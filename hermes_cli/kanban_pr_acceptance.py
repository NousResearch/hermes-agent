"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.
"""
from __future__ import annotations

import json
import re
import subprocess
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_WORKFLOW = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/actions/runs/([1-9][0-9]*)")


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False):
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=30, check=True)
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


def collect_acceptance(contract: str, published_pr: str | None, metadata: dict | None = None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": published_pr, "checks": [],
               "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                           "Use kanban_block if human input is needed; receipts remain on the task event log."}
    try:
        declared = _PR.fullmatch(contract)
        url = contract if declared else published_pr
        match = _PR.fullmatch(url or "")
        if not match or (not declared and match[1] != contract) or (declared and published_pr and published_pr != contract):
            receipt["detail"] = "Supply metadata.published_pr matching the persisted completion contract."
            return receipt
        repo, number = match[1], int(match[2])
        receipt["pr_url"] = url
        owner, name = repo.split("/")
        query = '''{repository(owner:%s,name:%s){pullRequest(number:%d){headRefOid baseRefName state
            baseRef{branchProtectionRule{requiredStatusChecks{context app{databaseId}}}}}}}''' % (
                json.dumps(owner), json.dumps(name), number)
        pr = _api("graphql", query=query)["data"]["repository"]["pullRequest"]
        sha, branch = pr["headRefOid"], pr["baseRefName"]
        receipt["head_sha"] = sha
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        protection = (pr.get("baseRef") or {}).get("branchProtectionRule") or {}
        required = {(r["context"], (r.get("app") or {}).get("databaseId")) for r in protection.get("requiredStatusChecks", [])}
        rules = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100", paginate=True)
        for page in rules:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    required.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        if not required:
            _collect_workflow_evidence(repo, number, sha, metadata, receipt)
            current = _api(f"repos/{repo}/pulls/{number}")
            if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
                receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
                return receipt
            receipt["classification"] = "success"
            receipt["ok"] = True
            return receipt
        pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest", paginate=True)
        runs = [run for page in pages for run in page["check_runs"]]
        if len({r["id"] for r in runs}) != pages[0]["total_count"]:
            raise ValueError("Incomplete check-run pagination")
        statuses = [{**s, "sha": sha} for page in _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100", paginate=True) for s in page]
        outcomes = []
        for context, app_id in sorted(required, key=str):
            matching = [r for r in runs if r["name"] == context and
                        (app_id in (None, -1) or r["app"]["id"] == app_id)]
            # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
            legacy = [s for s in statuses if s["context"] == context] if app_id in (None, -1) else []
            selected = matching + ([max(legacy, key=lambda s: s["id"])] if legacy else [])
            if not selected:
                outcomes.append("missing")
                receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
            for check in selected:
                is_run = "conclusion" in check
                outcome = check.get("conclusion") if is_run else check["state"]
                classification = _classify(check, sha, outcome, is_run)
                outcomes.append(classification)
                receipt["checks"].append({"name": context, "id": check["id"],
                    "url": check.get("html_url") or check.get("target_url"),
                    "head_sha": check.get("head_sha", check.get("sha")),
                    "classification": classification, "conclusion": outcome})
        # Re-read after all pages: old-head successes are never transferable.
        current = _api(f"repos/{repo}/pulls/{number}")
        if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
            receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
            return receipt
        receipt["classification"] = next((x for x in outcomes if x != "success"), "missing" if not outcomes else "success")
        receipt["ok"] = receipt["classification"] == "success"
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # Never persist gh stderr (credentials/host details); the failed phase is actionable.
        receipt.update(classification="infra", detail="GitHub acceptance evidence unavailable or incomplete; check gh authentication/API access and retry.")
        return receipt


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")


def _collect_workflow_evidence(repo: str, number: int, sha: str, metadata: dict | None, receipt: dict) -> None:
    """Accept only explicitly declared, API-confirmed successful Actions runs."""
    evidence = metadata.get("workflow_evidence") if isinstance(metadata, dict) else None
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("explicit workflow evidence is required when no checks are configured")
    seen = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("malformed workflow evidence")
        url = item.get("url")
        match = _WORKFLOW.fullmatch(url or "")
        if not match or match[1] != repo or match[2] in seen or item.get("head_sha") != sha:
            raise ValueError("workflow evidence is not tied to the current PR head")
        seen.add(match[2])
        if item.get("status") != "completed" or item.get("conclusion") != "success":
            raise ValueError("workflow evidence is not terminal-successful")
        jobs = item.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("workflow job evidence is missing or incomplete")
        run = _api(f"repos/{repo}/actions/runs/{match[2]}")
        if (run.get("head_sha") != sha or run.get("status") != "completed" or
                run.get("conclusion") != "success" or
                not any(p.get("number") == number for p in run.get("pull_requests", []))):
            raise ValueError("workflow run is stale, unsuccessful, or unrelated")
        job_pages = _api(f"repos/{repo}/actions/runs/{match[2]}/jobs?per_page=100", paginate=True)
        api_jobs = {job.get("id"): job for page in job_pages for job in page.get("jobs", [])}
        if not api_jobs:
            raise ValueError("workflow jobs are missing")
        for job in jobs:
            if not isinstance(job, dict) or job.get("id") not in api_jobs:
                raise ValueError("declared workflow job is missing")
            actual = api_jobs[job["id"]]
            if (job.get("name") != actual.get("name") or job.get("status") != "completed" or
                    job.get("conclusion") != "success" or actual.get("status") != "completed" or
                    actual.get("conclusion") != "success"):
                raise ValueError("workflow job is not terminal-successful")
        receipt["checks"].append({"name": item.get("name") or f"workflow-{match[2]}",
                                   "id": int(match[2]), "url": url, "head_sha": sha,
                                   "classification": "success", "conclusion": "success",
                                   "jobs": [job["id"] for job in jobs]})
