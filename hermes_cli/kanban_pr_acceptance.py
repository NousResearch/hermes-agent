"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any
from urllib.parse import quote

from hermes_cli.config import load_config

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_JOB = re.compile(r"https://github\.com/[^/]+/[^/]+/actions/runs/[0-9]+/job/([0-9]+)(?:\?.*)?")


def _plan_limited(error: subprocess.CalledProcessError) -> bool:
    # Do not interpret arbitrary 403s (bad credentials, insufficient scopes) as plan limits.
    return "Upgrade to GitHub Pro or make this repository public to enable this feature." in (error.stderr or "")


def _authorities(repo: str) -> list[str]:
    configured = load_config().get("kanban", {}).get("pr_acceptance_authorities", {})
    contexts = configured.get(repo) if isinstance(configured, dict) else None
    if not isinstance(contexts, list) or not contexts or any(
        not isinstance(c, str) or not c.strip() or c != c.strip() for c in contexts
    ) or len(set(contexts)) != len(contexts):
        return []
    return contexts


def _unexecuted_job(run: dict, repo: str, sha: str) -> int | None:
    if run.get("status") != "completed" or run.get("conclusion") not in {"failure", "skipped"}:
        return None
    if (run.get("app") or {}).get("slug") != "github-actions":
        return None
    match = _JOB.fullmatch(run.get("details_url") or "")
    if not match or not (run.get("details_url") or "").startswith(f"https://github.com/{repo}/actions/"):
        return None
    job = _api(f"repos/{repo}/actions/jobs/{match[1]}")
    if (job.get("check_run_url") == run.get("url") and job.get("head_sha") == sha and
            job.get("conclusion") == run["conclusion"] and job.get("status") == "completed" and
            not job.get("runner_name") and job.get("steps") == []):
        return job.get("run_id")
    return None


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
        command += ["--paginate"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=30, check=True)
    if paginate:
        decoder = json.JSONDecoder()
        value: Any = []
        text = result.stdout
        while text.strip():
            page, consumed = decoder.raw_decode(text.lstrip())
            value.append(page)
            text = text.lstrip()[consumed:]
    else:
        value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


def collect_acceptance(contract: str, published_pr: str | None) -> dict:
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
        plan_limited = False
        try:
            rules = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100", paginate=True)
        except subprocess.CalledProcessError as error:
            if not _plan_limited(error):
                raise
            plan_limited = True
            rules = []
        for page in rules:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    required.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        # Branch protection can be hidden independently of repository rules. Probe it
        # when GraphQL has no rule; a 403 is only a fallback trigger for this exact plan error.
        if not required and not plan_limited:
            try:
                protection_api = _api(f"repos/{repo}/branches/{quote(branch, safe='')}/protection/required_status_checks")
                required.update((c, None) for c in protection_api.get("contexts", []))
            except subprocess.CalledProcessError as error:
                if not _plan_limited(error):
                    raise
                plan_limited = True
        if plan_limited and not required:
            required = {(context, None) for context in _authorities(repo)}
            receipt["authority_source"] = "configured-plan-limited"
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        if not required:
            receipt["detail"] = "No repository-required checks are configured; explicitly use a local-only contract for non-CI tasks."
            return receipt
        pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest", paginate=True)
        runs = [run for page in pages for run in page["check_runs"]]
        if len({r["id"] for r in runs}) != pages[0]["total_count"]:
            raise ValueError("Incomplete check-run pagination")
        statuses = [{**s, "sha": s.get("sha", sha)} for page in _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100", paginate=True) for s in page]
        outcomes = []
        if receipt.get("authority_source") == "configured-plan-limited":
            # A non-authoritative failure is ignorable ONLY when the GitHub job never
            # acquired a runner and executed zero steps. Skipped dependents are ignorable
            # only in the same workflow run as such a failure. Executed tests fail closed.
            unexecuted = {r["id"]: _unexecuted_job(r, repo, sha) for r in runs
                          if r.get("conclusion") in {"failure", "skipped"} and
                          r.get("head_sha") == sha and not any(r["name"] == c for c, _ in required)}
            failed_runs = {unexecuted[r["id"]] for r in runs
                           if r.get("conclusion") == "failure" and r["id"] in unexecuted and
                           unexecuted[r["id"]] is not None}
            for run in runs:
                if any(run["name"] == c for c, _ in required):
                    continue
                classification = _classify(run, sha, run.get("conclusion"), True)
                if ((classification == "failure" and unexecuted.get(run["id"]) is not None) or
                        (run.get("conclusion") == "skipped" and unexecuted.get(run["id"]) in failed_runs
                         and unexecuted.get(run["id"]) is not None)):
                    receipt["checks"].append({"name": run["name"], "id": run["id"],
                        "head_sha": run.get("head_sha"), "classification":
                        "pre-runner-infra" if classification == "failure" else "skipped-after-pre-runner"})
                elif classification != "success":
                    outcomes.append(classification)
                    receipt["checks"].append({"name": run["name"], "id": run["id"],
                        "head_sha": run.get("head_sha"), "classification": classification})
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
        if (current["head"]["sha"] != sha or current["base"]["ref"] != branch or
                (current["state"] == "closed" and not current.get("merged")) or
                (pr["state"] == "MERGED" and not (current.get("merged") and current.get("merge_commit_sha")))):
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
