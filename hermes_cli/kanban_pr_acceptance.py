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


def _branch_rules(repo: str, branch: str, private: bool | None):
    try:
        rules = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100", paginate=True)
    except subprocess.CalledProcessError as exc:
        # gh --slurp wraps even an HTTP error in a page array. Only a first-page
        # plan denial is evidence of unavailable rules, not auth or partial policy.
        pages = json.loads(exc.stdout)
        if (private is True and exc.returncode == 1 and isinstance(pages, list)
                and len(pages) == 1 and isinstance(pages[0], dict)
                and (exc.stderr or "").rstrip().endswith("(HTTP 403)")
                and pages[0].get("status") == "403"
                and pages[0].get("message") ==
                "Upgrade to GitHub Pro or make this repository public to enable this feature."):
            return None
        raise
    if not isinstance(rules, list) or not rules or not all(isinstance(page, list) for page in rules):
        raise ValueError("Incomplete branch rules evidence")
    return rules


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
        query = '''{repository(owner:%s,name:%s){isPrivate pullRequest(number:%d){headRefOid baseRefName state
            baseRef{branchProtectionRule{requiredStatusChecks{context app{databaseId}}}}}}}''' % (
                json.dumps(owner), json.dumps(name), number)
        repository = _api("graphql", query=query)["data"]["repository"]
        pr = repository["pullRequest"]
        sha, branch = pr["headRefOid"], pr["baseRefName"]
        receipt["head_sha"] = sha
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        protection = pr["baseRef"]["branchProtectionRule"]
        required_checks = protection["requiredStatusChecks"] if protection is not None else []
        if not isinstance(required_checks, list):
            raise ValueError("Incomplete classic branch protection evidence")
        required = {(r["context"], (r.get("app") or {}).get("databaseId"))
                    for r in required_checks}
        rules = _branch_rules(repo, branch, repository.get("isPrivate"))
        receipt["rules_status"] = "plan_unavailable" if rules is None else "available"
        receipt["policy_source"] = "required_checks"
        for page in rules or []:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    required.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        if not required and rules is not None:
            receipt["detail"] = "No repository-required checks are configured; explicitly use a local-only contract for non-CI tasks."
            return receipt
        pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest", paginate=True)
        runs = [run for page in pages for run in page["check_runs"]]
        if len({r["id"] for r in runs}) != pages[0]["total_count"]:
            raise ValueError("Incomplete check-run pagination")
        statuses = [{**s, "sha": sha} for page in _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100", paginate=True) for s in page]
        outcomes = []
        selected = []
        if not required:
            # A plan-limited repo cannot mark CI required. Nothing observed is
            # optional here; keep every run and the newest legacy status/context.
            receipt["policy_source"] = "observed_checks"
            latest = {s["context"]: s for s in sorted(statuses, key=lambda s: s["id"])}
            selected = runs + list(latest.values())
            if not selected:
                receipt["detail"] = "No exact-head check runs or statuses were observed."
        for context, app_id in sorted(required, key=str):
            matching = [r for r in runs if r["name"] == context and
                        (app_id in (None, -1) or r["app"]["id"] == app_id)]
            # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
            legacy = [s for s in statuses if s["context"] == context] if app_id in (None, -1) else []
            checks = matching + ([max(legacy, key=lambda s: s["id"])] if legacy else [])
            if not checks:
                outcomes.append("missing")
                receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
            selected.extend(checks)
        for check in selected:
            is_run = "conclusion" in check
            outcome = check.get("conclusion") if is_run else check["state"]
            classification = _classify(check, sha, outcome, is_run)
            outcomes.append(classification)
            receipt["checks"].append({"name": check["name"] if is_run else check["context"], "id": check["id"],
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
