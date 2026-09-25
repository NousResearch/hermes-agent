"""Exact-head GitHub / GitLab acceptance for explicitly declared PR/MR tasks.

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
# GitLab: ``gitlab:HOST/GROUP/.../PROJECT`` (CI green on the MR head) or
# ``gitlab-merged:HOST/...`` (additionally the MR must be merged), or an exact MR
# URL; a trailing ``#merged`` on the URL carries the merged requirement.
_GL_PATH = r"([A-Za-z0-9.-]+)/([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)"
_GL_PROJECT = re.compile(r"(gitlab|gitlab-merged):" + _GL_PATH)
_GL_MR = re.compile(r"https://" + _GL_PATH + r"/-/merge_requests/([1-9][0-9]*)(#merged)?")
_GL_WAIT = {"created", "waiting_for_resource", "preparing", "pending", "running", "scheduled", "manual"}


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)
                                          or _GL_PROJECT.fullmatch(value) or _GL_MR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, an exact GitHub PR URL, "
                         "gitlab:HOST/PROJECT, gitlab-merged:HOST/PROJECT, or an exact GitLab MR URL")
    return value


def bind_contract(contract: str, published: str | None) -> str | None:
    """The contract to persist once ``published`` matches a repo/project-level
    contract; ``None`` when it does not bind (already exact, or a sibling)."""
    if not isinstance(published, str):
        return None
    gh = _PR.fullmatch(published)
    if gh and contract == gh[1]:
        return published
    project, mr = _GL_PROJECT.fullmatch(contract), _GL_MR.fullmatch(published)
    if project and mr and not mr[4] and (project[2], project[3]) == (mr[1], mr[2]):
        return published + ("#merged" if project[1] == "gitlab-merged" else "")
    return None


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


def collect_acceptance(contract: str, published_pr: str | None) -> dict:
    if _GL_PROJECT.fullmatch(contract) or _GL_MR.fullmatch(contract):
        return _collect_gitlab(contract, published_pr)
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
            receipt["detail"] = "No repository-required checks are configured; explicitly use a local-only contract for non-CI tasks."
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


def _collect_gitlab(contract: str, published_pr: str | None) -> dict:
    """GitLab MR: the head pipeline of the CURRENT head SHA must be ``success``;
    a ``#merged`` contract also needs ``state == merged``. Read-only (``glab api``)."""
    receipt = {"ok": False, "classification": "missing", "head_sha": None, "pr_url": None, "checks": [],
               "recovery": "Fix the failing pipeline or wait for it (and for the merge on a merged "
                           "contract), then retry completion. Use kanban_block if human input is needed."}
    match = _GL_MR.fullmatch(contract)
    if not match or (published_pr and published_pr != contract.removesuffix("#merged")):
        receipt["detail"] = "Supply metadata.published_pr with the MR URL of the declared GitLab project."
        return receipt
    host, path, number, merged = match[1], match[2], int(match[3]), bool(match[4])
    receipt["pr_url"] = contract.removesuffix("#merged")
    endpoint = f"projects/{quote(path, safe='')}/merge_requests/{number}"
    try:
        mr = _glab(host, endpoint)
        sha, target = mr["sha"], mr["target_branch"]
        receipt["head_sha"] = sha
        if not re.fullmatch(r"[0-9a-f]{40}", sha or "") or mr["state"] not in {"opened", "merged"}:
            raise ValueError("MR is closed or current head is unavailable")
        pipeline = mr.get("head_pipeline") or {}
        status = pipeline.get("status")
        if not pipeline:
            classification = "missing"
        elif pipeline.get("sha") != sha:
            classification = "stale"
        else:
            classification = {"success": "success", "failed": "failure", "canceled": "failure",
                              "skipped": "failure"}.get(status, "pending" if status in _GL_WAIT else "infra")
        if pipeline:
            receipt["checks"].append({"name": "pipeline", "id": pipeline.get("id"), "url": pipeline.get("web_url"),
                                      "head_sha": pipeline.get("sha"), "classification": classification,
                                      "conclusion": status})
        if classification == "success" and merged and mr["state"] != "merged":
            classification = "pending"
            receipt["detail"] = "Pipeline is green; the contract also requires the MR to be merged."
        current = _glab(host, endpoint)
        if current["sha"] != sha or current["target_branch"] != target or current["state"] == "closed":
            receipt.update(classification="stale", detail="MR head/target changed while collecting evidence; retry.")
            return receipt
        receipt["classification"] = classification
        receipt["ok"] = classification == "success"
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        receipt.update(classification="infra",
                       detail="GitLab acceptance evidence unavailable; check glab authentication/API access and retry.")
        return receipt


def _glab(host: str, endpoint: str) -> dict:
    result = subprocess.run(["glab", "api", endpoint, "--hostname", host], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=30, check=True)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("unexpected GitLab API response")
    return value


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")
