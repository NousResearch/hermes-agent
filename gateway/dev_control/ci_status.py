"""Fail-open GitHub CI status reader for Dev ship gates."""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


GITHUB_API_BASE = "https://api.github.com"
CI_STATES = {"success", "failure", "pending", "unknown"}
SUCCESS_CONCLUSIONS = {"success", "neutral", "skipped"}
FAILURE_CONCLUSIONS = {"failure", "timed_out", "cancelled", "action_required", "startup_failure"}
PENDING_CHECK_STATUSES = {"queued", "in_progress", "requested", "waiting", "pending"}


def fetch_ci_status(
    *,
    repo: str,
    ref: str,
    timeout_seconds: float = 8.0,
    opener: Any = None,
) -> Dict[str, Any]:
    """Return combined GitHub status/check-run state; fail open to unknown."""

    repo = str(repo or "").strip()
    ref = str(ref or "").strip()
    if not _valid_repo(repo) or not ref:
        return _unknown(repo=repo, ref=ref, warning="repo must be owner/name and ref is required.")
    opener = opener or urllib.request.urlopen
    token = _github_token()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "HermesDevCIStatus",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    }
    warnings: list[str] = []
    try:
        statuses = _get_json(
            f"{GITHUB_API_BASE}/repos/{repo}/commits/{_quote_ref(ref)}/status",
            headers=headers,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        checks = _get_json(
            f"{GITHUB_API_BASE}/repos/{repo}/commits/{_quote_ref(ref)}/check-runs",
            headers=headers,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
    except Exception as exc:
        return _unknown(repo=repo, ref=ref, warning=f"CI status unavailable: {exc}")
    status_items = _status_items(statuses)
    check_items = _check_items(checks)
    total = len(status_items) + len(check_items)
    if total == 0:
        warnings.append("No GitHub statuses or check runs were returned for this ref.")
        state = "unknown"
    elif any(item["state"] == "failure" for item in [*status_items, *check_items]):
        state = "failure"
    elif any(item["state"] == "pending" for item in [*status_items, *check_items]):
        state = "pending"
    else:
        state = "success"
    failing = [item for item in [*status_items, *check_items] if item["state"] == "failure"]
    return {
        "ok": True,
        "object": "hermes.dev_ci_status",
        "repo": repo,
        "ref": ref,
        "state": state,
        "total": total,
        "failing": failing,
        "checks": [*status_items, *check_items],
        "html_url": _html_url(statuses, checks),
        "warnings": warnings,
        "ship_gate": ci_ship_gate({"state": state, "failing": failing, "warnings": warnings}),
    }


def ci_ship_gate(ci_status: Dict[str, Any]) -> Dict[str, Any]:
    state = str((ci_status or {}).get("state") or "unknown").lower()
    if state == "failure":
        return {
            "status": "blocked_by_ci",
            "blocks_ship": True,
            "reason": "CI is failing for the release ref.",
        }
    if state == "pending":
        return {
            "status": "blocked_by_ci",
            "blocks_ship": True,
            "reason": "CI is still running for the release ref.",
        }
    if state == "success":
        return {"status": "ready", "blocks_ship": False, "reason": None}
    return {
        "status": "ci_unknown",
        "blocks_ship": False,
        "reason": "CI status could not be read; this is advisory and does not hard-block.",
    }


def _get_json(url: str, *, headers: Dict[str, str], timeout_seconds: float, opener: Any) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(body or f"GitHub API returned HTTP {exc.code}") from exc


def _status_items(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    statuses = payload.get("statuses") if isinstance(payload, dict) else []
    items = []
    for status in statuses or []:
        context = str(status.get("context") or "status").strip()
        raw_state = str(status.get("state") or "").lower()
        state = "success" if raw_state == "success" else "pending" if raw_state == "pending" else "failure"
        items.append({
            "name": context,
            "state": state,
            "raw_state": raw_state,
            "html_url": status.get("target_url"),
        })
    return items


def _check_items(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    check_runs = payload.get("check_runs") if isinstance(payload, dict) else []
    items = []
    for check in check_runs or []:
        name = str(check.get("name") or "check").strip()
        status = str(check.get("status") or "").lower()
        conclusion = str(check.get("conclusion") or "").lower()
        if conclusion in SUCCESS_CONCLUSIONS:
            state = "success"
        elif conclusion in FAILURE_CONCLUSIONS:
            state = "failure"
        elif status in PENDING_CHECK_STATUSES or not conclusion:
            state = "pending"
        else:
            state = "failure"
        items.append({
            "name": name,
            "state": state,
            "raw_status": status,
            "conclusion": conclusion,
            "html_url": check.get("html_url"),
        })
    return items


def _github_token() -> str:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        return token.strip()
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _valid_repo(repo: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", repo))


def _quote_ref(ref: str) -> str:
    return urllib.parse.quote(ref, safe="")


def _html_url(statuses: Dict[str, Any], checks: Dict[str, Any]) -> Optional[str]:
    for check in (checks.get("check_runs") if isinstance(checks, dict) else []) or []:
        if check.get("html_url"):
            return check.get("html_url")
    if isinstance(statuses, dict):
        return statuses.get("repository", {}).get("html_url") or statuses.get("url")
    return None


def _unknown(*, repo: str, ref: str, warning: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "object": "hermes.dev_ci_status",
        "repo": repo,
        "ref": ref,
        "state": "unknown",
        "total": 0,
        "failing": [],
        "checks": [],
        "html_url": None,
        "warnings": [warning],
        "ship_gate": ci_ship_gate({"state": "unknown"}),
    }
