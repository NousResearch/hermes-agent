"""Read-only, account-wide GitHub pull request operations for Desktop."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from typing import Any, Callable

SUMMARY_FIELDS = "number,title,url,state,isDraft,updatedAt,createdAt,repository,author,labels,commentsCount"
DETAIL_FIELDS = SUMMARY_FIELDS + ",body,headRefName,baseRefName,additions,deletions,changedFiles,reviewDecision,mergeStateStatus,mergedAt,statusCheckRollup"
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MAX_OUTPUT = 8 * 1024 * 1024


def validate_filter(kind: str, state: str, limit: int = 100) -> tuple[str, str, int]:
    if kind not in {"created", "review-requested"}:
        raise ValueError("Unsupported pull request filter")
    if state not in {"open", "closed"} or (kind == "review-requested" and state != "open"):
        raise ValueError("Unsupported pull request state")
    return kind, state, max(1, min(100, int(limit)))


def validate_ref(repository: str, number: int) -> tuple[str, int]:
    if not REPOSITORY_RE.fullmatch(repository or ""):
        raise ValueError("Invalid GitHub repository")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise ValueError("Invalid pull request number")
    return repository, number


def search_args(kind: str, state: str, limit: int = 100) -> list[str]:
    kind, state, limit = validate_filter(kind, state, limit)
    actor = ["--author", "@me"] if kind == "created" else ["--review-requested", "@me"]
    return ["search", "prs", *actor, "--state", state, "--sort", "updated", "--order", "desc", "--limit", str(limit), "--json", SUMMARY_FIELDS]


def _run(args: list[str], timeout: int = 30) -> dict[str, Any]:
    gh = shutil.which("gh")
    if not gh:
        return {"ok": False, "kind": "missing", "stdout": ""}
    try:
        result = subprocess.run([gh, *args], capture_output=True, text=True, timeout=timeout, shell=False, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "kind": "timeout", "stdout": ""}
    stdout = result.stdout[:MAX_OUTPUT]
    return {"ok": result.returncode == 0 and len(result.stdout) <= MAX_OUTPUT, "kind": "success" if result.returncode == 0 else "failure", "stdout": stdout}


def _repository(raw: Any) -> str | None:
    value = raw if isinstance(raw, str) else raw.get("nameWithOwner") if isinstance(raw, dict) else None
    return value if value and REPOSITORY_RE.fullmatch(value) else None


def normalize_summary(raw: Any, repository: str | None = None) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    repo = repository or _repository(raw.get("repository"))
    number = raw.get("number")
    if not repo or not isinstance(number, int) or number <= 0 or not isinstance(raw.get("title"), str) or not isinstance(raw.get("url"), str):
        return None
    state = str(raw.get("state", "UNKNOWN")).upper()
    if state not in {"OPEN", "CLOSED", "MERGED"}:
        state = "UNKNOWN"
    author = raw.get("author")
    normalized_author = {"login": author["login"]} if isinstance(author, dict) and isinstance(author.get("login"), str) else None
    if normalized_author is not None and isinstance(author.get("url"), str):
        normalized_author["url"] = author["url"]
    labels = [{"name": item["name"], **({"color": item["color"]} if isinstance(item.get("color"), str) else {})} for item in raw.get("labels", []) if isinstance(item, dict) and isinstance(item.get("name"), str)] if isinstance(raw.get("labels"), list) else []
    return {"id": f"{repo}#{number}", "repository": repo, "number": number, "title": raw["title"], "url": raw["url"], "state": state, "isDraft": bool(raw.get("isDraft")), "author": normalized_author, "labels": labels, "commentsCount": max(0, int(raw.get("commentsCount") or 0)), "createdAt": raw.get("createdAt") if isinstance(raw.get("createdAt"), str) else "", "updatedAt": raw.get("updatedAt") if isinstance(raw.get("updatedAt"), str) else ""}


def normalize_checks(raw: Any) -> dict[str, int]:
    out = {"total": 0, "pending": 0, "passed": 0, "failed": 0, "skipped": 0}
    for check in raw if isinstance(raw, list) else []:
        if not isinstance(check, dict):
            continue
        out["total"] += 1
        status, conclusion = str(check.get("status", "")).upper(), str(check.get("conclusion", check.get("state", ""))).upper()
        bucket = "pending" if status and status != "COMPLETED" else "passed" if conclusion in {"SUCCESS", "NEUTRAL"} else "skipped" if conclusion in {"SKIPPED", "STALE"} else "failed" if conclusion in {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"} else "pending"
        out[bucket] += 1
    return out


def list_pull_requests(kind: str, state: str, limit: int = 100, runner: Callable = _run) -> dict[str, Any]:
    fetched_at = int(time.time() * 1000)
    auth = runner(["auth", "status"])
    if not auth["ok"]:
        return {"authState": "gh-missing" if auth.get("kind") == "missing" else "not-authenticated", "items": [], "fetchedAt": fetched_at}
    result = runner(search_args(kind, state, limit))
    if not result["ok"]:
        return {"authState": "error", "items": [], "fetchedAt": fetched_at, "error": "GitHub CLI request timed out" if result.get("kind") == "timeout" else "Failed to load pull requests"}
    try:
        raw = json.loads(result["stdout"])
        if not isinstance(raw, list):
            raise ValueError
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"authState": "error", "items": [], "fetchedAt": fetched_at, "error": "GitHub CLI returned invalid data"}
    return {"authState": "ready", "items": [item for item in (normalize_summary(value) for value in raw) if item], "fetchedAt": fetched_at}


def pull_request_detail(repository: str, number: int, runner: Callable = _run) -> dict[str, Any]:
    repository, number = validate_ref(repository, number)
    result = runner(["pr", "view", str(number), "--repo", repository, "--json", DETAIL_FIELDS])
    if not result["ok"]:
        raise RuntimeError("GitHub CLI request timed out" if result.get("kind") == "timeout" else "Failed to load pull request details")
    try:
        raw = json.loads(result["stdout"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("GitHub CLI returned invalid data") from exc
    summary = normalize_summary(raw, repository)
    if not summary:
        raise RuntimeError("GitHub CLI returned invalid pull request data")
    merged_at = raw.get("mergedAt") if isinstance(raw.get("mergedAt"), str) else None
    return {**summary, "state": "MERGED" if merged_at else summary["state"], "body": raw.get("body") if isinstance(raw.get("body"), str) else "", "headRefName": raw.get("headRefName") if isinstance(raw.get("headRefName"), str) else "", "baseRefName": raw.get("baseRefName") if isinstance(raw.get("baseRefName"), str) else "", "additions": max(0, int(raw.get("additions") or 0)), "deletions": max(0, int(raw.get("deletions") or 0)), "changedFiles": max(0, int(raw.get("changedFiles") or 0)), "reviewDecision": raw.get("reviewDecision") or None, "mergeStateStatus": raw.get("mergeStateStatus") or None, "mergedAt": merged_at, "checks": normalize_checks(raw.get("statusCheckRollup"))}
