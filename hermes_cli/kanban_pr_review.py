"""GitHub PR review/CI polling for Kanban tasks.

The parser functions in this module are intentionally independent from
``gh`` so tests can feed fixture JSON directly. The CLI path only handles
fetching current GitHub data and applying the parsed state to a kanban task.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Any, Callable, Optional

from hermes_cli import kanban_db as kb


class PRReviewError(RuntimeError):
    pass


_FAIL_CONCLUSIONS = {
    "ACTION_REQUIRED",
    "CANCELLED",
    "FAILURE",
    "STARTUP_FAILURE",
    "TIMED_OUT",
}
_PASS_CONCLUSIONS = {"SUCCESS", "NEUTRAL", "SKIPPED"}
_PENDING_STATES = {
    "EXPECTED", "IN_PROGRESS", "PENDING", "QUEUED", "REQUESTED", "WAITING",
}


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _nodes(value: Any) -> list[dict]:
    if isinstance(value, dict):
        raw = value.get("nodes", [])
    else:
        raw = value
    return [v for v in (raw or []) if isinstance(v, dict)]


def _check_id(check: dict) -> str:
    for key in (
        "id", "databaseId", "node_id", "nodeId",
        "link", "url", "name", "context",
    ):
        value = check.get(key)
        if value is not None:
            return f"check:{value}"
    return "check:unknown"


def _review_id(review: dict) -> str:
    for key in ("id", "databaseId", "node_id", "nodeId", "author"):
        value = review.get(key)
        if value is not None:
            return f"review:{value}"
    return "review:changes_requested"


def _thread_action_id(thread: dict) -> str:
    comments = _nodes(thread.get("comments"))
    if comments:
        latest = comments[-1]
        for key in ("id", "databaseId", "node_id", "nodeId"):
            value = latest.get(key)
            if value is not None:
                return f"comment:{value}"
    return f"thread:{thread.get('id') or 'unknown'}"


def _extract_rollup_checks(pr_view: Optional[dict]) -> list[dict]:
    if not isinstance(pr_view, dict):
        return []
    rollup = pr_view.get("statusCheckRollup")
    if not isinstance(rollup, list):
        rollup = _nodes(rollup)
    checks: list[dict] = []
    for item in rollup or []:
        if not isinstance(item, dict):
            continue
        if "context" in item and isinstance(item["context"], dict):
            merged = dict(item["context"])
            merged.setdefault("id", item.get("id"))
            checks.append(merged)
        else:
            checks.append(item)
    return checks


def _extract_checks(checks: Optional[Any]) -> list[dict]:
    if isinstance(checks, list):
        return [c for c in checks if isinstance(c, dict)]
    if isinstance(checks, dict):
        for key in ("checks", "checkRuns", "nodes"):
            value = checks.get(key)
            if isinstance(value, list):
                return [c for c in value if isinstance(c, dict)]
            if isinstance(value, dict):
                return _nodes(value)
    return []


def _extract_review_threads(review_threads: Optional[dict | list]) -> list[dict]:
    if isinstance(review_threads, list):
        return [t for t in review_threads if isinstance(t, dict)]
    if not isinstance(review_threads, dict):
        return []
    data = review_threads.get("data")
    if isinstance(data, dict):
        repo = data.get("repository")
        if isinstance(repo, dict):
            pr = repo.get("pullRequest")
            if isinstance(pr, dict):
                threads = pr.get("reviewThreads")
                if isinstance(threads, dict):
                    return _nodes(threads)
    threads = review_threads.get("reviewThreads")
    if isinstance(threads, dict):
        return _nodes(threads)
    return _nodes(review_threads)


def evaluate_pr_review(
    *,
    pr_view: Optional[dict] = None,
    checks: Optional[list[dict]] = None,
    review_threads: Optional[dict | list] = None,
    seen_ids: Optional[set[str] | list[str]] = None,
) -> dict[str, Any]:
    """Classify a PR snapshot into a kanban review state.

    Returns a JSON-serializable dict with:

    - ``state`` / ``task_status``: ``merge_ready``, ``in_review``, or ``code_review``
    - ``actionable``: whether this poll found new feedback/failures
    - ``actionable_ids``: unseen review/check identifiers to persist as seen
    - ``observed_ids``: all currently open failure/feedback identifiers
    """
    seen = set(str(x) for x in (seen_ids or set()))
    all_checks = _extract_checks(checks) + _extract_rollup_checks(pr_view)
    actionable_ids: list[str] = []
    observed_ids: list[str] = []
    pending: list[str] = []
    open_seen: list[str] = []

    def note_open(identifier: str) -> None:
        observed_ids.append(identifier)
        if identifier in seen:
            open_seen.append(identifier)
        else:
            actionable_ids.append(identifier)

    for check in all_checks:
        if not isinstance(check, dict):
            continue
        cid = _check_id(check)
        bucket = _upper(check.get("bucket"))
        conclusion = _upper(check.get("conclusion"))
        state = _upper(check.get("state") or check.get("status"))
        if bucket == "FAIL" or conclusion in _FAIL_CONCLUSIONS:
            note_open(cid)
        elif bucket == "PENDING" or state in _PENDING_STATES or (
            not conclusion and state not in {"COMPLETED", "SUCCESS"}
        ):
            pending.append(cid)
        elif conclusion and conclusion not in _PASS_CONCLUSIONS:
            note_open(cid)

    if isinstance(pr_view, dict):
        if _upper(pr_view.get("reviewDecision")) == "CHANGES_REQUESTED":
            reviews = [
                r for r in (pr_view.get("reviews") or [])
                if isinstance(r, dict) and _upper(r.get("state")) == "CHANGES_REQUESTED"
            ]
            if reviews:
                for review in reviews:
                    note_open(_review_id(review))
            else:
                note_open("review:changes_requested")
        elif _upper(pr_view.get("reviewDecision")) in {
            "REVIEW_REQUIRED",
            "REVIEW_REQUESTED",
        }:
            pending.append("review:required")

    for thread in _extract_review_threads(review_threads):
        if thread.get("isResolved") is True:
            continue
        note_open(_thread_action_id(thread))

    # Deduplicate while preserving encounter order.
    actionable_ids = list(dict.fromkeys(actionable_ids))
    observed_ids = list(dict.fromkeys(observed_ids))
    pending = list(dict.fromkeys(pending))
    open_seen = list(dict.fromkeys(open_seen))

    if actionable_ids:
        state = "code_review"
        reason = "actionable_feedback"
    elif pending:
        state = "in_review"
        reason = "pending"
    elif open_seen:
        state = "code_review"
        reason = "already_seen_feedback"
    else:
        state = "merge_ready"
        reason = "green"

    return {
        "state": state,
        "task_status": state,
        "actionable": bool(actionable_ids),
        "reason": reason,
        "actionable_ids": actionable_ids,
        "observed_ids": observed_ids,
        "pending_ids": pending,
        "seen_ids": sorted(seen.union(actionable_ids)),
    }


def parse_pr_url(url: str) -> Optional[dict[str, str | int]]:
    m = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", str(url or ""))
    if not m:
        return None
    return {"owner": m.group(1), "repo": m.group(2), "number": int(m.group(3))}


def _run_gh_json(
    args: list[str],
    runner: Optional[Callable[[list[str]], str]] = None,
) -> Any:
    if runner is None:
        try:
            completed = subprocess.run(
                ["gh", *args],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise PRReviewError("gh CLI not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or exc.stdout or str(exc)).strip()
            raise PRReviewError(f"gh failed: {msg}") from exc
        raw = completed.stdout
    else:
        raw = runner(args)
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise PRReviewError(f"gh returned non-JSON for {' '.join(args)}") from exc


def _latest_pr_metadata(conn, task_id: str) -> dict:
    latest = kb.latest_run(conn, task_id)
    pr_meta = kb.extract_pr_metadata(latest.metadata if latest else None)
    if not pr_meta:
        raise PRReviewError(f"task {task_id} has no PR metadata on its latest run")
    return pr_meta


def _seen_review_ids(conn, task_id: str) -> set[str]:
    seen: set[str] = set()
    for event in kb.list_events(conn, task_id):
        payload = event.payload or {}
        review = payload.get("review") if isinstance(payload, dict) else None
        if isinstance(review, dict):
            seen.update(str(x) for x in review.get("seen_ids", []) if x)
    return seen


def _append_review_poll_event(conn, task_id: str, result: dict[str, Any]) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'review_polled', ?, ?)",
            (
                task_id,
                json.dumps({"review": result}, ensure_ascii=False),
                int(time.time()),
            ),
        )


def fetch_pr_snapshot(
    pr_meta: dict,
    runner: Optional[Callable[[list[str]], str]] = None,
) -> dict:
    pr_ref = str(pr_meta.get("pr_url") or pr_meta.get("pr_number") or "")
    if not pr_ref:
        raise PRReviewError("PR metadata lacks pr_url/pr_number")
    pr_view = _run_gh_json(
        [
            "pr",
            "view",
            pr_ref,
            "--json",
            (
                "number,url,state,isDraft,reviewDecision,mergeStateStatus,"
                "statusCheckRollup,reviews"
            ),
        ],
        runner,
    )
    checks = _run_gh_json(
        ["pr", "checks", pr_ref, "--json", "name,bucket,state,conclusion,link"],
        runner,
    )
    threads = {}
    parsed = parse_pr_url(str(pr_meta.get("pr_url") or pr_view.get("url") or ""))
    if parsed:
        query = """
        query($owner:String!, $repo:String!, $number:Int!) {
          repository(owner:$owner, name:$repo) {
            pullRequest(number:$number) {
              reviewThreads(first:100) {
                nodes {
                  id
                  isResolved
                  comments(first:50) {
                    nodes { id databaseId body path url author { login } }
                  }
                }
              }
            }
          }
        }
        """
        threads = _run_gh_json(
            [
                "api",
                "graphql",
                "-f",
                f"query={query}",
                "-F",
                f"owner={parsed['owner']}",
                "-F",
                f"repo={parsed['repo']}",
                "-F",
                f"number={parsed['number']}",
            ],
            runner,
        )
    return {"pr_view": pr_view, "checks": checks, "review_threads": threads}


def poll_task(
    conn,
    task_id: str,
    *,
    runner: Optional[Callable[[list[str]], str]] = None,
) -> dict[str, Any]:
    """Fetch one PR snapshot, classify it, and update the kanban task."""
    task = kb.get_task(conn, task_id)
    if task is None:
        raise PRReviewError(f"task {task_id} not found")
    pr_meta = _latest_pr_metadata(conn, task_id)
    snapshot = fetch_pr_snapshot(pr_meta, runner=runner)
    result = evaluate_pr_review(
        pr_view=snapshot.get("pr_view"),
        checks=snapshot.get("checks"),
        review_threads=snapshot.get("review_threads"),
        seen_ids=_seen_review_ids(conn, task_id),
    )
    result["pr"] = pr_meta
    kb.transition_review_task(
        conn,
        task_id,
        result["task_status"],
        summary=f"GitHub PR review poll: {result['reason']}",
        metadata={"github_review": result},
    )
    _append_review_poll_event(conn, task_id, result)
    return result
