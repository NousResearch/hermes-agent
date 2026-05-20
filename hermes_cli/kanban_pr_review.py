"""One-shot GitHub PR review polling for Kanban tasks."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli import kanban_db as kb

_MISSING = object()

BLOCKING_CHECK_CONCLUSIONS = {
    "ACTION_REQUIRED",
    "CANCELLED",
    "FAILURE",
    "STARTUP_FAILURE",
    "TIMED_OUT",
}
PENDING_CHECK_STATES = {
    "EXPECTED",
    "PENDING",
    "QUEUED",
    "REQUESTED",
    "WAITING",
    "IN_PROGRESS",
}
_GITHUB_PR_URL_RE = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")


@dataclass
class PRReference:
    selector: str
    number: Optional[int] = None
    repo: Optional[str] = None
    url: Optional[str] = None
    cwd: Optional[str] = None


@dataclass
class ActionItem:
    id: str
    kind: str
    title: str
    state: str
    url: Optional[str] = None
    body: Optional[str] = None


@dataclass
class PollResult:
    state: str
    summary: str
    pr: dict[str, Any]
    action_items: list[ActionItem] = field(default_factory=list)
    pending_items: list[ActionItem] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)
    closed_unmerged: bool = False
    merged: bool = False

    @property
    def has_new_action(self) -> bool:
        return bool(self.action_items)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "summary": self.summary,
            "pr": self.pr,
            "action_items": [item.__dict__ for item in self.action_items],
            "pending_items": [item.__dict__ for item in self.pending_items],
            "seen_ids": sorted(self.seen_ids),
            "closed_unmerged": self.closed_unmerged,
            "merged": self.merged,
        }


def seen_review_ids(events: list[kb.Event]) -> set[str]:
    seen: set[str] = set()
    for event in events:
        if event.kind != "pr_review_poll" or not isinstance(event.payload, dict):
            continue
        raw = event.payload.get("seen_ids")
        if isinstance(raw, list):
            seen.update(str(v) for v in raw if v)
        for key in ("action_items", "pending_items"):
            items = event.payload.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("id"):
                        seen.add(str(item["id"]))
    return seen


def pr_reference_from_task(task: kb.Task, run: Optional[kb.Run]) -> Optional[PRReference]:
    metadata = run.metadata if run else None
    pr = kb.extract_pr_metadata(metadata)
    if not pr:
        return None
    raw_number = pr.get("pr_number")
    number = int(raw_number) if str(raw_number).isdigit() else None
    repo = pr.get("repo") or pr.get("repository")
    url = pr.get("pr_url")
    if url and (not repo or not number):
        match = _GITHUB_PR_URL_RE.search(str(url))
        if match:
            repo = repo or match.group(1)
            number = number or int(match.group(2))
    selector = str(url or number or "").strip()
    if not selector:
        return None
    cwd = pr.get("worktree") or pr.get("worktree_path") or pr.get("repo_path")
    if not cwd and task.workspace_path:
        cwd = task.workspace_path
    return PRReference(selector=selector, number=number, repo=repo, url=url, cwd=cwd)


def parse_checks(
    raw: Any,
    *,
    seen_ids: set[str],
) -> tuple[list[ActionItem], list[ActionItem], set[str], bool]:
    checks = _coerce_checks(raw)
    failing: list[ActionItem] = []
    pending: list[ActionItem] = []
    all_ids: set[str] = set()
    has_blocking = False
    for check in checks:
        name = str(
            check.get("name")
            or check.get("workflow")
            or check.get("context")
            or "check"
        )
        state = str(
            check.get("state") or check.get("status") or check.get("bucket") or ""
        ).upper()
        conclusion = str(check.get("conclusion") or "").upper()
        bucket = str(check.get("bucket") or "").upper()
        url = check.get("link") or check.get("detailsUrl") or check.get("url")
        identifier = f"check:{check.get('id') or name}:{conclusion or state or bucket}"
        all_ids.add(identifier)
        if conclusion in BLOCKING_CHECK_CONCLUSIONS or bucket == "FAIL" or state == "FAIL":
            has_blocking = True
            if identifier not in seen_ids:
                failing.append(
                    ActionItem(
                        identifier,
                        "check",
                        name,
                        conclusion or state or bucket,
                        url=url,
                    )
                )
        elif state in PENDING_CHECK_STATES or bucket == "PENDING":
            pending.append(ActionItem(identifier, "check", name, state or bucket, url=url))
    return failing, pending, all_ids, has_blocking


def parse_pr_review_state(
    pr_view: dict[str, Any],
    checks: Any,
    *,
    review_threads: Optional[dict[str, Any]] = None,
    seen_ids: Optional[set[str]] = None,
) -> PollResult:
    seen_ids = set(seen_ids or set())
    pr = _normalize_pr(pr_view)
    action_items: list[ActionItem] = []
    pending_items: list[ActionItem] = []
    all_seen: set[str] = set()

    has_blocking = False
    check_actions, check_pending, check_ids, check_blocking = parse_checks(
        checks,
        seen_ids=seen_ids,
    )
    has_blocking = has_blocking or check_blocking
    action_items.extend(check_actions)
    pending_items.extend(check_pending)
    all_seen.update(check_ids)

    review_decision = str(pr_view.get("reviewDecision") or "").upper()
    if review_decision == "CHANGES_REQUESTED":
        has_blocking = True
        identifier = "review-decision:changes-requested"
        all_seen.add(identifier)
        if identifier not in seen_ids:
            action_items.append(
                ActionItem(identifier, "review", "Changes requested", review_decision)
            )

    for review in _iter_nodes(pr_view.get("latestReviews")):
        state = str(review.get("state") or "").upper()
        if state != "CHANGES_REQUESTED":
            continue
        has_blocking = True
        identifier = f"review:{review.get('id') or review.get('url') or review.get('submittedAt')}"
        all_seen.add(identifier)
        if identifier not in seen_ids:
            author = _author_login(review.get("author"))
            title = f"Changes requested by {author}" if author else "Changes requested"
            action_items.append(
                ActionItem(
                    identifier,
                    "review",
                    title,
                    state,
                    url=review.get("url"),
                    body=review.get("body"),
                )
            )

    thread_source = review_threads if review_threads is not None else pr_view.get("reviewThreads")
    for thread in _iter_nodes(thread_source):
        if bool(thread.get("isResolved")):
            continue
        has_blocking = True
        identifier = f"thread:{thread.get('id') or thread.get('url')}"
        all_seen.add(identifier)
        if identifier in seen_ids:
            continue
        first_comment = next(iter(_iter_nodes(thread.get("comments"))), {})
        path = thread.get("path") or first_comment.get("path")
        title = f"Unresolved review thread{f' in {path}' if path else ''}"
        action_items.append(
            ActionItem(
                identifier,
                "thread",
                title,
                "UNRESOLVED",
                url=first_comment.get("url") or thread.get("url"),
                body=first_comment.get("body"),
            )
        )

    merged = bool(pr_view.get("merged")) or str(pr_view.get("state") or "").upper() == "MERGED"
    closed = str(pr_view.get("state") or "").upper() == "CLOSED"
    closed_unmerged = closed and not merged
    if closed_unmerged:
        identifier = f"pr:{pr.get('number') or pr.get('url')}:closed-unmerged"
        all_seen.add(identifier)
        if identifier not in seen_ids:
            action_items.append(
                ActionItem(
                    identifier,
                    "pr",
                    "PR is closed without merge",
                    "CLOSED",
                    url=pr.get("url"),
                )
            )

    if closed_unmerged:
        state = "closed_unmerged"
        summary = "PR is closed without merge; task needs operator action."
    elif action_items:
        state = "action_required"
        summary = f"{len(action_items)} new PR review item(s) need attention."
    elif has_blocking:
        state = "action_required"
        summary = "PR still has blocking feedback already recorded on this task."
    elif pending_items:
        state = "pending"
        summary = "PR checks are still pending; task remains in review."
    elif merged:
        state = "merged"
        summary = "PR is merged; review agent may complete deployment/closure checks."
    else:
        state = "green"
        summary = "PR checks are green and no unresolved blocking feedback was found."
    return PollResult(
        state=state,
        summary=summary,
        pr=pr,
        action_items=action_items,
        pending_items=pending_items,
        seen_ids=seen_ids | all_seen,
        closed_unmerged=closed_unmerged,
        merged=merged,
    )


def poll_task(
    conn,
    task_id: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> PollResult:
    task = kb.get_task(conn, task_id)
    if not task:
        raise ValueError(f"unknown task {task_id}")
    run = kb.latest_run(conn, task_id)
    ref = pr_reference_from_task(task, run)
    if not ref:
        raise ValueError(f"task {task_id} has no PR metadata on its latest run")
    seen = seen_review_ids(kb.list_events(conn, task_id))
    pr_view = _gh_json(_gh_pr_view_cmd(ref), cwd=ref.cwd, runner=runner)
    checks = _gh_json(_gh_pr_checks_cmd(ref), cwd=ref.cwd, runner=runner, default=[])
    threads = pr_view.get("reviewThreads")
    if threads is None and ref.repo and ref.number:
        threads = _fetch_review_threads_graphql(ref, cwd=ref.cwd, runner=runner)
    result = parse_pr_review_state(pr_view, checks, review_threads=threads, seen_ids=seen)
    kb.record_pr_review_poll(
        conn,
        task_id,
        result.to_event_payload(),
        comment=_comment_for_result(result),
    )
    return result


def _gh_pr_view_cmd(ref: PRReference) -> list[str]:
    fields = [
        "number", "url", "state", "merged", "isDraft", "mergeStateStatus",
        "reviewDecision", "latestReviews", "headRefName", "baseRefName",
        "headRepository", "headRepositoryOwner",
    ]
    cmd = ["gh", "pr", "view", ref.selector, "--json", ",".join(fields)]
    if ref.repo:
        cmd.extend(["--repo", ref.repo])
    return cmd


def _gh_pr_checks_cmd(ref: PRReference) -> list[str]:
    cmd = [
        "gh", "pr", "checks", ref.selector, "--json",
        "name,state,conclusion,link,bucket,workflow",
    ]
    if ref.repo:
        cmd.extend(["--repo", ref.repo])
    return cmd


def _gh_json(
    cmd: list[str],
    *,
    cwd: Optional[str],
    runner,
    default: Any = _MISSING,
) -> Any:
    proc = runner(cmd, cwd=_safe_cwd(cwd), capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        if default is not _MISSING:
            return default
        raise RuntimeError((proc.stderr or proc.stdout or "gh command failed").strip())
    text = (proc.stdout or "").strip()
    if not text and default is not _MISSING:
        return default
    return json.loads(text)


def _fetch_review_threads_graphql(
    ref: PRReference,
    *,
    cwd: Optional[str],
    runner,
) -> Optional[dict[str, Any]]:
    if not ref.repo or not ref.number or "/" not in ref.repo:
        return None
    owner, name = ref.repo.split("/", 1)
    query = """
    query($owner:String!, $name:String!, $number:Int!) {
      repository(owner:$owner, name:$name) {
        pullRequest(number:$number) {
          reviewThreads(first:100) {
            nodes {
              id
              isResolved
              path
              comments(first:1) { nodes { id body url path author { login } } }
            }
          }
        }
      }
    }
    """
    cmd = [
        "gh", "api", "graphql", "-f", f"query={query}",
        "-F", f"owner={owner}", "-F", f"name={name}", "-F", f"number={ref.number}",
    ]
    data = _gh_json(cmd, cwd=cwd, runner=runner, default=None)
    if not isinstance(data, dict):
        return None
    return (
        ((data.get("data") or {}).get("repository") or {}).get("pullRequest") or {}
    ).get("reviewThreads")


def _comment_for_result(result: PollResult) -> str:
    lines = [f"PR review poll: {result.summary}"]
    for item in result.action_items:
        suffix = f" ({item.url})" if item.url else ""
        lines.append(f"- [{item.kind}] {item.title}: {item.state}{suffix}")
        if item.body:
            lines.append(f"  {item.body.strip()[:500]}")
    return "\n".join(lines)


def _normalize_pr(pr_view: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": pr_view.get("number"),
        "url": pr_view.get("url"),
        "state": pr_view.get("state"),
        "merged": bool(pr_view.get("merged")),
        "reviewDecision": pr_view.get("reviewDecision"),
        "mergeStateStatus": pr_view.get("mergeStateStatus"),
        "headRefName": pr_view.get("headRefName"),
        "baseRefName": pr_view.get("baseRefName"),
    }


def _coerce_checks(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, dict)]
    if isinstance(raw, dict):
        for key in ("checks", "nodes"):
            if isinstance(raw.get(key), list):
                return [c for c in raw[key] if isinstance(c, dict)]
        return [raw]
    if isinstance(raw, str):
        checks: list[dict[str, Any]] = []
        for line in raw.splitlines():
            parts = re.split(r"\s{2,}|\t", line.strip())
            if len(parts) >= 2:
                checks.append({"name": parts[0], "state": parts[1]})
        return checks
    return []


def _iter_nodes(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        nodes = value.get("nodes")
        if isinstance(nodes, list):
            return [n for n in nodes if isinstance(n, dict)]
        edges = value.get("edges")
        if isinstance(edges, list):
            return [
                e.get("node")
                for e in edges
                if isinstance(e, dict) and isinstance(e.get("node"), dict)
            ]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


def _author_login(author: Any) -> Optional[str]:
    return author.get("login") if isinstance(author, dict) else None


def _safe_cwd(cwd: Optional[str]) -> Optional[str]:
    if not cwd:
        return None
    path = Path(str(cwd)).expanduser()
    return str(path) if path.exists() else None
