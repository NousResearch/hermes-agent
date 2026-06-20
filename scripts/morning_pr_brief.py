#!/usr/bin/env python3
"""Build a compact morning review brief for open GitHub PRs.

Spec / plan:
- Input: JSON array shaped like `gh pr list --json ...`, from `--json-file`, stdin,
  or live `gh pr list` when no JSON is supplied.
- Output: deterministic Markdown grouped by action priority so Joe can review the
  few PRs that need attention first.
- Safety: read-only helper; it never mutates GitHub state, sends messages, or
  shells out except for `gh pr list`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

GH_FIELDS = (
    "number,title,url,headRefName,author,updatedAt,reviewDecision,isDraft,"
    "mergeStateStatus,statusCheckRollup"
)
PRIORITY_ORDER = ("Review now", "Stale / blocked", "Approved / low-touch")


@dataclasses.dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    url: str
    branch: str
    author: str
    updated_at: dt.datetime
    age_label: str
    review_decision: str
    is_draft: bool
    merge_state: str
    checks_summary: str
    priority: str


def _parse_time(value: str) -> dt.datetime:
    if not value:
        return dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _age_label(updated_at: dt.datetime, now: dt.datetime) -> str:
    seconds = max(0, int((now - updated_at).total_seconds()))
    if seconds < 3600:
        minutes = max(1, seconds // 60)
        return f"{minutes}m"
    if seconds < 86_400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86_400}d"


def _checks_summary(raw_checks: Any) -> str:
    if not raw_checks:
        return "checks unknown"
    nodes = raw_checks.get("nodes", []) if isinstance(raw_checks, dict) else []
    if not nodes:
        return "checks unknown"
    counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        state = node.get("conclusion") or node.get("status") or "UNKNOWN"
        counts[str(state).lower()] += 1
    if not counts:
        return "checks unknown"
    return ", ".join(f"{count} {state}" for state, count in sorted(counts.items()))


def _priority(is_draft: bool, review_decision: str, merge_state: str, age_days: int) -> str:
    merge_state = merge_state.upper()
    review_decision = review_decision.upper()
    if is_draft or merge_state in {"DIRTY", "BLOCKED", "UNKNOWN"} or age_days >= 7:
        return "Stale / blocked"
    if review_decision in {"REVIEW_REQUIRED", "CHANGES_REQUESTED", ""}:
        return "Review now"
    return "Approved / low-touch"


def normalize_pull_requests(raw_prs: Iterable[dict[str, Any]], *, now: dt.datetime | None = None) -> list[PullRequest]:
    """Normalize GitHub PR JSON and sort by Joe's morning-review priority."""
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    normalized: list[PullRequest] = []
    for raw in raw_prs:
        updated_at = _parse_time(str(raw.get("updatedAt") or ""))
        age_days = max(0, int((now - updated_at).total_seconds() // 86_400))
        review_decision = str(raw.get("reviewDecision") or "")
        merge_state = str(raw.get("mergeStateStatus") or "UNKNOWN")
        is_draft = bool(raw.get("isDraft", False))
        author = raw.get("author") or {}
        author_login = author.get("login", "unknown") if isinstance(author, dict) else str(author)
        normalized.append(
            PullRequest(
                number=int(raw.get("number", 0)),
                title=str(raw.get("title") or "(untitled)"),
                url=str(raw.get("url") or ""),
                branch=str(raw.get("headRefName") or ""),
                author=author_login,
                updated_at=updated_at,
                age_label=_age_label(updated_at, now),
                review_decision=review_decision or "UNKNOWN",
                is_draft=is_draft,
                merge_state=merge_state,
                checks_summary=_checks_summary(raw.get("statusCheckRollup")),
                priority=_priority(is_draft, review_decision, merge_state, age_days),
            )
        )

    priority_rank = {name: idx for idx, name in enumerate(PRIORITY_ORDER)}
    return sorted(normalized, key=lambda pr: (priority_rank[pr.priority], pr.updated_at), reverse=False)


def build_brief(raw_prs: Sequence[dict[str, Any]], *, now: dt.datetime | None = None, limit: int = 10) -> str:
    """Return a Markdown morning brief, or `[SILENT]` when there are no PRs."""
    prs = normalize_pull_requests(raw_prs, now=now)
    if not prs:
        return "[SILENT]"

    selected = prs[:limit]
    omitted = max(0, len(prs) - len(selected))
    lines = ["# Morning PR review brief", "", f"Open PRs scanned: {len(prs)}", ""]

    for priority in PRIORITY_ORDER:
        bucket = [pr for pr in selected if pr.priority == priority]
        if not bucket:
            continue
        lines.extend([f"## {priority}", ""])
        for pr in bucket:
            draft = " draft" if pr.is_draft else ""
            lines.append(f"- [#{pr.number} {pr.title}]({pr.url}) — {pr.age_label} ago{draft}")
            lines.append(
                f"  - branch `{pr.branch}` by @{pr.author}; review `{pr.review_decision}`; "
                f"merge `{pr.merge_state}`; {pr.checks_summary}"
            )
        lines.append("")

    if omitted:
        plural = "s" if omitted != 1 else ""
        lines.append(f"_{omitted} additional PR{plural} omitted by --limit._")
    return "\n".join(lines).rstrip() + "\n"


def _fetch_open_pr_json() -> str:
    cmd = ["gh", "pr", "list", "--state", "open", "--limit", "25", "--json", GH_FIELDS]
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(3):
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1 + attempt)
    stderr = (last_error.stderr or "").strip() if last_error else ""
    raise RuntimeError(f"gh pr list failed after retries: {stderr or last_error}")


def _load_json(path: str | None) -> list[dict[str, Any]]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        # Many automation shells attach a non-interactive stdin even when no
        # pipe is provided.  Treat empty stdin as "use live gh" rather than as
        # "there are no PRs", otherwise cron smoke checks falsely go silent.
        text = sys.stdin.read()
        if not text.strip():
            text = _fetch_open_pr_json()
    else:
        text = _fetch_open_pr_json()
    data = json.loads(text or "[]")
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of pull requests")
    return [item for item in data if isinstance(item, dict)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a compact morning PR review brief.")
    parser.add_argument("--json-file", help="Read gh-pr-list JSON from this file instead of stdin/live gh.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum PRs to include in the brief.")
    args = parser.parse_args(argv)

    try:
        raw_prs = _load_json(args.json_file)
        print(build_brief(raw_prs, limit=max(1, args.limit)), end="")
    except Exception as exc:  # pragma: no cover - CLI guardrail
        print(f"morning_pr_brief failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
