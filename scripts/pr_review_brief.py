#!/usr/bin/env python3
"""Build a concise morning PR review brief from exported GitHub PR metadata.

Example input:
    gh pr list --json number,title,state,url,author,isDraft,reviewDecision,\
statusCheckRollup,additions,deletions,updatedAt > /tmp/prs.json
    python scripts/pr_review_brief.py --input /tmp/prs.json --silent-empty

The script is intentionally local-first: it does not call GitHub, send messages,
or mutate repo state. Feed it JSON from `gh`, a webhook export, or stdin.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

SILENT_TOKEN = "[SILENT]"
OPEN_STATES = {"OPEN"}
FAILED_CHECK_CONCLUSIONS = {"FAILURE", "FAILED", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED"}
PENDING_CHECK_STATUSES = {"QUEUED", "IN_PROGRESS", "PENDING", "WAITING", "REQUESTED"}


@dataclass(frozen=True)
class PullRequest:
    number: int | None
    title: str
    state: str
    url: str
    author: str
    is_draft: bool
    review_decision: str
    failed_checks: tuple[str, ...]
    pending_checks: tuple[str, ...]
    changed_lines: int
    updated_at: datetime | None


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _author_login(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("login") or value.get("name") or "unknown")
    if value:
        return str(value)
    return "unknown"


def _iter_check_items(raw_rollup: Any) -> Iterable[dict[str, Any]]:
    if isinstance(raw_rollup, list):
        for item in raw_rollup:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(raw_rollup, dict):
        nodes = raw_rollup.get("nodes")
        if isinstance(nodes, list):
            for item in nodes:
                if isinstance(item, dict):
                    yield item
        elif raw_rollup:
            yield raw_rollup


def _check_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("context") or item.get("workflowName") or "check")


def _check_state(item: dict[str, Any]) -> tuple[str, str]:
    conclusion = str(item.get("conclusion") or "").upper()
    status = str(item.get("status") or item.get("state") or "").upper()
    return conclusion, status


def _check_summary(raw_rollup: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    failed: list[str] = []
    pending: list[str] = []
    for item in _iter_check_items(raw_rollup):
        conclusion, status = _check_state(item)
        name = _check_name(item)
        if conclusion in FAILED_CHECK_CONCLUSIONS:
            failed.append(name)
        elif status in PENDING_CHECK_STATUSES or (status and not conclusion and status != "COMPLETED"):
            pending.append(name)
    return tuple(failed), tuple(pending)


def _unwrap_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "pullRequests", "pull_requests", "prs", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def load_items(path: str | None) -> list[Any]:
    """Load JSON PR records from a path or stdin."""
    text = sys.stdin.read() if not path or path == "-" else Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return []
    return _unwrap_items(json.loads(text))


def normalize_pr(item: Any) -> PullRequest | None:
    if not isinstance(item, dict):
        return None
    failed_checks, pending_checks = _check_summary(
        item.get("statusCheckRollup")
        or item.get("status_check_rollup")
        or item.get("checks")
        or item.get("check_runs")
    )
    return PullRequest(
        number=_as_int(item.get("number")) or None,
        title=str(item.get("title") or "Untitled PR"),
        state=str(item.get("state") or "OPEN").upper(),
        url=str(item.get("url") or item.get("html_url") or ""),
        author=_author_login(item.get("author") or item.get("user")),
        is_draft=bool(item.get("isDraft") or item.get("draft") or item.get("is_draft")),
        review_decision=str(item.get("reviewDecision") or item.get("review_decision") or "").upper(),
        failed_checks=failed_checks,
        pending_checks=pending_checks,
        changed_lines=_as_int(item.get("additions")) + _as_int(item.get("deletions")),
        updated_at=_parse_time(item.get("updatedAt") or item.get("updated_at")),
    )


def _age_hours(pr: PullRequest, now: datetime) -> int | None:
    if pr.updated_at is None:
        return None
    delta = now - pr.updated_at
    return max(0, int(delta.total_seconds() // 3600))


def score_pr(pr: PullRequest, now: datetime) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    decision = pr.review_decision.replace("_", " ").lower()

    if pr.is_draft:
        score -= 20
        reasons.append("draft")
    if pr.failed_checks:
        score += 45
        reasons.append("failed checks")
    if pr.review_decision == "REVIEW_REQUIRED":
        score += 35
        reasons.append("review required")
    elif pr.review_decision == "CHANGES_REQUESTED":
        score += 30
        reasons.append("changes requested")
    elif pr.review_decision == "APPROVED":
        score -= 10
        reasons.append("approved")
    elif decision:
        score += 10
        reasons.append(decision)
    if pr.pending_checks:
        score += 8
        reasons.append("checks pending")

    age = _age_hours(pr, now)
    if age is not None:
        if age >= 72:
            score += 18
            reasons.append(f"stale {age}h")
        elif age >= 24:
            score += 8
            reasons.append(f"waiting {age}h")

    if pr.changed_lines >= 800:
        score += 12
        reasons.append("large diff")
    elif pr.changed_lines <= 50:
        score += 5
        reasons.append("small diff")

    if not reasons:
        reasons.append("open")
    return score, reasons


def _format_pr(pr: PullRequest, rank: int, now: datetime) -> str:
    score, reasons = score_pr(pr, now)
    title = f"#{pr.number}" if pr.number is not None else "PR"
    if pr.url:
        title = f"[{title}]({pr.url})"
    age = _age_hours(pr, now)
    age_text = "unknown age" if age is None else f"updated {age}h ago"
    check_text = ""
    if pr.failed_checks:
        check_text = f"; failed: {', '.join(pr.failed_checks[:3])}"
    elif pr.pending_checks:
        check_text = f"; pending: {', '.join(pr.pending_checks[:3])}"
    return (
        f"{rank}. {title} — {pr.title}\n"
        f"   - Priority: {score} ({', '.join(reasons)})\n"
        f"   - Author: {pr.author}; size: {pr.changed_lines} LOC; {age_text}{check_text}"
    )


def build_brief(items: Iterable[Any], *, now: str | datetime | None = None, silent_empty: bool = False) -> str:
    current = _parse_time(now) if isinstance(now, str) else now
    if current is None:
        current = datetime.now(UTC)
    prs = [pr for item in items if (pr := normalize_pr(item)) and pr.state in OPEN_STATES]
    if not prs:
        return SILENT_TOKEN if silent_empty else "# PR Review Brief\n\nNo open PRs found."

    prs.sort(key=lambda pr: score_pr(pr, current)[0], reverse=True)
    lines = ["# PR Review Brief", "", f"Open PRs: {len(prs)}", "", "## Review Queue"]
    lines.extend(_format_pr(pr, index, current) for index, pr in enumerate(prs, start=1))
    lines.extend(
        [
            "",
            "## Suggested next move",
            "- Start with failed checks / changes-requested PRs; they have the highest coordination cost if ignored.",
            "- Batch small clean PRs after blockers so review momentum compounds.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", help="JSON file to read; omit or use '-' for stdin")
    parser.add_argument("--now", help="ISO timestamp for deterministic age scoring")
    parser.add_argument("--silent-empty", action="store_true", help="print exactly [SILENT] when no open PRs exist")
    args = parser.parse_args(argv)

    print(build_brief(load_items(args.input), now=args.now, silent_empty=args.silent_empty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
