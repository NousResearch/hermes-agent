#!/usr/bin/env python3
"""Render a Joe-style Traditional Chinese review-debt brief from PR JSON.

Input is intentionally local-first: export metadata with a command such as

    gh pr list --json number,title,url,state,isDraft,mergedAt,createdAt,updatedAt,author,reviewRequests,reviewDecision,statusCheckRollup > prs.json

Then run this script against the saved JSON. It never mutates GitHub state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SILENT = "[SILENT]"
DEFAULT_JOE_LOGINS = {"joe102084"}


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    url: str
    state: str = "OPEN"
    is_draft: bool = False
    merged_at: str | None = None
    closed_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    author: str = "unknown"
    review_requests: tuple[str, ...] = ()
    review_decision: str = ""
    check_state: str = "unknown"

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "PullRequest":
        return cls(
            number=int(raw.get("number") or 0),
            title=str(raw.get("title") or "(untitled)"),
            url=str(raw.get("url") or raw.get("permalink") or ""),
            state=str(raw.get("state") or "OPEN").upper(),
            is_draft=bool(raw.get("isDraft") or raw.get("draft") or False),
            merged_at=_optional_str(raw.get("mergedAt") or raw.get("merged_at")),
            closed_at=_optional_str(raw.get("closedAt") or raw.get("closed_at")),
            created_at=_optional_str(raw.get("createdAt") or raw.get("created_at")),
            updated_at=_optional_str(raw.get("updatedAt") or raw.get("updated_at")),
            author=_extract_login(raw.get("author")) or "unknown",
            review_requests=tuple(_extract_review_requests(raw.get("reviewRequests"))),
            review_decision=str(raw.get("reviewDecision") or "").upper(),
            check_state=_extract_check_state(raw.get("statusCheckRollup")),
        )


@dataclass(frozen=True)
class ReviewItem:
    pr: PullRequest
    priority: str
    score: int
    age_days: int
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def number(self) -> int:
        return self.pr.number


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_login(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("login", "name", "username"):
            if value.get(key):
                return str(value[key])
    return ""


def _extract_review_requests(value: Any) -> list[str]:
    if not value:
        return []
    candidates: list[Any]
    if isinstance(value, dict):
        candidates = []
        for key in ("users", "nodes", "teams"):
            nested = value.get(key)
            if isinstance(nested, list):
                candidates.extend(nested)
    elif isinstance(value, list):
        candidates = value
    else:
        return []

    logins: list[str] = []
    for item in candidates:
        login = _extract_login(item)
        if login:
            logins.append(login)
    return sorted(set(logins), key=str.lower)


def _extract_check_state(value: Any) -> str:
    if not value:
        return "unknown"
    runs: Iterable[Any]
    if isinstance(value, dict):
        runs = value.get("nodes") or value.get("checkRuns") or value.get("contexts") or []
    elif isinstance(value, list):
        runs = value
    else:
        return str(value).lower()

    states: list[str] = []
    for run in runs:
        if isinstance(run, dict):
            states.append(
                str(
                    run.get("conclusion")
                    or run.get("state")
                    or run.get("status")
                    or "unknown"
                ).lower()
            )
        else:
            states.append(str(run).lower())
    if any(state in {"failure", "failed", "error", "timed_out", "cancelled"} for state in states):
        return "failure"
    if any(state in {"pending", "queued", "in_progress", "waiting", "requested"} for state in states):
        return "pending"
    if states and all(state in {"success", "completed", "neutral", "skipped"} for state in states):
        return "success"
    return "unknown"


def _raw_prs_from_json(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = []
        for key in ("pullRequests", "prs", "items", "nodes"):
            if isinstance(data.get(key), list):
                raw = data[key]
                break
    else:
        raw = []
    return [item for item in raw if isinstance(item, dict)]


def load_pull_requests(path: Path) -> list[PullRequest]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [PullRequest.from_raw(raw) for raw in _raw_prs_from_json(data)]


def _age_days(pr: PullRequest, now: datetime) -> int:
    created = parse_datetime(pr.created_at)
    if created is None:
        return 0
    return max(0, (now - created).days)


def _is_actionable(pr: PullRequest) -> bool:
    return pr.state == "OPEN" and not pr.is_draft and not pr.merged_at and not pr.closed_at


def _score(pr: PullRequest, *, joe_logins: set[str], now: datetime) -> ReviewItem:
    score = 0
    evidence: list[str] = []
    normalized_requests = {login.lower() for login in pr.review_requests}
    normalized_joe = {login.lower() for login in joe_logins}

    if normalized_requests & normalized_joe:
        score += 60
        evidence.append("review request includes Joe")
    if pr.check_state == "failure":
        score += 35
        evidence.append("failing checks")
    elif pr.check_state == "pending":
        score += 10
        evidence.append("checks pending")

    if pr.review_decision in {"CHANGES_REQUESTED", "REVIEW_REQUIRED"}:
        score += 25
        evidence.append(f"reviewDecision={pr.review_decision}")

    age = _age_days(pr, now)
    if age >= 14:
        score += 20
        evidence.append(f"stale {age}d")
    elif age >= 7:
        score += 10
        evidence.append(f"open {age}d")

    if not evidence:
        evidence.append("open PR")

    if score >= 90:
        priority = "P0"
    elif score >= 50:
        priority = "P1"
    elif score >= 20:
        priority = "P2"
    else:
        priority = "P3"

    return ReviewItem(pr=pr, priority=priority, score=score, age_days=age, evidence=tuple(evidence))


def actionable_items(
    pull_requests: Sequence[PullRequest],
    *,
    joe_logins: set[str] | None = None,
    now: datetime | None = None,
) -> list[ReviewItem]:
    joe_logins = joe_logins or DEFAULT_JOE_LOGINS
    now = now or datetime.now(timezone.utc)
    items = [_score(pr, joe_logins=joe_logins, now=now) for pr in pull_requests if _is_actionable(pr)]
    return sorted(items, key=lambda item: (-item.score, -item.age_days, item.pr.number))


def render_markdown(items: Sequence[ReviewItem], *, silent_if_empty: bool = False) -> str:
    if not items:
        if silent_if_empty:
            return SILENT
        return "## TL;DR\n- 沒有可行動的 PR review debt。\n"

    top = items[0]
    p0_p1 = sum(1 for item in items if item.priority in {"P0", "P1"})
    lines: list[str] = [
        "## TL;DR",
        f"- 目前有 {len(items)} 個可行動 PR；其中 {p0_p1} 個是 P0/P1。",
        f"- 最優先：#{top.pr.number} `{top.pr.title}`（{top.priority}, score {top.score}）。",
        "",
        "## Fact / verified",
    ]

    for item in items:
        pr = item.pr
        requests = ", ".join(pr.review_requests) if pr.review_requests else "無"
        lines.extend(
            [
                f"- **{item.priority} / score {item.score}** #{pr.number} {pr.title}",
                f"  - 年齡：{item.age_days} 天；checks：{pr.check_state}；reviewDecision：{pr.review_decision or '無'}",
                f"  - 作者：{pr.author}；review requests：{requests}",
                f"  - 證據：{'; '.join(item.evidence)}",
                f"  - URL：{pr.url or '未提供'}",
            ]
        )

    lines.extend(
        [
            "",
            "## Hypothesis",
            "- 分數高的項目通常代表 Joe 被明確 request、CI 已紅、或 PR 已經放太久；早上先處理這些可以降低 review queue 的認知負債。",
            "",
            "## Action for Joe",
            "- 先打開 P0/P1：決定要 review、要求作者修 CI，或直接略過不看。",
            "- P2/P3 只在 5 分鐘內能快速處理時才碰；不要讓低優先 review 吃掉深度工作時段。",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path, help="Path to JSON exported from gh pr list")
    parser.add_argument(
        "--joe-login",
        action="append",
        default=[],
        help="GitHub login that counts as Joe's direct review request; repeatable",
    )
    parser.add_argument("--now", help="Override current UTC time for deterministic output")
    parser.add_argument("--silent-if-empty", action="store_true", help="Print exact [SILENT] when no actionable PRs exist")
    parser.add_argument("--limit", type=int, default=10, help="Maximum PRs to render (default: 10)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    joe_logins = set(args.joe_login) if args.joe_login else DEFAULT_JOE_LOGINS
    now = parse_datetime(args.now) if args.now else datetime.now(timezone.utc)
    if now is None:
        raise SystemExit(f"Invalid --now value: {args.now}")
    items = actionable_items(load_pull_requests(args.json_path), joe_logins=joe_logins, now=now)
    if args.limit >= 0:
        items = items[: args.limit]
    print(render_markdown(items, silent_if_empty=args.silent_if_empty), end="\n" if items or args.silent_if_empty else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
