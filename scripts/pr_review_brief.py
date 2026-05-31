#!/usr/bin/env python3
"""Render a Joe-style morning review brief from local PR metadata.

Input is intentionally local-first: export PR data with a command such as:

    gh pr list --repo OWNER/REPO --state open \
      --json number,title,url,state,isDraft,createdAt,updatedAt,author,reviewRequests,reviewDecision,statusCheckRollup

Then run this script against the saved JSON file. The script does not call
GitHub, create reviews, send messages, or mutate any data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REVIEWER = "joe102084"
FAILED_CONCLUSIONS = {"ACTION_REQUIRED", "CANCELLED", "FAILURE", "STARTUP_FAILURE", "TIMED_OUT"}
PENDING_CONCLUSIONS = {"EXPECTED", "IN_PROGRESS", "PENDING", "QUEUED", "REQUESTED", "WAITING"}


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    url: str
    state: str = "OPEN"
    author: str = "unknown"
    created_at: str = ""
    updated_at: str = ""
    review_requests: tuple[str, ...] = ()
    review_decision: str = ""
    check_state: str = "unknown"
    age_days: int = 0
    priority_score: int = 0
    priority_label: str = "一般追蹤"
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _parse_today(today: str | date | None) -> date:
    if isinstance(today, date):
        return today
    if today:
        return date.fromisoformat(today)
    return datetime.now(timezone.utc).date()


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).date()
        except ValueError:
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
    return None


def _login(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("login", "name", "slug"):
            if value.get(key):
                return str(value[key])
    return ""


def _extract_rows(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        for key in ("pullRequests", "pull_requests", "prs", "items", "data"):
            rows = raw.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise ValueError("expected a JSON list of PRs or an object with pullRequests/prs/items")


def _review_requests(row: dict[str, Any]) -> tuple[str, ...]:
    raw = row.get("reviewRequests") or row.get("review_requests") or []
    if isinstance(raw, dict):
        raw = raw.get("nodes") or raw.get("users") or raw.get("teams") or []
    if not isinstance(raw, list):
        return ()
    names = []
    for item in raw:
        if isinstance(item, dict) and "requestedReviewer" in item:
            item = item["requestedReviewer"]
        name = _login(item)
        if name:
            names.append(name)
    return tuple(names)


def _check_state(row: dict[str, Any]) -> str:
    raw = row.get("statusCheckRollup") or row.get("status_check_rollup") or row.get("checks") or []
    if isinstance(raw, dict):
        raw = raw.get("nodes") or raw.get("contexts") or raw.get("checkRuns") or []
    if not isinstance(raw, list) or not raw:
        return "unknown"

    states: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            value = item.get("conclusion") or item.get("state") or item.get("status")
            if value:
                states.append(str(value).upper())
    if any(state in FAILED_CONCLUSIONS for state in states):
        return "failed"
    if any(state in PENDING_CONCLUSIONS for state in states):
        return "pending"
    if states and all(state in {"SUCCESS", "COMPLETED", "NEUTRAL", "SKIPPED"} for state in states):
        return "passing"
    return "unknown"


def _normalize_pr(row: dict[str, Any], today: date) -> PullRequest | None:
    state = str(row.get("state") or "OPEN").upper()
    if state not in {"OPEN", ""}:
        return None
    if row.get("isDraft") or row.get("draft"):
        return None
    if row.get("mergedAt") or row.get("closedAt"):
        return None

    number = int(row.get("number") or row.get("id") or 0)
    if number <= 0:
        return None

    created_at = str(row.get("createdAt") or row.get("created_at") or "")
    created_date = _parse_date(created_at)
    age_days = max((today - created_date).days, 0) if created_date else 0

    author = _login(row.get("author")) or str(row.get("author") or "unknown")
    return PullRequest(
        number=number,
        title=str(row.get("title") or "(untitled)"),
        url=str(row.get("url") or row.get("html_url") or ""),
        state=state or "OPEN",
        author=author,
        created_at=created_at,
        updated_at=str(row.get("updatedAt") or row.get("updated_at") or ""),
        review_requests=_review_requests(row),
        review_decision=str(row.get("reviewDecision") or row.get("review_decision") or "").upper(),
        check_state=_check_state(row),
        age_days=age_days,
    )


def load_prs(path: str | Path, today: str | date | None = None) -> list[PullRequest]:
    """Load open, non-draft PRs from a local JSON export."""

    today_date = _parse_today(today)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    prs = [_normalize_pr(row, today_date) for row in _extract_rows(raw)]
    return [pr for pr in prs if pr is not None]


def _score(pr: PullRequest, reviewer: str) -> tuple[int, str, tuple[str, ...]]:
    score = pr.age_days
    label = "一般追蹤"
    reasons: list[str] = []

    if reviewer and reviewer in pr.review_requests:
        score += 100
        label = "需要 Joe Review"
        reasons.append(f"review request 指向 {reviewer}")
    if pr.check_state == "failed":
        score += 80
        if label == "一般追蹤":
            label = "CI 失敗"
        reasons.append("CI / checks 有失敗訊號")
    elif pr.check_state == "pending":
        score += 20
        reasons.append("CI / checks 尚未完成")
    elif pr.check_state == "passing":
        score += 10
        reasons.append("checks 目前通過，可進入人工判斷")

    if pr.review_decision in {"REVIEW_REQUIRED", "CHANGES_REQUESTED"}:
        score += 50
        if label == "一般追蹤":
            label = "需要 Review 決策"
        reasons.append(f"reviewDecision={pr.review_decision}")
    elif pr.review_decision == "APPROVED":
        score += 15
        reasons.append("已核可，適合檢查是否可收尾")

    if pr.age_days >= 7:
        score += 25
        reasons.append(f"已開 {pr.age_days} 天，避免積壓")
    elif pr.age_days >= 3:
        score += 10
        reasons.append(f"已開 {pr.age_days} 天")

    if not reasons:
        reasons.append("開放中，尚無高優先級阻塞訊號")
    return score, label, tuple(reasons)


def prioritize(prs: Iterable[PullRequest], reviewer: str = DEFAULT_REVIEWER) -> list[PullRequest]:
    """Return PRs sorted by deterministic review priority."""

    enriched = []
    for pr in prs:
        score, label, reasons = _score(pr, reviewer)
        enriched.append(
            PullRequest(
                number=pr.number,
                title=pr.title,
                url=pr.url,
                state=pr.state,
                author=pr.author,
                created_at=pr.created_at,
                updated_at=pr.updated_at,
                review_requests=pr.review_requests,
                review_decision=pr.review_decision,
                check_state=pr.check_state,
                age_days=pr.age_days,
                priority_score=score,
                priority_label=label,
                reasons=reasons,
            )
        )
    return sorted(enriched, key=lambda pr: (-pr.priority_score, -pr.age_days, pr.number))


def render_brief(
    prs: Iterable[PullRequest],
    reviewer: str = DEFAULT_REVIEWER,
    max_items: int = 5,
    silent_if_empty: bool = False,
) -> str:
    ranked = prioritize(prs, reviewer=reviewer)[:max_items]
    if not ranked:
        if silent_if_empty:
            return "[SILENT]"
        return "## TL;DR\n- Fact / verified：目前沒有需要列入 review queue 的開放 PR。"

    need_joe = sum(1 for pr in ranked if pr.priority_label == "需要 Joe Review")
    failed = sum(1 for pr in ranked if pr.check_state == "failed")
    stale = sum(1 for pr in ranked if pr.age_days >= 7)

    lines = [
        "## TL;DR",
        f"- Fact / verified：找到 {len(ranked)} 個開放 PR 候選；其中 {need_joe} 個指向 Joe review、{failed} 個 CI 失敗、{stale} 個已開超過 7 天。",
        "- Action for Joe：先處理標記為「需要 Joe Review」或「CI 失敗」的項目；其餘只做快速掃描。",
        "",
        "## Fact / verified",
    ]
    for index, pr in enumerate(ranked, start=1):
        reasons = "；".join(pr.reasons)
        review_requests = ", ".join(pr.review_requests) if pr.review_requests else "無"
        lines.extend(
            [
                f"{index}. #{pr.number} — {pr.title}",
                f"   - Priority：{pr.priority_label}（score={pr.priority_score}，age={pr.age_days}d，checks={pr.check_state}）",
                f"   - Author：{pr.author}；Review requests：{review_requests}",
                f"   - Evidence：{reasons}",
                f"   - URL：{pr.url}",
            ]
        )

    lines.extend(
        [
            "",
            "## Hypothesis",
            "- 這份排序不是語義 code review；它只根據本地 PR metadata 判斷「早上最值得先看的隊列」。",
            "- 如果 CI 狀態或 review request 欄位缺失，優先級會偏保守；建議匯出 gh JSON 時包含 statusCheckRollup/reviewRequests/reviewDecision。",
            "",
            "## Action for Joe",
            "- 10 分鐘內：先打開前 1–3 個 URL，決定 review / comment / ignore。",
            "- 若要更新資料：`gh pr list --repo OWNER/REPO --state open --json number,title,url,state,isDraft,createdAt,updatedAt,author,reviewRequests,reviewDecision,statusCheckRollup > prs.json`，再跑本 script。",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a Traditional Chinese PR review brief from local JSON metadata.")
    parser.add_argument("input", type=Path, help="Path to gh/API JSON export.")
    parser.add_argument("--today", help="Override today's date as YYYY-MM-DD for deterministic runs.")
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER, help="Reviewer login to prioritize. Default: joe102084.")
    parser.add_argument("--max-items", type=int, default=5, help="Maximum PRs to include. Default: 5.")
    parser.add_argument("--silent-if-empty", action="store_true", help="Print exact [SILENT] when there are no actionable PRs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prs = load_prs(args.input, today=args.today)
    print(render_brief(prs, reviewer=args.reviewer, max_items=args.max_items, silent_if_empty=args.silent_if_empty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
