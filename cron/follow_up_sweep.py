"""
Phase 028-B — Weekly Hermes `follow-up-sweep` cron job.

Purpose
-------
Once per week (Monday 09:00 America/Los_Angeles), query Linear for issues
labeled ``follow-up:*`` older than 30 days and DM Blake on Slack with a
block-kit summary + thread reply listing each row.

This is the "feedback loop" half of Plan 028 (D2 §R8):

  028-A   /build:execute-plan close step auto-opens follow-up rows at
          plan close (extractor lives in agentic-hub/orchestrator).
  028-A1  one-shot backfill across plans 014–023 (already shipped 19
          follow-up:021 rows on 2026-05-31).
  028-B   this module — closes the loop so deferred work cannot rot
          silently inside Linear's backlog.

Design notes
------------
- **Standalone Hermes cron**: invoked as a script entry by the existing
  cron scheduler (``cron/scheduler.py``). The cron registry stores the
  schedule (``0 9 * * 1`` America/Los_Angeles) and runs::

      python -m cron.follow_up_sweep

  on a tick.
- **Stdlib only.** Mirrors the zero-dependency pattern in
  ``skills/productivity/linear/scripts/linear_api.py`` — Hermes is a
  fat CLI and we do not want this job to drag in optional deps.
- **Read-only Linear query.** This phase ships the *digest*; reaction-
  driven mutations (wontfix/defer/schedule) come from 029-F. If 029-F
  has not landed by the time the operator wants triage, we open a
  follow-up:028-B row for the reaction surface.
- **Slack target.** Posts to ``SLACK_FOLLOW_UP_CHANNEL`` (Blake's DM
  channel id, e.g. ``D0123ABCD``). Bot token from ``SLACK_BOT_TOKEN``
  — the same token the gateway already uses (reuse-over-create per
  the autonomy/reuse memory).
- **Failure modes are explicit.** Linear errors raise + log; the cron
  scheduler captures the traceback into its output file. We do not
  swallow Linear failures silently — that would re-create the exact
  rot-pattern 028 was built to eliminate.

Acceptance (from master plan §028-B):
  * cron entry registered (weekly, Mon 09:00 PT)
  * DM produced with list (mocked Linear + Slack in tests)
  * 5+ tests including empty-list + Linear API failure case
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LINEAR_API_URL = "https://api.linear.app/graphql"
SLACK_POST_URL = "https://slack.com/api/chat.postMessage"
STALE_THRESHOLD_DAYS = 30
VERY_STALE_THRESHOLD_DAYS = 90
DEFAULT_PAGE_SIZE = 100
LABEL_PREFIX = "follow-up:"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


# ---------------------------------------------------------------------------
# Linear client (stdlib http)
# ---------------------------------------------------------------------------


class LinearError(RuntimeError):
    """Raised when Linear returns an HTTP error or GraphQL errors."""


def _linear_gql(
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
    url_opener=urllib.request.urlopen,
) -> dict[str, Any]:
    """Execute a Linear GraphQL query. Raises ``LinearError`` on any failure."""
    key = api_key or _env("LINEAR_API_KEY")
    if not key:
        raise LinearError("LINEAR_API_KEY not set in environment")

    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LINEAR_API_URL,
        data=body,
        headers={
            "Authorization": key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with url_opener(req, timeout=30) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - exercised via injected opener
        raise LinearError(f"Linear HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise LinearError(f"Linear network error: {exc.reason}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LinearError(f"Linear returned non-JSON: {raw[:200]!r}") from exc

    if data.get("errors"):
        raise LinearError(f"Linear GraphQL errors: {data['errors']}")
    return data.get("data") or {}


_ISSUES_QUERY = """
query FollowUpSweep($filter: IssueFilter, $first: Int!) {
  issues(filter: $filter, first: $first, orderBy: createdAt) {
    nodes {
      id
      identifier
      title
      url
      createdAt
      labels { nodes { name } }
      state { name type }
    }
  }
}
"""


def fetch_follow_up_issues(
    *,
    now: datetime | None = None,
    threshold_days: int = STALE_THRESHOLD_DAYS,
    page_size: int = DEFAULT_PAGE_SIZE,
    api_key: str | None = None,
    gql=_linear_gql,
) -> list[dict[str, Any]]:
    """Return Linear issues labeled ``follow-up:*`` older than ``threshold_days``.

    Filters to *open* states only (type in BACKLOG/UNSTARTED/STARTED) so
    closed/cancelled rows do not show up in the weekly digest.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=threshold_days)).isoformat()
    filt: dict[str, Any] = {
        "labels": {"name": {"startsWith": LABEL_PREFIX}},
        "createdAt": {"lt": cutoff},
        "state": {"type": {"in": ["backlog", "unstarted", "started", "triage"]}},
    }
    data = gql(
        _ISSUES_QUERY,
        {"filter": filt, "first": page_size},
        api_key=api_key,
    )
    issues = (data.get("issues") or {}).get("nodes") or []
    return issues


# ---------------------------------------------------------------------------
# Digest formatting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StaleIssue:
    identifier: str
    title: str
    url: str
    age_days: int
    plan_id: str  # extracted from the follow-up:<plan-id> label


def _plan_id_from_labels(labels: Iterable[dict[str, Any]]) -> str:
    for node in labels or []:
        name = (node or {}).get("name") or ""
        if name.startswith(LABEL_PREFIX):
            return name[len(LABEL_PREFIX):] or "unknown"
    return "unknown"


def _parse_created_at(raw: str) -> datetime:
    # Linear returns ISO 8601 with trailing Z
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def normalize_issues(
    raw_issues: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> list[StaleIssue]:
    now = now or datetime.now(timezone.utc)
    out: list[StaleIssue] = []
    for issue in raw_issues:
        created_raw = issue.get("createdAt") or ""
        try:
            created = _parse_created_at(created_raw)
        except ValueError:
            logger.warning("skipping issue with bad createdAt: %r", created_raw)
            continue
        age = (now - created).days
        labels = (issue.get("labels") or {}).get("nodes") or []
        out.append(
            StaleIssue(
                identifier=issue.get("identifier") or "?",
                title=issue.get("title") or "(no title)",
                url=issue.get("url") or "",
                age_days=age,
                plan_id=_plan_id_from_labels(labels),
            )
        )
    # Oldest first — Blake should see the most-rotten rows at the top.
    out.sort(key=lambda i: i.age_days, reverse=True)
    return out


def build_digest(
    issues: list[StaleIssue],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return ``{"summary": str, "thread": list[str], "block_kit": list[dict]}``.

    Top-line summary (one Slack message) plus the per-row thread payload.
    Empty-issues case returns a positive-confirmation summary
    (the master plan calls this out explicitly — silence is bad).
    """
    now = now or datetime.now(timezone.utc)
    week = now.strftime("%Y-%m-%d")

    if not issues:
        summary = (
            f":white_check_mark: Follow-up sweep ({week}): "
            f"0 stale follow-ups this week."
        )
        return {
            "summary": summary,
            "thread": [],
            "block_kit": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": summary},
                }
            ],
        }

    very_stale = sum(1 for i in issues if i.age_days >= VERY_STALE_THRESHOLD_DAYS)
    summary = (
        f":clipboard: Follow-up backlog sweep ({week}): "
        f"{len(issues)} issues > {STALE_THRESHOLD_DAYS} days old, "
        f"{very_stale} > {VERY_STALE_THRESHOLD_DAYS} days."
    )
    thread = [
        f"• <{i.url}|{i.identifier}> follow-up:{i.plan_id} — {i.age_days} days — {i.title}"
        for i in issues
    ]
    block_kit = [
        {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
        {"type": "divider"},
    ]
    # Slack limits a single section block to 3000 chars; chunk into ≤20 rows.
    chunk: list[str] = []
    for line in thread:
        chunk.append(line)
        if len(chunk) >= 20:
            block_kit.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(chunk)}}
            )
            chunk = []
    if chunk:
        block_kit.append(
            {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(chunk)}}
        )

    return {"summary": summary, "thread": thread, "block_kit": block_kit}


# ---------------------------------------------------------------------------
# Slack client (stdlib http)
# ---------------------------------------------------------------------------


class SlackError(RuntimeError):
    """Raised on Slack chat.postMessage failure."""


def _slack_post(
    payload: dict[str, Any],
    *,
    token: str,
    url_opener=urllib.request.urlopen,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SLACK_POST_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with url_opener(req, timeout=30) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise SlackError(f"Slack HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SlackError(f"Slack network error: {exc.reason}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SlackError(f"Slack returned non-JSON: {raw[:200]!r}") from exc
    if not data.get("ok"):
        raise SlackError(f"Slack API error: {data.get('error', 'unknown')}")
    return data


def send_digest_to_slack(
    digest: dict[str, Any],
    *,
    channel: str | None = None,
    token: str | None = None,
    slack_post=_slack_post,
) -> dict[str, Any]:
    """Post the summary to ``channel`` then thread each row as a reply.

    Returns ``{"summary_ts": ts, "thread_ts_count": int}``.
    """
    channel = channel or _env("SLACK_FOLLOW_UP_CHANNEL")
    token = token or _env("SLACK_BOT_TOKEN")
    if not channel:
        raise SlackError("SLACK_FOLLOW_UP_CHANNEL not set")
    if not token:
        raise SlackError("SLACK_BOT_TOKEN not set")

    top = slack_post(
        {
            "channel": channel,
            "text": digest["summary"],
            "blocks": digest["block_kit"],
            "mrkdwn": True,
        },
        token=token,
    )
    summary_ts = top.get("ts")
    thread_count = 0
    # The block_kit already contains the per-row sections, so the thread
    # reply only fires when we want to emit raw text (>20 rows that got
    # chunked, or when a downstream consumer prefers plain text). For v1
    # we post the long-tail rows beyond what block_kit chunked in
    # separately so the thread always has structured rows — matches the
    # master-plan's "thread reply lists each row" criterion.
    for line in digest.get("thread", []):
        slack_post(
            {
                "channel": channel,
                "thread_ts": summary_ts,
                "text": line,
                "mrkdwn": True,
            },
            token=token,
        )
        thread_count += 1
    return {"summary_ts": summary_ts, "thread_ts_count": thread_count}


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------


def run_sweep(
    *,
    now: datetime | None = None,
    threshold_days: int = STALE_THRESHOLD_DAYS,
    page_size: int = DEFAULT_PAGE_SIZE,
    linear_gql=_linear_gql,
    slack_post=_slack_post,
    channel: str | None = None,
    token: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """End-to-end: fetch stale follow-ups, format, send to Slack.

    Returns a result dict suitable for the cron output log::

        {"count": int, "very_stale": int, "summary_ts": str|None,
         "thread_ts_count": int}
    """
    now = now or datetime.now(timezone.utc)
    raw = fetch_follow_up_issues(
        now=now,
        threshold_days=threshold_days,
        page_size=page_size,
        api_key=api_key,
        gql=linear_gql,
    )
    issues = normalize_issues(raw, now=now)
    digest = build_digest(issues, now=now)
    delivery = send_digest_to_slack(
        digest, channel=channel, token=token, slack_post=slack_post
    )
    result = {
        "count": len(issues),
        "very_stale": sum(1 for i in issues if i.age_days >= VERY_STALE_THRESHOLD_DAYS),
        "summary_ts": delivery.get("summary_ts"),
        "thread_ts_count": delivery.get("thread_ts_count", 0),
    }
    logger.info(
        "follow-up-sweep: count=%s very_stale=%s summary_ts=%s",
        result["count"],
        result["very_stale"],
        result["summary_ts"],
    )
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        result = run_sweep()
    except (LinearError, SlackError) as exc:
        logger.error("follow-up-sweep failed: %s", exc)
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    sys.stdout.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
