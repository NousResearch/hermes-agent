"""
Tests for Phase 028-B — weekly follow-up sweep cron job.

Covers (per master plan §028-B acceptance criteria, 5+ tests):
  * happy path: ≥1 stale issue → DM with summary + per-row thread
  * empty list: 0 stale issues → positive-confirmation message (not silent)
  * Linear API failure surfaces as ``LinearError`` (does not silently noop)
  * Slack API failure surfaces as ``SlackError``
  * idempotent normalization: oldest-first sort + plan-id extraction
  * digest formatting: very-stale count, block-kit chunking >20 rows
  * env-validation: missing channel/token raises clear error
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cron import follow_up_sweep as fus


UTC = timezone.utc


def _issue(identifier, title, label, created_at, url=None, state_type="backlog"):
    return {
        "id": f"id-{identifier}",
        "identifier": identifier,
        "title": title,
        "url": url or f"https://linear.app/example/issue/{identifier}",
        "createdAt": created_at,
        "labels": {"nodes": [{"name": label}]},
        "state": {"name": "Backlog", "type": state_type},
    }


# ---------------------------------------------------------------------------
# fetch_follow_up_issues
# ---------------------------------------------------------------------------


def test_fetch_passes_label_prefix_and_cutoff_to_linear():
    captured = {}

    def fake_gql(query, variables, *, api_key=None):
        captured["query"] = query
        captured["variables"] = variables
        captured["api_key"] = api_key
        return {"issues": {"nodes": []}}

    now = datetime(2026, 5, 31, 9, 0, tzinfo=UTC)
    fus.fetch_follow_up_issues(
        now=now, threshold_days=30, api_key="lin_test", gql=fake_gql
    )

    filt = captured["variables"]["filter"]
    assert filt["labels"]["name"]["startsWith"] == "follow-up:"
    # cutoff is 30 days before "now"
    cutoff = filt["createdAt"]["lt"]
    assert cutoff.startswith("2026-05-01")
    assert filt["state"]["type"]["in"] == [
        "backlog",
        "unstarted",
        "started",
        "triage",
    ]
    assert captured["api_key"] == "lin_test"


def test_fetch_returns_raw_issue_list():
    payload = {
        "issues": {
            "nodes": [
                _issue("PLAN-021-1", "rooben fork archive", "follow-up:021-G", "2026-04-01T00:00:00Z"),
            ]
        }
    }
    issues = fus.fetch_follow_up_issues(
        now=datetime(2026, 5, 31, tzinfo=UTC),
        gql=lambda q, v, *, api_key=None: payload,
        api_key="lin_test",
    )
    assert len(issues) == 1
    assert issues[0]["identifier"] == "PLAN-021-1"


def test_fetch_propagates_linear_errors():
    def boom(*_a, **_kw):
        raise fus.LinearError("Linear HTTP 500: Internal Server Error")

    with pytest.raises(fus.LinearError):
        fus.fetch_follow_up_issues(
            now=datetime(2026, 5, 31, tzinfo=UTC),
            gql=boom,
            api_key="lin_test",
        )


# ---------------------------------------------------------------------------
# normalize_issues
# ---------------------------------------------------------------------------


def test_normalize_extracts_plan_id_and_sorts_oldest_first():
    now = datetime(2026, 5, 31, tzinfo=UTC)
    raw = [
        _issue("PV-10", "newer", "follow-up:022-A", "2026-04-01T00:00:00Z"),  # 60d
        _issue("PV-11", "ancient", "follow-up:014-1", "2025-12-01T00:00:00Z"),  # 181d
        _issue("PV-12", "stale", "follow-up:021-G", "2026-05-01T00:00:00Z"),  # 30d
    ]
    normalized = fus.normalize_issues(raw, now=now)
    assert [i.identifier for i in normalized] == ["PV-11", "PV-10", "PV-12"]
    assert normalized[0].plan_id == "014-1"
    assert normalized[0].age_days == 181
    assert normalized[1].age_days == 60
    assert normalized[2].age_days == 30


def test_normalize_handles_missing_followup_label():
    now = datetime(2026, 5, 31, tzinfo=UTC)
    raw = [
        {
            "identifier": "PV-99",
            "title": "ghost",
            "url": "https://x/",
            "createdAt": "2026-04-01T00:00:00Z",
            "labels": {"nodes": [{"name": "bug"}]},
        }
    ]
    normalized = fus.normalize_issues(raw, now=now)
    assert normalized[0].plan_id == "unknown"


def test_normalize_skips_invalid_createdAt():
    now = datetime(2026, 5, 31, tzinfo=UTC)
    raw = [
        _issue("PV-1", "ok", "follow-up:021-A", "2026-04-01T00:00:00Z"),
        _issue("PV-2", "bad", "follow-up:021-B", "not-a-date"),
    ]
    normalized = fus.normalize_issues(raw, now=now)
    assert [i.identifier for i in normalized] == ["PV-1"]


# ---------------------------------------------------------------------------
# build_digest
# ---------------------------------------------------------------------------


def test_build_digest_empty_emits_positive_confirmation():
    now = datetime(2026, 5, 31, tzinfo=UTC)
    digest = fus.build_digest([], now=now)
    assert "0 stale follow-ups" in digest["summary"]
    assert digest["thread"] == []
    assert digest["block_kit"][0]["text"]["text"] == digest["summary"]


def test_build_digest_summary_matches_thread_count():
    now = datetime(2026, 5, 31, tzinfo=UTC)
    issues = fus.normalize_issues(
        [
            _issue("PV-1", "a", "follow-up:021-A", "2026-04-01T00:00:00Z"),  # 60d
            _issue("PV-2", "b", "follow-up:021-B", "2026-02-01T00:00:00Z"),  # 119d (>90)
            _issue("PV-3", "c", "follow-up:022-A", "2026-05-01T00:00:00Z"),  # 30d
        ],
        now=now,
    )
    digest = fus.build_digest(issues, now=now)
    assert "3 issues > 30 days old" in digest["summary"]
    assert "1 > 90 days" in digest["summary"]
    assert len(digest["thread"]) == 3
    # markdown link + plan id + age + title present
    line = digest["thread"][0]
    assert "PV-2" in line and "follow-up:021-B" in line and "days" in line


def test_build_digest_chunks_large_lists_into_block_kit_sections():
    now = datetime(2026, 5, 31, tzinfo=UTC)
    raw = [
        _issue(f"PV-{i}", f"row {i}", "follow-up:021-A", "2026-01-01T00:00:00Z")
        for i in range(45)
    ]
    digest = fus.build_digest(fus.normalize_issues(raw, now=now), now=now)
    # summary + divider + ceil(45/20) sections = 5 blocks
    sections = [b for b in digest["block_kit"] if b.get("type") == "section"]
    assert len(sections) == 1 + 3  # top summary + 3 chunks (20+20+5)


# ---------------------------------------------------------------------------
# send_digest_to_slack
# ---------------------------------------------------------------------------


def test_send_digest_posts_summary_then_threads_rows():
    calls = []

    def fake_post(payload, *, token):
        calls.append(payload)
        # First (summary) returns a ts; subsequent (thread) reuse the same.
        return {"ok": True, "ts": "171.0" if "thread_ts" not in payload else "171.X"}

    digest = {
        "summary": "top",
        "thread": ["row-a", "row-b"],
        "block_kit": [{"type": "section", "text": {"type": "mrkdwn", "text": "top"}}],
    }
    res = fus.send_digest_to_slack(
        digest,
        channel="D123",
        token="xoxb-test",
        slack_post=fake_post,
    )
    assert res["summary_ts"] == "171.0"
    assert res["thread_ts_count"] == 2
    assert calls[0]["channel"] == "D123"
    assert "blocks" in calls[0]
    assert calls[1]["thread_ts"] == "171.0"
    assert calls[2]["thread_ts"] == "171.0"


def test_send_digest_requires_channel():
    with pytest.raises(fus.SlackError, match="SLACK_FOLLOW_UP_CHANNEL"):
        fus.send_digest_to_slack(
            {"summary": "x", "thread": [], "block_kit": []},
            channel=None,
            token="xoxb-test",
            slack_post=lambda *_a, **_kw: {"ok": True, "ts": "1"},
        )


def test_send_digest_requires_token():
    with pytest.raises(fus.SlackError, match="SLACK_BOT_TOKEN"):
        fus.send_digest_to_slack(
            {"summary": "x", "thread": [], "block_kit": []},
            channel="D123",
            token=None,
            slack_post=lambda *_a, **_kw: {"ok": True, "ts": "1"},
        )


def test_send_digest_propagates_slack_failure():
    def fail(*_a, **_kw):
        raise fus.SlackError("Slack API error: invalid_auth")

    with pytest.raises(fus.SlackError):
        fus.send_digest_to_slack(
            {"summary": "x", "thread": [], "block_kit": []},
            channel="D123",
            token="xoxb-test",
            slack_post=fail,
        )


# ---------------------------------------------------------------------------
# run_sweep — end-to-end
# ---------------------------------------------------------------------------


def test_run_sweep_happy_path_with_mocked_linear_and_slack():
    now = datetime(2026, 5, 31, tzinfo=UTC)
    payload = {
        "issues": {
            "nodes": [
                _issue("PV-1", "a", "follow-up:021-A", "2026-04-01T00:00:00Z"),
                _issue("PV-2", "b", "follow-up:022-B", "2025-12-15T00:00:00Z"),
            ]
        }
    }
    slack_calls = []

    def fake_gql(_q, _v, *, api_key=None):
        return payload

    def fake_slack(p, *, token):
        slack_calls.append(p)
        return {"ok": True, "ts": "171.0"}

    result = fus.run_sweep(
        now=now,
        linear_gql=fake_gql,
        slack_post=fake_slack,
        channel="D123",
        token="xoxb-test",
        api_key="lin_test",
    )
    assert result["count"] == 2
    assert result["very_stale"] == 1  # PV-2 is ~167d old
    assert result["summary_ts"] == "171.0"
    assert result["thread_ts_count"] == 2
    # 1 top + 2 thread replies
    assert len(slack_calls) == 3


def test_run_sweep_empty_still_posts_confirmation():
    now = datetime(2026, 5, 31, tzinfo=UTC)
    slack_calls = []

    def fake_slack(p, *, token):
        slack_calls.append(p)
        return {"ok": True, "ts": "171.0"}

    result = fus.run_sweep(
        now=now,
        linear_gql=lambda *_a, **_kw: {"issues": {"nodes": []}},
        slack_post=fake_slack,
        channel="D123",
        token="xoxb-test",
        api_key="lin_test",
    )
    assert result["count"] == 0
    assert result["very_stale"] == 0
    assert result["thread_ts_count"] == 0
    # exactly one Slack call — the positive-confirmation summary
    assert len(slack_calls) == 1
    assert "0 stale follow-ups" in slack_calls[0]["text"]


def test_run_sweep_propagates_linear_failure():
    def boom(*_a, **_kw):
        raise fus.LinearError("Linear HTTP 500")

    with pytest.raises(fus.LinearError):
        fus.run_sweep(
            now=datetime(2026, 5, 31, tzinfo=UTC),
            linear_gql=boom,
            slack_post=lambda *_a, **_kw: {"ok": True, "ts": "1"},
            channel="D123",
            token="xoxb-test",
            api_key="lin_test",
        )
