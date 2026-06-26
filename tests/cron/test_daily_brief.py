"""
Tests for Phase 026-D — Hermes cron daily morning brief.

Covers (per master plan §026-D acceptance criteria, 5+ tests):
  * happy path: weekday → /daily invoked → DM posted with block-kit
  * weekend-skip: Saturday/Sunday → no Slack post, status=skipped
  * /daily invocation failure surfaces as status=failed (not silent)
  * Slack post failure raises SlackError (visible cron exit)
  * AgentDecision URN passes through to the result envelope
  * env-validation: missing channel/token raises clear SlackError
  * weekday gate edges (Mon, Fri included; Sat, Sun excluded)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from cron import daily_brief as db


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_payload(urn: str | None = "urn:atlas:agent_decision:fake") -> dict:
    payload: dict = {
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": "*Top 3 today*"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "• bullet one"}},
        ],
        "text": "Daily brief",
    }
    if urn is not None:
        payload["_daily_meta"] = {"agent_decision_urn": urn}
    return payload


def _make_invoker(payload: dict, captured: dict):
    async def _invoke(*, slack_channel: str):
        captured["slack_channel"] = slack_channel
        return payload

    return _invoke


def _make_slack_recorder(captured: dict, response: dict | None = None):
    response = response or {"ok": True, "ts": "1717000000.000100"}

    def _post(payload, *, token):
        captured["payload"] = payload
        captured["token"] = token
        return response

    return _post


# ---------------------------------------------------------------------------
# should_fire_now — weekday gate
# ---------------------------------------------------------------------------


def test_weekday_monday_fires():
    # Mon 2026-06-01 07:30 ET
    monday = datetime(2026, 6, 1, 7, 30, tzinfo=UTC)
    fire, reason = db.should_fire_now(monday)
    assert fire is True
    assert "weekday-fire" in reason


def test_weekday_friday_fires():
    # Fri 2026-06-05 07:30
    friday = datetime(2026, 6, 5, 7, 30, tzinfo=UTC)
    fire, _ = db.should_fire_now(friday)
    assert fire is True


def test_weekend_saturday_skips():
    sat = datetime(2026, 5, 30, 7, 30, tzinfo=UTC)
    fire, reason = db.should_fire_now(sat)
    assert fire is False
    assert "weekend-skip" in reason


def test_weekend_sunday_skips():
    sun = datetime(2026, 5, 31, 7, 30, tzinfo=UTC)
    fire, reason = db.should_fire_now(sun)
    assert fire is False
    assert "weekend-skip" in reason


# ---------------------------------------------------------------------------
# send_brief_to_slack — env validation
# ---------------------------------------------------------------------------


def test_send_brief_missing_channel_raises(monkeypatch):
    monkeypatch.delenv("SLACK_DAILY_DM_CHANNEL", raising=False)
    monkeypatch.delenv("SLACK_HOME_CHANNEL", raising=False)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    with pytest.raises(db.SlackError, match="SLACK_DAILY_DM_CHANNEL"):
        db.send_brief_to_slack(_fake_payload(), slack_post=lambda *a, **kw: {"ok": True})


def test_send_brief_missing_token_raises(monkeypatch):
    monkeypatch.setenv("SLACK_DAILY_DM_CHANNEL", "D0000ABCD")
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    with pytest.raises(db.SlackError, match="SLACK_BOT_TOKEN"):
        db.send_brief_to_slack(_fake_payload(), slack_post=lambda *a, **kw: {"ok": True})


def test_send_brief_posts_blocks_to_channel():
    captured: dict = {}
    out = db.send_brief_to_slack(
        _fake_payload(),
        channel="D0000ABCD",
        token="xoxb-test",
        slack_post=_make_slack_recorder(captured),
    )
    assert out["summary_ts"] == "1717000000.000100"
    assert out["channel"] == "D0000ABCD"
    sent = captured["payload"]
    assert sent["channel"] == "D0000ABCD"
    assert sent["text"] == "Daily brief"
    assert sent["blocks"][0]["type"] == "section"
    assert captured["token"] == "xoxb-test"


# ---------------------------------------------------------------------------
# run_daily_brief — orchestration paths
# ---------------------------------------------------------------------------


def test_run_daily_brief_weekday_happy_path():
    captured: dict = {}
    invoke_captured: dict = {}
    payload = _fake_payload(urn="urn:atlas:agent_decision:abc123")

    result = db.run_daily_brief(
        now=datetime(2026, 6, 1, 7, 30, tzinfo=UTC),  # Monday
        invoke_daily=_make_invoker(payload, invoke_captured),
        slack_post=_make_slack_recorder(captured),
        channel="D0000ABCD",
        token="xoxb-test",
    )

    assert result["status"] == "delivered"
    assert result["summary_ts"] == "1717000000.000100"
    assert result["channel"] == "D0000ABCD"
    assert result["agent_decision_urn"] == "urn:atlas:agent_decision:abc123"
    # The cron passes its resolved channel through to /daily so the
    # AgentDecision URN binds to the same DM the brief lands in.
    assert invoke_captured["slack_channel"] == "D0000ABCD"
    # The block-kit payload was forwarded verbatim.
    assert captured["payload"]["blocks"] == payload["blocks"]


# ---------------------------------------------------------------------------
# _invoke_daily_default — the REAL default invoker (regression for the
# DailyHandlerConfig kwarg mismatch). Every other test injects ``invoke_daily``
# and so never exercises this path, which is exactly how the bug shipped.
# ---------------------------------------------------------------------------


def test_invoke_daily_default_constructs_valid_config(monkeypatch):
    """The cron's default invoker must call ``build_daily_brief`` with a config
    the dataclass actually accepts.

    Regression: it previously built ``DailyHandlerConfig(writeback_enabled=True,
    slack_channel=...)`` — neither field exists on the dataclass — so the cron
    path raised ``TypeError`` the instant it fired (the ``/daily`` slash command
    uses a different path and was unaffected). Writeback (026-B) is not wired
    into ``build_daily_brief``, so the default invoker must not pass unsupported
    kwargs.
    """
    import plugins.slash.daily as daily_mod

    captured: dict = {}

    async def _fake_build(config=None):
        captured["config"] = config
        return _fake_payload()

    monkeypatch.setattr(daily_mod, "build_daily_brief", _fake_build)

    payload = asyncio.run(db._invoke_daily_default(slack_channel="D0000ABCD"))

    assert payload["blocks"], "default invoker should return the brief payload"
    # A config the dataclass accepts (or None) — never one built from
    # unsupported kwargs.
    assert captured["config"] is None or isinstance(
        captured["config"], daily_mod.DailyHandlerConfig
    )


def test_run_daily_brief_weekend_skips_without_slack_call():
    slack_calls: list = []

    def _slack(*args, **kwargs):
        slack_calls.append((args, kwargs))
        return {"ok": True}

    async def _invoke(*, slack_channel: str):
        slack_calls.append(("invoked",))
        return _fake_payload()

    result = db.run_daily_brief(
        now=datetime(2026, 5, 30, 7, 30, tzinfo=UTC),  # Saturday
        invoke_daily=_invoke,
        slack_post=_slack,
        channel="D0000ABCD",
        token="xoxb-test",
    )

    assert result["status"] == "skipped"
    assert "weekend-skip" in result["reason"]
    assert result["summary_ts"] is None
    assert slack_calls == []  # neither /daily nor Slack was called


def test_run_daily_brief_daily_failure_surfaces_as_failed_status():
    async def _bad_invoke(*, slack_channel: str):
        raise RuntimeError("atlas_ask timeout")

    slack_calls: list = []

    def _slack(*args, **kwargs):
        slack_calls.append(kwargs)
        return {"ok": True}

    result = db.run_daily_brief(
        now=datetime(2026, 6, 1, 7, 30, tzinfo=UTC),
        invoke_daily=_bad_invoke,
        slack_post=_slack,
        channel="D0000ABCD",
        token="xoxb-test",
    )

    assert result["status"] == "failed"
    assert "atlas_ask timeout" in result["reason"]
    # Slack was NOT called because the brief never built.
    assert slack_calls == []


def test_run_daily_brief_empty_payload_surfaces_as_failed():
    async def _empty(*, slack_channel: str):
        return {"blocks": []}

    result = db.run_daily_brief(
        now=datetime(2026, 6, 1, 7, 30, tzinfo=UTC),
        invoke_daily=_empty,
        slack_post=lambda *a, **kw: {"ok": True},
        channel="D0000ABCD",
        token="xoxb-test",
    )

    assert result["status"] == "failed"
    assert "no blocks" in result["reason"]


def test_run_daily_brief_slack_error_raises():
    def _broken_slack(payload, *, token):
        raise db.SlackError("Slack API error: channel_not_found")

    with pytest.raises(db.SlackError, match="channel_not_found"):
        db.run_daily_brief(
            now=datetime(2026, 6, 1, 7, 30, tzinfo=UTC),
            invoke_daily=_make_invoker(_fake_payload(), {}),
            slack_post=_broken_slack,
            channel="D0000ABCD",
            token="xoxb-test",
        )


def test_run_daily_brief_uses_env_channel_when_unset(monkeypatch):
    monkeypatch.setenv("SLACK_DAILY_DM_CHANNEL", "D_from_env")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env")
    captured: dict = {}
    invoke_captured: dict = {}

    db.run_daily_brief(
        now=datetime(2026, 6, 1, 7, 30, tzinfo=UTC),
        invoke_daily=_make_invoker(_fake_payload(), invoke_captured),
        slack_post=_make_slack_recorder(captured),
    )

    assert captured["payload"]["channel"] == "D_from_env"
    assert captured["token"] == "xoxb-env"
    assert invoke_captured["slack_channel"] == "D_from_env"


# ---------------------------------------------------------------------------
# main() script entry
# ---------------------------------------------------------------------------


def test_main_returns_zero_on_skipped(monkeypatch, capsys):
    # Force a Saturday "now" by monkeypatching the hermes_time clock.
    sat = datetime(2026, 5, 30, 7, 30, tzinfo=UTC)
    import hermes_time
    monkeypatch.setattr(hermes_time, "now", lambda: sat)
    monkeypatch.setenv("SLACK_DAILY_DM_CHANNEL", "D0000ABCD")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")

    rc = db.main([])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert '"status": "skipped"' in out
