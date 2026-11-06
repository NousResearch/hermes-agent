from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent

    return AIAgent(
        api_key="k", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
        api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="test-session",
        platform="cli", quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, enabled_toolsets=["file"],
    )


def _usage(read, write, prompt):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=7, total_tokens=prompt + 7,
        prompt_tokens_details=SimpleNamespace(cached_tokens=read, cache_write_tokens=write),
        completion_tokens_details=None,
    )


def test_moa_daily_spend_is_shared_across_profiles_and_alarms_once(tmp_path):
    from agent.moa_budget import record_moa_spend

    root = tmp_path / ".hermes"
    default_home = root / "profiles" / "default"
    predyktor_home = root / "profiles" / "predyktor"
    now = datetime(2026, 10, 28, 12, 0, tzinfo=ZoneInfo("Europe/Warsaw"))

    first = record_moa_spend(1.20, 2.0, "response-1", home=default_home, now=now)
    assert first.day == "2026-10-28"
    assert first.total_usd == 1.20
    assert first.event_added and not first.alarm_emitted

    crossed = record_moa_spend(0.90, 2.0, "response-2", home=predyktor_home, now=now)
    assert crossed.total_usd == 2.10
    assert crossed.event_added and crossed.alarm_emitted

    duplicate = record_moa_spend(0.90, 2.0, "response-2", home=predyktor_home, now=now)
    assert duplicate.total_usd == 2.10
    assert not duplicate.event_added and not duplicate.alarm_emitted

    later = record_moa_spend(0.05, 2.0, "response-3", home=default_home, now=now)
    assert later.total_usd == 2.15
    assert later.event_added and not later.alarm_emitted


def test_moa_daily_alarm_resets_on_warsaw_calendar_day(tmp_path):
    from agent.moa_budget import record_moa_spend

    home = tmp_path / ".hermes"
    before_midnight_utc = datetime(2026, 10, 28, 22, 30, tzinfo=ZoneInfo("UTC"))
    after_midnight_warsaw = datetime(2026, 10, 28, 23, 30, tzinfo=ZoneInfo("UTC"))

    first = record_moa_spend(2.01, 2.0, "response-1", home=home, now=before_midnight_utc)
    second = record_moa_spend(2.01, 2.0, "response-2", home=home, now=after_midnight_warsaw)

    assert first.day == "2026-10-28" and first.alarm_emitted
    assert second.day == "2026-10-29" and second.total_usd == 2.01 and second.alarm_emitted


def test_moa_budget_alarm_is_advisory_only(tmp_path):
    from agent.moa_budget import record_moa_spend

    result = record_moa_spend(2.01, 2.0, "response-1", home=tmp_path, now=datetime(2026, 10, 28, tzinfo=ZoneInfo("Europe/Warsaw")))
    assert result.alarm_emitted
    assert result.total_usd > result.budget_usd
    # Crossing the threshold is a reported state, not an exception or a call veto.


def test_unknown_moa_spend_is_persisted_and_deduplicated(tmp_path):
    import sqlite3

    from agent.moa_budget import _ledger_path, record_unknown_moa_spend

    home = tmp_path / ".hermes"
    now = datetime(2026, 10, 28, 12, 0, tzinfo=ZoneInfo("Europe/Warsaw"))
    first = record_unknown_moa_spend(
        "response-without-usage", model="anthropic/claude-sonnet-5-5",
        reason="response_usage_unavailable", home=home, now=now,
    )
    duplicate = record_unknown_moa_spend(
        "response-without-usage", model="anthropic/claude-sonnet-5-5",
        reason="response_usage_unavailable", home=home, now=now,
    )

    assert first.day == "2026-10-28" and first.event_added
    assert duplicate.day == first.day and not duplicate.event_added
    with sqlite3.connect(_ledger_path(home)) as conn:
        row = conn.execute(
            "SELECT model, reason FROM moa_spend_unknown_events"
        ).fetchone()
    assert row == ("anthropic/claude-sonnet-5-5", "response_usage_unavailable")


def test_turn_usage_emits_moa_alarm_without_blocking(tmp_path, monkeypatch, caplog):
    from decimal import Decimal
    import logging

    from agent import turn_usage
    from agent.usage_pricing import CanonicalUsage, CostResult
    from hermes_cli import config as config_module

    agent = _agent(tmp_path, monkeypatch)
    monkeypatch.setattr(agent, "provider", "moa", raising=False)
    monkeypatch.setattr(agent, "model", "default", raising=False)
    monkeypatch.setattr(agent, "client", SimpleNamespace(
        consume_reference_usage=lambda: (CanonicalUsage(), 0.20),
        last_aggregator_slot={"provider": "anthropic", "model": "claude-sonnet-5-5"},
        consume_and_save_trace=lambda *_args, **_kwargs: None,
    ), raising=False)
    monkeypatch.setattr(config_module, "load_config", lambda: {"moa": {"daily_budget_usd": 0.20}})
    monkeypatch.setattr(
        turn_usage,
        "estimate_usage_cost",
        lambda *_args, **_kwargs: CostResult(
            amount_usd=Decimal("0.10"), status="estimated", source="official_docs_snapshot", label="~$0.10"
        ),
    )
    response = SimpleNamespace(usage=_usage(0, 0, 100), id="anthropic-response-1", model="claude-sonnet-5-5")
    try:
        with caplog.at_level(logging.ERROR, logger="agent.conversation_loop"):
            outcome = turn_usage.record_response_usage(
                agent, response, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
            )
    finally:
        agent.close()

    assert outcome.compression_attempts == 0
    assert any("ALARM: MoA daily estimated spend exceeded" in record.getMessage() for record in caplog.records)


def test_turn_usage_persists_unknown_moa_spend_without_blocking(tmp_path, monkeypatch, caplog):
    import logging
    import sqlite3

    from agent import turn_usage
    from agent.moa_budget import _ledger_path

    agent = _agent(tmp_path, monkeypatch)
    monkeypatch.setattr(agent, "provider", "moa", raising=False)
    monkeypatch.setattr(agent, "model", "default", raising=False)
    response = SimpleNamespace(usage=None, id="response-without-usage", model="claude-sonnet-5-5")
    try:
        with caplog.at_level(logging.ERROR, logger="agent.conversation_loop"):
            outcome = turn_usage.record_response_usage(
                agent, response, messages=[{"role": "user", "content": "hi"}], api_call_count=1,
                api_duration=0.2, compression_attempts=0, max_compression_attempts=3,
            )
    finally:
        agent.close()

    assert outcome.compression_attempts == 0
    assert any("ALARM: MoA daily spend estimate unavailable" in record.getMessage() for record in caplog.records)
    with sqlite3.connect(_ledger_path(tmp_path)) as conn:
        rows = conn.execute("SELECT model, reason FROM moa_spend_unknown_events").fetchall()
    assert rows == [("claude-sonnet-5-5", "response_usage_unavailable")]
