from datetime import datetime, timedelta, timezone

from agent.cmh_subprocess.envelope import (
    ENVELOPE_STATE_FILENAME,
    EnvelopeDecision,
    allocation_cap,
    check_budget,
    default_envelope_state,
    envelope_state_path,
    increment_usage,
    load_envelope_state,
    save_envelope_state,
)


def test_default_envelope_caps_are_85_percent():
    state = default_envelope_state()

    assert allocation_cap(state["anthropic_max"]) == 191
    assert allocation_cap(state["chatgpt_pro"]) == 170


def test_envelope_state_path_uses_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert envelope_state_path() == tmp_path / "state" / ENVELOPE_STATE_FILENAME


def test_load_missing_state_returns_defaults_without_writing(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    state = load_envelope_state()

    assert state["anthropic_max"]["envelope_messages_used_5h"] == 0
    assert not envelope_state_path().exists()


def test_increment_usage_starts_window_and_persists(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = datetime(2026, 5, 17, 20, 0, tzinfo=timezone.utc)
    state = default_envelope_state()

    updated = increment_usage(state, "anthropic_max", now=now)
    save_envelope_state(updated)
    reloaded = load_envelope_state()

    assert reloaded["anthropic_max"]["envelope_messages_used_5h"] == 1
    assert reloaded["anthropic_max"]["window_start_iso"] == now.isoformat()
    assert reloaded["anthropic_max"]["last_invocation_iso"] == now.isoformat()


def test_increment_usage_resets_after_five_hours():
    start = datetime(2026, 5, 17, 20, 0, tzinfo=timezone.utc)
    later = start + timedelta(hours=5, minutes=1)
    state = default_envelope_state()
    state["anthropic_max"]["window_start_iso"] = start.isoformat()
    state["anthropic_max"]["envelope_messages_used_5h"] = 190

    updated = increment_usage(state, "anthropic_max", now=later)

    assert updated["anthropic_max"]["envelope_messages_used_5h"] == 1
    assert updated["anthropic_max"]["window_start_iso"] == later.isoformat()


def test_budget_blocks_non_priority_at_cap():
    state = default_envelope_state()
    state["anthropic_max"]["envelope_messages_used_5h"] = 191

    decision = check_budget(state, "anthropic_max", priority=False)

    assert decision == EnvelopeDecision(
        allowed=False,
        reason="budget_blocked",
        used=191,
        cap=191,
        available=0,
    )


def test_budget_allows_priority_at_cap_with_reason():
    state = default_envelope_state()
    state["anthropic_max"]["envelope_messages_used_5h"] = 191

    decision = check_budget(state, "anthropic_max", priority=True)

    assert decision.allowed is True
    assert decision.reason == "priority_override"
    assert decision.available == 0
