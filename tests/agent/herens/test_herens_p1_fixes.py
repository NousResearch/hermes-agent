"""Tests for P1-4, P1-5, P1-10, P1-11 fixes in turn_hooks + error_recovery.

P1-4: turn_hooks.py brain_networks config check (was always truthy).
P1-5: Pass real attempt_count to error_recovery (was hardcoded to 1).
P1-10: Implement actual retry execution in ErrorRecovery (was recommendation-only).
P1-11: Add LLM fallback for error classifier (was pattern-only).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def herens_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "agent.herens.config.is_herens_enabled", lambda cfg=None: True
    )
    monkeypatch.setattr(
        "agent.herens.config.load_herens_config",
        lambda cfg=None: {"enabled": True, "error_recovery": {"llm_fallback": False}},
    )
    return home


# ── P1-4: brain_networks config check ───────────────────────────────────────


def test_ecn_focus_skipped_when_brain_networks_disabled(herens_home, monkeypatch):
    """ECN focus must NOT run when brain_networks.enabled is False."""
    # Patch load_config to return brain_networks.enabled=False
    from agent.herens import turn_hooks

    fake_config = {
        "brain_networks": {"enabled": False},
        "experience": {"enabled": True, "lesson_limit": 3, "lesson_min_confidence": 0.6},
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    # Also patch _safe_ecn_focus to detect if it's called
    ecn_called = []
    monkeypatch.setattr(
        turn_hooks, "_safe_ecn_focus",
        lambda msg, **kwargs: ecn_called.append(msg) or "",
    )

    block = turn_hooks.build_turn_context_block(
        "test message", session_id="s_ecn_disabled", strategy="react"
    )
    assert ecn_called == [], "ECN focus ran even though brain_networks disabled"


def test_ecn_focus_runs_when_brain_networks_enabled(herens_home, monkeypatch):
    """ECN focus MUST run when brain_networks.enabled is True."""
    from agent.herens import turn_hooks

    fake_config = {
        "brain_networks": {"enabled": True},
        "experience": {"enabled": True, "lesson_limit": 3, "lesson_min_confidence": 0.6},
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    ecn_called = []
    monkeypatch.setattr(
        turn_hooks, "_safe_ecn_focus",
        lambda msg, **kwargs: ecn_called.append(msg) or "[ECN focus] test",
    )

    turn_hooks.build_turn_context_block(
        "test message", session_id="s_ecn_enabled", strategy="react"
    )
    assert len(ecn_called) == 1, "ECN focus was not called when brain_networks enabled"


# ── P1-5: real attempt_count from tracker ────────────────────────────────────


def test_attempt_tracker_starts_at_zero(herens_home):
    from agent.herens.error_recovery import get_attempt_count, clear_attempts

    clear_attempts("s_tracker_1")
    assert get_attempt_count("s_tracker_1") == 0


def test_record_attempt_increments_count(herens_home):
    from agent.herens.error_recovery import clear_attempts, get_attempt_count, record_attempt

    clear_attempts("s_tracker_2")
    assert record_attempt("s_tracker_2") == 1
    assert record_attempt("s_tracker_2") == 2
    assert record_attempt("s_tracker_2") == 3
    assert get_attempt_count("s_tracker_2") == 3


def test_clear_attempts_resets_count(herens_home):
    from agent.herens.error_recovery import clear_attempts, record_attempt, get_attempt_count

    record_attempt("s_tracker_3")
    record_attempt("s_tracker_3")
    assert get_attempt_count("s_tracker_3") == 2
    clear_attempts("s_tracker_3")
    assert get_attempt_count("s_tracker_3") == 0


def test_attempt_tracker_is_per_session(herens_home):
    from agent.herens.error_recovery import clear_attempts, get_attempt_count, record_attempt

    clear_attempts("s_a")
    clear_attempts("s_b")
    record_attempt("s_a")
    record_attempt("s_a")
    record_attempt("s_b")
    assert get_attempt_count("s_a") == 2
    assert get_attempt_count("s_b") == 1


def test_attempt_tracker_windowed(herens_home, monkeypatch):
    """Old attempts beyond the window must not count."""
    from agent.herens.error_recovery import _attempt_tracker, _ATTEMPT_WINDOW_S, get_attempt_count

    key = ("s_window", "_global")
    now = time.time()
    # Inject one old + one recent attempt
    _attempt_tracker[key] = [now - _ATTEMPT_WINDOW_S - 100, now - 1]
    assert get_attempt_count("s_window") == 1, "old attempt should be evicted"


# ── P1-10: execute_retry ─────────────────────────────────────────────────────


def test_execute_retry_succeeds_first_try(herens_home):
    from agent.herens.error_recovery import clear_attempts, execute_retry

    clear_attempts("s_retry_1")
    calls = []
    def fn():
        calls.append(1)
        return "ok"

    result = execute_retry(fn, session_id="s_retry_1", max_attempts=3)
    assert result["ok"] is True
    assert result["result"] == "ok"
    assert result["attempts"] == 1
    assert len(calls) == 1


def test_execute_retry_retries_on_transient(herens_home, monkeypatch):
    """A transient error should be retried with backoff."""
    from agent.herens.error_recovery import clear_attempts, execute_retry

    clear_attempts("s_retry_2")
    # Patch time.sleep so the test is fast
    monkeypatch.setattr("agent.herens.error_recovery.time.sleep", lambda s: None)

    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("Connection timed out")
        return "ok"

    result = execute_retry(fn, session_id="s_retry_2", max_attempts=5)
    assert result["ok"] is True
    assert result["result"] == "ok"
    assert result["attempts"] == 3
    assert len(calls) == 3


def test_execute_retry_gives_up_after_max(herens_home, monkeypatch):
    from agent.herens.error_recovery import clear_attempts, execute_retry

    clear_attempts("s_retry_3")
    monkeypatch.setattr("agent.herens.error_recovery.time.sleep", lambda s: None)

    calls = []
    def fn():
        calls.append(1)
        raise TimeoutError("Connection timed out")

    result = execute_retry(fn, session_id="s_retry_3", max_attempts=3)
    assert result["ok"] is False
    assert result["attempts"] == 3
    assert "Connection timed out" in result["error"]
    assert len(calls) == 3


def test_execute_retry_stops_on_abort(herens_home, monkeypatch):
    """An abort-classified error should stop retrying immediately."""
    from agent.herens.error_recovery import clear_attempts, execute_retry

    clear_attempts("s_retry_4")
    monkeypatch.setattr("agent.herens.error_recovery.time.sleep", lambda s: None)

    calls = []
    def fn():
        calls.append(1)
        raise RuntimeError("security violation: out of scope")

    result = execute_retry(fn, session_id="s_retry_4", max_attempts=5)
    assert result["ok"] is False
    assert result["final_strategy"] == "abort"
    assert len(calls) == 1, "should not retry abort-classified errors"


def test_execute_retry_clears_tracker_on_success(herens_home, monkeypatch):
    """A successful retry must clear the attempt tracker."""
    from agent.herens.error_recovery import clear_attempts, execute_retry, get_attempt_count

    clear_attempts("s_retry_5")
    monkeypatch.setattr("agent.herens.error_recovery.time.sleep", lambda s: None)

    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ConnectionError("Connection refused")
        return "ok"

    execute_retry(fn, session_id="s_retry_5", max_attempts=5)
    assert get_attempt_count("s_retry_5") == 0, "tracker not cleared after success"


def test_execute_retry_invokes_on_classify(herens_home, monkeypatch):
    """The on_classify callback should fire on each failure."""
    from agent.herens.error_recovery import clear_attempts, execute_retry

    clear_attempts("s_retry_6")
    monkeypatch.setattr("agent.herens.error_recovery.time.sleep", lambda s: None)

    classifications = []
    def fn():
        raise ConnectionError("Connection timed out")

    execute_retry(
        fn, session_id="s_retry_6", max_attempts=2,
        on_classify=lambda strat, exc: classifications.append(strat.strategy),
    )
    assert len(classifications) == 2
    assert all(c == "retry" for c in classifications)


# ── P1-11: LLM fallback for classifier ───────────────────────────────────────


def test_llm_classify_returns_none_on_exception(herens_home, monkeypatch):
    """When auxiliary_chat raises, _llm_classify must return None (not crash)."""
    from agent.herens.error_recovery import _llm_classify

    def boom(*a, **kw):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr("agent.auxiliary_client.auxiliary_chat", boom)

    result = _llm_classify("some weird error", attempt_count=1)
    assert result is None


def test_llm_classify_parses_valid_json(herens_home, monkeypatch):
    """A valid JSON response from the LLM should produce a RecoveryStrategy."""
    from agent.herens.error_recovery import _llm_classify

    fake_resp = '{"strategy": "retry", "confidence": 0.85, "guidance": "wait and retry", "error_class": "transient"}'
    monkeypatch.setattr(
        "agent.auxiliary_client.auxiliary_chat",
        lambda *a, **kw: fake_resp,
    )

    result = _llm_classify("some weird error", attempt_count=1)
    assert result is not None
    assert result.strategy == "retry"
    assert result.confidence == 0.85
    assert "wait and retry" in result.guidance
    assert "llm_fallback" in result.signals


def test_llm_classify_returns_none_on_invalid_strategy(herens_home, monkeypatch):
    """An invalid strategy value should make _llm_classify return None."""
    from agent.herens.error_recovery import _llm_classify

    fake_resp = '{"strategy": "explode", "confidence": 0.5}'
    monkeypatch.setattr(
        "agent.auxiliary_client.auxiliary_chat",
        lambda *a, **kw: fake_resp,
    )

    result = _llm_classify("some error", attempt_count=1)
    assert result is None


def test_llm_classify_handles_markdown_fences(herens_home, monkeypatch):
    """LLM responses wrapped in markdown code fences should still parse."""
    from agent.herens.error_recovery import _llm_classify

    fake_resp = '```json\n{"strategy": "pivot", "confidence": 0.7, "guidance": "try different approach"}\n```'
    monkeypatch.setattr(
        "agent.auxiliary_client.auxiliary_chat",
        lambda *a, **kw: fake_resp,
    )

    result = _llm_classify("some error", attempt_count=1)
    assert result is not None
    assert result.strategy == "pivot"


def test_classify_error_uses_llm_fallback_when_enabled(herens_home, monkeypatch):
    """When llm_fallback is enabled and no pattern matches, use the LLM."""
    from agent.herens import error_recovery

    # Enable llm_fallback
    monkeypatch.setattr(
        "agent.herens.config.load_herens_config",
        lambda cfg=None: {"enabled": True, "error_recovery": {"llm_fallback": True}},
    )
    fake_resp = '{"strategy": "delegate", "confidence": 0.8, "guidance": "delegate to specialist"}'
    monkeypatch.setattr(
        "agent.auxiliary_client.auxiliary_chat",
        lambda *a, **kw: fake_resp,
    )

    # An error that doesn't match any pattern
    strat = error_recovery.classify_error("some weird unique error xyz123", attempt_count=1)
    assert strat.strategy == "delegate"
    assert "llm_fallback" in strat.signals


def test_classify_error_skips_llm_when_disabled(herens_home, monkeypatch):
    """When llm_fallback is disabled, the LLM should NOT be called."""
    from agent.herens import error_recovery

    monkeypatch.setattr(
        "agent.herens.config.load_herens_config",
        lambda cfg=None: {"enabled": True, "error_recovery": {"llm_fallback": False}},
    )
    llm_called = []
    def fake_chat(*a, **kw):
        llm_called.append(1)
        return '{"strategy": "retry"}'
    monkeypatch.setattr("agent.auxiliary_client.auxiliary_chat", fake_chat)

    strat = error_recovery.classify_error("some weird unique error xyz123", attempt_count=1)
    assert llm_called == [], "LLM was called even though llm_fallback disabled"
    # Falls back to default pivot
    assert strat.strategy == "pivot"
    assert "default_pivot" in strat.signals
