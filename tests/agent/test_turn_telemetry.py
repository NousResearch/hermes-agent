"""Tests for the per-turn outcome telemetry primitive (agent/turn_telemetry.py).

The record is the retrospective self-observation channel: captured in the
run_conversation forwarder's finally (every exit path), it tells the next turn's
pre_llm_call hook what the previous turn DID. Contract: always the full schema,
never raises, describes the finished turn — including failed/interrupted turns.
"""

from __future__ import annotations

import types

from agent.turn_telemetry import capture_turn_telemetry, empty_telemetry


_SCHEMA_KEYS = {
    "has_data", "turn_id", "provider", "model", "base_url", "was_fallback",
    "primary_provider", "primary_model", "on_primary", "api_calls",
    "interrupted", "tool_failures",
}


def _guardrails(exact=None, same=None, halted=False):
    return types.SimpleNamespace(
        _exact_failure_counts=exact or {},
        _same_tool_failure_counts=same or {},
        halt_decision=(object() if halted else None),
    )


def _agent(**over):
    base = dict(
        provider="onyx", model="qwen", base_url="http://127.0.0.1:8020",
        _current_turn_id="turn-42",
        _fallback_activated=False,
        _primary_runtime={"provider": "onyx", "model": "qwen", "base_url": "http://127.0.0.1:8020"},
        _tool_guardrails=_guardrails(),
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_schema_always_complete():
    rec = capture_turn_telemetry(_agent(), result={})
    assert set(rec) == _SCHEMA_KEYS
    assert set(empty_telemetry()) == _SCHEMA_KEYS


def test_has_data_true_on_real_capture_false_on_empty():
    assert capture_turn_telemetry(_agent(), result={})["has_data"] is True
    assert empty_telemetry()["has_data"] is False


def test_on_primary_when_active_matches_primary():
    rec = capture_turn_telemetry(_agent(), result={})
    assert rec["was_fallback"] is False and rec["on_primary"] is True


def test_fallback_flag_detected_reports_active_and_primary():
    a = _agent(_fallback_activated=True, provider="anthropic", model="claude",
               base_url="https://api.anthropic.com")
    rec = capture_turn_telemetry(a, result={})
    assert rec["was_fallback"] is True and rec["on_primary"] is False
    assert rec["provider"] == "anthropic"       # active runtime
    assert rec["primary_provider"] == "onyx"    # primary preserved (backup-vs-home)


def test_runtime_divergence_detected_without_flag():
    a = _agent(_fallback_activated=False, base_url="http://100.111.117.39:8011")
    rec = capture_turn_telemetry(a, result={})
    assert rec["was_fallback"] is True and rec["on_primary"] is False


def test_tool_failures_are_unresolved_at_turn_end():
    a = _agent(_tool_guardrails=_guardrails(
        exact={("read_file",): 1, ("terminal",): 1}, same={"read_file": 4}, halted=True))
    rec = capture_turn_telemetry(a, result={})
    assert rec["tool_failures"] == {"exact": 2, "same_tool": 4, "halted": True}


def test_metadata_from_result_dict():
    rec = capture_turn_telemetry(_agent(), result={"api_calls": 7, "interrupted": True})
    assert rec["api_calls"] == 7 and rec["interrupted"] is True
    assert rec["turn_id"] == "turn-42"  # from agent, not result


def test_none_result_is_safe():
    # forwarder passes result=None when run_conversation raised
    rec = capture_turn_telemetry(_agent(), result=None)
    assert rec["has_data"] is True and rec["api_calls"] == 0 and rec["interrupted"] is False


def test_bare_agent_captures_empty_values_without_raising():
    # every field is read defensively, so a bare agent yields a complete record
    # with empty values (not the exception sentinel).
    rec = capture_turn_telemetry(types.SimpleNamespace(), result={})
    assert set(rec) == _SCHEMA_KEYS
    assert rec["has_data"] is True
    assert rec["was_fallback"] is False and rec["provider"] == ""


def test_capture_exception_returns_unactionable_sentinel():
    # a genuinely hostile shape (_primary_runtime is not a dict) makes .get raise;
    # capture must fall back to the has_data=False sentinel, never propagate.
    a = _agent(_primary_runtime="not-a-dict")
    rec = capture_turn_telemetry(a, result={})
    assert rec["has_data"] is False
    assert set(rec) == _SCHEMA_KEYS


def test_never_raises_on_hostile_guardrails():
    class Hostile:
        @property
        def _exact_failure_counts(self):
            raise RuntimeError("boom")
    rec = capture_turn_telemetry(_agent(_tool_guardrails=Hostile()), result={})
    assert rec["tool_failures"] == {"exact": 0, "same_tool": 0, "halted": False}
    assert rec["provider"] == "onyx"       # the rest of the record survives


def test_no_tools_offered_field():
    # dropped from v1 (was only for a deferred consumer); schema stays minimal
    assert "tools_offered" not in capture_turn_telemetry(_agent(), result={})


def test_hook_contract_consumer_can_read_last_turn():
    a = _agent(_fallback_activated=True, provider="anthropic", model="claude")
    last_turn = capture_turn_telemetry(a, result={})

    def consumer(**kwargs):
        lt = kwargs.get("last_turn") or {}
        if lt.get("has_data") and lt.get("was_fallback"):
            return {"context": f"last turn ran off-primary on {lt.get('provider')}"}
        return None

    assert consumer(last_turn=last_turn) == {"context": "last turn ran off-primary on anthropic"}
    # first-turn sentinel must NOT trigger the consumer (has_data False)
    assert consumer(last_turn=empty_telemetry()) is None


def test_forwarder_stamps_on_every_exit_path(monkeypatch):
    """C1 regression: the record is stamped even when run_conversation raises
    (an early-return/exception turn), not only on the happy finalize path."""
    import agent.conversation_loop as cl

    captured = {}

    class _Agent:
        provider = "onyx"; model = "qwen"; base_url = "http://127.0.0.1:8020"
        _current_turn_id = "t1"
        _fallback_activated = True  # this failing turn fell back
        _primary_runtime = {"provider": "onyx", "model": "qwen", "base_url": "http://127.0.0.1:8020"}
        _tool_guardrails = _guardrails(same={"terminal": 3})
        # bind the real forwarder as a method
        from run_agent import AIAgent as _AI
        run_conversation = _AI.run_conversation

    def _boom(*a, **k):
        raise RuntimeError("turn blew up before finalize_turn")

    monkeypatch.setattr(cl, "run_conversation", _boom)
    ag = _Agent()
    try:
        ag.run_conversation("hi")
    except RuntimeError:
        pass
    # the failed turn was still observed
    assert ag._last_turn_telemetry is not None
    assert ag._last_turn_telemetry["has_data"] is True
    assert ag._last_turn_telemetry["was_fallback"] is True
    assert ag._last_turn_telemetry["tool_failures"]["same_tool"] == 3
