"""Regression contract for PR #99053 finding G2: a turn's REAL usage must survive an
ephemeral model override.

The scope swaps in an isolated copy of the context compressor so the override's
model-derived settings (context length, thresholds, calibration) cannot leak onto the
session compressor. Real provider usage, however, is turn progress: it is recorded on
the copy by `update_from_response` and was discarded when the session compressor was
restored, so finalization read stale counters.
"""

from __future__ import annotations

from agent.runtime_override import apply_runtime_override
from tests.agent.test_runtime_override import (
    _OVERRIDE_MODEL,
    _SESSION_MODEL,
    _build_model_owned_agent,
    _patch_model_owned_resolution,
)


def test_real_usage_recorded_during_an_override_turn_reaches_the_session_compressor(
    monkeypatch,
):
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    session_cc = agent.context_compressor
    session_cc.last_prompt_tokens = 0
    session_cc.last_completion_tokens = 0
    session_cc.last_total_tokens = 0

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        scoped = agent.context_compressor
        assert scoped is not session_cc, "the scope must project onto an isolated copy"
        # What the API loop does with the provider's real usage for this attempt.
        scoped.update_from_response(
            {"prompt_tokens": 4242, "completion_tokens": 7, "total_tokens": 4249}
        )

    assert session_cc.last_prompt_tokens == 4242, (
        "the turn's real prompt tokens were dropped with the scoped copy"
    )
    assert session_cc.last_total_tokens == 4249, (
        "the turn's real total tokens were dropped with the scoped copy"
    )


def test_same_model_override_keeps_the_cached_system_prompt(monkeypatch):
    """G9: a no-op override (same model) must not invalidate the frozen system prompt.

    `AGENTS.md` requires the system prompt to stay byte-stable for the life of a
    conversation; the PR body promises nothing is invalidated when the override
    changes none of the model-owned state the turn reads.
    """
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    agent._cached_system_prompt = "FROZEN_SYSTEM_PROMPT_BYTES"

    with apply_runtime_override(agent, {"model": _SESSION_MODEL}):  # identical model
        inside = agent._cached_system_prompt

    assert inside == "FROZEN_SYSTEM_PROMPT_BYTES", (
        f"a no-op override invalidated the cached prompt: {inside!r}"
    )


def test_scope_exit_without_a_response_does_not_touch_session_counters(monkeypatch):
    """Three-way forwarding: a scope that produced no usage/compaction progress must
    leave every session counter exactly as it was (no zeroing, no clobbering)."""
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    session_cc = agent.context_compressor
    session_cc.last_prompt_tokens = 777
    session_cc.last_total_tokens = 888
    session_cc._ineffective_compression_count = 3
    session_cc._last_compression_savings_pct = 12.5

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        pass  # no response, no compaction

    assert session_cc.last_prompt_tokens == 777
    assert session_cc.last_total_tokens == 888
    assert session_cc._ineffective_compression_count == 3
    assert session_cc._last_compression_savings_pct == 12.5  # untouched field stays put


def test_anti_thrash_progress_from_the_turn_is_forwarded(monkeypatch):
    """Progress the turn really made -- including anti-thrash bookkeeping -- is kept."""
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    session_cc = agent.context_compressor
    session_cc._ineffective_compression_count = 0

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        agent.context_compressor._ineffective_compression_count = 2

    assert session_cc._ineffective_compression_count == 2


def test_committed_compression_count_survives_the_scope(monkeypatch):
    """A compression that committed inside the scope must be counted afterwards.

    Counterexample contributed by the J1/OpenCode/agy dispatch experiment
    (`coop-test-99053-g2-20261012/g2-count-probe.py`): the scoped copy incremented
    `compression_count` to 1 and the session compressor still read 0.
    """
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    session_cc = agent.context_compressor
    session_cc.compression_count = 0
    session_cc._previous_summary = None

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        scoped = agent.context_compressor
        scoped.compression_count += 1  # ContextCompressor.compress() does this on success
        scoped._previous_summary = "example valid summary"

    assert session_cc.compression_count == 1, "a committed compression was lost with the copy"
    assert session_cc._previous_summary == "example valid summary"


def test_compaction_outcome_state_from_the_turn_is_forwarded(monkeypatch):
    """A compaction that ran inside the scope leaves its outcome on the session
    compressor (audit: tools/audit_compressor_fields.py, progress candidates)."""
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    session_cc = agent.context_compressor
    session_cc._last_compression_savings_pct = None
    session_cc._last_summary_fallback_used = False

    with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
        scoped = agent.context_compressor
        scoped._last_compression_savings_pct = 42.5
        scoped._last_summary_fallback_used = True  # compaction outcome, turn progress

    assert session_cc._last_compression_savings_pct == 42.5
    assert session_cc._last_summary_fallback_used is True


def test_no_op_check_does_not_invoke_a_descriptor(monkeypatch):
    """Deciding "no-op" must read live values without running a property getter
    (third-party review: a getter with side effects would execute on every turn)."""
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent(model=_SESSION_MODEL)
    calls = {"n": 0}

    class _Probe:
        def __get__(self, obj, objtype=None):
            calls["n"] += 1
            return "probe"

    type(agent).probe_attr = _Probe()
    try:
        with apply_runtime_override(agent, {"model": _SESSION_MODEL}):
            pass
        assert calls["n"] == 0, f"the no-op check invoked a descriptor {calls['n']} times"
    finally:
        del type(agent).probe_attr
