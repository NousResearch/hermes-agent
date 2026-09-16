"""Engine-driven sub-threshold preflight maintenance (#20316, salvaged from #20424).

``build_turn_context`` historically consulted the context engine only through
``should_compress()`` — the optional ``ContextEngine.should_compress_preflight()``
hook was never called anywhere in the turn flow, so engines doing LCM-style
deferred maintenance (incremental leaf-chunk compaction below the token
threshold) could never run it.  The preflight block now falls through to the
engine hook when the threshold path does not fire.

Contracts pinned here:

* Byte-identical default: the built-in ``ContextCompressor`` inherits the
  default ``should_compress_preflight() -> False``, so the sub-threshold path
  performs no compression, emits no status, and leaves every piece of turn
  bookkeeping untouched.
* A True-returning engine gets its ``compress()`` pass exactly ONCE per turn
  (mutually exclusive with the cap-bounded threshold loop), regardless of the
  resolved ``compression.max_attempts`` value.
* No-op interplay (#64382): an engine pass that no-ops (``_compress_context``
  returns the input list object) must not set ``preflight_compression_blocked``
  — the sub-threshold pass proves nothing about over-threshold
  compressibility — and must not re-baseline the flush history.
* The hook is never consulted when the threshold path fires, when a
  compression-failure cooldown is active, or when the engine raises.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import TurnContext, build_turn_context
from tests.agent.test_turn_context import _FakeAgent


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    """Keep the aux runtime-main global from leaking into sibling tests."""
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _history(n_pairs: int = 6) -> list:
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def _stub_compressor(*, preflight=None, threshold_tokens=10**9):
    """Engine-shaped stub: below threshold, no defer, no cooldown."""
    comp = types.SimpleNamespace(
        protect_first_n=2,
        protect_last_n=2,
        threshold_tokens=threshold_tokens,
        last_prompt_tokens=0,
        should_compress=lambda _tokens=None: False,
        should_defer_preflight_to_real_usage=lambda _t: False,
        get_active_compression_failure_cooldown=lambda: None,
    )
    if preflight is not None:
        comp.should_compress_preflight = preflight
    return comp


def _make_agent(compressor):
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent.context_compressor = compressor
    agent._emit_status = MagicMock()
    agent._compress_context = MagicMock(
        side_effect=lambda messages, *_a, **_k: (messages, "SYSTEM")
    )
    return agent


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=_history(),
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


def test_default_false_hook_is_byte_identical_noop():
    """The default engine (hook returns False) changes nothing sub-threshold."""
    calls = []

    def _hook(messages):
        calls.append(list(messages))
        return False

    agent = _make_agent(_stub_compressor(preflight=_hook))

    ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    assert calls, "sub-threshold path must consult the engine hook"
    agent._compress_context.assert_not_called()
    agent._emit_status.assert_not_called()
    assert ctx.preflight_compression_blocked is False








def test_pending_sanitation_runs_before_short_transcript_cheap_gate():
    """A short transcript can still carry an engine sanitation operation."""
    hook = MagicMock(return_value=True)
    compressor = _stub_compressor(preflight=hook)
    compressor.protect_first_n = 3
    compressor.protect_last_n = 6
    agent = _make_agent(compressor)
    sanitized = [{"role": "user", "content": "[redacted]"}]
    agent._compress_context = MagicMock(return_value=(sanitized, "SYSTEM"))

    ctx = _build(agent, conversation_history=_history(2))

    hook.assert_called_once()
    agent._compress_context.assert_called_once()
    assert ctx.messages is sanitized


def test_engine_maintenance_gets_real_token_estimate_not_zero():
    """Round-7 finding: the short-history maintenance path must not forward approx_tokens=0.

    When the cheap preflight gate rejects but the engine hook opts in, the pass used to
    call ``_compress_context(approx_tokens=0)``; engines sizing compaction from the
    documented ``current_tokens`` then treat a nonempty prompt as empty. The real
    preflight estimate must be computed once the hook opts in.
    """
    hook = MagicMock(return_value=True)
    compressor = _stub_compressor(preflight=hook)
    compressor.protect_first_n = 3
    compressor.protect_last_n = 6
    agent = _make_agent(compressor)

    captured: dict = {}

    def _capture(messages, system_message=None, **kwargs):
        captured["approx_tokens"] = kwargs.get("approx_tokens")
        return (messages, "SYSTEM")

    agent._compress_context = MagicMock(side_effect=_capture)

    _build(agent, conversation_history=_history(2))

    hook.assert_called_once()
    agent._compress_context.assert_called_once()
    approx = captured.get("approx_tokens")
    assert isinstance(approx, int) and not isinstance(approx, bool), (
        "engine maintenance must receive a real integer token estimate"
    )
    assert approx > 0, (
        f"engine maintenance received approx_tokens={approx!r}: a nonempty prompt "
        "must never be sized as empty"
    )


@pytest.mark.parametrize("blocked_by", ["cooldown", "deferred", "codex_native"])
def test_short_transcript_sanitation_respects_preflight_skip_gates(blocked_by):
    hook = MagicMock(return_value=True)
    compressor = _stub_compressor(preflight=hook)
    compressor.protect_first_n = 3
    compressor.protect_last_n = 6
    agent = _make_agent(compressor)
    if blocked_by == "cooldown":
        compressor.get_active_compression_failure_cooldown = lambda: {
            "remaining_seconds": 30.0,
        }
    elif blocked_by == "deferred":
        compressor.should_defer_preflight_to_real_usage = lambda _tokens: True
    else:
        agent.api_mode = "codex_app_server"
        agent.codex_app_server_auto_compaction = "native"

    _build(agent, conversation_history=_history(2))

    hook.assert_not_called()
    agent._compress_context.assert_not_called()


def test_true_engine_noop_does_not_defeat_retry_loop_blocking():
    """#64382 interplay: an engine pass that no-ops must not touch the

    stale-budget blocking machinery — ``preflight_compression_blocked`` stays
    False (it was never proven ineffective at threshold pressure) and the
    flush baseline is not re-baselined for a compaction that never happened.
    """
    hook = MagicMock(return_value=True)
    agent = _make_agent(_stub_compressor(preflight=hook))
    # Skip path: _compress_context returns the INPUT list object.
    agent._compress_context = MagicMock(
        side_effect=lambda messages, *_a, **_k: (messages, "SYSTEM")
    )
    history = _history()

    ctx = _build(agent, conversation_history=history)

    hook.assert_called_once()
    agent._compress_context.assert_called_once()
    assert ctx.preflight_compression_blocked is False
    # No real compaction -> the caller-provided history object remains the
    # flush baseline (no re-baseline through
    # conversation_history_after_compression).
    assert ctx.conversation_history is history








def test_builtin_compressor_default_sub_threshold_path_unchanged(tmp_path):
    """Byte-identical-default pin against the REAL ContextCompressor.

    The built-in engine inherits ``should_compress_preflight() -> False``
    from ``ContextEngine``, so wiring the hook must not change the default
    sub-threshold behavior at all: no compress call, no status emission,
    no blocking flag.
    """
    from agent.context_compressor import ContextCompressor
    from agent.context_engine import ContextEngine

    # The default really is False (the dead-code premise of #20316).
    assert "should_compress_preflight" not in ContextCompressor.__dict__
    assert ContextEngine.should_compress_preflight(
        object.__new__(ContextCompressor), []
    ) is False

    with patch(
        "agent.context_compressor.get_model_context_length", return_value=1_000_000
    ):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    agent = _make_agent(compressor)

    ctx = _build(agent)

    agent._compress_context.assert_not_called()
    agent._emit_status.assert_not_called()
    assert ctx.preflight_compression_blocked is False
