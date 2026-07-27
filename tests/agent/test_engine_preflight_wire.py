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


def test_engine_without_hook_attribute_is_skipped():
    """Minimal engines lacking the optional hook must not break the turn."""
    agent = _make_agent(_stub_compressor(preflight=None))

    ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._compress_context.assert_not_called()
    assert ctx.preflight_compression_blocked is False


def test_true_engine_gets_exactly_one_compress_pass():
    """A True-returning engine triggers exactly one compress() per turn."""
    hook = MagicMock(return_value=True)
    agent = _make_agent(_stub_compressor(preflight=hook))
    # A real compaction: return a NEW list containing this turn's user row.
    agent._compress_context = MagicMock(
        side_effect=lambda messages, *_a, **_k: (
            [dict(m) for m in messages[-3:]],
            "SYSTEM",
        )
    )

    ctx = _build(agent)

    hook.assert_called_once()
    agent._compress_context.assert_called_once()
    # Real compaction re-anchors this turn's user message in the new list.
    assert ctx.messages[ctx.current_turn_user_idx]["content"] == "hello"
    # A sub-threshold maintenance pass proves nothing about over-threshold
    # compressibility: the retry-loop blocking flag must stay untouched.
    assert ctx.preflight_compression_blocked is False


def test_true_engine_single_pass_even_with_raised_attempt_cap():
    """The engine pass is once-per-turn regardless of compression.max_attempts."""
    hook = MagicMock(return_value=True)
    agent = _make_agent(_stub_compressor(preflight=hook))
    agent.max_compression_attempts = 6
    agent._compress_context = MagicMock(
        side_effect=lambda messages, *_a, **_k: (
            [dict(m) for m in messages],
            "SYSTEM",
        )
    )

    _build(agent)

    hook.assert_called_once()
    agent._compress_context.assert_called_once()


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


def test_engine_hook_not_consulted_when_threshold_path_fires():
    """Over-threshold turns take the cap-bounded loop, never the engine arm."""
    hook = MagicMock(return_value=True)
    comp = _stub_compressor(preflight=hook, threshold_tokens=100)
    comp.should_compress = lambda _tokens=None: True
    comp.context_length = 200_000
    agent = _make_agent(comp)

    with patch(
        "agent.turn_context.estimate_request_tokens_rough", return_value=999_999
    ):
        _build(agent)

    hook.assert_not_called()
    # The threshold loop ran instead (progress check breaks after pass 1
    # because the estimate never shrinks, but the pass itself happened).
    assert agent._compress_context.call_count >= 1


def test_engine_hook_not_consulted_during_failure_cooldown():
    """An active compression-failure cooldown gates engine maintenance too."""
    hook = MagicMock(return_value=True)
    comp = _stub_compressor(preflight=hook)
    comp.get_active_compression_failure_cooldown = lambda: {
        "remaining_seconds": 60.0
    }
    agent = _make_agent(comp)

    ctx = _build(agent)

    hook.assert_not_called()
    agent._compress_context.assert_not_called()
    assert ctx.preflight_compression_blocked is False


def test_engine_hook_exception_is_swallowed():
    """A buggy engine must not break an otherwise-healthy turn."""
    hook = MagicMock(side_effect=RuntimeError("buggy engine"))
    agent = _make_agent(_stub_compressor(preflight=hook))

    ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    hook.assert_called_once()
    agent._compress_context.assert_not_called()
    assert ctx.preflight_compression_blocked is False


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
