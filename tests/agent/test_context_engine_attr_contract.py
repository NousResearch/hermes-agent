"""The host must not read engine state that the ContextEngine ABC never declares.

``plugins/context_engine/load_context_engine()`` advertises pluggable context
engines, and ``ContextEngine`` is the published contract those engines
implement. But two host sites read attributes that live only on the built-in
``ContextCompressor``:

* ``agent/turn_context.py`` idle-compaction floor — ``summary_target_ratio``
* ``agent/turn_context.py`` deferred-preflight log — ``last_real_prompt_tokens``

Neither was declared on the ABC and neither read was guarded, so an engine that
implements exactly the four abstract members raises ``AttributeError`` mid-turn
and the turn dies.

These tests pin both halves of the fix:

1. ``ContextEngine`` declares both fields with defaults that match the built-in
   compressor, so the contract is honest about what the host reads.
2. The two host reads are defensive, so an engine constructed before the fields
   were declared (or a compressor double) still completes the turn.
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agent.context_engine import ContextEngine
from agent.turn_context import TurnContext, build_turn_context
from tests.agent.test_turn_context import _FakeAgent


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


class _MinimalEngine(ContextEngine):
    """An engine implementing EXACTLY the ContextEngine abstract members."""

    @property
    def name(self) -> str:
        return "minimal"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        return None

    def should_compress(self, prompt_tokens: int = None) -> bool:
        return False

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        return messages


# ── 1. contract: the ABC declares what the host reads ──────────────────────


def test_abc_declares_summary_target_ratio_with_builtin_default():
    """The idle-compaction floor multiplier is part of the published contract."""
    engine = _MinimalEngine()
    assert isinstance(engine.summary_target_ratio, float)
    # Pin the RELATION to the built-in engine's own default rather than a bare
    # literal, so the two can never silently drift apart.
    from agent.context_compressor import ContextCompressor
    import inspect

    builtin_default = inspect.signature(
        ContextCompressor.__init__
    ).parameters["summary_target_ratio"].default
    assert engine.summary_target_ratio == builtin_default


def test_abc_declares_last_real_prompt_tokens():
    """The deferred-preflight log reads this; the ABC must declare it."""
    engine = _MinimalEngine()
    assert engine.last_real_prompt_tokens == 0


def test_builtin_compressor_still_overrides_both_fields():
    """The ABC defaults must not shadow ContextCompressor's real values."""
    from agent.context_compressor import ContextCompressor

    comp = ContextCompressor(model="test/model", summary_target_ratio=0.35)
    assert comp.summary_target_ratio == 0.35
    comp.last_real_prompt_tokens = 4321
    assert comp.last_real_prompt_tokens == 4321


# ── 2. host: both reads survive an engine that lacks the attributes ────────


def _history(n_pairs: int = 6) -> list:
    msgs = []
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


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


def _legacy_engine_stub(**extra):
    """An engine-shaped double with NEITHER of the two attributes present.

    This is what a plugin engine written against an older ContextEngine looks
    like from the host's point of view: it satisfies every documented hook but
    carries no ``summary_target_ratio`` / ``last_real_prompt_tokens``.
    """
    comp = types.SimpleNamespace(
        protect_first_n=2,
        protect_last_n=2,
        threshold_tokens=1_000,
        last_prompt_tokens=0,
        should_compress=lambda _tokens=None: False,
        should_compress_info=lambda _tokens=None: (False, None),
        get_active_compression_failure_cooldown=lambda: None,
    )
    for k, v in extra.items():
        setattr(comp, k, v)
    assert not hasattr(comp, "summary_target_ratio")
    assert not hasattr(comp, "last_real_prompt_tokens")
    return comp


def test_idle_compaction_survives_engine_without_summary_target_ratio():
    """Idle compaction must not AttributeError on a minimal engine."""
    comp = _legacy_engine_stub(
        should_defer_preflight_to_real_usage=lambda _t: False,
    )
    agent = _make_agent(comp)
    # Arm the opt-in idle path so the floor calculation actually runs.
    agent.compression_idle_compact_after_seconds = 1
    agent._last_activity_ts = 0.0  # epoch => a huge idle gap

    ctx = _build(agent)

    assert isinstance(ctx, TurnContext)


def test_idle_floor_uses_abc_default_when_engine_omits_the_ratio():
    """The fallback must be the ABC default, not 0 (which would floor at 0)."""
    comp = _legacy_engine_stub(
        should_defer_preflight_to_real_usage=lambda _t: False,
    )
    agent = _make_agent(comp)
    agent.compression_idle_compact_after_seconds = 1
    agent._last_activity_ts = 0.0

    seen = {}

    def _capture(*_a, **kw):
        seen["floor"] = kw.get("floor_tokens")
        return False

    with patch("agent.turn_context._should_idle_compact", side_effect=_capture):
        _build(agent)

    assert seen["floor"] == int(
        comp.threshold_tokens * ContextEngine.summary_target_ratio
    )
    assert seen["floor"] > 0


def test_deferred_preflight_log_survives_engine_without_last_real_prompt_tokens():
    """A deferring engine that never tracks the real count must not crash."""
    comp = _legacy_engine_stub(
        # Force the deferral branch that reads last_real_prompt_tokens.
        should_defer_preflight_to_real_usage=lambda _t: True,
    )
    agent = _make_agent(comp)
    # Push the rough estimate over the threshold so preflight actually runs.
    comp.threshold_tokens = 1

    ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    # The deferral means no compaction was attempted.
    agent._compress_context.assert_not_called()


def test_non_numeric_summary_target_ratio_falls_back_to_abc_default():
    """A MagicMock-style engine double must not poison the floor arithmetic."""
    comp = _legacy_engine_stub(
        should_defer_preflight_to_real_usage=lambda _t: False,
    )
    comp.summary_target_ratio = MagicMock()  # truthy, non-numeric
    agent = _make_agent(comp)
    agent.compression_idle_compact_after_seconds = 1
    agent._last_activity_ts = 0.0

    seen = {}

    def _capture(*_a, **kw):
        seen["floor"] = kw.get("floor_tokens")
        return False

    with patch("agent.turn_context._should_idle_compact", side_effect=_capture):
        _build(agent)

    assert seen["floor"] == int(
        comp.threshold_tokens * ContextEngine.summary_target_ratio
    )
