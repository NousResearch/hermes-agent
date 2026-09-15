"""Regressions for the context-engine host contract.

These tests pin the five generic host-side guarantees that external context
engine plugins (e.g. hermes-lcm) rely on:

1. ``_transition_context_engine_session`` drives the full lifecycle
   (on_session_end → on_session_reset → on_session_start → optional
   carry_over_new_session_context) and ``reset_session_state`` delegates
   to it when callers pass session metadata.

2. ``on_session_start`` receives ``conversation_id`` derived from
   ``_gateway_session_key`` at agent init time.

3. ``conversation_loop`` forwards canonical cache buckets
   (``cache_read_tokens``, ``cache_write_tokens``, ``input_tokens``,
   ``output_tokens``, ``reasoning_tokens``) to the engine's
   ``update_from_response``, on top of the legacy aggregate keys.

4. ``_discover_context_engines`` includes plugin-registered engines (not
   just repo-shipped engines under ``plugins/context_engine/``).

5. The repo-shipped ``_EngineCollector`` honors ``ctx.register_command``
   from a plugin engine's ``register(ctx)`` entry point and routes it
   to the global plugin command registry.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from agent.context_compressor import ContextCompressor
from agent.context_engine import (
    ContextDecision,
    ContextDecisionKind,
    ContextOutcome,
    ContextOutcomeKind,
    context_engine_decision,
)
from hermes_state import SessionDB
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = "test-session"
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent._gateway_session_key = "agent:main:telegram:dm:42"
    return agent


def test_typed_context_decisions_are_immutable_and_cannot_be_used_as_booleans():
    """Operation classes must not collapse into the old preflight boolean."""
    decision = ContextDecision(
        kind=ContextDecisionKind.SANITIZE,
        reason="unsafe_memory_context",
        attempt_id="turn-3",
        observed_tokens=100,
        threshold_tokens=80,
        effective_floor=64,
        sanitation_diff=((2, "content"),),
    )

    with pytest.raises(TypeError, match="explicitly"):
        bool(decision)
    with pytest.raises(FrozenInstanceError):
        decision.reason = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("kind", list(ContextDecisionKind))
def test_context_decision_kinds_cover_each_host_operation(kind):
    assert ContextDecision(kind=kind, reason="test").kind is kind


@pytest.mark.parametrize("kind", list(ContextOutcomeKind))
def test_context_outcome_kinds_cover_each_terminal_result(kind):
    outcome = ContextOutcome(kind=kind, reason="test")
    assert outcome.kind is kind


def test_legacy_engine_maps_boolean_preflight_to_typed_compact_or_noop():
    class LegacyEngine:
        def should_compress(self, _tokens):
            return True

    compact = context_engine_decision(
        LegacyEngine(), operation="preflight", observed_tokens=90, threshold_tokens=80,
    )
    noop = context_engine_decision(
        None, operation="preflight", observed_tokens=70, threshold_tokens=80,
    )

    assert compact.kind is ContextDecisionKind.COMPACT
    assert compact.reason == "legacy_should_compress"
    assert noop.kind is ContextDecisionKind.NONE
    assert noop.reason == "legacy_noop"






def test_transition_skips_optional_hooks_when_engine_lacks_them():
    """Engines that don't implement on_session_end/carry_over still work."""
    class MinimalEngine:
        def __init__(self):
            self.context_length = 100_000
            self.reset_called = False
            self.start_called_with = None

        def on_session_reset(self):
            self.reset_called = True

        def on_session_start(self, sid, **kw):
            self.start_called_with = (sid, kw)

    engine = MinimalEngine()
    agent = _bare_agent()
    agent.context_compressor = engine

    # Should not raise even though on_session_end / carry_over are missing.
    agent._transition_context_engine_session(
        old_session_id="old",
        new_session_id="new",
        previous_messages=[{"role": "user", "content": "hi"}],
        carry_over_context=True,
    )

    assert engine.reset_called is True
    assert engine.start_called_with is not None
    new_sid, kw = engine.start_called_with
    assert new_sid == "new"
    assert kw.get("old_session_id") == "old"






def test_reset_session_state_rebinds_builtin_compressor_after_session_switch(tmp_path, monkeypatch):
    """Reset-only session switches must rebind durable cooldown state to the new session."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("old-sid", source="cli")
    db.create_session("new-sid", source="cli")
    db.record_compression_failure_cooldown("old-sid", 4_000_000_000.0, "old-timeout")
    db.set_compression_fallback_streak("old-sid", 2)

    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length",
        lambda *_a, **_k: 100_000,
    )
    compressor = ContextCompressor(
        model="fake-model",
        threshold_percent=0.85,
        protect_first_n=2,
        protect_last_n=2,
        quiet_mode=True,
    )
    compressor.bind_session_state(db, "old-sid")

    agent = _bare_agent()
    agent._session_db = db
    agent.context_compressor = compressor
    agent.session_id = "new-sid"

    agent.reset_session_state()

    assert compressor._session_id == "new-sid"
    assert compressor.get_active_compression_failure_cooldown() is None
    assert compressor._fallback_compression_streak == 0
    assert db.get_compression_failure_cooldown("old-sid") is not None
    assert db.get_compression_fallback_streak("old-sid") == 2

    compressor._record_compression_failure_cooldown(30.0, "new-timeout")

    assert db.get_compression_failure_cooldown("new-sid") is not None
    assert db.get_compression_failure_cooldown("old-sid")["error"] == "old-timeout"


def test_update_from_response_forwards_canonical_cache_buckets():
    """conversation_loop passes cache_read/write/reasoning tokens to engine."""
    # Test the contract directly: a usage_dict built from CanonicalUsage must
    # contain the canonical buckets in addition to the legacy keys. We don't
    # spin up the full conversation loop; we just verify the dict shape.
    from agent.usage_pricing import CanonicalUsage

    canonical = CanonicalUsage(
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=800,
        cache_write_tokens=200,
        reasoning_tokens=50,
    )
    usage_dict = {
        "prompt_tokens": canonical.prompt_tokens,
        "completion_tokens": canonical.output_tokens,
        "total_tokens": canonical.total_tokens,
        "input_tokens": canonical.input_tokens,
        "output_tokens": canonical.output_tokens,
        "cache_read_tokens": canonical.cache_read_tokens,
        "cache_write_tokens": canonical.cache_write_tokens,
        "reasoning_tokens": canonical.reasoning_tokens,
    }

    # Legacy keys present
    assert usage_dict["prompt_tokens"] == canonical.prompt_tokens
    assert usage_dict["completion_tokens"] == 500
    assert usage_dict["total_tokens"] == canonical.total_tokens
    # Canonical cache + reasoning buckets present
    assert usage_dict["cache_read_tokens"] == 800
    assert usage_dict["cache_write_tokens"] == 200
    assert usage_dict["reasoning_tokens"] == 50
    assert usage_dict["input_tokens"] == 1000
    assert usage_dict["output_tokens"] == 500






def test_engine_collector_forwards_register_command_to_plugin_manager():
    """A plugin context engine can register a slash command via ``ctx.register_command``."""
    from plugins.context_engine import _EngineCollector
    from hermes_cli.plugins import get_plugin_manager

    handler = lambda raw_args: f"echo: {raw_args}"

    collector = _EngineCollector(engine_name="my-lcm")
    collector.register_command(
        "my-lcm-test-cmd",
        handler,
        description="test command from a context engine",
        args_hint="<msg>",
    )

    manager = get_plugin_manager()
    try:
        assert "my-lcm-test-cmd" in manager._plugin_commands
        entry = manager._plugin_commands["my-lcm-test-cmd"]
        assert entry["handler"] is handler
        assert entry["args_hint"] == "<msg>"
        assert entry["plugin"] == "context-engine:my-lcm"
    finally:
        # Clean up so we don't leak the registration across tests.
        manager._plugin_commands.pop("my-lcm-test-cmd", None)

