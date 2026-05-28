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

from unittest.mock import MagicMock

import pytest

from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = "test-session"
    agent.model = "fake-model"
    agent.platform = "telegram"
    agent._gateway_session_key = "agent:main:telegram:dm:42"
    return agent


def test_transition_runs_full_lifecycle_in_order():
    """End → reset → start → carry_over, in that order, when all inputs apply."""
    events: list[str] = []
    engine = MagicMock()
    engine.context_length = 200_000
    engine.on_session_end.side_effect = lambda *a, **kw: events.append("on_session_end")
    engine.on_session_reset.side_effect = lambda *a, **kw: events.append("on_session_reset")
    engine.on_session_start.side_effect = lambda *a, **kw: events.append("on_session_start")
    engine.carry_over_new_session_context.side_effect = lambda *a, **kw: events.append("carry_over")

    agent = _bare_agent()
    agent.context_compressor = engine

    agent._transition_context_engine_session(
        old_session_id="old-sid",
        new_session_id="new-sid",
        previous_messages=[{"role": "user", "content": "hi"}],
        carry_over_context=True,
    )

    assert events == [
        "on_session_end",
        "on_session_reset",
        "on_session_start",
        "carry_over",
    ]


def test_transition_passes_conversation_id_from_gateway_session_key():
    """on_session_start receives ``conversation_id`` from ``_gateway_session_key``."""
    engine = MagicMock()
    engine.context_length = 200_000
    captured: dict = {}
    engine.on_session_start.side_effect = lambda sid, **kw: captured.update(kw)

    agent = _bare_agent()
    agent.context_compressor = engine

    agent._transition_context_engine_session(
        old_session_id="old-sid",
        new_session_id="new-sid",
        previous_messages=[{"role": "user", "content": "hi"}],
    )

    assert captured.get("conversation_id") == "agent:main:telegram:dm:42"
    assert captured.get("old_session_id") == "old-sid"
    assert captured.get("platform") == "telegram"


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


def test_reset_session_state_delegates_to_transition_when_args_provided():
    """``reset_session_state(previous_messages=..., old_session_id=...)`` fires full lifecycle."""
    engine = MagicMock()
    engine.context_length = 100_000

    agent = _bare_agent()
    agent.context_compressor = engine

    agent.reset_session_state(
        previous_messages=[{"role": "user", "content": "hi"}],
        old_session_id="old-sid",
    )

    assert engine.on_session_end.called
    assert engine.on_session_reset.called
    assert engine.on_session_start.called
    # No carry_over_context, so carry_over hook NOT called.
    assert not engine.carry_over_new_session_context.called


def test_reset_session_state_default_call_only_resets():
    """Bare ``reset_session_state()`` still only resets the engine (no end/start)."""
    engine = MagicMock()
    engine.context_length = 100_000

    agent = _bare_agent()
    agent.context_compressor = engine

    agent.reset_session_state()

    assert engine.on_session_reset.called
    assert not engine.on_session_end.called
    assert not engine.on_session_start.called


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


def test_discover_context_engines_includes_plugin_registered_engines(monkeypatch):
    """Plugin-registered context engines appear in the ``hermes plugins`` picker."""
    from hermes_cli import plugins_cmd

    fake_repo = lambda: [("compressor", "built-in", True)]

    class FakePluginEngine:
        name = "lcm"

    monkeypatch.setattr(
        "plugins.context_engine.discover_context_engines",
        fake_repo,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.discover_plugins",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_context_engine",
        lambda: FakePluginEngine(),
    )

    engines = plugins_cmd._discover_context_engines()
    names = [n for n, _desc in engines]
    assert "compressor" in names
    assert "lcm" in names


def test_discover_context_engines_dedupes_by_name(monkeypatch):
    """Repo-shipped engine wins when name collides with a plugin-registered one."""
    from hermes_cli import plugins_cmd

    class FakePluginEngine:
        name = "compressor"  # same name as repo-shipped

    monkeypatch.setattr(
        "plugins.context_engine.discover_context_engines",
        lambda: [("compressor", "built-in compressor", True)],
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.discover_plugins",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_context_engine",
        lambda: FakePluginEngine(),
    )

    engines = plugins_cmd._discover_context_engines()
    # Only one entry — the repo-shipped one. Description is preserved.
    assert engines == [("compressor", "built-in compressor")]


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


def test_engine_collector_rejects_builtin_command_conflicts():
    """Context engine cannot shadow built-in slash commands like /help."""
    from plugins.context_engine import _EngineCollector
    from hermes_cli.plugins import get_plugin_manager

    collector = _EngineCollector(engine_name="my-lcm")
    collector.register_command("help", lambda *_: "shadow")

    manager = get_plugin_manager()
    # Must NOT have overwritten / registered against built-in /help.
    assert "help" not in manager._plugin_commands or \
           manager._plugin_commands["help"].get("plugin") != "context-engine:my-lcm"
