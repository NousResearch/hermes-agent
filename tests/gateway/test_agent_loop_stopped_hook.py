"""Real plugin discovery and interrupt delivery, including idle and failure controls."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key


@pytest.fixture
def interrupt_plugin(tmp_path, monkeypatch):
    from hermes_cli.plugins import discover_plugins, get_plugin_manager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin_dir = tmp_path / "plugins" / "interrupt-observer"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: interrupt-observer\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "events = []\n"
        "fail = False\n"
        "def observe(*, session_key, platform, reason, invalidation_reason):\n"
        "    events.append(dict(session_key=session_key, platform=platform,\n"
        "                       reason=reason, invalidation_reason=invalidation_reason))\n"
        "    if fail:\n"
        "        raise RuntimeError('plugin exploded')\n"
        "def register(ctx):\n"
        "    ctx.register_hook('agent_loop_stopped', observe)\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        "plugins:\n  enabled: [interrupt-observer]\n", encoding="utf-8"
    )
    discover_plugins()
    return get_plugin_manager()._plugins["interrupt-observer"].module


class InterruptAdapter:
    def __init__(self):
        self.interrupted = []

    async def interrupt_session_activity(self, session_key, chat_id, *, metadata=None):
        self.interrupted.append((session_key, chat_id))


def _make_turn(agent_kind):
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", chat_type="dm"
    )
    session_key = build_session_key(source)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)}
    )
    adapter = InterruptAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    # Keep gateway-status persistence outside this plugin/turn boundary test.
    runner._persist_active_agents = Mock()
    state = runner._session_state(session_key)
    state.persistent.pending_command_text = "queued command"
    state.conversation.ephemeral_pin = "cached prompt context"
    agent = SimpleNamespace(hard_interrupt=Mock())
    state.turn.agent = {
        "running": agent, "pending": _AGENT_PENDING_SENTINEL, "missing": None
    }[agent_kind]
    generation = runner._begin_session_run_generation(session_key)
    return runner, source, session_key, agent, adapter, state, generation


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_kind", ["running", "pending", "missing"])
@pytest.mark.parametrize("reason,invalidation_reason", [
    ("user_stop", "stop_command"), ("session_reset", "new_command")
])
async def test_registered_plugin_observes_only_running_turns(
    interrupt_plugin, agent_kind, reason, invalidation_reason
):
    runner, source, key, agent, adapter, state, generation = _make_turn(agent_kind)

    await runner._interrupt_and_clear_session(
        key, source, interrupt_reason=reason, invalidation_reason=invalidation_reason
    )

    expected = [{"session_key": key, "platform": source.platform.value,
                 "reason": reason, "invalidation_reason": invalidation_reason}]
    assert interrupt_plugin.events == (expected if agent_kind == "running" else [])
    if agent_kind == "running":
        agent.hard_interrupt.assert_called_once_with(reason)
    else:
        agent.hard_interrupt.assert_not_called()
    assert adapter.interrupted == [(key, source.chat_id)]
    assert not runner._is_session_run_current(key, generation)
    assert state.turn.agent is None
    assert state.persistent.pending_command_text is None
    assert state.conversation.ephemeral_pin is None


@pytest.mark.asyncio
async def test_failing_plugin_cannot_prevent_interrupt_cleanup(interrupt_plugin):
    interrupt_plugin.fail = True
    runner, source, key, agent, adapter, state, generation = _make_turn("running")

    await runner._interrupt_and_clear_session(
        key, source, interrupt_reason="user_stop", invalidation_reason="stop_command"
    )

    assert interrupt_plugin.events == [{
        "session_key": key, "platform": source.platform.value,
        "reason": "user_stop", "invalidation_reason": "stop_command"
    }]
    agent.hard_interrupt.assert_called_once_with("user_stop")
    assert adapter.interrupted == [(key, source.chat_id)]
    assert not runner._is_session_run_current(key, generation)
    assert state.turn.agent is None
    assert state.persistent.pending_command_text is None
    assert state.conversation.ephemeral_pin is None
