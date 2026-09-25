"""Regression tests for SystemExit containment in memory providers and plugin probes (#123042).

A memory provider or plugin probe that triggers sys.exit() / SystemExit (for instance,
via tools.lazy_deps.install_specs -> stop_for_relaunch) must not terminate the agent
process. KeyboardInterrupt must still propagate cleanly to preserve user interruption.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from agent import agent_init, provider_registry
from plugins import plugin_loader


class SystemExitAvailableProvider:
    name = "system-exit-available"

    def is_available(self):
        raise SystemExit(1)

    def unavailable_reason(self):
        return "fail"


class SystemExitUnavailableReasonProvider:
    name = "system-exit-reason"

    def is_available(self):
        return False

    def unavailable_reason(self):
        raise SystemExit(2)


class KeyboardInterruptAvailableProvider:
    name = "keyboard-interrupt-available"

    def is_available(self):
        raise KeyboardInterrupt()

    def unavailable_reason(self):
        return "interrupt"


class KeyboardInterruptReasonProvider:
    name = "keyboard-interrupt-reason"

    def is_available(self):
        return False

    def unavailable_reason(self):
        raise KeyboardInterrupt()


def test_memory_provider_is_available_system_exit_contained():
    """SystemExit raised from a memory provider's is_available() does not kill the process."""
    agent = SimpleNamespace(
        enabled_toolsets=[],
        disabled_toolsets=[],
        _memory_store=None,
        _memory_manager=None,
        tools=[],
        valid_tool_names=set(),
    )
    agent_cfg = {"memory": {"provider": "system-exit-available"}}
    provider = SystemExitAvailableProvider()

    with (
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("tools.memory_tool.get_builtin_memory_config", return_value={"provider": "system-exit-available"}),
        patch("tools.memory_tool.get_builtin_memory_store_flags", return_value=(False, False)),
    ):
        agent_init._init_memory(agent, agent_cfg, skip_memory=False, platform=None)

    assert agent._memory_manager is None


def test_memory_provider_unavailable_reason_system_exit_contained():
    """SystemExit raised from unavailable_reason() is suppressed without terminating."""
    agent = SimpleNamespace(
        enabled_toolsets=[],
        disabled_toolsets=[],
        _memory_store=None,
        _memory_manager=None,
        tools=[],
        valid_tool_names=set(),
    )
    agent_cfg = {"memory": {"provider": "system-exit-reason"}}
    provider = SystemExitUnavailableReasonProvider()

    agent_init._warned_unavailable_providers.clear()
    with (
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("tools.memory_tool.get_builtin_memory_config", return_value={"provider": "system-exit-reason"}),
        patch("tools.memory_tool.get_builtin_memory_store_flags", return_value=(False, False)),
        patch("agent.agent_init._warn_memory_provider_unavailable") as mock_warn,
    ):
        agent_init._init_memory(agent, agent_cfg, skip_memory=False, platform=None)

    assert agent._memory_manager is None
    mock_warn.assert_called_once_with("system-exit-reason", "")


def test_memory_provider_keyboard_interrupt_propagates_from_is_available():
    """KeyboardInterrupt from is_available() must NOT be caught; must propagate."""
    agent = SimpleNamespace(
        enabled_toolsets=[],
        disabled_toolsets=[],
        _memory_store=None,
        _memory_manager=None,
        tools=[],
        valid_tool_names=set(),
    )
    agent_cfg = {"memory": {"provider": "keyboard-interrupt-available"}}
    provider = KeyboardInterruptAvailableProvider()

    with (
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("tools.memory_tool.get_builtin_memory_config", return_value={"provider": "keyboard-interrupt-available"}),
        patch("tools.memory_tool.get_builtin_memory_store_flags", return_value=(False, False)),
    ):
        with pytest.raises(KeyboardInterrupt):
            agent_init._init_memory(agent, agent_cfg, skip_memory=False, platform=None)


def test_memory_provider_keyboard_interrupt_propagates_from_unavailable_reason():
    """KeyboardInterrupt from unavailable_reason() must NOT be caught; must propagate."""
    agent = SimpleNamespace(
        enabled_toolsets=[],
        disabled_toolsets=[],
        _memory_store=None,
        _memory_manager=None,
        tools=[],
        valid_tool_names=set(),
    )
    agent_cfg = {"memory": {"provider": "keyboard-interrupt-reason"}}
    provider = KeyboardInterruptReasonProvider()

    agent_init._warned_unavailable_providers.clear()
    with (
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("tools.memory_tool.get_builtin_memory_config", return_value={"provider": "keyboard-interrupt-reason"}),
        patch("tools.memory_tool.get_builtin_memory_store_flags", return_value=(False, False)),
    ):
        with pytest.raises(KeyboardInterrupt):
            agent_init._init_memory(agent, agent_cfg, skip_memory=False, platform=None)


def test_is_available_safe_contains_system_exit():
    """provider_registry.is_available_safe catches SystemExit and returns False."""
    logger = MagicMock()
    provider = SystemExitAvailableProvider()
    assert provider_registry.is_available_safe(provider, logger, "test %s %s") is False


def test_is_available_safe_propagates_keyboard_interrupt():
    """provider_registry.is_available_safe lets KeyboardInterrupt propagate."""
    logger = MagicMock()
    provider = KeyboardInterruptAvailableProvider()
    with pytest.raises(KeyboardInterrupt):
        provider_registry.is_available_safe(provider, logger, "test %s %s")


def test_probe_availability_contains_system_exit():
    """plugin_loader.probe_availability catches SystemExit and returns False."""
    def _raising_load():
        raise SystemExit(1)

    assert plugin_loader.probe_availability(_raising_load) is False

    provider = SystemExitAvailableProvider()
    assert plugin_loader.probe_availability(lambda: provider) is False


def test_probe_availability_propagates_keyboard_interrupt():
    """plugin_loader.probe_availability lets KeyboardInterrupt propagate."""
    def _interrupting_load():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        plugin_loader.probe_availability(_interrupting_load)


def test_context_engine_selection_system_exit_contained():
    """_select_context_engine contains SystemExit from plugins/context_engine and falls back."""
    agent_cfg = {"context": {"engine": "custom_engine"}}
    with patch("plugins.context_engine.load_context_engine", side_effect=SystemExit(1)):
        result = agent_init._select_context_engine(agent_cfg)
        assert result is None
