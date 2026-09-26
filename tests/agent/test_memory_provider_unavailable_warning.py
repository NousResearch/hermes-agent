"""Regression tests for NousResearch/hermes-agent#2765.

A memory provider configured via ``memory.provider`` but reporting
``is_available() == False`` (e.g. missing credentials, or a systemd/gateway
service that didn't inherit ``~/.hermes/.env``) used to be dropped silently.
``agent_init`` now emits a one-time, deduped warning instead.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import agent_init


def test_warns_once_and_dedupes(caplog):
    agent_init._warned_unavailable_providers.clear()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight")
        agent_init._warn_memory_provider_unavailable("hindsight")

    warnings = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(warnings) == 1, "should warn exactly once per provider (gateway dedup)"
    assert "hindsight" in warnings[0].getMessage()


def test_distinct_providers_each_warn(caplog):
    agent_init._warned_unavailable_providers.clear()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight")
        agent_init._warn_memory_provider_unavailable("mem0")

    warnings = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(warnings) == 2


def test_provider_reason_is_appended(caplog):
    # A provider's unavailable_reason() (e.g. the local_embedded install hint,
    # #7718) reaches the user through this warning — the only path that runs
    # when the provider is unavailable and thus never initialized.
    agent_init._warned_unavailable_providers.clear()
    hint = "Install the embedded runtime with: uv pip install hindsight-all."
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight", hint)

    warnings = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(warnings) == 1
    assert hint in warnings[0].getMessage()



class _ExitProvider:
    name = "exit-provider"

    def __init__(self, *, available=True, reason_exit=False, interrupt=False):
        self.available = available
        self.reason_exit = reason_exit
        self.interrupt = interrupt

    def is_available(self):
        if self.interrupt:
            raise KeyboardInterrupt()
        if self.available:
            raise SystemExit(1)
        return False

    def unavailable_reason(self):
        if self.reason_exit:
            raise SystemExit(2)
        return "not available"


def _memory_init_fixture(monkeypatch, provider):
    import agent.memory_manager as memory_manager
    import plugins.memory as memory_plugins
    import tools.memory_tool as memory_tool

    monkeypatch.setattr(
        memory_tool,
        "get_builtin_memory_config",
        lambda _config: {
            "provider": "exit-provider",
            "memory_enabled": False,
            "user_profile_enabled": False,
        },
    )
    monkeypatch.setattr(memory_tool, "get_builtin_memory_store_flags", lambda _config: (False, False))
    monkeypatch.setattr(memory_plugins, "load_memory_provider", lambda _name: provider)
    monkeypatch.setattr(memory_manager, "inject_memory_provider_tools", lambda _agent: None)
    monkeypatch.setattr(agent_init, "is_core_memory_provider", lambda _name: False)

    logger = SimpleNamespace(debug=Mock(), info=Mock(), warning=Mock())
    monkeypatch.setattr(agent_init, "_ra", lambda: SimpleNamespace(logger=logger))
    agent = SimpleNamespace(enabled_toolsets=[], disabled_toolsets=[])

    return agent, logger


def test_memory_provider_systemexit_is_contained(monkeypatch):
    agent, logger = _memory_init_fixture(monkeypatch, _ExitProvider())

    agent_init._init_memory(agent, {}, False, "cli")

    assert agent._memory_manager is None
    logger.warning.assert_called_once()
    assert "Memory provider plugin init failed" in logger.warning.call_args.args[0]


def test_unavailable_reason_systemexit_does_not_abort_init(monkeypatch):
    agent_init._warned_unavailable_providers.clear()
    provider = _ExitProvider(available=False, reason_exit=True)
    agent, _logger = _memory_init_fixture(monkeypatch, provider)

    agent_init._init_memory(agent, {}, False, "cli")

    assert agent._memory_manager is None
    assert "exit-provider" in agent_init._warned_unavailable_providers


def test_memory_provider_keyboard_interrupt_still_propagates(monkeypatch):
    agent, _logger = _memory_init_fixture(monkeypatch, _ExitProvider(interrupt=True))

    with pytest.raises(KeyboardInterrupt):
        agent_init._init_memory(agent, {}, False, "cli")
