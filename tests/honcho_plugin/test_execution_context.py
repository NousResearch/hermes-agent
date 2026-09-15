"""Execution-context isolation policy for the Honcho provider."""

from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


@pytest.mark.parametrize("agent_context", ["subagent", "kanban", "cron", "flush"])
def test_pre_admit_rejects_non_primary_contexts(agent_context):
    """The cheap admission gate rejects workers before availability probing."""
    provider = HonchoMemoryProvider()

    assert provider.pre_admit(platform="cli", agent_context=agent_context) is False


def test_pre_admit_keeps_primary_and_legacy_cron_platform_behavior():
    """Primary agents are admitted while the cron platform is rejected independently."""
    provider = HonchoMemoryProvider()

    assert provider.pre_admit(platform="cli", agent_context="primary") is True
    assert provider.pre_admit(platform="cron", agent_context="primary") is False


@pytest.mark.parametrize("agent_context", ["subagent", "kanban", "cron", "flush"])
def test_non_primary_context_skips_configuration_and_session_initialization(monkeypatch, agent_context):
    """Worker contexts must return before config, client, or session initialization."""
    config_loader = MagicMock(side_effect=AssertionError("configuration must not load"))
    client_builder = MagicMock(side_effect=AssertionError("client must not initialize"))
    session_initializer = MagicMock(side_effect=AssertionError("session must not initialize"))
    monkeypatch.setattr(HonchoClientConfig, "from_global_config", config_loader)
    monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", client_builder)
    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", session_initializer)

    provider = HonchoMemoryProvider()
    provider.initialize("worker-session", agent_context=agent_context, platform="cli")

    assert provider._cron_skipped is True
    config_loader.assert_not_called()
    client_builder.assert_not_called()
    session_initializer.assert_not_called()


def test_cron_platform_skips_initialization_without_agent_context(monkeypatch):
    """The legacy cron platform gate remains isolated for backward compatibility."""
    config_loader = MagicMock(side_effect=AssertionError("configuration must not load"))
    monkeypatch.setattr(HonchoClientConfig, "from_global_config", config_loader)

    provider = HonchoMemoryProvider()
    provider.initialize("cron-session", platform="cron")

    assert provider._cron_skipped is True
    config_loader.assert_not_called()


def test_primary_context_keeps_normal_configuration_path(monkeypatch):
    """Primary agents continue loading provider configuration as before."""
    disabled_config = HonchoClientConfig(enabled=False)
    config_loader = MagicMock(return_value=disabled_config)
    monkeypatch.setattr(HonchoClientConfig, "from_global_config", config_loader)

    provider = HonchoMemoryProvider()
    provider.initialize("primary-session", agent_context="primary", platform="cli")

    assert provider._cron_skipped is False
    config_loader.assert_called_once_with()
