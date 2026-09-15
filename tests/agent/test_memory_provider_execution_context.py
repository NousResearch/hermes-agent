"""Coverage for external-memory execution-context scoping."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import agent_init
from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _memory_provider_init_kwargs
from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    delegated_child_context,
    non_dispatcher_owned_context,
)
from agent.memory_provider import MemoryProvider
from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


def _agent() -> SimpleNamespace:
    """Build the minimal agent shape consumed by provider initialization."""
    return SimpleNamespace(
        session_id="test-session",
        _session_db=None,
        _emit_warning=lambda _message: None,
        _emit_status=lambda _message: None,
        **{f"_{name}": None for name in _GATEWAY_IDENTITY_PARAMS},
    )


@pytest.mark.parametrize("platform", ["cli", "gateway"])
def test_primary_context_without_worker_markers_remains_backward_compatible(monkeypatch, platform):
    """Ordinary CLI and gateway agents retain the historical primary context."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)

    assert _memory_provider_init_kwargs(_agent(), platform)["agent_context"] == "primary"


def test_memory_provider_context_is_kanban_for_dispatcher_worker(monkeypatch):
    """Dispatcher-owned workers expose the dedicated kanban context."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_memory_context")
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)

    assert _memory_provider_init_kwargs(_agent(), "cli")["agent_context"] == "kanban"


def test_memory_provider_context_is_subagent_for_delegated_child(monkeypatch):
    """Delegated child processes expose the existing subagent context."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)

    with delegated_child_context():
        assert _memory_provider_init_kwargs(_agent(), "cli")["agent_context"] == "subagent"


def test_delegated_child_context_wins_over_inherited_kanban_marker(monkeypatch):
    """Delegated children may inherit parent env before launcher scrubbing."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent_worker")
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "delegated")

    assert _memory_provider_init_kwargs(_agent(), "cli")["agent_context"] == "subagent"


def test_primary_context_inside_non_dispatcher_guard_remains_backward_compatible(monkeypatch):
    """Cron-like in-process work must not impersonate its parent worker."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent_worker")
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)

    with non_dispatcher_owned_context():
        assert _memory_provider_init_kwargs(_agent(), "cli")["agent_context"] == "primary"


class _DefaultAdmissionProvider(MemoryProvider):
    """Minimal provider exercising the base-class compatibility path."""

    name = "default-admission"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []


def test_memory_provider_default_pre_admit_is_backward_compatible():
    """Existing subclasses inherit admission in every execution context."""
    provider = _DefaultAdmissionProvider()

    assert provider.pre_admit(platform="cli", agent_context="kanban") is True


def _agent_init_patches(config, provider):
    """Return the standard isolated AIAgent initialization patch set."""
    return (
        patch("hermes_cli.config.load_config", return_value=config),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    )


@pytest.mark.parametrize("worker_context", ["kanban", "subagent"])
def test_aiagent_denied_honcho_context_never_probes_or_initializes(monkeypatch, worker_context):
    """Real worker startup rejects Honcho before config, availability, warnings, or sessions."""
    config = {"memory": {"provider": "honcho"}, "agent": {}}
    provider = HonchoMemoryProvider()
    availability = MagicMock(side_effect=AssertionError("availability must not run"))
    unavailable_reason = MagicMock(side_effect=AssertionError("unavailable reason must not run"))
    config_loader = MagicMock(side_effect=AssertionError("configuration must not load"))
    client_builder = MagicMock(side_effect=AssertionError("client must not initialize"))
    session_initializer = MagicMock(side_effect=AssertionError("session must not initialize"))
    warning = MagicMock(side_effect=AssertionError("warning must not emit"))
    monkeypatch.setattr(provider, "is_available", availability)
    monkeypatch.setattr(provider, "unavailable_reason", unavailable_reason)
    monkeypatch.setattr(HonchoClientConfig, "from_global_config", config_loader)
    monkeypatch.setattr("plugins.memory.honcho.client.get_honcho_client", client_builder)
    monkeypatch.setattr(HonchoMemoryProvider, "_do_session_init", session_initializer)
    monkeypatch.setattr(agent_init, "_warn_memory_provider_unavailable", warning)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    context = (
        patch.dict("os.environ", {"HERMES_KANBAN_TASK": "t_memory_context"})
        if worker_context == "kanban"
        else delegated_child_context()
    )

    patches = _agent_init_patches(config, provider)
    with context:
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
            )

    assert agent._memory_manager is None
    availability.assert_not_called()
    unavailable_reason.assert_not_called()
    config_loader.assert_not_called()
    client_builder.assert_not_called()
    session_initializer.assert_not_called()
    warning.assert_not_called()


def test_aiagent_primary_honcho_still_checks_availability(monkeypatch):
    """Ordinary primary startup preserves the historical availability path."""
    config = {"memory": {"provider": "honcho"}, "agent": {}}
    provider = HonchoMemoryProvider()
    availability = MagicMock(return_value=False)
    unavailable_reason = MagicMock(return_value="not configured")
    monkeypatch.setattr(provider, "is_available", availability)
    monkeypatch.setattr(provider, "unavailable_reason", unavailable_reason)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    monkeypatch.setattr(agent_init, "_warned_unavailable_providers", set())

    patches = _agent_init_patches(config, provider)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    availability.assert_called_once_with()
    unavailable_reason.assert_called_once_with()
