"""Coverage for external-memory execution-context scoping."""

import json
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
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from plugins.memory.mem0 import Mem0MemoryProvider
from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from tools.memory_tool import apply_memory_pending, memory_tool
from tools.memory_tool_store import MemoryStore


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
    """A cron run inside a worker process keeps its platform-derived context."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent_worker")
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)

    with non_dispatcher_owned_context():
        assert _memory_provider_init_kwargs(_agent(), " CRON ")["agent_context"] == "cron"


@pytest.mark.parametrize(
    ("platform", "expected"),
    [(" CRON ", "cron"), ("Flush", "flush"), ("subagent", "subagent"), (None, "primary")],
)
def test_platform_context_is_normalized_safely(monkeypatch, platform, expected):
    """Lifecycle platform spellings produce the matching provider context."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)

    kwargs = _memory_provider_init_kwargs(_agent(), platform)

    assert kwargs["platform"] == (str(platform).strip().lower() if platform else "cli")
    assert kwargs["agent_context"] == expected


def test_enum_like_platform_value_is_normalized(monkeypatch):
    """Platform enum values are normalized without leaking their wrapper representation."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)

    kwargs = _memory_provider_init_kwargs(_agent(), SimpleNamespace(value=" CRON "))

    assert kwargs["platform"] == "cron"
    assert kwargs["agent_context"] == "cron"


def test_delegated_child_precedes_platform_context(monkeypatch):
    """A delegated child remains a subagent even when its transport says cron."""
    monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "delegated")

    assert _memory_provider_init_kwargs(_agent(), "cron")["agent_context"] == "subagent"


class _DefaultAdmissionProvider(MemoryProvider):
    """Minimal provider exercising the base-class compatibility path."""

    name = "default-admission"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []


class _LegacyMutatingProvider(_DefaultAdmissionProvider):
    """Legacy provider that ignores context and mutates from every lifecycle hook."""

    name = "legacy-mutator"

    def __init__(self):
        """Initialize an empty mutation-call ledger."""
        self.calls = []
        self.read_only_closed = False

    def get_tool_schemas(self):
        """Expose one representative legacy write tool."""
        return [{"name": "legacy_write", "description": "write", "parameters": {"type": "object"}}]

    def handle_tool_call(self, tool_name, args, **kwargs):
        """Record a legacy explicit tool mutation."""
        self.calls.append(("tool", tool_name))
        return json.dumps({"success": True})

    def system_prompt_block(self):
        """Record provider prompt generation, which may have legacy side effects."""
        self.calls.append(("system_prompt", "generated"))
        return "legacy provider prompt"

    def queue_prefetch(self, query, **kwargs):
        """Record a legacy queued-prefetch hook."""
        self.calls.append(("queue_prefetch", query))

    def on_turn_start(self, turn_number, message, **kwargs):
        """Record a legacy turn-start hook."""
        self.calls.append(("turn_start", turn_number))

    def sync_turn(self, user_content, assistant_content, **kwargs):
        """Record a legacy completed-turn mutation."""
        self.calls.append(("sync", user_content))

    def on_session_end(self, messages):
        """Record a legacy session-end mutation."""
        self.calls.append(("session_end", len(messages)))

    def on_session_switch(self, new_session_id, **kwargs):
        """Record a legacy session-switch mutation."""
        self.calls.append(("session_switch", new_session_id))

    def on_pre_compress(self, messages):
        """Record a legacy pre-compression mutation."""
        self.calls.append(("pre_compress", len(messages)))
        return "persisted"

    def on_memory_write(self, action, target, content, metadata=None):
        """Record a legacy built-in-memory mirror mutation."""
        self.calls.append(("memory_write", action))

    def on_delegation(self, task, result, **kwargs):
        """Record a legacy delegation mutation."""
        self.calls.append(("delegation", task))

    def shutdown(self):
        """Record a legacy shutdown flush mutation."""
        self.calls.append(("shutdown", "write-flush"))

    def shutdown_read_only(self):
        """Release read-only resources without recording a durable mutation."""
        self.read_only_closed = True


def test_memory_provider_default_pre_admit_is_backward_compatible():
    """Existing subclasses inherit admission in every execution context."""
    provider = _DefaultAdmissionProvider()

    assert provider.pre_admit(platform="cli", agent_context="kanban") is True


@pytest.mark.parametrize("agent_context", ["kanban", "subagent", "cron", "flush"])
def test_memory_manager_centrally_denies_legacy_provider_mutations(agent_context):
    """Legacy providers cannot bypass non-primary isolation through tools or lifecycle hooks."""
    provider = _LegacyMutatingProvider()
    manager = MemoryManager(agent_context=agent_context)
    manager.add_provider(provider)

    assert manager.get_all_tool_schemas() == []
    assert manager.has_tool("legacy_write") is False
    assert "disabled" in json.loads(manager.handle_tool_call("legacy_write", {}))["error"].lower()
    assert manager.build_system_prompt() == ""
    manager.queue_prefetch_all("user")
    manager.on_turn_start(1, "user")
    manager.sync_all("user", "assistant")
    manager.on_session_end([{"role": "user", "content": "user"}])
    manager.on_session_switch("new-session")
    assert manager.on_pre_compress([]) == ""
    manager.on_memory_write("add", "memory", "fact")
    manager.on_delegation("task", "result")
    manager.shutdown_all()

    assert provider.calls == []
    assert provider.read_only_closed is True


def test_read_only_memory_store_blocks_tool_and_staged_replay(tmp_path, monkeypatch):
    """Built-in writes stay denied through both explicit tools and staged replay."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MemoryStore(writes_enabled=False)

    direct = json.loads(memory_tool(action="add", target="memory", content="must not persist", store=store))
    replay = apply_memory_pending(
        {"action": "add", "target": "memory", "content": "must not replay"},
        store,
    )

    assert direct["success"] is False
    assert replay["success"] is False
    assert not (tmp_path / "memories" / "MEMORY.md").exists()


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


def test_scheduler_cron_aiagent_keeps_mem0_recall_but_denies_all_writes(monkeypatch):
    """The scheduler's real AIAgent path activates Mem0 read-only at the central fence."""
    config = {"memory": {"provider": "mem0"}, "agent": {}}
    provider = Mem0MemoryProvider()
    backend = MagicMock()
    backend.search.return_value = []
    monkeypatch.setattr(provider, "is_available", lambda: True)
    monkeypatch.setattr(provider, "_create_backend", lambda: backend)
    monkeypatch.setattr("plugins.memory.mem0._load_config", lambda: {})
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)

    patches = _agent_init_patches(config, provider)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        from cron.scheduler import _construct_cron_agent, _CronAgentSetup
        from run_agent import AIAgent

        setup = _CronAgentSetup(
            model="",
            runtime={
                "api_key": "test-key-1234567890",
                "base_url": "https://openrouter.ai/api/v1",
            },
            max_iterations=2,
        )
        agent = _construct_cron_agent(
            AIAgent,
            {"enabled_toolsets": ["memory"]},
            config,
            setup,
            workdir=None,
            session_id="cron-test-session",
            session_db=None,
        )

    manager = agent._memory_manager
    assert manager is not None
    assert manager.agent_context == "cron"
    assert manager.writes_enabled is False
    assert manager.prefetch_all("remember this", session_id=agent.session_id) == ""
    backend.search.assert_called_once()
    denied = json.loads(manager.handle_tool_call("mem0_add", {"content": "must not persist"}))
    assert "disabled" in denied["error"].lower()
    manager.sync_all("user", "assistant", session_id=agent.session_id)
    manager.flush_pending(timeout=1)
    backend.add.assert_not_called()
