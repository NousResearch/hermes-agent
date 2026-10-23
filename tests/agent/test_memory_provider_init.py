"""Regression tests for memory provider selection during AIAgent init."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def test_shutdown_memory_provider_is_idempotent():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    manager = MagicMock()
    agent = object.__new__(AIAgent)
    agent._memory_manager = manager
    agent.context_compressor = None
    agent.session_id = "session-1"

    agent.shutdown_memory_provider([{"role": "user", "content": "one"}])
    agent.shutdown_memory_provider([{"role": "user", "content": "two"}])

    manager.on_session_end.assert_called_once()
    manager.shutdown_all.assert_called_once()


def test_blank_memory_provider_stays_opt_out():
    """Blank memory.provider loads no external provider and never rewrites the config."""
    cfg = {"memory": {"provider": ""}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_close_shuts_down_memory_provider():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.context_compressor = None
    agent.session_id = ""
    agent._session_messages = []

    agent.close()

    agent._memory_manager.shutdown_all.assert_called_once()


def test_aiagent_forwards_user_id_alt_to_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-alt",
            platform="feishu",
            user_id="open-id",
            user_id_alt="union-id",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-alt"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
    assert provider.init_kwargs["platform"] == "feishu"
    assert "warning_callback" not in provider.init_kwargs
    assert "status_callback" not in provider.init_kwargs


class CoreShadowProvider:
    """Provider that tries to register tools shadowing built-in core tools."""

    name = "core-shadow"

    def get_tool_schemas(self):
        return [
            {"name": "clarify", "description": "shadows built-in clarify"},
            {"name": "delegate_task", "description": "shadows built-in delegate"},
            {"name": "honcho_search", "description": "legit memory tool"},
        ]


def test_core_tool_names_rejected_from_memory_routing_table():
    """Memory tools shadowing core tool names are rejected at registration (#40466).

    Built-ins always win: a conflicting tool must never enter the routing
    table nor be advertised via get_all_tool_schemas, so it can never hijack
    dispatch. The non-conflicting tool is preserved.
    """
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    mm.add_provider(CoreShadowProvider())

    # Reserved names never enter the routing table
    assert not mm.has_tool("clarify")
    assert not mm.has_tool("delegate_task")
    assert "clarify" not in mm._tool_to_provider
    assert "delegate_task" not in mm._tool_to_provider

    # Non-conflicting tool survives
    assert mm.has_tool("honcho_search")
    assert "honcho_search" in mm._tool_to_provider

    # Manager never advertises a schema it would refuse to route
    schema_names = {s.get("name") for s in mm.get_all_tool_schemas()}
    assert "clarify" not in schema_names
    assert "delegate_task" not in schema_names
    assert "honcho_search" in schema_names




def test_aiagent_reuses_handed_in_memory_manager_without_reinitializing():
    """A caller that rebuilds the agent per turn (gateway api_server) hands back the session's manager:
    the provider keeps its state — no second load, no second initialize (#120116)."""
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}
    common = dict(
        api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
        skip_context_files=True, skip_memory=False, session_id="sess-api", platform="api_server",
    )
    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider) as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        first = AIAgent(**common)
        manager = first._memory_manager
        assert manager is not None and load_memory_provider.call_count == 1
        provider.init_session_id = None  # a re-initialize would set it again

        second = AIAgent(memory_manager=manager, **common)

    assert second._memory_manager is manager
    assert load_memory_provider.call_count == 1
    assert provider.init_session_id is None


class UnavailableMemoryProvider:
    """Provider that loads but reports itself unavailable (broken add-on)."""

    name = "mnemosyne"

    def is_available(self):
        return False

    def unavailable_reason(self):
        return ""

    def initialize(self, session_id, **kwargs):  # pragma: no cover - never called
        raise AssertionError("initialize should not run for an unavailable provider")

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


@pytest.fixture()
def _fresh_memory_warn_once():
    """The warn-once sets are module level; isolate them per test."""
    from agent import agent_init
    from hermes_cli import memory_provider_migration as migration

    sets = (
        agent_init._warned_unavailable_providers,
        agent_init._warned_failed_providers,
        migration._attempted,
    )
    for s in sets:
        s.clear()
    yield
    for s in sets:
        s.clear()


def _build_agent_with_memory_provider(load_patch, *extra_patches, platform="telegram"):
    cfg = {"memory": {"provider": "mnemosyne"}, "agent": {}}
    with ExitStack() as stack:
        for p in (
            patch("hermes_cli.config.load_config", return_value=cfg),
            patch("hermes_cli.config.load_config_readonly", return_value=cfg),
            load_patch,
            patch("agent.model_metadata.get_model_context_length", return_value=204_800),
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            *extra_patches,
        ):
            stack.enter_context(p)
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-49200",
            platform=platform,
        )


def _pending_notice_texts(agent):
    return [n.text for n in getattr(agent, "_pending_startup_notices", [])]


def _present_but_unloadable():
    """The provider resolves on disk (e.g. a dangling plugin symlink) but does not load."""
    return (
        patch("plugins.memory.load_memory_provider", return_value=None),
        patch("pm.install.lazy_installs_allowed", return_value=True),
        patch("hermes_cli.memory_provider_migration.configured_provider", return_value="mnemosyne"),
        patch("hermes_cli.memory_provider_migration.provider_present", return_value=True),
    )


@pytest.mark.usefixtures("_fresh_memory_warn_once")
def test_configured_memory_provider_that_resolves_but_fails_to_load_warns_user():
    """#49200: a provider that resolves but fails to load must reach the user, not just DEBUG.

    Startup recovery stays silent here (the provider is present), so without this warning
    the agent quietly runs on built-in memory. The gateway binds its callbacks after init,
    so the warning is held as a startup notice and replayed on the first turn.
    """
    load_patch, *extra = _present_but_unloadable()
    agent = _build_agent_with_memory_provider(load_patch, *extra)

    assert agent._memory_manager is None
    notices = _pending_notice_texts(agent)
    assert len(notices) == 1
    assert "mnemosyne" in notices[0] and "failed to load" in notices[0]


@pytest.mark.usefixtures("_fresh_memory_warn_once")
def test_configured_memory_provider_failure_reaches_late_bound_notice_callback():
    """The gateway sets its callbacks after __init__; the held notice reaches them on replay."""
    load_patch, *extra = _present_but_unloadable()
    agent = _build_agent_with_memory_provider(load_patch, *extra)

    delivered = []
    agent.notice_callback = delivered.append
    agent._replay_startup_warnings()

    assert len(delivered) == 1
    assert "mnemosyne" in delivered[0].text and "failed to load" in delivered[0].text
    assert _pending_notice_texts(agent) == []


@pytest.mark.usefixtures("_fresh_memory_warn_once")
def test_configured_memory_provider_failure_warns_once_per_home():
    """The gateway builds an agent per message; the warning must not repeat every turn."""
    load_patch, *extra = _present_but_unloadable()
    first = _build_agent_with_memory_provider(load_patch, *extra)
    load_patch, *extra = _present_but_unloadable()
    second = _build_agent_with_memory_provider(load_patch, *extra)

    assert len(_pending_notice_texts(first)) == 1
    assert _pending_notice_texts(second) == []


@pytest.mark.usefixtures("_fresh_memory_warn_once")
def test_configured_memory_provider_failure_reaches_cli_warning():
    """On the CLI the startup warning prints through ``_emit_warning``."""
    load_patch, *extra = _present_but_unloadable()
    with patch("agent.status_output.StatusOutputMixin._emit_warning", autospec=True) as emit_warning:
        _build_agent_with_memory_provider(load_patch, *extra, platform="cli")

    messages = [call.args[1] for call in emit_warning.call_args_list]
    assert any("mnemosyne" in m and "failed to load" in m for m in messages)


@pytest.mark.usefixtures("_fresh_memory_warn_once")
def test_memory_provider_init_exception_warns_user():
    """A provider init that raises is reported to the user, not only to the log."""
    agent = _build_agent_with_memory_provider(
        patch("plugins.memory.load_memory_provider", side_effect=ImportError("boom")),
    )

    assert agent._memory_manager is None
    notices = _pending_notice_texts(agent)
    assert len(notices) == 1
    assert "mnemosyne" in notices[0] and "boom" in notices[0]


@pytest.mark.usefixtures("_fresh_memory_warn_once")
def test_unavailable_memory_provider_warns_user_once():
    """A provider that loads but is unavailable keeps its own warning, with no second one."""
    agent = _build_agent_with_memory_provider(
        patch("plugins.memory.load_memory_provider", return_value=UnavailableMemoryProvider()),
    )

    assert agent._memory_manager is None
    notices = _pending_notice_texts(agent)
    assert len(notices) == 1
    assert "reports unavailable" in notices[0]


@pytest.mark.usefixtures("_fresh_memory_warn_once")
def test_recovery_report_is_not_followed_by_a_second_warning():
    """When startup recovery already told the user why, no extra 'failed to load' notice."""
    agent = _build_agent_with_memory_provider(
        patch("plugins.memory.load_memory_provider", return_value=None),
        patch("pm.install.lazy_installs_allowed", return_value=False),
    )

    notices = _pending_notice_texts(agent)
    assert len(notices) == 1
    assert "not installed" in notices[0]
