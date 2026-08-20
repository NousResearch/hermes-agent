"""Built-in memory disabled in config must leave no dead surface behind.

Setting ``memory.memory_enabled: false`` and ``memory.user_profile_enabled:
false`` stops ``agent_init`` from building a ``MemoryStore``, so the ``memory``
tool dispatches against ``store=None`` and every call comes back "Memory is not
available". Before the fix the tool stayed in the schema and MEMORY_GUIDANCE
stayed in the system prompt, so users running a third-party provider (Hindsight,
Mem0, …) paid for both on every API call with no way to drop them — listing
``memory`` under ``disabled_toolsets`` takes the provider's tools down too.

These tests exercise the real resolution chain (config on disk → check_fn →
``get_tool_definitions``) against a temp ``HERMES_HOME``, not mocks.
"""

import pytest
import yaml

from model_tools import get_tool_definitions


@pytest.fixture(autouse=True)
def _clear_caches():
    """check_fn results and tool definitions are both cached; config written by
    a test only takes effect once those are dropped."""
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    yield
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


def _write_memory_config(home, **memory_section):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"memory": memory_section}), encoding="utf-8"
    )


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _memory_tool_names():
    tools = get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=True)
    return {tool["function"]["name"] for tool in tools}


class TestBuiltinMemoryToolAvailability:
    def test_tool_hidden_when_both_stores_disabled(self, hermes_home):
        _write_memory_config(
            hermes_home, memory_enabled=False, user_profile_enabled=False
        )
        assert "memory" not in _memory_tool_names()

    def test_tool_present_when_only_user_profile_enabled(self, hermes_home):
        _write_memory_config(
            hermes_home, memory_enabled=False, user_profile_enabled=True
        )
        assert "memory" in _memory_tool_names()

    def test_tool_present_when_only_memory_enabled(self, hermes_home):
        _write_memory_config(
            hermes_home, memory_enabled=True, user_profile_enabled=False
        )
        assert "memory" in _memory_tool_names()

    def test_tool_present_by_default(self, hermes_home):
        """No config file at all must not strip a working tool."""
        assert "memory" in _memory_tool_names()

    def test_unreadable_config_fails_open(self, hermes_home, monkeypatch):
        """A config read error must not silently remove the tool."""
        from tools import memory_tool as memory_tool_module

        def _boom():
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", _boom, raising=False
        )
        assert memory_tool_module.check_memory_requirements() is True


class TestExternalProviderSurvivesBuiltinDisable:
    """Dropping the built-in tool must not drop the external provider's tools.

    ``memory_provider_tools_enabled`` short-circuits on the built-in tool being
    present, so hiding that tool moves the decision onto the toolset gate. The
    provider must still be reachable for every way a caller can ask for memory.
    """

    def test_provider_tools_enabled_when_memory_toolset_requested(self):
        from agent.memory_manager import memory_provider_tools_enabled

        assert memory_provider_tools_enabled(
            ["memory", "file"], None, memory_tool_present=False
        )

    def test_provider_tools_enabled_for_unrestricted_toolsets(self):
        from agent.memory_manager import memory_provider_tools_enabled

        assert memory_provider_tools_enabled(None, None, memory_tool_present=False)

    def test_disabled_toolsets_still_takes_everything_down(self):
        """The heavy switch keeps its documented meaning."""
        from agent.memory_manager import memory_provider_tools_enabled

        assert not memory_provider_tools_enabled(
            None, ["memory"], memory_tool_present=False
        )


class TestInjectionEndToEnd:
    """The real ``inject_memory_provider_tools`` with no built-in memory tool."""

    def test_provider_tools_injected_without_builtin_memory_tool(self):
        from types import SimpleNamespace

        from agent.memory_manager import MemoryManager, inject_memory_provider_tools
        from agent.memory_provider import MemoryProvider

        class _Provider(MemoryProvider):
            @property
            def name(self):
                return "fake_hindsight"

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                pass

            def get_tool_schemas(self):
                return [
                    {
                        "name": "hindsight_retain",
                        "description": "retain",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]

        manager = MemoryManager()
        manager.add_provider(_Provider())
        agent = SimpleNamespace(
            _memory_manager=manager,
            enabled_toolsets=["memory"],
            disabled_toolsets=None,
            tools=[],
            valid_tool_names=set(),
        )

        added = inject_memory_provider_tools(agent)

        assert added == 1
        assert "hindsight_retain" in agent.valid_tool_names
