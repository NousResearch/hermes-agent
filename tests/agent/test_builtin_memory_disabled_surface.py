"""Employee memory stays available; native per-store write guards still apply."""

import json

import pytest
import hermes_yaml as yaml

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


def _memory_tool_definition():
    return next(
        tool["function"]
        for tool in get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=True)
        if tool["function"]["name"] == "memory"
    )


def _memory_tool_names():
    tools = get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=True)
    return {tool["function"]["name"] for tool in tools}


class TestBuiltinMemoryToolAvailability:
    @pytest.mark.parametrize("flags", [(False, False), (False, True), (True, False), ("false", "false")])
    def test_employee_memory_cannot_be_disabled_by_legacy_flags(self, hermes_home, flags):
        _write_memory_config(hermes_home, memory_enabled=flags[0], user_profile_enabled=flags[1])
        definition = _memory_tool_definition()
        assert definition["parameters"]["properties"]["target"]["enum"] == ["memory", "user"]

    def test_tool_present_by_default(self, hermes_home):
        """No config file at all must not strip a working tool."""
        assert "memory" in _memory_tool_names()

    def test_missing_memory_flags_fail_open_consistently(self):
        from tools.memory_tool import get_builtin_memory_store_flags

        assert get_builtin_memory_store_flags({"memory": {}}) == (True, True)
        assert get_builtin_memory_store_flags({}) == (True, True)

    def test_schema_reuses_availability_flag_snapshot(self, monkeypatch):
        """One definition pass must not reread config between check and schema."""
        from tools import memory_tool as memory_tool_module

        calls = 0

        def _flags():
            nonlocal calls
            calls += 1
            return (False, True) if calls == 1 else (True, False)

        monkeypatch.setattr(memory_tool_module, "get_builtin_memory_store_flags", _flags)
        definition = _memory_tool_definition()

        assert calls == 1
        assert definition["parameters"]["properties"]["target"]["enum"] == ["user"]

    def test_unavailable_snapshot_cannot_survive_failed_recheck(self, monkeypatch):
        from tools import memory_tool as memory_tool_module

        monkeypatch.setattr(
            memory_tool_module,
            "get_builtin_memory_store_flags",
            lambda: (False, False),
        )
        assert memory_tool_module.check_memory_requirements() is False

        def _boom():
            raise RuntimeError("config read failed")

        monkeypatch.setattr(memory_tool_module, "get_builtin_memory_store_flags", _boom)
        with pytest.raises(RuntimeError, match="config read failed"):
            memory_tool_module.check_memory_requirements()

        assert memory_tool_module._memory_surface_flags.get() is None

    def test_legacy_config_flip_keeps_employee_memory_available(self, hermes_home):
        _write_memory_config(
            hermes_home, memory_enabled=False, user_profile_enabled=False
        )
        assert "memory" in _memory_tool_names()

        _write_memory_config(
            hermes_home, memory_enabled=True, user_profile_enabled=False
        )
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

    def test_malformed_memory_section_normalizes_to_defaults(self):
        from tools.memory_tool import (
            get_builtin_memory_config,
            get_builtin_memory_store_flags,
        )

        config = {"memory": "not-a-mapping"}
        assert get_builtin_memory_config(config) == {}
        assert get_builtin_memory_store_flags(config) == (True, True)


class TestIndependentStoreWriteGates:
    def _store(self, *, memory_enabled, user_profile_enabled):
        from tools.memory_tool import MemoryStore

        store = MemoryStore(
            memory_char_limit=500,
            user_char_limit=500,
            memory_enabled=memory_enabled,
            user_profile_enabled=user_profile_enabled,
        )
        store.load_from_disk()
        return store

    def test_user_profile_only_rejects_memory_write(self, hermes_home):
        from tools.memory_tool import memory_tool

        store = self._store(memory_enabled=False, user_profile_enabled=True)
        denied = json.loads(memory_tool(action="add", target="memory", content="fact", store=store))
        allowed = json.loads(memory_tool(action="add", target="user", content="pref", store=store))

        assert denied["success"] is False
        assert denied["target"] == "memory"
        assert allowed["success"] is True

    def test_memory_only_rejects_user_profile_write(self, hermes_home):
        from tools.memory_tool import memory_tool

        store = self._store(memory_enabled=True, user_profile_enabled=False)
        denied = json.loads(memory_tool(action="add", target="user", content="pref", store=store))
        allowed = json.loads(memory_tool(action="add", target="memory", content="fact", store=store))

        assert denied["success"] is False
        assert denied["target"] == "user"
        assert allowed["success"] is True

    def test_staged_write_cannot_bypass_target_gate(self, hermes_home):
        from tools.memory_tool import apply_memory_pending

        store = self._store(memory_enabled=False, user_profile_enabled=True)
        result = apply_memory_pending(
            {"action": "add", "target": "memory", "content": "fact"}, store
        )

        assert result["success"] is False
        assert result["target"] == "memory"

    def test_invalid_target_error_is_bounded_and_carries_recovery_hint(self, hermes_home):
        """Model-supplied target is interpolated into the error: it must stay
        capped at the shared tool_error bound and keep the recovery hint."""
        from tools.memory_tool import memory_tool
        from tools.registry import _MAX_TOOL_ERROR_CHARS

        store = self._store(memory_enabled=True, user_profile_enabled=True)
        result = json.loads(
            memory_tool(action="add", target="x" * 10_000, content="fact", store=store)
        )

        assert result["success"] is False
        assert len(result["error"]) <= _MAX_TOOL_ERROR_CHARS + 32

        short = json.loads(
            memory_tool(action="add", target="bogus", content="fact", store=store)
        )
        assert short["success"] is False
        assert short["error"]


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
