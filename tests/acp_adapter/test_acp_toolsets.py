"""Behavioral tests for ACP session toolset resolution."""

from __future__ import annotations

import sys
from types import ModuleType

from acp_adapter import session as session_mod


class NoopDb:
    pass


def _module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_make_agent_includes_platform_resolved_native_plugin_toolsets(monkeypatch):
    """ACP sessions must include enabled native plugins, not only MCP tools."""
    captured = {}
    config = {
        "model": {"default": "test-model"},
        "agent": {"disabled_toolsets": ["browser"]},
        "mcp_servers": {
            "notes": {"enabled": True},
            "disabled": {"enabled": "false"},
        },
    }

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs["model"]

    def resolve_platform_toolsets(
        resolved_config,
        platform,
        *,
        include_default_mcp_servers=True,
    ):
        assert resolved_config is config
        assert platform == "acp"
        assert include_default_mcp_servers is False
        # The canonical resolver can return an explicitly selected MCP server
        # name even with default MCP expansion disabled. It must not cross the
        # native-toolset namespace before ACP adds the canonical mcp- prefix.
        return {"hermes-acp", "fbrk", "notes"}

    monkeypatch.setitem(
        sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent)
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: config),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=resolve_platform_toolsets,
            _get_plugin_toolset_keys=lambda: {"fbrk"},
            enabled_mcp_server_names=lambda resolved_config: {"notes"},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        _module(
            "hermes_cli.mcp_startup",
            ensure_mcp_discovery_before_agent_build=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(session_mod, "_register_task_cwd", lambda *_args: None)

    manager = session_mod.SessionManager(db=NoopDb())
    manager._make_agent(session_id="acp-test", cwd=".")

    assert set(captured["enabled_toolsets"]) == {
        "hermes-acp",
        "fbrk",
        "mcp-notes",
    }
    assert captured["disabled_toolsets"] == ["browser"]


def test_base_acp_toolset_includes_profile_skill_tools():
    from toolsets import resolve_toolset

    assert {"skills_list", "skill_view", "skill_manage"}.issubset(
        resolve_toolset("hermes-acp")
    )


def test_platform_resolver_failure_preserves_base_acp_toolset(monkeypatch):
    """A broken optional plugin must not remove Hermes' base ACP tools or skills."""

    def fail_resolution(*_args, **_kwargs):
        raise RuntimeError("plugin discovery failed")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=fail_resolution,
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: set(),
        ),
    )

    resolved = session_mod._resolve_acp_platform_toolsets({})

    assert resolved == ["hermes-acp"]
    assert session_mod._expand_acp_enabled_toolsets(resolved) == ["hermes-acp"]


def test_platform_resolver_discovers_plugins_before_resolving_toolsets(monkeypatch):
    call_order = []

    def discover_plugins():
        call_order.append("discover")

    def resolve_toolsets(_config, platform, **_kwargs):
        call_order.append("resolve")
        assert platform == "acp"
        return {"hermes-acp", "fbrk"}

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=discover_plugins),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=resolve_toolsets,
            _get_plugin_toolset_keys=lambda: {"fbrk"},
            enabled_mcp_server_names=lambda _config: set(),
        ),
    )

    assert session_mod._resolve_acp_platform_toolsets({}) == ["hermes-acp", "fbrk"]
    assert call_order == ["discover", "resolve"]


def test_mcp_selection_honors_acp_allowlist_and_no_mcp(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: {"notes", "calendar"},
        ),
    )

    allowlisted = {
        "mcp_servers": {
            "notes": {"enabled": True},
            "calendar": {"enabled": True},
        },
        "platform_toolsets": {"acp": ["hermes-acp", "notes"]},
    }
    opted_out = {
        "mcp_servers": {
            "notes": {"enabled": True},
            "calendar": {"enabled": True},
        },
        "platform_toolsets": {"acp": ["hermes-acp", "no_mcp"]},
    }

    assert session_mod._enabled_acp_mcp_server_names(allowlisted) == ["notes"]
    assert session_mod._enabled_acp_mcp_server_names(opted_out) == []


def test_mcp_policy_is_stable_after_raw_name_alias_registration(monkeypatch):
    """A live MCP alias must not be mistaken for a native toolset collision."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: {"notes"},
        ),
    )
    monkeypatch.setattr(
        "toolsets._get_registry_toolset_aliases",
        lambda: {"notes": "mcp-notes"},
    )
    config = {
        "mcp_servers": {"notes": {"enabled": True}},
        "platform_toolsets": {"acp": ["hermes-acp", "notes"]},
    }

    assert session_mod._resolve_acp_mcp_policy(config) == (
        ["notes"],
        frozenset({"notes"}),
    )


def test_disabled_toolset_name_denies_canonical_mcp_toolset(monkeypatch):
    """Global disabled_toolsets must survive ACP's mcp- namespace canonicalization."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: {"notes"},
        ),
    )
    config = {
        "agent": {"disabled_toolsets": ["notes"]},
        "mcp_servers": {"notes": {"enabled": True}},
    }

    assert session_mod._resolve_acp_mcp_policy(config) == ([], None)
    assert session_mod._disabled_acp_mcp_server_names(config) == frozenset(
        {"notes"}
    )


def test_disabled_mcp_allowlist_entry_does_not_enable_other_servers(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: {"calendar"},
        ),
    )
    config = {
        "mcp_servers": {
            "notes": {"enabled": False},
            "calendar": {"enabled": True},
        },
        "platform_toolsets": {"acp": ["hermes-acp", "notes"]},
    }

    assert session_mod._enabled_acp_mcp_server_names(config) == []


def test_native_plugin_survives_mcp_name_collision(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_args, **_kwargs: {"hermes-acp", "notes"},
            _get_plugin_toolset_keys=lambda: {"notes"},
            enabled_mcp_server_names=lambda _config: set(),
        ),
    )

    assert session_mod._resolve_acp_platform_toolsets(
        {}, mcp_server_names=["notes"]
    ) == ["hermes-acp", "notes"]
    assert session_mod._expand_acp_enabled_toolsets(
        ["hermes-acp", "notes"], mcp_server_names=["notes"]
    ) == ["hermes-acp", "notes", "mcp-notes"]


def test_explicit_known_plugin_survives_enabled_mcp_collision(monkeypatch):
    """A selected plugin keeps its native tools while its colliding MCP is denied."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda config, *_args, **_kwargs: {
                "hermes-acp",
                *(
                    {"fbrk"}
                    if "fbrk" in config["platform_toolsets"]["acp"]
                    else set()
                ),
            },
            _get_plugin_toolset_keys=lambda: {"fbrk"},
            enabled_mcp_server_names=lambda _config: {"fbrk"},
        ),
    )
    config = {
        "mcp_servers": {"fbrk": {"enabled": True}},
        "platform_toolsets": {"acp": ["hermes-acp", "fbrk"]},
        "known_plugin_toolsets": {"acp": ["fbrk"]},
    }

    mcp_names = session_mod._enabled_acp_mcp_server_names(config)
    native = session_mod._resolve_acp_platform_toolsets(
        config, mcp_server_names=mcp_names
    )

    assert mcp_names == []
    assert native == ["hermes-acp", "fbrk"]


def test_builtin_toolset_survives_mcp_name_collision(monkeypatch):
    """An MCP server named after a native toolset must not suppress native tools."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_args, **_kwargs: {"hermes-acp", "memory"},
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: set(),
        ),
    )

    resolved = session_mod._resolve_acp_platform_toolsets(
        {}, mcp_server_names=["memory"]
    )

    assert resolved == ["hermes-acp", "memory"]
    assert session_mod._expand_acp_enabled_toolsets(
        resolved, mcp_server_names=["memory"]
    ) == ["hermes-acp", "memory", "mcp-memory"]


def test_current_policy_reserves_native_names_from_acp_client_mcp_registration():
    policy = session_mod._current_acp_capability_policy({})

    assert {"all", "*", "memory", "coding"}.issubset(
        set(policy["mcp_denied_names"])
    )


def test_current_policy_reserves_static_native_names_when_plugin_discovery_fails(
    monkeypatch,
):
    def fail_discovery():
        raise RuntimeError("broken plugin")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=fail_discovery),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_args, **_kwargs: {"hermes-acp"},
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: set(),
        ),
    )

    policy = session_mod._current_acp_capability_policy({})

    assert {"all", "*", "memory", "coding"}.issubset(
        set(policy["mcp_denied_names"])
    )


def test_explicit_builtin_mcp_collision_fails_closed(monkeypatch):
    """An ambiguous explicit name must grant neither native nor MCP capability."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    def resolve_native(resolved_config, *_args, **_kwargs):
        selected = set(resolved_config["platform_toolsets"]["acp"])
        return {"hermes-acp", *(selected & {"cronjob"})}

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=resolve_native,
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: {"cronjob"},
        ),
    )
    config = {
        "mcp_servers": {"cronjob": {"enabled": True}},
        "platform_toolsets": {"acp": ["hermes-acp", "cronjob"]},
    }

    mcp_names = session_mod._enabled_acp_mcp_server_names(config)
    native = session_mod._resolve_acp_platform_toolsets(
        config, mcp_server_names=mcp_names
    )

    assert mcp_names == []
    assert native == ["hermes-acp"]
    assert session_mod._expand_acp_enabled_toolsets(
        native, mcp_server_names=mcp_names
    ) == ["hermes-acp"]


def test_explicit_builtin_mcp_collision_preserves_baseline_native_tools(monkeypatch):
    """A colliding MCP name cannot revoke a separately granted ACP baseline."""
    captured = {}
    config = {
        "model": {"default": "test-model"},
        "mcp_servers": {"memory": {"enabled": True}},
        "platform_toolsets": {"acp": ["hermes-acp", "memory"]},
    }

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs["model"]
            self.tools = [
                {"type": "function", "function": {"name": "memory"}},
                {"type": "function", "function": {"name": "read_file"}},
            ]
            self.valid_tool_names = {"memory", "read_file"}

    monkeypatch.setitem(
        sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent)
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: config),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_args, **_kwargs: {
                "hermes-acp",
                "memory",
            },
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: {"memory"},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        _module(
            "hermes_cli.mcp_startup",
            ensure_mcp_discovery_before_agent_build=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(session_mod, "_register_task_cwd", lambda *_args: None)

    manager = session_mod.SessionManager(db=NoopDb())
    agent = manager._make_agent(session_id="acp-test", cwd=".")

    assert captured["enabled_toolsets"] == ["hermes-acp"]
    assert set(captured["disabled_toolsets"]) == {"mcp-memory"}
    assert agent.valid_tool_names == {"memory", "read_file"}
    assert [tool["function"]["name"] for tool in agent.tools] == [
        "memory",
        "read_file",
    ]
    assert agent._acp_mcp_allowed_names == frozenset()


def test_explicit_bundle_mcp_collision_preserves_independent_baseline(monkeypatch):
    config = {
        "model": {"default": "test-model"},
        "mcp_servers": {"hermes-acp": {"enabled": True}},
        "platform_toolsets": {"acp": ["hermes-acp"]},
    }

    class FakeAgent:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.tools = [
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "function", "function": {"name": "terminal"}},
                {"type": "function", "function": {"name": "memory_search"}},
            ]
            self.valid_tool_names = {"read_file", "terminal", "memory_search"}

    monkeypatch.setitem(
        sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent)
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: config),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_args, **_kwargs: {"hermes-acp"},
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: {"hermes-acp"},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.mcp_startup",
        _module(
            "hermes_cli.mcp_startup",
            ensure_mcp_discovery_before_agent_build=lambda **_kwargs: None,
        ),
    )
    monkeypatch.setattr(session_mod, "_register_task_cwd", lambda *_args: None)

    manager = session_mod.SessionManager(db=NoopDb())
    agent = manager._make_agent(session_id="bundle-collision", cwd=".")

    assert [tool["function"]["name"] for tool in agent.tools] == [
        "read_file",
        "terminal",
        "memory_search",
    ]
    assert agent.valid_tool_names == {"read_file", "terminal", "memory_search"}
    assert agent._acp_mcp_allowed_names == frozenset()


def test_composite_mcp_collision_is_removed_before_native_resolution(monkeypatch):
    """An MCP named ``all`` must not expand into every native capability."""
    resolved_platform_toolsets = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )

    def resolve_native(resolved_config, *_args, **_kwargs):
        resolved_platform_toolsets.extend(resolved_config["platform_toolsets"]["acp"])
        return {"hermes-acp"}

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=resolve_native,
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: {"all"},
        ),
    )
    config = {
        "mcp_servers": {"all": {"enabled": True}},
        "platform_toolsets": {"acp": ["hermes-acp", "all"]},
    }

    mcp_names = session_mod._enabled_acp_mcp_server_names(config)

    assert mcp_names == []
    assert session_mod._resolve_acp_platform_toolsets(
        config, mcp_server_names=mcp_names
    ) == ["hermes-acp"]
    assert resolved_platform_toolsets == ["hermes-acp"]


def test_disabled_mcp_collision_does_not_suppress_native_toolset(monkeypatch):
    """A disabled MCP entry must not affect an independently selected native toolset."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_args, **_kwargs: {"hermes-acp", "memory"},
            _get_plugin_toolset_keys=lambda: set(),
            enabled_mcp_server_names=lambda _config: set(),
        ),
    )
    config = {
        "mcp_servers": {"memory": {"enabled": False}},
        "platform_toolsets": {"acp": ["hermes-acp", "memory"]},
    }

    assert session_mod._enabled_acp_mcp_server_names(config) == []
    assert session_mod._resolve_acp_platform_toolsets(
        config, mcp_server_names=[]
    ) == ["hermes-acp", "memory"]


def test_native_plugin_collision_does_not_widen_mcp_allowlist(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        _module("hermes_cli.plugins", discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_plugin_toolset_keys=lambda: {"notes"},
            enabled_mcp_server_names=lambda _config: {"notes", "calendar"},
        ),
    )
    config = {
        "mcp_servers": {
            "notes": {"enabled": True},
            "calendar": {"enabled": True},
        },
        "platform_toolsets": {
            "acp": ["hermes-acp", "notes", "calendar"],
        },
    }

    assert session_mod._enabled_acp_mcp_server_names(config) == ["calendar"]
