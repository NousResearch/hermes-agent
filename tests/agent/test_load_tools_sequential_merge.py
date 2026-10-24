"""Regressions for current-main tool loading and sequential delegation policy.

Exercise the installed native loader, schema selection/cache and sequential
policy. Discovery, registry availability and dispatch effects are doubled.
The original test_delegate_sequential.py remains a separate required join test.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields
import json
import os
import socket
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _write_profile(home, sequential):
    home.mkdir(parents=True, exist_ok=True)
    lines = [
        "agent:",
        "  disabled_toolsets: []",
        "delegation:",
        "  max_concurrent_children: 3",
        "  max_spawn_depth: 1",
        "  independent_completions: true",
    ]
    if sequential is not None:
        lines.append("  sequential: " + str(sequential).lower())
    lines += [
        "tool_search:",
        '  enabled: "off"',
        "tools:",
        "  cli:",
        "    disabled: []",
    ]
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def loader_case(tmp_path, monkeypatch):
    # Prepare the private home before any Hermes imports in this module.
    bootstrap = tmp_path / "bootstrap"
    _write_profile(bootstrap, False)
    monkeypatch.setenv("HERMES_HOME", str(bootstrap))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    def refuse_effect(*args, **kwargs):
        # PermissionError lets native optional version probes use their own
        # OSError fallback; no child process or network operation is performed.
        raise PermissionError("Loader regression fixture refuses external effects")

    monkeypatch.setattr(subprocess, "Popen", refuse_effect)
    monkeypatch.setattr(os, "system", refuse_effect)
    monkeypatch.setattr(os, "popen", refuse_effect)
    monkeypatch.setattr(threading.Thread, "start", refuse_effect)
    monkeypatch.setattr(socket, "create_connection", refuse_effect)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_effect)
    for name in ("connect", "connect_ex", "bind", "listen"):
        monkeypatch.setattr(socket.socket, name, refuse_effect)

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    home_token = set_hermes_home_override(bootstrap)
    secret_token = set_secret_scope({}, profile_home=str(bootstrap))
    set_multiplex_active(True)
    try:
        import hermes_cli.left_core_migration as migration
        from hermes_cli import plugins
        import tools.registry as registry_module

        # Install these before importing model_tools, whose module body performs
        # builtin/plugin discovery. Do not load plugins or probe tool backends.
        recovery = Mock()
        discovery = Mock()
        builtin_discovery = Mock()
        monkeypatch.setattr(migration, "recover_at_startup", recovery)
        monkeypatch.setattr(plugins, "discover_plugins", discovery)
        monkeypatch.setattr(
            registry_module, "discover_builtin_tools", builtin_discovery
        )

        from hermes_cli import config
        import model_tools
        import toolsets
        from agent import agent_init
        from agent import delegation_context
        import agent.oneshot_footprint as oneshot
        import tools.delegate_tool as delegate
        import tools.delegate_tool_config as policy
        import tools.delegate_tool_dispatch as dispatch
        from tools import tool_search

        registry = registry_module.registry
        monkeypatch.setattr(config, "_LOAD_CONFIG_CACHE", {})
        monkeypatch.setattr(model_tools, "_tool_defs_cache", {})
        monkeypatch.setattr(model_tools, "_last_resolved_tool_names", [])
        monkeypatch.setattr(
            delegation_context, "owned_kanban_task", Mock(return_value=None)
        )
        monkeypatch.setattr(
            tool_search, "load_config",
            Mock(return_value=SimpleNamespace(enabled="off")),
        )

        case = SimpleNamespace(
            active_agent=None,
            extra_disabled=[],
            resolutions=[],
            selected=[],
            bootstrap=bootstrap,
            config=config,
            model_tools=model_tools,
            policy=policy,
            dispatch=dispatch,
            recovery=recovery,
            discovery=discovery,
        )
        native_resolver = toolsets.session_disabled_toolsets

        def resolve(disabled_toolsets, platform):
            result = native_resolver(disabled_toolsets, platform)
            # One parametrized case adds a resolver-level exclusion after the
            # native helper. This makes raw versus resolved arguments differ,
            # without assuming uncaptured platform-config parsing internals.
            if case.extra_disabled:
                result = list(dict.fromkeys([*(result or []), *case.extra_disabled]))
            case.resolutions.append(result)
            return result

        case.resolver = Mock(side_effect=resolve)
        monkeypatch.setattr(toolsets, "session_disabled_toolsets", case.resolver)
        # Default: call the real upstream helper. Filter cases override its
        # return value to force a particular agent-drop decision at this seam.
        case.agent_drops = Mock(wraps=toolsets.agent_tool_drops)
        monkeypatch.setattr(toolsets, "agent_tool_drops", case.agent_drops)
        case.prune = Mock(wraps=oneshot.prune_oneshot_tools)
        monkeypatch.setattr(oneshot, "prune_oneshot_tools", case.prune)

        def definitions(names, *, quiet=False):
            agent = case.active_agent
            assert agent is not None
            # This runs at actual schema creation, after native toolset
            # selection, so storing exclusions after the call cannot pass.
            assert agent.disabled_toolsets is case.resolutions[-1]
            assert (
                policy._SEQUENTIAL_SCHEMA_POLICY.get()
                is agent._delegation_sequential
            )
            assert agent._tool_snapshot_generation == registry._generation
            case.selected.append(set(names))
            out = []
            if "delegate_task" in names:
                schema = deepcopy(delegate.DELEGATE_TASK_SCHEMA)
                # Use the real PR schema builder, not a hard-coded sequential
                # description or a replacement for get_tool_definitions.
                schema.update(delegate._build_dynamic_schema_overrides())
                out.append({"type": "function", "function": schema})
            if "terminal" in names:
                out.append({
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "description": "Fixture terminal; never executed.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                })
            return out

        case.build_definitions = definitions
        case.definitions = Mock(side_effect=definitions)
        monkeypatch.setattr(registry, "get_definitions", case.definitions)
        case.background = Mock(return_value='{"status":"dispatched"}')
        case.aggregate = Mock(return_value={"results": []})
        monkeypatch.setattr(dispatch, "begin_delegation_run", Mock())
        monkeypatch.setattr(dispatch, "_dispatch_background", case.background)
        monkeypatch.setattr(dispatch, "_execute_and_aggregate", case.aggregate)

        @contextmanager
        def profile(home):
            with monkeypatch.context() as env:
                env.setenv("HERMES_HOME", str(home))
                next_home = set_hermes_home_override(home)
                next_secret = set_secret_scope({}, profile_home=str(home))
                try:
                    yield
                finally:
                    reset_secret_scope(next_secret)
                    reset_hermes_home_override(next_home)

        def write(home, sequential):
            _write_profile(home, sequential)
            config._LOAD_CONFIG_CACHE.clear()
            # Deliberately keep the native schema cache across profile edits
            # and old/fresh sessions, including opposite policies at one stat.

        def load(agent=None, *, enabled=None, disabled=None):
            if agent is None:
                agent = SimpleNamespace(
                    quiet_mode=True, platform="cli", _delegate_depth=0
                )
            if enabled is None:
                enabled = ["delegation", "terminal"]
            case.active_agent = agent
            try:
                # No fabricated loader, AIAgent construction, or dispatch here.
                agent_init._load_tools(agent, enabled, disabled)
            finally:
                case.active_agent = None
            return agent

        case.profile = profile
        case.write = write
        case.load = load
        yield case
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        # Match the existing profile regression's isolated-process lifecycle.
        set_multiplex_active(False)


def _assert_schema_mode(agent, enabled):
    definition = next(
        tool["function"] for tool in agent.tools
        if tool["function"]["name"] == "delegate_task"
    )
    assert agent._delegation_sequential is enabled
    assert ("sequential mode is enabled" in definition["description"].lower()) == enabled
    tasks = definition["parameters"]["properties"]["tasks"]
    assert ("sequentially" in tasks["description"].lower()) == enabled
    # independent_completions is true in these private profiles; sequential
    # mode removes groups, ordinary mode retains its existing group surface.
    assert ("group" in tasks["items"]["properties"]) == (not enabled)


def _assert_dispatch_mode(case, agent, enabled):
    batch = case.dispatch._Batch(
        **{field.name: None for field in fields(case.dispatch._Batch)}
    )
    batch.task_list = []
    batch.children = []
    batch.parent_agent = agent
    batch.max_children = 3
    case.background.reset_mock()
    case.aggregate.reset_mock()
    result = json.loads(case.dispatch._run_batch(batch, background=True))
    if enabled:
        assert result == {"results": []}
        case.aggregate.assert_called_once()
        case.background.assert_not_called()
        assert case.aggregate.call_args.args[0].max_children == 1
        assert case.aggregate.call_args.args[0].parent_agent is agent
    else:
        assert result == {"status": "dispatched"}
        case.background.assert_called_once_with(batch)
        case.aggregate.assert_not_called()
    assert batch.max_children == 3


@pytest.mark.parametrize("disabled_source", ["explicit", "resolver-added"])
def test_sequential_mode_keeps_session_disabled_delegation_absent(
    loader_case, disabled_source
):
    case = loader_case
    case.write(case.bootstrap, True)
    excluded = ["delegation"] if disabled_source == "explicit" else []
    if disabled_source == "resolver-added":
        case.extra_disabled = ["delegation"]
    parent = case.load(disabled=excluded)
    assert parent._delegation_sequential is True
    assert "delegation" in parent.disabled_toolsets
    assert parent.valid_tool_names == {"terminal"}
    assert "delegate_task" not in case.selected[-1]
    assert case.resolver.call_args.args == (excluded, "cli")
    case.agent_drops.assert_called_once_with(parent)
    assert case.policy._SEQUENTIAL_SCHEMA_POLICY.get() is None
    # A later reload still honors the session's exclusions and frozen policy.
    case.write(case.bootstrap, False)
    case.load(parent, disabled=parent.disabled_toolsets)
    assert parent._delegation_sequential is True
    assert parent.valid_tool_names == {"terminal"}
    assert case.policy._get_sequential() is False


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize(
    ("enabled", "disabled", "drops", "expected", "selected"),
    [
        (["delegation", "terminal"], ["terminal"], set(),
         {"delegate_task"}, {"delegate_task"}),
        (["delegation", "terminal"], [], {"delegate_task"},
         {"terminal"}, {"delegate_task", "terminal"}),
        (["delegation", "terminal"], ["delegation"], {"terminal"},
         set(), {"terminal"}),
        (["terminal"], [], set(), {"terminal"}, {"terminal"}),
    ],
)
def test_toolset_selection_and_agent_drops_still_apply(
    loader_case, sequential, enabled, disabled, drops, expected, selected
):
    case = loader_case
    case.write(case.bootstrap, sequential)
    case.agent_drops.return_value = drops  # Explicit upstream-decision double.
    parent = case.load(enabled=enabled, disabled=disabled)
    assert parent.valid_tool_names == expected
    assert {
        tool["function"]["name"] for tool in parent.tools
    } == expected
    assert case.selected[-1].intersection({"delegate_task", "terminal"}) == selected
    case.agent_drops.assert_called_once_with(parent)
    assert parent._delegation_sequential is sequential
    if "delegate_task" in expected:
        _assert_schema_mode(parent, sequential)

    # Dropping tools on one agent must not prune the cached schema list for
    # another session with the same toolsets and profile.
    case.agent_drops.return_value = set()
    fresh = case.load(enabled=enabled, disabled=disabled)
    assert fresh.valid_tool_names == selected
    assert parent.valid_tool_names == expected


def test_oneshot_pruning_precedes_agent_specific_drops(loader_case):
    case = loader_case
    case.write(case.bootstrap, True)
    seen = []

    def prune(tools):
        seen.append(("prune", {tool["function"]["name"] for tool in tools}))
        return [
            tool for tool in tools
            if tool["function"]["name"] != "terminal"
        ]

    def drops(agent):
        seen.append(("drops", {tool["function"]["name"] for tool in agent.tools}))
        return {"delegate_task"}

    case.prune.side_effect = prune  # Force pruning without entering one-shot mode.
    case.agent_drops.side_effect = drops
    parent = case.load()
    assert seen == [
        ("prune", {"delegate_task", "terminal"}),
        ("drops", {"delegate_task"}),
    ]
    assert parent.tools == []
    assert parent.valid_tool_names == set()
    case.recovery.assert_called_once_with(say=None)
    assert case.discovery.call_count >= 1


@pytest.mark.parametrize("b_policy", [False, None], ids=["explicit-off", "default-off"])
def test_profile_a_b_a_freezes_schema_and_isolates_fresh_sessions(
    loader_case, tmp_path, b_policy
):
    case = loader_case
    homes = (tmp_path / "a", tmp_path / "b")
    case.write(homes[0], True)
    case.write(homes[1], b_policy)
    saved = []
    for home, configured in (
        (homes[0], True), (homes[1], b_policy), (homes[0], True)
    ):
        expected = configured is True
        with case.profile(home):
            assert case.policy._SEQUENTIAL_SCHEMA_POLICY.get() is None
            assert case.policy._get_sequential() is expected
            parent = case.load()
            _assert_schema_mode(parent, expected)
            original = json.dumps(parent.tools, sort_keys=True)
            saved.append((parent.tools, original))

            # Same home, changed config; do not clear the schema cache.
            case.write(home, not expected)
            assert case.policy._get_sequential() is (not expected)
            case.load(parent)
            assert json.dumps(parent.tools, sort_keys=True) == original
            _assert_schema_mode(parent, expected)
            _assert_dispatch_mode(case, parent, expected)

            fresh = case.load()
            _assert_schema_mode(fresh, not expected)
            _assert_dispatch_mode(case, fresh, not expected)
            saved.append((fresh.tools, json.dumps(fresh.tools, sort_keys=True)))

            # Rebuild the old session after the opposite fresh one. Cache
            # entries and nested dynamic descriptions must remain isolated.
            case.load(parent)
            assert json.dumps(parent.tools, sort_keys=True) == original
            assert case.policy._SEQUENTIAL_SCHEMA_POLICY.get() is None
            for tools, serialized in saved:
                assert json.dumps(tools, sort_keys=True) == serialized
            case.write(home, configured)


def test_schema_failure_resets_context_without_losing_exclusions(loader_case, tmp_path):
    case = loader_case
    case.write(case.bootstrap, True)
    parent = SimpleNamespace(quiet_mode=True, platform="cli", _delegate_depth=0)
    case.definitions.side_effect = RuntimeError("intentional schema failure")
    with pytest.raises(RuntimeError, match="intentional schema failure"):
        case.load(parent, disabled=["delegation"])
    assert "delegation" in parent.disabled_toolsets
    assert parent._delegation_sequential is True
    assert case.policy._SEQUENTIAL_SCHEMA_POLICY.get() is None
    case.prune.assert_not_called()
    case.agent_drops.assert_not_called()

    case.definitions.side_effect = case.build_definitions
    other = tmp_path / "after-error-default"
    case.write(other, None)
    with case.profile(other):
        fresh = case.load()
        _assert_schema_mode(fresh, False)
        _assert_dispatch_mode(case, fresh, False)
        assert case.policy._SEQUENTIAL_SCHEMA_POLICY.get() is None
