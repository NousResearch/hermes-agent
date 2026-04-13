"""Tests for the Hermes plugin system (hermes_cli.plugins)."""

import logging
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.plugins import (
    ENTRY_POINTS_GROUP,
    VALID_HOOKS,
    LoadedPlugin,
    PluginContext,
    PluginManager,
    PluginManifest,
    get_plugin_manager,
    get_plugin_tool_names,
    discover_plugins,
    invoke_hook,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_plugin_dir(base: Path, name: str, *, register_body: str = "pass",
                     manifest_extra: dict | None = None) -> Path:
    """Create a minimal plugin directory with plugin.yaml + __init__.py."""
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"name": name, "version": "0.1.0", "description": f"Test plugin {name}"}
    if manifest_extra:
        manifest.update(manifest_extra)

    (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest))
    (plugin_dir / "__init__.py").write_text(
        f"def register(ctx):\n    {register_body}\n"
    )
    return plugin_dir


# ── TestPluginDiscovery ────────────────────────────────────────────────────


class TestPluginDiscovery:
    """Tests for plugin discovery from directories and entry points."""

    def test_discover_user_plugins(self, tmp_path, monkeypatch):
        """Plugins in ~/.hermes/plugins/ are discovered."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "hello_plugin")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "hello_plugin" in mgr._plugins
        assert mgr._plugins["hello_plugin"].enabled

    def test_discover_project_plugins(self, tmp_path, monkeypatch):
        """Plugins in ./.hermes/plugins/ are discovered."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "true")
        plugins_dir = project_dir / ".hermes" / "plugins"
        _make_plugin_dir(plugins_dir, "proj_plugin")

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "proj_plugin" in mgr._plugins
        assert mgr._plugins["proj_plugin"].enabled

    def test_discover_project_plugins_skipped_by_default(self, tmp_path, monkeypatch):
        """Project plugins are not discovered unless explicitly enabled."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)
        plugins_dir = project_dir / ".hermes" / "plugins"
        _make_plugin_dir(plugins_dir, "proj_plugin")

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "proj_plugin" not in mgr._plugins

    def test_discover_is_idempotent(self, tmp_path, monkeypatch):
        """Calling discover_and_load() twice does not duplicate plugins."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "once_plugin")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()
        mgr.discover_and_load()  # second call should no-op

        assert len(mgr._plugins) == 1

    def test_discover_skips_dir_without_manifest(self, tmp_path, monkeypatch):
        """Directories without plugin.yaml are silently skipped."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        (plugins_dir / "no_manifest").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert len(mgr._plugins) == 0

    def test_entry_points_scanned(self, tmp_path, monkeypatch):
        """Entry-point based plugins are discovered (mocked)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        fake_module = types.ModuleType("fake_ep_plugin")
        fake_module.register = lambda ctx: None  # type: ignore[attr-defined]

        fake_ep = MagicMock()
        fake_ep.name = "ep_plugin"
        fake_ep.value = "fake_ep_plugin:register"
        fake_ep.group = ENTRY_POINTS_GROUP
        fake_ep.load.return_value = fake_module

        def fake_entry_points():
            result = MagicMock()
            result.select = MagicMock(return_value=[fake_ep])
            return result

        with patch("importlib.metadata.entry_points", fake_entry_points):
            mgr = PluginManager()
            mgr.discover_and_load()

        assert "ep_plugin" in mgr._plugins


# ── TestPluginLoading ──────────────────────────────────────────────────────


class TestPluginLoading:
    """Tests for plugin module loading."""

    def test_load_missing_init(self, tmp_path, monkeypatch):
        """Plugin dir without __init__.py records an error."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "bad_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "bad_plugin"}))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "bad_plugin" in mgr._plugins
        assert not mgr._plugins["bad_plugin"].enabled
        assert mgr._plugins["bad_plugin"].error is not None

    def test_load_missing_register_fn(self, tmp_path, monkeypatch):
        """Plugin without register() function records an error."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "no_reg"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "no_reg"}))
        (plugin_dir / "__init__.py").write_text("# no register function\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "no_reg" in mgr._plugins
        assert not mgr._plugins["no_reg"].enabled
        assert "no register()" in mgr._plugins["no_reg"].error

    def test_load_registers_namespace_module(self, tmp_path, monkeypatch):
        """Directory plugins are importable under hermes_plugins.<name>."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "ns_plugin")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        # Clean up any prior namespace module
        sys.modules.pop("hermes_plugins.ns_plugin", None)

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "hermes_plugins.ns_plugin" in sys.modules


# ── TestPluginHooks ────────────────────────────────────────────────────────


class TestPluginHooks:
    """Tests for lifecycle hook registration and invocation."""

    def test_valid_hooks_include_request_scoped_api_hooks(self):
        assert "pre_api_request" in VALID_HOOKS
        assert "post_api_request" in VALID_HOOKS

    def test_register_and_invoke_hook(self, tmp_path, monkeypatch):
        """Registered hooks are called on invoke_hook()."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "hook_plugin",
            register_body='ctx.register_hook("pre_tool_call", lambda **kw: None)',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        # Should not raise
        mgr.invoke_hook("pre_tool_call", tool_name="test", args={}, task_id="t1")

    def test_hook_exception_does_not_propagate(self, tmp_path, monkeypatch):
        """A hook callback that raises does NOT crash the caller."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "bad_hook",
            register_body='ctx.register_hook("post_tool_call", lambda **kw: 1/0)',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        # Should not raise despite 1/0
        mgr.invoke_hook("post_tool_call", tool_name="x", args={}, result="r", task_id="")

    def test_hook_return_values_collected(self, tmp_path, monkeypatch):
        """invoke_hook() collects non-None return values from callbacks."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "ctx_plugin",
            register_body=(
                'ctx.register_hook("pre_llm_call", '
                'lambda **kw: {"context": "memory from plugin"})'
            ),
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        results = mgr.invoke_hook("pre_llm_call", session_id="s1", user_message="hi",
                                  conversation_history=[], is_first_turn=True, model="test")
        assert len(results) == 1
        assert results[0] == {"context": "memory from plugin"}

    def test_hook_none_returns_excluded(self, tmp_path, monkeypatch):
        """invoke_hook() excludes None returns from the result list."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "none_hook",
            register_body='ctx.register_hook("post_llm_call", lambda **kw: None)',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        results = mgr.invoke_hook("post_llm_call", session_id="s1",
                                  user_message="hi", assistant_response="bye", model="test")
        assert results == []

    def test_request_hooks_are_invokeable(self, tmp_path, monkeypatch):
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "request_hook",
            register_body=(
                'ctx.register_hook("pre_api_request", '
                'lambda **kw: {"seen": kw.get("api_call_count"), '
                '"mc": kw.get("message_count"), "tc": kw.get("tool_count")})'
            ),
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        results = mgr.invoke_hook(
            "pre_api_request",
            session_id="s1",
            task_id="t1",
            model="test",
            api_call_count=2,
            message_count=5,
            tool_count=3,
            approx_input_tokens=100,
            request_char_count=400,
            max_tokens=8192,
        )
        assert results == [{"seen": 2, "mc": 5, "tc": 3}]

    def test_invalid_hook_name_warns(self, tmp_path, monkeypatch, caplog):
        """Registering an unknown hook name logs a warning."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "warn_plugin",
            register_body='ctx.register_hook("on_banana", lambda **kw: None)',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            mgr = PluginManager()
            mgr.discover_and_load()

        assert any("on_banana" in record.message for record in caplog.records)


# ── TestPluginContext ──────────────────────────────────────────────────────


class TestPluginContext:
    """Tests for the PluginContext facade."""

    def test_register_tool_adds_to_registry(self, tmp_path, monkeypatch):
        """PluginContext.register_tool() puts the tool in the global registry."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "tool_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "tool_plugin"}))
        (plugin_dir / "__init__.py").write_text(
            'def register(ctx):\n'
            '    ctx.register_tool(\n'
            '        name="plugin_echo",\n'
            '        toolset="plugin_tool_plugin",\n'
            '        schema={"name": "plugin_echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}},\n'
            '        handler=lambda args, **kw: "echo",\n'
            '    )\n'
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "plugin_echo" in mgr._plugin_tool_names

        from tools.registry import registry
        assert "plugin_echo" in registry._tools


# ── TestPluginToolVisibility ───────────────────────────────────────────────


class TestPluginToolVisibility:
    """Plugin-registered tools appear in get_tool_definitions()."""

    def test_plugin_tools_in_definitions(self, tmp_path, monkeypatch):
        """Plugin tools are included when their toolset is in enabled_toolsets."""
        import hermes_cli.plugins as plugins_mod

        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "vis_plugin"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "vis_plugin"}))
        (plugin_dir / "__init__.py").write_text(
            'def register(ctx):\n'
            '    ctx.register_tool(\n'
            '        name="vis_tool",\n'
            '        toolset="plugin_vis_plugin",\n'
            '        schema={"name": "vis_tool", "description": "Visible", "parameters": {"type": "object", "properties": {}}},\n'
            '        handler=lambda args, **kw: "ok",\n'
            '    )\n'
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()
        monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)

        from model_tools import get_tool_definitions

        # Plugin tools are included when their toolset is explicitly enabled
        tools = get_tool_definitions(enabled_toolsets=["terminal", "plugin_vis_plugin"], quiet_mode=True)
        tool_names = [t["function"]["name"] for t in tools]
        assert "vis_tool" in tool_names

        # Plugin tools are excluded when only other toolsets are enabled
        tools2 = get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
        tool_names2 = [t["function"]["name"] for t in tools2]
        assert "vis_tool" not in tool_names2

        # Plugin tools are included when no toolset filter is active (all enabled)
        tools3 = get_tool_definitions(quiet_mode=True)
        tool_names3 = [t["function"]["name"] for t in tools3]
        assert "vis_tool" in tool_names3


# ── TestPluginManagerList ──────────────────────────────────────────────────


class TestPluginManagerList:
    """Tests for PluginManager.list_plugins()."""

    def test_list_empty(self):
        """Empty manager returns empty list."""
        mgr = PluginManager()
        assert mgr.list_plugins() == []

    def test_list_returns_sorted(self, tmp_path, monkeypatch):
        """list_plugins() returns results sorted by name."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "zulu")
        _make_plugin_dir(plugins_dir, "alpha")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        listing = mgr.list_plugins()
        names = [p["name"] for p in listing]
        assert names == sorted(names)

    def test_list_with_plugins(self, tmp_path, monkeypatch):
        """list_plugins() returns info dicts for each discovered plugin."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(plugins_dir, "alpha")
        _make_plugin_dir(plugins_dir, "beta")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        listing = mgr.list_plugins()
        names = [p["name"] for p in listing]
        assert "alpha" in names
        assert "beta" in names
        for p in listing:
            assert "enabled" in p
            assert "tools" in p
            assert "hooks" in p



class TestPreLlmCallTargetRouting:
    """Tests for pre_llm_call hook return format with target-aware routing.

    The routing logic lives in run_agent.py, but the return format is collected
    by invoke_hook(). These tests verify the return format works correctly and
    that downstream code can route based on the 'target' key.
    """

    def _make_pre_llm_plugin(self, plugins_dir, name, return_expr):
        """Create a plugin that returns a specific value from pre_llm_call."""
        _make_plugin_dir(
            plugins_dir, name,
            register_body=(
                f'ctx.register_hook("pre_llm_call", lambda **kw: {return_expr})'
            ),
        )

    def test_context_dict_returned(self, tmp_path, monkeypatch):
        """Plugin returning a context dict is collected by invoke_hook."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        self._make_pre_llm_plugin(
            plugins_dir, "basic_plugin",
            '{"context": "basic context"}',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        results = mgr.invoke_hook(
            "pre_llm_call", session_id="s1", user_message="hi",
            conversation_history=[], is_first_turn=True, model="test",
        )
        assert len(results) == 1
        assert results[0]["context"] == "basic context"
        assert "target" not in results[0]

    def test_plain_string_return(self, tmp_path, monkeypatch):
        """Plain string returns are collected as-is (routing treats them as user_message)."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        self._make_pre_llm_plugin(
            plugins_dir, "str_plugin",
            '"plain string context"',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        results = mgr.invoke_hook(
            "pre_llm_call", session_id="s1", user_message="hi",
            conversation_history=[], is_first_turn=True, model="test",
        )
        assert len(results) == 1
        assert results[0] == "plain string context"

    def test_multiple_plugins_context_collected(self, tmp_path, monkeypatch):
        """Multiple plugins returning context are all collected."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        self._make_pre_llm_plugin(
            plugins_dir, "aaa_memory",
            '{"context": "memory context"}',
        )
        self._make_pre_llm_plugin(
            plugins_dir, "bbb_guardrail",
            '{"context": "guardrail text"}',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        results = mgr.invoke_hook(
            "pre_llm_call", session_id="s1", user_message="hi",
            conversation_history=[], is_first_turn=True, model="test",
        )
        assert len(results) == 2
        contexts = [r["context"] for r in results]
        assert "memory context" in contexts
        assert "guardrail text" in contexts

    def test_routing_logic_all_to_user_message(self, tmp_path, monkeypatch):
        """Simulate the routing logic from run_agent.py.

        All plugin context — dicts and plain strings — ends up in a single
        user message context string. There is no system_prompt target.
        """
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        self._make_pre_llm_plugin(
            plugins_dir, "aaa_mem",
            '{"context": "memory A"}',
        )
        self._make_pre_llm_plugin(
            plugins_dir, "bbb_guard",
            '{"context": "rule B"}',
        )
        self._make_pre_llm_plugin(
            plugins_dir, "ccc_plain",
            '"plain text C"',
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        results = mgr.invoke_hook(
            "pre_llm_call", session_id="s1", user_message="hi",
            conversation_history=[], is_first_turn=True, model="test",
        )

        # Replicate run_agent.py routing logic — everything goes to user msg
        _ctx_parts = []
        for r in results:
            if isinstance(r, dict) and r.get("context"):
                _ctx_parts.append(str(r["context"]))
            elif isinstance(r, str) and r.strip():
                _ctx_parts.append(r)

        assert _ctx_parts == ["memory A", "rule B", "plain text C"]
        _plugin_user_context = "\n\n".join(_ctx_parts)
        assert "memory A" in _plugin_user_context
        assert "rule B" in _plugin_user_context
        assert "plain text C" in _plugin_user_context


# NOTE: TestPluginCommands removed – register_command() was never implemented
# in PluginContext (hermes_cli/plugins.py).  The tests referenced _plugin_commands,
# commands_registered, get_plugin_command_handler, and GATEWAY_KNOWN_COMMANDS
# integration — all of which are unimplemented features.


class TestPluginManagerSkillRegistry:
    def test_register_plugin_skill_stores_entry(self, tmp_path):
        """_register_plugin_skill adds an entry keyed by qualified name."""
        from hermes_cli.plugins import PluginManager

        skill_path = tmp_path / "skill.md"
        skill_path.write_text("---\nname: foo\n---\n")

        pm = PluginManager()
        pm._register_plugin_skill(
            plugin_name="myplugin",
            skill_name="foo",
            path=skill_path,
            description="test desc",
        )

        assert "myplugin:foo" in pm._plugin_skills
        entry = pm._plugin_skills["myplugin:foo"]
        assert entry["path"] == skill_path
        assert entry["plugin"] == "myplugin"
        assert entry["bare_name"] == "foo"
        assert entry["description"] == "test desc"

    def test_find_plugin_skill_hit(self, tmp_path):
        from hermes_cli.plugins import PluginManager

        skill_path = tmp_path / "skill.md"
        skill_path.write_text("---\nname: foo\n---\n")

        pm = PluginManager()
        pm._register_plugin_skill("myplugin", "foo", skill_path, "")

        assert pm.find_plugin_skill("myplugin:foo") == skill_path

    def test_find_plugin_skill_miss_wrong_namespace(self, tmp_path):
        from hermes_cli.plugins import PluginManager

        skill_path = tmp_path / "skill.md"
        skill_path.write_text("---\nname: foo\n---\n")

        pm = PluginManager()
        pm._register_plugin_skill("myplugin", "foo", skill_path, "")

        assert pm.find_plugin_skill("other:foo") is None

    def test_find_plugin_skill_miss_wrong_skill(self, tmp_path):
        from hermes_cli.plugins import PluginManager

        skill_path = tmp_path / "skill.md"
        skill_path.write_text("---\nname: foo\n---\n")

        pm = PluginManager()
        pm._register_plugin_skill("myplugin", "foo", skill_path, "")

        assert pm.find_plugin_skill("myplugin:nonexistent") is None

    def test_list_plugin_skills_sorted(self):
        from hermes_cli.plugins import PluginManager
        from pathlib import Path

        pm = PluginManager()
        pm._register_plugin_skill("sp", "z-skill", Path("/a"), "")
        pm._register_plugin_skill("sp", "a-skill", Path("/b"), "")
        pm._register_plugin_skill("sp", "m-skill", Path("/c"), "")

        assert pm.list_plugin_skills("sp") == ["a-skill", "m-skill", "z-skill"]

    def test_list_plugin_skills_isolates_by_plugin(self):
        from hermes_cli.plugins import PluginManager
        from pathlib import Path

        pm = PluginManager()
        pm._register_plugin_skill("sp", "foo", Path("/a"), "")
        pm._register_plugin_skill("other", "bar", Path("/b"), "")

        assert pm.list_plugin_skills("sp") == ["foo"]
        assert pm.list_plugin_skills("other") == ["bar"]

    def test_list_plugin_skills_empty_for_unknown_plugin(self):
        from hermes_cli.plugins import PluginManager

        pm = PluginManager()
        assert pm.list_plugin_skills("nonexistent") == []

    def test_remove_plugin_skill(self, tmp_path):
        from hermes_cli.plugins import PluginManager

        skill_path = tmp_path / "skill.md"
        skill_path.write_text("---\nname: foo\n---\n")

        pm = PluginManager()
        pm._register_plugin_skill("myplugin", "foo", skill_path, "")
        assert pm.find_plugin_skill("myplugin:foo") is not None

        pm._remove_plugin_skill("myplugin:foo")
        assert pm.find_plugin_skill("myplugin:foo") is None

    def test_remove_plugin_skill_noop_on_missing(self):
        from hermes_cli.plugins import PluginManager

        pm = PluginManager()
        # Should not raise
        pm._remove_plugin_skill("nonexistent:skill")

    def test_register_plugin_skill_overwrites_on_duplicate(self, tmp_path):
        """Re-registering the same qualified name replaces the previous entry."""
        from hermes_cli.plugins import PluginManager
        old_path = tmp_path / "old.md"
        new_path = tmp_path / "new.md"
        old_path.write_text("")
        new_path.write_text("")

        pm = PluginManager()
        pm._register_plugin_skill("myplugin", "foo", old_path, "old desc")
        pm._register_plugin_skill("myplugin", "foo", new_path, "new desc")

        assert pm.find_plugin_skill("myplugin:foo") == new_path
        assert pm._plugin_skills["myplugin:foo"]["description"] == "new desc"

    def test_list_plugin_skills_no_prefix_collision(self):
        """Plugin 'foo' listing must not include skills from plugin 'foo-bar'."""
        from hermes_cli.plugins import PluginManager
        from pathlib import Path

        pm = PluginManager()
        pm._register_plugin_skill("foo", "alpha", Path("/a"), "")
        pm._register_plugin_skill("foo-bar", "beta", Path("/b"), "")

        assert pm.list_plugin_skills("foo") == ["alpha"]
        assert pm.list_plugin_skills("foo-bar") == ["beta"]


class TestPluginContextRegisterSkill:
    def _make_context(self, tmp_path, plugin_name="myplugin"):
        """Build a PluginContext backed by a fresh PluginManager."""
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

        manifest = PluginManifest(
            name=plugin_name,
            source="user",
            path=str(tmp_path),
        )
        manager = PluginManager()
        return PluginContext(manifest, manager), manager

    def test_register_skill_happy_path(self, tmp_path):
        ctx, manager = self._make_context(tmp_path)
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: foo\n---\n")

        ctx.register_skill(name="foo", path=skill_md, description="test")

        assert manager.find_plugin_skill("myplugin:foo") == skill_md
        entry = manager._plugin_skills["myplugin:foo"]
        assert entry["description"] == "test"

    def test_register_skill_rejects_colon_in_name(self, tmp_path):
        ctx, _ = self._make_context(tmp_path)
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: foo\n---\n")

        with pytest.raises(ValueError, match="reserved for namespace"):
            ctx.register_skill(name="bad:name", path=skill_md)

    def test_register_skill_rejects_dot_in_name(self, tmp_path):
        ctx, _ = self._make_context(tmp_path)
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: foo\n---\n")

        with pytest.raises(ValueError, match="Invalid skill name"):
            ctx.register_skill(name="bad.name", path=skill_md)

    def test_register_skill_rejects_empty_name(self, tmp_path):
        ctx, _ = self._make_context(tmp_path)
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: foo\n---\n")

        with pytest.raises(ValueError, match="Invalid skill name"):
            ctx.register_skill(name="", path=skill_md)

    def test_register_skill_rejects_missing_file(self, tmp_path):
        ctx, _ = self._make_context(tmp_path)

        with pytest.raises(FileNotFoundError):
            ctx.register_skill(name="foo", path=tmp_path / "does-not-exist.md")

    def test_register_skill_default_description_empty(self, tmp_path):
        ctx, manager = self._make_context(tmp_path)
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: foo\n---\n")

        ctx.register_skill(name="foo", path=skill_md)

        entry = manager._plugin_skills["myplugin:foo"]
        assert entry["description"] == ""


class TestAutoRegisterSkillsFromDirV1:
    def _fake_register(self, tmp_path, plugin_name="myplugin"):
        """Create a fake ctx + manager + skills dir with a few SKILL.md files."""
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()

        manifest = PluginManifest(name=plugin_name, source="user", path=str(plugin_dir))
        manager = PluginManager()
        ctx = PluginContext(manifest, manager)
        return ctx, manager, skills_dir

    def _make_skill(self, skills_dir, name):
        """Helper: create a skill directory with SKILL.md under skills_dir."""
        skill_dir = skills_dir / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: test\n---\n\nBody.\n"
        )

    def test_scans_all_valid_skills(self, tmp_path):
        from hermes_cli.plugins import _auto_register_skills_from_dir_v1

        ctx, manager, skills_dir = self._fake_register(tmp_path)
        self._make_skill(skills_dir, "a")
        self._make_skill(skills_dir, "b")
        self._make_skill(skills_dir, "c")

        count = _auto_register_skills_from_dir_v1(ctx, skills_dir)

        assert count == 3
        assert manager.find_plugin_skill("myplugin:a") is not None
        assert manager.find_plugin_skill("myplugin:b") is not None
        assert manager.find_plugin_skill("myplugin:c") is not None

    def test_skips_non_directory_files(self, tmp_path):
        from hermes_cli.plugins import _auto_register_skills_from_dir_v1

        ctx, manager, skills_dir = self._fake_register(tmp_path)
        self._make_skill(skills_dir, "real-skill")
        # Add garbage files that should be ignored
        (skills_dir / ".DS_Store").write_text("")
        (skills_dir / "README.md").write_text("# hi")

        count = _auto_register_skills_from_dir_v1(ctx, skills_dir)

        assert count == 1
        assert manager.find_plugin_skill("myplugin:real-skill") is not None

    def test_skips_subdir_without_skill_md(self, tmp_path):
        from hermes_cli.plugins import _auto_register_skills_from_dir_v1

        ctx, manager, skills_dir = self._fake_register(tmp_path)
        self._make_skill(skills_dir, "good")
        # Empty subdirectory
        (skills_dir / "empty").mkdir()

        count = _auto_register_skills_from_dir_v1(ctx, skills_dir)

        assert count == 1

    def test_one_broken_skill_doesnt_block_others(self, tmp_path, caplog):
        import logging
        from hermes_cli.plugins import _auto_register_skills_from_dir_v1

        ctx, manager, skills_dir = self._fake_register(tmp_path)
        self._make_skill(skills_dir, "good")
        # Skill with invalid name (dir name contains dot)
        bad_dir = skills_dir / "bad.name"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text("---\nname: bad.name\n---\n")

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            count = _auto_register_skills_from_dir_v1(ctx, skills_dir)

        # The good skill still registered
        assert count == 1
        assert manager.find_plugin_skill("myplugin:good") is not None

        # The bad skill's failure was logged with actionable context:
        # the failing skill name and the owning plugin name
        assert any("bad.name" in r.message for r in caplog.records), (
            "Expected 'bad.name' to appear in warning log records"
        )
        assert any("myplugin" in r.message for r in caplog.records), (
            "Expected plugin name 'myplugin' to appear in warning log records"
        )

    def test_empty_dir(self, tmp_path):
        from hermes_cli.plugins import _auto_register_skills_from_dir_v1

        ctx, _, skills_dir = self._fake_register(tmp_path)

        count = _auto_register_skills_from_dir_v1(ctx, skills_dir)

        assert count == 0

    def test_missing_dir(self, tmp_path):
        from hermes_cli.plugins import (
            PluginContext, PluginManager, PluginManifest,
            _auto_register_skills_from_dir_v1,
        )

        manifest = PluginManifest(name="p", source="user", path=str(tmp_path))
        manager = PluginManager()
        ctx = PluginContext(manifest, manager)

        count = _auto_register_skills_from_dir_v1(ctx, tmp_path / "does-not-exist")

        assert count == 0
