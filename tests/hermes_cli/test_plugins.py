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
    get_plugin_command_handler,
    get_plugin_commands,
    get_pre_tool_call_block_message,
    discover_plugins,
    invoke_hook,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_plugin_dir(base: Path, name: str, *, register_body: str = "pass",
                     manifest_extra: dict | None = None,
                     auto_enable: bool = True) -> Path:
    """Create a minimal plugin directory with plugin.yaml + __init__.py.

    If *auto_enable* is True (default), also write the plugin's name into
    ``<hermes_home>/config.yaml`` under ``plugins.enabled``. Plugins are
    opt-in by default, so tests that expect the plugin to actually load
    need this. Pass ``auto_enable=False`` for tests that exercise the
    unenabled path.

    *base* is expected to be ``<hermes_home>/plugins/``; we derive
    ``<hermes_home>`` from it by walking one level up.
    """
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"name": name, "version": "0.1.0", "description": f"Test plugin {name}"}
    if manifest_extra:
        manifest.update(manifest_extra)

    (plugin_dir / "plugin.yaml").write_text(yaml.dump(manifest))
    (plugin_dir / "__init__.py").write_text(
        f"def register(ctx):\n    {register_body}\n"
    )

    if auto_enable:
        # Write/merge plugins.enabled in <HERMES_HOME>/config.yaml.
        # Config is always read from HERMES_HOME (not from the project
        # dir for project plugins), so that's where we opt in.
        import os
        hermes_home_str = os.environ.get("HERMES_HOME")
        if hermes_home_str:
            hermes_home = Path(hermes_home_str)
        else:
            hermes_home = base.parent
        hermes_home.mkdir(parents=True, exist_ok=True)
        cfg_path = hermes_home / "config.yaml"
        cfg: dict = {}
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
            except Exception:
                cfg = {}
        plugins_cfg = cfg.setdefault("plugins", {})
        enabled = plugins_cfg.setdefault("enabled", [])
        if isinstance(enabled, list) and name not in enabled:
            enabled.append(name)
        cfg_path.write_text(yaml.safe_dump(cfg))

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

        # Filter out bundled plugins — they're always discovered.
        non_bundled = {
            n: p for n, p in mgr._plugins.items()
            if p.manifest.source != "bundled"
        }
        assert len(non_bundled) == 1

    def test_discover_skips_dir_without_manifest(self, tmp_path, monkeypatch):
        """Directories without plugin.yaml are silently skipped."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        (plugins_dir / "no_manifest").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        # Filter out bundled plugins — they're always discovered.
        non_bundled = {
            n: p for n, p in mgr._plugins.items()
            if p.manifest.source != "bundled"
        }
        assert len(non_bundled) == 0

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
        # Explicitly enable so the loader tries to import it and hits the
        # missing-init error.
        hermes_home = tmp_path / "hermes_test"
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": ["bad_plugin"]}})
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        mgr = PluginManager()
        mgr.discover_and_load()

        assert "bad_plugin" in mgr._plugins
        assert not mgr._plugins["bad_plugin"].enabled
        assert mgr._plugins["bad_plugin"].error is not None
        # Should be the missing-init error, not "not enabled".
        assert "not enabled" not in mgr._plugins["bad_plugin"].error

    def test_load_missing_register_fn(self, tmp_path, monkeypatch):
        """Plugin without register() function records an error."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        plugin_dir = plugins_dir / "no_reg"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": "no_reg"}))
        (plugin_dir / "__init__.py").write_text("# no register function\n")
        # Explicitly enable it so the loader actually tries to import.
        hermes_home = tmp_path / "hermes_test"
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": ["no_reg"]}})
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

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
        assert "transform_terminal_output" in VALID_HOOKS
        assert "transform_tool_result" in VALID_HOOKS

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

    def test_transform_terminal_output_hook_can_be_registered_and_invoked(self, tmp_path, monkeypatch):
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "transform_hook",
            register_body=(
                'ctx.register_hook("transform_terminal_output", '
                'lambda **kw: f"{kw[\'command\']}|{kw[\'returncode\']}|{kw[\'env_type\']}|{kw[\'task_id\']}|{len(kw[\'output\'])}")'
            ),
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        results = mgr.invoke_hook(
            "transform_terminal_output",
            command="echo hello",
            output="abcdef",
            returncode=7,
            task_id="task-1",
            env_type="local",
        )
        assert results == ["echo hello|7|local|task-1|6"]

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


class TestPreToolCallBlocking:
    """Tests for the pre_tool_call block directive helper."""

    def test_block_message_returned_for_valid_directive(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [{"action": "block", "message": "blocked by plugin"}],
        )
        assert get_pre_tool_call_block_message("todo", {}, task_id="t1") == "blocked by plugin"

    def test_invalid_returns_are_ignored(self, monkeypatch):
        """Various malformed hook returns should not trigger a block."""
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [
                "block",                                 # not a dict
                123,                                     # not a dict
                {"action": "block"},                     # missing message
                {"action": "deny", "message": "nope"},   # wrong action
                {"message": "missing action"},            # no action key
                {"action": "block", "message": 123},     # message not str
            ],
        )
        assert get_pre_tool_call_block_message("todo", {}, task_id="t1") is None

    def test_none_when_no_hooks(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [],
        )
        assert get_pre_tool_call_block_message("web_search", {"q": "test"}) is None

    def test_first_valid_block_wins(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [
                {"action": "allow"},
                {"action": "block", "message": "first blocker"},
                {"action": "block", "message": "second blocker"},
            ],
        )
        assert get_pre_tool_call_block_message("terminal", {}) == "first blocker"


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
        hermes_home = tmp_path / "hermes_test"
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": ["tool_plugin"]}})
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

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
        hermes_home = tmp_path / "hermes_test"
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": ["vis_plugin"]}})
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

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


# ── TestPluginCommands ────────────────────────────────────────────────────


class TestPluginCommands:
    """Tests for plugin slash command registration via register_command()."""

    def test_register_command_basic(self):
        """register_command() stores handler, description, and plugin name."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        handler = lambda args: f"echo {args}"
        ctx.register_command("mycmd", handler, description="My custom command")

        assert "mycmd" in mgr._plugin_commands
        entry = mgr._plugin_commands["mycmd"]
        assert entry["handler"] is handler
        assert entry["description"] == "My custom command"
        assert entry["plugin"] == "test-plugin"

    def test_register_command_normalizes_name(self):
        """Names are lowercased, stripped, and leading slashes removed."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        ctx.register_command("/MyCmd ", lambda a: a, description="test")
        assert "mycmd" in mgr._plugin_commands
        assert "/MyCmd " not in mgr._plugin_commands

    def test_register_command_empty_name_rejected(self, caplog):
        """Empty name after normalization is rejected with a warning."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            ctx.register_command("", lambda a: a)
        assert len(mgr._plugin_commands) == 0
        assert "empty name" in caplog.text

    def test_register_command_builtin_conflict_rejected(self, caplog):
        """Commands that conflict with built-in names are rejected."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
            ctx.register_command("help", lambda a: a)
        assert "help" not in mgr._plugin_commands
        assert "conflicts" in caplog.text.lower()

    def test_register_command_default_description(self):
        """Missing description defaults to 'Plugin command'."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        ctx.register_command("status-cmd", lambda a: a)
        assert mgr._plugin_commands["status-cmd"]["description"] == "Plugin command"

    def test_get_plugin_command_handler_found(self):
        """get_plugin_command_handler() returns the handler for a registered command."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        handler = lambda args: f"result: {args}"
        ctx.register_command("mycmd", handler, description="test")

        with patch("hermes_cli.plugins._plugin_manager", mgr):
            result = get_plugin_command_handler("mycmd")
            assert result is handler

    def test_get_plugin_command_handler_not_found(self):
        """get_plugin_command_handler() returns None for unregistered commands."""
        mgr = PluginManager()
        with patch("hermes_cli.plugins._plugin_manager", mgr):
            assert get_plugin_command_handler("nonexistent") is None

    def test_get_plugin_commands_returns_dict(self):
        """get_plugin_commands() returns the full commands dict."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)
        ctx.register_command("cmd-a", lambda a: a, description="A")
        ctx.register_command("cmd-b", lambda a: a, description="B")

        with patch("hermes_cli.plugins._plugin_manager", mgr):
            cmds = get_plugin_commands()
            assert "cmd-a" in cmds
            assert "cmd-b" in cmds
            assert cmds["cmd-a"]["description"] == "A"

    def test_commands_tracked_on_loaded_plugin(self, tmp_path, monkeypatch):
        """Commands registered during discover_and_load() are tracked on LoadedPlugin."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        _make_plugin_dir(
            plugins_dir, "cmd-plugin",
            register_body=(
                'ctx.register_command("mycmd", lambda a: "ok", description="Test")'
            ),
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))

        mgr = PluginManager()
        mgr.discover_and_load()

        loaded = mgr._plugins["cmd-plugin"]
        assert loaded.enabled
        assert "mycmd" in loaded.commands_registered

    def test_commands_in_list_plugins_output(self, tmp_path, monkeypatch):
        """list_plugins() includes command count."""
        plugins_dir = tmp_path / "hermes_test" / "plugins"
        # Set HERMES_HOME BEFORE _make_plugin_dir so auto-enable targets
        # the right config.yaml.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_test"))
        _make_plugin_dir(
            plugins_dir, "cmd-plugin",
            register_body=(
                'ctx.register_command("mycmd", lambda a: "ok", description="Test")'
            ),
        )

        mgr = PluginManager()
        mgr.discover_and_load()

        info = mgr.list_plugins()
        # Filter out bundled plugins — they're always discovered.
        cmd_info = [p for p in info if p["name"] == "cmd-plugin"]
        assert len(cmd_info) == 1
        assert cmd_info[0]["commands"] == 1

    def test_handler_receives_raw_args(self):
        """The handler is called with the raw argument string."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        received = []
        ctx.register_command("echo", lambda args: received.append(args) or "ok")

        handler = mgr._plugin_commands["echo"]["handler"]
        handler("hello world")
        assert received == ["hello world"]

    def test_multiple_plugins_register_different_commands(self):
        """Multiple plugins can each register their own commands."""
        mgr = PluginManager()

        for plugin_name, cmd_name in [("plugin-a", "cmd-a"), ("plugin-b", "cmd-b")]:
            manifest = PluginManifest(name=plugin_name, source="user")
            ctx = PluginContext(manifest, mgr)
            ctx.register_command(cmd_name, lambda a: a, description=f"From {plugin_name}")

        assert "cmd-a" in mgr._plugin_commands
        assert "cmd-b" in mgr._plugin_commands
        assert mgr._plugin_commands["cmd-a"]["plugin"] == "plugin-a"
        assert mgr._plugin_commands["cmd-b"]["plugin"] == "plugin-b"


# ── TestPluginDispatchTool ────────────────────────────────────────────────


class TestPluginDispatchTool:
    """Tests for PluginContext.dispatch_tool() — tool dispatch with agent context."""

    def test_dispatch_tool_calls_registry(self):
        """dispatch_tool() delegates to registry.dispatch()."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        mock_registry = MagicMock()
        mock_registry.dispatch.return_value = '{"result": "ok"}'

        with patch("hermes_cli.plugins.PluginContext.dispatch_tool.__module__", "hermes_cli.plugins"):
            with patch.dict("sys.modules", {}):
                with patch("tools.registry.registry", mock_registry):
                    result = ctx.dispatch_tool("web_search", {"query": "test"})

        assert result == '{"result": "ok"}'

    def test_dispatch_tool_injects_parent_agent_from_cli_ref(self):
        """When _cli_ref has an agent, it's passed as parent_agent."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        mock_agent = MagicMock()
        mock_cli = MagicMock()
        mock_cli.agent = mock_agent
        mgr._cli_ref = mock_cli

        mock_registry = MagicMock()
        mock_registry.dispatch.return_value = '{"ok": true}'

        with patch("tools.registry.registry", mock_registry):
            ctx.dispatch_tool("delegate_task", {"goal": "test"})

        mock_registry.dispatch.assert_called_once()
        call_kwargs = mock_registry.dispatch.call_args
        assert call_kwargs[1].get("parent_agent") is mock_agent

    def test_dispatch_tool_no_parent_agent_when_no_cli_ref(self):
        """When _cli_ref is None (gateway mode), no parent_agent is injected."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)
        mgr._cli_ref = None

        mock_registry = MagicMock()
        mock_registry.dispatch.return_value = '{"ok": true}'

        with patch("tools.registry.registry", mock_registry):
            ctx.dispatch_tool("delegate_task", {"goal": "test"})

        call_kwargs = mock_registry.dispatch.call_args
        assert "parent_agent" not in call_kwargs[1]

    def test_dispatch_tool_no_parent_agent_when_agent_is_none(self):
        """When cli_ref exists but agent is None (not yet initialized), skip parent_agent."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        mock_cli = MagicMock()
        mock_cli.agent = None
        mgr._cli_ref = mock_cli

        mock_registry = MagicMock()
        mock_registry.dispatch.return_value = '{"ok": true}'

        with patch("tools.registry.registry", mock_registry):
            ctx.dispatch_tool("delegate_task", {"goal": "test"})

        call_kwargs = mock_registry.dispatch.call_args
        assert "parent_agent" not in call_kwargs[1]

    def test_dispatch_tool_respects_explicit_parent_agent(self):
        """Explicit parent_agent kwarg is not overwritten by _cli_ref.agent."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)

        cli_agent = MagicMock(name="cli_agent")
        mock_cli = MagicMock()
        mock_cli.agent = cli_agent
        mgr._cli_ref = mock_cli

        explicit_agent = MagicMock(name="explicit_agent")

        mock_registry = MagicMock()
        mock_registry.dispatch.return_value = '{"ok": true}'

        with patch("tools.registry.registry", mock_registry):
            ctx.dispatch_tool("delegate_task", {"goal": "test"}, parent_agent=explicit_agent)

        call_kwargs = mock_registry.dispatch.call_args
        assert call_kwargs[1]["parent_agent"] is explicit_agent

    def test_dispatch_tool_forwards_extra_kwargs(self):
        """Extra kwargs are forwarded to registry.dispatch()."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)
        mgr._cli_ref = None

        mock_registry = MagicMock()
        mock_registry.dispatch.return_value = '{"ok": true}'

        with patch("tools.registry.registry", mock_registry):
            ctx.dispatch_tool("some_tool", {"x": 1}, task_id="test-123")

        call_kwargs = mock_registry.dispatch.call_args
        assert call_kwargs[1]["task_id"] == "test-123"

    def test_dispatch_tool_returns_json_string(self):
        """dispatch_tool() returns the raw JSON string from the registry."""
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin", source="user")
        ctx = PluginContext(manifest, mgr)
        mgr._cli_ref = None

        mock_registry = MagicMock()
        mock_registry.dispatch.return_value = '{"error": "Unknown tool: fake"}'

        with patch("tools.registry.registry", mock_registry):
            result = ctx.dispatch_tool("fake", {})

        assert '"error"' in result
