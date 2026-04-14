"""End-to-end integration tests for hongxing-enhancements plugin (Review Fix).

Verifies that the plugin system actually discovers, loads, and activates
the permission engine and plan mode hooks at runtime.
"""

import importlib.util
import json
import os
import shutil
import sys

import pytest


# ── Helpers ────────────────────────────────────────────────────────────

# Path to the real plugin source in the project
_PROJECT_PLUGIN_SRC = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir,
    "plugins", "hongxing-enhancements",
)
# Path to the real user plugin (for __init__.py with register())
_USER_PLUGIN_SRC = os.path.join(
    os.path.expanduser("~"), ".hermes", "plugins", "hongxing-enhancements",
)


def _load_plan_mode_hook():
    """Load plan_mode_hook via importlib (hyphenated directory)."""
    mod_name = "plan_mode_hook"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name, os.path.join(_PROJECT_PLUGIN_SRC, "plan_mode_hook.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _reset_plan_mode():
    """Ensure plan mode is off before and after each test."""
    mod = _load_plan_mode_hook()
    mod.exit_plan_mode()
    yield
    mod.exit_plan_mode()


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    """Set up a fake HERMES_HOME with the plugin properly installed."""
    fake_home = tmp_path / "hermes_int"
    fake_home.mkdir()
    (fake_home / "sessions").mkdir()
    (fake_home / "memories").mkdir()

    # Copy user plugin files (plugin.yaml + __init__.py) into fake home
    dest = fake_home / "plugins" / "hongxing-enhancements"
    dest.mkdir(parents=True)
    for fname in ["plugin.yaml", "__init__.py"]:
        src = os.path.join(_USER_PLUGIN_SRC, fname)
        if os.path.isfile(src):
            shutil.copy2(src, str(dest / fname))

    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    # Clear cached hermes home
    try:
        import hermes_constants
        hermes_constants.get_hermes_home.cache_clear()
    except (ImportError, AttributeError):
        pass
    return fake_home


# ── Plugin Discovery ──────────────────────────────────────────────────

class TestPluginDiscovery:
    def test_plugin_found(self, plugin_home):
        from hermes_cli.plugins import PluginManager
        mgr = PluginManager()
        mgr.discover_and_load()
        names = [p["name"] for p in mgr.list_plugins()]
        assert "hongxing-enhancements" in names

    def test_plugin_enabled(self, plugin_home):
        from hermes_cli.plugins import PluginManager
        mgr = PluginManager()
        mgr.discover_and_load()
        plugins = {p["name"]: p for p in mgr.list_plugins()}
        assert "hongxing-enhancements" in plugins
        # Plugin is enabled (register() was called without error)
        assert plugins["hongxing-enhancements"]["enabled"] is True


# ── Tool Discovery Chain ─────────────────────────────────────────────

class TestToolDiscovery:
    def test_plan_mode_in_registry(self):
        import model_tools  # noqa: F401 — triggers _discover_tools()
        from tools.registry import registry
        assert "plan_mode" in registry.get_all_tool_names()

    def test_coordinator_in_registry(self):
        import model_tools  # noqa: F401 — triggers _discover_tools()
        from tools.registry import registry
        assert "coordinate" in registry.get_all_tool_names()


# ── Plan Mode State Persistence ──────────────────────────────────────

class TestPlanModeState:
    def test_state_persistent_across_calls(self):
        from tools.plan_tool import plan_mode_handler
        enter_result = json.loads(plan_mode_handler({"action": "enter"}))
        assert enter_result["plan_mode"] is True
        status_result = json.loads(plan_mode_handler({"action": "status"}))
        assert status_result["plan_mode"] is True

    def test_exit_clears_state(self):
        from tools.plan_tool import plan_mode_handler
        plan_mode_handler({"action": "enter"})
        plan_mode_handler({"action": "exit"})
        status = json.loads(plan_mode_handler({"action": "status"}))
        assert status["plan_mode"] is False


# ── Plan Mode Denies Writes ──────────────────────────────────────────

class TestPlanModeDeniesWrites:
    def test_write_file_denied(self):
        import tools.file_tools  # noqa: F401

        mod = _load_plan_mode_hook()
        mod.enter_plan_mode()
        result = mod.pre_tool_call("write_file", {"path": "/tmp/x"})
        assert result is not None
        assert result["action"] == "deny"

    def test_read_file_allowed(self):
        import tools.file_tools  # noqa: F401

        mod = _load_plan_mode_hook()
        mod.enter_plan_mode()
        assert mod.pre_tool_call("read_file", {}) is None

    def test_terminal_denied(self):
        mod = _load_plan_mode_hook()
        mod.enter_plan_mode()
        result = mod.pre_tool_call("terminal", {"command": "ls"})
        assert result is not None
        assert result["action"] == "deny"

    def test_todo_denied(self):
        import tools.todo_tool  # noqa: F401

        mod = _load_plan_mode_hook()
        mod.enter_plan_mode()
        result = mod.pre_tool_call("todo", {"todos": []})
        assert result is not None
        assert result["action"] == "deny"

    def test_memory_read_allowed_write_denied(self):
        mod = _load_plan_mode_hook()
        mod.enter_plan_mode()
        assert mod.pre_tool_call("memory", {"action": "read"}) is None
        result = mod.pre_tool_call("memory", {"action": "add"})
        assert result["action"] == "deny"
