import importlib.util
import os
import sys

import pytest

# Load plugin __init__.py as a regular module (directory name contains a hyphen).
PLUGIN_INIT = os.path.join(os.path.dirname(__file__), "..", "__init__.py")
_spec = importlib.util.spec_from_file_location("subagent_handles_plugin", PLUGIN_INIT)
plugin = importlib.util.module_from_spec(_spec)
sys.modules["subagent_handles_plugin"] = plugin
_spec.loader.exec_module(plugin)

from subagent_handles.registry import SubagentRegistry


@pytest.fixture(autouse=True)
def _isolate_persist_store(tmp_path, monkeypatch):
    """Point the persist store at a temp dir and reset the registry per test.

    The plugin now persists on subagent_start/stop and restores on register(),
    so every test must isolate the disk store from the real HERMES_HOME store
    and start from a clean in-memory registry.
    """
    monkeypatch.setattr(
        plugin, "default_persist_root", lambda: str(tmp_path / "subagent-handles")
    )
    plugin.registry = SubagentRegistry()
    return plugin.registry


class MockCtx:
    def __init__(self) -> None:
        self._plugins = {}
        self._hooks: dict[str, list] = {}

    def register_plugin(self, name: str, obj: object) -> None:
        self._plugins[name] = obj

    def register_hook(self, hook_name: str, callback) -> None:
        self._hooks.setdefault(hook_name, []).append(callback)


def test_subagent_start_registers_handle():
    plugin.registry = SubagentRegistry()

    ctx = MockCtx()
    plugin.register(ctx)

    handler = ctx._hooks["subagent_start"][0]
    handler(
        parent_session_id="p_s1",
        parent_turn_id="p_t1",
        parent_subagent_id="p_sub1",
        child_session_id="c_s1",
        child_subagent_id="c_sub1",
        child_role="coder",
        child_goal="Write tests",
    )

    assert "c_sub1" in plugin.registry
    handle = plugin.registry.resolve("c_sub1")
    assert handle is not None
    assert handle.session_id == "c_s1"
    assert handle.goal == "Write tests"
    assert handle.parent_subagent_id == "p_sub1"
    assert handle.role == "coder"
    assert handle.state == "running"


def test_subagent_stop_marks_done():
    plugin.registry = SubagentRegistry()

    ctx = MockCtx()
    plugin.register(ctx)
    start_handler = ctx._hooks["subagent_start"][0]
    start_handler(
        parent_session_id="p_s1",
        parent_turn_id="p_t1",
        parent_subagent_id=None,
        child_session_id="c_s1",
        child_subagent_id="c_sub1",
        child_role="coder",
        child_goal="Write tests",
    )

    assert plugin.registry.resolve("c_sub1").state == "running"

    stop_handler = ctx._hooks["subagent_stop"][0]
    stop_handler(
        parent_session_id="p_s1",
        parent_turn_id="p_t1",
        child_session_id="c_s1",
        child_role="coder",
        child_summary="Done",
        child_status="success",
        tool_call_history=[],
        duration_ms=1000,
    )

    assert plugin.registry.resolve("c_sub1").state == "done"


def test_subagent_start_missing_kwargs_no_crash():
    plugin.registry = SubagentRegistry()

    ctx = MockCtx()
    plugin.register(ctx)
    handler = ctx._hooks["subagent_start"][0]

    # Missing child_subagent_id
    handler(child_session_id="c_s1", child_goal="g1")
    assert "c_sub1" not in plugin.registry

    # Missing child_session_id
    handler(child_subagent_id="c_sub1", child_goal="g1")
    assert "c_sub1" not in plugin.registry

    # Missing child_goal
    handler(child_subagent_id="c_sub1", child_session_id="c_s1")
    assert "c_sub1" not in plugin.registry


def test_subagent_stop_missing_kwargs_no_crash():
    plugin.registry = SubagentRegistry()

    ctx = MockCtx()
    plugin.register(ctx)
    handler = ctx._hooks["subagent_stop"][0]

    # Should not raise
    handler()
    assert "anything" not in plugin.registry
