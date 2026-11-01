import importlib.util
import json
import os
import sys

PLUGIN_INIT = os.path.join(os.path.dirname(__file__), "..", "__init__.py")
_spec = importlib.util.spec_from_file_location("subagent_handles_plugin", PLUGIN_INIT)
plugin = importlib.util.module_from_spec(_spec)
sys.modules["subagent_handles_plugin"] = plugin
_spec.loader.exec_module(plugin)

from subagent_handles import status
from subagent_handles.status import SCHEMA, handle_subagent_handles
from subagent_handles.registry import registry as shared_registry


def _load(out: str) -> dict:
    """Tool handlers return a JSON string; decode for assertion."""
    return json.loads(out)


def _install_registry():
    """Reset the shared singleton to a clean state for test isolation.

    Uses the SAME module-level singleton that the hook handlers and the
    status tool both reference — production wiring, not a throwaway
    instance. This exercises the real hook→tool data path.
    """
    for h in list(shared_registry):
        shared_registry.remove(h.subagent_id)
    return shared_registry


def _register(handle):
    shared_registry.register(handle)


def test_handles_missing_handle():
    _install_registry()
    out = _load(handle_subagent_handles({"subagent_id": "missing"}))
    assert "error" in out
    assert "not found" in out["error"]


def test_handles_resolve_one():
    registry = _install_registry()
    registry.register(
        type("H", (), {"subagent_id": "a1", "session_id": "s1", "goal": "g1", "state": "running", "parent_subagent_id": None, "role": ""})()
    )
    out = _load(handle_subagent_handles({"subagent_id": "a1"}))
    assert out["handle"]["subagent_id"] == "a1"
    assert out["handle"]["state"] == "running"
    assert out["handle"]["session_id"] == "s1"


def test_handles_list_all():
    registry = _install_registry()
    registry.register(
        type("H", (), {"subagent_id": "a1", "session_id": "s1", "goal": "g1", "state": "running", "parent_subagent_id": None, "role": ""})()
    )
    registry.register(
        type("H", (), {"subagent_id": "a2", "session_id": "s2", "goal": "g2", "state": "done", "parent_subagent_id": None, "role": "coder"})()
    )
    out = _load(handle_subagent_handles({}))
    assert out["count"] == 2
    ids = {h["subagent_id"] for h in out["handles"]}
    assert ids == {"a1", "a2"}


def test_schema_shape():
    assert SCHEMA["name"] == "subagent_handles"
    assert "description" in SCHEMA
    params = SCHEMA["parameters"]
    assert params.get("type") == "object"
    assert "subagent_id" in params.get("properties", {})
    # read-only tool: no required args
    assert params.get("required", []) == []


def test_register_tools_wiring():
    """register_tools must use the correct PluginContext.register_tool signature.

    Regression guard for the bug where SCHEMA dict was passed as `name` and
    the handler as `toolset`, silently registering nothing.
    """
    calls = []

    class FakeCtx:
        def register_tool(self, **kwargs):
            calls.append(kwargs)

        def register_hook(self, *a, **k):
            pass

    status.register_tools(FakeCtx())
    assert calls, "register_tool was never called"
    assert calls[0]["name"] == "subagent_handles"
    assert calls[0]["schema"]["name"] == "subagent_handles"
    assert calls[0]["handler"] is handle_subagent_handles


def test_plugin_registers_status_tool_not_steering():
    """The plugin must NOT register subagent_send/cancel_subagent (those names
    belong to the platform delegation toolset and would be rejected as
    shadowing without override=True)."""
    calls = []

    class FakeCtx:
        def register_tool(self, **kwargs):
            calls.append(kwargs["name"])

        def register_hook(self, *a, **k):
            pass

    plugin.register(FakeCtx())
    assert "subagent_send" not in calls, "must not shadow platform subagent_send"
    assert "cancel_subagent" not in calls, "must not shadow platform cancel_subagent"
    assert "subagent_handles" in calls
