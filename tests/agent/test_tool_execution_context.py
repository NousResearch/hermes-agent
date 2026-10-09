"""Runtime lineage is bound by dispatch, never accepted from a model's arguments."""
import json
import weakref
from dataclasses import dataclass
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from agent.tool_execution_context import (
    bind_tool_execution_context, current_tool_execution_context, dispatch_in_tool_context,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@dataclass
class Agent:
    session_id: str
    _delegate_depth: int = 0
    _delegate_parent_ref: object = None
    valid_tool_names: object = None
    _memory_manager: object = None


def test_nested_identity_is_immutable_and_restored(tmp_path):
    root = Agent("root")
    child = Agent("child", 1, weakref.ref(root))
    grandchild = Agent("grandchild", 2, weakref.ref(child))
    token = set_hermes_home_override(tmp_path)
    try:
        with bind_tool_execution_context(root):
            before = current_tool_execution_context()
            with bind_tool_execution_context(grandchild, tool_call_id="tc"):
                actual = current_tool_execution_context()
                assert actual["root_session_id"] == "root"
                assert actual["parent_session_id"] == "child"
                assert actual["delegate_depth"] == 2 and actual["lineage_valid"]
                assert actual["profile_home"] == str(tmp_path)
                with pytest.raises(TypeError):
                    actual["root_session_id"] = "forged"
            assert current_tool_execution_context() == before
    finally:
        reset_hermes_home_override(token)
    assert not current_tool_execution_context()["lineage_valid"]


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda root, child: setattr(child, "_delegate_parent_ref", None), id="missing"),
    pytest.param(lambda root, child: setattr(child, "_delegate_parent_ref", weakref.ref(child)), id="cycle"),
    pytest.param(lambda root, child: setattr(root, "_delegate_depth", 4), id="wrong-depth"),
    pytest.param(lambda root, child: setattr(child, "session_id", "root"), id="same-session"),
    pytest.param(lambda root, child: setattr(child, "_delegate_depth", True), id="boolean-depth"),
])
def test_incomplete_lineage_has_no_root_authority(mutate):
    root = Agent("root")
    child = Agent("child", 1, weakref.ref(root))
    mutate(root, child)
    actual = dispatch_in_tool_context(child, lambda: current_tool_execution_context())
    assert not actual["lineage_valid"] and actual["root_session_id"] == ""


def test_dead_parent_has_no_root_authority():
    root = Agent("root")
    child = Agent("child", 1, weakref.ref(root))
    del root
    actual = dispatch_in_tool_context(child, current_tool_execution_context)
    assert not actual["lineage_valid"] and actual["root_session_id"] == ""


def test_invoke_tool_binds_runtime_identity_not_forged_arguments(monkeypatch):
    from agent.agent_runtime_helpers import invoke_tool
    import model_tools

    def handler(*args, **kwargs):
        return json.dumps(dict(current_tool_execution_context()))

    monkeypatch.setattr(model_tools, "handle_function_call", handler)
    actual = json.loads(invoke_tool(
        Agent("real"), "terminal", {"root_session_id": "forged"}, "task", "tc",
        pre_tool_block_checked=True, skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    ))
    assert actual["root_session_id"] == actual["session_id"] == "real"
    assert actual["lineage_valid"] and actual["tool_call_id"] == "tc"
    assert not current_tool_execution_context()["lineage_valid"]


def test_boolean_ancestor_depth_is_invalid():
    root = Agent("root")
    child = Agent("child", True, weakref.ref(root))
    grandchild = Agent("grandchild", 2, weakref.ref(child))
    actual = dispatch_in_tool_context(grandchild, current_tool_execution_context)
    assert actual["lineage_valid"] is False


def test_managed_dispatch_binds_worker_identity_in_actual_thread(tmp_path, monkeypatch):
    from agent import tool_executor as executor

    root = Agent("root")
    child = Agent("worker", 1, weakref.ref(root))
    child._tool_guardrails = SimpleNamespace(
        before_call=lambda *args: SimpleNamespace(allows_execution=True))
    monkeypatch.setattr(executor, "_pre_tool_block", lambda agent, ref: (None, ref.args))
    monkeypatch.setattr(executor, "_begin_tool_execution", lambda *args: None)
    ref = executor._ToolCallRef("probe", {}, "worker-task", "worker-call", [])
    token = set_hermes_home_override(tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            def dispatch():
                actual = executor._dispatch_authorized_once(
                    child, SimpleNamespace(), ref,
                    execute=lambda args: dict(current_tool_execution_context()),
                    scope_block=None, display_index=None, begin_execution=None,
                    authorization_gate=None,
                )
                assert not current_tool_execution_context()["lineage_valid"]
                return actual
            actual = pool.submit(copy_context().run, dispatch).result(timeout=10)
    finally:
        reset_hermes_home_override(token)
    assert actual["session_id"] == "worker" and actual["root_session_id"] == "root"
    assert actual["profile_home"] == str(tmp_path)
    assert actual["task_id"] == "worker-task" and actual["tool_call_id"] == "worker-call"
