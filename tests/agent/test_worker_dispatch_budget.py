"""Counterexamples for worker budgets at nested-RPC and Codex reconnect sends."""

import contextvars
import json
from types import SimpleNamespace

import httpx
import pytest

from agent.delegation_model_routing import ExecutionLimits
from agent.subagent_lifecycle import (
    SubagentLifecycleService,
    bind_subagent_parent,
)
from agent.worker_store import WorkerStore
from hermes_state import SessionDB
from tools.code_execution_rpc import _default_dispatch
from tools.code_kernel import CellAuthority


class Child:
    def __init__(self, ident, *, depth=1):
        self._subagent_id = ident
        self._delegate_role = "orchestrator"
        self._delegate_depth = depth
        self.provider = "fixture"
        self.model = "fixture-model"
        self.ephemeral_system_prompt = "fixed prompt"
        self.valid_tool_names = {"read_file", "delegate_task"}
        self._executable_tool_names = set(self.valid_tool_names)
        self.tools = []
        self._worker_route_receipt = {}
        self._interrupt_requested = False

    def hard_interrupt(self, _message=None, **_kwargs):
        self._interrupt_requested = True
        return True

    def close(self):
        return None


def _creds(*, iterations=4, tools=4):
    return {
        "requested_profile": None,
        "resolved_provider": "fixture",
        "resolved_model": "fixture-model",
        "resolved_reasoning_effort": None,
        "max_iterations": iterations,
        "execution_limits": ExecutionLimits(
            max_iterations=iterations, max_tool_calls=tools, timeout_seconds=60),
    }


def _finish(child, status="completed"):
    child._worker_last_history = [{"role": "assistant", "content": "done"}]
    SubagentLifecycleService.complete_adopted_child(child, {
        "status": status, "summary": "done", "api_calls": 0,
        "exit_reason": "completed" if status == "completed" else "interrupted",
    })


def test_local_and_remote_execute_code_rpc_share_tree_budget(tmp_path, monkeypatch):
    """New cells and a descendant's larger ceiling cannot multiply the root budget."""
    import model_tools

    db = SessionDB(tmp_path / "state.db")
    parent = SimpleNamespace(session_id="owner", _session_db=db)
    root = Child("root")
    root_service = SubagentLifecycleService(lambda: parent)
    root_service.adopt_delegate_child(
        root, goal="root", context=None, profile=None, creds=_creds(tools=3),
        cfg={"max_concurrent_children": 2},
    )
    nested = Child("nested", depth=2)
    nested_service = SubagentLifecycleService(lambda: root)
    nested_service.adopt_delegate_child(
        nested, goal="nested", context=None, profile=None, creds=_creds(tools=10),
        cfg={"max_concurrent_children": 2},
    )

    executed = []

    def execute(function_name, function_args, *_args, **_kwargs):
        executed.append((function_name, function_args["value"]))
        return json.dumps({"value": function_args["value"]})

    monkeypatch.setattr(model_tools, "_execute_tool", execute)
    with bind_subagent_parent(nested):
        first_cell = CellAuthority("task")
        inherited = contextvars.copy_context()
    assert json.loads(first_cell.dispatch("fixture_tool", {"value": "cell-1"}))["value"] == "cell-1"
    first_cell.retire()

    with bind_subagent_parent(nested):
        second_cell = CellAuthority("task")
    assert json.loads(second_cell.dispatch("fixture_tool", {"value": "cell-2"}))["value"] == "cell-2"
    second_cell.retire()

    remote_dispatch = inherited.run(_default_dispatch, "task")
    assert json.loads(inherited.run(remote_dispatch, "fixture_tool", {"value": "remote"}))["value"] == "remote"
    denied = json.loads(inherited.run(remote_dispatch, "fixture_tool", {"value": "denied"}))
    assert "tree tool call budget exhausted" in denied["error"]
    assert executed == [
        ("fixture_tool", "cell-1"),
        ("fixture_tool", "cell-2"),
        ("fixture_tool", "remote"),
    ]
    store = WorkerStore(db)
    assert store.budget_snapshot(nested._worker_run_id, "owner")["used_tool_calls"] == 3
    assert store.get_run(nested._worker_run_id, "owner")["tool_inflight_count"] == 0
    _finish(nested)
    _finish(root)
    db.close()


class _RetryingResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise httpx.RemoteProtocolError("synthetic reconnect")
        return iter([
            SimpleNamespace(type="response.output_text.delta", delta="done"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id="response-2", status="completed", usage=None,
                    model="fixture-model", output=None,
                ),
            ),
        ])


def _send_codex(child, responses):
    from agent.codex_runtime import run_codex_stream
    from agent.turn_api_call import perform_api_call

    child.api_mode = "codex_responses"
    child.session_id = "worker-session"
    child.platform = "subagent"
    child.base_url = "https://example.invalid"
    child._fallback_index = 0
    child.is_subagent = True
    child.show_commentary = False
    child.interim_assistant_callback = None
    child._pending_redirect_lock = None
    child._has_pending_redirect = lambda: False
    child._has_stream_consumers = lambda: True
    child._is_copilot_url = lambda: False
    child._is_codex_backend = lambda: True
    child._get_transport = lambda: SimpleNamespace(
        preflight_kwargs=lambda kwargs, **_options: dict(kwargs))
    child._fire_stream_delta = lambda _text: None
    child._fire_reasoning_delta = lambda _text: None
    child._touch_activity = lambda _text: None
    child._client_log_context = lambda: "fixture"
    child._abort_request_openai_client = lambda *_args, **_kwargs: None
    client = SimpleNamespace(responses=responses)
    child._interruptible_streaming_api_call = lambda kwargs, on_first_delta=None: run_codex_stream(
        child, kwargs, client=client, on_first_delta=on_first_delta)
    return perform_api_call(
        child, api_kwargs={"model": "fixture-model", "input": "synthetic"},
        _original_api_kwargs={}, _llm_middleware_trace=[], _moa_prepared_request=None,
        _retry=SimpleNamespace(), thinking_spinner=None, retry_count=0, api_call_count=0,
        api_request_id="request", effective_task_id="task", turn_id="turn", interrupted=False,
    )


@pytest.mark.parametrize("allocation,succeeds", [(1, False), (2, True)])
def test_codex_reconnect_reserves_each_physical_send_without_outer_charge(
    tmp_path, monkeypatch, allocation, succeeds,
):
    from agent import relay_llm
    from hermes_cli import middleware

    db = SessionDB(tmp_path / f"state-{allocation}.db")
    parent = SimpleNamespace(session_id=f"owner-{allocation}", _session_db=db)
    child = Child(f"codex-{allocation}")
    SubagentLifecycleService(lambda: parent).adopt_delegate_child(
        child, goal="retry", context=None, profile=None,
        creds=_creds(iterations=allocation, tools=1),
        cfg={"max_iterations": allocation},
    )
    responses = _RetryingResponses()
    monkeypatch.setattr(
        middleware, "run_llm_execution_middleware",
        lambda kwargs, send, **_options: send(kwargs),
    )
    monkeypatch.setattr(
        relay_llm, "stream", lambda request, stream_factory, **_options: stream_factory(request),
    )

    if succeeds:
        verdict = _send_codex(child, responses)
        assert verdict.response.status == "completed"
        assert len(responses.calls) == 2
        assert len(child._worker_route_receipt["execution_attempts"]) == 2
        _finish(child)
    else:
        with pytest.raises(InterruptedError, match="tree_iteration_budget_exhausted"):
            _send_codex(child, responses)
        assert len(responses.calls) == 1
        assert len(child._worker_route_receipt["execution_attempts"]) == 1
        _finish(child, "interrupted")
    snapshot = WorkerStore(db).budget_snapshot(child._worker_run_id, parent.session_id)
    assert snapshot["used_iterations"] == allocation
    db.close()
