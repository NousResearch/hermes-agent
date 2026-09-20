"""A parent's final response must not withdraw its live detached child's approvals."""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gateway.run_turn_runner import TurnRunner
from tools import approval, async_delegation, delegate_tool
from tools.approval_context import get_current_session_key
from tools.approval_gateway_wait import _await_gateway_decision
from tools.process_registry import process_registry


@pytest.mark.parametrize("request_phase", ["pending", "after_parent", "queued"])
def test_detached_child_can_approve_before_and_after_parent_returns(request_phase, monkeypatch):
    """Exercise real turn cleanup, dispatch/context propagation, delivery lookup, and resolution."""
    session_key = "agent:main:telegram:group:approval-lifetime:1"
    published = queue.Queue()
    start_request = threading.Event()
    decision = {}
    async_delegation._reset_for_tests()
    monkeypatch.setattr(delegate_tool, "_get_worktree_isolation", lambda: False)
    start_worker = threading.Event()
    queued_executor = None
    if request_phase == "queued":
        queued_executor = ThreadPoolExecutor(max_workers=1)
        queued_executor.submit(start_worker.wait, 10)
        monkeypatch.setattr(async_delegation, "_get_executor", lambda _: queued_executor)

    def child():
        assert start_request.wait(10)
        key = get_current_session_key()
        notify = approval._gateway_notify_cb(key)
        if notify is None:
            published.put(None)
            return {"status": "error", "error": "live child lost approval delivery"}
        decision.update(_await_gateway_decision(key, notify, {
            "command": "synthetic operation (never executed)",
            "description": "test approval", "pattern_key": "test",
        }))
        return {"status": "completed", "summary": "approval answered"}

    class ApprovalChild:
        tool_progress_callback = None
        _credential_pool = None
        _delegate_saved_tool_names = []
        _delegate_role = "leaf"
        _delegate_depth = 1
        _subagent_id = None
        model = "test-model"
        session_prompt_tokens = session_completion_tokens = 0
        session_estimated_cost_usd = 0.0
        session_cost_status = "unknown"

        def run_conversation(self, **kwargs):
            result = child()
            return {**result, "completed": result["status"] == "completed",
                    "final_response": result.get("summary", ""), "api_calls": 1, "messages": []}

        def get_activity_summary(self):
            return {"api_call_count": 1}

        def close(self):
            pass

    child_agent = ApprovalChild()
    first_request = []

    def parent_conversation(message, **kwargs):
        dispatched = async_delegation.dispatch_async_delegation(
            goal="test child", context=None, toolsets=None, role="leaf", model=None,
            session_key=session_key, max_async_children=1,
            runner=lambda: delegate_tool._run_single_child(
                task_index=0, goal="test child", child=child_agent, parent_agent=parent),
        )
        assert dispatched["status"] == "dispatched"
        if request_phase == "pending":
            start_request.set()
            first_request.append(published.get(timeout=10))
        return {"final_response": "The child is still working."}

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        session_key=session_key, session_id="parent-session", message="delegate",
        source=SimpleNamespace(user_id="test-user", user_name="test"),
        persist_user_display_kind=None, persist_user_display_metadata=None, moa_config=None, inbound_message_id=None,
        mute_notification_reply=False,
    )
    runner._native_image_run_message = lambda: "delegate"
    runner._approval_notify_sync = published.put
    parent = SimpleNamespace(run_conversation=parent_conversation, session_id="parent-session",
                             _current_task_id=None, _active_children=[], _active_children_lock=threading.Lock())
    try:
        runner._run_conversation_with_approval(parent, [], None, None, None)
        start_request.set()
        start_worker.set()
        request = first_request[0] if request_phase == "pending" else published.get(timeout=10)
        assert request is not None, "parent return removed delivery for a live child's next approval"
        assert approval.resolve_gateway_approval(session_key, "once", request_id=request["request_id"]) == 1
        event = process_registry.completion_queue.get(timeout=10)
        assert event["type"] == "async_delegation"
        assert decision["choice"] == "once"
        assert not decision.get("cancelled")
    finally:
        start_request.set()
        start_worker.set()
        approval.unregister_gateway_notify(session_key)
        executor = async_delegation._executor
        if executor is not None:
            executor.shutdown(wait=True)
        if queued_executor is not None:
            queued_executor.shutdown(wait=True)
        async_delegation._reset_for_tests()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()
