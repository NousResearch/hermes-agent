"""Offline full-parent-loop closeout canary with real queue and SQLite storage."""

import json
import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from run_agent import AIAgent
from tools import async_delegation as ad
from tools.process_registry import process_registry
from tools.process_registry_notifications import format_process_notification


@pytest.mark.parametrize("finish_before_seal", [True, False])
@pytest.mark.parametrize("replacement", [False, True])
def test_parent_reconciles_actual_batch_results_before_one_final(
    monkeypatch, finish_before_seal, replacement
):
    ad._reset_for_tests()
    pending = queue.Queue()
    monkeypatch.setattr(process_registry, "completion_queue", pending)
    monkeypatch.setattr(ad, "task_scoped_closeout_enabled", lambda config=None: True)
    monkeypatch.setattr(
        "gateway.session_context.closeout_delivery_supported", lambda: True
    )
    tool = {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": "delegate",
            "parameters": {
                "type": "object",
                "properties": {"tasks": {"type": "array"}},
            },
        },
    }
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kw: [tool])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda **_kw: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    deferred = []

    class Executor:
        def submit(self, fn):
            if finish_before_seal:
                fn()
            else:
                deferred.append(fn)

    monkeypatch.setattr(ad, "_get_executor", lambda _limit: Executor())
    agent = AIAgent(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        max_iterations=10,
    )
    agent.api_mode = "chat_completions"
    dispatched = []

    def dispatch(**kw):
        goal = kw["tasks"][0]["goal"]
        outcomes = [
            {"task_index": 0, "status": "completed", "summary": goal + "-evidence"}
        ]
        if replacement and goal == "review-2":
            outcomes[0].update(
                status="error", error="required-review-blocker", schema_valid=False
            )
        handle = ad.dispatch_async_delegation_batch(
            goals=[goal],
            context=None,
            toolsets=None,
            role="leaf",
            model="test/model",
            session_key=agent.session_id,
            parent_session_id=agent.session_id,
            runner=lambda: {"results": outcomes},
            **{
                k: kw[k]
                for k in (
                    "origin_work_id",
                    "work_generation",
                    "owner_turn_id",
                    "closeout_delivery_id",
                    "closeout_claim_id",
                )
            },
        )
        dispatched.append(handle)
        return json.dumps(handle)

    monkeypatch.setattr("tools.delegate_tool.delegate_task", dispatch)
    visible = []
    agent.interim_assistant_callback = lambda text, **_kw: visible.append(text)
    calls = 0

    def provider(kwargs, **_options):
        nonlocal calls
        calls += 1
        if calls <= 2:
            call = SimpleNamespace(
                id=f"call-{calls}",
                type="function",
                function=SimpleNamespace(
                    name="delegate_task",
                    arguments=json.dumps({"tasks": [{"goal": f"review-{calls}"}]}),
                ),
            )
            content, tools = "premature tool prose", [call]
        elif calls == 3:
            content, tools = "premature final", None
        else:
            received = json.dumps(kwargs["messages"])
            assert "review-1-evidence" in received and "review-2-evidence" in received
            if replacement:
                assert "required-review-blocker" in received
            if replacement and calls == 4:
                tools = [
                    SimpleNamespace(
                        id="replacement",
                        type="function",
                        function=SimpleNamespace(
                            name="delegate_task",
                            arguments=json.dumps({"tasks": [{"goal": "review-3"}]}),
                        ),
                    )
                ]
                content = "premature replacement prose"
            elif replacement and calls == 5:
                content, tools = "premature replacement final", None
            else:
                assert calls == (6 if replacement else 4)
                if replacement:
                    assert "review-3-evidence" in received
                content, tools = "Reconciled final answer", None
        agent._fire_stream_delta(content)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=content,
                        tool_calls=tools,
                        reasoning_content=None,
                    ),
                    finish_reason="tool_calls" if tools else "stop",
                )
            ],
            usage=None,
            model="test/model",
        )

    monkeypatch.setattr(agent, "_interruptible_api_call", provider)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", provider)
    try:
        first = agent.run_conversation(
            "Review both parts and give one final answer",
            stream_callback=visible.append,
        )
        assert len(dispatched) == 2 and all(
            h["status"] == "dispatched" for h in dispatched
        )
        assert first["waiting_on_delegates"] and not first["final_response"]
        assert visible == []

        def reconcile(history):
            for fn in list(deferred):
                fn()
            deferred.clear()
            event = pending.get(timeout=2)
            assert pending.empty()
            return agent.run_conversation(
                format_process_notification(event),
                conversation_history=history,
                stream_callback=visible.append,
                origin_work_id=event["origin_work_id"],
                work_generation=event["work_generation"],
                work_delivery_id=event["delivery_id"],
                work_claim_id=event["claim_id"],
            )

        second = reconcile(first["messages"])
        if replacement:
            assert second["waiting_on_delegates"] and not second["final_response"]
            assert visible == []
            second = reconcile(second["messages"])
        assert (
            second["completed"]
            and second["final_response"] == "Reconciled final answer"
        )
        assert visible == ["Reconciled final answer"]
        from tui_gateway import server

        displayed = server._history_to_messages(second["messages"])
        assert [m["text"] for m in displayed if m["role"] == "assistant"] == [
            "Reconciled final answer"
        ]
        tools = [m for m in displayed if m["role"] == "tool"]
        assert [m["args"]["tasks"][0]["goal"] for m in tools] == [
            f"review-{index}" for index in range(1, 4 if replacement else 3)
        ]
        assert ad.recover_and_enqueue_work_groups(target_queue=pending) == []
        assert pending.empty()
        with ad._transaction() as conn:
            assert conn.execute(
                "SELECT state FROM async_delegation_work_groups"
            ).fetchone() == ("closed",)
    finally:
        agent.close()
        ad._reset_for_tests()
