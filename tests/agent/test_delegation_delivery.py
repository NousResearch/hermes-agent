"""Contracts for durable async delegation delivery and tool-boundary evidence."""

from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import re
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from typing import Any
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agent import delegation_inject as inject
from agent.delegation_inject import (
    acknowledge_pending_injects,
    attach_ready_injects_to_tool_results,
    release_pending_injects,
)
from tools import async_delegation as ad
from tools import delegate_tool
from tools.budget_config import BudgetConfig
from tools.process_registry import process_registry
from tools.tool_result_storage import PERSISTED_OUTPUT_TAG
from hermes_state import SessionDB
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _clean_async_state():
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    deadline = time.monotonic() + 2
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.01)
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()


def _record(
    *,
    goals=("audit",),
    turn_id="turn-current",
    delivery="inject",
    parent_session_id="parent-session",
):
    delegation_id = f"deleg_test_{uuid.uuid4().hex}"
    record = {
        "delegation_id": delegation_id,
        "goal": goals[0] if len(goals) == 1 else f"{len(goals)} tasks",
        "goals": list(goals),
        "context": "parent context",
        "toolsets": ["file"],
        "role": "leaf",
        "model": "child-model",
        "session_key": "agent:main:cli:dm:local",
        "origin_ui_session_id": "",
        "origin_session_id": "",
        "parent_session_id": parent_session_id,
        "parent_turn_id": turn_id,
        "status": "running",
        "dispatched_at": time.time(),
        "completed_at": None,
        "is_batch": True,
        "result_delivery": delivery,
    }
    with ad._records_lock:
        ad._records[delegation_id] = record
    ad._persist_dispatch(record)
    return delegation_id


def _child(index: int, summary: str, *, status="completed", error=None):
    return {
        "task_index": index,
        "status": status,
        "summary": summary,
        "error": error,
        "api_calls": 2,
        "duration_seconds": 0.25,
    }


def _queue_contents():
    items = []
    while not process_registry.completion_queue.empty():
        items.append(process_registry.completion_queue.get_nowait())
    for item in items:
        process_registry.completion_queue.put(item)
    return items


def _event_state(delegation_id: str, event_key: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT delivery_state, delivery_attempts FROM async_delegation_events "
            "WHERE delegation_id=? AND event_key=?",
            (delegation_id, event_key),
        ).fetchone()


def _parent_state(delegation_id: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT state, delivery_state FROM async_delegations "
            "WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()


def _durable_event_keys(delegation_id: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return [
            row[0]
            for row in conn.execute(
                "SELECT event_key FROM async_delegation_events "
                "WHERE delegation_id=? ORDER BY event_key",
                (delegation_id,),
            ).fetchall()
        ]


def _loop_response(*, content, finish_reason="stop", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _loop_tool_call():
    return SimpleNamespace(
        id="call-inject",
        type="function",
        function=SimpleNamespace(name="terminal", arguments="{}"),
    )


def _make_loop_agent(tmp_path: Path) -> AIAgent:
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "test boundary",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        contextlib.redirect_stdout(io.StringIO()),
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai",
            api_mode="chat_completions",
            model="test/model",
            quiet_mode=True,
            max_iterations=4,
            skip_context_files=True,
            skip_memory=True,
            session_db=SessionDB(db_path=tmp_path / "state.db"),
            session_id=f"inject-loop-{uuid.uuid4().hex}",
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.tool_delay = 0
    agent.valid_tool_names = {"terminal"}
    return agent


def _tool_boundary_agent():
    return SimpleNamespace(
        session_id="parent-session",
        _active_turn_id="turn-current",
        _pending_delegation_inject_claims=[],
        _session_messages=[],
    )


def test_tool_boundary_inject_enriches_only_new_tool_result_without_user_tail():
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "review changed the implementation")
    )
    agent = _tool_boundary_agent()
    messages = [
        {"role": "user", "content": "implement"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "old", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "old", "content": "historical result"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "new", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "new", "content": "new result"},
    ]
    agent._session_messages = messages

    assert attach_ready_injects_to_tool_results(agent, messages, num_tool_msgs=1) == 1

    assert [message["role"] for message in messages] == [
        "user", "assistant", "tool", "assistant", "tool"
    ]
    assert messages[2]["content"] == "historical result"
    assert messages[4]["content"].startswith("new result")
    assert "review changed the implementation" in messages[4]["content"]
    assert _event_state(delegation_id, "task:0") == ("pending", 1)


def test_tool_boundary_inject_without_new_tool_carrier_stays_pending():
    delegation_id = _record()
    assert ad.publish_batch_child_completion(delegation_id, 0, _child(0, "late audit"))
    agent = _tool_boundary_agent()
    messages = [{"role": "user", "content": "implement"}]
    agent._session_messages = messages

    assert attach_ready_injects_to_tool_results(agent, messages, num_tool_msgs=0) == 0

    assert messages == [{"role": "user", "content": "implement"}]
    assert _event_state(delegation_id, "task:0") == ("pending", 0)
    assert [event["delivery_event_key"] for event in _queue_contents()] == ["task:0"]


def test_tool_boundary_inject_without_followup_budget_stays_after_turn_pending():
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "arrived at exhausted boundary")
    )
    agent = _tool_boundary_agent()
    agent.max_iterations = 1
    agent._api_call_count = 1
    agent.iteration_budget = SimpleNamespace(remaining=0)
    agent._budget_grace_call = False
    messages = [{"role": "tool", "tool_call_id": "new", "content": "result"}]

    assert attach_ready_injects_to_tool_results(
        agent, messages, num_tool_msgs=1
    ) == 0

    assert messages[0]["content"] == "result"
    assert _event_state(delegation_id, "task:0") == ("pending", 0)
    assert [event["delivery_event_key"] for event in _queue_contents()] == ["task:0"]


def test_unconsumed_tool_boundary_inject_restores_carrier_and_requeues():
    delegation_id = _record()
    assert ad.publish_batch_child_completion(delegation_id, 0, _child(0, "retry later"))
    agent = _tool_boundary_agent()
    messages = [{"role": "tool", "tool_call_id": "new", "content": "original"}]
    agent._session_messages = messages

    assert attach_ready_injects_to_tool_results(agent, messages, num_tool_msgs=1) == 1
    assert release_pending_injects(agent, messages, turn_id="turn-current") == 1

    assert messages == [{"role": "tool", "tool_call_id": "new", "content": "original"}]
    assert _event_state(delegation_id, "task:0") == ("pending", 1)
    assert [event["delivery_event_key"] for event in _queue_contents()] == ["task:0"]


def test_consumed_tool_boundary_inject_acknowledges_without_removing_carrier():
    delegation_id = _record()
    assert ad.publish_batch_child_completion(delegation_id, 0, _child(0, "accepted audit"))
    agent = _tool_boundary_agent()
    messages = [{"role": "tool", "tool_call_id": "new", "content": "original"}]
    agent._session_messages = messages

    assert attach_ready_injects_to_tool_results(agent, messages, num_tool_msgs=1) == 1
    assert acknowledge_pending_injects(agent, turn_id="turn-current") == 1

    assert "accepted audit" in messages[0]["content"]
    assert _event_state(delegation_id, "task:0") == ("delivered", 1)


def test_large_tool_boundary_inject_uses_canonical_spill_budget():
    delegation_id = _record()
    huge_summary = "HUGE_CHILD_REPORT:" + ("x" * 250_000)
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, huge_summary)
    )
    agent = _tool_boundary_agent()
    messages = [
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "tc-large-carrier",
            "content": "original tool result",
        }
    ]
    env = MagicMock()
    env.execute.return_value = {"output": "", "returncode": 0}
    env.get_temp_dir.return_value = ""
    budget = BudgetConfig(
        default_result_size=10_000,
        turn_budget=20_000,
        preview_size=512,
    )

    assert attach_ready_injects_to_tool_results(
        agent,
        messages,
        num_tool_msgs=1,
        storage_env=env,
        budget_config=budget,
    ) == 1

    assert len(messages[0]["content"]) <= budget.default_result_size
    assert PERSISTED_OUTPUT_TAG in messages[0]["content"]
    assert delegation_id in messages[0]["content"]
    assert env.execute.call_count == 1
    # Canonical spill home: the full report is written host-side by
    # maybe_persist_tool_result ($HERMES_HOME/cache/spillover); env.execute
    # only probes sandbox visibility, so the payload rides on disk, not stdin.
    saved_path = re.search(
        r"Full output saved to: (.+)", messages[0]["content"]
    ).group(1)
    persisted_report = Path(saved_path).read_text()
    assert "HUGE_CHILD_REPORT:" in persisted_report
    assert persisted_report.endswith("x" * 1_000)
    assert len(persisted_report) > len(huge_summary)


def test_oversized_inject_without_storage_defers_without_claim_or_truncation():
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "FULL_REPORT_MUST_SURVIVE" + ("x" * 20_000))
    )
    agent = _tool_boundary_agent()
    messages = [
        {"role": "tool", "tool_call_id": "tc-no-env", "content": "ORIGINAL"}
    ]
    budget = BudgetConfig(
        default_result_size=8_000,
        turn_budget=16_000,
        preview_size=512,
    )

    assert attach_ready_injects_to_tool_results(
        agent,
        messages,
        num_tool_msgs=1,
        storage_env=None,
        budget_config=budget,
    ) == 0

    assert messages == [
        {"role": "tool", "tool_call_id": "tc-no-env", "content": "ORIGINAL"}
    ]
    assert _event_state(delegation_id, "task:0") == ("pending", 0)
    assert any(
        event.get("delegation_id") == delegation_id for event in _queue_contents()
    )


def test_attach_exception_after_carrier_mutation_restores_target_and_requeues():
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "FAULT_INJECTED_EVIDENCE")
    )

    class RaisingSessionAgent:
        session_id = "parent-session"
        _active_turn_id = "turn-current"
        _session_messages = []

        @property
        def _pending_delegation_inject_claims(self):
            return []

        @_pending_delegation_inject_claims.setter
        def _pending_delegation_inject_claims(self, _value):
            raise RuntimeError("fault after carrier mutation")

    messages = [{
        "role": "tool",
        "tool_call_id": "tc",
        "content": "ORIGINAL_CONTENT",
        "display_metadata": {"risk": "keep"},
    }]

    with pytest.raises(RuntimeError, match="fault after carrier mutation"):
        attach_ready_injects_to_tool_results(
            RaisingSessionAgent(), messages, num_tool_msgs=1
        )

    assert messages == [{
        "role": "tool",
        "tool_call_id": "tc",
        "content": "ORIGINAL_CONTENT",
        "display_metadata": {"risk": "keep"},
    }]
    assert _event_state(delegation_id, "task:0") == ("pending", 1)
    assert any(
        event.get("delegation_id") == delegation_id for event in _queue_contents()
    )


def test_tool_carrier_preserves_provider_roles_and_existing_prefix_items():
    from agent.anthropic_adapter import convert_messages_to_anthropic
    from agent.codex_responses_adapter import _chat_messages_to_responses_input
    from agent.gemini_native_adapter import _build_gemini_contents
    from agent.prompt_caching import apply_anthropic_cache_control

    prefix = [{"role": "user", "content": "continue the current task"}]
    messages = prefix + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "base result\n[DELEGATION RESULT READY] MATRIX_MARKER",
        },
    ]

    chat = AIAgent._sanitize_api_messages(deepcopy(messages))
    assert chat[: len(prefix)] == prefix
    assert chat[-1]["role"] == "tool"
    assert "MATRIX_MARKER" in chat[-1]["content"]

    _system_prefix, anthropic_prefix = convert_messages_to_anthropic(prefix)
    _system, anthropic = convert_messages_to_anthropic(messages)
    assert anthropic[: len(anthropic_prefix)] == anthropic_prefix
    assert anthropic[-1]["role"] == "user"
    assert anthropic[-1]["content"][0]["type"] == "tool_result"
    assert "MATRIX_MARKER" in anthropic[-1]["content"][0]["content"]

    gemini_prefix, _ = _build_gemini_contents(prefix)
    gemini, _ = _build_gemini_contents(messages)
    assert gemini[: len(gemini_prefix)] == gemini_prefix
    function_response = gemini[-1]["parts"][0]["functionResponse"]
    assert "MATRIX_MARKER" in function_response["response"]["output"]

    responses_prefix = _chat_messages_to_responses_input(prefix)
    responses = _chat_messages_to_responses_input(messages)
    assert responses[: len(responses_prefix)] == responses_prefix
    assert responses[-1]["type"] == "function_call_output"
    assert "MATRIX_MARKER" in responses[-1]["output"]

    cache_prefix = [
        {"role": "system", "content": "stable system prefix"},
        {"role": "user", "content": "stable user prefix"},
    ]
    cached_prefix = apply_anthropic_cache_control(
        deepcopy(cache_prefix), native_anthropic=True
    )
    cached_extended = apply_anthropic_cache_control(
        deepcopy(cache_prefix + messages[1:]), native_anthropic=True
    )
    assert cached_extended[: len(cached_prefix)] == cached_prefix


def test_tool_boundary_opens_only_after_the_complete_result_batch():
    from agent.tool_executor import _completed_tool_batch_size

    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "function": {"name": "first"}},
                {"id": "b", "function": {"name": "second"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "one"},
    ]
    assert _completed_tool_batch_size(messages) == 0

    messages.append({"role": "tool", "tool_call_id": "b", "content": "two"})
    assert _completed_tool_batch_size(messages) == 2

    messages[-1]["_external_input_boundary_checked"] = True
    assert _completed_tool_batch_size(messages) == 0


def test_tool_boundary_flush_failure_restores_carrier_and_requeues_event():
    from agent.tool_executor import _flush_session_db_after_tool_progress

    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "must retry after flush failure")
    )
    agent = _tool_boundary_agent()
    agent._incremental_persistence_failed = False
    agent._flush_messages_to_session_db = lambda _messages: False
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "tc", "function": {"name": "terminal"}}],
        },
        {"role": "tool", "tool_call_id": "tc", "content": "original"},
    ]

    assert not _flush_session_db_after_tool_progress(
        agent, messages, stage="test flush failure"
    )
    assert messages[-1]["content"] == "original"
    assert agent._incremental_persistence_failed is True
    assert not agent._pending_delegation_inject_claims
    assert _event_state(delegation_id, "task:0") == ("pending", 1)
    assert any(
        event.get("delegation_id") == delegation_id for event in _queue_contents()
    )


def test_inject_batch_publishes_first_child_without_waiting_for_sibling():
    gate = threading.Event()
    ready = threading.Event()
    delegation_id = f"deleg_partial_{uuid.uuid4().hex}"

    def runner():
        assert ad.publish_batch_child_completion(
            delegation_id, 0, _child(0, "fast result")
        )
        ready.set()
        gate.wait(timeout=5)
        return {
            "results": [_child(0, "fast result"), _child(1, "slow result")],
            "total_duration_seconds": 0.5,
        }

    dispatched = ad.dispatch_async_delegation_batch(
        goals=["fast", "slow"],
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="agent:main:cli:dm:local",
        parent_session_id="parent",
        parent_turn_id="turn-current",
        runner=runner,
        max_async_children=1,
        delegation_id=delegation_id,
        result_delivery="inject",
    )
    assert dispatched["status"] == "dispatched"
    assert ready.wait(timeout=2)
    first = process_registry.completion_queue.get(timeout=2)
    assert first["delivery_event_key"] == "task:0"
    assert first["results"][0]["summary"] == "fast result"
    assert ad.get_durable_delegation(delegation_id)["state"] == "running"

    gate.set()
    deadline = time.monotonic() + 3
    second = None
    while time.monotonic() < deadline:
        try:
            candidate = process_registry.completion_queue.get(timeout=0.05)
        except Exception:
            continue
        if candidate.get("delivery_event_key") == "task:1":
            second = candidate
            break
    assert second is not None
    assert second["results"][0]["summary"] == "slow result"
    assert "delivery_event_key" not in {
        e.get("delivery_event_key") for e in _queue_contents()
    }
    deadline = time.monotonic() + 2
    while ad.active_count() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ad.get_durable_delegation(delegation_id)["delivery_state"] == "delivered"

    restored_queue = __import__("queue").Queue()
    assert ad.restore_undelivered_completions(restored_queue) == 2
    restored = [restored_queue.get_nowait(), restored_queue.get_nowait()]
    assert {event.get("delivery_event_key") for event in restored} == {
        "task:0",
        "task:1",
    }
    assert all(event.get("delivery_event_key") for event in restored)


def test_after_turn_remains_default_and_coalesces_all_ready_children():
    gate = threading.Event()

    def runner():
        gate.wait(timeout=5)
        return {
            "results": [_child(0, "A"), _child(1, "B")],
            "total_duration_seconds": 0.1,
        }

    dispatched = ad.dispatch_async_delegation_batch(
        goals=["A", "B"], context=None, toolsets=None, role="leaf", model="m",
        session_key="", runner=runner, max_async_children=1,
    )
    assert process_registry.completion_queue.empty()
    gate.set()
    deadline = time.time() + 3
    while process_registry.completion_queue.qsize() < 2 and time.time() < deadline:
        time.sleep(0.01)
    drained = process_registry.drain_notifications(
        owns_event=lambda _event: True,
        skip_poll_observed=False,
    )
    assert len(drained) == 1
    event, text = drained[0]
    assert event["delegation_id"] == dispatched["delegation_id"]
    assert event["result_delivery"] == "after_turn"
    assert event["delivery_event_keys"] == ["task:0", "task:1"]
    assert [r["summary"] for r in event["results"]] == ["A", "B"]
    assert "RESULTS READY" in text
    claim = ad.claim_event_delivery(event, "test-after-turn-default")
    assert claim
    assert ad.complete_event_delivery(event, claim)


def test_batch_finalization_enqueues_ready_set_under_routing_lock(monkeypatch):
    delegation_id = _record(goals=("A", "B"), delivery="after_turn")
    with ad._records_lock:
        record = dict(ad._records[delegation_id])
    children = [_child(0, "A"), _child(1, "B")]

    class TrackingLock:
        def __init__(self):
            self.inner = threading.RLock()
            self.depth = 0

        def __enter__(self):
            self.inner.acquire()
            self.depth += 1
            return self

        def __exit__(self, *_exc):
            self.depth -= 1
            self.inner.release()

    tracking_lock = TrackingLock()

    class GuardedQueue(__import__("queue").Queue):
        def put(self, item, *args, **kwargs):
            assert tracking_lock.depth > 0
            return super().put(item, *args, **kwargs)

    guarded_queue = GuardedQueue()
    monkeypatch.setattr(process_registry, "completion_routing_lock", tracking_lock)
    monkeypatch.setattr(process_registry, "completion_queue", guarded_queue)
    combined = {"results": children, "total_duration_seconds": 1.0}
    parent_event = {
        **record,
        "type": "async_delegation",
        "status": "completed",
        "is_batch": True,
        "results": children,
        "completed_at": time.time(),
    }

    ad._persist_batch_child_finalization(
        record, parent_event, combined, "completed"
    )

    grouped = guarded_queue.get_nowait()
    assert grouped["delivery_event_keys"] == ["task:0", "task:1"]
    assert [result["summary"] for result in grouped["results"]] == ["A", "B"]
    assert guarded_queue.empty()


def test_requeued_after_turn_envelope_absorbs_newly_ready_sibling():
    delegation_id = _record(goals=("A", "B", "C"), delivery="after_turn")
    for index, summary in enumerate(("A", "B", "C")):
        assert ad.publish_batch_child_completion(
            delegation_id, index, _child(index, summary)
        )
    children = [
        process_registry.completion_queue.get_nowait(),
        process_registry.completion_queue.get_nowait(),
        process_registry.completion_queue.get_nowait(),
    ]
    requeued_envelope = ad.coalesce_ready_after_turn_events(children[:2])[0]
    process_registry.completion_queue.put(children[2])

    merged = process_registry.collect_ready_after_turn_siblings(requeued_envelope)

    assert merged["delivery_event_keys"] == ["task:0", "task:1", "task:2"]
    assert [result["summary"] for result in merged["results"]] == ["A", "B", "C"]
    assert process_registry.completion_queue.empty()
    claim = ad.claim_event_delivery(merged, "requeued-group-consumer")
    assert claim
    assert ad.complete_event_delivery(merged, claim)


def test_child_timeout_error_is_injectable_and_durable():
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id,
        0,
        _child(0, "", status="timeout", error="child exceeded 10s"),
    )
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "done"}]
    agent = _tool_boundary_agent()
    assert attach_ready_injects_to_tool_results(
        agent, messages, num_tool_msgs=1, turn_id="turn-current"
    ) == 1
    assert "timeout" in messages[-1]["content"]
    assert "child exceeded 10s" in messages[-1]["content"]
    assert _event_state(delegation_id, "task:0") == ("pending", 1)
    assert acknowledge_pending_injects(agent, turn_id="turn-current") == 1
    assert _event_state(delegation_id, "task:0") == ("delivered", 1)


def test_failed_ack_keeps_ram_claim_for_later_reconciliation(monkeypatch):
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "ack must commit")
    )
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "done"}]
    agent = _tool_boundary_agent()
    assert attach_ready_injects_to_tool_results(
        agent, messages, num_tool_msgs=1, turn_id="turn-current"
    ) == 1

    monkeypatch.setattr(ad, "complete_event_delivery", lambda *_args: False)
    assert acknowledge_pending_injects(agent, turn_id="turn-current") == 0
    assert len(agent._pending_delegation_inject_claims) == 1
    assert _event_state(delegation_id, "task:0") == ("pending", 1)


def test_failed_release_neither_requeues_nor_forgets_claim(monkeypatch):
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "release must commit")
    )
    original = {"role": "tool", "tool_call_id": "tc", "content": "done"}
    messages = [original]
    agent = _tool_boundary_agent()
    assert attach_ready_injects_to_tool_results(
        agent, messages, num_tool_msgs=1, turn_id="turn-current"
    ) == 1

    monkeypatch.setattr(ad, "release_event_delivery", lambda *_args: False)
    assert release_pending_injects(agent, messages, turn_id="turn-current") == 0
    assert messages == [original]
    assert process_registry.completion_queue.empty()
    assert len(agent._pending_delegation_inject_claims) == 1


def test_pending_claim_heartbeat_renews_until_ack(monkeypatch):
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "slow provider")
    )
    renewed = threading.Event()

    def fake_renew(_event, _claim_id):
        renewed.set()
        return True

    monkeypatch.setattr(ad, "renew_event_delivery", fake_renew, raising=False)
    monkeypatch.setattr(
        inject, "_CLAIM_HEARTBEAT_INTERVAL_SECONDS", 0.01, raising=False
    )
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "done"}]
    agent = _tool_boundary_agent()

    assert attach_ready_injects_to_tool_results(
        agent, messages, num_tool_msgs=1, turn_id="turn-current"
    ) == 1
    assert inject.ensure_pending_inject_heartbeat(agent) is True
    assert renewed.wait(timeout=1), "pending inject claim was not renewed"
    assert acknowledge_pending_injects(agent, turn_id="turn-current") == 1
    heartbeat = agent._delegation_inject_claim_heartbeat
    heartbeat["thread"].join(timeout=1)
    assert not heartbeat["thread"].is_alive()


def test_run_conversation_inject_transport_normalize_and_ack(monkeypatch, tmp_path):
    agent = _make_loop_agent(tmp_path)
    cached_system_prompt = deepcopy(getattr(agent, "_cached_system_prompt"))
    requests = []
    responses = [
        _loop_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_loop_tool_call()],
        ),
        _loop_response(content="model consumed LIVE_LOOP_INJECT"),
    ]

    def create(**kwargs):
        requests.append(deepcopy(kwargs["messages"]))
        return responses.pop(0)

    agent.client.chat.completions.create.side_effect = create
    published = {}

    def handle_tool(*_args, **_kwargs):
        delegation_id = _record(
            turn_id=str(agent._active_turn_id),
            parent_session_id=agent.session_id,
        )
        published["delegation_id"] = delegation_id
        assert ad.publish_batch_child_completion(
            delegation_id, 0, _child(0, "LIVE_LOOP_INJECT")
        )
        return "tool boundary complete"

    with (
        patch("run_agent.handle_function_call", side_effect=handle_tool),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("exercise inject lifecycle")

    delegation_id = published["delegation_id"]
    assert result["completed"] is True
    assert len(requests) == 2
    assert requests[1][: len(requests[0])] == requests[0]
    assert [message["role"] for message in requests[1][len(requests[0]):]] == [
        "assistant",
        "tool",
    ]
    assert requests[1][-1]["role"] == "tool"
    assert "display_metadata" not in requests[1][-1]
    assert not any(key.startswith("_") for key in requests[1][-1])
    assert getattr(agent, "_cached_system_prompt") == cached_system_prompt
    assert "LIVE_LOOP_INJECT" in str(requests[1])
    assert "not a new user request" in requests[1][-1]["content"]
    assert _event_state(delegation_id, "task:0") == ("delivered", 1)
    assert not agent._pending_delegation_inject_claims
    heartbeat = agent._delegation_inject_claim_heartbeat
    heartbeat["thread"].join(timeout=1)
    assert not heartbeat["thread"].is_alive()


def test_run_conversation_persists_tool_carrier_before_provider_error(
    monkeypatch, tmp_path
):
    agent = _make_loop_agent(tmp_path)
    calls = {"provider": 0, "compress": 0, "marker_compress": 0}
    provider_error = RuntimeError("provider rejected request")
    provider_error.status_code = 400

    def create(**_kwargs):
        calls["provider"] += 1
        if calls["provider"] == 1:
            return _loop_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_loop_tool_call()],
            )
        raise provider_error

    agent.client.chat.completions.create.side_effect = create
    published = {}

    def handle_tool(*_args, **_kwargs):
        delegation_id = _record(
            turn_id=str(agent._active_turn_id),
            parent_session_id=agent.session_id,
        )
        published["delegation_id"] = delegation_id
        assert ad.publish_batch_child_completion(
            delegation_id, 0, _child(0, "COPY_THEN_RELEASE")
        )
        agent.compression_enabled = True
        agent.context_compressor.should_compress = lambda _tokens: True
        return "tool boundary complete"

    def copy_compress(messages, system_message, **_kwargs):
        calls["compress"] += 1
        if "COPY_THEN_RELEASE" not in str(messages):
            return messages, system_message
        calls["marker_compress"] += 1
        heartbeat = agent._delegation_inject_claim_heartbeat
        assert heartbeat["stop"].is_set()
        agent.compression_enabled = False
        return deepcopy(messages), system_message

    def persist_with_durable_carrier(messages, *_args, **_kwargs):
        if any(message.get("role") == "tool" for message in messages):
            assert "COPY_THEN_RELEASE" in str(messages)

    with (
        patch("run_agent.handle_function_call", side_effect=handle_tool),
        patch.object(agent, "_compress_context", side_effect=copy_compress),
        patch.object(
            agent,
            "_persist_session",
            side_effect=persist_with_durable_carrier,
        ),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("exercise rollback lifecycle")

    delegation_id = published["delegation_id"]
    assert result["failed"] is True
    assert calls["compress"] >= 1
    assert calls["marker_compress"] == 1
    assert "COPY_THEN_RELEASE" in str(agent._session_messages)
    assert _event_state(delegation_id, "task:0") == ("delivered", 1)
    assert not any(
        event.get("delegation_id") == delegation_id for event in _queue_contents()
    )
    assert not agent._pending_delegation_inject_claims
    heartbeat = agent._delegation_inject_claim_heartbeat
    heartbeat["thread"].join(timeout=1)
    assert not heartbeat["thread"].is_alive()


def test_persisted_bounded_carrier_survives_production_compressor_payload(
    tmp_path,
):
    from agent.context_compressor import ContextCompressor
    from agent.tool_executor import _flush_session_db_after_tool_progress

    agent = _make_loop_agent(tmp_path)
    agent._active_turn_id = "turn-current"
    agent._pending_delegation_inject_claims = []
    agent._incremental_persistence_failed = False
    delegation_id = _record(
        turn_id="turn-current",
        parent_session_id=agent.session_id,
    )
    huge_summary = "COMPRESSOR_CARRIER_EVIDENCE:" + ("z" * 250_000)
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, huge_summary)
    )

    messages = [{"role": "system", "content": "system"}]
    for index in range(8):
        messages.extend(
            [
                {"role": "user", "content": f"old user {index}"},
                {"role": "assistant", "content": f"old assistant {index}"},
            ]
        )
    messages.extend(
        [
            {"role": "user", "content": "current task"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc-compress",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "terminal",
                "tool_call_id": "tc-compress",
                "content": "tool boundary complete",
            },
        ]
    )
    env = MagicMock()
    env.execute.return_value = {"output": "", "returncode": 0}
    env.get_temp_dir.return_value = ""
    budget = BudgetConfig(
        default_result_size=10_000,
        turn_budget=20_000,
        preview_size=512,
    )

    assert _flush_session_db_after_tool_progress(
        agent,
        messages,
        stage="persist bounded carrier before compression",
        storage_env=env,
        budget_config=budget,
    )
    assert _event_state(delegation_id, "task:0") == ("delivered", 1)

    durable = agent._session_db.get_messages_as_conversation(agent.session_id)
    durable_str = str(durable)
    assert PERSISTED_OUTPUT_TAG in durable_str
    assert delegation_id in durable_str
    # Carrier evidence survives in the canonical host-side spillover file.
    carrier_content = next(
        message.get("content", "")
        for message in durable
        if isinstance(message, dict)
        and isinstance(message.get("content"), str)
        and PERSISTED_OUTPUT_TAG in message["content"]
    )
    saved_path = re.search(r"Full output saved to: (.+)", carrier_content).group(1)
    assert "COMPRESSOR_CARRIER_EVIDENCE:" in Path(saved_path).read_text()

    compressor = ContextCompressor(
        model="test/model",
        provider="openai",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        config_context_length=100_000,
        protect_first_n=3,
        protect_last_n=3,
        summary_target_ratio=0.10,
        quiet_mode=True,
    )
    compressor.threshold_tokens = 2_000
    compressor.tail_token_budget = 1_000
    compressor._generate_summary = lambda *_args, **_kwargs: "compressed old context"

    compressed = compressor.compress(durable, current_tokens=80_000, force=True)
    provider_payload = AIAgent._sanitize_api_messages(deepcopy(compressed))

    assert compressor._last_compression_made_progress is True
    assert delegation_id in str(provider_payload)
    assert PERSISTED_OUTPUT_TAG in str(provider_payload)
    assert len(provider_payload[-1]["content"]) <= budget.default_result_size


def test_same_turn_claim_conflict_defers_pending_event(monkeypatch):
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "leased elsewhere")
    )
    deferred = []
    monkeypatch.setattr(ad, "claim_event_delivery", lambda *_args: None)
    monkeypatch.setattr(
        process_registry,
        "defer_unclaimed_delivery",
        lambda evt: deferred.append(evt) or True,
    )

    messages = [{"role": "tool", "tool_call_id": "tc", "content": "done"}]
    assert attach_ready_injects_to_tool_results(
        _tool_boundary_agent(), messages, num_tool_msgs=1, turn_id="turn-current"
    ) == 0

    assert len(deferred) == 1
    assert deferred[0]["delegation_id"] == delegation_id
    assert deferred[0]["delivery_event_key"] == "task:0"
    assert len(messages) == 1


def test_formatter_failure_does_not_consume_delivery_attempts(monkeypatch):
    delegation_id = _record()
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "format me")
    )

    def broken_formatter(_event):
        raise ValueError("broken spill")

    monkeypatch.setattr(
        "tools.process_registry._format_async_delegation", broken_formatter
    )
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "done"}]
    for _ in range(ad._MAX_DELIVERY_ATTEMPTS + 2):
        assert attach_ready_injects_to_tool_results(
            _tool_boundary_agent(), messages, num_tool_msgs=1, turn_id="turn-current"
        ) == 0

    assert _event_state(delegation_id, "task:0") == ("pending", 0)
    assert len(_queue_contents()) == 1


def test_quick_restart_requeues_after_live_delivery_lease_expires(
    tmp_path, monkeypatch
):
    """A restored live-claimed row wakes without another process restart."""
    import queue as queue_module

    import tools.process_registry as registry_mod

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(ad, "_DELIVERY_CLAIM_LEASE_SECONDS", 0.15)
    monkeypatch.setattr(
        registry_mod, "CHECKPOINT_PATH", tmp_path / "processes.json"
    )

    delegation_id = _record(goals=("survive quick restart",))
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "restored after lease")
    )
    original = process_registry.completion_queue.get_nowait()
    old_claim = ad.claim_event_delivery(original, "old-process")
    assert old_claim

    restarted = registry_mod.ProcessRegistry()
    restored = restarted.completion_queue.get_nowait()
    assert restored["restored"] is True
    assert ad.claim_event_delivery(restored, "new-process") is None

    assert restarted.defer_unclaimed_delivery(restored) is True
    with pytest.raises(queue_module.Empty):
        restarted.completion_queue.get_nowait()

    woke = restarted.completion_queue.get(timeout=1)
    assert woke["delegation_id"] == delegation_id
    assert woke["delivery_event_key"] == "task:0"
    new_claim = ad.claim_event_delivery(woke, "new-process")
    assert new_claim
    assert ad.complete_event_delivery(woke, new_claim)
    assert _event_state(delegation_id, "task:0") == ("delivered", 2)
    assert restarted.defer_unclaimed_delivery(woke) is False
    with pytest.raises(queue_module.Empty):
        restarted.completion_queue.get_nowait()


def test_live_lease_retry_prunes_terminal_group_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(ad, "_DELIVERY_CLAIM_LEASE_SECONDS", 0.15)

    delegation_id = _record(
        goals=("already delivered", "still leased"), delivery="after_turn"
    )
    assert ad.publish_batch_child_completion(
        delegation_id, 0, _child(0, "delivered result")
    )
    assert ad.publish_batch_child_completion(
        delegation_id, 1, _child(1, "leased result")
    )
    children = [
        process_registry.completion_queue.get_nowait(),
        process_registry.completion_queue.get_nowait(),
    ]
    grouped = ad.coalesce_ready_after_turn_events(children)[0]
    by_key = {event["delivery_event_key"]: event for event in children}

    delivered_claim = ad.claim_event_delivery(by_key["task:0"], "first-consumer")
    assert delivered_claim
    assert ad.complete_event_delivery(by_key["task:0"], delivered_claim)
    live_claim = ad.claim_event_delivery(by_key["task:1"], "old-process")
    assert live_claim
    assert ad.claim_event_delivery(grouped, "new-process") is None

    assert process_registry.defer_unclaimed_delivery(grouped)
    assert grouped["delivery_event_keys"] == ["task:1"]
    assert [result["task_index"] for result in grouped["results"]] == [1]

    woke = process_registry.completion_queue.get(timeout=1)
    assert woke["delivery_event_keys"] == ["task:1"]
    retry_claim = ad.claim_event_delivery(woke, "new-process")
    assert retry_claim
    assert ad.complete_event_delivery(woke, retry_claim)
    assert _event_state(delegation_id, "task:0") == ("delivered", 1)
    assert _event_state(delegation_id, "task:1") == ("delivered", 2)


def test_model_schema_defaults_after_turn_and_dispatch_forwards_explicit_mode(monkeypatch):
    delivery_schema = delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"][
        "result_delivery"
    ]
    assert delivery_schema["enum"] == ["inject", "after_turn"]
    assert delivery_schema["default"] == "after_turn"

    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(delegate_tool, "delegate_task", fake_delegate_task)
    from run_agent import AIAgent

    result = AIAgent._dispatch_delegate_task(
        SimpleNamespace(_delegate_depth=0),
        {"goal": "audit", "result_delivery": "inject"},
    )
    assert result == "ok"
    assert captured["background"] is True
    assert captured["result_delivery"] == "inject"

    # The registry fallback is a distinct model-facing dispatch path. It must
    # preserve the same delivery choice if the run_agent intercept is bypassed.
    captured.clear()
    from tools.registry import registry

    entry = registry.get_entry("delegate_task")
    assert entry is not None
    result = entry.handler(
        {"goal": "audit", "result_delivery": "inject"},
        parent_agent=SimpleNamespace(_delegate_depth=0),
    )
    assert result == "ok"
    assert captured["background"] is True
    assert captured["result_delivery"] == "inject"
