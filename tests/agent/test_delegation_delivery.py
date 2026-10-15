"""Semantic oracle from #76230, adapted to #104299 completion-unit identities.

Old coalescer/second-ledger tests remain on the reference PR. Production unit
finalization, grouping, persistence and restart paths are exercised here.
"""

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


def _complete_unit(delegation_id: str, child: dict) -> bool:
    """Complete an existing unit through the production finalizer, not a child-row shim."""
    result = {"results": [child], "total_duration_seconds": child["duration_seconds"]}
    ad._finalize(delegation_id, result, ad._batch_status(result))
    return _event_state(delegation_id) == ("pending", 0)


def _queue_contents():
    items = []
    while not process_registry.completion_queue.empty():
        items.append(process_registry.completion_queue.get_nowait())
    for item in items:
        process_registry.completion_queue.put(item)
    return items


def _event_state(delegation_id: str):
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT delivery_state, delivery_attempts FROM async_delegations "
            "WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()


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
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
    assert _complete_unit(delegation_id, _child(0, "review changed the implementation")
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
    assert _event_state(delegation_id) == ("pending", 1)


def test_tool_boundary_inject_without_new_tool_carrier_stays_pending():
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "late audit"))
    agent = _tool_boundary_agent()
    messages = [{"role": "user", "content": "implement"}]
    agent._session_messages = messages

    assert attach_ready_injects_to_tool_results(agent, messages, num_tool_msgs=0) == 0

    assert messages == [{"role": "user", "content": "implement"}]
    assert _event_state(delegation_id) == ("pending", 0)
    assert [event["delegation_id"] for event in _queue_contents()] == [delegation_id]


def test_tool_boundary_inject_without_followup_budget_stays_after_turn_pending():
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "arrived at exhausted boundary")
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
    assert _event_state(delegation_id) == ("pending", 0)
    assert [event["delegation_id"] for event in _queue_contents()] == [delegation_id]


def test_unconsumed_tool_boundary_inject_restores_carrier_and_requeues():
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "retry later"))
    agent = _tool_boundary_agent()
    messages = [{"role": "tool", "tool_call_id": "new", "content": "original"}]
    agent._session_messages = messages

    assert attach_ready_injects_to_tool_results(agent, messages, num_tool_msgs=1) == 1
    assert release_pending_injects(agent, messages, turn_id="turn-current") == 1

    assert messages == [{"role": "tool", "tool_call_id": "new", "content": "original"}]
    assert _event_state(delegation_id) == ("pending", 1)
    assert [event["delegation_id"] for event in _queue_contents()] == [delegation_id]


def test_consumed_tool_boundary_inject_acknowledges_without_removing_carrier():
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "accepted audit"))
    agent = _tool_boundary_agent()
    messages = [{"role": "tool", "tool_call_id": "new", "content": "original"}]
    agent._session_messages = messages

    assert attach_ready_injects_to_tool_results(agent, messages, num_tool_msgs=1) == 1
    messages[-1]["_db_persisted"] = True
    assert acknowledge_pending_injects(agent, turn_id="turn-current") == 1

    assert "accepted audit" in messages[0]["content"]
    assert _event_state(delegation_id) == ("delivered", 1)


def test_large_tool_boundary_inject_uses_canonical_spill_budget():
    delegation_id = _record()
    huge_summary = "HUGE_CHILD_REPORT:" + ("x" * 250_000)
    assert _complete_unit(delegation_id, _child(0, huge_summary)
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
    assert _complete_unit(delegation_id, _child(0, "FULL_REPORT_MUST_SURVIVE" + ("x" * 20_000))
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
    assert _event_state(delegation_id) == ("pending", 0)
    assert any(
        event.get("delegation_id") == delegation_id for event in _queue_contents()
    )


def test_attach_exception_after_carrier_mutation_restores_target_and_requeues():
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "FAULT_INJECTED_EVIDENCE")
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
    assert _event_state(delegation_id) == ("pending", 1)
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
    assert _complete_unit(delegation_id, _child(0, "must retry after flush failure")
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
    assert _event_state(delegation_id) == ("pending", 1)
    assert any(
        event.get("delegation_id") == delegation_id for event in _queue_contents()
    )


def test_child_timeout_error_is_injectable_and_durable():
    delegation_id = _record()
    assert _complete_unit(
        delegation_id,
        _child(0, "", status="timeout", error="child exceeded 10s"),
    )
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "done"}]
    agent = _tool_boundary_agent()
    assert attach_ready_injects_to_tool_results(
        agent, messages, num_tool_msgs=1, turn_id="turn-current"
    ) == 1
    assert "timeout" in messages[-1]["content"]
    assert "child exceeded 10s" in messages[-1]["content"]
    assert _event_state(delegation_id) == ("pending", 1)
    messages[-1]["_db_persisted"] = True
    assert acknowledge_pending_injects(agent, turn_id="turn-current") == 1
    assert _event_state(delegation_id) == ("delivered", 1)


def test_failed_ack_keeps_ram_claim_for_later_reconciliation(monkeypatch):
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "ack must commit")
    )
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "done"}]
    agent = _tool_boundary_agent()
    assert attach_ready_injects_to_tool_results(
        agent, messages, num_tool_msgs=1, turn_id="turn-current"
    ) == 1

    messages[-1]["_db_persisted"] = True
    monkeypatch.setattr(ad, "complete_event_delivery", lambda *_args: False)
    assert acknowledge_pending_injects(agent, turn_id="turn-current") == 0
    assert len(agent._pending_delegation_inject_claims) == 1
    assert _event_state(delegation_id) == ("pending", 1)


def test_failed_release_neither_requeues_nor_forgets_claim(monkeypatch):
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "release must commit")
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
    assert _complete_unit(delegation_id, _child(0, "slow provider")
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
    messages[-1]["_db_persisted"] = True
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
        assert _complete_unit(delegation_id, _child(0, "LIVE_LOOP_INJECT")
        )
        return "tool boundary complete"

    with (
        patch("model_tools.handle_function_call", side_effect=handle_tool),
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
    assert _event_state(delegation_id) == ("delivered", 1)
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
        assert _complete_unit(delegation_id, _child(0, "COPY_THEN_RELEASE")
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
        patch("model_tools.handle_function_call", side_effect=handle_tool),
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
    assert _event_state(delegation_id) == ("delivered", 1)
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
    assert _complete_unit(delegation_id, _child(0, huge_summary)
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
    assert _event_state(delegation_id) == ("delivered", 1)

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


def test_formatter_failure_does_not_consume_delivery_attempts(monkeypatch):
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "format me")
    )

    def broken_formatter(_event):
        raise ValueError("broken spill")

    monkeypatch.setattr(
        "tools.process_registry_notifications._format_async_delegation", broken_formatter
    )
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "done"}]
    for _ in range(ad._MAX_DELIVERY_ATTEMPTS + 2):
        assert attach_ready_injects_to_tool_results(
            _tool_boundary_agent(), messages, num_tool_msgs=1, turn_id="turn-current"
        ) == 0

    assert _event_state(delegation_id) == ("pending", 0)
    assert len(_queue_contents()) == 1


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


@pytest.mark.parametrize("flush_result", [None, True])
def test_flush_without_durable_marker_does_not_acknowledge(flush_result):
    from agent.tool_executor import _flush_session_db_after_tool_progress

    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "must remain pending"))
    agent = _tool_boundary_agent()
    agent._flush_messages_to_session_db = lambda _messages: flush_result
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "tc"}]},
        {"role": "tool", "tool_call_id": "tc", "content": "original"},
    ]
    assert _flush_session_db_after_tool_progress(agent, messages, stage="no durable receipt")
    assert messages[-1]["content"] == "original"
    assert _event_state(delegation_id) == ("pending", 1)
    assert not agent._pending_delegation_inject_claims
    assert len(_queue_contents()) == 1


def test_ack_requires_actual_transcript_marker():
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "not persisted"))
    agent = _tool_boundary_agent()
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "original"}]
    assert attach_ready_injects_to_tool_results(agent, messages, 1) == 1
    assert acknowledge_pending_injects(agent) == 0
    assert _event_state(delegation_id) == ("pending", 1)
    assert release_pending_injects(agent, messages) == 1


def test_persisted_or_invalid_tool_tail_is_never_a_carrier():
    from agent.tool_executor import _completed_tool_batch_size

    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "new evidence"))
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "tc"}]},
        {"role": "tool", "tool_call_id": "tc", "content": "historical", "_db_persisted": True},
    ]
    before = deepcopy(messages)
    assert _completed_tool_batch_size(messages) == 0
    assert attach_ready_injects_to_tool_results(_tool_boundary_agent(), messages, 1) == 0
    assert messages == before
    assert _event_state(delegation_id) == ("pending", 0)
    malformed = [
        {"role": "assistant", "tool_calls": [{"id": "tc"}, {"id": "tc"}]},
        {"role": "tool", "tool_call_id": "tc", "content": "first"},
        {"role": "tool", "tool_call_id": "tc", "content": "duplicate"},
    ]
    assert _completed_tool_batch_size(malformed) == 0


def test_batch_siblings_leave_no_space_so_carrier_is_deferred():
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "report " * 300))
    messages = [
        {"role": "tool", "tool_call_id": "first", "content": "s" * 4_900},
        {"role": "tool", "tool_call_id": "last", "content": "last"},
    ]
    before = deepcopy(messages)
    budget = BudgetConfig(default_result_size=6_000, turn_budget=6_000)
    assert attach_ready_injects_to_tool_results(
        _tool_boundary_agent(), messages, 2, budget_config=budget,
    ) == 0
    assert messages == before
    assert _event_state(delegation_id) == ("pending", 0)
    assert len(_queue_contents()) == 1


def test_spilled_carrier_survives_subsequent_aggregate_budgeting():
    from agent.tool_executor import _finalize_tool_batch

    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "report " * 50_000))
    agent = _tool_boundary_agent()
    agent._apply_pending_steer_to_tool_results = lambda *_args: None
    messages = [
        {"role": "tool", "tool_call_id": "first", "content": "s" * 9_000},
        {"role": "tool", "tool_call_id": "last", "content": "last"},
    ]
    budget = BudgetConfig(default_result_size=10_000, turn_budget=10_000, preview_size=512)
    env = MagicMock()
    env.execute.return_value = {"output": "", "returncode": 0}
    env.get_temp_dir.return_value = ""
    assert attach_ready_injects_to_tool_results(agent, messages, 2, storage_env=env, budget_config=budget) == 1
    assert sum(len(m["content"]) for m in messages) <= budget.turn_budget
    before = deepcopy(messages)
    with patch("agent.tool_executor.get_active_env", return_value=env):
        _finalize_tool_batch(agent, messages, "test", 2, budget)
    assert messages == before
    assert PERSISTED_OUTPUT_TAG in messages[-1]["content"]
    assert release_pending_injects(agent, messages) == 1


def test_spill_exception_requeues_unclaimed_event(monkeypatch):
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "spill failure"))
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "original"}]

    def broken_spill(*_args, **_kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(inject, "_bounded_carrier_text", broken_spill)
    assert attach_ready_injects_to_tool_results(_tool_boundary_agent(), messages, 1) == 0
    assert messages[-1]["content"] == "original"
    assert _event_state(delegation_id) == ("pending", 0)
    assert len(_queue_contents()) == 1


def test_heartbeat_preserves_parent_context(monkeypatch):
    from contextvars import ContextVar

    profile = ContextVar("inject_test_profile", default="wrong-profile")
    profile.set("active-profile")
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "profile-scoped lease"))
    seen = []
    renewed = threading.Event()

    def renew(*_args):
        seen.append(profile.get())
        renewed.set()
        return True

    monkeypatch.setattr(ad, "renew_event_delivery", renew)
    monkeypatch.setattr(inject, "_CLAIM_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    agent = _tool_boundary_agent()
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "original"}]
    assert attach_ready_injects_to_tool_results(agent, messages, 1) == 1
    assert renewed.wait(timeout=1)
    assert seen and set(seen) == {"active-profile"}
    assert release_pending_injects(agent, messages) == 1
    agent._delegation_inject_claim_heartbeat["thread"].join(timeout=1)


def test_ack_exception_does_not_reclassify_successful_transcript_flush(monkeypatch):
    from agent.tool_executor import _flush_session_db_after_tool_progress

    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "durably stored"))
    agent = _tool_boundary_agent()
    agent._incremental_persistence_failed = False
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "tc"}]},
        {"role": "tool", "tool_call_id": "tc", "content": "original"},
    ]

    def flush(messages):
        for message in messages:
            message["_db_persisted"] = True
        return True

    agent._flush_messages_to_session_db = flush
    with patch.object(ad, "complete_event_delivery", side_effect=OSError("ack temporarily unavailable")):
        assert _flush_session_db_after_tool_progress(agent, messages, stage="ack retry")
    assert not agent._incremental_persistence_failed
    assert "durably stored" in messages[-1]["content"]
    assert len(agent._pending_delegation_inject_claims) == 1
    assert acknowledge_pending_injects(agent) == 1


def test_restart_does_not_redeliver_committed_carrier_without_ack(tmp_path, monkeypatch):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "receipt.db")
    db = SessionDB(db_path=tmp_path / "receipt.db")
    db.create_session("parent-session", source="cli", model="test")
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "receipt survives crash"))
    agent = _tool_boundary_agent()
    messages = [{"role": "tool", "tool_call_id": "tc", "tool_name": "terminal", "content": "original"}]
    assert attach_ready_injects_to_tool_results(agent, messages, 1) == 1
    event = agent._pending_delegation_inject_claims[0]["event"]
    db.append_message("parent-session", "tool", messages[0]["content"], tool_call_id="tc",
                      tool_name="terminal", display_metadata=messages[0]["display_metadata"])
    # Simulate a hard exit after the transcript transaction but BEFORE ack.
    agent._delegation_inject_claim_heartbeat["stop"].set()
    agent._delegation_inject_claim_heartbeat["thread"].join(timeout=1)
    agent._pending_delegation_inject_claims = []
    ad._reset_for_tests()
    assert _event_state(delegation_id) == ("pending", 1)
    assert ad.claim_event_delivery(event, "restarted-after-turn") is None
    assert _event_state(delegation_id) == ("delivered", 1)
    assert not ad.claim_completion_delivery(event["delegation_id"], "direct-gateway-claim")
    db.close()


@pytest.mark.parametrize("consumer", ["poller", "post_turn"])
def test_reconciled_carrier_claim_rejection_leaves_tui_runnable(
    tmp_path, monkeypatch, consumer,
):
    """A restart receipt rejection must not reserve ``session['running']``.

    Exercise both TUI consumers named in the review: the background notification
    poller and the immediate post-turn safety drain.  The durable row is still
    ``pending`` when queued, but the committed tool transcript is authoritative;
    its claim reconciles to ``delivered`` and no synthetic turn is dispatched.
    """
    from tui_gateway import server

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "receipt.db")
    db = SessionDB(db_path=tmp_path / "receipt.db")
    db.create_session("parent-session", source="tui", model="test")
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "receipt already committed"))
    agent = _tool_boundary_agent()
    messages = [{
        "role": "tool", "tool_call_id": "tc", "tool_name": "terminal",
        "content": "original",
    }]
    assert attach_ready_injects_to_tool_results(agent, messages, 1) == 1
    event = agent._pending_delegation_inject_claims[0]["event"]
    db.append_message(
        "parent-session", "tool", messages[0]["content"], tool_call_id="tc",
        tool_name="terminal", display_metadata=messages[0]["display_metadata"],
    )

    # Hard exit after transcript commit and before durable acknowledgement.
    heartbeat = agent._delegation_inject_claim_heartbeat
    heartbeat["stop"].set()
    heartbeat["thread"].join(timeout=1)
    agent._pending_delegation_inject_claims = []
    ad._reset_for_tests()
    process_registry.completion_queue.put(event)

    sid = f"review-{consumer}"
    session = {
        "session_key": event["session_key"],
        "history_lock": threading.Lock(),
        "running": False,
        "_finalized": False,
    }
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server, "_run_prompt_submit",
        lambda *_args, **_kwargs: pytest.fail("reconciled carrier was redelivered"),
    )
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        server, "_dispatch_followup_turn",
        lambda *_args, **_kwargs: pytest.fail("reconciled carrier started a follow-up"),
    )
    try:
        if consumer == "poller":
            stop = threading.Event()
            stop.set()  # one bounded shutdown drain
            server._notification_poller_loop(stop, sid, session)
        else:
            server._run_post_turn_followups("rid", sid, session, {}, None)

        assert session["running"] is False
        assert process_registry.completion_queue.empty()
        assert _event_state(delegation_id) == ("delivered", 1)
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_tui_busy_dequeue_cannot_hide_ready_inject_from_tool_boundary(monkeypatch):
    """A busy TUI requeues under routing ownership before the active drain proceeds."""
    from tui_gateway import server

    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "tui race result"))
    event = _queue_contents()[0]
    agent = _tool_boundary_agent()
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "foreground result"}]
    session = {
        "session_key": event["session_key"],
        "history_lock": threading.Lock(),
        "running": True,
        "_finalized": False,
    }

    tui_dequeued = threading.Event()
    release_tui = threading.Event()
    original_owns = server._session_owns_notification_event

    def paused_owns(sid, candidate_session, candidate):
        tui_dequeued.set()
        assert release_tui.wait(timeout=2)
        return original_owns(sid, candidate_session, candidate)

    monkeypatch.setattr(server, "_session_owns_notification_event", paused_owns)
    reserved = []

    def reserve():
        with process_registry.completion_routing_lock:
            reserved.append(server._notif_reserve_event(
                "tui-race", session, process_registry))

    tui_thread = threading.Thread(target=reserve)
    tui_thread.start()
    assert tui_dequeued.wait(timeout=2)

    attached: list[int] = []
    attach_done = threading.Event()

    def attach():
        attached.append(attach_ready_injects_to_tool_results(agent, messages, 1))
        attach_done.set()

    attach_thread = threading.Thread(target=attach)
    attach_thread.start()
    assert not attach_done.wait(timeout=0.05)

    release_tui.set()
    tui_thread.join(timeout=2)
    attach_thread.join(timeout=2)
    assert not tui_thread.is_alive()
    assert not attach_thread.is_alive()
    assert reserved and reserved[0][1] == "busy"
    assert attached == [1]
    assert "tui race result" in messages[0]["content"]

    messages[0]["_db_persisted"] = True
    assert acknowledge_pending_injects(agent) == 1
    agent._delegation_inject_claim_heartbeat["thread"].join(timeout=1)


@pytest.mark.parametrize("wrong_field", ["session", "role", "identity", "kind", "malformed"])
def test_unrelated_transcript_metadata_cannot_suppress_delivery(tmp_path, monkeypatch, wrong_field):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "receipt.db")
    db = SessionDB(db_path=tmp_path / "receipt.db")
    db.create_session("parent-session", source="cli", model="test")
    db.create_session("different-session", source="cli", model="test")
    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "must deliver"))
    event = process_registry.completion_queue.get_nowait()
    metadata = {"delegation_delivery": "tool_boundary", "delegation_event_ids": [delegation_id]}
    if wrong_field == "identity":
        metadata["delegation_event_ids"] = [f"{delegation_id}-different-unit"]
    elif wrong_field == "kind":
        metadata["delegation_delivery"] = "not_a_receipt"
    sid = "different-session" if wrong_field == "session" else "parent-session"
    role = "assistant" if wrong_field == "role" else "tool"
    db.append_message(sid, role, "unrelated row", display_metadata=metadata)
    if wrong_field == "malformed":
        with ad._DB_LOCK, ad._transaction() as conn:
            conn.execute("UPDATE messages SET display_metadata=?", (f"broken metadata {delegation_id}",))
    claim = ad.claim_event_delivery(event, "actual-consumer")
    assert claim
    assert _event_state(delegation_id) == ("pending", 1)
    assert ad.complete_event_delivery(event, claim)
    db.close()


@pytest.mark.parametrize("parent_id", ["", "different-parent"])
def test_inject_requires_positive_parent_session_ownership(parent_id):
    delegation_id = _record(parent_session_id=parent_id)
    assert _complete_unit(delegation_id, _child(0, "not for this session"))
    messages = [{"role": "tool", "tool_call_id": "tc", "content": "original"}]
    assert attach_ready_injects_to_tool_results(_tool_boundary_agent(), messages, 1) == 0
    assert messages[-1]["content"] == "original"
    assert _event_state(delegation_id) == ("pending", 0)
    assert len(_queue_contents()) == 1


@pytest.mark.parametrize("disabled_state", ["explicit", "missing_db"])
def test_disabled_persistence_never_burns_delivery_attempts(disabled_state):
    from agent.tool_executor import _flush_session_db_after_tool_progress

    delegation_id = _record()
    assert _complete_unit(delegation_id, _child(0, "after-turn fallback"))
    agent = _tool_boundary_agent()
    if disabled_state == "explicit":
        agent._persist_disabled = True
    else:
        agent._session_db = None
    agent._flush_messages_to_session_db = lambda _messages: None
    for index in range(ad._MAX_DELIVERY_ATTEMPTS + 2):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": f"tc-{index}"}]},
            {"role": "tool", "tool_call_id": f"tc-{index}", "content": "original"},
        ]
        assert _flush_session_db_after_tool_progress(agent, messages, stage="persistence disabled")
        assert messages[-1]["content"] == "original"
    assert _event_state(delegation_id) == ("pending", 0)
    assert len(_queue_contents()) == 1


# #104299 integration: preserve its real completion-unit partition/finalizer.
def _gated_unit_call(monkeypatch, gates, *, delivery="inject"):
    import json

    parent = MagicMock()
    parent._delegate_depth = 0
    parent.session_id = "parent-session"
    parent._active_turn_id = "turn-current"
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = None

    def build(**kw):
        child = MagicMock()
        child._delegate_role = "leaf"
        child._subagent_id = f"unit-child-{kw['task_index']}"
        return child

    def run(task_index, goal, **kw):
        assert gates[task_index].wait(timeout=10), "test did not release child"
        return _child(task_index, f"completed: {goal}")

    creds = dict(model="child-model", provider=None, base_url=None, api_key=None,
                 api_mode=None, command=None, args=None)
    monkeypatch.setattr(delegate_tool, "_build_child_agent", build)
    monkeypatch.setattr(delegate_tool, "_run_single_child", run)
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(delegate_tool, "_get_max_async_children", lambda: 3)
    monkeypatch.setattr(delegate_tool, "_get_max_concurrent_children", lambda: 3)
    return json.loads(delegate_tool.delegate_task(
        tasks=[
            {"goal": "compare implementation alpha in detail", "group": "compare"},
            {"goal": "compare implementation beta in detail", "group": "compare"},
            {"goal": "audit the independent security boundary"},
        ], background=True, parent_agent=parent, result_delivery=delivery,
    ))


def test_inject_uses_completed_units_without_splitting_groups_or_capacity(monkeypatch, tmp_path):
    import json
    import queue

    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "units.db")
    db = SessionDB(db_path=tmp_path / "units.db")
    db.create_session("parent-session", source="cli", model="test")
    gates = [threading.Event() for _ in range(3)]
    agent = _tool_boundary_agent()
    messages = []

    def consume(event):
        nonlocal messages
        process_registry.completion_queue.put(event)
        messages = [{"role": "tool", "tool_call_id": "boundary", "content": "fresh tool result"}]
        assert attach_ready_injects_to_tool_results(agent, messages, 1) == 1
        metadata = messages[-1]["display_metadata"]
        assert metadata["delegation_event_ids"] == [event["delegation_id"]]
        db.append_message("parent-session", "tool", messages[-1]["content"], display_metadata=metadata)
        messages[-1]["_db_persisted"] = True
        assert acknowledge_pending_injects(agent) == 1
        assert _event_state(event["delegation_id"]) == ("delivered", 1)
        return messages[-1]["content"]

    try:
        handle = _gated_unit_call(monkeypatch, gates)
        assert handle["status"] == "dispatched"
        units = {tuple(u["task_indexes"]): u for u in handle["units"]}
        assert set(units) == {(0, 1), (2,)}
        assert units[(0, 1)]["group"] == "compare"
        assert units[(2,)]["group"] is None
        assert ad.active_count() == 2
        assert ad.active_task_count() == 3
        with ad._records_lock:
            records = [dict(ad._records[u["delegation_id"]]) for u in units.values()]
        assert len({r["slot_key"] for r in records}) == 1
        with ad._DB_LOCK, ad._transaction() as conn:
            rows = conn.execute("SELECT delegation_id, task_json FROM async_delegations").fetchall()
        assert {r[0] for r in rows} == {u["delegation_id"] for u in units.values()}
        for _, payload in rows:
            task = json.loads(payload)
            assert task["result_delivery"] == "inject"
            assert task["parent_turn_id"] == "turn-current"

        rejected = ad.dispatch_async_delegation(
            goal="another call", context=None, toolsets=None, role="leaf", model="test", session_key="",
            runner=lambda: {"status": "completed", "summary": "must not run"}, max_async_children=1,
        )
        assert rejected["status"] == "rejected"

        gates[2].set()
        first = process_registry.completion_queue.get(timeout=5)
        assert first["delegation_id"] == units[(2,)]["delegation_id"]
        assert [r["task_index"] for r in first["results"]] == [2]
        assert first["parent_session_id"] == "parent-session"
        assert first["parent_turn_id"] == "turn-current"
        assert first["result_delivery"] == "inject"
        assert "independent security" in consume(first)

        gates[0].set()
        with pytest.raises(queue.Empty):
            process_registry.completion_queue.get(timeout=0.15)
        gates[1].set()
        grouped = process_registry.completion_queue.get(timeout=5)
        assert grouped["delegation_id"] == units[(0, 1)]["delegation_id"]
        assert grouped["group"] == "compare"
        assert [r["task_index"] for r in grouped["results"]] == [0, 1]
        content = consume(grouped)
        assert "implementation alpha" in content and "implementation beta" in content
        assert "group 'compare'" in content
    finally:
        for gate in gates:
            gate.set()
        release_pending_injects(agent, messages)
        db.close()


@pytest.mark.parametrize("requested,expected", [
    (None, "after_turn"), ("unknown-mode", "after_turn"),
    ("after_turn", "after_turn"), (" INJECT ", "inject"),
])
def test_unit_dispatch_propagates_policy_and_default_uses_same_after_turn_claim(monkeypatch, requested, expected):
    gates = [threading.Event() for _ in range(3)]
    agent = _tool_boundary_agent()
    messages = [{"role": "tool", "tool_call_id": "current", "content": "unchanged"}]
    try:
        handle = _gated_unit_call(monkeypatch, gates, delivery=requested)
        assert handle["status"] == "dispatched"
        assert handle["result_delivery"] == expected
        for gate in gates:
            gate.set()
        events = [process_registry.completion_queue.get(timeout=5) for _ in range(2)]
        assert {e["delegation_id"] for e in events} == {u["delegation_id"] for u in handle["units"]}
        for event in events:
            assert event["result_delivery"] == expected
            assert event["parent_turn_id"] == "turn-current"
            process_registry.completion_queue.put(event)
        attached = attach_ready_injects_to_tool_results(agent, messages, 1)
        if expected == "inject":
            assert attached == 2  # two units, not three child rows
            release_pending_injects(agent, messages)
        else:
            assert attached == 0
        assert messages[-1]["content"] == "unchanged"
        for _ in range(2):
            event = process_registry.completion_queue.get(timeout=5)
            claim = ad.claim_event_delivery(event, "ordinary-after-turn")
            assert claim
            assert ad.complete_event_delivery(event, claim)
            assert ad.get_event_delivery_state(event) == "delivered"
    finally:
        for gate in gates:
            gate.set()
        release_pending_injects(agent, messages)


@pytest.mark.parametrize("batch", [False, True])
def test_direct_dispatch_entry_points_persist_unit_policy(batch):
    common = dict(context=None, toolsets=None, role="leaf", model="test", session_key="",
                  parent_session_id="parent-session", parent_turn_id="turn-current", result_delivery="inject")
    if batch:
        handle = ad.dispatch_async_delegation_batch(
            goals=["audit"], runner=lambda: {"results": [_child(0, "ready")], "total_duration_seconds": 0.1},
            **common,
        )
    else:
        handle = ad.dispatch_async_delegation(
            goal="audit", runner=lambda: {"status": "completed", "summary": "ready"}, **common,
        )
    event = process_registry.completion_queue.get(timeout=5)
    assert event["delegation_id"] == handle["delegation_id"]
    assert event["parent_turn_id"] == "turn-current"
    assert event["result_delivery"] == "inject"
    claim = ad.claim_event_delivery(event, "after-turn")
    assert claim and ad.complete_event_delivery(event, claim)


def test_abandoned_unit_retains_dispatch_policy_and_turn_on_recovery(monkeypatch):
    import json

    delegation_id = _record()
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: False)
    assert ad.recover_abandoned_delegations() == 1
    with ad._DB_LOCK, ad._transaction() as conn:
        row = conn.execute("SELECT event_json FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone()
    event = json.loads(row[0])
    assert event["delegation_id"] == delegation_id
    assert event["status"] == "unknown"
    assert event["result_delivery"] == "inject"
    assert event["parent_turn_id"] == "turn-current"
    # A later turn cannot steal an abandoned unit's opt-in window.
    process_registry.completion_queue.put(event)
    agent = _tool_boundary_agent()
    agent._active_turn_id = "new-turn"
    messages = [{"role": "tool", "tool_call_id": "late", "content": "untouched"}]
    assert attach_ready_injects_to_tool_results(agent, messages, 1) == 0
    claim = ad.claim_event_delivery(event, "recovered-after-turn")
    assert claim and ad.complete_event_delivery(event, claim)
