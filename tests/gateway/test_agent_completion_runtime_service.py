"""DEAD path: not imported by gateway/run.py — contract-only unit tests.
See gateway/RUNTIME_SERVICES.md.
"""
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.agent_completion_runtime_service import (
    build_gateway_agent_end_payload,
    apply_gateway_reasoning_display,
    drain_pending_process_watchers,
    log_gateway_response_ready,
    prepare_gateway_agent_completion,
    stop_gateway_typing_indicator,
    sync_gateway_session_entry_id,
)


def test_apply_gateway_reasoning_display_returns_original_without_reasoning():
    assert apply_gateway_reasoning_display(
        response="hello",
        show_reasoning=False,
        last_reasoning="step 1",
    ) == "hello"


def test_apply_gateway_reasoning_display_collapses_long_reasoning():
    reasoning = "\n".join(f"line {idx}" for idx in range(20))

    result = apply_gateway_reasoning_display(
        response="answer",
        show_reasoning=True,
        last_reasoning=reasoning,
    )

    assert result.startswith("💭 **Reasoning:**")
    assert "line 0" in result
    assert "line 14" in result
    assert "_... (5 more lines)_" in result
    assert result.endswith("\n\nanswer")


def test_drain_pending_process_watchers_schedules_all_and_logs_resumed_watchers():
    scheduled = []
    logger = MagicMock()

    async def _run_process_watcher(watcher):
        return watcher

    def _create_task(coro):
        scheduled.append(coro)
        coro.close()

    process_registry = type(
        "_Registry",
        (),
        {
            "pending_watchers": [
                {"session_id": "sess-1"},
                {"session_id": "sess-2"},
            ]
        },
    )()

    result = drain_pending_process_watchers(
        process_registry=process_registry,
        run_process_watcher=_run_process_watcher,
        create_task=_create_task,
        logger=logger,
        resumed_log_template="Resumed watcher for recovered process %s",
    )

    assert result == 2
    assert process_registry.pending_watchers == []
    assert len(scheduled) == 2
    assert logger.info.call_count == 2


def test_drain_pending_process_watchers_without_log_template_still_schedules():
    scheduled = []

    async def _run_process_watcher(watcher):
        return watcher

    def _create_task(coro):
        scheduled.append(coro)
        coro.close()

    process_registry = type(
        "_Registry",
        (),
        {"pending_watchers": [{"session_id": "sess-1"}]},
    )()

    result = drain_pending_process_watchers(
        process_registry=process_registry,
        run_process_watcher=_run_process_watcher,
        create_task=_create_task,
    )

    assert result == 1
    assert process_registry.pending_watchers == []
    assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_stop_gateway_typing_indicator_calls_adapter_when_supported():
    adapter = SimpleNamespace(stop_typing=AsyncMock())

    await stop_gateway_typing_indicator(
        adapters={"qq": adapter},
        platform="qq",
        chat_id="chat-1",
    )

    adapter.stop_typing.assert_awaited_once_with("chat-1")


@pytest.mark.asyncio
async def test_stop_gateway_typing_indicator_ignores_adapter_errors():
    adapter = SimpleNamespace(stop_typing=AsyncMock(side_effect=RuntimeError("boom")))

    await stop_gateway_typing_indicator(
        adapters={"qq": adapter},
        platform="qq",
        chat_id="chat-1",
    )

    adapter.stop_typing.assert_awaited_once_with("chat-1")


def test_log_gateway_response_ready_uses_standard_log_shape():
    logger = MagicMock()

    log_gateway_response_ready(
        logger=logger,
        platform_name="qq",
        chat_id="group-1",
        msg_start_time=0.0,
        agent_result={"api_calls": 3},
        response="hello",
        response_state="sent",
    )

    logger.info.assert_called_once()
    args = logger.info.call_args.args
    assert args[0].startswith("response ready:")
    assert args[1] == "qq"
    assert args[2] == "group-1"
    assert args[4] == 3
    assert args[5] == 5
    assert args[6] == "sent"


def test_sync_gateway_session_entry_id_updates_changed_session():
    session_entry = SimpleNamespace(session_id="old")

    changed = sync_gateway_session_entry_id(
        session_entry=session_entry,
        agent_result={"session_id": "new"},
    )

    assert changed is True
    assert session_entry.session_id == "new"


def test_build_gateway_agent_end_payload_truncates_response():
    payload = build_gateway_agent_end_payload(
        hook_ctx={"session_id": "sess-1"},
        response="x" * 600,
    )

    assert payload["session_id"] == "sess-1"
    assert len(payload["response"]) == 500


@pytest.mark.asyncio
async def test_prepare_gateway_agent_completion_handles_reasoning_hook_and_watchers():
    hooks = SimpleNamespace(emit=AsyncMock())
    logger = MagicMock()
    session_entry = SimpleNamespace(session_id="sess-1")
    scheduled = []

    async def _run_process_watcher(watcher):
        return watcher

    def _create_task(coro):
        scheduled.append(coro)
        coro.close()

    process_registry = SimpleNamespace(pending_watchers=[{"session_id": "sess-w"}])

    result = await prepare_gateway_agent_completion(
        agent_result={
            "final_response": "answer",
            "last_reasoning": "\n".join(f"line {idx}" for idx in range(3)),
            "messages": [{"role": "assistant", "content": "answer"}],
            "api_calls": 2,
            "session_id": "sess-2",
        },
        history_len=4,
        empty_response_fallback=lambda kind: None,
        session_entry=session_entry,
        show_reasoning=True,
        hook_ctx={"session_id": "sess-1", "platform": "qq"},
        hooks=hooks,
        logger=logger,
        platform_name="qq",
        chat_id="group-1",
        msg_start_time=0.0,
        process_registry=process_registry,
        run_process_watcher=_run_process_watcher,
        create_task=_create_task,
    )

    assert result.suppress_reply is False
    assert result.response_state == "sent"
    assert result.agent_messages == [{"role": "assistant", "content": "answer"}]
    assert result.response.startswith("💭 **Reasoning:**")
    assert session_entry.session_id == "sess-2"
    hooks.emit.assert_awaited_once_with(
        "agent:end",
        {
            "session_id": "sess-1",
            "platform": "qq",
            "response": ANY,
        },
    )
    assert process_registry.pending_watchers == []
    assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_prepare_gateway_agent_completion_marks_synthetic_fallback_on_agent_result():
    hooks = SimpleNamespace(emit=AsyncMock())
    logger = MagicMock()
    session_entry = SimpleNamespace(session_id="sess-1")
    agent_result = {
        "final_response": "(empty)",
        "messages": [],
        "api_calls": 1,
    }

    result = await prepare_gateway_agent_completion(
        agent_result=agent_result,
        history_len=2,
        empty_response_fallback=lambda kind: "刚才接口空转了",
        session_entry=session_entry,
        show_reasoning=False,
        hook_ctx={"session_id": "sess-1"},
        hooks=hooks,
        logger=logger,
        platform_name="qq",
        chat_id="group-1",
        msg_start_time=0.0,
    )

    assert result.response == "刚才接口空转了"
    assert agent_result["response_state"] == "qq_explicit_fallback"
    assert agent_result["synthetic_fallback"] is True
