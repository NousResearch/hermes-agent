from unittest.mock import MagicMock

from gateway.agent_completion_runtime_service import (
    apply_gateway_reasoning_display,
    drain_pending_process_watchers,
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
