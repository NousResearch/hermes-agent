import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.orchestration_runtime import (
    OrchestrationRuntime,
    get_orchestration_runtime,
    set_orchestration_runtime,
)
from agent.task_registry import STATUS_CANCELLED
from agent.worker_lanes import CancelToken, ThreadWorkerLane, WorkerSpec
from gateway.platforms.base import MessageType, build_session_key
from tests.gateway.test_busy_session_ack import _make_adapter, _make_event, _make_runner


@pytest.mark.asyncio
async def test_short_korean_chat_question_falls_through_to_model_when_frontdesk_live_enabled():
    from gateway.run import GatewayRunner

    runner, _sentinel = _make_runner()
    runner.frontdesk_live_enabled = True
    runner._orchestration_status_queries_enabled = True
    adapter = _make_adapter()
    event = _make_event(text="지금 뭐 하고 있어?")
    runner.adapters[event.source.platform] = adapter

    with patch.object(
        GatewayRunner,
        "_handle_message_with_agent",
        new=AsyncMock(return_value="model"),
    ) as handle_with_agent:
        result = await GatewayRunner._handle_message(runner, event)

    handle_with_agent.assert_called_once()
    assert result == "model"


@pytest.mark.asyncio
async def test_stop_never_enters_pending_queue_when_busy():
    runner, _sentinel = _make_runner()
    runner.frontdesk_live_enabled = True
    runner._busy_input_mode = "queue"
    adapter = _make_adapter()
    event = _make_event(text="멈춰")
    sk = build_session_key(event.source)
    agent = MagicMock()
    runner._running_agents[sk] = agent
    runner._running_agents_ts[sk] = time.time()
    runner.adapters[event.source.platform] = adapter

    with patch("gateway.run.merge_pending_message_event") as merge:
        handled = await runner._handle_active_session_busy_message(event, sk)

    assert handled is True
    merge.assert_not_called()
    assert sk not in adapter._pending_messages
    agent.interrupt.assert_called_once_with("멈춰")


@pytest.mark.asyncio
async def test_worker_request_starts_default_worker_when_frontdesk_enabled():
    from gateway.run import GatewayRunner

    runner, _sentinel = _make_runner()
    runner.frontdesk_live_enabled = True
    adapter = _make_adapter()
    event = _make_event(text="워커 레인에 배당해서 이 회귀를 조사해줘")
    runner.adapters[event.source.platform] = adapter

    with patch.object(GatewayRunner, "_run_agent", autospec=True) as run_agent, patch(
        "agent.frontdesk_live._run_default_worker_subprocess",
        return_value="worker done",
    ):
        result = await GatewayRunner._handle_message(runner, event)

    run_agent.assert_not_called()
    assert isinstance(result, str)
    assert "worker started" in result
    runtime = get_orchestration_runtime(runner)
    assert runtime is not None
    assert "main" in runtime.worker_registry.lane_names()
    tasks = runtime.task_registry.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].status != STATUS_CANCELLED
    assert tasks[0].active_worker_id
    assert "worker started" in " ".join(tasks[0].notes)

    assert runtime.worker_registry.wait(tasks[0].active_worker_id, timeout=2.0)
    assert tasks[0].result is not None
    assert tasks[0].result["status"] == "succeeded"
    assert tasks[0].result["summary"] == "worker done"
    assert tasks[0].result["review_status"] == "pending_review"


@pytest.mark.asyncio
async def test_korean_followup_steers_active_agent_when_live_enabled():
    runner, _sentinel = _make_runner()
    runner.frontdesk_live_enabled = True
    adapter = _make_adapter()
    event = _make_event(text="중국집은 없나")
    sk = build_session_key(event.source)
    agent = MagicMock()
    agent.steer.return_value = True
    runner._running_agents[sk] = agent
    runner.adapters[event.source.platform] = adapter

    with patch("gateway.run.merge_pending_message_event") as merge:
        handled = await runner._handle_active_session_busy_message(event, sk)

    assert handled is True
    agent.steer.assert_called_once_with("중국집은 없나")
    merge.assert_not_called()
    assert sk not in adapter._pending_messages


@pytest.mark.asyncio
async def test_gateway_pending_sentinel_does_not_consume_korean_followup():
    from gateway.run import GatewayRunner

    runner, sentinel = _make_runner()
    runner.frontdesk_live_enabled = True
    adapter = _make_adapter()
    event = _make_event(text="빠니니를 파는 곳도 찾아보고 있어야지")
    sk = build_session_key(event.source)
    runner._running_agents[sk] = sentinel
    runner.adapters[event.source.platform] = adapter

    with patch.object(GatewayRunner, "_run_agent", autospec=True) as run_agent:
        result = await GatewayRunner._handle_message(runner, event)

    assert result is None
    run_agent.assert_not_called()
    assert sk in adapter._pending_messages


@pytest.mark.asyncio
async def test_gateway_korean_followup_attaches_to_active_worker_task():
    from gateway.run import GatewayRunner

    runner, _sentinel = _make_runner()
    runner.frontdesk_live_enabled = True
    adapter = _make_adapter()
    event = _make_event(text="중국집은 없나")
    sk = build_session_key(event.source)
    runner.adapters[event.source.platform] = adapter

    runtime = OrchestrationRuntime.create()
    entered = threading.Event()
    release = threading.Event()

    def worker(spec: WorkerSpec, token: CancelToken):  # noqa: ARG001
        entered.set()
        release.wait(2.0)
        token.raise_if_cancelled()
        return "done"

    lane = ThreadWorkerLane(runner=worker, name="thread")
    runtime.worker_registry.register(lane)
    set_orchestration_runtime(runner, runtime)
    started = runtime.handle_frontdesk_input(
        "워커 레인에 배당해서 이 회귀를 조사해줘",
        frontdesk_mode_active=True,
        session_key=sk,
        source_surface="gateway",
    )
    assert started.worker_id is not None
    assert started.task_id is not None
    assert entered.wait(2.0)

    try:
        with patch.object(GatewayRunner, "_run_agent", autospec=True) as run_agent:
            result = await GatewayRunner._handle_message(runner, event)

        run_agent.assert_not_called()
        assert isinstance(result, str)
        assert "follow-up attached" in result
        task = runtime.task_registry.get_task(started.task_id)
        assert task is not None
        assert [item.text for item in task.pending_followups] == ["중국집은 없나"]
        assert [item.text for item in lane.followups(started.worker_id)] == ["중국집은 없나"]
    finally:
        runtime.worker_registry.cancel(started.worker_id)
        release.set()
        runtime.worker_registry.wait(started.worker_id, timeout=2.0)


@pytest.mark.asyncio
async def test_frontdesk_status_command_reports_live_gate_and_worker_lane():
    from gateway.run import GatewayRunner

    runner, _sentinel = _make_runner()
    runner.frontdesk_live_enabled = True
    event = _make_event(text="/frontdesk status")

    result = await GatewayRunner._handle_frontdesk_command(runner, event)

    assert "Frontdesk live: enabled" in result
    assert "Default worker lane: available" in result
    assert "Available worker lanes: main" in result


@pytest.mark.asyncio
async def test_frontdesk_mode_enabled_alone_does_not_enable_gateway_live_interception():
    from gateway.run import GatewayRunner

    runner, _sentinel = _make_runner()
    runner.frontdesk_mode_enabled = True
    runner.frontdesk_live_enabled = False
    adapter = _make_adapter()
    event = _make_event(text="지금 뭐 하고 있어?")
    runner.adapters[event.source.platform] = adapter

    with patch.object(
        GatewayRunner,
        "_handle_message_with_agent",
        new=AsyncMock(return_value="model"),
    ) as handle_with_agent:
        result = await GatewayRunner._handle_message(runner, event)

    assert result == "model"
    handle_with_agent.assert_called_once()


@pytest.mark.asyncio
async def test_frontdesk_disabled_preserves_existing_busy_behavior():
    runner, _sentinel = _make_runner()
    runner.frontdesk_live_enabled = False
    runner._busy_input_mode = "queue"
    adapter = _make_adapter()
    event = _make_event(text="멈춰")
    event.message_type = MessageType.TEXT
    sk = build_session_key(event.source)
    agent = MagicMock()
    runner._running_agents[sk] = agent
    runner.adapters[event.source.platform] = adapter

    with patch("gateway.run.merge_pending_message_event") as merge:
        await runner._handle_active_session_busy_message(event, sk)

    merge.assert_called_once()
    agent.interrupt.assert_not_called()
