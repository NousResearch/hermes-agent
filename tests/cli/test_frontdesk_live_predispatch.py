import os
import queue
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.orchestration_runtime import get_orchestration_runtime
from agent.task_registry import STATUS_CANCELLED
from tests.cli.test_cli_init import _make_cli


def test_short_korean_chat_question_falls_through_to_model_when_frontdesk_live_enabled():
    cli = _make_cli()
    cli.frontdesk_live_enabled = True
    cli.agent = MagicMock()
    cli.agent.max_iterations = 90
    cli.agent.run_conversation.return_value = {
        "final_response": "model",
        "messages": [],
        "completed": True,
        "response_previewed": True,
    }
    cli._active_agent_route_signature = "sig"

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": "sig",
             "model": None,
             "runtime": None,
             "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        response = cli.chat("지금 뭐 하고 있어?")

    cli.agent.run_conversation.assert_called_once()
    assert response == "model"


def test_stop_never_enters_pending_queue_when_busy():
    cli = _make_cli(config_overrides={"display": {"busy_input_mode": "queue"}})
    cli.frontdesk_live_enabled = True
    cli._agent_running = True
    cli._pending_input = MagicMock()
    cli._interrupt_queue = queue.Queue()
    cli.agent = MagicMock()

    result = cli._handle_frontdesk_live_input("멈춰", main_in_flight=True)

    assert result is not None
    assert result.action == "stopped"
    cli._pending_input.put.assert_not_called()
    cli.agent.interrupt.assert_called_once_with("멈춰")


def test_worker_request_starts_default_worker_when_frontdesk_enabled():
    cli = _make_cli()
    cli.frontdesk_live_enabled = True
    cli.agent = MagicMock()
    cli.agent.run_conversation.return_value = {"final_response": "model"}

    with patch("agent.frontdesk_live._run_default_worker_subprocess", return_value="worker done"):
        response = cli.chat("워커 레인에 배당해서 이 회귀를 조사해줘")

    cli.agent.run_conversation.assert_not_called()
    assert "worker started" in response
    runtime = get_orchestration_runtime(cli)
    assert runtime is not None
    assert "main" in runtime.worker_registry.lane_names()
    tasks = runtime.task_registry.list_tasks()
    assert len(tasks) == 1
    assert tasks[0].status != STATUS_CANCELLED
    assert tasks[0].active_worker_id
    assert "worker started" in " ".join(tasks[0].notes)


def test_korean_followups_steer_when_live_and_agent_accepts():
    cli = _make_cli()
    cli.frontdesk_live_enabled = True
    cli.agent = MagicMock()
    cli.agent.steer.return_value = True

    for text in ("중국집은 없나", "빠니니를 파는 곳도 찾아보고 있어야지"):
        result = cli._handle_frontdesk_live_input(text, main_in_flight=True)
        assert result is not None
        assert result.action == "steered"

    assert cli.agent.steer.call_args_list[0].args == ("중국집은 없나",)
    assert cli.agent.steer.call_args_list[1].args == ("빠니니를 파는 곳도 찾아보고 있어야지",)


def test_korean_followup_rejected_by_steer_callback_falls_through():
    cli = _make_cli()
    cli.frontdesk_live_enabled = True
    cli.agent = MagicMock()
    cli.agent.steer.return_value = False

    result = cli._handle_frontdesk_live_input("중국집은 없나", main_in_flight=True)

    assert result is None
    cli.agent.steer.assert_called_once_with("중국집은 없나")


def test_frontdesk_mode_enabled_alone_does_not_enable_live_interception():
    cli = _make_cli()
    cli.frontdesk_mode_enabled = True
    cli.agent = MagicMock()
    cli.agent.max_iterations = 90
    cli.agent.run_conversation.return_value = {"final_response": "model"}
    cli._active_agent_route_signature = "sig"

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": "sig",
             "model": None,
             "runtime": None,
             "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        response = cli.chat("지금 뭐 하고 있어?")

    cli.agent.run_conversation.assert_called_once()
    assert response == "model"


def test_frontdesk_disabled_preserves_existing_behavior():
    cli = _make_cli()
    cli.frontdesk_live_enabled = False
    cli.agent = MagicMock()
    cli.agent.max_iterations = 90
    cli.agent.run_conversation.return_value = {"final_response": "model"}
    cli._active_agent_route_signature = "sig"

    with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
         patch.object(cli, "_resolve_turn_agent_config", return_value={
             "signature": "sig",
             "model": None,
             "runtime": None,
             "request_overrides": None,
         }), \
         patch.object(cli, "_init_agent", return_value=True):
        response = cli.chat("지금 뭐 하고 있어?")

    cli.agent.run_conversation.assert_called_once()
    assert response == "model"
