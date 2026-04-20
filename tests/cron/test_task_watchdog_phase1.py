import json
from unittest.mock import MagicMock, patch

from cron.scheduler import _build_job_prompt, run_job


def _runtime_provider_stub():
    return {
        "api_key": "***",
        "base_url": "https://example.invalid/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
    }


def test_build_job_prompt_includes_blocked_guidance():
    prompt = _build_job_prompt({"id": "job-1", "name": "Test", "prompt": "hello"})

    assert "[SILENT]" in prompt
    assert "[BLOCKED]" in prompt
    assert "user action" in prompt.lower()


def test_run_job_completed_writes_completed_watchdog_state(tmp_path):
    job = {"id": "job-1", "name": "Test Job", "prompt": "hello"}
    fake_db = MagicMock()
    tasks_file = tmp_path / "cron" / "task_heartbeat.json"

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("dotenv.load_dotenv"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_runtime_provider_stub()), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "done"}
        mock_agent.get_activity_summary.return_value = {
            "seconds_since_activity": 0.0,
            "last_activity_desc": "tool_call",
        }
        mock_agent_cls.return_value = mock_agent

        success, output, final_response, error = run_job(job)

    assert success is True
    assert error is None
    assert final_response == "done"
    assert "done" in output

    payload = json.loads(tasks_file.read_text())
    task = payload["tasks"][0]
    assert task["job_id"] == "job-1"
    assert task["job_name"] == "Test Job"
    assert task["status"] == "completed"
    assert task["delivery_target"] is None


def test_run_job_blocked_writes_waiting_external_and_suppresses_final_delivery(tmp_path):
    job = {
        "id": "job-2",
        "name": "Blocked Job",
        "prompt": "hello",
        "deliver": "telegram:123",
    }
    fake_db = MagicMock()
    tasks_file = tmp_path / "cron" / "task_heartbeat.json"

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("dotenv.load_dotenv"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_runtime_provider_stub()), \
         patch("cron.scheduler._deliver_result") as deliver_mock, \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {
            "final_response": "[BLOCKED] Please connect the GitHub token for this workspace."
        }
        mock_agent.get_activity_summary.return_value = {
            "seconds_since_activity": 0.0,
            "last_activity_desc": "waiting_on_user",
        }
        mock_agent_cls.return_value = mock_agent

        success, output, final_response, error = run_job(job)

    assert success is True
    assert error is None
    assert final_response == ""
    assert "[BLOCKED]" in output
    deliver_mock.assert_called_once()
    delivered_text = deliver_mock.call_args.args[1]
    assert "Blocked on Blocked Job" in delivered_text
    assert "GitHub token" in delivered_text

    payload = json.loads(tasks_file.read_text())
    task = payload["tasks"][0]
    assert task["status"] == "waiting_external"
    assert task["blocker_reason"] == "Please connect the GitHub token for this workspace."
    assert task["user_action_needed"] == "Please connect the GitHub token for this workspace."
