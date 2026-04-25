from unittest.mock import MagicMock, patch

import cron.scheduler as cron_scheduler
from gateway.runtime_governor import RuntimeGovernorDecision, RuntimeGovernorHeartbeat


class _Governor:
    def __init__(self, decision):
        self.decision = decision
        self.calls = []

    def admit(self, **kwargs):
        self.calls.append(("admit", kwargs))
        return self.decision

    def start(self, lease_id):
        self.calls.append(("start", lease_id))

    def heartbeat(self, lease_id):
        self.calls.append(("heartbeat", lease_id))
        return RuntimeGovernorHeartbeat(should_stop=False)

    def finish(self, lease_id, *, reason):
        self.calls.append(("finish", lease_id, reason))

    def fail(self, lease_id, *, reason):
        self.calls.append(("fail", lease_id, reason))


def _patch_runtime_provider():
    return patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value={
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
        },
    )


def test_cron_runtime_governor_denial_skips_agent(monkeypatch, tmp_path):
    governor = _Governor(
        RuntimeGovernorDecision(
            allowed=False,
            reason="daily_cap",
            user_message="Daily cap reached.",
        )
    )
    monkeypatch.setattr("gateway.runtime_governor.get_runtime_governor", lambda: governor)

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("dotenv.load_dotenv"), \
         patch("hermes_state.SessionDB", return_value=MagicMock()), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        success, output, final_response, error = cron_scheduler.run_job(
            {"id": "job-1", "name": "test", "prompt": "hello"}
        )

    assert success is True
    assert error is None
    assert final_response == "Daily cap reached."
    assert "Daily cap reached." in output
    mock_agent_cls.assert_not_called()
    assert governor.calls[0][0] == "admit"
    assert governor.calls[0][1]["platform"] == "cron"


def test_cron_runtime_governor_finishes_successful_lease(monkeypatch, tmp_path):
    governor = _Governor(
        RuntimeGovernorDecision(
            allowed=True,
            lease_id="lease-1",
            reason="allowed",
        )
    )
    monkeypatch.setattr("gateway.runtime_governor.get_runtime_governor", lambda: governor)

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("dotenv.load_dotenv"), \
         patch("hermes_state.SessionDB", return_value=MagicMock()), \
         _patch_runtime_provider(), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent

        success, _output, final_response, error = cron_scheduler.run_job(
            {"id": "job-1", "name": "test", "prompt": "hello"}
        )

    assert success is True
    assert error is None
    assert final_response == "ok"
    assert [call[0] for call in governor.calls] == ["admit", "start", "finish"]
    assert governor.calls[-1] == ("finish", "lease-1", "agent_end")
