"""Test that cron.scheduler.run_job calls load_hermes_dotenv (which resolve BSM)."""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import run_job


def test_run_job_uses_load_hermes_dotenv(tmp_path, monkeypatch):
    """run_job must call load_hermes_dotenv so BSM-managed secrets are resolved."""
    job = {
        "name": "test-bsm",
        "prompt": "hi",
        "schedule": "0 0 * * *",
        "model": "gpt-4",
    }
    fake_db = MagicMock()
    captured = {}

    def _fake_load_hermes_dotenv(*, hermes_home):
        captured["called"] = True
        captured["hermes_home"] = hermes_home

    monkeypatch.setattr("cron.scheduler.load_hermes_dotenv", _fake_load_hermes_dotenv)
    monkeypatch.setattr("cron.scheduler._resolve_origin", lambda jid: None)
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch("cron.scheduler._resolve_delivery_target", return_value=None), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openrouter",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        run_job(job, job_id="j123")

    assert captured.get("called") is True
    assert isinstance(captured.get("hermes_home"), (Path, str))
