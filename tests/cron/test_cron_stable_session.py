"""Tests for stable cron session IDs (issue #88268)."""

from unittest.mock import patch, MagicMock
from cron.scheduler import run_job


def test_cron_uses_latest_session_key():
    """A recurring cron job should map to ONE stable session key
    (cron_<job_id>_latest) instead of a new timestamped key per tick."""
    job = {
        "id": "test-job",
        "name": "Test Job",
        "prompt": "Say hello",
        "schedule": "*/15 * * * *",
    }

    fake_db = MagicMock()
    fake_db.get_compression_tip.return_value = None

    with patch("cron.scheduler._hermes_home", "/tmp/hermes"), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
             "api_key": "test-key",
             "base_url": "https://example.invalid/v1",
             "provider": "openrouter",
             "api_mode": "chat_completions",
         }), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        success, output, final, error = run_job(job)

    assert success
    # The session_id should be stable (no timestamp)
    call_kwargs = mock_agent_cls.call_args.kwargs
    session_id = call_kwargs["session_id"]
    assert session_id == "cron_test-job_latest", f"unexpected session_id: {session_id}"
