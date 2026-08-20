"""Regression test: script failure must not be masked as last_status=ok by LLM fallback.

Issue #36845: When a script-backed cron job's pre-run script fails (non-zero exit
or timeout), the LLM fallback path still runs and produces an error report.
Previously the agent's successful run was treated as overall job success,
masking the underlying script failure in cron metadata (last_status=ok).

The fix ensures _run_job_impl returns success=False when _script_failed=True,
even though the agent completed without error.
"""
import pytest
from unittest.mock import MagicMock, patch


class TestScriptFailureNotMasked:
    """Script failure should propagate through LLM fallback as job failure."""

    def test_script_failure_propagates_through_agent_fallback(self, tmp_path):
        """When script fails but agent runs, job should be marked failed."""
        from cron.scheduler import run_job

        script = tmp_path / "scripts" / "fail.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("import sys; sys.exit(1)")

        job = {
            "id": "script-fail-test",
            "name": "backup",
            "prompt": "Summarize the script error",
            "script": str(script),
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
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
            mock_agent.run_conversation.return_value = {
                "final_response": "The backup script failed with exit code 1."
            }
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        # The job should be marked as FAILED even though the agent ran fine
        assert success is False, \
            "Job should be marked failed when script fails, even if agent succeeds"
        assert error is not None, "Error message should be present"
        assert "script" in error.lower() or "failed" in error.lower(), \
            f"Error should mention script failure, got: {error}"
        # The agent response should still be available for delivery
        assert final_response == "The backup script failed with exit code 1."

    def test_script_success_still_returns_true(self, tmp_path):
        """When script succeeds and agent runs, job should succeed normally."""
        from cron.scheduler import run_job

        script = tmp_path / "scripts" / "ok.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("print('data collected')")

        job = {
            "id": "script-ok-test",
            "name": "collect",
            "prompt": "Analyze the data",
            "script": str(script),
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
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
            mock_agent.run_conversation.return_value = {
                "final_response": "Analysis complete."
            }
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True, "Job should succeed when script and agent both succeed"
        assert error is None
        assert final_response == "Analysis complete."

    def test_no_script_job_succeeds_normally(self, tmp_path):
        """Jobs without scripts should not be affected by the fix."""
        from cron.scheduler import run_job

        job = {
            "id": "no-script-test",
            "name": "daily",
            "prompt": "Generate a summary",
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
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
            mock_agent.run_conversation.return_value = {
                "final_response": "Summary done."
            }
            mock_agent_cls.return_value = mock_agent

            success, output, final_response, error = run_job(job)

        assert success is True
        assert error is None
