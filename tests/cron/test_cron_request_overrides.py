"""Standalone regression test for cron runtime request_overrides forwarding.

Split out of tests/cron/test_scheduler.py as an independent, upgrade-safe file
(registered in the local-patch ledger's allowed_untracked list) so the local
request_overrides forwarding hotfix no longer collides with upstream inserting
new tests around its anchor in the large test_scheduler.py module.

The functional change under test lives in cron/scheduler.py (run_job forwards
runtime['request_overrides'] into the ephemeral AIAgent); that hunk stays in the
source patch. Only this test moved here.
"""

from unittest.mock import patch, MagicMock

from cron.scheduler import run_job


class TestRunJobRequestOverrides:
    def test_run_job_forwards_runtime_request_overrides_to_agent(self, tmp_path):
        # runtime_provider may resolve provider-specific request_overrides;
        # cron must pass them into the ephemeral AIAgent or scheduled jobs
        # regress to SDK-default request settings.
        job = {
            "id": "request-overrides-job",
            "name": "request-overrides",
            "prompt": "hello",
        }
        fake_db = MagicMock()
        overrides = {
            "extra_headers": {
                "User-Agent": "codex_cli_rs/0.138.0 (Windows 10.0.26100; x86_64)"
            }
        }

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("dotenv.load_dotenv"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "custom",
                     "api_mode": "codex_responses",
                     "request_overrides": overrides,
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok"}
            mock_agent_cls.return_value = mock_agent

            success, _output, final_response, error = run_job(job)

        assert success is True
        assert error is None
        assert final_response == "ok"
        kwargs = mock_agent_cls.call_args.kwargs
        assert kwargs["request_overrides"] == overrides
