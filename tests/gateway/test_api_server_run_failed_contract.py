"""Regression tests for the /v1/runs failed-turn contract (#90436).

The ``codex_app_server`` runtime reported a failed turn via
``completed: False`` + ``error`` without ever setting ``failed: True``, so
``_handle_runs`` fell through to the green branch and emitted
``run.completed`` with empty output and zero usage — reintroducing #15561
through a second producer. The producer now sets ``failed`` on both of its
error returns, and the consumer treats ``error``/``completed=False`` as
failure even if a runtime omits the flag.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_turn(error=None, interrupted=False, final_text=""):
    return SimpleNamespace(
        error=error,
        interrupted=interrupted,
        final_text=final_text,
        tool_iterations=1,
        thread_id="th1",
        turn_id="tu1",
        projected_messages=[],
        should_retire=False,
    )


class TestCodexRuntimeSetsFailed:
    def _run_turn(self, turn):
        """Drive run_codex_app_server_turn's terminal return with a stub agent."""
        import agent.codex_runtime as cr

        agent = MagicMock()
        agent._codex_session = MagicMock()
        agent._codex_session.run_turn.return_value = turn
        agent._session_db = None
        agent._flush_messages_to_session_db = lambda msgs: True
        agent._sync_external_memory_for_turn = MagicMock()
        agent._skill_nudge_interval = 0
        agent.valid_tool_names = set()
        agent._iters_since_skill = 0

        with patch.object(cr, "_record_codex_app_server_compaction"), \
             patch.object(cr, "_record_codex_app_server_usage", return_value={}):
            return cr.run_codex_app_server_turn(
                agent,
                user_message="hi",
                original_user_message="hi",
                messages=[],
                effective_task_id="t1",
            )

    def test_turn_error_sets_failed(self):
        """A turn that errored (not interrupted) must carry failed=True so
        api_server emits run.failed (#90436)."""
        result = self._run_turn(
            _make_turn(error="Codex authentication failed", final_text="")
        )
        assert result["failed"] is True
        assert result["completed"] is False
        assert result["error"] == "Codex authentication failed"

    def test_clean_turn_not_failed(self):
        result = self._run_turn(_make_turn(final_text="done"))
        assert result.get("failed") is False
        assert result["completed"] is True
        assert result["error"] is None


class TestHandleRunsFailureContract:
    @pytest.mark.asyncio
    async def test_result_with_error_emits_run_failed(self, tmp_path):
        """Consumer defence: even without the failed flag, an error-bearing
        result must terminate run.failed, never a green run.completed."""
        from gateway.config import PlatformConfig
        from gateway.platforms.api_server import APIServerAdapter
        from tests.gateway.test_api_server_runs import _create_runs_app

        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
        mock_agent = MagicMock()
        # The exact codex_app_server failure shape BEFORE the producer fix.
        mock_agent.run_conversation.return_value = {
            "final_response": "",
            "completed": False,
            "error": "Codex authentication failed",
        }
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        from aiohttp.test_utils import TestClient, TestServer

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]

                status = None
                import asyncio as _aio

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status.get("status") in {"failed", "completed"}:
                        break
                    await _aio.sleep(0.05)

        assert status["status"] == "failed", (
            "an error-bearing result must terminate the run as failed, not green"
        )
        assert status["last_event"] == "run.failed"
