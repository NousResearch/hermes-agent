"""Behavior-contract test: cron run_job wires session_search to the job's
session DB (issue #6581).

When a cron job's agent calls the ``session_search`` tool, the query must reach
the SessionDB that ``run_job`` provisioned for that run — with the cron-default
filters (tool output excluded, no role filter) — rather than a detached or
missing store. This asserts the wiring contract end-to-end through ``run_job``,
not a snapshot of any particular result.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from cron.scheduler import run_job


def test_cron_session_search_uses_passed_session_db(tmp_path):
    job = {
        "id": "session-search-job",
        "name": "session search test",
        "prompt": "hello",
    }
    fake_db = MagicMock()

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            self._session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]

        def run_conversation(self, *args, **kwargs):
            result = self._invoke_tool(
                "session_search", {"query": "cron jobs"}, effective_task_id="cron"
            )
            return {"final_response": result}

        def _invoke_tool(
            self, function_name, function_args, effective_task_id=None, tool_call_id=None
        ):
            if function_name == "session_search":
                from tools.session_search_tool import session_search as _session_search

                return _session_search(
                    query=function_args.get("query", ""),
                    role_filter=function_args.get("role_filter"),
                    limit=function_args.get("limit", 3),
                    db=self._session_db,
                    current_session_id=self.session_id,
                )
            raise AssertionError(f"Unexpected tool: {function_name}")

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("dotenv.load_dotenv"), \
         patch("hermes_state.SessionDB", return_value=fake_db), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "***",
                 "base_url": "https://example.invalid/v1",
                 "provider": "openrouter",
                 "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent", FakeAgent):
        fake_db.search_messages.return_value = []
        success, output, final_response, error = run_job(job)

    assert success is True
    assert error is None
    payload = json.loads(final_response)
    assert payload["success"] is True
    assert payload["count"] == 0
    # The query must reach the job's own SessionDB (the wiring contract).
    # We assert the load-bearing arguments rather than freezing every default,
    # so this doesn't become a change-detector when tool defaults evolve.
    fake_db.search_messages.assert_called_once()
    call = fake_db.search_messages.call_args
    assert call.kwargs["query"] == "cron jobs"
    # Cron search excludes tool output by default so noise doesn't crowd recall.
    assert "tool" in call.kwargs["exclude_sources"]
