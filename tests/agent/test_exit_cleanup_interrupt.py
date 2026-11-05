"""A second Ctrl+C during cron finalization must not skip resource cleanup."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_keyboard_interrupt_in_end_session_does_not_skip_close(tmp_path, monkeypatch):
    """Exercise owner execution and release its borrowed store despite Ctrl+C."""
    from gateway.session_contract import SessionRef
    from gateway.session_cron import current_execution, execute
    from persistence_assembly import Persistence

    home = tmp_path / "owner"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    (home / "config.yaml").write_text("model:\n  default: test/model\n  provider: custom\n")
    store = storage_handle(home)
    ref = SessionRef(str(home), "cron-cleanup-session")
    seed_session(store, ref.session_id, "cron")
    holder = raw_holder(store)
    authority = SimpleNamespace(
        persistence=store, sessions={ref.session_id: SimpleNamespace(
            source=SimpleNamespace(user_id="cron-owner"))}, pending_results={})
    job = {"id": "test-job-1", "name": "test cleanup", "prompt": "hello",
           "schedule": "0 9 * * *", "model": "test/model"}
    row = {"admission_id": "cleanup-admission", "request_id": "cleanup-request",
           "principal_id": "cron-owner", "payload": {"text": ""}}
    policy = SimpleNamespace(request_json=json.dumps({
        "cron_job": job, "extra_prompt": None, "request_id": row["request_id"]}))
    runtime = {"provider": "custom", "api_key": "test-key",
               "base_url": "http://127.0.0.1:1/v1", "model": "test/model",
               "api_mode": "chat_completions"}
    reached = []

    def fail_turn(*args, **kwargs):
        reached.append(current_execution())
        raise RuntimeError("boom")

    try:
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime), \
             patch("run_agent.AIAgent") as agent_class, \
             patch.object(type(authority.persistence.sessions), "end", side_effect=KeyboardInterrupt) as end_session, \
             patch.object(Persistence, "close", autospec=True, side_effect=Persistence.close) as close:
            agent = agent_class.return_value
            agent.session_id = ref.session_id
            agent.run_conversation.side_effect = fail_turn
            previous = current_execution()
            with pytest.raises(RuntimeError, match="RuntimeError: boom"):
                await execute(authority, ref, row, policy)

            # The owner tuple carries the scheduler's execution id (the admission's request id), the
            # id the cron/executions ledger knows; never the admission id (df12cce832c).
            assert reached == [(authority, ref.session_id, job["id"], row["request_id"])]
            assert raw_holder(agent_class.call_args.kwargs["session_db"]) is holder
            end_session.assert_called_once_with(ref.session_id, "cron_incomplete_no_output")
            close.assert_called_once_with(agent_class.call_args.kwargs["session_db"])
            agent.close.assert_called_once()
            assert current_execution() is previous
            assert not authority._cron_cancellations
            result = authority.pending_results[row["admission_id"]]["result"]["cron_result"]
            assert result[0] is False and result[3] == "RuntimeError: boom"
            # Cron released its borrow, not the owner's live SQLite connection.
            saved = session_row(store, ref.session_id)
            assert saved is not None and saved["title"].startswith(job["name"])
    finally:
        assert raw_holder(store) is holder
