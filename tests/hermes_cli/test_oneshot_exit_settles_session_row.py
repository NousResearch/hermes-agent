"""One-shot exit settles the session row through the agent's storage handle.

An agent built on a writer's storage interface handle (``AIAgent(persistence=...)``, or the handle
``_get_session_db_for_recall`` opens) holds no ``SessionDB``. Exit must still drain its queued token
counts and end the row with ``cli_close``, as it does for an agent on the old store.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import cli as cli_mod
from persistence.conversation.usage import TokenCounts, UsageDelta
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.fixture(autouse=True)
def _reset_finalize_state(monkeypatch):
    monkeypatch.setattr(cli_mod, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli_mod, "_handed_off_session_ids", set())
    monkeypatch.setattr(cli_mod, "_cleanup_done", False, raising=False)
    monkeypatch.setattr(cli_mod, "_run_cleanup", lambda **kw: None)
    monkeypatch.setattr(cli_mod, "_notify_single_query_session_finalize", lambda _c: None)


def _agent_on(store, session_id="oneshot-handle"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                        quiet_mode=True, persistence=store, session_id=session_id,
                        skip_context_files=True, skip_memory=True)
    agent._ensure_db_session()
    return agent


def test_one_shot_exit_drains_token_counts_and_ends_the_row(tmp_path, monkeypatch):
    store = storage_handle(tmp_path)
    agent = _agent_on(store)
    agent._session_messages = []  # the turn is persisted already: no final persist retry drains the queue
    store.usage.queue_usage(agent.session_id, UsageDelta(tokens=TokenCounts(input_tokens=7), api_call_count=1))
    drains = []
    flush_usage = store.usage.flush_usage
    monkeypatch.setattr(store.usage, "flush_usage", lambda *a, **k: drains.append(a) or flush_usage(*a, **k))

    cli_mod._finalize_single_query(SimpleNamespace(
        agent=agent, session_id=agent.session_id, conversation_history=[], _session_db=None,
        _release_active_session=lambda: None))

    assert drains == [()], "one-shot exit must drain the queued token counts"
    with raw_holder(store)._read_ctx() as conn:
        tokens = conn.execute("SELECT input_tokens FROM sessions WHERE id = ?", (agent.session_id,)).fetchone()[0]
    assert tokens == 7
    row = session_row(store, agent.session_id)
    assert row["ended_at"] is not None
    assert row["end_reason"] == "cli_close"
