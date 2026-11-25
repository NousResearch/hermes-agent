"""Gateway readers of a session's pending admissions: the messages /new names as left behind a lost
turn, and the fresh-session check an exhausted turn's replay makes before re-running its request."""
from types import SimpleNamespace

import pytest

from tests.fixtures.storage_seed import seed_admission, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _runner(monkeypatch, store):
    import gateway.session_authorities as authorities

    authority = SimpleNamespace(persistence=store, logical_owner=lambda session_id: session_id)
    monkeypatch.setattr(authorities, "active_authority", lambda _runner: authority)
    return SimpleNamespace()


@pytest.mark.asyncio
async def test_paused_followers_counts_chat_messages_queued_behind_a_lost_turn(tmp_path, monkeypatch):
    from gateway.slash_commands_session import paused_followers

    store = storage_handle(tmp_path)
    seed_session(store, "paused", "telegram")
    seed_session(store, "running", "telegram")
    seed_admission(store, "paused", "lost", status="unknown", principal_id="messaging:42")
    seed_admission(store, "paused", "chat-1", principal_id="messaging:42")
    seed_admission(store, "paused", "chat-2", principal_id="messaging:42")
    seed_admission(store, "paused", "cron", principal_id="automation:tick")
    seed_admission(store, "running", "chat-3", principal_id="messaging:42")
    runner = _runner(monkeypatch, store)

    assert await paused_followers(runner, "paused") == 2
    assert await paused_followers(runner, "running") == 0
    assert await paused_followers(runner, "") == 0


def test_fresh_session_pending_input_reads_the_sessions_fifo(tmp_path, monkeypatch):
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin

    store = storage_handle(tmp_path)
    seed_session(store, "fresh", "telegram")
    seed_session(store, "settled", "telegram")
    seed_admission(store, "fresh", "queued")
    seed_admission(store, "settled", "done", status="terminal", outcome="completed")
    runner = _runner(monkeypatch, store)
    pending = GatewayTurnPersistenceMixin._hmwa_fresh_session_has_pending_input

    assert pending(runner, SimpleNamespace(session_id="fresh")) is True
    assert pending(runner, SimpleNamespace(session_id="settled")) is False
    assert pending(runner, None) is False
