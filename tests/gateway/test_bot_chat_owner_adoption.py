"""Owner login history survives either delivery-first or open-first Bot Chat adoption."""

import pytest

from tests.gateway import test_peer_dm_bot_chat_headless as peer
from tests.gateway.test_bot_chat_unadoptable_local_title import _deliver
from tests.waiting import async_wait_until

headless = peer.headless


@pytest.mark.parametrize("open_first", [False, True])
@pytest.mark.parametrize(
    "stamp", ["basic:alice", "self-hosted:alice", 'auth:v1:["basic","","alice"]']
)
@pytest.mark.asyncio
async def test_owner_chat_retains_identity_history_and_replies_in_both_orders(
    headless, monkeypatch, stamp, open_first
):
    import gateway.session_worker_reservation as reservation

    monkeypatch.setattr(reservation, "owner_worker_subject", lambda: "uid:1000")
    db = headless.authority.db
    sid = "existing-bot-chat"
    db.create_session(sid, source="tui", user_id=stamp)
    db.set_session_title(sid, "Bot Chat")
    db.append_message(sid, "user", "existing project context")
    db.append_message(sid, "assistant", "existing answer")
    history = db.get_messages_as_conversation(sid)
    owner = peer._local_owner(headless.authority)
    try:
        if open_first:
            await peer._rpc(owner, "session.resume", session_id=sid)
        first = await _deliver(headless.authority, "a")
        assert first["session_id"] == sid
        await async_wait_until(
            lambda: len(headless.executed) == 1, timeout=10, interval=0.05
        )
        await peer._rpc(owner, "session.resume", session_id=sid)
        response = await headless.cli.post(
            f"/api/sessions/{sid}/chat",
            json={"message": "follow-up from peer", "author": peer.AUTHOR},
        )
        body = await response.json()
        assert response.status == 200, body
        assert body["message"]["content"] == "pong"
        assert [entry[0] for entry in headless.executed] == [sid, sid]
        assert db.get_session_by_title("Bot Chat")["id"] == sid
        assert db.get_messages_as_conversation(sid)[: len(history)] == history
        assert [row[0] for row in db._read_all("SELECT id FROM sessions")] == [sid]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_owner_login_chat_is_not_replaced_while_its_legacy_writer_is_live(
    headless, monkeypatch
):
    from pathlib import Path

    import gateway.session_worker_reservation as reservation
    from hermes_cli.active_sessions import (
        release_active_session,
        try_acquire_active_session,
    )
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setattr(reservation, "owner_worker_subject", lambda: "uid:1000")
    db = headless.authority.db
    sid = "busy-existing-bot-chat"
    db.create_session(sid, source="tui", user_id="basic:alice")
    db.set_session_title(sid, "Bot Chat")
    db.append_message(sid, "assistant", "existing project context")
    lease, refusal = try_acquire_active_session(
        session_id=sid,
        surface="desktop",
        registry_home=Path(db.db_path).parent,
        config={},
    )
    assert lease is not None, refusal
    try:
        with pytest.raises(RuntimeStoreError) as refused:
            await _deliver(headless.authority, "b")
        assert refused.value.reason == "runtime_coordination_required"
        assert db.get_session_by_title("Bot Chat")["id"] == sid
        assert not headless.executed
        assert not db.get_session(sid)["chat_id"]
        assert [row[0] for row in db._read_all("SELECT id FROM sessions")] == [sid]
    finally:
        release_active_session(lease)
    delivered = await _deliver(headless.authority, "c")
    assert delivered["session_id"] == sid
    await async_wait_until(
        lambda: len(headless.executed) == 1, timeout=10, interval=0.05
    )
    assert (
        db.get_messages_as_conversation(sid)[0]["content"] == "existing project context"
    )
