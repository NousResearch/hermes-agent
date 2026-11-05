"""Owner login history survives either delivery-first or open-first Bot Chat adoption, a live
legacy writer refuses the delivery without losing the chat, and a forwarded login delivers into
its own chat but never moves the owner's aside."""

import pytest

from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import conversation, session_row
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway import test_peer_dm_bot_chat_headless as peer
from tests.gateway.test_bot_chat_unadoptable_local_title import _deliver
from tests.waiting import async_wait_until

headless = peer.headless


def _session_ids(db):
    with raw_holder(db)._read_ctx() as conn:
        return [row[0] for row in conn.execute("SELECT id FROM sessions")]


@pytest.mark.parametrize("open_first", [False, True], ids=["delivery-first", "open-first"])
@pytest.mark.parametrize(
    "source, stamp",
    [
        ("tui", "basic:alice"),
        ("tui", "self-hosted:alice"),
        ("tui", 'auth:v1:["basic","","alice"]'),
        ("cli", "basic:alice"),  # CLI rows follow the TUI rule; one case pins it
    ],
)
@pytest.mark.asyncio
async def test_owner_chat_retains_identity_history_and_replies_in_both_orders(
    headless, monkeypatch, source, stamp, open_first
):
    import gateway.session_worker_reservation as reservation

    monkeypatch.setattr(reservation, "owner_worker_subject", lambda: "uid:1000")
    db = headless.authority.persistence
    sid = "existing-bot-chat"
    seed_session(db, sid, source, user_id=stamp, title="Bot Chat")
    seed_messages(db, sid, {"role": "user", "content": "existing project context"},
                  {"role": "assistant", "content": "existing answer"})
    history = conversation(db, sid)
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
        assert db.sessions.get_by_title("Bot Chat").session_id == sid
        assert conversation(db, sid)[: len(history)] == history
        assert _session_ids(db) == [sid]
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
    from persistence.core.errors import AuthorityRefusal
    from persistence_assembly import store_location

    monkeypatch.setattr(reservation, "owner_worker_subject", lambda: "uid:1000")
    db = headless.authority.persistence
    sid = "busy-existing-bot-chat"
    seed_session(db, sid, "tui", user_id="basic:alice", title="Bot Chat")
    seed_messages(db, sid, {"role": "assistant", "content": "existing project context"})
    lease, refusal = try_acquire_active_session(
        session_id=sid,
        surface="desktop",
        registry_home=Path(store_location(db.scope)).parent,
        config={},
    )
    assert lease is not None, refusal
    try:
        with pytest.raises(AuthorityRefusal) as refused:
            await _deliver(headless.authority, "b")
        assert refused.value.reason == "runtime_coordination_required"
        assert db.sessions.get_by_title("Bot Chat").session_id == sid
        assert not headless.executed
        assert not session_row(db, sid)["chat_id"]
        assert _session_ids(db) == [sid]
    finally:
        release_active_session(lease)
    delivered = await _deliver(headless.authority, "c")
    assert delivered["session_id"] == sid
    await async_wait_until(
        lambda: len(headless.executed) == 1, timeout=10, interval=0.05
    )
    assert (
        conversation(db, sid)[0]["content"] == "existing project context"
    )


# A login a same-user peer forwards over the control socket (#138915); no registered provider
# names it, so it keeps its own subject (``principal_subject``) and owns its own stamps.
FORWARDED = 'auth:v1:["github","","alice"]'


@pytest.mark.parametrize("open_first", [False, True], ids=["delivery-first", "open-first"])
@pytest.mark.parametrize("stamp", [FORWARDED, "github:alice"], ids=["subject", "label"])
@pytest.mark.asyncio
async def test_a_forwarded_login_delivers_into_its_own_chat_in_both_orders(
    headless, monkeypatch, stamp, open_first
):
    # The forwarded login adopts the chat it stamped, so its delivery must not move that chat
    # aside, whether or not it opened it first.
    import gateway.session_worker_reservation as reservation

    monkeypatch.setattr(reservation, "owner_worker_subject", lambda: "uid:1000")
    db = headless.authority.persistence
    sid = "forwarded-bot-chat"
    seed_session(db, sid, "tui", user_id=stamp, title="Bot Chat")
    seed_messages(db, sid, {"role": "user", "content": "existing project context"})
    login = peer._local_owner(headless.authority, FORWARDED)
    try:
        if open_first:
            await peer._rpc(login, "session.resume", session_id=sid)
        record = await _deliver(headless.authority, "4", connection=login)
        assert record["session_id"] == sid
        assert db.sessions.get_by_title("Bot Chat").session_id == sid
        assert _session_ids(db) == [sid]
    finally:
        await login.close()


@pytest.mark.asyncio
async def test_a_forwarded_login_never_moves_the_owners_chat_aside(headless, monkeypatch):
    # The owner's own Bot Chat stays where it is when a forwarded login that cannot adopt it
    # delivers: that delivery is refused, the owner's history keeps the title.
    import gateway.session_worker_reservation as reservation
    from persistence.core.errors import AuthorityRefusal

    monkeypatch.setattr(reservation, "owner_worker_subject", lambda: "uid:1000")
    db = headless.authority.persistence
    sid = "owner-bot-chat"
    seed_session(db, sid, "tui", user_id="basic:alice", title="Bot Chat")
    seed_messages(db, sid, {"role": "user", "content": "existing project context"})
    login = peer._local_owner(headless.authority, FORWARDED)
    try:
        with pytest.raises(AuthorityRefusal) as refused:
            await _deliver(headless.authority, "5", connection=login)
        assert refused.value.reason == "not_found"
        assert db.sessions.get_by_title("Bot Chat").session_id == sid
        assert _session_ids(db) == [sid]
    finally:
        await login.close()
