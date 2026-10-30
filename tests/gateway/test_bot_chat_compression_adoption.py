"""Bot delivery tests adoption at the compression tip while protecting bound lineage rows."""

import pytest

from tests.gateway import test_peer_dm_bot_chat_headless as peer
from tests.gateway.test_bot_chat_unadoptable_local_title import _deliver
from tests.gateway.test_local_resume_compression_child import _compress
from tests.waiting import async_wait_until

headless = peer.headless


def _chain(db, root_stamp):
    db.create_session("root", source="tui", user_id=root_stamp)
    db.set_session_title("root", "Bot Chat")
    db.append_message("root", "user", "original context")
    _compress(db, "root", "tip", "tui")


@pytest.mark.parametrize("root_stamp", [None, "basic:alice"])
@pytest.mark.parametrize(
    "column,value",
    [
        ("session_key", "agent:main:tui:dm:stale"),
        ("user_id", "uid:1001"),
        ("profile_name", "another-profile"),
    ],
)
@pytest.mark.asyncio
async def test_adoptable_root_does_not_pin_an_unadoptable_tip(
    headless, monkeypatch, root_stamp, column, value
):
    import gateway.session_worker_reservation as reservation

    monkeypatch.setattr(reservation, "owner_worker_subject", lambda: "uid:1000")
    db = headless.authority.db
    _chain(db, root_stamp)
    db._execute_write(
        lambda conn: conn.execute(
            f"UPDATE sessions SET {column}=? WHERE id='tip'", (value,)
        )
    )
    root_history = db.get_messages_as_conversation("root")
    tip_history = db.get_messages_as_conversation("tip")
    owner = peer._local_owner(headless.authority)
    try:
        refused = await owner.dispatch(
            {"id": 1, "method": "session.resume", "params": {"session_id": "tip"}}
        )
        assert refused["error"]["message"] == "not_found", refused
    finally:
        await owner.close()
    first = await _deliver(headless.authority, "a")
    new = first["session_id"]
    assert new not in {"root", "tip"}
    assert db.get_session_by_title("Bot Chat")["id"] == new
    assert db.get_session("root")["title"] == "Bot Chat (tui root)"
    assert db.get_compression_tip("root") == "tip"
    assert db.get_messages_as_conversation("root") == root_history
    assert db.get_messages_as_conversation("tip") == tip_history
    second = await _deliver(headless.authority, "b")
    assert second["session_id"] == new
    await async_wait_until(
        lambda: len(headless.executed) == 2, timeout=10, interval=0.05
    )
    assert [sid for sid, _, _ in headless.executed] == [new, new]


@pytest.mark.parametrize("root_stamp", ["uid:1001", "basic:alice"])
@pytest.mark.asyncio
async def test_adoptable_tip_keeps_the_compressed_conversation(
    headless, monkeypatch, root_stamp
):
    import gateway.session_worker_reservation as reservation

    monkeypatch.setattr(reservation, "owner_worker_subject", lambda: "uid:1000")
    db = headless.authority.db
    _chain(db, root_stamp)
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET user_id='basic:alice' WHERE id='tip'"
        )
    )
    history = db.get_messages_as_conversation("tip")
    first = await _deliver(headless.authority, "a")
    second = await _deliver(headless.authority, "b")
    assert first["session_id"] == second["session_id"] == "tip"
    await async_wait_until(
        lambda: len(headless.executed) == 2, timeout=10, interval=0.05
    )
    assert db.get_session_by_title("Bot Chat")["id"] == "root"
    assert db.get_messages_as_conversation("tip")[: len(history)] == history
    assert len(db._read_all("SELECT id FROM sessions")) == 2


@pytest.mark.parametrize("protected", ["root", "tip"])
@pytest.mark.parametrize("binding", ["chat_id", "policy", "legacy"])
@pytest.mark.asyncio
async def test_existing_binding_on_either_segment_prevents_displacement(
    headless, protected, binding
):
    from gateway.session_bot import displaceable_bot_chat
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX

    db = headless.authority.db
    _chain(db, "uid:foreign")
    if binding == "chat_id":
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET chat_id='local-bound' WHERE id=?", (protected,)
            )
        )
    else:
        prefix = POLICY_PREFIX if binding == "policy" else LEGACY_PREFIX
        db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO state_meta(key,value) VALUES(?,?)",
                (prefix + protected, "{}"),
            )
        )
    assert displaceable_bot_chat(headless.authority, db.get_session("root")) is None
    assert db.get_session_by_title("Bot Chat")["id"] == "root"
