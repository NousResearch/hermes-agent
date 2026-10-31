"""REST display reads follow a canonical local reset without reviving its old history."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [False, True])
async def test_local_reset_history_reads_current_transcript(tmp_path, monkeypatch, compressed):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from hermes_cli.web_routers import sessions
    from hermes_state_local import local_receipt

    monkeypatch.setattr(run, "_load_gateway_config", dict)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="fixture")
    actor = Principal("owner", "default", frozenset({"session:create", "session:read"}), "socket")
    ref = create_local_session(authority, actor, {
        "request_id": "history", "source": "gui", "cwd": str(tmp_path), "model": "frozen", "toolsets": [],
    })
    db = authority.db
    old_prompt = db.append_message(ref.session_id, "user", "before reset")
    db.append_message(ref.session_id, "assistant", "old reply")
    reset = store.reset_session(authority.sessions[ref.session_id].route)
    history = [{"role": "user", "content": "after reset"}, {"role": "assistant", "content": "current reply"}]
    for message in history:
        db.append_message(reset.session_id, **message)
    if compressed:
        assert db.try_acquire_compression_lock(reset.session_id, "fixture")
        try:
            db.publish_compression_child(
                parent_session_id=reset.session_id, child_session_id="compressed", source="gui",
                messages=history, compression_lock_holder="fixture",
            )
        finally:
            db.release_compression_lock(reset.session_id, "fixture")
    target = local_receipt(db, ref.session_id)["entry"]["session_id"]
    before = local_receipt(db, ref.session_id)
    monkeypatch.setattr(sessions, "_with_db", lambda profile, fn, read_only: fn(db))
    monkeypatch.setattr(sessions, "_cron_default_profile", lambda: "default")

    page = await sessions.get_session_messages(
        ref.session_id, limit=120, offset=0, order="latest", include_compacted=True, inline_images=False,
    )
    assert [row["content"] for row in page["messages"]] == ["after reset", "current reply"]
    assert page["session_id"] == ref.session_id
    timeline = await sessions.get_session_timeline(ref.session_id, limit=10, after_row_id=0)
    assert [row["preview"] for row in timeline["entries"]] == ["after reset"]
    assert timeline["session_id"] == ref.session_id
    around = await sessions.get_session_messages_around(
        ref.session_id, row_id=timeline["entries"][0]["row_id"], limit=120,
    )
    assert [row["content"] for row in around["messages"]] == ["after reset", "current reply"]
    assert around["session_id"] == ref.session_id
    with pytest.raises(HTTPException) as exc:
        await sessions.get_session_messages_around(ref.session_id, row_id=old_prompt, limit=120)
    assert exc.value.status_code == 404
    assert local_receipt(db, ref.session_id) == before
    assert db.get_session(target) is not None
    assert db.get_session(ref.session_id)["end_reason"] == "session_reset"
    assert [row["content"] for row in db.get_messages(ref.session_id)] == ["before reset", "old reply"]


@pytest.mark.asyncio
async def test_legacy_reset_without_local_receipt_keeps_exact_history(tmp_path, monkeypatch):
    from hermes_cli.web_routers import sessions
    from hermes_state import SessionDB

    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("parent", source="desktop")
        db.append_message("parent", "user", "parent history")
        db.end_session("parent", "session_reset")
        db.create_session("child", source="desktop", parent_session_id="parent",
                          model_config={"_reset_from": "parent"})
        db.append_message("child", "user", "child history")
        monkeypatch.setattr(sessions, "_with_db", lambda profile, fn, read_only: fn(db))
        monkeypatch.setattr(sessions, "_cron_default_profile", lambda: "default")
        for sid in ("parent", "child"):
            page = await sessions.get_session_messages(
                sid, limit=120, offset=0, order="latest", include_compacted=True, inline_images=False,
            )
            assert page["session_id"] == sid
            assert [row["content"] for row in page["messages"]] == [f"{sid} history"]
            timeline = await sessions.get_session_timeline(sid, limit=10, after_row_id=0)
            assert timeline["session_id"] == sid
            around = await sessions.get_session_messages_around(
                sid, row_id=timeline["entries"][0]["row_id"], limit=120,
            )
            assert around["session_id"] == sid
            assert [row["content"] for row in around["messages"]] == [f"{sid} history"]
