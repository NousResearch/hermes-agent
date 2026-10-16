"""A durable messaging conversation remains resumable through Desktop after restart."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_desktop_resume_restores_a_durable_messaging_session_after_restart(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    import hermes_state_runtime as rt

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread-1", user_id="user-1", chat_type="thread")

    first_store = SessionStore(tmp_path / "sessions", GatewayConfig())
    entry = first_store.get_or_create_session(source)
    first_store._db.append_message(entry.session_id, role="user", content="before restart")
    first_store._db.publish_compression_child(
        parent_session_id=entry.session_id,
        child_session_id="compressed-tip",
        source="discord",
        messages=[{"role": "assistant", "content": "compressed summary"}],
        require_compression_lease=False,
    )
    first_store.advance_compression_session(entry.session_key, entry.session_id, "compressed-tip")
    first_store._db.append_message("compressed-tip", role="user", content="after restart")
    first_store.close_all_db_handles()

    cold_store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=cold_store, _session_db=cold_store._db, adapters={}, _draining=False)
    authority = SessionAuthority(
        runner,
        profile_id="owned",
        instance_id="restarted",
        db=cold_store._db,
        epoch=rt.begin_runtime_epoch(cold_store._db, instance_id="restarted"),
    )
    owner = AuthorityConnection(authority, object(), {"user_id": "human"})

    try:
        response = await owner.dispatch(
            {"id": "resume", "method": "session.resume", "params": {"session_id": entry.session_id}}
        )
        assert response["result"]["stored_session_id"] == entry.session_id
        assert [message["content"] for message in response["result"]["messages"]] == [
            "compressed summary", "after restart"
        ]
    finally:
        await owner.close()
        cold_store.close_all_db_handles()
