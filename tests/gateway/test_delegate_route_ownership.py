"""A delegate's process completion cannot turn its child into a human chat route.

Regression for the Discord incidents in #131578 and #134522.
"""

import pytest

from gateway.run import GatewayRunner


@pytest.fixture
def discord_store(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="channel", chat_type="thread",
        thread_id="thread", user_id="person",
    )
    entry = store.get_or_create_session(source)
    return store, entry, store._db_for_key(entry.session_key)


def test_switch_session_rejects_a_delegate_child_before_ending_the_chat(discord_store):
    store, parent, db = discord_store
    db.create_session(
        "delegate-child", source="subagent", parent_session_id=parent.session_id,
        model_config={"_delegate_from": parent.session_id},
    )

    result = store.switch_session(parent.session_key, "delegate-child", expected_session_id=parent.session_id)

    assert result is None
    assert store.lookup_by_session_key(parent.session_key).session_id == parent.session_id
    assert db.get_session(parent.session_id)["ended_at"] is None
    child = db.get_session("delegate-child")
    assert child["source"] == child["created_source"] == "subagent"
    assert child["session_key"] is None


@pytest.mark.asyncio
async def test_nested_promoted_child_cannot_repoint_chat_via_process_completion(discord_store):
    from gateway.session import AsyncSessionStore
    from hermes_state import AsyncSessionDB

    store, parent, db = discord_store
    db.create_session("delegate-parent", source="subagent", parent_session_id=parent.session_id)
    db.create_session("delegate-child", source="subagent", parent_session_id="delegate-parent")
    # Simulate a previously corrupted installation: mutable source/key were overwritten, while
    # immutable created_source still records the child's real origin.
    db._write_sql(
        "UPDATE sessions SET source = ?, session_key = ? WHERE id = ?",
        ("discord", parent.session_key, "delegate-child"),
    )
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)

    resolved = await runner._resolve_async_delegation_session(parent, "delegate-child")

    assert resolved is parent
    assert store.lookup_by_session_key(parent.session_key).session_id == parent.session_id
    assert db.get_session(parent.session_id)["ended_at"] is None
    assert db.get_session("delegate-child")["created_source"] == "subagent"
