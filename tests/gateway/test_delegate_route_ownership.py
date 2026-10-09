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


def test_inbound_restores_a_poisoned_route_without_waiting_on_the_child_lease(discord_store):
    """Recreate the legacy on-disk hijack, including a still-running delegate's lease."""
    from gateway.session import SessionSource, SessionStore

    store, parent, db = discord_store
    source = parent.origin
    assert source is not None
    sibling_source = SessionSource(
        platform=source.platform, chat_id="other-channel", chat_type="thread",
        thread_id="other-thread", user_id=source.user_id,
    )
    sibling = store.get_or_create_session(sibling_source)
    db.create_session("delegate-child", source="subagent", parent_session_id=parent.session_id)
    db._write_sql(
        "UPDATE sessions SET source = ?, session_key = ?, user_id = ?, chat_id = ?, "
        "chat_type = ?, thread_id = ? WHERE id = ?",
        ("discord", parent.session_key, source.user_id, source.chat_id,
         source.chat_type, source.thread_id, "delegate-child"),
    )
    db.end_session(parent.session_id, "session_switch")
    with store._lock:
        store._replace_route_locked(
            parent.session_key, parent, "delegate-child", parent.updated_at,
            display_name=parent.display_name, prompt_pin=parent.prompt_pin,
        )
    assert db.try_acquire_session_turn_lease("delegate-child", "worker", ttl_seconds=60)

    # The next normal Discord inbound (also used by idle /stop) must not inherit the child's
    # lease or mutate that independent execution row.
    restored = store.get_or_create_session(source)

    assert restored.session_id == parent.session_id
    assert db.get_session(parent.session_id)["end_reason"] is None
    child_row = db.get_session("delegate-child")
    assert child_row["ended_at"] is None
    assert child_row["source"] == "subagent" and child_row["session_key"] is None
    assert not db.try_acquire_session_turn_lease("delegate-child", "human", ttl_seconds=60)
    assert db.try_acquire_session_turn_lease(parent.session_id, "human", ttl_seconds=60)
    assert store.lookup_by_session_key(sibling.session_key).session_id == sibling.session_id

    # Durable routing has to stay repaired after a gateway restart.
    restarted = SessionStore(store.sessions_dir, store.config)
    assert restarted.get_or_create_session(source).session_id == parent.session_id
    assert restarted.lookup_by_session_key(sibling.session_key).session_id == sibling.session_id


def test_unverifiable_poisoned_route_opens_a_fresh_chat_without_ending_child(discord_store):
    """An explicit boundary forbids guessing the old owner, but cannot leave the chat bricked."""
    store, parent, db = discord_store
    source = parent.origin
    assert source is not None
    db.create_session("delegate-child", source="subagent", parent_session_id=parent.session_id)
    db._write_sql(
        "UPDATE sessions SET source = ?, session_key = ?, user_id = ?, chat_id = ?, "
        "chat_type = ?, thread_id = ? WHERE id = ?",
        ("discord", parent.session_key, source.user_id, source.chat_id,
         source.chat_type, source.thread_id, "delegate-child"),
    )
    db.end_session(parent.session_id, "session_reset")
    with store._lock:
        store._replace_route_locked(parent.session_key, parent, "delegate-child", parent.updated_at)

    fresh = store.get_or_create_session(source)

    assert fresh.session_id not in {parent.session_id, "delegate-child"}
    assert db.get_session("delegate-child")["ended_at"] is None
    assert db.get_session(fresh.session_id)["source"] == "discord"


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


@pytest.mark.asyncio
async def test_child_completion_reaches_owner_after_its_chat_compresses(discord_store):
    """The child retains its pre-compression parent while the chat route moves to its tip."""
    from gateway.session import AsyncSessionStore
    from hermes_state import AsyncSessionDB

    store, parent, db = discord_store
    original_owner_id = parent.session_id
    db.create_session("delegate-child", source="subagent", parent_session_id=original_owner_id)
    db.end_session(original_owner_id, "compression")
    db.create_session(
        "chat-tip", source="discord", parent_session_id=original_owner_id,
        session_key=parent.session_key, chat_id=parent.origin.chat_id,
        chat_type=parent.origin.chat_type, thread_id=parent.origin.thread_id,
        user_id=parent.origin.user_id,
    )
    tip = store.advance_compression_session(parent.session_key, original_owner_id, "chat-tip")
    assert tip is not None and tip.session_id == "chat-tip"
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)

    resolved = await runner._resolve_async_delegation_session(tip, "delegate-child")

    assert resolved is not None and resolved.session_id == "chat-tip"
    assert db.get_session(original_owner_id)["end_reason"] == "compression"
    assert db.get_session("delegate-child")["source"] == "subagent"
