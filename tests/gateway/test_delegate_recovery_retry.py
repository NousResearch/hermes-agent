"""Delegate route repair must preserve its proof across transient storage failures."""

import json
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


@pytest.fixture
def poisoned_route(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="channel", chat_type="thread",
        thread_id="thread", user_id="person",
    )
    owner = store.get_or_create_session(source)
    db = store._db_for_key(owner.session_key)
    db.create_session("delegate-child", source="subagent", parent_session_id=owner.session_id)
    db._write_sql(
        "UPDATE sessions SET source = ?, session_key = ?, user_id = ?, chat_id = ?, "
        "chat_type = ?, thread_id = ? WHERE id = ?",
        ("discord", owner.session_key, source.user_id, source.chat_id,
         source.chat_type, source.thread_id, "delegate-child"),
    )
    db.end_session(owner.session_id, "session_switch")
    with store._lock:
        store._replace_route_locked(
            owner.session_key, owner, "delegate-child", owner.updated_at,
        )
    yield store, owner, db
    store.close_all_db_handles()


@pytest.mark.parametrize("failure_point", ["ancestry", "owner_lookup", "reopen"])
def test_temporary_recovery_failure_preserves_route_until_retry(poisoned_route, monkeypatch, failure_point):
    store, owner, db = poisoned_route
    original_route = store.lookup_by_session_key(owner.session_key)
    method_name = {"ancestry": "_read_all", "owner_lookup": "get_session", "reopen": "reopen_session"}[
        failure_point
    ]
    original = getattr(db, method_name)

    def unavailable(*args, **kwargs):
        if failure_point != "owner_lookup" or args[0] == owner.session_id:
            raise OSError("temporary recovery storage failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(db, method_name, unavailable)
        with pytest.raises(RuntimeError, match="retry"):
            store.get_or_create_session(owner.origin)

    assert store.lookup_by_session_key(owner.session_key) is original_route
    assert original_route.session_id == "delegate-child"
    durable = store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())
    assert json.loads(durable[owner.session_key])["session_id"] == "delegate-child"
    assert db.get_session("delegate-child")["session_key"] == owner.session_key
    assert store.get_or_create_session(owner.origin).session_id == owner.session_id
    assert db.get_session("delegate-child")["ended_at"] is None


def test_failed_initial_db_load_does_not_admit_legacy_route(poisoned_route, monkeypatch):
    store, owner, _db = poisoned_route
    restarted = SessionStore(store.sessions_dir, store.config)
    with monkeypatch.context() as patch:
        patch.setattr(SessionStore, "_open_session_db_for_active_scope", lambda *_a, **_k: None)
        with pytest.raises(RuntimeError, match="provenance"):
            restarted.get_or_create_session(owner.origin)
        assert restarted.lookup_by_session_key(owner.session_key).session_id == "delegate-child"
    assert restarted.get_or_create_session(owner.origin).session_id == owner.session_id
    restarted.close_all_db_handles()


def test_new_jsonl_session_remains_usable_during_initial_db_outage(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(SessionStore, "_open_session_db_for_active_scope", lambda *_a, **_k: None)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="new-chat")
    created = store.get_or_create_session(source)
    assert store.get_or_create_session(source).session_id == created.session_id


@pytest.mark.parametrize("break_mirror", [False, True])
def test_failed_primary_repair_preserves_peer_proof_for_restart(poisoned_route, monkeypatch, break_mirror):
    store, owner, db = poisoned_route
    original_route = store.lookup_by_session_key(owner.session_key)

    def unavailable(*_args, **_kwargs):
        raise OSError("temporary routing write failure")

    with monkeypatch.context() as patch:
        patch.setattr(store._routing_db, "replace_gateway_routing_entries", unavailable)
        if break_mirror:
            patch.setattr(store, "_save_sessions_json", unavailable)
        with pytest.raises((OSError, RuntimeError)):
            store.get_or_create_session(owner.origin)

    assert store.lookup_by_session_key(owner.session_key) is original_route
    assert original_route.session_id == "delegate-child"
    assert db.get_session("delegate-child")["session_key"] == owner.session_key
    restarted = SessionStore(store.sessions_dir, store.config)
    assert restarted.get_or_create_session(owner.origin).session_id == owner.session_id
    assert db.get_session("delegate-child")["ended_at"] is None
    restarted.close_all_db_handles()


@pytest.mark.parametrize("boundary_point", ["after_proof", "before_reopen"])
def test_concurrent_owner_reset_survives_delegate_repair(poisoned_route, monkeypatch, boundary_point):
    """A separate lifecycle writer can reset after the ancestry snapshot was verified."""
    from hermes_state import SessionDB

    store, owner, db = poisoned_route
    original_route = store.lookup_by_session_key(owner.session_key)
    writer = SessionDB(db_path=db.db_path)

    def reset_from_another_writer():
        writer.reopen_session(owner.session_id)
        assert writer.promote_to_session_reset(owner.session_id, "session_reset")

    original_verdict = store._poisoned_delegate_route_verdict
    original_reopen = db.reopen_session

    def reset_after_proof(**kwargs):
        verdict = original_verdict(**kwargs)
        assert verdict.kind == "recoverable"
        reset_from_another_writer()
        return verdict

    def reset_before_reopen(*args, **kwargs):
        reset_from_another_writer()
        return original_reopen(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            if boundary_point == "after_proof":
                patch.setattr(store, "_poisoned_delegate_route_verdict", reset_after_proof)
            else:
                patch.setattr(db, "reopen_session", reset_before_reopen)
            with pytest.raises(RuntimeError, match="retry"):
                store.get_or_create_session(owner.origin)
        assert db.get_session(owner.session_id)["end_reason"] == "session_reset"
        assert store.lookup_by_session_key(owner.session_key) is original_route
        assert db.get_session("delegate-child")["session_key"] == owner.session_key
        fresh = store.get_or_create_session(owner.origin)
        assert fresh.session_id not in {owner.session_id, "delegate-child"}
        assert db.get_session(owner.session_id)["end_reason"] == "session_reset"
    finally:
        writer.close()


def _completion_runner(store, db):
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore
    from hermes_state import AsyncSessionDB

    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    runner._session_key_for_source = store._generate_session_key
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("pin_kind", ["child", "owner"])
async def test_completion_preparation_repairs_poisoned_route_before_acceptance(poisoned_route, pin_kind):
    from gateway.run_notifications_receipts import prepare_completion_owner

    store, owner, db = poisoned_route
    runner = _completion_runner(store, db)
    pin = "delegate-child" if pin_kind == "child" else owner.session_id
    event = SimpleNamespace(source=owner.origin, metadata={
        "gateway_session_key": owner.session_key, "gateway_session_id": pin,
    })
    assert await runner._classify_completion_target(pin, owner.session_key) == "deliver"
    assert await prepare_completion_owner(runner, event)
    assert event._completion_owner_receipt.session_id == owner.session_id
    assert store.peek_session_id(owner.session_key) == owner.session_id
    assert db.get_session("delegate-child")["session_key"] is None


@pytest.mark.asyncio
async def test_unproven_completion_cannot_mutate_poisoned_route(poisoned_route):
    from gateway.run_notifications_receipts import prepare_completion_owner

    store, owner, db = poisoned_route
    runner = _completion_runner(store, db)
    db.reopen_session(owner.session_id)
    assert db.promote_to_session_reset(owner.session_id)
    event = SimpleNamespace(source=owner.origin, metadata={
        "gateway_session_key": owner.session_key, "gateway_session_id": "delegate-child",
    })
    assert not await prepare_completion_owner(runner, event)
    assert store.peek_session_id(owner.session_key) == "delegate-child"
    assert db.get_session(owner.session_id)["end_reason"] == "session_reset"


@pytest.mark.asyncio
async def test_old_parent_preflight_preserves_a_boundary_on_the_routed_child(poisoned_route):
    store, owner, db = poisoned_route
    runner = _completion_runner(store, db)
    db.end_session("delegate-child", "session_reset")
    assert await runner._classify_completion_target(owner.session_id, owner.session_key) == "terminal"
    assert store.peek_session_id(owner.session_key) == "delegate-child"
    assert db.get_session("delegate-child")["end_reason"] == "session_reset"
