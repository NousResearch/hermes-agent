"""Durable delivery retries uncertain delegate ancestry without moving the route."""

import sqlite3
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from tools import async_delegation as ad


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["readiness", "claimed"])
@pytest.mark.parametrize("failed_ancestor", ["worker", "owner"])
async def test_transient_ownership_ancestor_failure_retries_durable_delivery_once(
    tmp_path, monkeypatch, phase, failed_ancestor, private_db_probe_cleanup,
):
    """An ancestor read error refunds the claim; recovery admits once to the owner.

    Exercise the real SQLite ledger, ownership walk, admission and route resolver.
    Only the transient database read and the external transport are substituted.
    Exactly-once here means adapter admission during retry/replay, not crash-atomic
    delivery or exactly-once model execution.
    """
    with _profile_runtime_scope(tmp_path):
        config = GatewayConfig(sessions_dir=tmp_path / "sessions")
        runner = GatewayRunner(config)
        store = runner.session_store
        replay_runner = None
        try:
            entry = store.get_or_create_session(SessionSource(
                platform=Platform.TELEGRAM, chat_id="ownership-retry", chat_type="dm",
            ))
            db = cast(Any, store._db)
            owner = entry.session_id
            db.create_session("worker", source="subagent", model_config={"_delegate_from": owner})
            db.create_session("nested-worker", source="subagent", model_config={"_delegate_from": "worker"})
            before = {sid: db.get_session(sid) for sid in (owner, "worker", "nested-worker")}
            ancestor = owner if failed_ancestor == "owner" else "worker"
            async_db = runner._session_db
            get_session = async_db.get_session
            walk_count = 0
            failures = []
            fault_enabled = True

            async def lookup_session(sid):
                nonlocal walk_count
                if sid == "nested-worker":
                    walk_count += 1
                if fault_enabled and sid == ancestor and (phase == "readiness" or walk_count >= 2):
                    failures.append(sid)
                    raise RuntimeError("temporary ownership ancestor lookup failure")
                return await get_session(sid)

            monkeypatch.setattr(async_db, "get_session", lookup_session)
            admitted = []

            async def accept(event):
                assert event.source.chat_id == "ownership-retry"
                assert event.metadata["gateway_session_key"] == entry.session_key
                assert event.metadata["gateway_session_id"] == owner
                current = store.lookup_by_session_key(entry.session_key)
                assert current is not None
                resolved = await runner._resolve_async_delegation_session(
                    current, event.metadata["gateway_session_id"],
                )
                assert resolved is not None and resolved.session_id == owner
                admitted.append(resolved.session_id)
                event._gateway_accepted = True

            adapter = cast(BasePlatformAdapter, SimpleNamespace(handle_message=accept))
            runner.adapters[Platform.TELEGRAM] = adapter
            event = {
                "type": "async_delegation", "delegation_id": "owner-retry",
                "session_key": entry.session_key, "parent_session_id": "nested-worker",
                "dispatched_at": 1.0, "status": "completed", "summary": "completed result",
            }
            ad._persist_dispatch(event)
            ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})

            assert await runner._deliver_completion_notification("completed result", event) is False
            assert failures == [ancestor]
            assert walk_count == (1 if phase == "readiness" else 2)
            pending = ad.get_durable_delegation(event["delegation_id"])
            assert pending is not None
            assert (pending["delivery_state"], pending["delivery_attempts"]) == ("pending", 0)
            with sqlite3.connect(ad._db_path()) as conn:
                assert conn.execute(
                    "SELECT delivery_claim FROM async_delegations WHERE delegation_id=?",
                    (event["delegation_id"],),
                ).fetchone() == (None,)
            assert admitted == []
            current = store.lookup_by_session_key(entry.session_key)
            assert current is not None and current.session_id == owner
            assert {sid: db.get_session(sid) for sid in before} == before

            fault_enabled = False
            assert await runner._deliver_completion_notification("completed result", event) is True
            delivered = ad.get_durable_delegation(event["delegation_id"])
            assert delivered is not None
            assert (delivered["delivery_state"], delivered["delivery_attempts"]) == ("delivered", 1)
            assert admitted == [owner]
            assert await runner._deliver_completion_notification("completed result", event) is None

            # A fresh runner has no lifecycle dedup cache: the durable ack prevents replay.
            replay_runner = GatewayRunner(config)
            replay_runner.adapters[Platform.TELEGRAM] = adapter
            assert await replay_runner._deliver_completion_notification("completed result", event) is None
            assert admitted == [owner]
            assert ad.get_durable_delegation(event["delegation_id"]) == delivered
            current = store.lookup_by_session_key(entry.session_key)
            replay_current = replay_runner.session_store.lookup_by_session_key(entry.session_key)
            assert current is not None and current.session_id == owner
            assert replay_current is not None and replay_current.session_id == owner
            assert {sid: db.get_session(sid) for sid in before} == before
        finally:
            if replay_runner is not None:
                replay_runner.session_store.close_all_db_handles()
                replay_runner.close_all_session_db_handles()
            store.close_all_db_handles()
            runner.close_all_session_db_handles()


@pytest.mark.asyncio
async def test_claimed_ownership_retries_do_not_spend_the_delivery_budget(
    tmp_path, monkeypatch, private_db_probe_cleanup,
):
    """More transient owner-read failures than the attempt budget leave the result pending; recovery admits it once."""
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        try:
            entry = store.get_or_create_session(SessionSource(
                platform=Platform.TELEGRAM, chat_id="ownership-budget", chat_type="dm",
            ))
            owner = entry.session_id
            cast(Any, store._db).create_session("worker", source="subagent", model_config={"_delegate_from": owner})
            async_db = runner._session_db
            get_session = async_db.get_session
            walks = 0
            failing = True

            async def lookup_session(sid):
                nonlocal walks
                if sid == "worker":
                    walks += 1
                # Each delivery walks twice, readiness then the claimed preflight; only the second fails.
                if failing and sid == owner and walks % 2 == 0:
                    raise RuntimeError("temporary owner lookup failure")
                return await get_session(sid)

            monkeypatch.setattr(async_db, "get_session", lookup_session)
            admitted = []

            async def accept(event):
                admitted.append(event.metadata["gateway_session_id"])
                event._gateway_accepted = True

            runner.adapters[Platform.TELEGRAM] = cast(BasePlatformAdapter, SimpleNamespace(handle_message=accept))
            event = {
                "type": "async_delegation", "delegation_id": "owner-budget",
                "session_key": entry.session_key, "parent_session_id": "worker",
                "dispatched_at": 1.0, "status": "completed", "summary": "completed result",
            }
            ad._persist_dispatch(event)
            ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})

            for _ in range(ad._MAX_DELIVERY_ATTEMPTS + 1):
                assert await runner._deliver_completion_notification("completed result", event) is False
            pending = ad.get_durable_delegation(event["delegation_id"])
            assert pending is not None
            assert (pending["delivery_state"], pending["delivery_attempts"]) == ("pending", 0)
            assert admitted == []

            failing = False
            assert await runner._deliver_completion_notification("completed result", event) is True
            assert len(admitted) == 1
            delivered = ad.get_durable_delegation(event["delegation_id"])
            assert delivered is not None and delivered["delivery_state"] == "delivered"
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()


async def _admit_with_session_rows_unreadable(runner, store, entry, monkeypatch, delegation_id, pinned):
    """Deliver one completion; every session-row read raises once the adapter has accepted it."""
    async_db = runner._session_db
    get_session = async_db.get_session
    admitted_phase = False

    async def lookup_session(sid):
        if admitted_phase:
            raise RuntimeError("session rows unreadable after admission")
        return await get_session(sid)

    monkeypatch.setattr(async_db, "get_session", lookup_session)
    resolved = []

    async def accept(event):
        nonlocal admitted_phase
        admitted_phase = True
        current = store.lookup_by_session_key(entry.session_key)
        assert current is not None
        result = await runner._resolve_async_delegation_session(
            current, event.metadata["gateway_session_id"],
            proven_session_id=event.metadata.get("gateway_proven_session_id", ""),
        )
        resolved.append(result.session_id if result else None)
        event._gateway_accepted = True

    runner.adapters[Platform.TELEGRAM] = cast(BasePlatformAdapter, SimpleNamespace(handle_message=accept))
    event = {
        "type": "async_delegation", "delegation_id": delegation_id,
        "session_key": entry.session_key, "parent_session_id": pinned,
        "dispatched_at": 1.0, "status": "completed", "summary": "completed result",
    }
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})
    assert await runner._deliver_completion_notification("completed result", event) is True
    delivered = ad.get_durable_delegation(delegation_id)
    assert delivered is not None and delivered["delivery_state"] == "delivered"
    return resolved


@pytest.mark.asyncio
async def test_admitted_delegate_completion_reaches_owner_without_reading_ownership_again(
    tmp_path, monkeypatch, private_db_probe_cleanup,
):
    """Admission acknowledges the durable claim, so the resolver must reach the owner preflight proved
    even when no session row can be read afterwards."""
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        try:
            entry = store.get_or_create_session(SessionSource(
                platform=Platform.TELEGRAM, chat_id="ownership-admitted", chat_type="dm",
            ))
            db = cast(Any, store._db)
            owner = entry.session_id
            db.create_session("worker", source="subagent", model_config={"_delegate_from": owner})
            db.create_session("nested-worker", source="subagent", model_config={"_delegate_from": "worker"})

            resolved = await _admit_with_session_rows_unreadable(
                runner, store, entry, monkeypatch, "owner-admitted", "nested-worker")

            assert resolved == [owner]
            current = store.lookup_by_session_key(entry.session_key)
            assert current is not None and current.session_id == owner
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()


@pytest.mark.asyncio
async def test_admitted_completion_reaches_compressed_owners_tip_without_reading_lineage_again(
    tmp_path, monkeypatch, private_db_probe_cleanup,
):
    """A compressed owner is proven through its live tip before admission; delivery reads no lineage after it."""
    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        try:
            entry = store.get_or_create_session(SessionSource(
                platform=Platform.TELEGRAM, chat_id="ownership-compressed", chat_type="dm",
            ))
            db = cast(Any, store._db)
            owner = entry.session_id
            db.create_session("worker", source="subagent", model_config={"_delegate_from": owner})
            db.create_session("owner-tip", source="telegram", parent_session_id=owner)
            db.end_session(owner, end_reason="compression")
            assert store.advance_compression_session(entry.session_key, owner, "owner-tip") is not None

            resolved = await _admit_with_session_rows_unreadable(
                runner, store, entry, monkeypatch, "tip-admitted", "worker")

            assert resolved == ["owner-tip"]
            current = store.lookup_by_session_key(entry.session_key)
            assert current is not None and current.session_id == "owner-tip"
        finally:
            store.close_all_db_handles()
            runner.close_all_session_db_handles()
