"""An automatic empty-context reset keeps accepted native followers on their FIFO owner."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest_asyncio.fixture
async def native(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(
        PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False)
    )

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="fixture-reply")

    monkeypatch.setattr(adapter, "send", send)
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="fixture"
    )
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    try:
        yield SimpleNamespace(
            runner=runner, adapter=adapter, authority=authority, source=source
        )
    finally:
        from gateway.session_runtime_workers import mutation_tasks

        await asyncio.gather(*mutation_tasks(authority))
        authority.db.close()


async def _admit(native, *texts):
    events = [
        MessageEvent(text=text, source=native.source, message_id=text) for text in texts
    ]
    receipts = []
    for event in events:
        with native_callback(native.runner, event, get_hermes_home()):
            receipts.append(await native.authority.admit_native(event))
    return events, receipts


@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [False, True])
async def test_exhaustion_preserves_accepted_native_fifo(
    native, monkeypatch, compressed
):
    from gateway.session_results import execution_result

    runner, authority = native.runner, native.authority
    events, receipts = await _admit(native, "first", "second", "third")
    ref = receipts[0].ref
    db = authority.db
    db.append_message(ref.session_id, "user", "earlier history")
    db.append_message(ref.session_id, "assistant", "earlier reply")
    executed = []
    reset_target = None

    async def turn(event, source, session_key, run_generation):
        nonlocal reset_target
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append((event.text, entry.session_id))
        if event.text == "first":
            result = {"failed": True, "compression_exhausted": True, "messages": []}
            captured = execution_result.get()
            assert captured is not None
            captured["result"] = result
            if compressed:
                child = entry.session_id + "-compressed"
                db.publish_compression_child(
                    parent_session_id=entry.session_id,
                    child_session_id=child,
                    source="discord",
                    messages=[{"role": "user", "content": "still oversized"}],
                    require_compression_lease=False,
                )
                entry = runner.session_store.advance_compression_session(
                    session_key, entry.session_id, child
                )
            _, fresh = await runner._hmwa_compression_exhaustion_reset(
                result,
                "context exhausted",
                entry,
                session_key,
                source,
            )
            reset_target = fresh.session_id
        else:
            assert entry.session_id == reset_target
            assert runner.session_store.load_transcript(entry.session_id) == []
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    try:
        await authority._drain(ref)
        assert [text for text, _ in executed] == ["first", "second", "third"]
        from gateway.session_ingress import executing_claim

        assert executing_claim.get() is None
        assert reset_target != ref.session_id
        rows = list_session_admissions(
            db, session_id=ref.session_id, pending_only=False
        )
        assert [(row["admission_id"], row["status"]) for row in rows] == [
            (receipt.admission_id, "terminal") for receipt in receipts
        ]
        # A transport retry after reset must still resolve to the accepted receipt.
        with native_callback(runner, events[1], get_hermes_home()):
            retry = await authority.admit_native(events[1])
        assert retry.admission_id == receipts[1].admission_id
        assert [row["content"] for row in db.get_messages(ref.session_id)] == [
            "earlier history",
            "earlier reply",
        ]
        assert db.delete_session(reset_target, exclude_active_write_guards=True)
        assert db.get_session(reset_target) is None
        with db._read_ctx() as conn:
            assert not conn.execute(
                "SELECT 1 FROM state_meta WHERE key GLOB ?",
                ("gateway.native_context_reset.v1.*",),
            ).fetchone()
    finally:
        await asyncio.sleep(0)


@pytest_asyncio.fixture
async def claimed(native):
    from hermes_state_runtime import claim_session_input

    _, receipts = await _admit(native, "first", "second", "third")
    ref = receipts[0].ref
    row = claim_session_input(
        native.authority.db, epoch=native.authority.epoch, session_id=ref.session_id
    )
    route = native.authority.sessions[ref.session_id].route
    return SimpleNamespace(
        ref=ref,
        row=row,
        route=route,
        entry=native.runner.session_store.lookup_by_session_key(route),
    )


async def _reset_claim(native, claimed):
    from gateway.session_ingress import executing_claim

    token = executing_claim.set((native.authority, claimed.ref, claimed.row))
    result = {
        "failed": True,
        "compression_exhausted": True,
        "last_prompt_tokens": 90000,
    }
    try:
        _, entry = await native.runner._hmwa_compression_exhaustion_reset(
            result, "context exhausted", claimed.entry, claimed.route, native.source
        )
        return entry, result
    finally:
        executing_claim.reset(token)


@pytest.mark.asyncio
async def test_context_reset_rolls_back_after_child_write_failure(
    native, claimed, monkeypatch
):
    db = native.authority.db
    publish = db._publish_child_session_row

    def fail_after_insert(*args, **kwargs):
        publish(*args, **kwargs)
        raise OSError("fixture disk failure")

    monkeypatch.setattr(db, "_publish_child_session_row", fail_after_insert)
    with pytest.raises(OSError, match="fixture disk failure"):
        await _reset_claim(native, claimed)
    assert db.get_compression_tip(claimed.ref.session_id) == claimed.ref.session_id
    assert db.get_session(claimed.ref.session_id)["end_reason"] is None
    assert (
        native.runner.session_store.peek_session_id(claimed.route)
        == claimed.ref.session_id
    )
    with db._read_ctx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert not conn.execute(
            "SELECT 1 FROM state_meta WHERE key GLOB ?",
            ("gateway.native_context_reset.v1.*",),
        ).fetchone()
    assert [
        row["status"]
        for row in list_session_admissions(db, session_id=claimed.ref.session_id)
    ] == ["started", "queued", "queued"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["stale_generation", "stale_epoch", "unknown_execution"]
)
async def test_context_reset_refuses_a_claim_it_no_longer_owns(
    native, claimed, failure
):
    from dataclasses import replace
    from hermes_state_runtime import (
        RuntimeStoreError,
        begin_runtime_epoch,
        recover_session_inputs,
    )

    authority, db = native.authority, native.authority.db
    if failure == "stale_generation":
        claimed.row = dict(claimed.row, generation=claimed.row["generation"] + 1)
    elif failure == "stale_epoch":
        authority.epoch -= 1
    else:
        authority.epoch = begin_runtime_epoch(db, instance_id="restarted")
        recover_session_inputs(db, epoch=authority.epoch)
    original = replace(claimed.entry)
    with pytest.raises(RuntimeStoreError) as exc:
        await _reset_claim(native, claimed)
    assert exc.value.reason == failure
    assert native.runner.session_store.lookup_by_session_key(claimed.route) == original
    assert db.get_compression_tip(claimed.ref.session_id) == claimed.ref.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["live", "restart"])
async def test_committed_reset_recovers_fresh_fields_from_a_stale_route(
    native, claimed, monkeypatch, recovery
):
    from dataclasses import replace
    from gateway.session import SessionStore

    store, db = native.runner.session_store, native.authority.db
    claimed.entry.last_prompt_tokens = 90000
    claimed.entry.model_override = {"provider": "custom", "model": "fixture/old"}
    stale = replace(claimed.entry)
    store._save_entries()

    def fail_save():
        raise OSError("fixture routing save failure")

    with monkeypatch.context() as fault:
        fault.setattr(store, "_save_entries", fail_save)
        with pytest.raises(OSError, match="fixture routing save failure"):
            await _reset_claim(native, claimed)
    target = db.get_compression_tip(claimed.ref.session_id)
    assert target != claimed.ref.session_id
    if recovery == "live":
        store._entries[claimed.route] = stale
    else:
        store = SessionStore(get_hermes_home() / "sessions", native.runner.config)
        store._db = db
    entry = store.get_or_create_session(native.source)
    assert entry.session_id == target
    assert entry.is_fresh_reset
    assert entry.last_prompt_tokens == 0
    assert entry.model_override is None
    assert store.load_transcript(target) == []
    assert [
        row["status"]
        for row in list_session_admissions(db, session_id=claimed.ref.session_id)
    ] == ["started", "queued", "queued"]


@pytest.mark.asyncio
async def test_failed_turn_statistics_do_not_overwrite_the_empty_context(
    native, claimed
):
    entry, result = await _reset_claim(native, claimed)
    await native.runner._hmwa_persist_turn_transcript(
        event=MessageEvent(text="first", source=native.source),
        source=native.source,
        session_entry=entry,
        session_key=claimed.route,
        agent_result=result,
        agent_messages=[],
        prepared=SimpleNamespace(history=[]),
        response="context exhausted",
        agent_failed_early=True,
        hidden_reasoning_incomplete=False,
        is_context_overflow_failure=True,
    )
    assert (
        native.runner.session_store.lookup_by_session_key(
            claimed.route
        ).last_prompt_tokens
        == 0
    )
    assert native.runner.session_store.load_transcript(entry.session_id) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [{}, {"compression_exhausted": True, "compression_deferred": True}]
)
async def test_normal_or_deferred_result_keeps_the_existing_context(native, result):
    entry = native.runner.session_store.get_or_create_session(native.source)
    response, unchanged = await native.runner._hmwa_compression_exhaustion_reset(
        result, "reply", entry, entry.session_key, native.source
    )
    assert response == "reply"
    assert unchanged is entry
    assert (
        native.runner.session_store.peek_session_id(entry.session_key)
        == entry.session_id
    )


@pytest.mark.asyncio
async def test_reset_rejects_a_different_profile_database(
    native, claimed, monkeypatch, tmp_path
):
    from hermes_state import SessionDB
    from hermes_state_runtime import RuntimeStoreError

    other = SessionDB(tmp_path / "other-profile.db")
    try:
        monkeypatch.setattr(
            native.runner.session_store, "_db_for_key", lambda route: other
        )
        with pytest.raises(RuntimeStoreError) as exc:
            await _reset_claim(native, claimed)
        assert exc.value.reason == "profile_mismatch"
        assert (
            native.authority.db.get_compression_tip(claimed.ref.session_id)
            == claimed.ref.session_id
        )
        with other._read_ctx() as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
    finally:
        other.close()


@pytest.mark.asyncio
async def test_cancelled_reset_caller_cannot_abandon_committed_publication(
    native, claimed, monkeypatch
):
    import threading
    from gateway import session_native_context_reset as reset
    from gateway.session_runtime_workers import mutation_tasks
    from tests.waiting import async_wait_until

    committed, release = threading.Event(), threading.Event()
    commit = reset._commit

    def paused_after_commit(*args):
        result = commit(*args)
        committed.set()
        assert release.wait(15), "test did not release the committed publisher"
        return result

    monkeypatch.setattr(reset, "_commit", paused_after_commit)
    caller = asyncio.create_task(_reset_claim(native, claimed))
    try:
        await async_wait_until(committed.is_set, timeout=15, what="reset commit")
        # A new input can heal the committed child before its publisher resumes.
        _, receipts = await _admit(native, "fourth")
        assert receipts[0].ref == claimed.ref
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert mutation_tasks(native.authority)
    finally:
        release.set()
        await asyncio.gather(*mutation_tasks(native.authority))
        if not caller.done():
            caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
    target = native.authority.physical_target(claimed.ref)
    assert native.runner.session_store.peek_session_id(claimed.route) == target
    assert target != claimed.ref.session_id
    rows = list_session_admissions(
        native.authority.db, session_id=claimed.ref.session_id
    )
    assert [row["request_id"] for row in rows] == ["first", "second", "third", "fourth"]
    assert [row["status"] for row in rows] == ["started", "queued", "queued", "queued"]


def _completion(native, route, target, identity):
    return MessageEvent(
        text=identity,
        source=native.source,
        internal=True,
        message_id=identity,
        metadata={"gateway_session_key": route, "gateway_session_id": target},
    )


@pytest.mark.asyncio
async def test_reset_retires_old_completions_but_keeps_route_wakes_and_humans(
    native, monkeypatch
):
    from gateway.session_automation import admit_route_wake
    from gateway.session_results import execution_result
    from hermes_state_runtime import RuntimeStoreError

    runner, authority = native.runner, native.authority
    _, receipts = await _admit(native, "first")
    ref = receipts[0].ref
    route = authority.sessions[ref.session_id].route
    old = _completion(native, route, ref.session_id, '["process","old"]')
    prior = await authority.admit_automation(native.adapter, old, old.message_id)
    assert await admit_route_wake(
        runner, native.adapter, native.source, "wake", "kanban-wake"
    )
    await _admit(native, "human")
    executed = []

    async def turn(event, source, key, generation):
        executed.append(event.text)
        if event.text == "first":
            entry = runner.session_store.lookup_by_session_key(key)
            result = {"failed": True, "compression_exhausted": True, "messages": []}
            captured = execution_result.get()
            assert captured is not None
            captured["result"] = result
            _, fresh = await runner._hmwa_compression_exhaustion_reset(
                result, "context exhausted", entry, key, source
            )
            late = _completion(native, route, ref.session_id, '["process","late-old"]')
            with pytest.raises(RuntimeStoreError, match="admission_conflict"):
                await authority.admit_automation(native.adapter, late, late.message_id)
            current = _completion(
                native, route, fresh.session_id, '["process","fresh"]'
            )
            await authority.admit_automation(
                native.adapter, current, current.message_id
            )
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await authority._drain(ref)
    assert executed == ["first", "wake", "human", '["process","fresh"]']
    retry = await authority.admit_automation(native.adapter, old, old.message_id)
    assert retry.admission_id == prior.admission_id
    assert retry.outcome == "cancelled"
    rows = list_session_admissions(
        authority.db, session_id=ref.session_id, pending_only=False
    )
    assert all(row["status"] == "terminal" for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_reset", [False, True])
async def test_completion_snapshot_obeys_empty_reset_not_ordinary_compression(
    native, claimed, empty_reset
):
    from gateway.session_automation import snapshot_automation
    from gateway.session_native_context_reset import automation_write_guard
    from hermes_state_runtime import RuntimeStoreError, admit_session_input

    event = _completion(
        native, claimed.route, claimed.entry.session_id, '["process","race"]'
    )
    payload, _ = snapshot_automation(
        native.authority, native.adapter, event, event.message_id
    )
    guard = automation_write_guard(native.authority, event, event.message_id)
    if empty_reset:
        await _reset_claim(native, claimed)
    else:
        child = claimed.entry.session_id + "-compressed"
        native.authority.db.publish_compression_child(
            parent_session_id=claimed.entry.session_id,
            child_session_id=child,
            source="discord",
            messages=[{"role": "user", "content": "summary"}],
            require_compression_lease=False,
        )
        native.runner.session_store.advance_compression_session(
            claimed.route, claimed.entry.session_id, child
        )

    def commit():
        return admit_session_input(
            native.authority.db,
            epoch=native.authority.epoch,
            principal_id="automation:" + claimed.route,
            session_id=claimed.ref.session_id,
            request_id=event.message_id,
            payload=payload,
            _authorize_write=guard,
        )

    if empty_reset:
        with pytest.raises(RuntimeStoreError, match="admission_conflict"):
            commit()
    else:
        assert commit()["status"] == "queued"
    assert len(
        list_session_admissions(native.authority.db, session_id=claimed.ref.session_id)
    ) == (3 if empty_reset else 4)
