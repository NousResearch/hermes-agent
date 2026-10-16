"""Native messages queued behind a turn that exhausts its context run once, in order, on the fresh
session the auto-reset creates; the exhausted turn is not replayed and a transport retry of a
re-homed message still dedupes. Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over
SQLite; the turn is a stand-in. Adapted from helix4u's #139673 headline test."""
import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_contract import SessionRef
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter

_REAL_SCHEDULE = SessionAuthority._schedule  # the fixture parks scheduling; races restore it


@pytest_asyncio.fixture
async def native(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="fixture-reply")

    monkeypatch.setattr(adapter, "send", send)
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="fixture")
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    try:
        yield SimpleNamespace(runner=runner, adapter=adapter, authority=authority, source=source)
    finally:
        from gateway.session_runtime_workers import mutation_tasks
        await asyncio.gather(*mutation_tasks(authority))
        authority.db.close()


async def _admit(native, text, event=None):
    event = event or MessageEvent(text=text, source=native.source, message_id=text)
    with native_callback(native.runner, event, get_hermes_home()):
        return event, await native.authority.admit_native(event)


@pytest.mark.asyncio
async def test_exhaustion_reset_runs_queued_native_messages_on_the_fresh_session(native, monkeypatch):
    from gateway.session_results import execution_result
    runner, authority, db = native.runner, native.authority, native.authority.db
    admitted = [await _admit(native, text) for text in ("A", "B", "C")]
    old = admitted[0][1].ref
    executed, fresh = [], {}

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append((event.text, entry.session_id))
        captured = execution_result.get()
        if event.text == "A":
            result = {"failed": True, "compression_exhausted": True, "messages": []}
            captured["result"] = result
            _, new_entry = await runner._hmwa_compression_exhaustion_reset(
                result, "context exhausted", entry, session_key, source)
            fresh["sid"] = new_entry.session_id
            fresh["D"] = await _admit(native, "D")  # admitted after the reset, behind B and C
        else:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await authority._drain(old)
    new = fresh["sid"]
    assert new != old.session_id
    if new in authority.sessions:
        await authority._drain(SessionRef(authority.profile_id, new))
    assert executed == [("A", old.session_id), ("B", new), ("C", new), ("D", new)]
    old_rows = list_session_admissions(db, session_id=old.session_id, pending_only=False)
    assert [(r["request_id"], r["status"]) for r in old_rows] == [("A", "terminal")]
    new_rows = list_session_admissions(db, session_id=new, pending_only=False)
    assert [(r["request_id"], r["status"], r["outcome"]) for r in new_rows] == [
        (t, "terminal", "completed") for t in ("B", "C", "D")]
    # The re-home keeps each message's admission identity; a transport retry of B dedupes onto it.
    assert [r["admission_id"] for r in new_rows[:2]] == [admitted[1][1].admission_id, admitted[2][1].admission_id]
    _, retry = await _admit(native, "B", admitted[1][0])  # the same delivery, redelivered
    assert retry.admission_id == admitted[1][1].admission_id
    assert [t for t, _ in executed].count("B") == 1



@pytest.mark.asyncio
@pytest.mark.parametrize("redeliver", [True, False])
async def test_exhaustion_reset_is_ordered_with_redelivery_and_cleanup(native, monkeypatch, redeliver):
    """A redelivery of queued B that resolves the route to the fresh session mid-reset dedupes onto
    the moved row (no UNIQUE collision rolling the whole move back); and the reset schedules the
    moved rows only after the conversation cleanup and topic binding sync."""
    from gateway.session_results import execution_result
    runner, authority, db = native.runner, native.authority, native.authority.db
    loop = asyncio.get_running_loop()
    admitted = [await _admit(native, text) for text in ("A", "B", "C")]
    old = admitted[0][1].ref
    events, redelivery = [], {}
    real_reset = runner.session_store.reset_session

    def reset_then_redeliver(session_key, *args, **kwargs):
        entry = real_reset(session_key, *args, **kwargs)
        if redeliver:  # the transport redelivers B right after the route rotated
            redelivery["B"] = future = asyncio.run_coroutine_threadsafe(_admit(native, "B", admitted[1][0]), loop)
            try:
                future.result(timeout=1)  # times out when (correctly) ordered behind the move
            except TimeoutError:
                pass
        return entry

    monkeypatch.setattr(runner.session_store, "reset_session", reset_then_redeliver)
    monkeypatch.setattr(runner, "_clear_conversation_scope", lambda *a, **k: events.append("clear"))
    monkeypatch.setattr(runner, "_sync_telegram_topic_binding", lambda *a, **k: events.append("sync"))
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: events.append("schedule"))
    executed, fresh = [], {}

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append(event.text)
        captured = execution_result.get()
        if event.text == "A":
            result = {"failed": True, "compression_exhausted": True, "messages": []}
            captured["result"] = result
            _, new_entry = await runner._hmwa_compression_exhaustion_reset(
                result, "context exhausted", entry, session_key, source)
            fresh["sid"] = new_entry.session_id
        else:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await authority._drain(old)
    new = fresh["sid"]
    if redeliver:
        _, retry = await asyncio.wrap_future(redelivery["B"])
        assert retry.admission_id == admitted[1][1].admission_id
    else:
        assert events == ["clear", "sync", "schedule"]
    await authority._drain(SessionRef(authority.profile_id, new))
    assert executed == ["A", "B", "C"]
    new_rows = list_session_admissions(db, session_id=new, pending_only=False)
    assert [(r["request_id"], r["admission_id"], r["status"]) for r in new_rows] == [
        ("B", admitted[1][1].admission_id, "terminal"), ("C", admitted[2][1].admission_id, "terminal")]


@pytest.mark.asyncio
async def test_no_scheduler_starts_the_fresh_fifo_before_reset_cleanup_finishes(native, monkeypatch):
    """With the real scheduler, a redelivery of B while the topic binding sync is still running
    (it schedules the fresh FIFO itself) must not start B there before the reset finishes."""
    import threading
    from gateway.session_results import execution_result
    runner, authority = native.runner, native.authority
    loop = asyncio.get_running_loop()
    admitted = [await _admit(native, text) for text in ("A", "B", "C")]
    old = admitted[0][1].ref
    events, executed, entered, release = [], [], asyncio.Event(), threading.Event()

    def held_sync(*args, **kwargs):
        events.append("sync:entered")
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10)
        events.append("sync:done")

    monkeypatch.setattr(runner, "_sync_telegram_topic_binding", held_sync)

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append((event.text, entry.session_id))
        events.append("turn:" + event.text)
        captured = execution_result.get()
        if event.text == "A":
            result = {"failed": True, "compression_exhausted": True, "messages": []}
            captured["result"] = result
            # Production's turn cleanup releases the running slot before the reset runs.
            runner._release_running_agent_state(session_key, run_generation=run_generation)
            await runner._hmwa_compression_exhaustion_reset(result, "context exhausted", entry, session_key, source)
        else:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(SessionAuthority, "_schedule", _REAL_SCHEDULE)
    authority._schedule(old)
    await asyncio.wait_for(entered.wait(), 10)
    _, retry = await _admit(native, "B", admitted[1][0])  # dedupes, then schedules its FIFO
    assert retry.admission_id == admitted[1][1].admission_id
    ran_early = asyncio.Event()
    monkeypatch.setattr(runner, "_handle_message_with_agent", lambda *a: ran_early.set() or turn(*a))
    with pytest.raises(asyncio.TimeoutError):  # nothing runs on the fresh session during the sync
        await asyncio.wait_for(ran_early.wait(), 0.5)
    assert [t for t, _ in executed] == ["A"]
    release.set()
    for _ in range(300):
        if len(executed) == 3:
            break
        await asyncio.sleep(0.02)
    new = runner.session_store.lookup_by_session_key(authority.sessions[old.session_id].route).session_id
    assert executed == [("A", old.session_id), ("B", new), ("C", new)]
    assert events.index("sync:done") < events.index("turn:B")


@pytest.mark.asyncio
async def test_admission_reserved_on_the_old_route_commits_on_the_fresh_session(native, monkeypatch):
    """A retry of queued B and a new D that reserved the old route before the reset commit after
    the re-home: B dedupes onto its moved row (same admission id), D queues behind it there."""
    import threading
    from gateway.session_results import execution_result
    runner, authority, db = native.runner, native.authority, native.authority.db
    loop = asyncio.get_running_loop()
    admitted = [await _admit(native, text) for text in ("A", "B", "C")]
    old = admitted[0][1].ref
    reserve_real, held, release = authority._reserve_route, [], threading.Event()

    def held_reserve(src):
        answer = reserve_real(src)
        loop.call_soon_threadsafe(held.append, answer)
        assert release.wait(10)
        return answer

    monkeypatch.setattr(authority, "_reserve_route", held_reserve)
    late = [asyncio.create_task(_admit(native, "B", admitted[1][0])), asyncio.create_task(_admit(native, "D"))]
    for _ in range(250):
        if len(held) == 2:
            break
        await asyncio.sleep(0.02)
    assert [sid for sid, _ in held] == [old.session_id] * 2
    executed, fresh = [], {}

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append((event.text, entry.session_id))
        captured = execution_result.get()
        if event.text == "A":
            result = {"failed": True, "compression_exhausted": True, "messages": []}
            captured["result"] = result
            _, new_entry = await runner._hmwa_compression_exhaustion_reset(
                result, "context exhausted", entry, session_key, source)
            fresh["sid"] = new_entry.session_id
        else:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await authority._drain(old)  # A exhausts; reset + re-home finish while B/D hold old reservations
    release.set()
    (_, retry), (_, late_d) = await asyncio.gather(*late)
    new = fresh["sid"]
    assert retry.admission_id == admitted[1][1].admission_id
    assert retry.ref.session_id == late_d.ref.session_id == new
    await authority._drain(SessionRef(authority.profile_id, new))
    assert executed == [("A", old.session_id), ("B", new), ("C", new), ("D", new)]
    assert [r["request_id"] for r in list_session_admissions(db, session_id=old.session_id, pending_only=False)] == ["A"]
