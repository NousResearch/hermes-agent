"""Canonical ingress/compaction plumbing, not a model-meaning evaluation.

Synthetic IDs and fake summaries deliberately provide no semantic evidence for
report rows 5/6/9. Receipt association must come from disk ownership, not quotes.
"""
import asyncio
import copy
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl, shutdown_flush as spool
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from tests.gateway.test_durable_batch_transfer import inputs, world  # noqa: F401


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("network is forbidden in synthetic replay fixtures")

    monkeypatch.setattr(socket.socket, "connect", refused)
    monkeypatch.setattr(socket.socket, "connect_ex", refused)
    # This constructor preflight may probe auxiliary providers. It is not the
    # compaction persistence path under test, and must never contact a provider.
    monkeypatch.setattr("agent.conversation_compression.check_compression_model_feasibility",
                        lambda agent: None)


def ingress_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace()
    runner._pending_native_image_paths = {}
    return runner


def scalar_values(value):
    if isinstance(value, dict):
        return [item for child in value.values() for item in scalar_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in scalar_values(child)]
    return [value]


@pytest.mark.asyncio
async def test_adapter_suppresses_answered_redelivery_but_admits_fresh_identity(world):
    adapter, source, key = world
    event = MessageEvent(text="repeat deliberately", source=source, message_id="first")
    assert spool.record_durable_inbound_event(key, event)
    saved = spool._serialise_inbound_event(event)
    await adapter.send_final_ledgered(event, key, "already answered", {}, reply_to=None)
    adapter._message_handler = AsyncMock(return_value=None)
    await adapter.handle_message(spool._deserialise_inbound_event(saved))
    assert adapter._message_handler.await_count == 0
    fresh = MessageEvent(text=event.text, source=source, message_id="second")
    await adapter.handle_message(fresh)
    await asyncio.gather(*list(adapter._session_tasks.values()))
    assert adapter._message_handler.await_count == 1
    assert adapter._message_handler.await_args.args[0].message_id == fresh.message_id
    # No terminal answer was produced for the fresh request; restart must keep it.
    assert [e.message_id for e in spool.recover_durable_inbound_events()] == [fresh.message_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_batched_ingress_exposes_original_typed_members_without_history_mutation(world, internal):
    _, source, key = world
    events, batch = inputs(source, key)
    if internal:
        # Internal events are not durable owner work. Preserve this distinction
        # even if the pending envelope has explicit batched provenance.
        batch.input_members = [{**m, "internal": True, "inbound_id": ""}
                               for m in batch.original_inputs()]
        batch.internal = True
    before = copy.deepcopy(batch.original_inputs())
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "answer"}]
    saved_history = copy.deepcopy(history)
    text = await ingress_runner()._prepare_inbound_message_text(
        event=batch, source=source, history=history, session_key=key)
    provenance = json.loads(text.split("\n", 2)[1])
    assert history == saved_history
    assert len(provenance) == len(before)
    for actual, original in zip(provenance, before):
        assert {name: actual[name] for name in original} == original
        assert actual["kind"] == ("internal_notification" if internal else "owner_message")
        assert actual["delivery_state"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("known_receipt", [False, True])
async def test_fresh_reply_resolves_delivered_output_after_disk_compaction(world, tmp_path, compact, known_receipt):
    from hermes_state import SessionDB
    from agent.conversation_compression import compress_context
    from tests.run_agent.test_in_place_compaction import _make_agent

    adapter, source, key = world
    _, batch = inputs(source, key)
    original_members = copy.deepcopy(batch.original_inputs())
    await adapter.send_final_ledgered(batch, key, "prior result", {}, reply_to=None)
    with dl._transaction() as conn:
        oid = conn.execute("SELECT obligation_id FROM delivery_obligations").fetchone()[0]
    db_path = tmp_path / "conversation.db"
    db = SessionDB(db_path=db_path)
    sid = "synthetic-replay-session"
    db.create_session(sid, "telegram", model="test/model")
    history = [{"role": "user", "content": "prior work"},
               {"role": "assistant", "content": "prior result"}]
    for item in history:
        db.append_message(session_id=sid, **item)
    if compact:
        agent = _make_agent(db, sid, in_place=True)
        compress_context(agent, history * 4, approx_tokens=100_000, system_message="fixed system")
        assert agent._last_compaction_in_place is True
    db.close()
    reopened = SessionDB(db_path=db_path)
    history = reopened.get_messages_as_conversation(sid)
    assert history
    if compact:
        assert any(not row.get("active", 1) for row in reopened.get_messages(sid, include_inactive=True))
    reopened.close()
    followup = MessageEvent(text="Investigate the repeated response, not the old work.",
                            source=source, message_id="new-complaint",
                            reply_to_message_id="out" if known_receipt else "unknown-receipt",
                            # Untrusted quoted words may NOT supply authoritative disposition.
                            reply_to_text="delivered", reply_to_is_own_message=True)
    assert spool.record_durable_inbound_event(key, followup)
    # A real disk spool reload, not a constructed replacement identity.
    restored = spool.recover_durable_inbound_events()[0]
    text = await ingress_runner()._prepare_inbound_message_text(
        event=restored, source=source, history=history, session_key=key)
    assert text is not None, "a fresh complaint is not a duplicate input"
    provenance = json.loads(text.split("\n", 2)[1])
    assert provenance[0]["message_id"] == followup.message_id
    assert provenance[0]["reply_to_message_id"] == followup.reply_to_message_id
    # Do not prescribe a new API/schema: the typed provenance must associate the
    # authoritative output and every original input somewhere in its structure.
    values = scalar_values(provenance)
    if known_receipt:
        assert oid in values, "known platform receipt must resolve to the authoritative output"
        assert all(member["inbound_id"] in values for member in original_members)
        # Exclude the quote when checking a machine disposition, to reject copying.
        assert any(value == "delivered" for value in scalar_values([
            {k: v for k, v in item.items() if k not in {"text", "reply_to_text"}}
            for item in provenance]))
    else:
        assert oid not in values, "unknown receipts cannot infer prior ownership from text"


@pytest.mark.asyncio
async def test_no_change_internal_notification_stays_silent(world):
    adapter, source, key = world
    event = MessageEvent(text="unchanged internal status", source=source, message_id="internal-only",
                         internal=True, allow_gateway_control=False)
    adapter._message_handler = AsyncMock(return_value=None)
    adapter._active_sessions[key] = asyncio.Event()
    await adapter._process_message_background(event, key)
    adapter._send_with_retry.assert_not_awaited()
    assert spool.recover_durable_inbound_events() == []
    assert dl.sweep_recoverable() == []


@pytest.mark.asyncio
async def test_compaction_keeps_retained_complaint_identity_metadata(world, tmp_path):
    from hermes_state import SessionDB
    from agent.conversation_compression import compress_context
    from tests.run_agent.test_in_place_compaction import _make_agent

    _, source, key = world
    _, complaint = inputs(source, key, count=2)
    text = await ingress_runner()._prepare_inbound_message_text(
        event=complaint, source=source, history=[], session_key=key)
    provenance = json.loads(text.split("\n", 2)[1])
    # display_metadata is the existing lossless transcript API, not a new
    # ingress API. Compression must preserve metadata of retained messages.
    retained = {"role": "user", "content": text,
                "platform_message_id": complaint.message_id,
                "display_metadata": {"original_inputs": provenance}}
    recent = [retained, {"role": "assistant", "content": "Investigating repetition."}]
    path = tmp_path / "compaction.db"
    db = SessionDB(db_path=path)
    sid = "retained-complaint"
    db.create_session(sid, "telegram", model="test/model")
    for item in recent:
        db.append_message(session_id=sid, **item)
    agent = _make_agent(db, sid, in_place=True)
    agent.context_compressor.compress = lambda *args, **kwargs: copy.deepcopy(recent)
    compress_context(agent, recent * 4, approx_tokens=100_000, system_message="stable")
    assert agent._last_compaction_in_place is True
    db.close()
    reopened = SessionDB(db_path=path)
    rows = reopened.get_messages(sid)
    originals = reopened.get_messages(sid, include_inactive=True)
    reopened.close()
    archived = [row for row in originals if not row.get("active", 1)]
    assert any(row.get("platform_message_id") == complaint.message_id for row in archived)
    current = next(row for row in rows if row["role"] == "user")
    assert current.get("platform_message_id") == complaint.message_id
    metadata = current.get("display_metadata")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    assert metadata == retained["display_metadata"]
