"""Replay contract against real disk owners; every transport is synthetic.

The child interpreter reopens the spool/SQLite and runs the production recovery
sender. No parent-process adapter, pending queue, or ledger connection survives.
"""
import asyncio
import json
import os
import socket
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl, shutdown_flush as spool
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_durable_batch_transfer import inputs, world  # noqa: F401


class Crash(BaseException):
    """Simulate process death without being swallowed as a transport exception."""


def refuse_network(*args, **kwargs):
    raise AssertionError("network is forbidden in synthetic replay fixtures")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_network)


def recovery_snapshot(home):
    os.environ["HERMES_HOME"] = home
    socket.socket.connect = refuse_network
    socket.socket.connect_ex = refuse_network
    sent, documents = [], []
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture", extra={}))

    async def send(chat_id, content, **kwargs):
        sent.append({"chat_id": chat_id, "content": content, **kwargs})
        return SendResult(success=True, message_id="recovery-text")

    async def document(**kwargs):
        if not Path(kwargs["file_path"]).is_file():
            return SendResult(success=False, error="missing attachment")
        documents.append(kwargs)
        return SendResult(success=True, message_id="recovery-file")

    adapter.send = send
    adapter._send_with_retry = send
    adapter.send_document = document
    runner = object.__new__(GatewayRunner)
    runner._obligation_adapter = AsyncMock(return_value=adapter)
    runner._arm_flood_timers_for_waiting_rows = AsyncMock()
    # The simulated dead producer is still our live pytest parent. Only its
    # liveness probe is replaced; the real atomic claim and sender are exercised.
    dl._owner_alive = lambda *args: False
    pending = spool.recover_durable_inbound_events()
    claimed = dl.sweep_recoverable()
    asyncio.run(runner._redeliver_claimed_obligations(claimed))
    with dl._transaction() as conn:
        rows = conn.execute("SELECT obligation_id, state FROM delivery_obligations").fetchall()
        links = conn.execute("SELECT inbound_id, obligation_id FROM delivery_input_links").fetchall()
    return {"pending": [e.original_inputs() for e in pending], "claimed": claimed,
            "sent": sent, "documents": documents, "rows": rows, "links": links}


def restart(home):
    code = ("import json,sys; "
            "from tests.gateway.test_durable_output_recovery import recovery_snapshot; "
            "print(json.dumps(recovery_snapshot(sys.argv[1])))")
    result = subprocess.run([sys.executable, "-c", code, str(home)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout.splitlines()[-1])


def runner_for(adapter):
    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_source = lambda *args: {}
    runner._reply_anchor_for_event = lambda event: event.reply_to_message_id
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before_record", "record_failed", "record_before_ack",
                                      "ack_before_send", "send_before_receipt", "after_send", "refused"])
async def test_restart_dispatch_preserves_every_member(world, tmp_path, monkeypatch, boundary):
    adapter, source, key = world
    events, batch = inputs(source, key)
    originals = [dict(member) for member in batch.original_inputs()]
    if boundary == "record_failed":
        monkeypatch.setattr(dl, "record_obligation", lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    if boundary == "record_before_ack":
        monkeypatch.setattr(spool, "acknowledge_durable_inbound_event", lambda event: (_ for _ in ()).throw(Crash()))
    if boundary == "ack_before_send":
        adapter._send_with_retry.side_effect = Crash()
    if boundary == "send_before_receipt":
        adapter._finalize_delivery_obligation = AsyncMock(side_effect=Crash())
    if boundary == "refused":
        adapter._send_with_retry.return_value = SendResult(success=False, error="rejected")
    if boundary != "before_record":
        try:
            await adapter.send_final_ledgered(batch, key, "saved output", {}, reply_to=None)
        except (Crash, OSError):
            assert boundary in {"record_failed", "record_before_ack", "ack_before_send", "send_before_receipt"}
    snapshot = restart(tmp_path)
    if boundary in {"before_record", "record_failed"}:
        assert [m for group in snapshot["pending"] for m in group] == originals
        assert snapshot["links"] == snapshot["rows"] == snapshot["sent"] == []
        adapter._send_with_retry.assert_not_awaited()
    else:
        assert snapshot["pending"] == []
        assert len(snapshot["rows"]) == 1
        oid, state = snapshot["rows"][0]
        assert state == "delivered"
        assert dict(snapshot["links"]) == {m["inbound_id"]: oid for m in originals}
        assert len(snapshot["sent"]) == (0 if boundary == "after_send" else 1)
        if boundary == "send_before_receipt":
            # Accepted transport with no persisted receipt is ambiguous, not
            # an exactly-once guarantee: redelivery carries the restart marker.
            assert snapshot["claimed"][0]["needs_marker"] is True
    fresh = MessageEvent(text=events[0].text, source=source, message_id="intentional-repeat")
    assert spool.record_durable_inbound_event(key, fresh)
    assert "intentional-repeat" in [e.message_id for e in spool.recover_durable_inbound_events()]


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("failure", ["crash_before_media", "refused_media", "missing_file"])
async def test_queued_media_survives_text_success_and_restart(world, tmp_path, streamed, failure):
    adapter, source, key = world
    _, batch = inputs(source, key)
    document = tmp_path / "deliverable.txt"
    document.write_text("synthetic deliverable", encoding="utf-8")
    runner = runner_for(adapter)
    if failure == "crash_before_media":
        adapter.send_document = AsyncMock(side_effect=Crash())
    else:
        adapter.send_document = AsyncMock(return_value=SendResult(success=False, error="refused"))
    if failure == "missing_file":
        document.unlink()
    try:
        await runner._deliver_queued_first_response(
            f"result\nMEDIA:{document}", source, adapter, session_key=key,
            inbound_event=batch, text_already_delivered=streamed,
            stream_consumer=SimpleNamespace(message_id="stream-receipt") if streamed else None,
            metadata={})
    except Crash:
        pass
    snapshot = restart(tmp_path)
    assert snapshot["pending"] == [], "recover the output, never rerun the answered inputs"
    if failure == "missing_file":
        assert snapshot["rows"][0][1] != "delivered", "missing attachment cannot silently complete"
    else:
        assert [item["file_path"] for item in snapshot["documents"]] == [str(document)]
        assert snapshot["sent"] == [], "confirmed text must not be sent twice to recover media"
        assert snapshot["rows"][0][1] == "delivered"


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_failed_queued_turn_never_uploads_artifacts(world, tmp_path, streamed):
    adapter, source, key = world
    _, batch = inputs(source, key)
    artifact = tmp_path / "not-a-success.txt"
    artifact.write_text("partial", encoding="utf-8")
    adapter.send_document = AsyncMock()
    await runner_for(adapter)._deliver_queued_first_response(
        f"failed\nMEDIA:{artifact}", source, adapter, session_key=key,
        inbound_event=batch, text_already_delivered=streamed, deliver_media=False)
    adapter.send_document.assert_not_awaited()
    assert restart(tmp_path)["documents"] == []


def test_shutdown_flush_keeps_unfinished_originals_and_media(world, tmp_path):
    _, source, key = world
    attachment = tmp_path / "input.txt"
    attachment.write_text("fixture", encoding="utf-8")
    event = MessageEvent(text="unfinished", source=source, message_id="pending-with-file",
                         media_urls=[str(attachment)], media_types=["text/plain"],
                         media_text_inlined=[True], reply_to_message_id="anchor")
    assert spool.record_durable_inbound_event(key, event)
    assert spool.flush_pending_to_file({key: event}) == 0  # already durably owned
    snapshot = restart(tmp_path)
    assert snapshot["pending"] == [event.original_inputs()]
    assert snapshot["rows"] == []
    assert attachment.read_text(encoding="utf-8") == "fixture"


@pytest.mark.asyncio
async def test_recursive_terminal_recovers_only_unconfirmed_attachment(world, tmp_path):
    from gateway.turn_context import TurnContext

    adapter, source, key = world
    _, batch = inputs(source, key)
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    for path in (first, second):
        path.write_text("synthetic", encoding="utf-8")
    opening = MessageEvent(text="opening", source=source, message_id="opening")
    assert spool.record_durable_inbound_event(key, opening)
    runner = runner_for(adapter)
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=key)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="followup")
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._pop_post_delivery_callback = MagicMock(return_value=None)
    terminal = {"final_response": f"terminal\nMEDIA:{first}\nMEDIA:{second}", "messages": [],
                "queued_terminal_event": batch, "queued_terminal_inbound_id": batch.message_id}
    runner._run_agent = AsyncMock(return_value=terminal)
    ctx = TurnContext(source=source, session_id="sid", session_key=key,
                      inbound_event=opening, inbound_message_id=opening.message_id,
                      _interrupt_depth=2, history=[])
    result = await runner._run_agent_queued_followup(
        ctx, adapter, "pending", batch, {"final_response": "opening answer"}, {"messages": []}, None)
    assert runner._run_agent.await_args.kwargs["inbound_event"] is batch
    adapter.send_document = AsyncMock(side_effect=[
        SendResult(success=True, message_id="first-file-receipt"),
        SendResult(success=False, error="refused")])
    await runner._deliver_queued_first_response(
        result["final_response"], source, adapter, session_key=key,
        inbound_event=result["queued_terminal_event"], metadata={})
    assert [call.kwargs["file_path"] for call in adapter.send_document.await_args_list] == [str(first), str(second)]
    snapshot = restart(tmp_path)
    assert snapshot["pending"] == []
    assert len(snapshot["rows"]) == 2  # opening output plus ONE terminal output
    assert [item["file_path"] for item in snapshot["documents"]] == [str(second)]
    assert snapshot["sent"] == []
