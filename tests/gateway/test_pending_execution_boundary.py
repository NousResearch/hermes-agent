"""RAM restoration cannot transfer execution to a reset or acknowledged input."""

import asyncio
from unittest.mock import AsyncMock

import hermes_state
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.input_owner import gateway_input_owner
from gateway.pending_execution import PendingExecutionOwner
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.shutdown_pending import PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_event
from gateway.turn_context import TurnContext
from hermes_constants import get_hermes_home
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "current",
        "reset-after-prepare",
        "receipt-after-prepare",
        "reset-after-pins",
        "receipt-after-pins",
        "reset-during-lease",
        "receipt-during-lease",
        "followup-current",
        "receipt-during-followup",
    ],
)
async def test_restored_input_rechecks_execution_owner_after_final_await(
    monkeypatch, boundary
):
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    runner = GatewayRunner(GatewayConfig())
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@hermes:example.org",
            },
        )
    )
    runner.adapters[Platform.MATRIX] = adapter
    adapter.gateway_runner = runner
    adapter._mark_connected()
    source = adapter.build_source(
        chat_id="!room:example.org", user_id="@alice:example.org", chat_type="group"
    )
    assert adapter._canonicalize(source) is not None
    entry = runner.session_store.get_or_create_session(source)
    snapshot = PendingQueueSnapshot.capture(
        entry.session_key,
        [MessageEvent(text="authored", source=source, message_id="$original")],
    )
    event = decode_pending_event(snapshot.events[0], adapter=adapter)
    assert adapter._canonicalize(event.source) is not None
    owner = gateway_input_owner(event, event.source)
    event._pending_execution_owner = PendingExecutionOwner(
        get_hermes_home().resolve(), entry.session_key, entry.session_id, owner
    )
    executions = []
    marked = []

    def change(stage):
        if not boundary.endswith(stage):
            return
        if boundary.startswith("reset"):
            runner.session_store.reset_session(entry.session_key, source=event.source)
        else:
            db = runner.session_store._db_for_session_id(entry.session_id)
            db.append_message(
                entry.session_id,
                "user",
                "executed original",
                display_metadata={"gateway_input_owner": owner},
            )

    async def hook(*args):
        change("prepare")

    async def pins(*args):
        change("pins")

    async def model(**kwargs):
        executions.append((
            kwargs["session_id"],
            kwargs["persist_user_display_metadata"]["gateway_input_owner"],
        ))
        raise asyncio.CancelledError

    monkeypatch.setattr(runner.hooks, "emit", hook)
    if boundary.endswith("lease"):
        acquire = runner._hmwa_acquire_turn_lease

        async def lease(*args):
            await acquire(*args)
            change("lease")

        monkeypatch.setattr(runner, "_hmwa_acquire_turn_lease", lease)
        monkeypatch.setattr(runner, "_hmwa_first_contact_notes", AsyncMock())
        monkeypatch.setattr(
            runner, "_hmwa_run_session_hygiene", AsyncMock(return_value=[])
        )
        monkeypatch.setattr(
            runner,
            "_prepare_profile_scoped_inbound_message_text",
            AsyncMock(return_value="authored"),
        )

        async def mark(*args):
            marked.append((args[0].message_id, args[1]))

        monkeypatch.setattr(runner, "_mark_durable_active_turn", mark)
    else:
        monkeypatch.setattr(
            runner,
            "_hmwa_prepare_turn",
            AsyncMock(
                return_value=(
                    runner._PreparedTurn(
                        [],
                        "",
                        event.text,
                        event.text,
                        None,
                        None,
                        entry.session_id,
                        owner,
                    ),
                    {},
                )
            ),
        )
    monkeypatch.setattr(runner, "_persist_prompt_pins", pins)
    monkeypatch.setattr(runner, "_run_agent", model)
    try:
        if "followup" in boundary:

            async def refresh(*args):
                change("followup")

            monkeypatch.setattr(runner, "_refresh_agent_cache_message_count", refresh)
            monkeypatch.setattr(
                runner,
                "_prepare_profile_scoped_inbound_message_text",
                AsyncMock(return_value="authored"),
            )
            await runner._run_agent_queued_followup(
                TurnContext(
                    source=source,
                    session_id=entry.session_id,
                    session_key=entry.session_key,
                    history=[],
                    run_generation=1,
                ),
                adapter,
                event.text,
                event,
                "done",
                {"interrupted": True, "messages": []},
                None,
            )
        else:
            await runner._handle_message_with_agent(
                event, event.source, entry.session_key, 1
            )
    except asyncio.CancelledError:
        pass
    expected = (
        [(entry.session_id, owner)]
        if boundary in {"current", "followup-current"}
        else []
    )
    assert (executions, marked) == (expected, [])


@pytest.mark.parametrize("capacity", [1, 2])
def test_restored_record_owners_use_bounded_fifo_without_media_coalescing(
    monkeypatch, capacity
):
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    runner = GatewayRunner(GatewayConfig())
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@hermes:example.org",
            },
        )
    )
    runner.adapters[Platform.MATRIX] = adapter
    adapter.gateway_runner = runner
    adapter._mark_connected()
    source = adapter.build_source(
        chat_id="!room:example.org", user_id="@alice:example.org", chat_type="group"
    )
    assert adapter._canonicalize(source) is not None
    entry = runner.session_store.get_or_create_session(source)
    originals = [
        MessageEvent(
            text=text,
            source=source,
            message_id=text,
            message_type=MessageType.PHOTO,
            media_urls=[str(get_hermes_home() / "cache" / text)],
        )
        for text in ("first", "second")
    ]
    snapshot = PendingQueueSnapshot.capture(entry.session_key, originals)
    events = [
        decode_pending_event(record, adapter=adapter) for record in snapshot.events
    ]
    for event in events:
        assert adapter._canonicalize(event.source) is not None
        event._pending_execution_owner = PendingExecutionOwner(
            get_hermes_home().resolve(),
            entry.session_key,
            entry.session_id,
            gateway_input_owner(event, event.source),
        )
        event.defer_until_idle = True
    runner._BUSY_QUEUE_MAX_PENDING = capacity
    accepted = [
        runner._queue_or_replace_pending_event(entry.session_key, event)
        for event in events
    ]
    queued = [
        adapter._pending_messages[entry.session_key],
        *(runner._overflow_queue(entry.session_key) or []),
    ]
    expected = events[:capacity]
    assert (
        accepted,
        [
            (
                event.text,
                event.media_urls,
                getattr(event, "_pending_snapshot_uid", None),
                event._pending_execution_owner,
            )
            for event in queued
        ],
    ) == (
        [True, capacity == 2],
        [
            (
                record["event"]["text"],
                record["event"]["media_urls"],
                record["uid"],
                event._pending_execution_owner,
            )
            for event, record in zip(expected, snapshot.events)
        ],
    )
