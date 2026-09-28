"""Private ``/group N send`` through the real canonical event and task path."""
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
from gateway.hosted_room_discussion import MAX_USER_TEXT_BYTES
from gateway.platforms.event import MessageType
from hermes_cli.commands import resolve_command
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_canonical_group_messaging_list import consumer, route
from tests.gateway.test_messaging_inventory_binding import bound, enrolled
from tests.gateway.test_messaging_room_read_binding import room_grant_params, room_rpc
from tests.gateway.test_messaging_room_send_binding import (
    _native,
    send_grant_params,
    send_revoke_params,
    send_rpc,
)

_REAL_ROOM_STATE = rooms.room_state
_REAL_READ_EVENTS = rooms.read_events


def send_event(c, text="review this", *, message_id="message-send-1", room_ref=1, **changes):
    event = c.event(f"/group {room_ref} send {text}", message_id=message_id, **changes)
    event.message_id = message_id
    return event


def canonical_rows(c, room_id=None):
    room_id = room_id or getattr(c, "send_room_id", "alice-room")
    page = rooms.read_events(c.db.db_path, room_id=room_id)
    return [row for row in page["events"] if row["kind"] == "message.user"]


def canonical_tasks(c, room_id=None):
    room_id = room_id or getattr(c, "send_room_id", "alice-room")
    return driver.list_tasks(c.db.db_path, room_id=room_id)


@pytest_asyncio.fixture
async def authorized_send(consumer, monkeypatch):
    c = consumer
    monkeypatch.setattr(rooms, "room_state", _REAL_ROOM_STATE)
    monkeypatch.setattr(rooms, "read_events", _REAL_READ_EVENTS)
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "inert-gateway")
    # The accepted read fixture installs tripwires on mutation. Reveal the real
    # class method only for this focused Send fixture.
    monkeypatch.delattr(c.service, "send")
    # Proof-only harness compatibility: the exact Route/lifetime/permission
    # composition predates the later HostedRoomService constructor field from
    # #98073's broader consumer tree.  Supply the same canonical store to the
    # fixture without changing any production method or Send assertion.
    if not hasattr(c.service, "attachments"):
        from gateway.hosted_room_attachments import HostedRoomAttachmentStore

        c.service.attachments = HostedRoomAttachmentStore(c.db.db_path)
    c.service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
    c.service.local_profiles = lambda: ("default", "reviewer")
    c.service.authorize_room(c.alice.actor.subject, "send-room", create=True)
    rooms.create_room(
        c.db.db_path,
        room_id="send-room",
        name="Send room",
        members=[
            {"member_id": "writer", "profile": "default", "handle": "writer"},
            {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
        ],
        authority_gateway_id="inert-gateway",
    )
    inventory = await enrolled(c)
    read_grant = (await room_rpc(
        c.alice, params=room_grant_params(inventory, room_id="send-room")
    ))["result"]
    owner = _native(c)
    send_grant = (await send_rpc(
        owner, params=send_grant_params(read_grant, room_id="send-room")
    ))["result"]
    c.send_room_id = "send-room"
    c.inventory_grant = inventory
    c.room_grant = read_grant
    c.send_owner = owner
    c.send_grant = send_grant
    try:
        yield c
    finally:
        c.service.runtime._thread = None


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["idle", "runner_busy", "adapter_busy"])
async def test_registered_send_reaches_one_canonical_event_task_and_ack_in_every_lane(
    authorized_send, lane
):
    c = authorized_send
    command = resolve_command("group")
    assert command is not None and command.busy_policy == "dispatch"

    event = send_event(c, "@writer review — this – exactly")
    await route(c, event, lane)

    rows = canonical_rows(c)
    tasks = canonical_tasks(c)
    assert len(rows) == len(tasks) == 1
    assert rows[0]["payload"]["text"] == "@writer review — this – exactly"
    assert rows[0]["payload"]["thread_id"].startswith("msg:")
    assert tasks[0]["identity"].thread_id == rows[0]["payload"]["thread_id"]
    assert len(c.adapter.sent) == 1 and not c.adapter.generic_sent
    assert c.adapter.sent[0] == (
        "private-chat",
        "Queued in Group 1. Check: /group 1",
        None,
        {"_interim_send": True},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["idle", "runner_busy", "adapter_busy"])
async def test_registered_bridge_send_never_enters_legacy_rooms(authorized_send, monkeypatch, lane):
    c = authorized_send
    monkeypatch.setattr(c.runner, "_handle_legacy_rooms_command", c.forbidden)
    await route(c, send_event(c, "@writer bridge intent"), lane)
    assert len(canonical_rows(c)) == len(canonical_tasks(c)) == 1
    assert canonical_rows(c)[0]["payload"]["text"] == "@writer bridge intent"
    assert c.adapter.sent == [
        ("private-chat", "Queued in Group 1. Check: /group 1", None,
         {"_interim_send": True})
    ]
    assert not c.adapter.generic_sent


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["1", "1 bots", "１"])
async def test_native_numeric_reference_never_routes_to_legacy_detail(
    authorized_send, monkeypatch, query
):
    c = authorized_send
    monkeypatch.setattr(c.runner, "_handle_legacy_rooms_command", c.forbidden)
    observed = []

    async def native_detail(event):
        observed.append(event.text)
        return "native detail selected"

    # Routing invariant only; the separately-owned read module tests the
    # actual detail implementation in its own composition.
    monkeypatch.setattr(c.runner, "_handle_group_command", native_detail)
    event = c.event(f"/group {query}")
    handled, result = await c.runner._hm_dispatch_canonical_command(
        event, event.source, c.runner._session_key_for_source(event.source), "group")
    assert handled and result == "native detail selected"
    assert observed == [f"/group {query}"]
    assert c.adapter.sent == c.adapter.generic_sent == []


@pytest.mark.asyncio
async def test_read_consent_is_not_send_consent(consumer, monkeypatch):
    c = consumer
    monkeypatch.setattr(rooms, "room_state", _REAL_ROOM_STATE)
    monkeypatch.setattr(rooms, "read_events", _REAL_READ_EVENTS)
    inventory = await enrolled(c)
    read_grant = (await room_rpc(c.alice, params=room_grant_params(inventory)))["result"]
    c.service.runtime._thread = SimpleNamespace(is_alive=lambda: True)
    try:
        result = await c.runner._handle_group_command(
            send_event(c, room_ref=read_grant["room_ref"])
        )
    finally:
        c.service.runtime._thread = None
    assert result
    assert c.adapter.sent == c.adapter.generic_sent == []
    assert canonical_rows(c) == []
    assert canonical_tasks(c) == []


@pytest.mark.asyncio
async def test_same_transport_message_replays_one_event_and_task_but_changed_text_conflicts(
    authorized_send,
):
    c = authorized_send
    await route(c, send_event(c, "@writer exact"))
    first_rows = canonical_rows(c)
    first_tasks = canonical_tasks(c)
    c.adapter.sent.clear()

    await route(c, send_event(c, "@writer exact"))
    assert canonical_rows(c) == first_rows
    assert canonical_tasks(c) == first_tasks
    assert len(c.adapter.sent) == 1

    c.adapter.sent.clear()
    result = await c.runner._handle_group_command(send_event(c, "@writer changed"))
    assert result
    assert c.adapter.sent == []
    assert canonical_rows(c) == first_rows
    assert canonical_tasks(c) == first_tasks

    await route(c, send_event(c, "@writer new intent", message_id="message-send-2"))
    assert len(canonical_rows(c)) == 2
    # The canonical planner keeps the second intent behind the already queued
    # task instead of creating concurrent work in the same room.
    assert canonical_tasks(c) == first_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_id", "source_mismatch", "edited", "machine", "relay_unknown",
        "internal", "media", "empty", "over_limit",
    ],
)
async def test_unstable_or_nonhuman_send_is_rejected_before_canonical_admission(
    authorized_send, mutation
):
    c = authorized_send
    text = "x"
    event = send_event(c, text)
    if mutation == "missing_id":
        event.message_id = None
        event.source.message_id = None
    elif mutation == "source_mismatch":
        event.source.message_id = "different-source-message"
    elif mutation == "edited":
        event.metadata["message_is_edit"] = True
    elif mutation == "machine":
        event.source.is_bot = True
    elif mutation == "relay_unknown":
        event.source.delivered_via_upstream_relay = True
    elif mutation == "internal":
        event.internal = True
    elif mutation == "media":
        event.media_urls = [str(c.home / "not-opened")]
        event.media_types = ["text/plain"]
        event.message_type = MessageType.DOCUMENT
    elif mutation == "empty":
        event = send_event(c, "   ")
    else:
        event = send_event(c, "x" * (MAX_USER_TEXT_BYTES + 1))

    result = await c.runner._handle_group_command(event)
    assert result
    assert c.adapter.sent == c.adapter.generic_sent == []
    assert canonical_rows(c) == []
    assert canonical_tasks(c) == []


@pytest.mark.asyncio
async def test_revoke_denies_new_but_exact_accepted_receipt_remains_visible_on_same_lineage(
    authorized_send,
):
    c = authorized_send
    original = send_event(c, "@writer durable")
    await route(c, original)
    rows = canonical_rows(c)
    tasks = canonical_tasks(c)
    revoked = (await send_rpc(
        c.send_owner,
        "revoke",
        send_revoke_params(c.room_grant, c.send_grant, room_id=c.send_room_id),
    ))["result"]
    c.adapter.sent.clear()

    await route(c, send_event(c, "@writer durable"))
    assert canonical_rows(c) == rows
    assert canonical_tasks(c) == tasks
    assert len(c.adapter.sent) == 1

    c.adapter.sent.clear()
    result = await c.runner._handle_group_command(
        send_event(c, "@writer forbidden", message_id="message-after-revoke")
    )
    assert result and c.adapter.sent == []
    assert canonical_rows(c) == rows

    replacement = (await send_rpc(
        c.send_owner,
        params=send_grant_params(
            c.room_grant,
            request_id="send-regrant-after-accept",
            room_id=c.send_room_id,
            expected_generation=revoked["generation"],
        ),
    ))["result"]
    c.send_grant = replacement
    result = await c.runner._handle_group_command(send_event(c, "@writer durable"))
    assert result and c.adapter.sent == []
    assert canonical_rows(c) == rows
    assert canonical_tasks(c) == tasks


@pytest.mark.asyncio
async def test_same_source_cannot_retarget_a_second_consented_room(authorized_send):
    c = authorized_send
    await route(c, send_event(c, "@writer same source"))
    first_rows = canonical_rows(c)

    c.service.authorize_room(c.send_owner.actor.subject, "alice-second", create=True)
    rooms.create_room(
        c.db.db_path,
        room_id="alice-second",
        name="Second",
        members=[{"member_id": "writer", "profile": "default", "handle": "writer"}],
        authority_gateway_id="inert-gateway",
    )
    second_read = (await room_rpc(
        c.alice,
        params=room_grant_params(
            c.inventory_grant,
            request_id="second-read",
            room_id="alice-second",
        ),
    ))["result"]
    second_send = (await send_rpc(
        c.send_owner,
        params=send_grant_params(
            second_read,
            request_id="second-send",
            room_id="alice-second",
        ),
    ))["result"]
    assert second_send["binding_id"] != c.send_grant["binding_id"]

    c.adapter.sent.clear()
    result = await c.runner._handle_group_command(
        send_event(c, "@writer same source", room_ref=second_read["room_ref"])
    )
    assert result and c.adapter.sent == []
    assert canonical_rows(c) == first_rows
    assert canonical_rows(c, "alice-second") == []
    assert driver.list_tasks(c.db.db_path, room_id="alice-second") == []


@pytest.mark.asyncio
async def test_new_writer_callback_never_reenters_sessiondb_read_lock(
    authorized_send, monkeypatch
):
    from gateway.session_group_messaging_send import _MessagingRoomSend

    c = authorized_send
    original_authorize = _MessagingRoomSend.authorize_new_event
    original_read_ctx = c.db._read_ctx
    inside = []

    def authorize(context, conn):
        inside.append(True)
        try:
            return original_authorize(context, conn)
        finally:
            inside.pop()

    @contextmanager
    def guarded_read_ctx():
        if inside:
            raise AssertionError("NEW writer callback re-entered SessionDB._read_ctx")
        with original_read_ctx() as conn:
            yield conn

    monkeypatch.setattr(_MessagingRoomSend, "authorize_new_event", authorize)
    monkeypatch.setattr(c.db, "_checkout_read_conn", lambda: None)
    monkeypatch.setattr(c.db, "_read_ctx", guarded_read_ctx)
    assert await c.runner._handle_group_command(send_event(c)) == ""
    assert len(canonical_rows(c)) == len(canonical_tasks(c)) == 1
    assert len(c.adapter.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("regrant", [False, True])
async def test_native_revoke_and_aba_after_worker_entry_fail_at_new_writer_barrier(
    authorized_send, monkeypatch, regrant
):
    from gateway.session_group_messaging_send import (
        commit_native_send_binding,
        prepare_native_send_binding,
    )

    c = authorized_send
    revoke = prepare_native_send_binding(
        c.send_owner,
        "groups.messaging.room.send.revoke",
        send_revoke_params(c.room_grant, c.send_grant) | {"room_id": c.send_room_id},
    )
    original_send = c.service.send

    def raced_send(**kwargs):
        revoked = commit_native_send_binding(revoke)
        if regrant:
            replacement = prepare_native_send_binding(
                c.send_owner,
                "groups.messaging.room.send.grant",
                send_grant_params(
                    c.room_grant,
                    request_id="writer-barrier-regrant",
                    room_id=c.send_room_id,
                    expected_generation=revoked["generation"],
                ),
            )
            commit_native_send_binding(replacement)
        return original_send(**kwargs)

    monkeypatch.setattr(c.service, "send", raced_send)
    result = await c.runner._handle_group_command(send_event(c))
    assert result and c.adapter.sent == []
    assert canonical_rows(c) == []
    assert canonical_tasks(c) == []


@pytest.mark.asyncio
async def test_closing_after_worker_entry_fails_at_new_writer_barrier(
    authorized_send, monkeypatch
):
    c = authorized_send
    original_send = c.service.send

    def closing_send(**kwargs):
        c.service.runtime._stop.set()
        return original_send(**kwargs)

    monkeypatch.setattr(c.service, "send", closing_send)
    try:
        result = await c.runner._handle_group_command(send_event(c))
    finally:
        c.service.runtime._stop.clear()
    assert result and c.adapter.sent == []
    assert canonical_rows(c) == []
    assert canonical_tasks(c) == []


@pytest.mark.asyncio
async def test_closing_before_new_admission_writes_nothing(authorized_send):
    c = authorized_send
    c.service.runtime._stop.set()
    try:
        result = await c.runner._handle_group_command(send_event(c))
    finally:
        c.service.runtime._stop.clear()
    assert result and c.adapter.sent == []
    assert canonical_rows(c) == []
    assert canonical_tasks(c) == []


@pytest.mark.asyncio
async def test_send_binding_replacement_after_commit_suppresses_ack(
    authorized_send, monkeypatch
):
    from gateway.session_group_messaging_send import (
        commit_native_send_binding,
        prepare_native_send_binding,
    )

    c = authorized_send
    revoke = prepare_native_send_binding(
        c.send_owner,
        "groups.messaging.room.send.revoke",
        send_revoke_params(
            c.room_grant, c.send_grant, room_id=c.send_room_id,
            request_id="egress-revoke",
        ),
    )
    original_prepare = c.service.prepare_room

    def drift_after_commit(binding):
        result = original_prepare(binding)
        revoked = commit_native_send_binding(revoke)
        replacement = prepare_native_send_binding(
            c.send_owner,
            "groups.messaging.room.send.grant",
            send_grant_params(
                c.room_grant,
                request_id="egress-regrant",
                room_id=c.send_room_id,
                expected_generation=revoked["generation"],
            ),
        )
        commit_native_send_binding(replacement)
        return result

    monkeypatch.setattr(c.service, "prepare_room", drift_after_commit)
    assert await c.runner._handle_group_command(send_event(c)) == ""
    assert len(canonical_rows(c)) == len(canonical_tasks(c)) == 1
    assert c.adapter.sent == c.adapter.generic_sent == []


@pytest.mark.asyncio
async def test_receiver_replacement_after_commit_never_hands_private_ack_to_new_adapter(
    authorized_send, monkeypatch
):
    from gateway.config import Platform
    from tests.gateway.test_canonical_group_messaging_list import CapturingReceiver

    c = authorized_send
    replacement = CapturingReceiver(c.runner, c.adapter.config)
    original_prepare = c.service.prepare_room

    def replace_after_commit(binding):
        result = original_prepare(binding)
        c.runner.adapters[Platform.SIGNAL] = replacement
        return result

    monkeypatch.setattr(c.service, "prepare_room", replace_after_commit)
    assert await c.runner._handle_group_command(send_event(c)) == ""
    assert len(canonical_rows(c)) == len(canonical_tasks(c)) == 1
    assert c.adapter.sent == c.adapter.generic_sent == []
    assert replacement.sent == replacement.generic_sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["generic", "runtime_store"])
async def test_post_commit_exception_uses_one_uncertainty_handoff_and_no_automatic_retry(
    authorized_send, monkeypatch, failure
):
    c = authorized_send
    calls = []
    original_prepare = c.service.prepare_room

    def fail_after_commit(_binding):
        calls.append("prepare")
        if failure == "runtime_store":
            raise RuntimeStoreError("permission_denied")
        raise RuntimeError("private body sentinel must never be logged")

    monkeypatch.setattr(c.service, "prepare_room", fail_after_commit)
    result = await c.runner._handle_group_command(send_event(c, "private body sentinel"))
    assert result == ""
    assert calls == ["prepare"]
    assert len(canonical_rows(c)) == 1
    assert canonical_tasks(c) == []
    assert c.adapter.sent == [
        (
            "private-chat",
            "Hermes couldn't confirm whether that message was queued. "
            "Check /group 1 before sending a new message.",
            None,
            {"_interim_send": True},
        )
    ]
    assert c.adapter.generic_sent == []

    monkeypatch.setattr(c.service, "prepare_room", original_prepare)
    c.adapter.sent.clear()
    await route(c, send_event(c, "private body sentinel"))
    assert len(canonical_rows(c)) == 1
    assert canonical_tasks(c) == []
    assert len(c.adapter.sent) == 1


@pytest.mark.asyncio
async def test_ack_adapter_failure_is_once_only_without_generic_fallback(
    authorized_send, monkeypatch
):
    c = authorized_send
    sends = []

    async def fail(*args, **kwargs):
        sends.append((args, kwargs))
        raise RuntimeError("private adapter error")

    monkeypatch.setattr(c.adapter, "send", fail)
    assert await c.runner._handle_group_command(send_event(c)) == ""
    assert len(sends) == 1
    assert len(canonical_rows(c)) == 1
    assert len(canonical_tasks(c)) == 1
    assert c.adapter.generic_sent == []
