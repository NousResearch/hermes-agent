"""Published-owner Send admission, receipt miss, lazy attachments and bounded Stop."""
import sqlite3
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.session_hosted_attachments import append_user_event
from hermes_state_runtime import RuntimeStoreError
from tui_gateway.hosted_room_driver import HostedRoomRuntime


def _room(path):
    return hosted_rooms.create_room(
        path, room_id="room-1", name="Send room",
        members=[{"profile": "ops", "handle": "ops"}],
        authority_gateway_id="gateway-a", now=time.time())


def test_new_event_callbacks_run_inside_the_writer_and_receipt_miss_appends_nothing(tmp_path):
    path = tmp_path / "state.db"
    room = _room(path)
    seen = []

    def authorize_new(conn):
        seen.append(("new", conn.in_transaction))

    def authorize_commit(conn):
        seen.append(("commit", conn.in_transaction))

    with pytest.raises(hosted_rooms.EventNotFoundError):
        hosted_rooms.append_event(
            path, room_id="room-1", event_id="missing", kind="message.user",
            actor={"kind": "user", "id": "desktop"}, payload={"text": "nope", "thread_id": "missing"},
            authority_gateway_id="gateway-a", authority_epoch=room["authority_epoch"],
            existing_only=True)
    with sqlite3.connect(path) as conn:
        missing = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_events WHERE event_id='missing'").fetchone()[0]
    assert missing == 0

    event = hosted_rooms.append_event(
        path, room_id="room-1", event_id="event-1", kind="message.user",
        actor={"kind": "user", "id": "desktop"}, payload={"text": "hello", "thread_id": "event-1"},
        authority_gateway_id="gateway-a", authority_epoch=room["authority_epoch"],
        authorize_new=authorize_new, authorize_commit=authorize_commit)
    assert event["idempotent"] is False
    assert seen == [("new", True), ("commit", True)]
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_hosted_room_events_message_thread'"
        ).fetchone()
    assert row is not None


def test_text_only_append_does_not_construct_an_attachment_store(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    room = _room(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("text-only Send constructed an attachment store")

    monkeypatch.setattr(
        "gateway.session_hosted_attachments.HostedRoomAttachmentStore", forbidden)
    service = SimpleNamespace(db_path=path, attachments=None)
    event = append_user_event(
        service, room_id="room-1", event_id="event-2", payload={"text": "plain", "thread_id": "event-2"},
        gateway_id="gateway-a", epoch=room["authority_epoch"])
    assert event["event_id"] == "event-2"


def _runtime(path):
    return HostedRoomRuntime(
        db_path=path, rooms=(), turn_lock=lambda _profile: nullcontext(),
        transport_resolver=lambda _binding: None)


def test_stop_deadline_includes_lifecycle_lock_wait(tmp_path):
    runtime = _runtime(tmp_path / "state.db")
    assert runtime.stop(timeout=0.05) is True
    runtime._lifecycle_lock.acquire()
    try:
        started = time.monotonic()
        assert runtime.stop(timeout=0.05) is False
        assert time.monotonic() - started < 0.5
    finally:
        runtime._lifecycle_lock.release()
    with pytest.raises(RuntimeStoreError, match="runtime_coordination_required"):
        with runtime.new_event_admission():
            pass


def test_running_admission_excludes_stop_until_released(tmp_path):
    runtime = _runtime(tmp_path / "state.db")
    runtime._thread = threading.current_thread()
    runtime._stop.clear()
    entered = threading.Event()
    release = threading.Event()

    def hold():
        with runtime.new_event_admission():
            entered.set()
            release.wait(2)

    worker = threading.Thread(target=hold)
    worker.start()
    assert entered.wait(2)
    started = time.monotonic()
    assert runtime.stop(timeout=0.05) is False
    assert time.monotonic() - started < 0.5
    release.set()
    worker.join(2)
    assert not worker.is_alive()
