"""Real consumer helpers with an authorized fake native adapter and real store."""

import asyncio
import builtins
import os
import re
import sqlite3
from pathlib import Path

import pytest

from gateway.choice_picker import ChoicePage, ChoiceProgress
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.group_chat_slash import GroupChatSlashCommandsMixin
from gateway.native_document_guard import check_document_fallback, mark_native_document_guard
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource
from gateway.slash_commands_model import GatewayModelCommandsMixin
from gateway.run import GatewayRunner
from gateway import hosted_room_messaging as rooms
from gateway import hosted_rooms
from gateway.hosted_room_file_lookup import selection_digest
from tests.gateway.test_hosted_room_file_access import file_state, publish


class NativeAdapter:
    supports_choice_pages = True
    typed_command_prefix = "/"
    _owner_profile = "default"
    name = "test-native"

    def __init__(self):
        self.pages, self.documents, self.notices = [], [], []
        self.release = None
        self.started = asyncio.Event()
        self.fallback = False
        self.unknown = False

    async def send_choice_picker(self, **kwargs):
        self.pages.append(kwargs)
        return SendResult(success=True, message_id="menu")

    async def send(self, **kwargs):
        self.notices.append(kwargs)
        return SendResult(success=True, message_id="notice")

    async def _send_media_fallback_notice(
        self, _method, _kind, _path, chat_id, caption, reply_to, metadata, **_kwargs
    ):
        return await self.send(
            chat_id=chat_id,
            content=caption or "File delivery unavailable.",
            reply_to=reply_to,
            metadata=metadata,
        )

    @mark_native_document_guard
    async def send_document(self, *, chat_id, file_path, file_name=None, **kwargs):
        path = Path(file_path)
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
        self.documents.append((chat_id, path.read_bytes(), file_name, path, kwargs))
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.fallback:
            check_document_fallback()
            return await BasePlatformAdapter.send_document(
                self, chat_id, file_path, file_name=file_name
            )
        return SendResult(success=not self.unknown, message_id="document")


class Runner(GroupChatSlashCommandsMixin, GatewayModelCommandsMixin):
    _track_deferred_agent_worker = GatewayRunner._track_deferred_agent_worker
    _active_deferred_agent_worker_count = GatewayRunner._active_deferred_agent_worker_count

    def __init__(self, adapter, platform=Platform.SIGNAL):
        self.adapter = adapter
        self.config = GatewayConfig(
            platforms={
                platform: PlatformConfig(
                    enabled=True, extra={"allow_admin_from": ["owner"]}
                )
            }
        )
        adapter.config = self.config.platforms[platform]
        self.adapters = {platform: adapter}

    def _adapter_for_source(self, source):
        return self.adapter

    def _is_user_authorized_for_source(self, source):
        return source.user_id == "owner"

    def _typed_command_prefix_for(self, source):
        return "/"

    def _normalize_source_for_session_key(self, source):
        return source

    def _session_key_for_source(self, source):
        return f"{source.chat_id}:{source.user_id}"

    def _reply_anchor_for_event(self, event):
        return event.message_id

    def _thread_metadata_for_source(self, source, anchor):
        return {"thread_id": source.thread_id, "reply_to_message_id": anchor}


def event(text, *, message_id="request-1", platform=Platform.SIGNAL):
    source = SessionSource(
        platform=platform,
        chat_id="chat",
        chat_type="dm",
        user_id="owner",
        thread_id="topic",
    )
    source.profile = "default"
    source.is_one_to_one = True
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=source,
        user_id="owner",
        message_id=message_id,
    )


@pytest.fixture
def consumer(file_state, monkeypatch):
    state = file_state
    state.backend.service.status = lambda room_id: {
        "working": False,
        "blocked": False,
        "counts": {},
        "pending_actions": [],
    }
    adapter = NativeAdapter()
    runner = Runner(adapter)
    monkeypatch.setattr(rooms, "current_room_backend", lambda: state.backend)
    assert type(adapter).send_document.strict_native_document_guard is True
    return state, runner, adapter


def choice(page, prefix):
    return next(
        item["value"] for item in page.choices if item["label"].startswith(prefix)
    )


async def open_files(runner, adapter):
    assert await runner._handle_rooms_command(event("/group 1 files")) is None
    call = adapter.pages[-1]
    return call["on_choice_selected"], ChoicePage(call["title"], call["choices"])


@pytest.mark.asyncio
async def test_native_tap_gets_exact_file_and_new_arrivals_do_not_retarget(consumer):
    state, runner, adapter = consumer
    item = publish(state, "same.md", b"original bytes")
    callback, page = await open_files(runner, adapter)
    publish(state, "same.md", b"new bytes")
    progress = await callback("chat", choice(page, "same.md"))
    assert isinstance(progress, ChoiceProgress)
    sent = await progress.complete()
    assert isinstance(sent, ChoicePage) and sent.title == "File sent."
    assert [row[1] for row in adapter.documents] == [b"original bytes"]
    assert not adapter.documents[0][3].exists()
    assert adapter.documents[0][4]["metadata"]["thread_id"] == "topic"
    assert adapter.documents[0][4]["metadata"]["group_file_delivery_id"]
    await callback("chat", choice(page, "same.md"))
    assert len(adapter.documents) == 1


@pytest.mark.asyncio
async def test_double_inflight_completion_and_replayed_command_do_not_send_twice(
    consumer,
):
    state, runner, adapter = consumer
    item = publish(state)
    callback, page = await open_files(runner, adapter)
    progress = await callback("chat", choice(page, "shared.txt"))
    adapter.release = asyncio.Event()
    first = asyncio.create_task(progress.complete())
    await adapter.started.wait()
    second = await progress.complete()
    assert "already" in second
    adapter.release.set()
    await first
    await progress.complete()
    assert len(adapter.documents) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["owner", "profile", "chat", "authority", "adapter"])
async def test_stale_action_scope_is_refused(consumer, change):
    state, runner, adapter = consumer
    publish(state)
    command = event("/group 1 files")
    await runner._handle_rooms_command(command)
    call = adapter.pages[-1]
    selected = call["choices"][0]["value"]
    if change == "owner":
        runner.config.platforms[Platform.SIGNAL].extra["allow_admin_from"] = [
            "different"
        ]
    elif change == "profile":
        command.source.profile = "ops"
    elif change == "chat":
        command.source.chat_id = "different"
    elif change == "authority":
        with sqlite3.connect(state.db) as conn:
            conn.execute("UPDATE hosted_rooms SET authority_gateway_id='different'")
    else:
        runner.adapter = NativeAdapter()
    result = await call["on_choice_selected"]("chat", selected)
    if isinstance(result, ChoiceProgress):
        result = await result.complete()
    assert not adapter.documents
    assert "File sent." not in str(result)


@pytest.mark.asyncio
async def test_revocation_after_read_prevents_submit_and_cleans_bytes(
    consumer, monkeypatch
):
    state, runner, adapter = consumer
    item = publish(state)
    original = state.backend.read_file

    def revoke(**kwargs):
        result = original(**kwargs)
        runner.config.platforms[Platform.SIGNAL].extra["allow_admin_from"] = []
        return result

    monkeypatch.setattr(state.backend, "read_file", revoke)
    code = selection_digest(state.room, item)[:8]
    result = await runner._handle_rooms_command(event(f"/group 1 file {code}"))
    assert not adapter.documents
    assert "authorized" in result
    assert not list(state.db.parent.glob("group-file-delivery-tmp/send-*"))


@pytest.mark.asyncio
async def test_fallback_notice_is_never_reported_as_a_file_send(consumer):
    state, runner, adapter = consumer
    item = publish(state)
    adapter.fallback = True
    adapter.supports_choice_pages = False
    # Capability checks are class-owned; use a non-paged subclass for plain UX.
    type(adapter).supports_choice_pages = False
    try:
        code = selection_digest(state.room, item)[:8]
        result = await runner._handle_rooms_command(event(f"/group 1 file {code}"))
        assert "Delivery wasn’t confirmed" in result
        assert "File sent." not in result
        assert not any(
            "Couldn't deliver" in notice["content"] for notice in adapter.notices
        )
        assert not adapter.documents[0][3].exists()
    finally:
        type(adapter).supports_choice_pages = True


@pytest.mark.asyncio
async def test_unmarked_instance_document_override_never_receives_bytes(consumer):
    state, runner, adapter = consumer
    item = publish(state)
    calls = []

    async def unmarked_override(**kwargs):
        calls.append(kwargs)
        return SendResult(success=True, message_id="unexpected")

    adapter.send_document = unmarked_override
    code = selection_digest(state.room, item)[:8]
    result = await runner._handle_rooms_command(event(f"/group 1 file {code}"))

    assert "Open Files in Hermes Desktop" in result
    assert calls == []


@pytest.mark.asyncio
async def test_ambiguous_native_send_requires_an_explicit_send_again(consumer):
    state, runner, adapter = consumer
    publish(state)
    callback, page = await open_files(runner, adapter)
    adapter.unknown = True
    progress = await callback("chat", choice(page, "shared.txt"))
    unknown = await progress.complete()
    assert "Delivery wasn’t confirmed" in unknown.title
    assert len(adapter.documents) == 1
    adapter.unknown = False
    explicit = await callback("chat", choice(unknown, "Send again"))
    assert isinstance(explicit, ChoiceProgress)
    await explicit.complete()
    assert len(adapter.documents) == 2


@pytest.mark.asyncio
async def test_plain_codes_and_native_room_files_entry(consumer, monkeypatch):
    state, runner, adapter = consumer
    publish(state)
    monkeypatch.setattr(type(adapter), "supports_choice_pages", False)
    result = await runner._handle_rooms_command(event("/group 1 files"))
    assert re.search(r"/group 1 file [0-9a-f]{8}`", result)
    monkeypatch.setattr(type(adapter), "supports_choice_pages", True)
    await runner._handle_rooms_command(event("/group"))
    room_menu = adapter.pages[-1]
    page = await room_menu["on_choice_selected"](
        "chat", room_menu["choices"][0]["value"]
    )
    assert {item["label"] for item in page.choices} >= {"View files", "View Bots"}


@pytest.mark.asyncio
async def test_full_reply_uses_actual_latest_event_not_the_preview(consumer):
    state, runner, adapter = consumer
    text = "# Complete work\n" + "not a preview\n" * 100
    hosted_rooms.append_event(
        state.db,
        room_id="room-1",
        event_id="bot-reply",
        kind="message.member",
        actor={"kind": "member", "id": "ops"},
        payload={"text": text},
        authority_gateway_id=state.authority,
        authority_epoch=1,
    )
    for index in range(85):
        hosted_rooms.append_event(
            state.db,
            room_id="room-1",
            event_id=f"ordinary-{index}",
            kind="message.user",
            actor={"kind": "user", "id": "desktop"},
            payload={"text": "ordinary"},
            authority_gateway_id=state.authority,
            authority_epoch=1,
        )
    result = await runner._handle_rooms_command(event("/group 1 reply"))
    assert result == "File sent."
    assert adapter.documents[0][1] == text.encode()
    assert adapter.documents[0][2] == "reply.md"


@pytest.mark.asyncio
async def test_text_only_source_does_not_import_files_for_help_or_status(
    consumer, monkeypatch
):
    state, runner, adapter = consumer
    monkeypatch.setattr(type(adapter), "supports_choice_pages", False)
    original = builtins.__import__

    def without_files(name, *args, **kwargs):
        if name.startswith((
            "gateway.hosted_room_file",
            "gateway.hosted_room_control_files",
            "gateway.hosted_room_shared_message",
        )):
            raise ImportError("Files not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_files)
    assert "Group Chats" in await runner._handle_rooms_command(event("/group help"))
    assert (
        "couldn" not in (await runner._handle_rooms_command(event("/group 1"))).lower()
    )
    result = await runner._handle_rooms_command(event("/group 1 file abcdef12"))
    assert "isn't available" in result


@pytest.mark.asyncio
async def test_collision_choices_use_exact_matches_not_an_unrelated_first_page(
    consumer, monkeypatch
):
    from gateway import hosted_room_file_lookup as lookup
    import hashlib

    state, runner, adapter = consumer
    first, second = publish(state, "old-a.txt", b"a"), publish(state, "old-b.txt", b"b")
    original = lookup.selection_digest
    selected_ids = {first["attachment_id"], second["attachment_id"]}

    def collision(room, item):
        if item["attachment_id"] in selected_ids:
            return (
                "deadbeef"
                + hashlib.sha256(item["attachment_id"].encode()).hexdigest()[8:]
            )
        return original(room, item)

    monkeypatch.setattr(lookup, "selection_digest", collision)
    for index in range(10):
        publish(state, f"unrelated-{index}.txt")
    result = await runner._handle_rooms_command(event("/group 1 file deadbeef"))
    assert result is None
    call = adapter.pages[-1]
    assert any(item["label"].startswith("old-b.txt") for item in call["choices"])
    assert not any(item["label"].startswith("unrelated") for item in call["choices"])
    token = next(
        item["value"]
        for item in call["choices"]
        if item["label"].startswith("old-b.txt")
    )
    progress = await call["on_choice_selected"]("chat", token)
    await progress.complete()
    assert [document[1] for document in adapter.documents] == [b"b"]


@pytest.mark.asyncio
async def test_plain_older_newer_keep_snapshot_and_reject_another_actor(
    consumer, monkeypatch
):
    state, runner, adapter = consumer
    for index in range(9):
        publish(state, f"file-{index}.txt")
    monkeypatch.setattr(type(adapter), "supports_choice_pages", False)
    first = await runner._handle_rooms_command(event("/group 1 files"))
    handle = re.search(r"--older ([0-9a-f]+)", first).group(1)
    publish(state, "new-arrival.txt")
    second = await runner._handle_rooms_command(
        event(f"/group 1 files --older {handle}", message_id="older")
    )
    assert "file-0.txt" in second and "new-arrival" not in second
    newer = re.search(r"--newer ([0-9a-f]+)", second).group(1)
    back = await runner._handle_rooms_command(
        event(f"/group 1 files --newer {newer}", message_id="newer")
    )
    assert "file-8.txt" in back and "new-arrival" not in back
    foreign = event(f"/group 1 files --older {handle}")
    foreign.source.user_id = "other"
    assert "file-0.txt" not in await runner._handle_rooms_command(foreign)


@pytest.mark.asyncio
async def test_stop_remains_responsive_while_document_send_is_inflight(
    consumer, monkeypatch
):
    state, runner, adapter = consumer
    publish(state)
    monkeypatch.setattr(state.backend, "stop_room", lambda *args, **kwargs: 0)
    callback, page = await open_files(runner, adapter)
    progress = await callback("chat", choice(page, "shared.txt"))
    adapter.release = asyncio.Event()
    sending = asyncio.create_task(progress.complete())
    await adapter.started.wait()
    try:
        stopped = await asyncio.wait_for(
            runner._handle_rooms_command(event("/group 1 stop", message_id="stop")), 2
        )
        assert "Stop requested" in stopped
    finally:
        adapter.release.set()
        await sending


@pytest.mark.asyncio
async def test_expired_menu_and_unsafe_labels_do_not_retarget(consumer):
    from gateway.hosted_room_messaging_files import label
    import time

    state, runner, adapter = consumer
    publish(state)
    callback, page = await open_files(runner, adapter)
    callback.__self__.deadline = time.monotonic() - 1
    assert "expired" in await callback("chat", choice(page, "shared.txt"))
    assert not adapter.documents
    safe = label("/group 7 stop\u202e @all **name**")
    assert not safe.startswith("/") and "\u202e" not in safe and "@" not in safe


@pytest.mark.asyncio
async def test_transient_error_retry_and_cursor_reset_return_to_latest(
    consumer, monkeypatch
):
    from gateway.hosted_room_file_contract import FileAccessError

    state, runner, adapter = consumer
    for index in range(9):
        publish(state, f"file-{index}.txt")
    callback, page = await open_files(runner, adapter)
    original = state.backend.list_files

    def fail(**kwargs):
        raise FileAccessError("file_host_unavailable", retryable=True)

    monkeypatch.setattr(state.backend, "list_files", fail)
    failed = await callback("chat", choice(page, "Older"))
    assert isinstance(failed, ChoicePage)
    assert {item["label"] for item in failed.choices} == {
        "Retry",
        "Show latest",
        "Back",
    }
    monkeypatch.setattr(state.backend, "list_files", original)
    older = await callback("chat", choice(failed, "Retry"))
    assert any(item["label"].startswith("file-0") for item in older.choices)

    def reset(**kwargs):
        raise FileAccessError("attachment_cursor_reset_required")

    monkeypatch.setattr(state.backend, "list_files", reset)
    failed = await callback("chat", choice(older, "Show latest"))
    assert {item["label"] for item in failed.choices} == {"Show latest", "Back"}
    monkeypatch.setattr(state.backend, "list_files", original)
    latest = await callback("chat", choice(failed, "Show latest"))
    assert any(item["label"].startswith("file-8") for item in latest.choices)


@pytest.mark.asyncio
async def test_file_rate_bucket_is_separate_bounded_and_does_not_block_stop(
    consumer, monkeypatch
):
    from gateway import hosted_room_messaging_files as files

    state, runner, adapter = consumer
    publish(state)
    monkeypatch.setattr(files, "_RATE_LIMITS", {"read": 1, "send": 1})
    await open_files(runner, adapter)
    monkeypatch.setattr(
        state.backend, "list_files", lambda **kwargs: pytest.fail("rate-limited browse")
    )
    assert "Too many file requests" in await runner._handle_rooms_command(
        event("/group 1 files")
    )
    monkeypatch.setattr(state.backend, "stop_room", lambda *args, **kwargs: 0)
    assert "Stop requested" in await runner._handle_rooms_command(
        event("/group 1 stop")
    )
    source = files._source_key(runner, event(""))
    assert files._rate(runner, source, "send")
    assert not files._rate(runner, source, "send")
    for index in range(2050):
        files._rate(runner, f"source-{index}", "read")
    assert len(runner._group_file_rates) == 2048
