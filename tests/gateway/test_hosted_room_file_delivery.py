"""Durable at-most-one attempts, native-only success, caps and private cleanup."""

import asyncio
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gateway import hosted_room_file_delivery as delivery
from gateway.native_document_guard import (
    NativeDocumentFallback,
    mark_native_document_guard,
    require_native_document,
)
from gateway.platforms.base import BasePlatformAdapter
from gateway.config import Platform
from gateway.choice_picker import ChoicePage, ChoiceProgress
from tests.gateway.test_hosted_room_file_access import file_state, publish
from tests.gateway.test_hosted_room_messaging_files import (
    consumer,
    event,
    choice,
    open_files,
    Runner,
)


def test_receipt_claim_is_atomic_and_identity_bound(file_state):
    db = file_state.db
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: delivery.reserve_delivery(db, "key", "scope"), range(8))
        )
    assert results.count("new") == 1
    assert results.count("busy") == 7
    with pytest.raises(delivery.FileDeliveryError):
        delivery.reserve_delivery(db, "key", "different-scope")
    delivery.mark_delivery(db, "key", "sending")
    assert delivery.reserve_delivery(db, "key", "scope") == "busy"
    delivery.mark_delivery(db, "key", "unknown")
    assert delivery.reserve_delivery(db, "key", "scope") == "unknown"
    assert delivery.reserve_delivery(db, "explicit-new-request", "scope") == "new"


def test_receipt_capacity_never_evicts_recent_successes_for_replay(
    file_state, monkeypatch
):
    db = file_state.db
    monkeypatch.setattr(delivery, "MAX_RECEIPTS", 1)
    assert delivery.reserve_delivery(db, "key", "scope") == "new"
    delivery.mark_delivery(db, "key", "sending")
    delivery.mark_delivery(db, "key", "delivered")
    assert delivery.reserve_delivery(db, "key", "scope") == "delivered"
    with pytest.raises(delivery.FileDeliveryError):
        delivery.reserve_delivery(db, "different", "scope")


@pytest.mark.asyncio
async def test_native_fallback_guard_is_opt_in_and_context_local(consumer, tmp_path):
    _state, _runner, adapter = consumer
    path = tmp_path / "fallback.txt"
    path.write_text("fallback", encoding="utf-8")
    path.chmod(0o600)
    adapter.fallback = True
    result = await adapter.send_document(
        chat_id="chat", file_path=str(path), file_name="visible.txt"
    )
    assert result.success is True
    before = len(adapter.notices)
    with require_native_document():
        with pytest.raises(NativeDocumentFallback):
            await adapter.send_document(
                chat_id="chat", file_path=str(path), file_name="visible.txt"
            )
    assert len(adapter.notices) == before
    assert (
        await adapter.send_document(chat_id="chat", file_path=str(path))
    ).success


@pytest.mark.asyncio
async def test_missing_production_guard_fails_closed_before_loading(
    consumer, monkeypatch
):
    state, runner, adapter = consumer
    publish(state)

    async def missing_guard(*_args, **_kwargs):
        pytest.fail("an unadvertised native document fallback was called")

    monkeypatch.setattr(type(adapter), "send_document", missing_guard)
    monkeypatch.setattr(
        state.backend,
        "read_file",
        lambda **kwargs: pytest.fail("unsupported delivery downloaded bytes"),
    )
    callback, page = await open_files(runner, adapter)
    result = await callback("chat", choice(page, "shared.txt"))
    assert isinstance(result, ChoicePage) and "Desktop" in result.title
    assert adapter.documents == []


@pytest.mark.asyncio
async def test_discord_twelve_mb_is_rejected_before_download(consumer, monkeypatch):
    state, _runner, adapter = consumer
    publish(state, "large.txt", b"x" * 12_000_000)
    runner = Runner(adapter, Platform.DISCORD)
    monkeypatch.setattr(
        state.backend,
        "read_file",
        lambda **kwargs: pytest.fail("over-cap file was fetched"),
    )
    await runner._handle_rooms_command(
        event("/group 1 files", platform=Platform.DISCORD)
    )
    call = adapter.pages[-1]
    result = await call["on_choice_selected"]("chat", call["choices"][0]["value"])
    assert isinstance(result, ChoicePage) and "too large" in result.title
    assert adapter.documents == []


@pytest.mark.asyncio
async def test_large_supported_file_requires_explicit_confirmation(
    consumer, monkeypatch
):
    state, runner, adapter = consumer
    publish(state, "large.txt", b"x" * 11_000_000)
    original = state.backend.read_file
    monkeypatch.setattr(
        state.backend,
        "read_file",
        lambda **kwargs: pytest.fail("confirmation fetched bytes"),
    )
    callback, page = await open_files(runner, adapter)
    confirm = await callback("chat", choice(page, "large.txt"))
    assert isinstance(confirm, ChoicePage) and "11.0 MB" in confirm.title
    assert adapter.documents == []
    monkeypatch.setattr(state.backend, "read_file", original)
    progress = await callback("chat", choice(confirm, "Send"))
    assert isinstance(progress, ChoiceProgress)
    await progress.complete()
    assert len(adapter.documents) == 1
    assert not adapter.documents[0][3].exists()


@pytest.mark.asyncio
async def test_cancelled_send_is_unknown_and_temp_bytes_are_removed(consumer):
    state, runner, adapter = consumer
    publish(state)
    callback, page = await open_files(runner, adapter)
    progress = await callback("chat", choice(page, "shared.txt"))
    adapter.release = asyncio.Event()
    pending = asyncio.create_task(progress.complete())
    await adapter.started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not adapter.documents[0][3].exists()
    repeated = await progress.complete()
    assert (
        isinstance(repeated, ChoicePage) and "Delivery wasn’t confirmed" in repeated.title
    )
    assert len(adapter.documents) == 1
    with sqlite3.connect(state.db) as conn:
        assert (
            conn.execute("SELECT state FROM hosted_room_file_deliveries").fetchone()[0]
            == "unknown"
        )


@pytest.mark.asyncio
async def test_delivery_success_is_not_relabelled_failure_when_menu_expires(
    consumer, monkeypatch
):
    _state, runner, adapter = consumer
    publish(_state)
    callback, page = await open_files(runner, adapter)
    menu = callback.__self__
    original = adapter.send_document

    @mark_native_document_guard
    async def expires(**kwargs):
        result = await original(**kwargs)
        menu.deadline = time.monotonic() - 1
        return result

    monkeypatch.setattr(adapter, "send_document", expires)
    progress = await callback("chat", choice(page, "shared.txt"))
    assert await progress.complete() == "File sent."


def test_stale_private_temp_cleanup_never_follows_symlink(file_state, tmp_path):
    root = file_state.db.parent / "group-file-delivery-tmp"
    root.mkdir(mode=0o700)
    stale = root / "send-stale"
    stale.mkdir(mode=0o700)
    (stale / "document").write_bytes(b"old private bytes")
    os.utime(stale, (time.time() - 7200, time.time() - 7200))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_bytes(b"keep")
    (root / "send-link").symlink_to(outside, target_is_directory=True)
    folder, path = delivery._temporary_document(
        file_state.db, delivery.Document("file.txt", b"new")
    )
    try:
        assert path.read_bytes() == b"new"
        assert not stale.exists()
        assert (outside / "keep").read_bytes() == b"keep"
    finally:
        folder.cleanup()


def test_long_unicode_filename_is_safe_for_the_temporary_filesystem(file_state):
    folder, path = delivery._temporary_document(
        file_state.db, delivery.Document("\u7814" * 250 + ".txt", b"exact")
    )
    try:
        assert len(path.name.encode("utf-8")) <= 220
        assert path.suffix == ".txt"
        assert path.read_bytes() == b"exact"
    finally:
        folder.cleanup()


@pytest.mark.asyncio
async def test_final_authority_check_runs_after_the_sending_receipt(
    consumer, monkeypatch
):
    state, runner, adapter = consumer
    publish(state)
    original = delivery.mark_delivery

    def revoke(db, key, status):
        original(db, key, status)
        if status == "sending":
            runner.config.platforms[Platform.SIGNAL].extra["allow_admin_from"] = []

    monkeypatch.setattr(delivery, "mark_delivery", revoke)
    callback, page = await open_files(runner, adapter)
    progress = await callback("chat", choice(page, "shared.txt"))
    result = await progress.complete()
    assert "You can no longer get this file here" in result
    assert not adapter.documents
    assert list((state.db.parent / "group-file-delivery-tmp").iterdir()) == []


@pytest.mark.asyncio
async def test_full_reply_change_before_submission_cannot_change_the_selected_document(
    consumer, monkeypatch
):
    state, runner, adapter = consumer
    publish(state, actor={"kind": "member", "id": "ops"})
    original = delivery.mark_delivery

    def alter_reply(db, key, status):
        original(db, key, status)
        if status == "sending":
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "UPDATE hosted_room_events SET payload_json=json_set(payload_json, '$.text', 'changed') WHERE kind='message.member'"
                )

    monkeypatch.setattr(delivery, "mark_delivery", alter_reply)
    result = await runner._handle_rooms_command(event("/group 1 reply"))
    assert "You can no longer get this file here" in result
    assert not adapter.documents
