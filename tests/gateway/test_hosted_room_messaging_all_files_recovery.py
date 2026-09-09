"""Interrupted pages and final native labels retain the intended file selection."""

import asyncio
import threading
import time

import pytest

from gateway import hosted_room_messaging_all_files as all_files, hosted_rooms
from gateway.config import Platform
from gateway.hosted_room_messaging_files import text
from tests.gateway.test_hosted_room_file_access import file_state
from tests.gateway.test_hosted_room_file_captions import batch, telegram_labels, locale
from tests.gateway.test_hosted_room_messaging_files import consumer, event
from tests.gateway.test_hosted_room_messaging_all_files import open_menu, share


@pytest.mark.asyncio
async def test_cancelled_next_page_does_not_drop_snapshot_rows(consumer, monkeypatch):
    state, _, _ = consumer
    for index in range(18):
        share(state, state.room, index, when=time.time() - 100 + index)
    menu, _ = await open_menu(consumer)
    collected = list(menu.pages[0]["rows"])
    entered = asyncio.Event()
    original_inventory = menu._inventory
    calls = 0

    async def paused_inventory():
        nonlocal calls
        calls += 1
        if calls == 3:
            entered.set()
            await asyncio.Event().wait()
        return await original_inventory()

    with monkeypatch.context() as patch:
        patch.setattr(menu, "_inventory", paused_inventory)
        pending = asyncio.create_task(menu.open_page(1))
        try:
            await asyncio.wait_for(entered.wait(), 3)
        finally:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
    assert len(menu.pages) == 1
    for page in range(1, 4):
        if not menu.pages[-1]["has_more"]:
            break
        await menu.open_page(page)
        collected.extend(menu.pages[page - menu.first_page]["rows"])
    actual = {item["event_id"] for _, item in collected}
    assert actual == {f"event-{index}" for index in range(18)}
    assert not menu.incomplete


@pytest.mark.asyncio
@pytest.mark.parametrize("spacing", [0, 0.125, 35])
@pytest.mark.parametrize("name", ["brief.md", "quarterly-report-long-name-" * 7 + ".md"])
async def test_complete_global_telegram_captions_preserve_versions(consumer, spacing, name):
    state, runner, _ = consumer
    hosted_rooms.rename_room(state.db, room_id="room-1", event_id="caption-rename", name="Workshop coordination room")
    for serial in range(2):
        batch(state, [name], at=1_788_509_527 + serial * spacing, serial=serial, producer="You")
    runner.config.platforms[Platform.TELEGRAM] = runner.config.platforms[Platform.SIGNAL]
    menu = all_files.AllFilesMenu(runner, event("/group files", platform=Platform.TELEGRAM), state.backend, "/group")
    page = await menu.open_page()
    labels, callback_ids = telegram_labels(page, 2)
    assert len(set(callback_ids)) == len(set(labels)) == 2
    assert all(len(caption) <= 64 for caption in labels)
    assert all(".md" in caption for caption in labels)
    if name == "brief.md":
        assert all(name in caption for caption in labels)
    assert [menu.actions[choice["value"]][1] for choice in page.choices[:2]] == menu.pages[0]["rows"]


@pytest.mark.asyncio
async def test_expired_read_budget_retains_candidates_for_page_reload(consumer, monkeypatch):
    from gateway import hosted_room_file_lookup

    state, _, _ = consumer
    for index in range(18):
        share(state, state.room, index, when=time.time() - 100 + index)
    menu, _ = await open_menu(consumer)
    original = menu._verify_rows
    release = threading.Event()
    real_resolve = hosted_room_file_lookup.resolve_file

    def blocked(**kwargs):
        assert release.wait(5)
        return real_resolve(**kwargs)

    async def expire_before_verification(rows, current):
        menu.read_deadline = time.monotonic() - 1
        return await original(rows, current)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(menu, "_verify_rows", expire_before_verification)
            patch.setattr(hosted_room_file_lookup, "resolve_file", blocked)
            page = await menu.open_page(1)
    finally:
        release.set()

        async def drained():
            while menu.runner._all_group_files_read_slots._value != 4:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(drained(), 3)
    assert menu.incomplete
    assert text("reload_page") in [choice["label"] for choice in page.choices]
    assert len(menu.pages[1]["candidates"]) == 8
    await menu.open_page(1)
    assert len(menu.pages[1]["rows"]) == 8


@pytest.mark.asyncio
async def test_empty_global_view_has_no_noop_search_or_refresh(consumer):
    menu, call = await open_menu(consumer)
    assert list(menu.actions.values()) == [("groups", None)]
    assert "files <text>" not in menu.plain_files()
    assert text("all_empty") in call["title"]


@pytest.mark.asyncio
async def test_refill_timeout_offers_reload_without_losing_snapshot_cursor(consumer, monkeypatch):
    state, runner, adapter = consumer
    for index in range(18):
        share(state, state.room, index, when=time.time() - 100 + index)
    menu, _ = await open_menu(consumer)
    cursor = next(iter(menu.streams.values()))["cursor"]
    original_list = state.backend.list_files
    release = threading.Event()

    def blocked_refill(**kwargs):
        assert kwargs["cursor"] == cursor
        assert release.wait(5)
        return original_list(**kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(all_files, "BATCH_TIMEOUT", 0.03)
            patch.setattr(state.backend, "list_files", blocked_refill)
            assert await runner._handle_rooms_command(event(f"/group files --page {menu.handle} 2")) is None
    finally:
        release.set()

        async def drained():
            while runner._all_group_files_read_slots._value != 4:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(drained(), 3)
    assert len(menu.pages) == 1 and menu.failed_page == 1
    assert next(iter(menu.streams.values()))["cursor"] == cursor
    failed = adapter.pages[-1]
    assert failed["title"] == text("error")
    assert failed["choices"][0]["label"] == text("reload_page")
    assert f"files --page {menu.handle} 2`" in menu.plain_files()
    await menu.choose("chat", failed["choices"][0]["value"])
    assert menu.failed_page is None
    assert len(menu.pages[1]["rows"]) == 8
    await menu.open_page(2)
    seen = {item["event_id"] for page in menu.pages for _, item in page["rows"]}
    assert seen == {f"event-{index}" for index in range(18)}
