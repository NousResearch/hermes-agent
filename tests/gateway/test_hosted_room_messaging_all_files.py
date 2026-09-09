"""Cross-room files retain exact identity, current authorization and bounded I/O."""

import asyncio
import threading
import time

import pytest

from gateway import hosted_room_messaging_all_files as all_files, hosted_rooms
from gateway.choice_picker import ChoicePage, ChoiceProgress
from gateway.hosted_room_file_contract import MANIFEST_FIELDS
from gateway.hosted_room_messaging_files import text
from tests.gateway.test_hosted_room_file_access import file_state, publish
from tests.gateway.test_hosted_room_messaging_files import consumer, event


def share(state, room, index, *, name=None, when=None, viewer=True):
    data = f"{room['room_id']}:{index}".encode()
    stored = state.store.put(
        room_id=room["room_id"], upload_id=f"upload-{index}", kind="file",
        name=name or f"result-{index}.txt", mime="text/plain", data=data,
    )
    manifest = {key: stored[key] for key in MANIFEST_FIELDS}
    state.store.commit_message(
        room_id=room["room_id"], event_id=f"event-{index}", manifest=[manifest],
        recipient_member_ids=[member["member_id"] for member in room["members"]],
        viewer_access=viewer, hold_until_event=True,
    )
    if not viewer:
        return data
    hosted_rooms.append_event(
        state.db, room_id=room["room_id"], event_id=f"event-{index}",
        kind="message.user", actor={"kind": "user", "id": "desktop"},
        authority_gateway_id=state.authority, authority_epoch=1,
        payload={"text": "Shared result", "attachments": [manifest]}, now=when,
    )
    return data


def second_room(state):
    return hosted_rooms.create_room(
        state.db, room_id="room-2", name="Other project", members=[
            {"member_id": "writer", "profile": "writer", "handle": "writer"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ], authority_gateway_id=state.authority,
    )


async def open_menu(consumer, query=""):
    state, runner, adapter = consumer
    assert await runner._handle_rooms_command(event("/group files " + query)) is None
    menu = next(reversed(runner._all_group_file_menus.values()))
    return menu, adapter.pages[-1]


@pytest.mark.asyncio
async def test_reserved_files_command_does_not_open_room_named_files(consumer):
    state, _runner, _adapter = consumer
    publish(state)
    menu, call = await open_menu(consumer)
    assert menu.room is None
    assert text("all_title") in call["title"]
    assert len(menu.pages[0]["rows"]) == 1


@pytest.mark.asyncio
async def test_cross_room_native_selection_sends_exact_bytes_and_no_private_files(consumer):
    state, _runner, adapter = consumer
    other = second_room(state)
    now = time.time()
    share(state, state.room, 1, name="same.txt", when=now - 30)
    expected = share(state, other, 1, name="same.txt", when=now - 5)
    share(state, other, 2, name="private.txt", when=now, viewer=False)
    menu, call = await open_menu(consumer)
    assert [room["room_id"] for room, _ in menu.pages[0]["rows"]] == ["room-2", "room-1"]
    assert "private.txt" not in str(call)
    assert len({choice["label"] for choice in call["choices"][:2]}) == 2
    action = await menu.choose("chat", call["choices"][0]["value"])
    assert isinstance(action, ChoiceProgress)
    await action.complete()
    assert [document[1] for document in adapter.documents] == [expected]
    assert await menu.choose("chat", call["choices"][0]["value"]) == text("expired")
    assert len(adapter.documents) == 1


@pytest.mark.asyncio
async def test_global_pages_cross_per_room_cursor_and_do_not_repeat_new_arrivals(consumer):
    state, runner, _adapter = consumer
    other = second_room(state)
    now = time.time()
    for index in range(12):
        share(state, state.room, index, when=now - 100 + 2 * index)
        share(state, other, index, when=now - 99 + 2 * index)
    menu, _call = await open_menu(consumer)
    rows = list(menu.pages[0]["rows"])
    share(state, other, 20, name="new-arrival.txt", when=now)
    for page in range(2, 4):
        assert await runner._handle_rooms_command(event(f"/group files --page {menu.handle} {page}")) is None
        rows.extend(menu.pages[menu.position]["rows"])
    assert len(rows) == 24
    assert len({(room["room_id"], item["event_id"], item["attachment_id"]) for room, item in rows}) == 24
    assert [item["shared_at"] for _, item in rows] == sorted([item["shared_at"] for _, item in rows], reverse=True)
    assert not menu.pages[-1]["has_more"]
    assert not any(item["name"] == "new-arrival.txt" for _, item in rows)


@pytest.mark.asyncio
async def test_old_menu_checks_current_owner_before_any_catalog_reads(consumer, monkeypatch):
    state, runner, adapter = consumer
    publish(state)
    menu, call = await open_menu(consumer)
    adapter.config.extra["allow_admin_from"] = ["another-owner"]
    monkeypatch.setattr(state.backend, "list_files", lambda **kwargs: pytest.fail("unauthorized catalog read"))
    assert text("delivered") not in str(await menu.choose("chat", call["choices"][0]["value"]))
    assert await runner._handle_rooms_command(event(f"/group files --page {menu.handle} 1"))
    assert adapter.documents == []


@pytest.mark.asyncio
async def test_removed_room_drops_cached_rows_and_stale_download(consumer):
    state, _runner, adapter = consumer
    publish(state)
    menu, call = await open_menu(consumer)
    hosted_rooms.disband_room(state.db, room_id="room-1", expected_gateway_id=state.authority, expected_epoch=1)
    result = await menu.choose("chat", call["choices"][0]["value"])
    assert not isinstance(result, ChoiceProgress)
    assert adapter.documents == []
    await menu.open_page(0)
    assert menu.pages[0]["rows"] == []


@pytest.mark.asyncio
async def test_partial_catalog_failure_does_not_hide_healthy_files_or_claim_complete(consumer, monkeypatch):
    state, _runner, _adapter = consumer
    other = second_room(state)
    publish(state)
    share(state, other, 1)
    original = state.backend.list_files

    def unavailable(**kwargs):
        if kwargs["room"]["room_id"] == "room-2":
            raise ConnectionError("offline")
        return original(**kwargs)

    monkeypatch.setattr(state.backend, "list_files", unavailable)
    menu, call = await open_menu(consumer)
    assert text("some_unavailable") in call["title"]
    assert len(menu.pages[0]["rows"]) == 1
    assert text("some_unavailable") in menu.plain_files()


@pytest.mark.asyncio
async def test_global_search_and_plain_download_hints_keep_client_prefix(consumer, monkeypatch):
    state, runner, adapter = consumer
    other = second_room(state)
    share(state, state.room, 1, name="plan.md")
    share(state, other, 1, name="other.md")
    runner._typed_command_prefix_for = lambda source: "!"
    monkeypatch.setattr(type(adapter), "supports_choice_pages", False)
    result = await runner._handle_rooms_command(event("!group files plan"))
    assert "plan.md" in result and "other.md" not in result
    assert "Download: `!group 1 file " in result
    assert "!group files <text>" in result
    assert "reply`" not in result and "Back:" not in result


@pytest.mark.asyncio
async def test_timeout_does_not_release_thread_slots_before_io_completes(consumer, monkeypatch):
    _state, runner, _adapter = consumer
    menu = all_files.AllFilesMenu(runner, event("/group files"), consumer[0].backend, "/group")
    release = threading.Event()
    entered = []
    monkeypatch.setattr(all_files, "BATCH_TIMEOUT", 0.1)

    def blocked():
        entered.append(1)
        release.wait(5)

    try:
        await menu._batch([menu._read(blocked) for _ in range(12)])
        assert len(entered) == 4
        await menu._batch([menu._read(blocked) for _ in range(4)])
        assert len(entered) == 4
    finally:
        release.set()
        for _ in range(100):
            if runner._all_group_files_read_slots._value == 4:
                break
            await asyncio.sleep(0.01)
    assert runner._all_group_files_read_slots._value == 4


@pytest.mark.asyncio
async def test_plain_page_tokens_cannot_cross_people_or_chats(consumer):
    state, runner, _adapter = consumer
    publish(state)
    menu, _ = await open_menu(consumer)
    command = event(f"/group files --page {menu.handle} 1")
    command.source.chat_id = "different-chat"
    assert await runner._handle_rooms_command(command) == text("expired")


@pytest.mark.asyncio
async def test_global_snapshots_do_not_duplicate_into_larger_room_menu_cache(consumer):
    publish(consumer[0])
    menu, _ = await open_menu(consumer)
    assert menu.handle not in getattr(consumer[1], "_group_file_menus", {})


@pytest.mark.asyncio
async def test_refill_midpage_keeps_uneven_room_streams_in_order(consumer):
    state, _runner, _ = consumer
    other = second_room(state)
    now = time.time()
    for index in range(20):
        share(state, state.room, index, when=now - 200 + index * 7)
        share(state, other, index, when=now - 100 + index)
    menu, _ = await open_menu(consumer)
    rows = list(menu.pages[0]["rows"])
    for page in range(1, 5):
        await menu.open_page(page)
        rows.extend(menu.pages[page]["rows"])
    stamps = [item["shared_at"] for _, item in rows]
    assert len(rows) == 40
    assert stamps == sorted(stamps, reverse=True)


@pytest.mark.asyncio
async def test_window_eviction_keeps_page_numbers_stable(consumer):
    state, runner, _ = consumer
    now = time.time()
    for index in range(73):
        share(state, state.room, index, when=now - 100 + index)
    menu, _ = await open_menu(consumer)
    for page in range(1, 10):
        await menu.open_page(page)
    assert len(menu.pages) == 8 and menu.first_page == 2 and menu.position == 9
    assert "page 10" in menu.plain_files()
    assert f"files --page {menu.handle} 9`" in menu.plain_files()
    assert await runner._handle_rooms_command(event(f"/group files --page {menu.handle} 1")) == text("expired")
    assert await runner._handle_rooms_command(event(f"/group files --page {menu.handle} 9")) is None
    assert menu.position == 8


@pytest.mark.asyncio
async def test_same_room_receiving_identity_change_discards_metadata_before_display(consumer, monkeypatch):
    state, runner, adapter = consumer
    publish(state, "should-not-leak.txt")
    original = state.backend.list_files

    def change_owner(**kwargs):
        result = original(**kwargs)
        adapter.config.extra["allow_admin_from"] = ["another-owner"]
        return result

    monkeypatch.setattr(state.backend, "list_files", change_owner)
    result = await runner._handle_rooms_command(event("/group files"))
    assert "should-not-leak.txt" not in str(result)
    assert adapter.pages == [] and adapter.documents == []


@pytest.mark.asyncio
async def test_global_menu_eviction_expires_native_callbacks_and_clears_buffers(consumer, monkeypatch):
    state, runner, _ = consumer
    publish(state)
    monkeypatch.setattr(all_files, "MAX_GLOBAL_MENUS", 1)
    previous, call = await open_menu(consumer)
    current, _ = await open_menu(consumer)
    assert list(runner._all_group_file_menus) == [current.handle]
    assert previous.streams == {} and previous.pages == []
    assert await previous.choose("chat", call["choices"][0]["value"]) == text("expired")


@pytest.mark.asyncio
async def test_text_group_views_discover_files_without_probing_empty_catalogs(consumer, monkeypatch):
    state, runner, adapter = consumer
    monkeypatch.setattr(type(adapter), "supports_choice_pages", False)
    runner._typed_command_prefix_for = lambda source: "!"
    monkeypatch.setattr(state.backend, "list_files", lambda **kwargs: pytest.fail("eager catalog read"))
    detail = await runner._handle_rooms_command(event("!group 1"))
    assert "View files: `!group 1 files`" in detail
    listing = await runner._handle_rooms_command(event("!group list"))
    assert "View files: `!group files`" in listing
    help_text = runner._group_chat_help("!group")
    assert "`!group files [query]`" in help_text
