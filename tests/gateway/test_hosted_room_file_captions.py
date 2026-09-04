"""Final renderer captions stay distinct for valid canonical file shares."""

from copy import deepcopy
import sys
from types import ModuleType

import pytest

from agent import i18n
from gateway import hosted_rooms
from gateway import hosted_room_messaging_files as files
from gateway.choice_picker import choice_label
from gateway.config import Platform
from gateway.hosted_room_file_contract import MANIFEST_FIELDS
from tests.gateway.test_hosted_room_file_access import file_state
from tests.gateway.test_hosted_room_messaging_files import consumer, event


@pytest.fixture(autouse=True)
def locale(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    monkeypatch.setenv("HERMES_TIMEZONE", "UTC")
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


def batch(state, names, *, at, serial, producer="Release automation team"):
    state.store.clock = lambda: at
    manifest = []
    for index, name in enumerate(names):
        stored = state.store.put(
            room_id="room-1", upload_id=f"caption-{serial}-{index}",
            kind="file", name=name, mime="text/plain",
            data=f"original output {serial}-{index}".encode(),
        )
        manifest.append({key: stored[key] for key in MANIFEST_FIELDS})
    event_id = f"caption-{serial}"
    state.store.commit_message(
        room_id="room-1", event_id=event_id, manifest=manifest,
        recipient_member_ids=["peer", "ops"], viewer_access=True, hold_until_event=True,
    )
    hosted_rooms.append_event(
        state.db, room_id="room-1", event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "desktop", "display_name": producer},
        authority_gateway_id=state.authority, authority_epoch=1,
        payload={"text": "caption fixture", "thread_id": "thread-1", "attachments": manifest},
        now=at,
    )


async def menu_for(consumer, platform=Platform.SIGNAL):
    state, runner, _ = consumer
    runner.config.platforms[platform] = runner.config.platforms[Platform.SIGNAL]
    menu = files.FilesMenu(runner, event("/group 1 files", platform=platform), state.backend, "/group")
    await menu.bind("1")
    return menu, await menu.files_page()


def telegram_labels(page, count):
    if not isinstance(sys.modules.get("telegram"), ModuleType):
        for name in list(sys.modules):
            if name == "telegram" or name.startswith("telegram."):
                sys.modules.pop(name)
    import telegram
    from plugins.platforms.telegram.choice_picker import _keyboard

    assert isinstance(telegram.__file__, str) and isinstance(telegram.__version__, str)
    markup = _keyboard(page.choices[:count], "caption-test", 1,
                       telegram.InlineKeyboardButton, telegram.InlineKeyboardMarkup)
    buttons = [button for row in markup.inline_keyboard for button in row]
    return [button.text for button in buttons], [button.callback_data for button in buttons]


@pytest.mark.asyncio
async def test_real_batches_preserve_filename_tail_after_final_100_character_budget(consumer):
    state, _, _ = consumer
    names = ["project-" * 5 + "A.md", "project-" * 5 + "B.md"]
    for serial in range(2):
        batch(state, names, at=1_788_509_527 + serial * 0.125, serial=serial)
    menu, page = await menu_for(consumer)
    items = deepcopy(menu.pages[0]["items"])
    assert len(items) == 4
    assert {item["shared_at"] for item in items} == {1_788_509_527, 1_788_509_527.125}
    captions = [choice_label(choice) for choice in page.choices[:4]]
    assert [menu.actions[choice["value"]][1][0] for choice in page.choices[:4]] == items
    assert len(set(captions)) == 4
    assert all(item["name"][-4:] in caption for item, caption in zip(items, captions))
    assert all(caption in menu.plain_files() for caption in captions)


@pytest.mark.asyncio
@pytest.mark.parametrize("spacing", [35, 0])
async def test_real_telegram_markup_preserves_distinct_versions_at_64_characters(consumer, spacing):
    state, _, _ = consumer
    name = "quarterly-report-long-name-" * 7 + ".md"
    for serial in range(2):
        batch(state, [name], at=1_788_509_527 + serial * spacing, serial=serial, producer="You")
    menu, page = await menu_for(consumer, Platform.TELEGRAM)
    captions, actions = telegram_labels(page, 2)
    assert len(set(actions)) == 2
    assert len(set(captions)) == 2
    assert all(len(caption) <= 64 for caption in captions)
    assert all(caption in menu.plain_files() for caption in captions)
    assert [menu.actions[choice["value"]][1][0] for choice in page.choices[:2]] == menu.pages[0]["items"]


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.SIGNAL, Platform.TELEGRAM])
async def test_normal_rows_keep_names_and_information_without_display_codes(consumer, monkeypatch, platform):
    from gateway import hosted_room_file_lookup as lookup

    state, _, _ = consumer
    batch(state, ["brief.md", "notes.md"], at=1_788_509_527, serial=0, producer="You")
    menu, _ = await menu_for(consumer, platform)
    monkeypatch.setattr(lookup, "selection_digest", lambda *args: pytest.fail("unambiguous captions computed a code"))
    page = menu.render_files()
    captions = ([choice_label(choice) for choice in page.choices[:2]]
                if platform == Platform.SIGNAL else telegram_labels(page, 2)[0])
    for item, caption in zip(menu.pages[0]["items"], captions):
        assert caption.startswith(item["name"])
        assert "You" in caption and "08:12" in caption and "1 KB" in caption
        assert "[" not in caption


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_prefix", [0, 8, 60])
async def test_actual_residual_collisions_reuse_unique_codes_that_survive_telegram(consumer, monkeypatch, shared_prefix):
    from gateway import hosted_room_file_lookup as lookup

    state, _, _ = consumer
    names = ["prefix-" * 5 + marker + "-suffix" * 5 + ".md" for marker in ("A", "B")]
    batch(state, names + ["ordinary.md"], at=1_788_509_527, serial=0)
    menu, _ = await menu_for(consumer, Platform.TELEGRAM)
    items = menu.pages[0]["items"]
    ambiguous = [item for item in items if item["name"] != "ordinary.md"]
    codes = {item["attachment_id"]: "a" * shared_prefix + str(index) * (64 - shared_prefix)
             for index, item in enumerate(ambiguous)}
    original = lookup.selection_digest
    calls = []

    def digest(room, item):
        calls.append(item["attachment_id"])
        return codes.get(item["attachment_id"]) or original(room, item)

    monkeypatch.setattr(lookup, "selection_digest", digest)
    expected = deepcopy(items)
    page = menu.render_files()
    captions, callback_ids = telegram_labels(page, 3)
    assert len(set(captions)) == len(set(callback_ids)) == 3
    assert all(len(caption) <= 64 for caption in captions)
    assert set(calls) == set(codes) and len(calls) == 2
    for item, caption in zip(items, captions):
        if item["attachment_id"] in codes:
            length = max(8, shared_prefix + 4)
            assert codes[item["attachment_id"]][:length] in caption
        else:
            assert caption.startswith("ordinary.md") and "[" not in caption
    assert [menu.actions[choice["value"]][1][0] for choice in page.choices[:3]] == expected
    assert all(caption in menu.plain_files() for caption in captions)


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", i18n.SUPPORTED_LANGUAGES)
@pytest.mark.parametrize("platform", [Platform.SIGNAL, Platform.TELEGRAM])
async def test_all_locale_final_captions_keep_the_same_exact_selections(consumer, monkeypatch, lang, platform):
    state, _, _ = consumer
    names = ["project-" * 5 + "A.md", "project-" * 5 + "B.md"]
    for serial in range(2):
        batch(state, names, at=1_788_509_527 + serial * 0.125, serial=serial)
    menu, _ = await menu_for(consumer, platform)
    expected = deepcopy(menu.pages[0]["items"])
    source_key = menu.source_key
    monkeypatch.setenv("HERMES_LANGUAGE", lang)
    monkeypatch.setenv("HERMES_TIMEZONE", "Asia/Kolkata")
    page = menu.render_files()
    captions = ([choice_label(choice) for choice in page.choices[:4]]
                if platform == Platform.SIGNAL else telegram_labels(page, 4)[0])
    assert len(set(captions)) == 4
    assert all("13:42:07" in caption for caption in captions)
    assert all(caption in menu.plain_files() for caption in captions)
    assert [menu.actions[choice["value"]][1][0] for choice in page.choices[:4]] == expected
    assert menu.source_key == source_key == files._source_key(menu.runner, menu.event)


@pytest.mark.asyncio
async def test_loaded_page_rendering_is_batched_and_does_not_fetch(consumer, monkeypatch):
    import hermes_time

    state, _, _ = consumer
    for serial in range(8):
        batch(state, [f"file-{serial}-{index}.md" for index in range(8)],
              at=1_788_509_527 + serial, serial=serial)
    menu, _ = await menu_for(consumer, Platform.TELEGRAM)
    while menu.pages[menu.position]["has_more"]:
        await menu.files_page(menu.pages[menu.position]["next_cursor"], direction="older")
    assert len(menu.pages) == 8
    count = sum(len(page["items"]) for page in menu.pages)
    assert count == 64
    calls = []
    zones = []
    original_text, original_zone = files.text, hermes_time.get_timezone

    def counted(key, **values):
        calls.append(key)
        return original_text(key, **values)

    def zone():
        zones.append(1)
        return original_zone()

    monkeypatch.setattr(files, "text", counted)
    monkeypatch.setattr(hermes_time, "get_timezone", zone)
    monkeypatch.setattr(menu, "file_label", lambda item: pytest.fail("per-row batch recomputation"))
    monkeypatch.setattr(state.backend, "list_files", lambda **kwargs: pytest.fail("caption catalog fetch"))
    monkeypatch.setattr(state.backend, "read_file", lambda **kwargs: pytest.fail("caption byte fetch"))
    for render in (menu.render_files, menu.plain_files):
        calls.clear()
        zones.clear()
        render()
        assert len(zones) == 1 and calls.count("date_format") == 1
        assert calls.count("file_label") <= count * 2


@pytest.mark.asyncio
async def test_only_files_request_full_width_using_existing_telegram_layout(consumer):
    state, _, _ = consumer
    batch(state, ["brief.md", "notes.md"], at=1_788_509_527, serial=0, producer="You")
    menu, page = await menu_for(consumer, Platform.TELEGRAM)
    telegram_labels(page, 2)  # Load and verify the installed SDK, not its optional mock.
    import telegram
    from plugins.platforms.telegram.choice_picker import _keyboard

    def markup(value):
        return _keyboard(value.choices, "caption-test", 1,
                         telegram.InlineKeyboardButton, telegram.InlineKeyboardMarkup)

    assert all(choice["full_width"] for choice in page.choices[:2])
    assert not any(choice["full_width"] for choice in page.choices[2:])
    assert all(len(row) == 1 for row in markup(page).inline_keyboard)
    room = await menu.room_page(detail="Unchanged room menu")
    assert not any(choice["full_width"] for choice in room.choices)
    assert len(markup(room).inline_keyboard[0]) == 2
    item = {**menu.pages[0]["items"][0], "size": 11_000_000}
    confirmation = await menu.prepare_file(item)
    assert not any(choice["full_width"] for choice in confirmation.choices)
    assert len(markup(confirmation).inline_keyboard[0]) == 2
