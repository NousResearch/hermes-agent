"""Native Group Chat navigation retains the canonical picker and useful actions."""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import i18n
from gateway import hosted_room_messaging as rooms
from gateway import hosted_room_messaging_files as files
from gateway.choice_picker import ChoicePage
from gateway.config import Platform
from tests.gateway.test_hosted_room_file_access import file_state, publish
from tests.gateway.test_hosted_room_messaging_files import consumer, event
from tests.gateway.test_telegram_choice_pages import adapter as telegram_adapter
from tests.gateway.test_telegram_choice_picker import _query
from tests.gateway.test_discord_choice_pages import picker as discord_picker
from tests.gateway.test_matrix_choice_pages import adapter as matrix_adapter


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


async def menu_for(consumer):
    state, runner, _ = consumer
    menu = files.FilesMenu(runner, event("/group 1"), state.backend, "/group")
    await menu.bind("1")
    return menu


def token(menu, page, kind):
    return next(c["value"] for c in page.choices if menu.actions[c["value"]][0] == kind)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/group", "!group"])
async def test_detail_bots_back_uses_canonical_status_labels_and_layout(consumer, command):
    menu = await menu_for(consumer)
    menu.command = command
    detail = await menu.room_page()
    bots = await menu.choose("chat", token(menu, detail, "bots"))
    expected_bots = rooms.room_bot_picker_choices(menu.backend, await menu.fresh_room())
    bot_rows = [c for c in bots.choices if menu.actions[c["value"]][0] == "bot"]
    assert [c["label"] for c in bot_rows] == [c["label"] for c in expected_bots]
    assert all(c["full_width"] for c in bot_rows)
    assert [menu.actions[c["value"]][1] for c in bot_rows] == [
        c["value"] for c in expected_bots
    ]
    # Roster order must not retarget a native participant selection.
    state = consumer[0]
    with sqlite3.connect(state.db) as conn:
        conn.execute("UPDATE hosted_rooms SET members_json=? WHERE room_id='room-1'",
                     (json.dumps(list(reversed(state.room["members"]))),))
    bot_detail = await menu.choose("chat", bot_rows[0]["value"])
    assert bot_detail.title == rooms.format_room_bot_detail(
        menu.backend, await menu.fresh_room(), expected_bots[0]["value"], room_command=command,
    )
    assert f"{command} 1 send @reviewer" in bot_detail.title
    bots = await menu.choose("chat", token(menu, bot_detail, "bots"))
    activity = await menu.choose("chat", token(menu, bots, "room"))
    bots = await menu.choose("chat", token(menu, activity, "bots"))
    catalog = await menu.choose("chat", token(menu, bots, "groups"))
    expected = rooms.room_picker_choices(menu.backend, await menu.fresh_room())
    assert [c["label"] for c in catalog.choices] == [c["label"] for c in expected]
    assert all(c["full_width"] for c in catalog.choices)
    assert f"{command} list" in catalog.title
    reopened = await menu.choose("chat", catalog.choices[0]["value"])
    assert isinstance(reopened, ChoicePage)
    assert menu.reference == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    "wrong_chat", "stale_action", "expired", "source", "adapter", "permission",
    "authority", "removed_room", "removed_member", "member_during_detail",
    "member_during_navigation", "member_during_list", "member_during_recheck",
    "authority_during_recheck",
])
async def test_nested_bot_navigation_rejects_changed_scope(consumer, monkeypatch, change):
    menu = await menu_for(consumer)
    state, runner, adapter = consumer
    detail = await menu.room_page()
    selected = token(menu, detail, "bots")
    if change != "member_during_list":
        bots = await menu.choose("chat", selected)
        selected = token(menu, bots, "bot")

    def update_room(statement, parameters=()):
        with sqlite3.connect(state.db) as conn:
            conn.execute(statement, parameters)

    def remove_member():
        update_room("UPDATE hosted_rooms SET members_json=? WHERE room_id='room-1'",
                    (json.dumps(state.room["members"][1:]),))

    def change_authority():
        update_room("UPDATE hosted_rooms SET authority_epoch=authority_epoch+1")

    def during_read(name, mutate=remove_member, *, on_call=1):
        original = getattr(rooms, name)
        calls = 0

        def changed(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == on_call:
                mutate()
            return result

        monkeypatch.setattr(rooms, name, changed)

    original_navigation = menu.room_content_actions

    async def changed_navigation(current):
        actions = await original_navigation(current)
        remove_member()
        return actions

    changes = {
        "stale_action": lambda: menu.page(detail.title, [(files.text("back"), ("room", None))]),
        "expired": lambda: setattr(menu, "deadline", 0),
        "source": lambda: setattr(menu.event.source, "thread_id", "different-topic"),
        "adapter": lambda: setattr(runner, "adapter", type(adapter)()),
        "permission": lambda: runner.config.platforms[Platform.SIGNAL].extra.update(
            allow_admin_from=["other"],
        ),
        "authority": change_authority,
        "removed_room": lambda: update_room("DELETE FROM hosted_rooms WHERE room_id='room-1'"),
        "removed_member": remove_member,
        "member_during_detail": lambda: during_read("format_room_bot_detail"),
        "member_during_list": lambda: during_read("format_room_bot_list"),
        "member_during_recheck": lambda: during_read("room_bot_picker_choices", on_call=2),
        "authority_during_recheck": lambda: during_read(
            "room_bot_picker_choices", change_authority, on_call=2,
        ),
        "member_during_navigation": lambda: monkeypatch.setattr(
            menu, "room_content_actions", changed_navigation,
        ),
    }
    if change in changes:
        changes[change]()
    chat = "other-chat" if change == "wrong_chat" else "chat"
    result = await menu.choose(chat, selected)
    rendered = result.title if isinstance(result, ChoicePage) else result
    assert "reviewer" not in rendered and "send @" not in rendered
    assert rendered in {files.text("denied"), files.text("expired"), files.text("error")}
    assert not adapter.documents and not adapter.notices


@pytest.mark.asyncio
@pytest.mark.parametrize("view", ["bots", "bot"])
async def test_nested_bot_read_failure_uses_safe_text_without_execution_controls(consumer, monkeypatch, view):
    menu = await menu_for(consumer)
    page = await menu.room_page()
    if view == "bot":
        page = await menu.choose("chat", token(menu, page, "bots"))

    def unavailable(*args, **kwargs):
        raise RuntimeError("private backend diagnostic")

    monkeypatch.setattr(rooms, "format_room_bot_detail" if view == "bot" else "format_room_bot_list",
                        unavailable)
    result = await menu.choose("chat", token(menu, page, view))
    assert result == files.text("error")
    assert not consumer[2].documents and not consumer[2].notices


@pytest.mark.asyncio
async def test_current_view_is_not_an_action_and_navigation_uses_verbs(consumer):
    menu = await menu_for(consumer)
    detail = await menu.room_page()
    assert "room" not in [menu.actions[c["value"]][0] for c in detail.choices]
    assert files.text("bots") == "View Bots"
    bots = await menu.choose("chat", token(menu, detail, "bots"))
    assert "bots" not in [menu.actions[c["value"]][0] for c in bots.choices]
    assert files.text("activity") == "View recent activity"
    assert files.text("activity") in [c["label"] for c in bots.choices]


@pytest.mark.asyncio
async def test_empty_files_and_reply_actions_are_absent_then_refresh(consumer):
    menu = await menu_for(consumer)
    detail = await menu.room_page()
    assert not {"files", "reply"} & {menu.actions[c["value"]][0] for c in detail.choices}
    publish(consumer[0], "welcome.md", b"hello")
    detail = await menu.room_page()
    assert "files" in {menu.actions[c["value"]][0] for c in detail.choices}
    assert files.text("files") == "View files"


@pytest.mark.asyncio
async def test_empty_catalog_finishes_without_a_self_link(consumer, monkeypatch):
    menu = await menu_for(consumer)
    detail = await menu.room_page()
    monkeypatch.setattr(rooms, "list_messaging_rooms", lambda *a, **k: [])
    result = await menu.choose("chat", token(menu, detail, "groups"))
    assert isinstance(result, str)
    assert not menu.actions


@pytest.mark.asyncio
async def test_telegram_real_callback_roundtrip_preserves_full_width_picker(consumer, telegram_adapter):
    menu = await menu_for(consumer)
    page = await menu.room_page()
    adapter = telegram_adapter
    await adapter.send_choice_picker(
        "chat-1", page.title, page.choices, "session",
        lambda chat, value: menu.choose("chat", value),
        {"choice_pages": True, "requester_user_id": "user-1"},
    )
    markup = adapter._send_message_with_thread_fallback.await_args.kwargs["reply_markup"]
    expected_bots = rooms.room_bot_picker_choices(menu.backend, await menu.fresh_room())
    for caption in (files.text("bots"), expected_bots[0]["label"],
                    files.text("bots"), files.text("back_groups")):
        button = next(b for row in markup.inline_keyboard for b in row if b.text == caption)
        if caption == expected_bots[0]["label"]:
            assert all(len(row) == 1 for row in markup.inline_keyboard)
        query = _query()
        await adapter._handle_choice_picker_callback(query, button.callback_data, "chat-1")
        markup = query.edit_message_text.await_args.kwargs["reply_markup"]
    assert all(len(row) == 1 for row in markup.inline_keyboard)
    expected = rooms.room_picker_choices(menu.backend, await menu.fresh_room())
    assert [row[0].text for row in markup.inline_keyboard] == [c["label"] for c in expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.DISCORD, Platform.MATRIX, Platform.SLACK])
async def test_shared_menu_contract_has_identical_action_labels(consumer, platform):
    state, runner, adapter = consumer
    command = event("/group 1", platform=platform)
    # Transport configuration is irrelevant to rendering; retain real owner authorization.
    runner.config.platforms[platform] = next(iter(runner.config.platforms.values()))
    menu = files.FilesMenu(runner, command, state.backend, "/group")
    await menu.bind("1")
    page = await menu.room_page()
    expected = ["View Bots", "Back to Group Chats"]
    if platform == Platform.TELEGRAM:
        expected = ["🤖 View Bots", "‹ Group Chats"]
    assert [c["label"] for c in page.choices] == expected


@pytest.mark.asyncio
async def test_unavailable_metadata_does_not_break_navigation(consumer, monkeypatch):
    menu = await menu_for(consumer)
    def unavailable(**kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(menu.backend, "list_files", unavailable)
    page = await menu.room_page()
    assert "files" not in {menu.actions[c["value"]][0] for c in page.choices}
    assert "bots" in {menu.actions[c["value"]][0] for c in page.choices}


@pytest.mark.asyncio
@pytest.mark.parametrize("language", i18n.SUPPORTED_LANGUAGES)
async def test_localized_telegram_buttons_fit_without_changing_action_values(consumer, monkeypatch, language):
    from plugins.platforms.telegram.choice_picker import _keyboard

    menu = await menu_for(consumer)
    menu.event.source.platform = Platform.TELEGRAM
    monkeypatch.setenv("HERMES_LANGUAGE", language)
    actions = [(files.text(key), (kind, None)) for key, kind in (
        ("files", "files"), ("full_reply", "reply"), ("activity", "room"),
        ("bots", "bots"), ("back_groups", "groups"),
    )]
    page = menu.page(files.text("groups"), actions)
    markup = _keyboard(page.choices, "token", 1,
                       lambda text, **kw: SimpleNamespace(text=text, **kw), lambda rows: rows)
    assert [b.text for row in markup for b in row] == [c["label"] for c in page.choices]
    assert all(len(b.text) <= 64 and len(b.callback_data.encode()) <= 64 for row in markup for b in row)
    assert list(menu.actions.values()) == [action for _, action in actions]


@pytest.mark.asyncio
async def test_discord_renderer_keeps_action_verbs_without_telegram_decoration(consumer):
    menu = await menu_for(consumer)
    page = await menu.room_page()
    view, _ = await discord_picker(AsyncMock(), choices=page.choices)
    try:
        assert [o.label for o in view.children[0].options] == [c["label"] for c in page.choices]
    finally:
        view.stop()


@pytest.mark.asyncio
async def test_matrix_renderer_keeps_action_verbs_without_telegram_decoration(consumer, matrix_adapter):
    menu = await menu_for(consumer)
    page = await menu.room_page()
    result = await matrix_adapter.send_choice_picker(
        "!room:example.org", page.title, page.choices, "session", AsyncMock(),
        {"choice_pages": True, "requester_user_id": "@owner:example.org"},
    )
    assert result.success
    content = matrix_adapter.send.await_args.args[1]
    assert all(c["label"] in content for c in page.choices)
    assert "🤖 View Bots" not in content and "‹ Group Chats" not in content


@pytest.mark.asyncio
async def test_plain_files_does_not_inherit_telegram_icons(consumer):
    publish(consumer[0])
    menu = await menu_for(consumer)
    await menu.files_page()
    plain = menu.plain_files()
    assert "📎" not in plain and "🕘" not in plain and "🤖" not in plain
    assert "/group 1 file " in plain and "/group 1 reply" not in plain


@pytest.mark.asyncio
async def test_removed_room_cannot_be_retargeted_after_back_to_catalog(consumer, monkeypatch):
    menu = await menu_for(consumer)
    detail = await menu.room_page()
    catalog = await menu.choose("chat", token(menu, detail, "groups"))
    original = (await menu.fresh_room())[0]
    replacement = {**original, "room_id": "replacement", "name": original["name"]}
    monkeypatch.setattr(rooms, "list_messaging_rooms", lambda *a, **k: [replacement])
    result = await menu.choose("chat", catalog.choices[0]["value"])
    assert menu.room is None
    assert not isinstance(result, ChoicePage) or result.title != detail.title


@pytest.mark.asyncio
async def test_no_native_menu_support_preserves_text_callback(consumer):
    from plugins.platforms.slack.adapter import SlackAdapter

    _, runner, _ = consumer
    runner.adapter = object.__new__(SlackAdapter)
    fallback = AsyncMock(return_value="plain room detail")
    callback, reusable = files.room_picker_callback(
        runner, event("/group", platform=Platform.SLACK), consumer[0].backend, "/group", fallback,
    )
    assert not reusable and callback is fallback
    assert await callback("chat", "room-token") == "plain room detail"


@pytest.mark.asyncio
async def test_full_reply_action_appears_only_after_a_bot_reply(consumer):
    from gateway import hosted_rooms

    state, _, _ = consumer
    menu = await menu_for(consumer)
    before = await menu.room_page()
    assert "reply" not in {menu.actions[c["value"]][0] for c in before.choices}
    hosted_rooms.append_event(
        state.db, room_id="room-1", event_id="bot-reply", kind="message.member",
        actor={"kind": "member", "id": "ops"}, payload={"text": "Finished work"},
        authority_gateway_id=state.authority, authority_epoch=1,
    )
    after = await menu.room_page()
    assert "Get full reply" in [c["label"] for c in after.choices]


@pytest.mark.asyncio
async def test_unsupported_document_delivery_omits_content_controls(consumer, monkeypatch):
    publish(consumer[0])
    menu = await menu_for(consumer)
    monkeypatch.setattr(menu.adapter, "send_document", None)
    page = await menu.room_page()
    assert not {"files", "reply"} & {menu.actions[c["value"]][0] for c in page.choices}
