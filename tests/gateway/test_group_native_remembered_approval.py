"""The rich group menu uses the same confirmed, revocable permission path as slash commands."""

import pytest

from gateway.hosted_room_messaging_files import FilesMenu
from gateway.choice_picker import ChoicePage
from tests.gateway.test_group_approval_permissions import owner


def token(menu, page, kind, argument_prefix=""):
    return next(choice["value"] for choice in page.choices if (
        menu.actions[choice["value"]][0] == kind
        and str(menu.actions[choice["value"]][1]).startswith(argument_prefix)))


@pytest.mark.asyncio
async def test_group_detail_confirmation_and_bot_settings_revoke_use_one_permission(owner):
    menu = FilesMenu(owner.runner, owner.event, owner.service, "/group")
    await menu.bind("1")
    page = await menu.room_page()
    assert not any(action[0] == "permissions" for action in menu.actions.values())
    page = await menu.choose(owner.event.source.chat_id, token(menu, page, "approvals"))
    page = await menu.choose(owner.event.source.chat_id, token(menu, page, "permission_action", "remember:"))
    assert "without asking" in page.title and owner.calls == []
    page = await menu.choose(owner.event.source.chat_id, token(menu, page, "permission_action", "confirm:"))
    assert "remembered for writer" in page.title
    assert [call["choice"] for call in owner.calls] == ["once"]
    owner.service.room_status["pending_actions"] = []
    page = await menu.choose(owner.event.source.chat_id, token(menu, page, "room"))
    page = await menu.choose(owner.event.source.chat_id, token(menu, page, "bots"))
    page = await menu.choose(owner.event.source.chat_id, token(menu, page, "permissions"))
    page = await menu.choose(owner.event.source.chat_id, token(menu, page, "permission_action", "rule:"))
    old_forget = token(menu, page, "permission_action", "forget:")
    page = await menu.choose(owner.event.source.chat_id, old_forget)
    assert "Future requests will ask again" in page.title
    assert not isinstance(await menu.choose(owner.event.source.chat_id, old_forget), ChoicePage)
    assert len(owner.calls) == 1
    page = await menu.choose(owner.event.source.chat_id, token(menu, page, "room"))
    assert isinstance(page, ChoicePage) and "group-a" in page.title


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["owner", "home", "expired", "wrong-chat"])
async def test_native_remember_confirmation_rechecks_owner_destination_and_menu_lifetime(owner, change):
    menu = FilesMenu(owner.runner, owner.event, owner.service, "/group")
    await menu.bind("1")
    page = await menu.permission_page(approvals=True)
    page = await menu.choose(owner.event.source.chat_id, token(menu, page, "permission_action", "remember:"))
    confirm = token(menu, page, "permission_action", "confirm:")
    if change == "owner":
        owner.event.source.user_id = "user-2"
    elif change == "home":
        owner.runner.config.get_home_channel(owner.event.source.platform).selection_id = "replaced"
    elif change == "expired":
        menu.deadline = 0
    result = await menu.choose("different-chat" if change == "wrong-chat" else owner.event.source.chat_id, confirm)
    assert owner.calls == []
    assert "remembered for writer" not in str(result)
