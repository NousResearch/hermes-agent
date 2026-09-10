"""Current source talk permission is separate from unchanged Group Chat admin policy."""

import asyncio
from threading import Event

import pytest

from gateway import hosted_room_messaging as rooms
from gateway.config import Platform
from gateway.group_chat_policy import group_policy_for_source
from tests.gateway.test_group_home_consent import home, command
from tests.gateway.group_chat_picker_fixtures import picker_home, choose_first


@pytest.fixture
def live(picker_home, monkeypatch):
    picker_home.runner.__dict__.pop("_is_user_authorized_for_source")
    picker_home.adapter.config.extra["allow_admin_from"] = ["user-1"]
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-1")
    assert picker_home.runner._is_user_authorized_for_source(picker_home.event.source)
    return picker_home


def revoke(state, monkeypatch):
    policy = group_policy_for_source(state.runner, state.event.source)
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-2")
    assert not state.runner._is_user_authorized_for_source(state.event.source)
    assert group_policy_for_source(state.runner, state.event.source) == policy
    assert policy.is_admin("user-1")


async def prepare(state, scope):
    if scope == "shared":
        await command(state, "/group")
        await command(state, "/group confirm")
    else:
        state.event.source.chat_type = "dm"
        state.event.source.is_one_to_one = True
        if scope == "non-home":
            state.runner.config.get_home_channel(
                Platform.TELEGRAM
            ).chat_id = "elsewhere"


def pause(monkeypatch, name):
    entered, release = Event(), Event()
    original = getattr(rooms, name)

    def held(*args, **kwargs):
        value = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return value

    monkeypatch.setattr(rooms, name, held)
    return entered, release


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["private", "non-home", "shared"])
@pytest.mark.parametrize("action", ["status", "send"])
async def test_real_talk_revocation_withholds_read_and_submission(
    live, monkeypatch, scope, action
):
    await prepare(live, scope)
    entered, release = pause(
        monkeypatch,
        "format_room_detail" if action == "status" else "list_messaging_rooms",
    )
    pending = asyncio.create_task(
        command(live, "/group 1" if action == "status" else "/group 1 send forbidden")
    )
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        revoke(live, monkeypatch)
    finally:
        release.set()
    result = await pending
    assert not live.runner._can_control_group_chats(live.event)
    assert "Release room" not in str(result) and not live.service.sent


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["private", "shared"])
async def test_core_callback_uses_fresh_talk_auth_after_captured_read(
    live, monkeypatch, scope
):
    await prepare(live, scope)
    # The picker closes over the imported formatter, so install the barrier first.
    entered, release = pause(monkeypatch, "format_room_detail")
    assert await command(live, "/group") is None
    pending = asyncio.create_task(choose_first(live))
    assert await asyncio.to_thread(entered.wait, 5)
    try:
        revoke(live, monkeypatch)
    finally:
        release.set()
    result = await pending
    assert "Release room" not in str(result)


@pytest.mark.asyncio
async def test_still_authorized_private_source_retains_good_path(live):
    await prepare(live, "private")
    assert "Release room" in await command(live, "/group 1")
    assert "Queued" in await command(live, "/group 1 send allowed")
    assert live.service.sent
