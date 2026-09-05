"""A delivery-home selection is not an owner enrollment mechanism."""

from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.config import HomeChannel, Platform, PlatformConfig
from tests.gateway.test_hosted_room_messaging import (
    _FakeService,
    _event,
    _runner,
    _seed_rooms,
)


@pytest.mark.asyncio
async def test_talk_contact_cannot_self_select_into_group_history(
    tmp_path, monkeypatch
):
    from tests.gateway.test_slash_access_dispatch import (
        _make_event,
        _make_runner,
        _make_source,
    )

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-1,user-2,user-3")
    db, _, _ = _seed_rooms(tmp_path)
    hosted_rooms.append_event(
        db,
        room_id="release-room",
        event_id="private-event",
        kind="message.user",
        actor={"kind": "user", "id": "owner"},
        authority_gateway_id="install:test-gateway",
        authority_epoch=1,
        payload={"text": "PrivateRoomSentinel", "thread_id": "private-thread"},
    )
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: _FakeService(db),
    )
    runner = _make_runner(
        platform=Platform.TELEGRAM,
        platform_extra={"allow_from": ["user-1", "user-2", "user-3"]},
    )
    del runner.__dict__["_is_user_authorized"]
    runner.adapters[Platform.TELEGRAM] = SimpleNamespace(typed_command_prefix="/")
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="owner-home",
        name="Owner home",
        user_id="user-1",
    )
    source = _make_source(
        platform=Platform.TELEGRAM,
        user_id="user-2",
        chat_type="group",
        chat_id="contact-home",
    )
    source.is_one_to_one = False
    event = _make_event("/sethome", source)
    assert runner._is_user_authorized_for_source(source)
    assert not runner._can_control_group_chats(event)
    await runner._handle_message(event)
    assert runner.config.get_home_channel(Platform.TELEGRAM).user_id == "user-2"
    event.text = "/group list"
    assert "Release room" not in await runner._handle_rooms_command(event)
    event.text = "/group 1"
    assert "PrivateRoomSentinel" not in await runner._handle_rooms_command(event)


@pytest.mark.parametrize("chat_type", ["group", "dm"])
@pytest.mark.parametrize("change", ["owner", "clear-owner", "home", "chat", "scope"])
def test_lost_binding_cannot_fall_back_to_single_contact(
    monkeypatch, chat_type, change
):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-1")
    runner = _runner(platform=Platform.TELEGRAM, extra={"allow_from": ["user-1"]})
    event = _event(
        "/group list",
        platform=Platform.TELEGRAM,
        chat_type=chat_type,
        is_one_to_one=chat_type == "dm",
    )
    home = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id=event.source.chat_id,
        name="Owner home",
        user_id="user-1",
    )
    runner.config.platforms[Platform.TELEGRAM].home_channel = home

    def authorized(source):
        if change == "owner":
            home.user_id = "user-2"
        elif change == "clear-owner":
            home.user_id = None
        elif change == "home":
            runner.config.platforms[Platform.TELEGRAM].home_channel = None
        elif change == "chat":
            home.chat_id = "other-chat"
        else:
            home.scope_id = "other-scope"
        return True

    runner._is_user_authorized_for_source = authorized
    assert not runner._can_control_group_chats(event)


@pytest.mark.parametrize("admin", [False, True])
def test_explicit_home_uses_enrolled_admin_not_unrestricted_slash_policy(
    monkeypatch, admin
):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-1,user-2,user-3")
    extra = {"allow_from": ["user-1", "user-2", "user-3"]}
    if admin:
        extra["group_allow_admin_from"] = ["user-1"]
    runner = _runner(platform=Platform.TELEGRAM, extra=extra)
    event = _event(
        "/group list",
        platform=Platform.TELEGRAM,
        chat_type="group",
        is_one_to_one=False,
    )
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id=event.source.chat_id,
        name="Owner home",
        user_id="user-1",
    )
    assert runner._is_user_authorized_for_source(event.source)
    from gateway.group_home_identity import acknowledgement

    home = runner.config.platforms[Platform.TELEGRAM].home_channel
    home.group_audience_ack = acknowledgement(home)
    assert runner._can_control_group_chats(event) is admin


def test_unbound_single_contact_legacy_dm_still_works(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-1")
    runner = _runner(platform=Platform.TELEGRAM, extra={"allow_from": ["user-1"]})
    event = _event("/group list", platform=Platform.TELEGRAM)
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id=event.source.chat_id, name="Home"
    )
    assert runner._can_control_group_chats(event)


@pytest.mark.parametrize("transport_owner", ["user-1", "user-2"])
def test_home_uses_receiving_adapters_admin_policy(monkeypatch, transport_owner):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-1,user-2")
    runner = _runner(
        platform=Platform.TELEGRAM,
        extra={
            "allow_from": ["user-1", "user-2"],
            "group_allow_admin_from": [
                "user-2" if transport_owner == "user-1" else "user-1"
            ],
        },
    )
    event = _event(
        "/group list",
        platform=Platform.TELEGRAM,
        chat_type="group",
        is_one_to_one=False,
    )
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id=event.source.chat_id,
        name="Selected",
        user_id="user-1",
    )
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.config = PlatformConfig(
        enabled=True,
        extra={
            "allow_from": ["user-1", "user-2"],
            "group_allow_admin_from": [transport_owner],
        },
    )
    event.source._transport_adapter_ref = lambda: adapter
    from gateway.group_home_identity import acknowledgement

    home = runner.config.platforms[Platform.TELEGRAM].home_channel
    home.group_audience_ack = acknowledgement(home)
    assert runner._can_control_group_chats(event) is (transport_owner == "user-1")
