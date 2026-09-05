"""An explicitly selected home owner is not the gateway-wide talk allowlist."""

import pytest

from gateway.config import HomeChannel, Platform
from tests.gateway.test_hosted_room_messaging import (
    _FakeService,
    _event,
    _runner,
    _seed_rooms,
)


def selected_home(monkeypatch, platform=Platform.TELEGRAM, *, accepted=True):
    monkeypatch.setenv(
        f"{platform.value.upper()}_ALLOWED_USERS", "user-1,user-2,user-3"
    )
    runner = _runner(
        platform=platform,
        extra={
            "allow_from": ["user-1", "user-2", "user-3"],
            "group_allow_admin_from": ["user-1"],
        },
    )
    runner._is_user_authorized_for_source = lambda source: (
        source.user_id in {"user-1", "user-2", "user-3"}
    )
    event = _event(
        "/group list", platform=platform, chat_type="group", is_one_to_one=False
    )
    runner.config.platforms[platform].home_channel = HomeChannel(
        platform=platform,
        chat_id=event.source.chat_id,
        name="Owner home",
        user_id="user-1",
    )
    if accepted:
        from gateway.group_home_identity import acknowledgement

        home = runner.config.platforms[platform].home_channel
        home.group_audience_ack = acknowledgement(home)
    return runner, event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform", [Platform.TELEGRAM, Platform.SIGNAL, Platform.WHATSAPP]
)
async def test_selected_home_owner_can_read_and_send_with_other_allowed_contacts(
    tmp_path, monkeypatch, platform
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    runner, event = selected_home(monkeypatch, platform)

    listing = await runner._handle_rooms_command(event)
    assert "Release room" in listing
    event.text = "/group 1 send selected owner"
    result = await runner._handle_room_command(event)
    assert result.startswith("Queued in Release room")
    assert service.sent[0]["payload"]["text"] == "selected owner"


@pytest.mark.parametrize(
    "change",
    [
        "other-user",
        "other-chat",
        "no-owner",
        "wrong-scope",
        "bot",
        "relay",
        "unauthorized",
        "owner-changed",
    ],
)
def test_selected_home_binding_cannot_be_borrowed(monkeypatch, change):
    runner, event = selected_home(monkeypatch)
    home = runner.config.platforms[Platform.TELEGRAM].home_channel
    if change == "other-user":
        event.source.user_id = "user-2"
    elif change == "other-chat":
        event.source.chat_id = "another-chat"
    elif change == "no-owner":
        home.user_id = None
    elif change == "wrong-scope":
        home.scope_id = "workspace-a"
        event.source.scope_id = "workspace-b"
    elif change == "bot":
        event.source.is_bot = True
    elif change == "relay":
        event.source.delivered_via_upstream_relay = True
    elif change == "unauthorized":
        runner._is_user_authorized_for_source = lambda source: False
    else:

        def changed_owner(source):
            home.user_id = "user-2"
            return True

        runner._is_user_authorized_for_source = changed_owner
    assert runner._can_control_group_chats(event) is False


def test_explicit_home_owner_does_not_promote_other_users_when_talk_is_public(
    monkeypatch,
):
    runner, event = selected_home(monkeypatch)
    monkeypatch.setenv("TELEGRAM_ALLOW_ALL_USERS", "true")
    runner._is_user_authorized_for_source = lambda source: True
    assert runner._can_control_group_chats(event) is True
    event.source.user_id = "unlisted-user"
    assert runner._can_control_group_chats(event) is False


def test_legacy_unbound_home_dm_with_multiple_users_stays_restricted(monkeypatch):
    runner, event = selected_home(monkeypatch)
    runner.config.platforms[Platform.TELEGRAM].home_channel.user_id = None
    event.source.chat_type = "dm"
    event.source.is_one_to_one = True
    assert runner._can_control_group_chats(event) is False
