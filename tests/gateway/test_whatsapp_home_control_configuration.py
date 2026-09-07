"""Post-ingress Group Chat policy with a shared-line WhatsApp configuration.

The source carries native one-to-one provenance, as the installed ingress does.
These tests exercise config loading and real authorization, not bridge intake.
"""

import json
from dataclasses import replace

import pytest
import yaml

from gateway.config import HomeChannel, Platform, load_gateway_config
from gateway.group_home_consent import text
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.slash_access import is_home_control_source, policy_for_source
from tests.gateway.test_hosted_room_messaging import _runner


PHONE = "15550001001"
OWNER = "90000000001001@lid"
CONTACT = "90000000001002@lid"
DELIVERY_GROUP = "120363000000001@g.us"


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("WHATSAPP_HOME_CHANNEL", DELIVERY_GROUP)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", f"{PHONE},{OWNER},{CONTACT}")
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    session = tmp_path / "platforms" / "whatsapp" / "session"
    session.mkdir(parents=True)
    (session / f"lid-mapping-{PHONE}.json").write_text(
        json.dumps(OWNER.split("@")[0]), encoding="utf-8"
    )
    (session / f"lid-mapping-{OWNER.split('@')[0]}_reverse.json").write_text(
        json.dumps(PHONE), encoding="utf-8"
    )

    def load(*, admin=False):
        whatsapp = {"dm_policy": "allowlist", "allow_from": [PHONE, OWNER, CONTACT]}
        if admin:
            whatsapp["allow_admin_from"] = [OWNER]
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({
                "platforms": {"whatsapp": {"enabled": True}},
                "whatsapp": whatsapp,
            }),
            encoding="utf-8",
        )
        runner = _runner(platform=Platform.WHATSAPP, extra={})
        runner.config = load_gateway_config()
        runner.adapters[Platform.WHATSAPP].config = runner.config.platforms[Platform.WHATSAPP]
        event = MessageEvent(
            text="/group",
            message_type=MessageType.COMMAND,
            message_id="synthetic-message",
            source=SessionSource(
                platform=Platform.WHATSAPP,
                chat_id=OWNER,
                user_id=OWNER,
                chat_type="dm",
                is_one_to_one=True,
            ),
        )
        return runner, event

    return load


@pytest.mark.asyncio
async def test_allowed_dm_and_unrestricted_slash_policy_do_not_enroll_owner(configured):
    runner, event = configured()
    assert runner._is_user_authorized_for_source(event.source)
    assert not policy_for_source(runner.config, event.source).enabled
    assert "Tier: unrestricted" in await runner._handle_whoami_command(event)
    assert not is_home_control_source(runner.config, event.source)
    assert not runner._can_control_group_chats(event)
    result = await runner._handle_rooms_command(event)
    assert result == text("setup")


def test_phone_alias_authorization_does_not_turn_shared_delivery_home_into_dm(configured, monkeypatch):
    runner, event = configured()
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", PHONE)
    runner.config.platforms[Platform.WHATSAPP].extra["allow_from"] = [PHONE]
    assert runner._is_user_authorized_for_source(event.source)
    assert not is_home_control_source(runner.config, event.source)
    assert not runner._can_control_group_chats(event)


def test_selecting_dm_home_alone_does_not_enroll_shared_line_contact(configured):
    runner, event = configured()
    runner.config.platforms[Platform.WHATSAPP].home_channel = HomeChannel(
        Platform.WHATSAPP, OWNER, "Synthetic home", user_id=OWNER
    )
    assert is_home_control_source(runner.config, event.source, require_owner_identity=True)
    assert not runner._can_control_group_chats(event)


@pytest.mark.asyncio
async def test_explicit_dm_admin_works_without_changing_delivery_or_talk_access(configured):
    runner, event = configured(admin=True)
    assert runner.config.get_home_channel(Platform.WHATSAPP).chat_id == DELIVERY_GROUP
    assert runner._is_user_authorized_for_source(event.source)
    assert runner._can_control_group_chats(event)
    assert "Tier: **admin**" in await runner._handle_whoami_command(event)
    other = replace(event, source=replace(event.source, chat_id=CONTACT, user_id=CONTACT))
    assert runner._is_user_authorized_for_source(other.source)
    assert not runner._can_control_group_chats(other)
    policy = policy_for_source(runner.config, other.source)
    assert policy.can_run(CONTACT, "whoami")
    assert not policy.can_run(CONTACT, "model")
    group = replace(event, source=replace(event.source, chat_id=DELIVERY_GROUP, chat_type="group", is_one_to_one=False))
    assert not runner._can_control_group_chats(group)


def test_dm_admin_uses_literal_transport_id_not_phone_alias(configured):
    runner, event = configured(admin=True)
    runner.config.platforms[Platform.WHATSAPP].extra["allow_admin_from"] = [PHONE]
    assert runner._is_user_authorized_for_source(event.source)
    assert not runner._can_control_group_chats(event)


@pytest.mark.parametrize("weakness", ["display-name", "raw-owner", "unknown-private", "machine", "edit"])
def test_admin_does_not_authorize_unverified_identity_or_provenance(configured, weakness):
    runner, event = configured(admin=True)
    if weakness in {"display-name", "raw-owner"}:
        event.source = replace(event.source, chat_id=CONTACT, user_id=CONTACT)
        if weakness == "display-name":
            event.source.user_name = OWNER
        else:
            event.raw_message = {"senderPn": PHONE, "senderId": OWNER, "fromOwner": True}
    elif weakness == "unknown-private":
        event.source.is_one_to_one = None
    elif weakness == "machine":
        event.source.is_bot = True
    else:
        event.source.message_is_edit = True
    assert not runner._can_control_group_chats(event)
