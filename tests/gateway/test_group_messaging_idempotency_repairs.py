"""Regression coverage for Home replacement and classic Stop replay."""

import hashlib

import pytest

from gateway import desktop_room_mailbox
from gateway.config import Platform
from gateway.home_channel_config import _replace_home
from gateway.hosted_room_messaging import messaging_event_id, send_to_room, stop_room
from hermes_cli import config as cli_config
from hermes_cli import managed_scope
from tests.gateway.test_hosted_room_messaging import _FakeService, _event, _runner


def _classic_room(*, projected_message: bool) -> dict:
    log = []
    if projected_message:
        log.append({
            "from": {"kind": "user"},
            "thread": "desktop-thread",
            "eventId": "desktop-message",
        })
    return {
        "room_id": "classic-room",
        "name": "Classic room",
        "members": [{"name": "default"}, {"name": "reviewer"}],
        "log": log,
        "desktop_authority_hash": hashlib.sha256(b"authority:test").hexdigest(),
        "desktop_available": False,
        "_room_mode": "desktop",
    }


def test_sethome_refuses_managed_legacy_delivery_before_saving(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "managed-old-chat")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "managed-old-topic")
    monkeypatch.setattr(
        managed_scope,
        "is_env_managed",
        lambda key: key.startswith("TELEGRAM_HOME_CHANNEL"),
    )
    cli_config.save_config({"platforms": {"telegram": {"enabled": True}}})
    runner = _runner(platform=Platform.TELEGRAM)
    event = _event(
        "/sethome",
        platform=Platform.TELEGRAM,
        chat_type="group",
        is_one_to_one=False,
    )
    event.source.chat_id = "selected-new-chat"
    event.source.thread_id = "selected-new-topic"

    result = _replace_home(runner, event)

    assert "could not be saved" in result.casefold()
    assert "home_channel" not in cli_config.load_config()["platforms"]["telegram"]
    assert runner.config.get_home_channel(Platform.TELEGRAM) is None


@pytest.mark.parametrize("projected_message", [False, True])
def test_same_stop_replay_after_superseding_send_keeps_original_effect(
    tmp_path, monkeypatch, projected_message
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "gateway.hosted_rooms.local_authority_gateway_id",
        lambda: "install:idempotency-test",
    )
    service = _FakeService(tmp_path / "state.db")
    room = _classic_room(projected_message=projected_message)
    send_to_room(
        service,
        room,
        _event("/group 1 send hello", message_id="send-delivery"),
        "hello",
    )
    event = _event("/group 1 stop", message_id="stop-delivery")

    first = stop_room(service, room, event)
    second = stop_room(service, room, event)

    assert second == first
    command = desktop_room_mailbox.command_state(
        desktop_room_mailbox.default_db_path(),
        f"stop:{messaging_event_id(event)}",
    )
    assert command is not None and command["action"] == "stop"
