"""Regression coverage for Home replacement and classic Stop replay."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

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


def test_failed_atomic_home_pair_preserves_previous_canonical_live_and_legacy(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "old-chat")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "old-topic")
    cli_config.save_config({"platforms": {"telegram": {"enabled": False}}})
    runner = _runner(platform=Platform.TELEGRAM)
    event = _event("/sethome", platform=Platform.TELEGRAM)
    event.source.chat_id, event.source.thread_id = "old-chat", "old-topic"
    assert "This is now" in _replace_home(runner, event)
    previous = runner.config.get_home_channel(Platform.TELEGRAM).to_dict()
    env_path = cli_config.get_env_path()
    original_env = env_path.read_bytes()
    event.source.chat_id, event.source.thread_id = "new-chat", "new-topic"

    def reject_write(path, lines, **kwargs):
        assert "TELEGRAM_HOME_CHANNEL=new-chat\n" in lines
        assert "TELEGRAM_HOME_CHANNEL_THREAD_ID=new-topic\n" in lines
        raise OSError("disk write unavailable")

    monkeypatch.setattr(cli_config, "_write_env_lines", reject_write)
    assert "could not be saved" in _replace_home(runner, event)
    assert env_path.read_bytes() == original_env
    assert cli_config.load_config()["platforms"]["telegram"]["home_channel"] == previous
    assert runner.config.get_home_channel(Platform.TELEGRAM).to_dict() == previous
    loaded = load_gateway_config().get_home_channel(Platform.TELEGRAM)
    assert (loaded.chat_id, loaded.thread_id) == ("old-chat", "old-topic")


def test_concurrent_identical_stop_deliveries_replay_one_frozen_target(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.hosted_rooms.local_authority_gateway_id", lambda: "install:test")
    service = _FakeService(tmp_path / "state.db")
    room = _classic_room(projected_message=False)
    send_to_room(service, room, _event("/group 1 send hello", message_id="send"), "hello")
    event = _event("/group 1 stop", message_id="simultaneous-stop")
    barrier = Barrier(8)

    def deliver(_index):
        barrier.wait(timeout=10)
        return stop_room(service, room, event)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(deliver, range(8)))
    assert len(set(results)) == 1
    stored = desktop_room_mailbox.command_state(
        desktop_room_mailbox.default_db_path(), f"stop:{messaging_event_id(event)}",
    )
    assert stored["payload"]["target_command_id"]


def test_stop_replay_does_not_accept_changed_room_or_authority(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    service = _FakeService(tmp_path / "state.db")
    room = _classic_room(projected_message=True)
    event = _event("/group 1 stop", message_id="original-stop")
    stop_room(service, room, event)
    for change in ({"room_id": "other-room"}, {"desktop_authority_hash": "c" * 64}):
        with pytest.raises(ValueError):
            stop_room(service, {**room, **change}, event)
