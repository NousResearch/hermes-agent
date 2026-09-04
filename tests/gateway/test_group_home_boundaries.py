"""Exact source identity, reload, and settings-race boundaries for home consent."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from threading import Event

import pytest

from gateway.config import HomeChannel, Platform, _apply_env_overrides
from gateway.group_home_identity import acknowledgement, trusted_person
from gateway.slash_access import is_home_control_source
from hermes_cli.config import load_config, save_config
from tests.gateway.test_group_home_consent import home, command


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "anonymous",
        "unknown",
        "irc",
        "channel",
        "sender_chat",
        "bot",
        "relay",
        "rejected_profile",
    ],
)
def test_weak_native_identity_never_becomes_a_control_principal(home, change):
    event = home.event
    if change in {"missing", "anonymous", "unknown"}:
        event.source.user_id = None if change == "missing" else change
    elif change == "irc":
        event.source.platform = Platform("irc")
    elif change == "channel":
        event.source.chat_type = "channel"
    elif change == "sender_chat":
        event.raw_message = SimpleNamespace(sender_chat=SimpleNamespace(id=-100))
    elif change == "bot":
        event.source.is_bot = True
    elif change == "relay":
        event.source.delivered_via_upstream_relay = True
        event.metadata = {
            "relay_author_classified": False,
            "relay_edit_classified": True,
        }
        event.source.is_one_to_one = True
    else:
        event.source.profile_route_rejected = True
    assert not trusted_person(event)
    assert not home.runner._can_control_group_chats(event, require_audience=False)


@pytest.mark.parametrize(
    "platform", [Platform.TELEGRAM, Platform.DISCORD, Platform.SLACK, Platform.MATRIX]
)
def test_home_scope_and_real_topic_match_exactly(home, platform):
    source = home.event.source
    source.platform, source.scope_id, source.thread_id = platform, "scope", "topic"
    config = home.runner.config
    config.platforms[platform] = config.platforms[Platform.TELEGRAM]
    config.platforms[platform].home_channel = HomeChannel(
        platform, source.chat_id, "Home", "topic", source.user_id, "scope"
    )
    assert is_home_control_source(config, source, require_owner_identity=True)
    source.thread_id = "other"
    assert not is_home_control_source(config, source)
    source.thread_id, source.scope_id = "topic", "other"
    assert not is_home_control_source(config, source)
    source.scope_id = None
    assert not is_home_control_source(config, source)
    source.scope_id = "scope"
    config.platforms[platform].home_channel.platform = Platform.SIGNAL
    assert not is_home_control_source(config, source)


def test_slack_synthetic_thread_is_not_the_home_location(home):
    source = home.event.source
    source.platform, source.thread_id, source.message_id = (
        Platform.SLACK,
        "new-message",
        "new-message",
    )
    config = home.runner.config
    config.platforms[Platform.SLACK] = config.platforms[Platform.TELEGRAM]
    config.platforms[Platform.SLACK].home_channel = HomeChannel(
        Platform.SLACK, source.chat_id, "Home", user_id=source.user_id
    )
    assert is_home_control_source(config, source)
    source.thread_id = "real-parent"
    assert not is_home_control_source(config, source)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["same", "other"])
async def test_ack_survives_only_same_target_env_reload(home, monkeypatch, target):
    await command(home, "/group")
    await command(home, "/group confirm")
    selected = home.runner.config.get_home_channel(Platform.TELEGRAM)
    monkeypatch.setenv(
        "TELEGRAM_HOME_CHANNEL", selected.chat_id if target == "same" else "other"
    )
    _apply_env_overrides(home.runner.config)
    loaded = home.runner.config.get_home_channel(Platform.TELEGRAM)
    if target == "same":
        assert loaded.group_audience_ack == acknowledgement(loaded)
    else:
        assert loaded.group_audience_ack is None and loaded.user_id is None


@pytest.mark.asyncio
async def test_disk_binding_replacement_and_unrelated_settings_are_not_overwritten(
    home,
):
    await command(home, "/group")
    config = load_config()
    config["platforms"]["telegram"]["home_channel"]["chat_id"] = "new-home"
    config["unrelated-home-ux-control"] = "keep"
    save_config(config)
    assert "expired" in await command(home, "/group confirm")
    saved = load_config()
    assert saved["unrelated-home-ux-control"] == "keep"
    assert saved["platforms"]["telegram"]["home_channel"]["chat_id"] == "new-home"


@pytest.mark.asyncio
async def test_replacement_serializes_after_confirmation_and_clears_ack(
    home, monkeypatch
):
    await command(home, "/group")
    entered, release = Event(), Event()
    original = save_config

    def blocked(config, **kwargs):
        if (
            config
            .get("platforms", {})
            .get("telegram", {})
            .get("home_channel", {})
            .get("group_audience_ack")
        ):
            entered.set()
            assert release.wait(5)
        return original(config, **kwargs)

    monkeypatch.setattr("hermes_cli.config.save_config", blocked)
    confirm = asyncio.create_task(command(home, "/group confirm"))
    assert await asyncio.to_thread(entered.wait, 5)
    replacement = asyncio.create_task(
        home.runner._handle_set_home_command(replace(home.event, text="/sethome"))
    )
    release.set()
    await asyncio.gather(confirm, replacement)
    current = home.runner.config.get_home_channel(Platform.TELEGRAM)
    assert current.group_audience_ack is None and current.selection_id
    assert load_config()["platforms"]["telegram"]["home_channel"] == current.to_dict()


@pytest.mark.asyncio
async def test_ack_revoked_during_room_read_withholds_the_private_response(
    home, monkeypatch
):
    await command(home, "/group")
    await command(home, "/group confirm")
    from gateway import hosted_room_messaging as rooms

    original = rooms.format_room_detail

    def revoke(*args, **kwargs):
        value = original(*args, **kwargs)
        home.runner.config.get_home_channel(Platform.TELEGRAM).group_audience_ack = None
        return value

    monkeypatch.setattr(rooms, "format_room_detail", revoke)
    result = await command(home, "/group 1")
    assert "Release room" not in result and "expired" in result


def test_single_talk_contact_is_not_shared_home_admin_enrollment(home, monkeypatch):
    config = home.runner.config.platforms[Platform.TELEGRAM]
    config.extra = {"allow_from": ["user-1"]}
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "user-1")
    config.home_channel.group_audience_ack = acknowledgement(config.home_channel)
    assert not home.runner._can_control_group_chats(home.event)


@pytest.mark.asyncio
async def test_binding_changed_during_resolution_prevents_command_submission(
    home, monkeypatch
):
    from gateway import hosted_room_messaging as rooms

    await command(home, "/group")
    await command(home, "/group confirm")
    original = rooms.list_messaging_rooms

    def revoke(*args, **kwargs):
        result = original(*args, **kwargs)
        home.runner.config.get_home_channel(Platform.TELEGRAM).group_audience_ack = None
        return result

    monkeypatch.setattr(rooms, "list_messaging_rooms", revoke)
    result = await home.runner._handle_room_command(
        replace(home.event, text="/group 1 send never-send")
    )
    assert not home.service.sent and "Queued" not in result


@pytest.mark.asyncio
async def test_ack_and_revision_survive_supported_cold_dotenv_reload(home):
    from hermes_cli.config import get_config_path
    from tests.gateway.test_home_binding_reload import _COLD_LOAD

    await home.runner._handle_set_home_command(replace(home.event, text="/sethome"))
    await command(home, "/group")
    await command(home, "/group confirm")
    folder = get_config_path().parent
    child = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", _COLD_LOAD],
        cwd=folder,
        env={
            "PATH": os.environ["PATH"],
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "HOME": str(folder.parent),
            "HERMES_HOME": str(folder),
            "HERMES_TEST_ISOLATION": str(folder),
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    result = json.loads(
        next(
            line.removeprefix("HOME_BINDING=")
            for line in child.stdout.splitlines()
            if line.startswith("HOME_BINDING=")
        )
    )
    assert result == home.runner.config.get_home_channel(Platform.TELEGRAM).to_dict()
    assert result["selection_id"] and result["group_audience_ack"]
