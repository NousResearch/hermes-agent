"""Audience opt-in uses real home config persistence and existing room helpers."""

import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform, load_gateway_config
from gateway.group_home_identity import acknowledgement
from gateway import group_home_consent as consent
from hermes_cli.config import load_config, save_config
from tests.gateway.test_group_chat_selected_home_owner import selected_home
from tests.gateway.test_hosted_room_messaging import _FakeService, _seed_rooms


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    runner, event = selected_home(monkeypatch, accepted=False)
    save_config({
        "platforms": {"telegram": runner.config.platforms[Platform.TELEGRAM].to_dict()}
    })
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    return SimpleNamespace(runner=runner, event=event, service=service)


async def command(home, text):
    return await home.runner._handle_rooms_command(replace(home.event, text=text))


@pytest.mark.asyncio
async def test_first_warning_has_no_private_fetch_then_confirms_once(home, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(
            "gateway.hosted_room_messaging.current_room_backend",
            lambda: pytest.fail("private fetch before consent"),
        )
        warning = await command(home, "/group 1")
        assert "Everyone who can read this chat" in warning
        assert "/group confirm" in warning and "/approve" not in warning
        assert not home.runner._can_control_group_chats(home.event)
        assert home.runner._can_control_group_chats(home.event, require_audience=False)
    result = await command(home, "/group confirm")
    assert "Release room" in result
    saved = load_config()["platforms"]["telegram"]["home_channel"]
    assert saved["group_audience_ack"] == acknowledgement(
        home.runner.config.get_home_channel(Platform.TELEGRAM)
    )
    assert "Release room" in await command(home, "/group list")
    home.runner.config = load_gateway_config()
    assert home.runner._can_control_group_chats(home.event)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["user", "topic", "profile", "expired", "replacement"]
)
async def test_prompt_cannot_cross_actor_location_profile_or_replacement(home, change):
    await command(home, "/group")
    if change == "user":
        home.event.source.user_id = "user-2"
    elif change == "topic":
        home.event.source.thread_id = "another"
    elif change == "profile":
        home.event.source.profile = "other"
    elif change == "expired":
        next(iter(home.runner._group_home_confirmations.values())).deadline = (
            time.monotonic() - 1
        )
    else:
        await home.runner._handle_set_home_command(replace(home.event, text="/sethome"))
    result = await command(home, "/group confirm")
    assert "Release room" not in str(result)
    assert not load_config()["platforms"]["telegram"]["home_channel"].get(
        "group_audience_ack"
    )


@pytest.mark.asyncio
async def test_failed_save_never_sets_live_ack(home, monkeypatch):
    await command(home, "/group")
    monkeypatch.setattr("hermes_cli.config.save_config", lambda *args, **kwargs: None)
    assert "could not be saved" in await command(home, "/group confirm")
    assert not home.runner._can_control_group_chats(home.event)


@pytest.mark.asyncio
async def test_stop_help_and_cancel_do_not_require_consent(home, monkeypatch):
    assert "Stop requested" in await command(home, "/group 1 stop")
    assert home.service.stopped
    assert home.runner._can_approve_group_chats(home.event, require_audience=False)
    denied = await command(home, "/group 1 deny abcdef12")
    assert denied == consent.text("control_unavailable") and "Everyone" not in denied
    with monkeypatch.context() as patch:
        patch.setattr(
            "gateway.hosted_room_messaging.current_room_backend",
            lambda: pytest.fail("generic command fetched rooms"),
        )
        help_text = await command(home, "/group help")
        assert "Choose a Group Chat" in help_text
        assert "Shared homes require" not in help_text
        assert "/sethome" not in help_text
        assert "Confirmation cancelled" in await command(home, "/group cancel")


@pytest.mark.asyncio
async def test_sethome_replacement_clears_ack_and_changes_revision(home):
    await command(home, "/group")
    await command(home, "/group confirm")
    await home.runner._handle_set_home_command(replace(home.event, text="/sethome"))
    first = home.runner.config.get_home_channel(Platform.TELEGRAM)
    assert first.group_audience_ack is None and first.selection_id
    await command(home, "/group")
    await command(home, "/group confirm")
    await home.runner._handle_set_home_command(replace(home.event, text="/sethome"))
    second = home.runner.config.get_home_channel(Platform.TELEGRAM)
    assert (
        second.selection_id != first.selection_id and second.group_audience_ack is None
    )
