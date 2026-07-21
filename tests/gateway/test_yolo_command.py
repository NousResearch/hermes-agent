"""Tests for gateway /yolo session scoping."""

import os

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply, MessageEvent
from gateway.session import SessionSource
from tools.approval import disable_session_yolo, is_session_yolo_enabled


@pytest.fixture(autouse=True)
def _clean_yolo_state(monkeypatch):
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    import tools.approval as approval_module

    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")
    yield
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _make_event(chat_id: str, args: str = "") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )
    suffix = f" {args}" if args else ""
    return MessageEvent(text=f"/yolo{suffix}", source=source)


@pytest.mark.asyncio
async def test_yolo_command_toggles_only_current_session(monkeypatch):
    runner = _make_runner()

    event_a = _make_event("chat-a")
    session_a = runner._session_key_for_source(event_a.source)
    session_b = runner._session_key_for_source(_make_event("chat-b").source)

    result_on = await runner._handle_yolo_command(event_a)

    assert "ON" in result_on
    assert is_session_yolo_enabled(session_a) is True
    assert is_session_yolo_enabled(session_b) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None

    result_off = await runner._handle_yolo_command(event_a)

    assert "OFF" in result_off
    assert is_session_yolo_enabled(session_a) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None


@pytest.mark.asyncio
async def test_yolo_status_is_ephemeral_read_only_and_session_scoped():
    runner = _make_runner()
    event_a = _make_event("chat-a", "status")
    session_a = runner._session_key_for_source(event_a.source)
    session_b = runner._session_key_for_source(_make_event("chat-b").source)

    result = await runner._handle_yolo_command(event_a)

    assert "yolo mode" in str(result).lower()
    assert "OFF" in str(result)
    assert isinstance(result, EphemeralReply)
    assert is_session_yolo_enabled(session_a) is False
    assert is_session_yolo_enabled(session_b) is False

    from tools.approval import enable_session_yolo

    enable_session_yolo(session_a)
    result_on = await runner._handle_yolo_command(event_a)
    assert "ON" in str(result_on)
    assert is_session_yolo_enabled(session_a) is True
    assert is_session_yolo_enabled(session_b) is False


@pytest.mark.asyncio
async def test_yolo_status_uses_the_source_profiles_approval_mode(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from tools import approval

    default_home = tmp_path / "default"
    target_home = tmp_path / "profiles" / "target"
    default_home.mkdir(parents=True)
    target_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(
        approval,
        "_get_approval_mode",
        lambda: "off" if get_hermes_home() == target_home else "manual",
    )

    runner = _make_runner()
    runner._resolve_profile_home_for_source = lambda _source: target_home
    event = _make_event("chat-a", "status")
    event.source.profile = "target"

    result = await runner._handle_yolo_command(event)

    assert "ON" in str(result)
    assert get_hermes_home() == default_home


@pytest.mark.asyncio
async def test_yolo_toggle_response_reports_the_session_source_not_effective_status(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from tools import approval

    default_home = tmp_path / "default"
    target_home = tmp_path / "profiles" / "target"
    default_home.mkdir(parents=True)
    target_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(
        approval,
        "_get_approval_mode",
        lambda: "off" if get_hermes_home() == target_home else "manual",
    )

    runner = _make_runner()
    runner._resolve_profile_home_for_source = lambda _source: target_home
    event = _make_event("chat-a")
    event.source.profile = "target"
    session_key = runner._session_key_for_source(event.source)
    approval.enable_session_yolo(session_key)

    result = await runner._handle_yolo_command(event)

    assert is_session_yolo_enabled(session_key) is False
    assert "OFF" in str(result)
    assert "for this session" in str(result)
    assert get_hermes_home() == default_home


@pytest.mark.asyncio
async def test_yolo_invalid_argument_uses_localized_usage(monkeypatch):
    import gateway.slash_commands as slash_commands

    monkeypatch.setattr(
        slash_commands,
        "t",
        lambda key, **_kwargs: "localized usage" if key == "gateway.yolo.usage" else key,
    )

    result = await _make_runner()._handle_yolo_command(_make_event("chat-a", "invalid"))

    assert isinstance(result, EphemeralReply)
    assert str(result) == "localized usage"


def test_yolo_status_is_discoverable_in_registry_help_and_completion():
    from hermes_cli.commands import COMMANDS, gateway_help_lines

    assert "[status]" in COMMANDS["/yolo"]
    assert any("/yolo" in line and "[status]" in line for line in gateway_help_lines())
