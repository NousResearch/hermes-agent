import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.platforms.base import MessageEvent
from hermes_constants import get_hermes_home
from hermes_cli.commands import resolve_command


def _load_gateway_run():
    """Import the runner without bootstrapping the host installation during unit tests."""
    bootstrap = ModuleType("hermes_bootstrap")
    bootstrap.__dict__["_happy_eyeballs_create_connection"] = lambda *args, **kwargs: None
    with patch.dict(sys.modules, {"hermes_bootstrap": bootstrap}):
        import gateway.run as gateway_run
    return gateway_run


class RunningProcess:
    returncode = None


@pytest.mark.asyncio
async def test_speaker_start_resumes_current_media_without_interrupting(monkeypatch):
    gateway_run = _load_gateway_run()

    event = MessageEvent(text="/speaker start", source=SimpleNamespace())
    process = RunningProcess()
    create_process = AsyncMock(return_value=process)
    monkeypatch.setattr(gateway_run.asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(gateway_run.asyncio, "sleep", AsyncMock())

    runner = object.__new__(gateway_run.GatewayRunner)
    result = await runner._handle_speaker_command(event)

    assert result == "Speaker command `start` started. It will not interrupt current playback."
    speaker_root = get_hermes_home() / "sonos-brain"
    create_process.assert_awaited_once_with(
        sys.executable,
        str(speaker_root / "sonos_brain.py"),
        "start",
        cwd=str(speaker_root),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )


@pytest.mark.asyncio
async def test_speaker_schedule_is_a_deterministic_help_route(monkeypatch):
    gateway_run = _load_gateway_run()

    event = MessageEvent(text="/speaker schedule", source=SimpleNamespace())
    create_process = AsyncMock()
    monkeypatch.setattr(gateway_run.asyncio, "create_subprocess_exec", create_process)

    runner = object.__new__(gateway_run.GatewayRunner)
    result = await runner._handle_speaker_command(event)

    assert "FM schedule" in result
    assert "08:00 1h weekdays" in result
    create_process.assert_not_awaited()


def test_speaker_registry_documents_start_command():
    command = resolve_command("speaker")

    assert command is not None
    assert "start" in command.subcommands
