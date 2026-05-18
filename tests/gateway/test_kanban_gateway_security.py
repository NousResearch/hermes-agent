from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: MagicMock(send=AsyncMock())}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
    )
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '/kanban create "do it" --assignee admin',
        '/kanban create "do it" --assignee=admin',
        '/kanban --board ops create "do it" --assignee admin',
    ],
)
async def test_gateway_kanban_denies_create_with_assignee(monkeypatch, text):
    """Gateway users must not create work that auto-spawns arbitrary profiles."""
    from hermes_cli import kanban as kanban_cli

    runner = _make_runner()
    run_slash = MagicMock(return_value="Created t_abcd (ready, assignee=admin)")
    monkeypatch.setattr(kanban_cli, "run_slash", run_slash)

    result = await runner._handle_kanban_command(_make_event(text))

    assert "creating assigned tasks from the gateway is disabled" in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_kanban_denies_dispatch(monkeypatch):
    """Gateway users must not manually trigger the profile-spawning dispatcher."""
    from hermes_cli import kanban as kanban_cli

    runner = _make_runner()
    run_slash = MagicMock(return_value="Spawned: 1")
    monkeypatch.setattr(kanban_cli, "run_slash", run_slash)

    result = await runner._handle_kanban_command(
        _make_event("/kanban dispatch --max 1")
    )

    assert "only available from the local CLI" in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "/kanban assign t_1234 admin",
        "/kanban reassign t_1234 admin --reclaim",
        "/kanban unblock t_1234",
    ],
)
async def test_gateway_kanban_denies_profile_launching_actions(monkeypatch, text):
    """Gateway users must not assign or relaunch worker profile work."""
    from hermes_cli import kanban as kanban_cli

    runner = _make_runner()
    run_slash = MagicMock(return_value="ok")
    monkeypatch.setattr(kanban_cli, "run_slash", run_slash)

    result = await runner._handle_kanban_command(_make_event(text))

    assert "gateway" in result or "local CLI" in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_kanban_allows_unassigned_create(monkeypatch):
    """Unassigned gateway-created tasks remain triage items until a local assign."""
    from hermes_cli import kanban as kanban_cli

    runner = _make_runner()
    run_slash = MagicMock(return_value="Created t_abcd  (ready, assignee=-)")
    monkeypatch.setattr(kanban_cli, "run_slash", run_slash)

    result = await runner._handle_kanban_command(
        _make_event('/kanban create "triage me"')
    )

    assert result.startswith("Created t_abcd")
    run_slash.assert_called_once_with('create "triage me"')
