"""Gateway contract and live dispatch for /approvals."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _event(text: str = "/approvals") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="user-1",
            chat_id="chat-1",
            chat_type="dm",
        ),
    )


def _runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace(platforms={})
    runner.hooks = MagicMock(loaded_hooks=[])
    runner.hooks.emit = AsyncMock(return_value=[])
    runner._running_agents = {}
    runner._get_or_create_gateway_honcho = lambda _key: (None, None)
    runner._is_user_authorized = lambda _source: True
    runner.session_store = SimpleNamespace(get_or_create_session=lambda _source: None)
    return runner


@pytest.mark.asyncio
async def test_gateway_rejects_non_admin_persistent_approval_change():
    runner = _runner()
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(
                extra={
                    "allow_admin_from": ["admin-1"],
                    "user_allowed_commands": ["approvals"],
                }
            )
        }
    )

    with patch("hermes_cli.approval_mode.run_approval_mode_command") as run:
        output = await runner._handle_approvals_command(_event("/approvals off"))

    assert "admin" in output.lower()
    run.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_rejects_approval_change_without_configured_policy():
    """An unconfigured slash policy (gating disabled) must NOT authorize a
    persistent approval-mode change: with ``enabled=False``,
    ``policy.is_admin`` returns True for everyone, which would let any caller
    persist ``approvals.mode: off`` and disable approval checks (#81108)."""
    runner = _runner()  # platforms={} → policy_for_source returns disabled policy

    with patch("hermes_cli.approval_mode.run_approval_mode_command") as run:
        output = await runner._handle_approvals_command(_event("/approvals off"))

    assert "admin" in output.lower()
    run.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_allows_query_without_configured_policy():
    """Reading the current approval mode stays open even with no policy."""
    runner = _runner()

    with patch("hermes_cli.approval_mode.run_approval_mode_command") as run:
        run.return_value = SimpleNamespace(message="Approval mode: manual")
        output = await runner._handle_approvals_command(_event("/approvals"))

    assert output == "Approval mode: manual"
    run.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_gateway_allows_admin_change_with_configured_policy():
    runner = _runner()
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(
                extra={
                    "allow_admin_from": ["admin-1"],
                    "user_allowed_commands": ["approvals"],
                }
            )
        }
    )

    with patch("hermes_cli.approval_mode.run_approval_mode_command") as run:
        run.return_value = SimpleNamespace(message="Approval mode: off")
        event = _event("/approvals off")
        event.source.user_id = "admin-1"
        output = await runner._handle_approvals_command(event)

    assert output == "Approval mode: off"
    run.assert_called_once_with("off")


