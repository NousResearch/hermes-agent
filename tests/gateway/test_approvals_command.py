"""Gateway contract and live dispatch for /approvals."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
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
async def test_gateway_approval_request_waits_for_existing_operator_confirmation_surface():
    runner = _runner()
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(
                extra={"allow_admin_from": ["user-1"], "user_allowed_commands": ["approvals"]}
            )
        }
    )
    runner._session_key_for_source = lambda _source: "session-1"
    runner._adapter_for_source = lambda _source: None
    runner._thread_metadata_for_source = lambda *_args: {}
    runner._reply_anchor_for_event = lambda _event: None
    runner._slash_confirm_counter = iter([1])

    pending = SimpleNamespace(request_id="request-1", session_id="session-1")
    proof = object()
    broker = MagicMock()
    broker.request.return_value = pending
    broker.operator_confirm.return_value = proof
    result = SimpleNamespace(message="Approval mode: off (persistent profile setting).")

    with (
        patch("hermes_cli.policy_mutation.PolicyMutationBroker", return_value=broker),
        patch("hermes_cli.approval_mode.run_approval_mode_command", return_value=result) as run,
    ):
        prompt = await runner._handle_approvals_command(_event("/approvals off"))

    assert "confirm" in prompt.lower()
    broker.operator_confirm.assert_not_called()
    run.assert_not_called()

    from tools import slash_confirm
    resolved = await slash_confirm.resolve("session-1", "1", "once")

    assert resolved == result.message
    broker.operator_confirm.assert_called_once_with("request-1")
    run.assert_called_once_with("off", proof=proof, session_id="session-1")


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


