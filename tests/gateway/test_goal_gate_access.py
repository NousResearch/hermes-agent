"""Gateway authorization tests for shell-backed goal quality gates."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def _source(user_id: str = "user-1", *, chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id=user_id,
        chat_id=f"{chat_type}-1",
        chat_type=chat_type,
    )


def _runner(*, admins=(), group_admins=()):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                extra={
                    "allow_admin_from": list(admins),
                    "group_allow_admin_from": list(group_admins),
                },
            )
        }
    )
    manager = MagicMock()
    manager.add_gate.return_value = SimpleNamespace(
        command="touch /tmp/host-in-the-shell",
        max_retries=3,
        timeout_seconds=300,
    )
    runner._get_goal_manager_for_event = AsyncMock(
        return_value=(manager, SimpleNamespace(session_id="session-1"))
    )
    return runner, manager


def _event(user_id: str = "user-1", *, chat_type: str = "dm"):
    event = MagicMock()
    event.source = _source(user_id, chat_type=chat_type)
    event.get_command_args.return_value = (
        "gate add touch /tmp/host-in-the-shell"
    )
    return event


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("admins", "user_id"),
    [
        ((), "user-1"),
        (("admin-1",), "user-1"),
    ],
)
async def test_gateway_gate_add_requires_explicit_admin(admins, user_id):
    """Allowed chat users must not turn /goal into an unrestricted host shell."""
    from gateway.run import GatewayRunner

    runner, manager = _runner(admins=admins)

    result = await GatewayRunner._handle_goal_command(runner, _event(user_id))

    assert "explicitly configured gateway admin" in result
    manager.add_gate.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_explicit_admin_can_add_goal_gate():
    """The fix preserves the documented quality-gate capability for operators."""
    from gateway.run import GatewayRunner

    runner, manager = _runner(admins=("admin-1",))

    result = await GatewayRunner._handle_goal_command(runner, _event("admin-1"))

    assert "Gate added" in result
    manager.add_gate.assert_called_once_with(
        "touch /tmp/host-in-the-shell"
    )


@pytest.mark.asyncio
async def test_gateway_group_gate_add_uses_group_admin_scope():
    """DM admin status must not silently grant host-shell access in groups."""
    from gateway.run import GatewayRunner

    runner, manager = _runner(
        admins=("dm-admin",),
        group_admins=("group-admin",),
    )

    denied = await GatewayRunner._handle_goal_command(
        runner,
        _event("dm-admin", chat_type="group"),
    )
    allowed = await GatewayRunner._handle_goal_command(
        runner,
        _event("group-admin", chat_type="group"),
    )

    assert "explicitly configured gateway admin" in denied
    assert "Gate added" in allowed
    manager.add_gate.assert_called_once_with(
        "touch /tmp/host-in-the-shell"
    )


@pytest.mark.asyncio
async def test_gateway_non_admin_can_still_list_goal_gates():
    """Non-admins retain read-only visibility into the active goal's gates."""
    from gateway.run import GatewayRunner

    runner, manager = _runner(admins=("admin-1",))
    manager.render_gates.return_value = "- 1. $ scripts/run_tests.sh"
    event = _event("user-1")
    event.get_command_args.return_value = "gate list"

    result = await GatewayRunner._handle_goal_command(runner, event)

    assert result == "- 1. $ scripts/run_tests.sh"
    manager.render_gates.assert_called_once_with()
