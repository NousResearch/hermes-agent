"""Gateway access control for raw Kanban worker logs."""
from __future__ import annotations

from typing import cast
from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _runner(
    *,
    admins: list[str],
    user_commands: list[str],
    dm_admins: list[str] | None = None,
):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(
                enabled=True,
                token="synthetic-token-not-used",
                extra={
                    "allow_admin_from": dm_admins or [],
                    "group_allow_admin_from": admins,
                    "group_user_allowed_commands": user_commands,
                },
            )
        }
    )
    return runner


def _event(
    user_id: str | None,
    text: str = "/kanban log t_abcdef12",
    *,
    chat_type: str | None = "channel",
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.SLACK,
            user_id=user_id,
            chat_id="C123",
            chat_type=cast(str, chat_type),
        ),
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_non_admin_allowed_kanban_caller_cannot_read_worker_log(monkeypatch):
    runner = _runner(admins=["admin"], user_commands=["kanban"])
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(_event("member"))

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "/kanban --board default log t_abcdef12",
        "/kanban --board=default log t_abcdef12",
        "/kanban --bo default log t_abcdef12",
        "/kanban --bo=default log t_abcdef12",
        "/kanban -- log t_abcdef12",
        "/kanban --board default -- log t_abcdef12",
        "/kanban --board=default -- log t_abcdef12",
    ],
)
async def test_non_admin_cannot_bypass_worker_log_guard_with_board_flag_forms(
    monkeypatch, text
):
    runner = _runner(admins=["admin"], user_commands=["kanban"])
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(_event("member", text))

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_admin_can_read_worker_log(monkeypatch):
    runner = _runner(admins=["admin"], user_commands=[])
    run_slash = Mock(return_value="worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(_event("admin"))

    assert result == "worker output"
    run_slash.assert_called_once_with("log t_abcdef12")


@pytest.mark.asyncio
async def test_explicit_dm_admin_can_read_worker_log(monkeypatch):
    runner = _runner(admins=[], user_commands=[], dm_admins=["admin"])
    run_slash = Mock(return_value="worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(_event("admin", chat_type="dm"))

    assert result == "worker output"
    run_slash.assert_called_once_with("log t_abcdef12")


@pytest.mark.asyncio
async def test_worker_log_fails_closed_without_configured_admin(monkeypatch):
    runner = _runner(admins=[], user_commands=[])
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(_event("member"))

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_admins", [True, [True], {"admin": True}])
async def test_worker_log_rejects_non_identity_admin_config(
    monkeypatch, malformed_admins
):
    runner = _runner(admins=[], user_commands=[])
    runner.config.platforms[Platform.SLACK].extra[
        "group_allow_admin_from"
    ] = malformed_admins
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(_event("True"))

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_admins",
    [["admin", True], ["admin", {"nested": True}], ["admin", ["nested"]]],
)
async def test_worker_log_rejects_mixed_malformed_admin_config(
    monkeypatch, malformed_admins
):
    runner = _runner(admins=[], user_commands=[])
    runner.config.platforms[Platform.SLACK].extra[
        "group_allow_admin_from"
    ] = malformed_admins
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(_event("admin"))

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
async def test_worker_log_rejects_malformed_platform_and_config_types(monkeypatch):
    runner = _runner(admins=[], user_commands=[])
    runner.config.platforms = cast(
        dict[Platform, PlatformConfig],
        {"slack": {"group_allow_admin_from": ["admin"]}},
    )
    event = _event("admin")
    event.source.platform = cast(Platform, "slack")
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(event)

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
async def test_worker_log_fails_closed_without_authenticated_user_id(monkeypatch):
    runner = _runner(admins=["admin"], user_commands=["kanban"])
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(_event(None))

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
async def test_worker_log_fails_closed_when_authorization_resolution_is_malformed(
    monkeypatch,
):
    runner = _runner(admins=["admin"], user_commands=["kanban"])
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    def malformed_policy(*_args, **_kwargs):
        raise ValueError("synthetic malformed authorization")

    monkeypatch.setattr("gateway.slash_access.policy_for_source", malformed_policy)

    result = await runner._handle_kanban_command(_event("admin"))

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_type", [None, "", "unknown-scope"])
async def test_worker_log_fails_closed_for_unrecognized_chat_scope(
    monkeypatch, chat_type
):
    runner = _runner(admins=["admin"], user_commands=["kanban"])
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(
        _event("admin", chat_type=chat_type)
    )

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()


@pytest.mark.asyncio
async def test_group_admin_is_not_implicitly_admin_in_dm_scope(monkeypatch):
    runner = _runner(admins=["admin"], user_commands=["kanban"])
    run_slash = Mock(return_value="raw worker output")
    monkeypatch.setattr("hermes_cli.kanban.run_slash", run_slash)

    result = await runner._handle_kanban_command(_event("admin", chat_type="dm"))

    assert "admin-only" in result
    assert "raw worker output" not in result
    run_slash.assert_not_called()
