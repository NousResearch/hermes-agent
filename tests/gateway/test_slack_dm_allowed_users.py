"""Regression tests for Slack 1:1-DM-scoped user authorization."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from plugins.platforms.slack.adapter import _apply_yaml_config


_SLACK_AUTH_ENV = (
    "SLACK_ALLOWED_USERS",
    "SLACK_DM_ALLOWED_USERS",
    "SLACK_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
)


@pytest.fixture(autouse=True)
def _isolate_slack_auth_env(monkeypatch):
    for key in _SLACK_AUTH_ENV:
        monkeypatch.delenv(key, raising=False)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True)},
    )
    runner.adapters = {Platform.SLACK: SimpleNamespace(send=AsyncMock())}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    return runner


def _slack_source(user_id: str, *, chat_id: str = "D_OWNER", chat_type: str = "dm"):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_id,
    )


def test_owner_dm_allowed_by_dm_allowlist(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER,U_COLLAB")
    monkeypatch.setenv("SLACK_DM_ALLOWED_USERS", "U_OWNER")

    assert _make_runner()._is_user_authorized(_slack_source("U_OWNER")) is True


def test_channel_only_collaborator_dm_denied_by_nonempty_dm_allowlist(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER,U_COLLAB")
    monkeypatch.setenv("SLACK_DM_ALLOWED_USERS", "U_OWNER")

    assert _make_runner()._is_user_authorized(_slack_source("U_COLLAB")) is False


def test_collaborator_channel_mention_still_uses_slack_allowed_users(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER,U_COLLAB")
    monkeypatch.setenv("SLACK_DM_ALLOWED_USERS", "U_OWNER")

    source = _slack_source("U_COLLAB", chat_id="C_SHARED", chat_type="group")

    assert _make_runner()._is_user_authorized(source) is True


def test_empty_dm_allowlist_falls_back_to_slack_allowed_users(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER,U_COLLAB")
    monkeypatch.setenv("SLACK_DM_ALLOWED_USERS", "")

    assert _make_runner()._is_user_authorized(_slack_source("U_COLLAB")) is True


def test_mpim_keeps_platform_allowlist_even_with_dm_allowlist(monkeypatch):
    """Slack MPIMs are shared surfaces; only 1:1 IMs use DM auth."""
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_COLLAB")
    monkeypatch.setenv("SLACK_DM_ALLOWED_USERS", "U_OWNER")

    source = _slack_source("U_COLLAB", chat_id="G_MPIM", chat_type="dm")

    assert _make_runner()._is_user_authorized(source) is True


@pytest.mark.asyncio
async def test_active_thread_continuation_still_runs_authz(monkeypatch):
    """Busy/active Slack thread follow-ups must not bypass authz."""
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_OWNER")

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized = MagicMock(return_value=False)
    event = MessageEvent(
        text="thread follow-up",
        source=_slack_source("U_COLLAB", chat_id="C_SHARED", chat_type="group"),
        message_id="1700000000.000002",
    )
    event.source.thread_id = "1700000000.000001"

    handled = await runner._handle_active_session_busy_message(
        event,
        "agent:main:slack:group:C_SHARED:1700000000.000001",
    )

    assert handled is True
    runner._is_user_authorized.assert_called_once_with(event.source)


def test_slack_dm_allowlist_makes_unauthorized_dm_behavior_ignore(monkeypatch):
    monkeypatch.setenv("SLACK_DM_ALLOWED_USERS", "U_OWNER")

    assert _make_runner()._get_unauthorized_dm_behavior(Platform.SLACK) == "ignore"


def test_yaml_config_bridges_dm_allowed_users_to_env(monkeypatch):
    monkeypatch.delenv("SLACK_DM_ALLOWED_USERS", raising=False)

    _apply_yaml_config({}, {"dm_allowed_users": ["U_OWNER", "U_ASSISTANT"]})

    assert os.environ["SLACK_DM_ALLOWED_USERS"] == "U_OWNER,U_ASSISTANT"


def test_yaml_config_does_not_override_explicit_dm_allowed_users_env(monkeypatch):
    monkeypatch.setenv("SLACK_DM_ALLOWED_USERS", "U_ENV")

    _apply_yaml_config({}, {"dm_allowed_users": "U_YAML"})

    assert os.environ["SLACK_DM_ALLOWED_USERS"] == "U_ENV"
