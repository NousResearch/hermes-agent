"""Regression guard for Slack bot/workflow-sender authorization bypass.

Mirrors tests/gateway/test_feishu_bot_auth_bypass.py for Platform.SLACK.

Slack Workflow Builder posts (and other app/bot messages) arrive as
``subtype=bot_message`` with ``user=None``, so the SessionSource carries
``is_bot=True`` and ``user_id=None``. Without the #4466 bot bypass running
*before* the no-user-id guard, these senders are rejected at
``_is_user_authorized`` even when the operator enabled ``SLACK_ALLOW_BOTS`` --
the bug that makes @mentioning the bot from a Slack workflow do nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_slack_env(monkeypatch):
    for var in (
        "SLACK_ALLOW_BOTS",
        "SLACK_ALLOWED_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_slack_bot_source():
    # Workflow Builder / app posts: subtype=bot_message, user=None.
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C0123",
        chat_type="group",
        user_id=None,
        user_name="",
        is_bot=True,
    )


def _make_slack_human_source(user_id="U_human"):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C0123",
        chat_type="group",
        user_id=user_id,
        user_name="Human",
        is_bot=False,
    )


def _make_slack_thread_source(user_id="U_human"):
    source = _make_slack_human_source(user_id)
    source.chat_type = "thread"
    source.thread_id = "1000.0"
    return source


def test_slack_bot_authorized_when_allow_bots_all(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "all")
    assert runner._is_user_authorized(_make_slack_bot_source()) is True


def test_slack_bot_authorized_when_allow_bots_mentions(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "mentions")
    assert runner._is_user_authorized(_make_slack_bot_source()) is True


def test_slack_bot_denied_when_allow_bots_unset(monkeypatch):
    # No SLACK_ALLOW_BOTS + no user_id => denied (no bypass, hits guard).
    runner = _make_bare_runner()
    assert runner._is_user_authorized(_make_slack_bot_source()) is False


def test_slack_bot_denied_when_allow_bots_none(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "none")
    assert runner._is_user_authorized(_make_slack_bot_source()) is False


def test_slack_human_unaffected_by_bot_bypass(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_ALL_USERS", "true")
    assert runner._is_user_authorized(_make_slack_human_source()) is True


def test_slack_group_allow_from_authorizes_configured_channel_only():
    runner = _make_bare_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(
                enabled=True,
                extra={"group_allow_from": ["C0123"]},
            )
        }
    )

    assert runner._is_user_authorized(_make_slack_human_source("U_teammate")) is True

    other_channel = _make_slack_human_source("U_teammate")
    other_channel.chat_id = "C9999"
    assert runner._is_user_authorized(other_channel) is False


def test_slack_group_allow_from_does_not_open_dms():
    runner = _make_bare_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(
                enabled=True,
                extra={"group_allow_from": ["C0123"]},
            )
        }
    )

    dm = _make_slack_human_source("U_teammate")
    dm.chat_id = "D0123"
    dm.chat_type = "dm"
    assert runner._is_user_authorized(dm) is False


def test_slack_group_allow_from_authorizes_thread_context_in_configured_channel():
    runner = _make_bare_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(
                enabled=True,
                extra={"group_allow_from": ["C0123"]},
            )
        }
    )

    assert runner._is_user_authorized(_make_slack_thread_source("U_teammate")) is True


def test_slack_group_allow_from_supports_team_scoped_channel_entries():
    runner = _make_bare_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(
                enabled=True,
                extra={"group_allow_from": ["T_ALLOWED:C0123"]},
            )
        }
    )

    allowed = _make_slack_human_source("U_teammate")
    allowed.scope_id = "T_ALLOWED"
    assert runner._is_user_authorized(allowed) is True

    other_workspace = _make_slack_human_source("U_teammate")
    other_workspace.scope_id = "T_OTHER"
    assert runner._is_user_authorized(other_workspace) is False


def test_slack_group_allow_from_does_not_authorize_interactive_approvals():
    runner = _make_bare_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(
                enabled=True,
                extra={"group_allow_from": ["C0123"]},
            )
        }
    )

    interactive = _make_slack_human_source("U_teammate")
    interactive.chat_type = "interactive"
    assert runner._is_user_authorized(interactive) is False
