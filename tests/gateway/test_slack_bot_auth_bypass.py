"""Regression guard for Slack bot-origin authorization."""

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_slack_env(monkeypatch):
    for var in (
        "SLACK_ALLOW_BOTS",
        "SLACK_ALLOWED_USERS",
        "SLACK_ALLOW_ALL_USERS",
        "SLACK_GROUP_ALLOWED_USERS",
        "SLACK_GROUP_ALLOWED_CHATS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_slack_bot_source(bot_id: str = "U_OTHER_BOT"):
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C1",
        chat_type="group",
        user_id=bot_id,
        user_name="grainfs-dev",
        is_bot=True,
    )


def test_slack_bot_authorized_when_allow_bots_mentions(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_HUMAN")

    assert runner._is_user_authorized(_make_slack_bot_source()) is True


def test_slack_bot_not_authorized_when_allow_bots_unset(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U_HUMAN")

    assert runner._is_user_authorized(_make_slack_bot_source()) is False
