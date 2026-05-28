"""Regression guard for Feishu bot-sender authorization bypass.

Mirrors tests/gateway/test_discord_bot_auth_bypass.py for Platform.FEISHU.
Without the bypass in gateway/run.py, Feishu bot senders admitted by the
adapter would be rejected at _is_user_authorized with "Unauthorized user"
— same class of bug as Discord #4466.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_feishu_env(monkeypatch):
    for var in (
        "FEISHU_ALLOW_BOTS",
        "FEISHU_ALLOWED_USERS",
        "FEISHU_ALLOW_ALL_USERS",
        "FEISHU_GROUP_ALLOWED_CHATS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_feishu_bot_source(open_id: str = "ou_peer"):
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_1",
        chat_type="group",
        user_id=open_id,
        user_name="PeerBot",
        is_bot=True,
    )


def _make_feishu_human_source(open_id: str = "ou_human", chat_id: str = "oc_1"):
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id=chat_id,
        chat_type="group",
        user_id=open_id,
        user_name="Human",
        is_bot=False,
    )


def test_feishu_bot_authorized_when_allow_bots_mentions(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "mentions")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_bot_source("ou_peer")) is True


def test_feishu_bot_authorized_when_allow_bots_all(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_bot_source()) is True


def test_feishu_bot_NOT_authorized_when_allow_bots_none(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "none")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_bot_source("ou_peer")) is False


def test_feishu_bot_NOT_authorized_when_allow_bots_unset(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_bot_source("ou_peer")) is False


def test_feishu_human_still_checked_against_allowlist_when_bot_policy_set(monkeypatch):
    """FEISHU_ALLOW_BOTS=all must NOT open the gate for humans."""
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_human_source("ou_stranger")) is False
    assert runner._is_user_authorized(_make_feishu_human_source("ou_human")) is True


def test_feishu_group_allowed_chats_authorizes_group_without_opening_dm(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_GROUP_ALLOWED_CHATS", "oc_1")

    group_source = _make_feishu_human_source("ou_stranger")
    assert runner._is_user_authorized(group_source) is True

    dm_source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_dm",
        chat_type="dm",
        user_id="ou_stranger",
        user_name="Human",
        is_bot=False,
    )
    assert runner._is_user_authorized(dm_source) is False


def test_feishu_group_allowed_chats_rejects_other_groups_even_for_allowed_user(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_GROUP_ALLOWED_CHATS", "oc_allowed")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    allowed_group_source = _make_feishu_human_source("ou_human", chat_id="oc_allowed")
    other_group_source = _make_feishu_human_source("ou_human", chat_id="oc_random")

    assert runner._is_user_authorized(allowed_group_source) is True
    assert runner._is_user_authorized(other_group_source) is False


def test_feishu_group_allowed_chats_makes_unauthorized_dms_silent(monkeypatch):
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_GROUP_ALLOWED_CHATS", "oc_allowed")

    assert runner._get_unauthorized_dm_behavior(Platform.FEISHU) == "ignore"


def test_feishu_bot_bypass_does_not_leak_to_other_platforms(monkeypatch):
    """FEISHU_ALLOW_BOTS=all must not authorize Telegram/Discord bot sources."""
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")

    telegram_bot = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="channel",
        user_id="999",
        is_bot=True,
    )
    assert runner._is_user_authorized(telegram_bot) is False
