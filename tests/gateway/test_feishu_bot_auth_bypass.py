"""Regression guard for Feishu bot-sender authorization bypass.

Mirrors tests/gateway/test_discord_bot_auth_bypass.py for Platform.FEISHU.
Without the bypass in gateway/run.py, Feishu bot senders admitted by the
adapter would be rejected at _is_user_authorized with "Unauthorized user"
— same class of bug as Discord #4466.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_feishu_env(monkeypatch):
    for var in (
        "FEISHU_ALLOW_BOTS",
        "FEISHU_ALLOWED_USERS",
        "FEISHU_ALLOW_ALL_USERS",
        "FEISHU_GROUP_POLICY",
        "FEISHU_ALLOWED_GROUP_USERS",
        "TELEGRAM_ALLOW_BOTS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_group_rule_runner(group_rules, **extra):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    platform_config = PlatformConfig(
        enabled=True,
        extra={"group_rules": group_rules, **extra},
    )
    adapter = FeishuAdapter(platform_config)
    runner = _make_bare_runner()
    runner.adapters = {Platform.FEISHU: adapter}
    runner._profile_adapters = {}
    runner.config = GatewayConfig(
        platforms={Platform.FEISHU: platform_config},
    )
    return runner, adapter


def _make_feishu_bot_source(open_id: str = "ou_peer"):
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_1",
        chat_type="group",
        user_id=open_id,
        user_name="PeerBot",
        is_bot=True,
    )


def _make_feishu_human_source(open_id: str = "ou_human"):
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_1",
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


def test_explicit_open_group_rule_authorizes_only_that_chat(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_dm_owner")
    runner, adapter = _make_group_rule_runner({
        "oc_1": {"policy": "open"},
    })

    assert adapter.enforces_own_access_policy is True
    assert runner._is_user_authorized(_make_feishu_human_source("ou_guest")) is True

    other_group = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_2",
        chat_type="group",
        user_id="ou_guest",
        user_name="Guest",
    )
    dm = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_dm",
        chat_type="dm",
        user_id="ou_guest",
        user_name="Guest",
    )
    assert runner._is_user_authorized(other_group) is False
    assert runner._is_user_authorized(dm) is False


def test_disabled_group_rule_does_not_authorize_chat(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_dm_owner")
    runner, _adapter = _make_group_rule_runner({
        "oc_1": {"policy": "disabled"},
    })

    assert runner._is_user_authorized(_make_feishu_human_source("ou_guest")) is False


def test_default_open_group_policy_is_not_a_chat_allowlist(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_dm_owner")
    runner, _adapter = _make_group_rule_runner(
        {}, default_group_policy="open",
    )

    assert runner._is_user_authorized(_make_feishu_human_source("ou_guest")) is False


def test_explicit_sender_allowlist_is_enforced_before_gateway_authorization(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_dm_owner")
    runner, adapter = _make_group_rule_runner({
        "oc_1": {
            "policy": "allowlist",
            "allowlist": ["ou_group_member"],
        },
    })
    allowed_sender = SimpleNamespace(
        open_id="ou_group_member", user_id=None,
    )
    denied_sender = SimpleNamespace(
        open_id="ou_guest", user_id=None,
    )

    assert adapter._allow_group_message(allowed_sender, "oc_1") is True
    assert adapter._allow_group_message(denied_sender, "oc_1") is False
    assert runner._is_user_authorized(
        _make_feishu_human_source("ou_group_member")
    ) is True


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
