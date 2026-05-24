"""Feishu settings and Hermes fallback group-policy checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.feishu.adapter import FeishuGroupRule


# --- FeishuAdapterSettings wiring ------------------------------------------


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("none", "none"),
        ("mentions", "mentions"),
        ("all", "all"),
        ("  Mentions  ", "mentions"),
    ],
)
def test_feishu_load_settings_populates_allow_bots(monkeypatch, env_value, expected):
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", env_value)

    settings = FeishuAdapter._load_settings(extra={})
    assert settings.allow_bots == expected


def test_feishu_load_settings_allow_bots_defaults_to_none(monkeypatch):
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.delenv("FEISHU_ALLOW_BOTS", raising=False)

    settings = FeishuAdapter._load_settings(extra={})
    assert settings.allow_bots == "none"


def test_feishu_load_settings_ignores_extra_allow_bots(monkeypatch):
    # extra is ignored: yaml is bridged to env before adapter settings load.
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.delenv("FEISHU_ALLOW_BOTS", raising=False)

    settings = FeishuAdapter._load_settings(extra={"allow_bots": "all"})
    assert settings.allow_bots == "none"


def test_feishu_load_settings_falls_back_to_env_when_extra_missing(monkeypatch):
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "mentions")

    settings = FeishuAdapter._load_settings(extra={})
    assert settings.allow_bots == "mentions"


def test_feishu_load_settings_warns_on_unknown_allow_bots(monkeypatch, caplog):
    import logging

    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "menton")

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.feishu"):
        settings = FeishuAdapter._load_settings(extra={})

    assert settings.allow_bots == "none"
    assert any("allow_bots" in r.message and "menton" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "env_value, extra, expected",
    [
        (None, {}, True),
        ("false", {}, False),
        ("true", {}, True),
        ("true", {"require_mention": False}, False),
    ],
)
def test_feishu_load_settings_require_mention(monkeypatch, env_value, extra, expected):
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
    if env_value is None:
        monkeypatch.delenv("FEISHU_REQUIRE_MENTION", raising=False)
    else:
        monkeypatch.setenv("FEISHU_REQUIRE_MENTION", env_value)

    settings = FeishuAdapter._load_settings(extra=extra)
    assert settings.require_mention is expected


def test_feishu_load_settings_parses_per_group_require_mention(monkeypatch):
    from gateway.platforms.feishu import FeishuAdapter

    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")

    settings = FeishuAdapter._load_settings(extra={
        "group_rules": {
            "oc_free": {"policy": "open", "require_mention": False},
            "oc_strict": {"policy": "open", "require_mention": True},
            "oc_inherit": {"policy": "open"},
        },
    })
    assert settings.group_rules["oc_free"].require_mention is False
    assert settings.group_rules["oc_strict"].require_mention is True
    assert settings.group_rules["oc_inherit"].require_mention is None


# --- Hermes fallback group policy ------------------------------------------


def _make_adapter(
    *,
    policy: str = "allowlist",
    allowed: set[str] | None = None,
    admins: set[str] | None = None,
    group_rules: dict[str, FeishuGroupRule] | None = None,
):
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "app_id": "cli_test",
                "app_secret": "secret_test",
                "domain": "feishu",
            },
        )
    )
    adapter._group_policy = policy
    adapter._default_group_policy = policy
    adapter._allowed_group_users = set(allowed or set())
    adapter._admins = set(admins or set())
    adapter._group_rules = group_rules or {}
    return adapter


def _sender_id(
    *,
    open_id: str | None = None,
    user_id: str | None = None,
    union_id: str | None = None,
):
    return SimpleNamespace(open_id=open_id, user_id=user_id, union_id=union_id)


def test_allow_group_message_admin_bypasses_locked_policy():
    adapter = _make_adapter(policy="disabled", admins={"ou_admin"})

    assert adapter._allow_group_message(_sender_id(open_id="ou_admin"), "oc_team") is True


@pytest.mark.parametrize("policy", ["disabled", "admin_only"])
def test_allow_group_message_locked_policies_reject_non_admins(policy):
    adapter = _make_adapter(policy=policy)

    assert adapter._allow_group_message(_sender_id(open_id="ou_user"), "oc_team") is False


def test_allow_group_message_open_policy_accepts_any_identified_sender():
    adapter = _make_adapter(policy="open")

    assert adapter._allow_group_message(_sender_id(open_id="ou_user"), "oc_team") is True


@pytest.mark.parametrize(
    "sender",
    [
        _sender_id(open_id="ou_allowed"),
        _sender_id(user_id="u_allowed"),
        _sender_id(union_id="on_allowed"),
    ],
)
def test_allow_group_message_allowlist_matches_any_sender_id(sender):
    adapter = _make_adapter(
        policy="allowlist",
        allowed={"ou_allowed", "u_allowed", "on_allowed"},
    )

    assert adapter._allow_group_message(sender, "oc_team") is True


def test_allow_group_message_allowlist_rejects_unlisted_sender():
    adapter = _make_adapter(policy="allowlist", allowed={"ou_allowed"})

    assert adapter._allow_group_message(_sender_id(open_id="ou_other"), "oc_team") is False


def test_allow_group_message_blacklist_rejects_listed_sender_and_accepts_others():
    adapter = _make_adapter(
        policy="blacklist",
        group_rules={
            "oc_team": FeishuGroupRule(policy="blacklist", blacklist={"ou_blocked"}),
        },
    )

    assert adapter._allow_group_message(_sender_id(open_id="ou_blocked"), "oc_team") is False
    assert adapter._allow_group_message(_sender_id(open_id="ou_other"), "oc_team") is True


def test_allow_group_message_per_chat_rule_overrides_default_policy():
    adapter = _make_adapter(
        policy="disabled",
        group_rules={
            "oc_team": FeishuGroupRule(policy="open"),
        },
    )

    assert adapter._allow_group_message(_sender_id(open_id="ou_user"), "oc_team") is True


def test_allow_group_message_admitted_bots_skip_human_allowlist_filters():
    adapter = _make_adapter(policy="allowlist", allowed={"ou_human"})

    assert adapter._allow_group_message(_sender_id(open_id="ou_peer_bot"), "oc_team", is_bot=True) is True


@pytest.mark.parametrize("policy", ["disabled", "admin_only"])
def test_allow_group_message_admitted_bots_do_not_bypass_channel_locks(policy):
    adapter = _make_adapter(policy=policy)

    assert adapter._allow_group_message(_sender_id(open_id="ou_peer_bot"), "oc_team", is_bot=True) is False
