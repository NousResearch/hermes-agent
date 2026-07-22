"""Per-sender WhatsApp toolset isolation for shared business lines."""

from types import SimpleNamespace

import gateway.run as gateway_run
import pytest

from gateway.config import Platform
from gateway.session import SessionSource


FULL_TOOLSETS = ["context_engine", "file", "terminal", "web"]
EXTERNAL_TOOLSETS = ["context_engine"]


@pytest.fixture(autouse=True)
def _clear_home_channel_env(monkeypatch):
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)


def _config(**whatsapp):
    return {
        "whatsapp": {
            "nonowner_enabled_toolsets": EXTERNAL_TOOLSETS,
            **whatsapp,
        }
    }


def _source(*, chat_id="contact@s.whatsapp.net", user_id=None, chat_type="dm"):
    return SimpleNamespace(
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_id_alt=None,
    )


def test_unconfigured_and_other_platform_are_unchanged():
    source = _source()
    assert gateway_run._whatsapp_owner_tool_gate(
        source, "whatsapp", {"whatsapp": {}}, FULL_TOOLSETS
    ) is FULL_TOOLSETS
    assert gateway_run._whatsapp_owner_tool_gate(
        source, "telegram", _config(), FULL_TOOLSETS
    ) is FULL_TOOLSETS


def test_nonowner_gets_only_explicit_allowlist():
    assert gateway_run._whatsapp_owner_tool_gate(
        _source(user_id="contact@s.whatsapp.net"),
        "whatsapp",
        _config(owner_users=["owner"]),
        FULL_TOOLSETS,
    ) == EXTERNAL_TOOLSETS


def test_nonowner_allowlist_only_reduces_enabled_toolsets_in_platform_order():
    enabled = ["web", "context_engine", "file"]
    assert gateway_run._whatsapp_owner_tool_gate(
        _source(user_id="contact@s.whatsapp.net"),
        "whatsapp",
        _config(
            owner_users=["owner"],
            nonowner_enabled_toolsets=["terminal", "context_engine", "web"],
        ),
        enabled,
    ) == ["web", "context_engine"]


def test_nonowner_policy_stays_explicit_when_allowlist_matches_platform_tools():
    toolsets = ["context_engine"]
    gated, restricted = gateway_run._whatsapp_owner_tool_policy(
        _source(user_id="contact@s.whatsapp.net"),
        "whatsapp",
        _config(owner_users=["owner"]),
        toolsets,
    )
    assert gated == toolsets
    assert restricted is True


def test_owner_matches_phone_lid_aliases(monkeypatch):
    aliases = {
        "owner-phone": {"owner-phone", "owner-lid"},
        "owner-lid": {"owner-phone", "owner-lid"},
    }
    monkeypatch.setattr(
        gateway_run,
        "_expand_whatsapp_auth_aliases",
        lambda value: aliases.get(value, {value}),
    )

    result = gateway_run._whatsapp_owner_tool_gate(
        _source(user_id="owner-lid"),
        "whatsapp",
        _config(owner_users=["owner-phone"]),
        FULL_TOOLSETS,
    )
    assert result is FULL_TOOLSETS


def test_standard_home_channel_env_identifies_owner(monkeypatch):
    monkeypatch.setenv("WHATSAPP_HOME_CHANNEL", "owner")
    assert gateway_run._whatsapp_owner_tool_gate(
        _source(user_id="owner@s.whatsapp.net"),
        "whatsapp",
        _config(),
        FULL_TOOLSETS,
    ) is FULL_TOOLSETS


def test_home_channel_env_overrides_stale_yaml(monkeypatch):
    monkeypatch.setenv("WHATSAPP_HOME_CHANNEL", "current-owner")
    config = _config(home_channel={"chat_id": "stale-owner"})

    assert gateway_run._whatsapp_owner_tool_gate(
        _source(user_id="stale-owner@s.whatsapp.net"),
        "whatsapp",
        config,
        FULL_TOOLSETS,
    ) == EXTERNAL_TOOLSETS


def test_group_chat_id_never_grants_owner_tools():
    config = _config(home_channel={"chat_id": "home-group@g.us"})
    result = gateway_run._whatsapp_owner_tool_gate(
        _source(
            chat_id="home-group@g.us",
            user_id="home-group@s.whatsapp.net",
            chat_type="group",
        ),
        "whatsapp",
        config,
        FULL_TOOLSETS,
    )
    assert result == EXTERNAL_TOOLSETS


def test_group_owner_is_matched_from_sender():
    config = _config(owner_users=["owner"])
    result = gateway_run._whatsapp_owner_tool_gate(
        _source(
            chat_id="group@g.us",
            user_id="owner@s.whatsapp.net",
            chat_type="group",
        ),
        "whatsapp",
        config,
        FULL_TOOLSETS,
    )
    assert result is FULL_TOOLSETS


def test_home_channel_env_is_profile_scoped(monkeypatch):
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    monkeypatch.setenv("WHATSAPP_HOME_CHANNEL", "default-owner")
    set_multiplex_active(True)
    token = set_secret_scope({"WHATSAPP_HOME_CHANNEL": "profile-owner"})
    try:
        assert gateway_run._whatsapp_owner_tool_gate(
            _source(user_id="profile-owner@s.whatsapp.net"),
            "whatsapp",
            _config(),
            FULL_TOOLSETS,
        ) is FULL_TOOLSETS
        assert gateway_run._whatsapp_owner_tool_gate(
            _source(user_id="default-owner@s.whatsapp.net"),
            "whatsapp",
            _config(),
            FULL_TOOLSETS,
        ) == EXTERNAL_TOOLSETS
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_missing_dm_sender_falls_back_to_chat_id():
    config = _config(home_channel={"chat_id": "owner"})
    assert gateway_run._whatsapp_owner_tool_gate(
        _source(chat_id="owner@s.whatsapp.net"),
        "whatsapp",
        config,
        FULL_TOOLSETS,
    ) is FULL_TOOLSETS


@pytest.mark.parametrize(
    "config_path",
    [("platforms",), ("gateway", "platforms")],
)
def test_nested_platform_home_channel_identifies_owner(config_path):
    config = _config()
    target = config
    for key in config_path:
        target = target.setdefault(key, {})
    target["whatsapp"] = {"home_channel": {"chat_id": "owner"}}

    assert gateway_run._whatsapp_owner_tool_gate(
        _source(user_id="owner@s.whatsapp.net"),
        "whatsapp",
        config,
        FULL_TOOLSETS,
    ) is FULL_TOOLSETS


def test_invalid_config_or_identity_resolution_fails_closed(monkeypatch):
    assert gateway_run._whatsapp_owner_tool_gate(
        _source(user_id="owner"),
        "whatsapp",
        {"whatsapp": {"nonowner_enabled_toolsets": "context_engine"}},
        FULL_TOOLSETS,
    ) == []

    monkeypatch.setattr(
        gateway_run,
        "_expand_whatsapp_auth_aliases",
        lambda _value: (_ for _ in ()).throw(OSError("broken alias map")),
    )
    assert gateway_run._whatsapp_owner_tool_gate(
        _source(user_id="owner"),
        "whatsapp",
        _config(owner_users=["owner"]),
        FULL_TOOLSETS,
    ) == EXTERNAL_TOOLSETS


def test_toolset_change_alters_agent_cache_signature():
    owner_signature = gateway_run.GatewayRunner._agent_config_signature(
        "model", {}, FULL_TOOLSETS, ""
    )
    external_signature = gateway_run.GatewayRunner._agent_config_signature(
        "model", {}, EXTERNAL_TOOLSETS, ""
    )
    assert owner_signature != external_signature
