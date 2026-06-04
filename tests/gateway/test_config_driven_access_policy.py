"""Tests for config-driven platform access policies at the gateway layer.

Background (#34515): WeCom, Weixin, Yuanbao, QQBot, and WhatsApp expose a
documented config-driven access surface (``dm_policy`` / ``group_policy`` /
``allow_from`` / ``group_allow_from`` in ``PlatformConfig.extra``) and enforce
it at intake —
a message is dropped inside the adapter and never reaches the gateway unless it
already passed that policy.

The gateway's env-based allowlist check (``_is_user_authorized``) runs *after*
the adapter. A prior compatibility fix trusted any adapter that declared
``enforces_own_access_policy`` when gateway env allowlists were empty. That made
``dm_policy: open`` / ``group_policy: open`` equivalent to network-wide gateway
authorization.

The security contract is narrower: adapters may declare that they own an intake
policy, but the gateway only treats that intake decision as authorization when
the current source is backed by a concrete configured allowlist.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


# Platforms whose adapters own their access policy at intake.
_OWN_POLICY_PLATFORMS = [
    Platform.WECOM,
    Platform.WEIXIN,
    Platform.YUANBAO,
    Platform.QQBOT,
    Platform.WHATSAPP,
]


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "WECOM_ALLOWED_USERS",
        "WEIXIN_ALLOWED_USERS",
        "YUANBAO_ALLOWED_USERS",
        "QQ_ALLOWED_USERS",
        "QQ_GROUP_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "TELEGRAM_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "WECOM_ALLOW_ALL_USERS",
        "WEIXIN_ALLOW_ALL_USERS",
        "YUANBAO_ALLOW_ALL_USERS",
        "QQ_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_runner(platform: Platform, config: GatewayConfig, *, enforces: bool):
    """Build a bare GatewayRunner with one adapter for *platform*.

    ``enforces`` controls whether the adapter declares
    ``enforces_own_access_policy`` — i.e. whether it owns its access gate.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock(), enforces_own_access_policy=enforces)
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    return runner, adapter


def _source(platform: Platform, *, chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="some-user",
        chat_id="some-chat",
        user_name="tester",
        chat_type=chat_type,
    )


# ---------------------------------------------------------------------------
# Layer 1: the base-class contract and per-adapter overrides
# ---------------------------------------------------------------------------


def test_base_adapter_defaults_to_not_owning_access_policy():
    """Adapters that don't override the property delegate to the gateway."""
    from gateway.platforms.base import BasePlatformAdapter

    # The default lives on the base property descriptor.
    assert BasePlatformAdapter.enforces_own_access_policy.fget(object()) is False


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("gateway.platforms.wecom", "WeComAdapter"),
        ("gateway.platforms.weixin", "WeixinAdapter"),
        ("gateway.platforms.yuanbao", "YuanbaoAdapter"),
        ("gateway.platforms.qqbot.adapter", "QQAdapter"),
        ("gateway.platforms.whatsapp", "WhatsAppAdapter"),
    ],
)
def test_own_policy_adapters_declare_the_flag(module_path, class_name):
    """The config-policy adapters override the flag to True."""
    import importlib

    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    # Property is overridden on the subclass and returns True regardless of
    # instance state (it reflects a static capability, not runtime config).
    value = adapter_cls.enforces_own_access_policy.fget(object.__new__(adapter_cls))
    assert value is True


# ---------------------------------------------------------------------------
# Layer 2: gateway trusts only concrete adapter allowlists
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_open_dm_default_denies_without_allowlist(monkeypatch, platform):
    """Open adapter policy is not enough to authorize network callers."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra={"dm_policy": "open"})}
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)

    assert runner._is_user_authorized(_source(platform)) is False


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_allowlisted_dm_is_authorized_without_env_allowlist(monkeypatch, platform):
    """Config-only adapter allowlists remain valid authorization evidence."""
    _clear_auth_env(monkeypatch)
    allowlist_key = "dm_allow_from" if platform == Platform.YUANBAO else "allow_from"
    config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                extra={"dm_policy": "allowlist", allowlist_key: ["some-user"]},
            )
        }
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)

    assert runner._is_user_authorized(_source(platform)) is True


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_allowlisted_dm_rejects_non_matching_sender(monkeypatch, platform):
    """The gateway re-checks the concrete adapter allowlist before trusting it."""
    _clear_auth_env(monkeypatch)
    allowlist_key = "dm_allow_from" if platform == Platform.YUANBAO else "allow_from"
    config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                extra={"dm_policy": "allowlist", allowlist_key: ["allowed-user"]},
            )
        }
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)

    assert runner._is_user_authorized(_source(platform)) is False


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_open_group_default_denies_without_allowlist(monkeypatch, platform):
    """Open group policy is not enough to authorize network callers."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra={"group_policy": "open"})}
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)

    assert runner._is_user_authorized(_source(platform, chat_type="group")) is False


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_allowlisted_group_is_authorized_without_env_allowlist(monkeypatch, platform):
    """Config-only group allowlists remain valid authorization evidence."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["some-chat"]},
            )
        }
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)

    assert runner._is_user_authorized(_source(platform, chat_type="group")) is True


def test_yuanbao_env_parsed_dm_allowlist_is_authorized(monkeypatch):
    """Adapter-parsed env allowlists are trusted without duplicating env reads."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(platforms={Platform.YUANBAO: PlatformConfig(enabled=True)})
    runner, adapter = _make_runner(Platform.YUANBAO, config, enforces=True)
    adapter._access_policy = SimpleNamespace(
        dm_policy="allowlist",
        dm_allow_from=["some-user"],
        group_policy="open",
        group_allow_from=[],
    )

    assert runner._is_user_authorized(_source(Platform.YUANBAO)) is True


def test_weixin_env_parsed_group_allowlist_is_authorized(monkeypatch):
    """Group allowlists parsed by the adapter remain valid gateway evidence."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(platforms={Platform.WEIXIN: PlatformConfig(enabled=True)})
    runner, adapter = _make_runner(Platform.WEIXIN, config, enforces=True)
    adapter._group_policy = "allowlist"
    adapter._group_allow_from = ["some-chat"]

    assert runner._is_user_authorized(_source(Platform.WEIXIN, chat_type="group")) is True


def test_yuanbao_group_allowlist_matches_raw_group_code(monkeypatch):
    """Yuanbao stores raw group codes while SessionSource chat_id is prefixed."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(platforms={Platform.YUANBAO: PlatformConfig(enabled=True)})
    runner, adapter = _make_runner(Platform.YUANBAO, config, enforces=True)
    adapter._access_policy = SimpleNamespace(
        dm_policy="open",
        dm_allow_from=[],
        group_policy="allowlist",
        group_allow_from=["some-chat"],
    )

    source = _source(Platform.YUANBAO, chat_type="group")
    source.chat_id = "group:some-chat"
    assert runner._is_user_authorized(source) is True


def test_wecom_config_allowlist_uses_adapter_normalization(monkeypatch):
    """WeCom prefixes and case are normalized like the adapter intake check."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={
            Platform.WECOM: PlatformConfig(
                enabled=True,
                extra={"dm_policy": "allowlist", "allow_from": ["wecom:USER:Some-User"]},
            )
        }
    )
    runner, _adapter = _make_runner(Platform.WECOM, config, enforces=True)

    assert runner._is_user_authorized(_source(Platform.WECOM, chat_type="dm")) is True


def test_qqbot_guild_group_allowlist_matches_guild_id(monkeypatch):
    """QQBot guild messages are allowlisted by guild id, not channel id."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={
            Platform.QQBOT: PlatformConfig(
                enabled=True,
                extra={"group_policy": "allowlist", "group_allow_from": ["guild-1"]},
            )
        }
    )
    runner, _adapter = _make_runner(Platform.QQBOT, config, enforces=True)
    source = _source(Platform.QQBOT, chat_type="group")
    source.chat_id = "channel-1"
    source.guild_id = "guild-1"

    assert runner._is_user_authorized(source) is True


def test_non_owning_platform_still_default_denies(monkeypatch):
    """Adapters that don't own their policy keep the env-only default-deny."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")}
    )
    runner, _adapter = _make_runner(Platform.TELEGRAM, config, enforces=False)

    assert runner._is_user_authorized(_source(Platform.TELEGRAM)) is False


def test_env_allowlist_still_takes_precedence_for_own_policy_platform(monkeypatch):
    """When an env allowlist IS set, it governs — adapter trust is a fallback.

    The adapter-trust branch only fires when no env allowlist exists, so an
    operator who sets ``WECOM_ALLOWED_USERS`` still gets env-based gating and
    a non-listed user is denied.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WECOM_ALLOWED_USERS", "allowed-user")
    config = GatewayConfig(
        platforms={Platform.WECOM: PlatformConfig(enabled=True, extra={"dm_policy": "open"})}
    )
    runner, _adapter = _make_runner(Platform.WECOM, config, enforces=True)

    listed = SessionSource(
        platform=Platform.WECOM, user_id="allowed-user", chat_id="c",
        user_name="t", chat_type="dm",
    )
    stranger = SessionSource(
        platform=Platform.WECOM, user_id="stranger", chat_id="c",
        user_name="t", chat_type="dm",
    )
    assert runner._is_user_authorized(listed) is True
    assert runner._is_user_authorized(stranger) is False


def test_unknown_adapter_does_not_crash_trust_check(monkeypatch):
    """No adapter registered for the platform → safe default-deny."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(platforms={Platform.WECOM: PlatformConfig(enabled=True)})
    runner, _adapter = _make_runner(Platform.WECOM, config, enforces=True)
    runner.adapters = {}  # nothing registered

    assert runner._adapter_enforces_own_access_policy(Platform.WECOM) is False
    assert runner._is_user_authorized(_source(Platform.WECOM)) is False


# ---------------------------------------------------------------------------
# Layer 3: unauthorized-DM behavior reads config dm_policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dm_policy, expected",
    [
        ("allowlist", "ignore"),
        ("disabled", "ignore"),
        ("pairing", "pair"),
    ],
)
def test_unauthorized_dm_behavior_follows_config_dm_policy(monkeypatch, dm_policy, expected):
    """A restrictive dm_policy drops unauthorized DMs; pairing opts back in."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.WECOM: PlatformConfig(enabled=True, extra={"dm_policy": dm_policy})}
    )
    runner, _adapter = _make_runner(Platform.WECOM, config, enforces=True)

    assert runner._get_unauthorized_dm_behavior(Platform.WECOM) == expected


def test_unauthorized_dm_behavior_open_policy_keeps_default(monkeypatch):
    """``dm_policy: open`` is not restrictive → falls through to the default."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.WECOM: PlatformConfig(enabled=True, extra={"dm_policy": "open"})}
    )
    runner, _adapter = _make_runner(Platform.WECOM, config, enforces=True)

    # No allowlist + no restrictive policy → open-gateway pairing default.
    assert runner._get_unauthorized_dm_behavior(Platform.WECOM) == "pair"
