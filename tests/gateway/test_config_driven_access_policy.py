"""Tests for config-driven platform access policies at the gateway layer.

Background (#34515): WeCom, Weixin, Yuanbao, QQBot, and WhatsApp expose a
documented config-driven access surface (``dm_policy`` / ``group_policy`` /
``allow_from`` / ``group_allow_from`` in ``PlatformConfig.extra``) and enforce
it at intake —
a message is dropped inside the adapter and never reaches the gateway unless it
already passed that policy.

The gateway's env-based allowlist check (``_is_user_authorized``) runs *after*
the adapter. Adapters that own their access policy declare
``enforces_own_access_policy`` (a ``BasePlatformAdapter`` property, default
``False``) so the gateway can honor a config-only ``dm_policy: allowlist`` /
``allow_from`` (which the adapter already enforced) instead of double-denying it
when no ``PLATFORM_ALLOWED_USERS`` env var is set.

Crucially, the flag is NOT a blanket "already authorized" pass. These adapters
default ``dm_policy`` / ``group_policy`` to ``"open"``, which forwards *every*
sender, so the gateway trusts the adapter only when its effective policy for the
chat type is an actual ``"allowlist"`` restriction. Trusting ``"open"`` here
admitted the whole external network with no operator-configured allowlist — the
fail-open SECURITY.md §2.6 forbids for network-exposed adapters ("an allowlist
is required for every enabled network-exposed adapter ... code paths that fail
open when no allowlist is configured are code bugs"). Open access requires an
explicit ``{PLATFORM}_ALLOW_ALL_USERS`` / ``GATEWAY_ALLOW_ALL_USERS`` opt-in.
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
# Layer 2: gateway trusts the adapter-enforced flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_allowlist_authorized_without_env_allowlist(monkeypatch, platform):
    """A config-only ``dm_policy: allowlist`` is trusted without an env allowlist.

    The adapter only forwards an allowlisted sender under ``allowlist`` policy,
    so a message reaching the gateway *was* authorized for this specific sender.
    The gateway must honor that instead of double-denying (the #34515 case).
    """
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra={"dm_policy": "allowlist"})}
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)

    assert runner._is_user_authorized(_source(platform)) is True


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_open_dm_not_authorized_without_allowlist(monkeypatch, platform):
    """``dm_policy: open`` forwards everyone → NOT authorization (SECURITY.md §2.6).

    With no env allowlist and no per-platform allow-all flag, an own-policy
    adapter running ``open`` (the default) must NOT fail open: the gateway falls
    through to default-deny so the whole external network can't reach the agent.
    """
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra={"dm_policy": "open"})}
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)

    assert runner._is_user_authorized(_source(platform)) is False


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_default_open_dm_is_fail_closed(monkeypatch, platform):
    """The adapters' *default* ``open`` policy (no config at all) fails closed.

    Operators who enable an own-policy adapter with only credentials get
    ``dm_policy = "open"`` resolved on the live adapter. Simulate that resolved
    state (empty config.extra, adapter ``_dm_policy = "open"``) and confirm the
    gateway denies — the do-nothing default must not be open to the world.
    """
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(platforms={platform: PlatformConfig(enabled=True, extra={})})
    runner, adapter = _make_runner(platform, config, enforces=True)
    adapter._dm_policy = "open"  # as the live adapter resolves the default

    assert runner._is_user_authorized(_source(platform)) is False


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_allowlist_authorized_for_group_chat(monkeypatch, platform):
    """A config-only ``group_policy: allowlist`` is trusted for group traffic."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra={"group_policy": "allowlist"})}
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)

    assert runner._is_user_authorized(_source(platform, chat_type="group")) is True


@pytest.mark.parametrize("platform", _OWN_POLICY_PLATFORMS)
def test_own_policy_open_group_not_authorized_without_allowlist(monkeypatch, platform):
    """``group_policy: open`` is the same fail-open class as DM open → deny."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra={"group_policy": "open"})}
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)

    assert runner._is_user_authorized(_source(platform, chat_type="group")) is False


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
# Layer 2b: `dm_policy: pairing` is NOT blanket-trusted
# ---------------------------------------------------------------------------
#
# Regression: WeCom/Weixin document ``dm_policy: pairing`` and declare
# ``enforces_own_access_policy=True``, but their intake helper only special-cases
# ``disabled`` / ``allowlist`` — ``pairing`` falls through and forwards the DM so
# the gateway can run its pairing handshake. With no env allowlist, the
# adapter-trust shortcut above then authorized *every* unpaired sender, silently
# degrading pairing mode to open access. The shortcut must skip pairing-mode DMs
# so an unpaired sender falls through to default-deny (and gets a pairing code).


@pytest.mark.parametrize("platform", [Platform.WECOM, Platform.WEIXIN])
def test_pairing_dm_policy_not_blanket_authorized(monkeypatch, platform):
    """An unpaired sender in ``dm_policy: pairing`` is NOT authorized."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, extra={"dm_policy": "pairing"})}
    )
    runner, _adapter = _make_runner(platform, config, enforces=True)
    # pairing_store.is_approved already returns False (set in _make_runner).

    assert runner._is_user_authorized(_source(platform)) is False


def test_pairing_dm_policy_authorizes_paired_user(monkeypatch):
    """Once approved in the pairing store, the sender authorizes normally."""
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.WECOM: PlatformConfig(enabled=True, extra={"dm_policy": "pairing"})}
    )
    runner, _adapter = _make_runner(Platform.WECOM, config, enforces=True)
    runner.pairing_store.is_approved.return_value = True

    assert runner._is_user_authorized(_source(Platform.WECOM)) is True


def test_pairing_carveout_reads_adapter_when_env_set(monkeypatch):
    """Env-only ``WECOM_DM_POLICY=pairing`` (absent from config.extra) is honored.

    The adapter resolves ``dm_policy`` from the env var, so its ``_dm_policy`` is
    authoritative even when ``config.extra`` is empty. The carve-out must read
    that, not just config.
    """
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={Platform.WECOM: PlatformConfig(enabled=True, extra={})}
    )
    runner, adapter = _make_runner(Platform.WECOM, config, enforces=True)
    adapter._dm_policy = "pairing"  # as the adapter would resolve from the env var

    assert runner._is_user_authorized(_source(Platform.WECOM)) is False


def test_pairing_dm_policy_group_chat_still_trusted(monkeypatch):
    """Pairing is DM-only — the DM pairing carve-out doesn't gate group traffic.

    Group access is governed by ``group_policy``, so an allowlisted group is
    still trusted even while DMs are in ``pairing`` mode.
    """
    _clear_auth_env(monkeypatch)
    config = GatewayConfig(
        platforms={
            Platform.WECOM: PlatformConfig(
                enabled=True, extra={"dm_policy": "pairing", "group_policy": "allowlist"}
            )
        }
    )
    runner, _adapter = _make_runner(Platform.WECOM, config, enforces=True)

    assert runner._is_user_authorized(_source(Platform.WECOM, chat_type="group")) is True


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
