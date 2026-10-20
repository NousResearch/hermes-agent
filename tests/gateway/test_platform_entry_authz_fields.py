"""Bot-bypass and own-policy startup gates read the plugin's PlatformEntry, not a core table.

A platform plugin that declares ``allow_bots_env`` / ``own_policy_env`` gets the same
authorization and startup-abort behaviour core's bundled platforms do.
"""

from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource

_NAME = "authzprobe"


@pytest.fixture
def probe_platform(monkeypatch):
    from gateway.platform_registry import PlatformEntry, platform_registry

    for var in ("AUTHZPROBE_ALLOW_BOTS", "AUTHZPROBE_DM_POLICY", "AUTHZPROBE_GROUP_POLICY",
                "AUTHZPROBE_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(var, raising=False)
    platform_registry.register(PlatformEntry(
        name=_NAME, label="Authz Probe", adapter_factory=lambda cfg: None, check_fn=lambda: True,
        allow_all_env="AUTHZPROBE_ALLOW_ALL_USERS", allow_bots_env="AUTHZPROBE_ALLOW_BOTS",
        own_policy_env=("AUTHZPROBE_DM_POLICY", "AUTHZPROBE_GROUP_POLICY"),
    ))
    yield Platform(_NAME)
    platform_registry.unregister(_NAME)


def test_declared_allow_bots_env_admits_bot_without_user_id(probe_platform, monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    source = SessionSource(platform=probe_platform, chat_id="c1", chat_type="group", user_id=None,
                           user_name="", is_bot=True)
    assert runner._is_user_authorized(source) is False
    monkeypatch.setenv("AUTHZPROBE_ALLOW_BOTS", "all")
    assert runner._is_user_authorized(source) is True


def test_declared_own_policy_env_refuses_open_policy_without_allow_all(probe_platform, monkeypatch):
    from gateway.run import _own_policy_open_startup_violation

    cfg = GatewayConfig(platforms={probe_platform: PlatformConfig(enabled=True)})
    assert _own_policy_open_startup_violation(cfg) is None
    monkeypatch.setenv("AUTHZPROBE_GROUP_POLICY", "open")
    assert _own_policy_open_startup_violation(cfg) == "authzprobe: open policy without allow-all opt-in"
    monkeypatch.setenv("AUTHZPROBE_ALLOW_ALL_USERS", "true")
    assert _own_policy_open_startup_violation(cfg) is None


@pytest.mark.parametrize("declared", ["both", "group_only", "none"])
def test_declared_group_chat_env_and_pairing_suppression(monkeypatch, declared):
    """``group_allowed_chats_env`` authorizes group chats by id; only ``allowlist_suppresses_pairing``
    turns the unauthorized-DM default from "pair" into "ignore" (a group list alone never does)."""
    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.run import GatewayRunner

    for var in ("GRPPROBE_ALLOWED_USERS", "GRPPROBE_GROUP_CHATS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(var, raising=False)
    fields = {"both": dict(group_allowed_chats_env="GRPPROBE_GROUP_CHATS", allowlist_suppresses_pairing=True),
              "group_only": dict(group_allowed_chats_env="GRPPROBE_GROUP_CHATS"), "none": {}}[declared]
    platform_registry.register(PlatformEntry(
        name="grpprobe", label="Group Probe", adapter_factory=lambda cfg: None, check_fn=lambda: True,
        allowed_users_env="GRPPROBE_ALLOWED_USERS", **fields))
    try:
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={Platform("grpprobe"): PlatformConfig(enabled=True)})
        runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
        group = SessionSource(platform=Platform("grpprobe"), chat_id="g1", chat_type="group", user_id=None,
                              user_name="")
        monkeypatch.setenv("GRPPROBE_GROUP_CHATS", "g1")
        assert runner._is_user_authorized(group) is (declared != "none")
        dm = runner._get_unauthorized_dm_behavior(Platform("grpprobe"))
        assert dm == ("ignore" if declared == "both" else "pair")
        monkeypatch.delenv("GRPPROBE_GROUP_CHATS")
        monkeypatch.setenv("GRPPROBE_ALLOWED_USERS", "owner")
        assert runner._get_unauthorized_dm_behavior(Platform("grpprobe")) == ("ignore" if declared == "both" else "pair")
    finally:
        platform_registry.unregister("grpprobe")
