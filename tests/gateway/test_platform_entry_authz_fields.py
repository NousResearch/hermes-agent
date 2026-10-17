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
