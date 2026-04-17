"""Sandbox type helpers."""

from sandbox.types import IsolationProfile, isolation_profile_from_config


def test_isolation_profile_defaults():
    p = IsolationProfile()
    assert p.network_policy == "bridge"
    assert p.mem_limit_mb == 2048


def test_isolation_profile_from_config_nested():
    cfg = {
        "profiles": {
            "strict": {
                "network_policy": "none",
                "mem_limit_mb": 512,
            }
        }
    }
    p = isolation_profile_from_config(cfg, "strict")
    assert p.network_policy == "none"
    assert p.mem_limit_mb == 512
