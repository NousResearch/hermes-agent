"""RBAC config cache must be keyed by the config file's path (multiplex fix 1).

``hermes profile create --clone`` copies config.yaml with ``shutil.copy2``,
which preserves mtime. A cache keyed only on ``(mtime_ns, size)`` therefore
serves one profile's ``user_roles`` to the other whenever the two files happen
to have the same size — which an edited clone often does.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from gateway import tool_access
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _write(home, user):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"slack:\n  user_roles:\n    {user}: admin\n", encoding="utf-8"
    )


@pytest.fixture()
def two_homes(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write(a, "UAAAAAAA1")
    _write(b, "UBBBBBBB2")  # same length → same size
    st = (a / "config.yaml").stat()
    os.utime(b / "config.yaml", ns=(st.st_atime_ns, st.st_mtime_ns))
    assert (a / "config.yaml").stat().st_size == (b / "config.yaml").stat().st_size
    assert (a / "config.yaml").stat().st_mtime_ns == (b / "config.yaml").stat().st_mtime_ns
    tool_access._config_cache.clear()
    yield a, b
    tool_access._config_cache.clear()


def _grant(home, user):
    tok = set_hermes_home_override(str(home))
    try:
        policy = tool_access.policy_for_platform("slack")
        return policy.grant_for(user, None) or set()
    finally:
        reset_hermes_home_override(tok)


def test_cloned_profile_same_mtime_size_isolated(two_homes):
    a, b = two_homes
    assert "*" in _grant(a, "UAAAAAAA1")
    # Before the fix the cache hit returned home A's config for home B.
    assert "*" in _grant(b, "UBBBBBBB2")
    assert "*" not in _grant(b, "UAAAAAAA1")
    assert "*" not in _grant(a, "UBBBBBBB2")


def test_cache_still_reused_within_one_home(two_homes):
    a, _ = two_homes
    from gateway import config as gw_config

    with patch.object(gw_config, "load_gateway_config", wraps=gw_config.load_gateway_config) as spy:
        _grant(a, "UAAAAAAA1")
        _grant(a, "UAAAAAAA1")
    assert spy.call_count == 1
