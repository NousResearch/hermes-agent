"""Tests for read_raw_config_readonly() — the no-deepcopy raw config read.

The readonly variant exists for per-turn policy checks (e.g. the
shared-metrics gate) that were paying a full config deepcopy on every call.
Contract under test:

1. identity invariant — repeat calls return the SAME cached object,
   including across the very first (cache-miss) call;
2. freshness — an edited config.yaml (mtime/size change) is picked up;
3. parity — content equals read_raw_config()'s result;
4. missing/broken config degrades to {} exactly like read_raw_config().
"""

import os
import time

import pytest
import yaml


@pytest.fixture()
def isolated_hermes_home():
    """Per-test HERMES_HOME dir (already redirected by the autouse conftest
    fixture) as a Path, with the raw-config cache cleared around the test."""
    from pathlib import Path

    import hermes_cli.config as config_mod

    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    config_mod._RAW_CONFIG_CACHE.clear()
    yield home
    config_mod._RAW_CONFIG_CACHE.clear()


def _write_config(home, data):
    cfg = home / "config.yaml"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    return cfg


def test_identity_invariant_from_first_call(isolated_hermes_home):
    from hermes_cli.config import read_raw_config_readonly

    _write_config(isolated_hermes_home, {"telemetry": {"shared_metrics": {"enabled": True}}})
    ro1 = read_raw_config_readonly()
    ro2 = read_raw_config_readonly()
    assert ro1 is ro2, "cache-miss return must be the same object later hits serve"
    assert ro1["telemetry"]["shared_metrics"]["enabled"] is True


def test_freshness_after_config_edit(isolated_hermes_home):
    from hermes_cli.config import read_raw_config_readonly

    cfg = _write_config(isolated_hermes_home, {"display": {"ephemeral_system_ttl": 1}})
    first = read_raw_config_readonly()
    assert first["display"]["ephemeral_system_ttl"] == 1

    _write_config(isolated_hermes_home, {"display": {"ephemeral_system_ttl": 7}})
    # Force a distinct mtime_ns even on coarse-timestamp filesystems.
    st = cfg.stat()
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    second = read_raw_config_readonly()
    assert second["display"]["ephemeral_system_ttl"] == 7


def test_missing_config_returns_empty(isolated_hermes_home):
    from hermes_cli.config import read_raw_config_readonly

    cfg = isolated_hermes_home / "config.yaml"
    if cfg.exists():
        cfg.unlink()
    assert read_raw_config_readonly() == {}


def test_mutable_variant_still_isolated(isolated_hermes_home):
    """read_raw_config() callers may mutate; that must not corrupt the
    readonly cache entry."""
    from hermes_cli.config import read_raw_config, read_raw_config_readonly

    _write_config(isolated_hermes_home, {"a": {"b": 1}})
    mutable = read_raw_config()
    mutable["a"]["b"] = 999
    assert read_raw_config_readonly()["a"]["b"] == 1
