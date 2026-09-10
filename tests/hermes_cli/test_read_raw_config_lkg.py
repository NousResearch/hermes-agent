"""Tests for read_raw_config() LKG fallback on YAML parse errors.

Similar to _load_config_impl's last-known-good guard, read_raw_config()
must serve the previously-cached raw config when the file becomes
unparseable, rather than silently returning {} and letting a
read→mutate→save_config() caller overwrite the corrupted file.
"""

import os

import pytest
import yaml


@pytest.fixture()
def isolated_hermes_home():
    """Per-test HERMES_HOME dir with the raw-config cache cleared."""
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


def test_lkg_on_broken_yaml(isolated_hermes_home):
    """read_raw_config returns last-known-good when YAML becomes broken."""
    from hermes_cli.config import read_raw_config

    # Populate cache with valid config
    _write_config(
        isolated_hermes_home,
        {"model": {"default": "claude-opus-4-8"}, "agent": {"max_turns": 90}},
    )
    first = read_raw_config()
    assert first == {"model": {"default": "claude-opus-4-8"}, "agent": {"max_turns": 90}}

    # Corrupt the file
    cfg = isolated_hermes_home / "config.yaml"
    cfg.write_text("model:\n  default: claude\n  provider: [broken YAML")
    # Bump mtime to invalidate the signature-based cache hit
    st = cfg.stat()
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    # Should return LKG, not {}
    second = read_raw_config()
    assert second == {"model": {"default": "claude-opus-4-8"}, "agent": {"max_turns": 90}}, (
        f"Expected LKG config, got {second}"
    )


def test_cold_start_broken_yaml_still_returns_empty(isolated_hermes_home):
    """Cold start with no cache: broken YAML still returns {} (existing behavior)."""
    from hermes_cli.config import read_raw_config

    cfg = isolated_hermes_home / "config.yaml"
    cfg.write_text("model:\n  default: claude\n  provider: [broken YAML")

    result = read_raw_config()
    assert result == {}, f"Cold start with broken YAML should return {{}}, got {result}"
