"""Unit tests for the TTL cache on _load_gateway_config.

Covers:
- Cache hits within the 60-second TTL window (no repeated yaml.safe_load)
- _invalidate_gateway_config_cache forces a reload on the next call
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import gateway.run as gw_run
from gateway.run import _invalidate_gateway_config_cache, _load_gateway_config


@pytest.fixture(autouse=True)
def _reset_ttl_cache():
    """Always start each test with a cold cache and restore afterwards."""
    _invalidate_gateway_config_cache()
    yield
    _invalidate_gateway_config_cache()


class TestLoadGatewayConfigTTLCacheHitsWithinWindow:
    """Calling _load_gateway_config twice within 60 s must only hit the
    underlying loader once — the second call returns the cached value."""

    def test_cache_hit_skips_yaml_safe_load(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("model:\n  default: gpt-4\n", encoding="utf-8")

        load_count = {"n": 0}
        real_safe_load = __import__("yaml").safe_load

        def counting_safe_load(stream):
            load_count["n"] += 1
            return real_safe_load(stream)

        # Force the fallback path (direct yaml read) by making get_config_path
        # return a different path so the read_raw_config fast-path is bypassed.
        with (
            patch.object(gw_run, "_hermes_home", tmp_path),
            patch("hermes_cli.config.get_config_path", return_value=Path("/nonexistent")),
            patch("yaml.safe_load", side_effect=counting_safe_load),
        ):
            result1 = _load_gateway_config()
            result2 = _load_gateway_config()

        assert result1 == result2, "Both calls must return the same dict"
        assert load_count["n"] == 1, (
            f"yaml.safe_load should be called exactly once within the TTL window; "
            f"was called {load_count['n']} time(s)"
        )

    def test_cache_returns_same_object(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("model:\n  default: claude-3\n", encoding="utf-8")

        with (
            patch.object(gw_run, "_hermes_home", tmp_path),
            patch("hermes_cli.config.get_config_path", return_value=Path("/nonexistent")),
        ):
            r1 = _load_gateway_config()
            r2 = _load_gateway_config()

        # Within the TTL window the exact same dict object is returned.
        assert r1 is r2, "TTL cache must return the identical dict object on a hit"


class TestInvalidateGatewayConfigCacheForcesReload:
    """_invalidate_gateway_config_cache must cause the next call to
    re-read config from disk even if the TTL has not expired."""

    def test_invalidate_forces_reload(self, tmp_path):
        config_yaml = tmp_path / "config.yaml"
        config_yaml.write_text("model:\n  default: first-model\n", encoding="utf-8")

        load_count = {"n": 0}
        real_safe_load = __import__("yaml").safe_load

        def counting_safe_load(stream):
            load_count["n"] += 1
            return real_safe_load(stream)

        with (
            patch.object(gw_run, "_hermes_home", tmp_path),
            patch("hermes_cli.config.get_config_path", return_value=Path("/nonexistent")),
            patch("yaml.safe_load", side_effect=counting_safe_load),
        ):
            r1 = _load_gateway_config()
            assert load_count["n"] == 1

            # Invalidate and update the file on disk
            _invalidate_gateway_config_cache()
            config_yaml.write_text("model:\n  default: second-model\n", encoding="utf-8")

            r2 = _load_gateway_config()

        assert load_count["n"] == 2, (
            f"After invalidation, yaml.safe_load must be called again; "
            f"total calls: {load_count['n']}"
        )
        assert r1 != r2, "Second load must reflect updated file content"
        assert r1.get("model", {}).get("default") == "first-model"
        assert r2.get("model", {}).get("default") == "second-model"

    def test_invalidate_resets_cache_vars(self):
        """After invalidation the module-level cache vars are in cold state."""
        with patch("hermes_cli.config.get_config_path", return_value=Path("/nonexistent")):
            _load_gateway_config()

        _invalidate_gateway_config_cache()

        assert gw_run._config_cache_value is None
        assert gw_run._config_cache_expires_at == 0.0
