"""#141857: load_config() cache must be keyed by scope so scoped and unscoped callers
don't evict each other's entries and re-parse config.yaml on every crossing.
"""
import os
from unittest.mock import patch

from hermes_cli import config as config_mod


def test_scoped_and_unscoped_cache_entries_are_separate(tmp_path):
    """Alternating scoped and unscoped load_config() calls must not evict each other."""
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_mod._LOAD_CONFIG_CACHE.clear()
        config_mod._RAW_CONFIG_CACHE.clear()
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model:\n  default: test-route\n", encoding="utf-8")

        # Unscoped load (no active secret scope)
        with patch("agent.secret_scope.current_secret_scope", return_value=None):
            config_mod._LOAD_CONFIG_CACHE.clear()
            first = config_mod._load_config_impl(want_deepcopy=False)
            assert first["model"]["default"] == "test-route"
            assert len(config_mod._LOAD_CONFIG_CACHE) == 1

            # Scoped load (active secret scope for a profile)
            with patch("agent.secret_scope.current_secret_scope", return_value={"TEST_VAR": "scoped_value"}), \
                 patch("agent.secret_scope.current_secret_scope_home", return_value="/home/user/.hermes/profiles/work"):
                second = config_mod._load_config_impl(want_deepcopy=False)
                assert second["model"]["default"] == "test-route"
                # Both entries must coexist
                assert len(config_mod._LOAD_CONFIG_CACHE) == 2

                # Scoped again — must be a cache hit (no eviction)
                third = config_mod._load_config_impl(want_deepcopy=False)
                assert third is second  # same object from cache

            # Back to unscoped — must still be a cache hit
            fourth = config_mod._load_config_impl(want_deepcopy=False)
            assert fourth is first  # same object from cache


def test_cache_key_includes_scope(tmp_path):
    """_load_config_cache_key must return different keys for scoped vs unscoped."""
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        path_key = str(tmp_path / "config.yaml")

        # Unscoped
        with patch("agent.secret_scope.current_secret_scope", return_value=None):
            key_unscoped = config_mod._load_config_cache_key(path_key)
            assert key_unscoped == (path_key, False, None)

        # Scoped
        with patch("agent.secret_scope.current_secret_scope", return_value={"FOO": "bar"}), \
             patch("agent.secret_scope.current_secret_scope_home", return_value="/home/user/.hermes/profiles/work"):
            key_scoped = config_mod._load_config_cache_key(path_key)
            assert key_scoped == (path_key, True, "/home/user/.hermes/profiles/work")

        # Different scope home = different key
        with patch("agent.secret_scope.current_secret_scope", return_value={"FOO": "bar"}), \
             patch("agent.secret_scope.current_secret_scope_home", return_value="/home/user/.hermes/profiles/personal"):
            key_other = config_mod._load_config_cache_key(path_key)
            assert key_other == (path_key, True, "/home/user/.hermes/profiles/personal")
            assert key_other != key_scoped


def test_alternating_calls_do_not_reparse(tmp_path):
    """The original bug: 200 alternating calls caused 199 parses. With the fix, only 2."""
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_mod._LOAD_CONFIG_CACHE.clear()
        config_mod._RAW_CONFIG_CACHE.clear()
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model:\n  default: test-route\n", encoding="utf-8")

        parse_count = 0
        original_fast_safe_load = config_mod.fast_safe_load

        def counting_fast_safe_load(*args, **kwargs):
            nonlocal parse_count
            parse_count += 1
            return original_fast_safe_load(*args, **kwargs)

        with patch("agent.secret_scope.current_secret_scope", return_value=None), \
             patch.object(config_mod, "fast_safe_load", side_effect=counting_fast_safe_load):
            config_mod._LOAD_CONFIG_CACHE.clear()
            # First unscoped load
            config_mod._load_config_impl(want_deepcopy=False)
            assert parse_count == 1

        # Now alternate scoped/unscoped many times
        with patch("agent.secret_scope.current_secret_scope", return_value={"TEST": "scoped"}), \
             patch("agent.secret_scope.current_secret_scope_home", return_value="/home/user/.hermes/profiles/work"), \
             patch.object(config_mod, "fast_safe_load", side_effect=counting_fast_safe_load):
            config_mod._load_config_impl(want_deepcopy=False)
            assert parse_count == 2  # first scoped load parses

            for _ in range(50):
                config_mod._load_config_impl(want_deepcopy=False)
            # All scoped calls after the first must be cache hits
            assert parse_count == 2

        # Back to unscoped — must be a cache hit (no re-parse)
        with patch("agent.secret_scope.current_secret_scope", return_value=None), \
             patch.object(config_mod, "fast_safe_load", side_effect=counting_fast_safe_load):
            for _ in range(50):
                config_mod._load_config_impl(want_deepcopy=False)
            assert parse_count == 2  # still 2 — no re-parse
