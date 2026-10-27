"""#141856: require_readable_config_before_write must cache successful parses to avoid
re-parsing config.yaml on every call from plugin tool hooks.
"""
from unittest.mock import patch

from hermes_cli import config as config_mod
from hermes_cli import config_readable_cache as cache_mod


def test_repeated_calls_use_cache(tmp_path):
    """Second call with the same file must be a cache hit (no re-parse)."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  default: test-route\n", encoding="utf-8")

    cache_mod.clear_readable_config_cache()
    parse_count = 0
    original_read_config_doc = config_mod.read_config_doc

    def counting_read_config_doc(*args, **kwargs):
        nonlocal parse_count
        parse_count += 1
        return original_read_config_doc(*args, **kwargs)

    with patch.object(config_mod, "read_config_doc", side_effect=counting_read_config_doc), \
         patch.object(cache_mod, "mtime_is_racy", return_value=False):
        result1 = config_mod.require_readable_config_before_write(cfg)
        assert result1["model"]["default"] == "test-route"
        assert parse_count == 1

        result2 = config_mod.require_readable_config_before_write(cfg)
        assert result2["model"]["default"] == "test-route"
        assert parse_count == 1  # cache hit — no re-parse


def test_file_edit_invalidates_cache(tmp_path):
    """Editing the file must invalidate the cache."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  default: aaaa\n", encoding="utf-8")

    cache_mod.clear_readable_config_cache()
    result1 = config_mod.require_readable_config_before_write(cfg)
    assert result1["model"]["default"] == "aaaa"

    # Edit the file
    cfg.write_text("model:\n  default: bbbb\n", encoding="utf-8")
    result2 = config_mod.require_readable_config_before_write(cfg)
    assert result2["model"]["default"] == "bbbb"


def test_clear_cache_works(tmp_path):
    """clear_readable_config_cache() must reset the cache."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  default: test\n", encoding="utf-8")

    cache_mod.clear_readable_config_cache()
    with patch.object(cache_mod, "mtime_is_racy", return_value=False):
        config_mod.require_readable_config_before_write(cfg)
    assert len(cache_mod._CACHE) == 1

    cache_mod.clear_readable_config_cache()
    assert len(cache_mod._CACHE) == 0


def test_each_call_returns_fresh_dict(tmp_path):
    """Mutating a returned dict must not poison the cache — including the call that cached it."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  default: test\n", encoding="utf-8")

    cache_mod.clear_readable_config_cache()
    with patch.object(cache_mod, "mtime_is_racy", return_value=False):
        result1 = config_mod.require_readable_config_before_write(cfg)
        assert len(cache_mod._CACHE) == 1  # this call cached its parse
        result1["model"]["default"] = "MUTATED"
        result1["new_key"] = "new_value"

        result2 = config_mod.require_readable_config_before_write(cfg)
        assert result2["model"]["default"] == "test"
        assert "new_key" not in result2


def test_cache_holds_one_entry_per_path(tmp_path):
    """Repeated edits keep a single entry per path, so a long-lived process cannot grow the cache."""
    cfg = tmp_path / "config.yaml"
    cache_mod.clear_readable_config_cache()
    with patch.object(cache_mod, "mtime_is_racy", return_value=False):
        for value in ("one", "two", "three"):
            cfg.write_text(f"model:\n  default: {value}\n", encoding="utf-8")
            assert config_mod.require_readable_config_before_write(cfg)["model"]["default"] == value
            assert len(cache_mod._CACHE) == 1


def test_failure_is_not_cached(tmp_path):
    """A malformed file must not be cached — every call re-validates."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model:\n  default: test\n", encoding="utf-8")

    cache_mod.clear_readable_config_cache()
    # First call succeeds
    result1 = config_mod.require_readable_config_before_write(cfg)
    assert result1["model"]["default"] == "test"

    # Corrupt the file
    cfg.write_text("invalid: yaml: content: [\n", encoding="utf-8")
    # Every call must re-validate and raise
    for _ in range(3):
        try:
            config_mod.require_readable_config_before_write(cfg)
            assert False, "Should have raised"
        except Exception:
            pass

    # Fix the file — must work again
    cfg.write_text("model:\n  default: fixed\n", encoding="utf-8")
    result2 = config_mod.require_readable_config_before_write(cfg)
    assert result2["model"]["default"] == "fixed"


def test_missing_file_returns_empty_dict(tmp_path):
    """A missing file returns {} and is not cached."""
    cfg = tmp_path / "nonexistent.yaml"
    cache_mod.clear_readable_config_cache()
    result = config_mod.require_readable_config_before_write(cfg)
    assert result == {}
    assert len(cache_mod._CACHE) == 0
