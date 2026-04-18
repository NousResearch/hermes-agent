"""Tests for the rate limit cooldown tracker (rate_limit_cooldown.py)."""

import json
import time
from pathlib import Path

import pytest

from rate_limit_cooldown import (
    DEFAULT_COOLDOWN_SECONDS,
    RATE_LIMIT_DIR_NAME,
    _get_provider_file,
    _load_cache,
    _save_cache,
    clear_all_cooldowns,
    clear_cooldown,
    get_cooldown_remaining,
    is_in_cooldown,
    record_rate_limit,
)


@pytest.fixture
def hermes_home(tmp_path):
    """Provide a temporary Hermes home directory."""
    return str(tmp_path)


class TestProviderFileResolution:
    """Tests for _get_provider_file path logic."""

    def test_simple_provider_name(self, hermes_home, tmp_path):
        path = _get_provider_file("zai", hermes_home)
        expected = tmp_path / RATE_LIMIT_DIR_NAME / "zai.json"
        assert path == expected

    def test_dashes_normalized_to_underscores(self, hermes_home, tmp_path):
        path = _get_provider_file("open-router", hermes_home)
        expected = tmp_path / RATE_LIMIT_DIR_NAME / "open_router.json"
        assert path == expected

    def test_slashes_normalized(self, hermes_home, tmp_path):
        path = _get_provider_file("org/provider", hermes_home)
        expected = tmp_path / RATE_LIMIT_DIR_NAME / "org_provider.json"
        assert path == expected


class TestCachePersistence:
    """Tests for _load_cache and _save_cache file I/O."""

    def test_load_missing_file(self, tmp_path):
        result = _load_cache(tmp_path / "nonexistent.json")
        assert result == {}

    def test_load_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not json{{{", encoding="utf-8")
        result = _load_cache(f)
        assert result == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        f = tmp_path / "test.json"
        data = {"model_a": {"error_type": "rate_limit", "reset_after": 60}}
        _save_cache(f, data)
        loaded = _load_cache(f)
        assert loaded == data

    def test_save_creates_parent_dir(self, tmp_path):
        f = tmp_path / "nested" / "dir" / "test.json"
        _save_cache(f, {"a": 1})
        assert f.exists()

    def test_save_atomic(self, tmp_path):
        """Verify temp file is cleaned up after save."""
        f = tmp_path / "test.json"
        _save_cache(f, {"a": 1})
        # No leftover .tmp file
        assert not f.with_suffix(".tmp").exists()


class TestRecordRateLimit:
    """Tests for record_rate_limit."""

    def test_record_creates_cache_file(self, hermes_home, tmp_path):
        record_rate_limit("zai", "glm-5-turbo", hermes_home=hermes_home)
        cache_file = tmp_path / RATE_LIMIT_DIR_NAME / "zai.json"
        assert cache_file.exists()
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "glm-5-turbo" in cache

    def test_record_stores_metadata(self, hermes_home, tmp_path):
        record_rate_limit(
            "zai", "glm-5-turbo",
            error_type="rate_limit",
            reset_after=120,
            http_status=429,
            hermes_home=hermes_home,
        )
        cache_file = tmp_path / RATE_LIMIT_DIR_NAME / "zai.json"
        entry = json.loads(cache_file.read_text(encoding="utf-8"))["glm-5-turbo"]
        assert entry["error_type"] == "rate_limit"
        assert entry["reset_after"] == 120
        assert entry["http_status"] == 429
        assert "timestamp" in entry

    def test_record_strips_provider_prefix(self, hermes_home, tmp_path):
        record_rate_limit("zai", "zai/glm-5-turbo", hermes_home=hermes_home)
        cache_file = tmp_path / RATE_LIMIT_DIR_NAME / "zai.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        # Key should be "glm-5-turbo", not "zai/glm-5-turbo"
        assert "glm-5-turbo" in cache
        assert "zai/glm-5-turbo" not in cache

    def test_record_default_cooldown(self, hermes_home, tmp_path):
        record_rate_limit("openai", "gpt-4o", hermes_home=hermes_home)
        cache_file = tmp_path / RATE_LIMIT_DIR_NAME / "openai.json"
        entry = json.loads(cache_file.read_text(encoding="utf-8"))["gpt-4o"]
        assert entry["reset_after"] == DEFAULT_COOLDOWN_SECONDS

    def test_record_multiple_models(self, hermes_home, tmp_path):
        record_rate_limit("zai", "glm-5-turbo", hermes_home=hermes_home)
        record_rate_limit("zai", "glm-4", hermes_home=hermes_home)
        cache_file = tmp_path / RATE_LIMIT_DIR_NAME / "zai.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "glm-5-turbo" in cache
        assert "glm-4" in cache

    def test_record_overwrites_existing(self, hermes_home, tmp_path):
        record_rate_limit("zai", "glm-5-turbo", reset_after=60, hermes_home=hermes_home)
        record_rate_limit("zai", "glm-5-turbo", reset_after=120, hermes_home=hermes_home)
        cache_file = tmp_path / RATE_LIMIT_DIR_NAME / "zai.json"
        entry = json.loads(cache_file.read_text(encoding="utf-8"))["glm-5-turbo"]
        assert entry["reset_after"] == 120


class TestIsInCooldown:
    """Tests for is_in_cooldown."""

    def test_not_recorded(self, hermes_home):
        assert not is_in_cooldown("zai", "glm-5-turbo", hermes_home=hermes_home)

    def test_in_cooldown(self, hermes_home):
        record_rate_limit("zai", "glm-5-turbo", reset_after=60, hermes_home=hermes_home)
        assert is_in_cooldown("zai", "glm-5-turbo", hermes_home=hermes_home)

    def test_cooldown_expires(self, hermes_home):
        # Use a very short cooldown and wait
        record_rate_limit("zai", "glm-5-turbo", reset_after=1, hermes_home=hermes_home)
        time.sleep(1.1)
        assert not is_in_cooldown("zai", "glm-5-turbo", hermes_home=hermes_home)

    def test_expired_entry_cleaned(self, hermes_home, tmp_path):
        record_rate_limit("zai", "glm-5-turbo", reset_after=1, hermes_home=hermes_home)
        time.sleep(1.1)
        is_in_cooldown("zai", "glm-5-turbo", hermes_home=hermes_home)
        cache_file = tmp_path / RATE_LIMIT_DIR_NAME / "zai.json"
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "glm-5-turbo" not in cache

    def test_different_model_not_affected(self, hermes_home):
        record_rate_limit("zai", "glm-5-turbo", reset_after=60, hermes_home=hermes_home)
        assert not is_in_cooldown("zai", "glm-4", hermes_home=hermes_home)

    def test_different_provider_not_affected(self, hermes_home):
        record_rate_limit("zai", "glm-5-turbo", reset_after=60, hermes_home=hermes_home)
        assert not is_in_cooldown("openai", "glm-5-turbo", hermes_home=hermes_home)

    def test_strips_provider_prefix(self, hermes_home):
        record_rate_limit("zai", "zai/glm-5-turbo", reset_after=60, hermes_home=hermes_home)
        assert is_in_cooldown("zai", "glm-5-turbo", hermes_home=hermes_home)


class TestGetCooldownRemaining:
    """Tests for get_cooldown_remaining."""

    def test_not_recorded(self, hermes_home):
        assert get_cooldown_remaining("zai", "glm-5-turbo", hermes_home=hermes_home) == 0

    def test_returns_remaining_seconds(self, hermes_home):
        record_rate_limit("zai", "glm-5-turbo", reset_after=60, hermes_home=hermes_home)
        remaining = get_cooldown_remaining("zai", "glm-5-turbo", hermes_home=hermes_home)
        assert 0 < remaining <= 60

    def test_returns_zero_after_expiry(self, hermes_home):
        record_rate_limit("zai", "glm-5-turbo", reset_after=1, hermes_home=hermes_home)
        time.sleep(1.1)
        assert get_cooldown_remaining("zai", "glm-5-turbo", hermes_home=hermes_home) == 0

    def test_strips_provider_prefix(self, hermes_home):
        record_rate_limit("zai", "zai/glm-5-turbo", reset_after=60, hermes_home=hermes_home)
        assert get_cooldown_remaining("zai", "glm-5-turbo", hermes_home=hermes_home) > 0


class TestClearCooldown:
    """Tests for clear_cooldown."""

    def test_clear_existing(self, hermes_home):
        record_rate_limit("zai", "glm-5-turbo", reset_after=60, hermes_home=hermes_home)
        assert clear_cooldown("zai", "glm-5-turbo", hermes_home=hermes_home) is True
        assert not is_in_cooldown("zai", "glm-5-turbo", hermes_home=hermes_home)

    def test_clear_nonexistent(self, hermes_home):
        assert clear_cooldown("zai", "glm-5-turbo", hermes_home=hermes_home) is False

    def test_strips_provider_prefix(self, hermes_home):
        record_rate_limit("zai", "zai/glm-5-turbo", reset_after=60, hermes_home=hermes_home)
        assert clear_cooldown("zai", "glm-5-turbo", hermes_home=hermes_home) is True


class TestClearAllCooldowns:
    """Tests for clear_all_cooldowns."""

    def test_clear_multiple_providers(self, hermes_home, tmp_path):
        record_rate_limit("zai", "glm-5-turbo", hermes_home=hermes_home)
        record_rate_limit("openai", "gpt-4o", hermes_home=hermes_home)
        count = clear_all_cooldowns(hermes_home=hermes_home)
        assert count == 2
        assert not (tmp_path / RATE_LIMIT_DIR_NAME / "zai.json").exists()
        assert not (tmp_path / RATE_LIMIT_DIR_NAME / "openai.json").exists()

    def test_clear_empty_dir(self, hermes_home):
        assert clear_all_cooldowns(hermes_home=hermes_home) == 0

    def test_clear_nonexistent_dir(self, hermes_home):
        assert clear_all_cooldowns(hermes_home=hermes_home) == 0
