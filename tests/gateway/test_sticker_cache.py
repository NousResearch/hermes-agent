"""Tests for gateway/sticker_cache.py — sticker description cache."""

import json
import time
from unittest.mock import patch

from gateway.sticker_cache import (
    _load_cache,
    _save_cache,
    get_cached_description,
    cache_sticker_description,
    build_sticker_injection,
    build_animated_sticker_injection,
    STICKER_VISION_PROMPT,
)


class TestLoadSaveCache:
    def test_load_missing_file(self, tmp_path):
        with patch("gateway.sticker_cache.CACHE_PATH", tmp_path / "nope.json"):
            assert _load_cache() == {}

    def test_load_corrupt_file(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json{{{")
        with patch("gateway.sticker_cache.CACHE_PATH", bad_file):
            assert _load_cache() == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        data = {"abc123": {"description": "A cat", "emoji": "", "set_name": "", "cached_at": 1.0}}
        with patch("gateway.sticker_cache.CACHE_PATH", cache_file):
            _save_cache(data)
            loaded = _load_cache()
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        cache_file = tmp_path / "sub" / "dir" / "cache.json"
        with patch("gateway.sticker_cache.CACHE_PATH", cache_file):
            _save_cache({"key": "value"})
        assert cache_file.exists()


class TestCacheSticker:
    def test_cache_and_retrieve(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        with patch("gateway.sticker_cache.CACHE_PATH", cache_file):
            cache_sticker_description("uid_1", "A happy dog", emoji="🐕", set_name="Dogs")
            result = get_cached_description("uid_1")

        assert result is not None
        assert result["description"] == "A happy dog"
        assert result["emoji"] == "🐕"
        assert result["set_name"] == "Dogs"
        assert "cached_at" in result

    def test_missing_sticker_returns_none(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        with patch("gateway.sticker_cache.CACHE_PATH", cache_file):
            result = get_cached_description("nonexistent")
        assert result is None

    def test_overwrite_existing(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        with patch("gateway.sticker_cache.CACHE_PATH", cache_file):
            cache_sticker_description("uid_1", "Old description")
            cache_sticker_description("uid_1", "New description")
            result = get_cached_description("uid_1")

        assert result["description"] == "New description"

    def test_multiple_stickers(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        with patch("gateway.sticker_cache.CACHE_PATH", cache_file):
            cache_sticker_description("uid_1", "Cat")
            cache_sticker_description("uid_2", "Dog")
            r1 = get_cached_description("uid_1")
            r2 = get_cached_description("uid_2")

        assert r1["description"] == "Cat"
        assert r2["description"] == "Dog"


class TestBuildStickerInjection:
    def test_basic_injection(self):
        result = build_sticker_injection("A cat waving")
        assert "A cat waving" in result
        assert "sticker" in result.lower()

    def test_with_emoji(self):
        result = build_sticker_injection("A cat", emoji="😀")
        assert "😀" in result

    def test_with_emoji_and_set_name(self):
        result = build_sticker_injection("A cat", emoji="😀", set_name="MyPack")
        assert "😀" in result
        assert "MyPack" in result

    def test_no_emoji_no_set_name(self):
        result = build_sticker_injection("A cat")
        assert "A cat" in result
        assert "sticker" in result.lower()


class TestBuildAnimatedStickerInjection:
    def test_with_emoji(self):
        result = build_animated_sticker_injection(emoji="🎉")
        assert "animated sticker" in result.lower()
        assert "🎉" in result

    def test_without_emoji(self):
        result = build_animated_sticker_injection()
        assert "animated sticker" in result.lower()
        assert "can't see" in result.lower()


class TestVisionPrompt:
    def test_prompt_exists(self):
        assert len(STICKER_VISION_PROMPT) > 0
        assert "sticker" in STICKER_VISION_PROMPT.lower()
