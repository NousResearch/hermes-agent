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
    def test_exact_format_no_context(self):
        result = build_sticker_injection("A cat waving")
        assert result == '[The user sent a sticker~ It shows: "A cat waving" (=^.w.^=)]'

    def test_exact_format_emoji_only(self):
        result = build_sticker_injection("A cat", emoji="😀")
        assert result == '[The user sent a sticker 😀~ It shows: "A cat" (=^.w.^=)]'

    def test_exact_format_emoji_and_set_name(self):
        result = build_sticker_injection("A cat", emoji="😀", set_name="MyPack")
        assert result == '[The user sent a sticker 😀 from "MyPack"~ It shows: "A cat" (=^.w.^=)]'

    def test_set_name_without_emoji_ignored(self):
        """set_name alone (no emoji) produces no context — only emoji+set_name triggers 'from' clause."""
        result = build_sticker_injection("A cat", set_name="MyPack")
        assert result == '[The user sent a sticker~ It shows: "A cat" (=^.w.^=)]'
        assert "MyPack" not in result

    def test_description_with_quotes(self):
        result = build_sticker_injection('A "happy" dog')
        assert '"A \\"happy\\" dog"' not in result  # no escaping happens
        assert 'A "happy" dog' in result

    def test_empty_description(self):
        result = build_sticker_injection("")
        assert result == '[The user sent a sticker~ It shows: "" (=^.w.^=)]'


class TestBuildAnimatedStickerInjection:
    def test_exact_format_with_emoji(self):
        result = build_animated_sticker_injection(emoji="🎉")
        assert result == (
            "[The user sent an animated sticker 🎉~ "
            "I can't see animated ones yet, but the emoji suggests: 🎉]"
        )

    def test_exact_format_without_emoji(self):
        result = build_animated_sticker_injection()
        assert result == "[The user sent an animated sticker~ I can't see animated ones yet]"

    def test_empty_emoji_same_as_no_emoji(self):
        result = build_animated_sticker_injection(emoji="")
        assert result == build_animated_sticker_injection()
