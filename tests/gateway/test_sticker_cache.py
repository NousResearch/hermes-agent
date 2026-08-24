"""Tests for gateway/sticker_cache.py — sticker description cache."""

from unittest.mock import patch

from gateway.sticker_cache import (
    _load_cache,
    _save_cache,
    get_cached_description,
    cache_sticker_description,
    build_sticker_injection,
    build_animated_sticker_injection,
)


class TestLoadSaveCache:

    def test_load_corrupt_file(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json{{{")
        with patch("gateway.sticker_cache.CACHE_PATH", bad_file):
            assert _load_cache() == {}


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


class TestBuildStickerInjection:
    def test_exact_format_no_context(self):
        result = build_sticker_injection("A cat waving")
        assert result == '[The user sent a sticker~ It shows: "A cat waving" (=^.w.^=)]'


    def test_set_name_without_emoji_ignored(self):
        """set_name alone (no emoji) produces no context — only emoji+set_name triggers 'from' clause."""
        result = build_sticker_injection("A cat", set_name="MyPack")
        assert result == '[The user sent a sticker~ It shows: "A cat" (=^.w.^=)]'
        assert "MyPack" not in result


class TestBuildAnimatedStickerInjection:
    def test_exact_format_with_emoji(self):
        result = build_animated_sticker_injection(emoji="🎉")
        assert result == (
            "[The user sent an animated sticker 🎉~ "
            "I can't see animated ones yet, but the emoji suggests: 🎉]"
        )


