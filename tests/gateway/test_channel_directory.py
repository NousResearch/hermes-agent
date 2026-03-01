"""Tests for gateway/channel_directory.py — channel resolution and display."""

import json
from pathlib import Path
from unittest.mock import patch

from gateway.channel_directory import (
    resolve_channel_name,
    format_directory_for_display,
    load_directory,
    _build_from_sessions,
    DIRECTORY_PATH,
)


def _write_directory(tmp_path, platforms):
    """Helper to write a fake channel directory."""
    data = {"updated_at": "2026-01-01T00:00:00", "platforms": platforms}
    cache_file = tmp_path / "channel_directory.json"
    cache_file.write_text(json.dumps(data))
    return cache_file


class TestLoadDirectory:
    def test_missing_file(self, tmp_path):
        with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "nope.json"):
            result = load_directory()
        assert result["updated_at"] is None
        assert result["platforms"] == {}

    def test_valid_file(self, tmp_path):
        cache_file = _write_directory(tmp_path, {
            "telegram": [{"id": "123", "name": "John", "type": "dm"}]
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = load_directory()
        assert result["platforms"]["telegram"][0]["name"] == "John"

    def test_corrupt_file(self, tmp_path):
        cache_file = tmp_path / "channel_directory.json"
        cache_file.write_text("{bad json")
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = load_directory()
        assert result["updated_at"] is None


class TestResolveChannelName:
    def _setup(self, tmp_path, platforms):
        cache_file = _write_directory(tmp_path, platforms)
        return patch("gateway.channel_directory.DIRECTORY_PATH", cache_file)

    def test_exact_match(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "111", "name": "bot-home", "guild": "MyServer", "type": "channel"},
                {"id": "222", "name": "general", "guild": "MyServer", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("discord", "bot-home") == "111"
            assert resolve_channel_name("discord", "#bot-home") == "111"

    def test_case_insensitive(self, tmp_path):
        platforms = {
            "slack": [{"id": "C01", "name": "Engineering", "type": "channel"}]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("slack", "engineering") == "C01"
            assert resolve_channel_name("slack", "ENGINEERING") == "C01"

    def test_guild_qualified_match(self, tmp_path):
        platforms = {
            "discord": [
                {"id": "111", "name": "general", "guild": "ServerA", "type": "channel"},
                {"id": "222", "name": "general", "guild": "ServerB", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("discord", "ServerA/general") == "111"
            assert resolve_channel_name("discord", "ServerB/general") == "222"

    def test_prefix_match_unambiguous(self, tmp_path):
        platforms = {
            "slack": [
                {"id": "C01", "name": "engineering-backend", "type": "channel"},
                {"id": "C02", "name": "design-team", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            # "engineering" prefix matches only one channel
            assert resolve_channel_name("slack", "engineering") == "C01"

    def test_prefix_match_ambiguous_returns_none(self, tmp_path):
        platforms = {
            "slack": [
                {"id": "C01", "name": "eng-backend", "type": "channel"},
                {"id": "C02", "name": "eng-frontend", "type": "channel"},
            ]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("slack", "eng") is None

    def test_no_channels_returns_none(self, tmp_path):
        with self._setup(tmp_path, {}):
            assert resolve_channel_name("telegram", "someone") is None

    def test_no_match_returns_none(self, tmp_path):
        platforms = {
            "telegram": [{"id": "123", "name": "John", "type": "dm"}]
        }
        with self._setup(tmp_path, platforms):
            assert resolve_channel_name("telegram", "nonexistent") is None


class TestBuildFromSessions:
    def _write_sessions(self, tmp_path, sessions_data):
        """Write sessions.json at the path _build_from_sessions expects."""
        sessions_path = tmp_path / ".hermes" / "sessions" / "sessions.json"
        sessions_path.parent.mkdir(parents=True)
        sessions_path.write_text(json.dumps(sessions_data))

    def test_builds_from_sessions_json(self, tmp_path):
        self._write_sessions(tmp_path, {
            "session_1": {
                "origin": {
                    "platform": "telegram",
                    "chat_id": "12345",
                    "chat_name": "Alice",
                },
                "chat_type": "dm",
            },
            "session_2": {
                "origin": {
                    "platform": "telegram",
                    "chat_id": "67890",
                    "user_name": "Bob",
                },
                "chat_type": "group",
            },
            "session_3": {
                "origin": {
                    "platform": "discord",
                    "chat_id": "99999",
                },
            },
        })

        with patch.object(Path, "home", return_value=tmp_path):
            entries = _build_from_sessions("telegram")

        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert "Alice" in names
        assert "Bob" in names

    def test_missing_sessions_file(self, tmp_path):
        with patch.object(Path, "home", return_value=tmp_path):
            entries = _build_from_sessions("telegram")
        assert entries == []

    def test_deduplication_by_chat_id(self, tmp_path):
        self._write_sessions(tmp_path, {
            "s1": {"origin": {"platform": "telegram", "chat_id": "123", "chat_name": "X"}},
            "s2": {"origin": {"platform": "telegram", "chat_id": "123", "chat_name": "X"}},
        })

        with patch.object(Path, "home", return_value=tmp_path):
            entries = _build_from_sessions("telegram")

        assert len(entries) == 1


class TestFormatDirectoryForDisplay:
    def test_empty_directory(self, tmp_path):
        with patch("gateway.channel_directory.DIRECTORY_PATH", tmp_path / "nope.json"):
            result = format_directory_for_display()
        assert "No messaging platforms" in result

    def test_telegram_display(self, tmp_path):
        cache_file = _write_directory(tmp_path, {
            "telegram": [
                {"id": "123", "name": "Alice", "type": "dm"},
                {"id": "456", "name": "Dev Group", "type": "group"},
            ]
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = format_directory_for_display()

        assert "Telegram:" in result
        assert "telegram:Alice" in result
        assert "telegram:Dev Group" in result

    def test_discord_grouped_by_guild(self, tmp_path):
        cache_file = _write_directory(tmp_path, {
            "discord": [
                {"id": "1", "name": "general", "guild": "Server1", "type": "channel"},
                {"id": "2", "name": "bot-home", "guild": "Server1", "type": "channel"},
                {"id": "3", "name": "chat", "guild": "Server2", "type": "channel"},
            ]
        })
        with patch("gateway.channel_directory.DIRECTORY_PATH", cache_file):
            result = format_directory_for_display()

        assert "Discord (Server1):" in result
        assert "Discord (Server2):" in result
        assert "discord:#general" in result
