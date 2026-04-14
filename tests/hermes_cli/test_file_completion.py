"""Tests for async project file scanning in @-completion."""

import os
import threading
import time
from unittest.mock import patch

import pytest

from hermes_cli.commands import SlashCommandCompleter
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

# ---------------------------------------------------------------------------
# Async project file scanning
# ---------------------------------------------------------------------------

class TestAsyncProjectFiles:
    """Tests for _get_project_files background refresh behaviour."""

    def test_cache_hit_returns_immediately(self):
        """Within TTL, _get_project_files should return cached data."""
        completer = SlashCommandCompleter()
        completer._file_cache = ["a.py", "b.py"]
        completer._file_cache_time = time.monotonic()
        completer._file_cache_cwd = os.getcwd()

        result = completer._get_project_files()
        assert result == ["a.py", "b.py"]
        assert completer._file_refreshing is False

    def test_stale_cache_triggers_background_refresh(self):
        """When cache expires, a background thread should start."""
        completer = SlashCommandCompleter()
        completer._file_cache = ["old.py"]
        completer._file_cache_time = time.monotonic() - 10.0  # expired
        completer._file_cache_cwd = os.getcwd()

        with patch.object(completer, "_refresh_file_cache") as mock_refresh:
            result = completer._get_project_files()

        # Should return stale data immediately
        assert result == ["old.py"]
        # Should have started a background refresh
        mock_refresh.assert_called_once()

    def test_stale_cache_returns_empty_when_cwd_changed(self):
        """When cwd changes and cache is stale, return empty list."""
        completer = SlashCommandCompleter()
        completer._file_cache = ["old.py"]
        completer._file_cache_time = time.monotonic() - 10.0
        completer._file_cache_cwd = "/some/other/dir"

        with patch.object(completer, "_refresh_file_cache") as mock_refresh:
            result = completer._get_project_files()

        assert result == []
        mock_refresh.assert_called_once()

    def test_refreshing_flag_prevents_duplicate_threads(self):
        """While a refresh is in progress, don't spawn another thread."""
        completer = SlashCommandCompleter()
        completer._file_cache = []
        completer._file_cache_time = 0.0
        completer._file_cache_cwd = os.getcwd()
        completer._file_refreshing = True  # simulate in-progress refresh

        with patch.object(completer, "_refresh_file_cache") as mock_refresh:
            result = completer._get_project_files()

        # Should return stale (empty) but NOT start another refresh
        mock_refresh.assert_not_called()

    def test_refresh_clears_refreshing_flag(self, tmp_path):
        """After _refresh_file_cache completes, _file_refreshing should be False."""
        completer = SlashCommandCompleter()
        completer._file_cache = []
        completer._file_cache_time = 0.0
        completer._file_cache_cwd = str(tmp_path)
        completer._file_refreshing = True

        completer._refresh_file_cache(str(tmp_path), time.monotonic())

        assert completer._file_refreshing is False
        # Cache should now contain at least the test file listing
        assert isinstance(completer._file_cache, list)

    def test_refresh_updates_cache_and_timestamp(self, tmp_path):
        """_refresh_file_cache should populate cache and update timestamp."""
        completer = SlashCommandCompleter()
        start_time = time.monotonic() - 1.0

        completer._refresh_file_cache(str(tmp_path), start_time)

        assert completer._file_cache_time == start_time
        assert completer._file_cache_cwd == str(tmp_path)

    def test_concurrent_calls_return_stale_data(self):
        """Multiple rapid calls should all return stale data, not block."""
        completer = SlashCommandCompleter()
        completer._file_cache = ["cached.py"]
        completer._file_cache_time = time.monotonic() - 10.0  # expired
        completer._file_cache_cwd = os.getcwd()

        results = []
        with patch.object(completer, "_refresh_file_cache", wraps=lambda *a: (
            time.sleep(0.1),
            SlashCommandCompleter._refresh_file_cache(completer, *a),
        )):
            for _ in range(5):
                results.append(completer._get_project_files())

        # All calls should return the stale cache, not block
        for r in results:
            assert r == ["cached.py"]

    def test_file_refreshing_init_false(self):
        """_file_refreshing should be False on a fresh instance."""
        completer = SlashCommandCompleter()
        assert completer._file_refreshing is False
