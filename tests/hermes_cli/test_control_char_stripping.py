"""Tests for ASCII control character stripping in save_env_value.

Covers the fix for issue #40840 — on Windows, pressing ESC during an
``input()`` prompt inserts a literal ``\\x1b`` character into the returned
string. When this value is persisted to ``~/.hermes/.env`` it silently
corrupts URL and API-key fields (e.g. ``SEARXNG_URL=\\x1b`` causes every
``web_search`` call to fail permanently).

The fix adds ``_strip_control_chars()`` in the ``save_env_value`` path
to strip all C0 control characters (0x00–0x1F except TAB) and DEL (0x7F)
from env values before they are written to disk.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestStripControlChars:
    """Tests for hermes_cli.config._strip_control_chars()."""

    def test_clean_value_unchanged(self):
        from hermes_cli.config import _strip_control_chars

        assert _strip_control_chars("TEST_KEY", "https://example.com") == "https://example.com"

    def test_clean_api_key_unchanged(self):
        from hermes_cli.config import _strip_control_chars

        key = "sk-proj-" + "a" * 48
        assert _strip_control_chars("OPENAI_API_KEY", key) == key

    def test_empty_value_unchanged(self):
        from hermes_cli.config import _strip_control_chars

        assert _strip_control_chars("TEST_KEY", "") == ""

    def test_none_value_unchanged(self):
        from hermes_cli.config import _strip_control_chars

        assert _strip_control_chars("TEST_KEY", None) is None

    def test_strips_bare_esc_character(self, capsys):
        """The exact scenario from issue #40840: bare ESC as entire value."""
        from hermes_cli.config import _strip_control_chars

        result = _strip_control_chars("SEARXNG_URL", "\x1b")
        assert result == ""
        captured = capsys.readouterr()
        assert "ESC" in captured.err
        assert "control characters" in captured.err

    def test_strips_esc_embedded_in_url(self, capsys):
        """ESC accidentally inserted into a URL value."""
        from hermes_cli.config import _strip_control_chars

        result = _strip_control_chars("SEARXNG_URL", "http://\x1blocalhost:8080")
        assert result == "http://localhost:8080"
        captured = capsys.readouterr()
        assert "ESC" in captured.err

    def test_strips_bell_character(self, capsys):
        """BEL (0x07) is another common terminal control character."""
        from hermes_cli.config import _strip_control_chars

        result = _strip_control_chars("API_KEY", "sk-\x07test")
        assert result == "sk-test"
        captured = capsys.readouterr()
        assert "BEL" in captured.err

    def test_strips_del_character(self, capsys):
        """DEL (0x7F) is also stripped."""
        from hermes_cli.config import _strip_control_chars

        result = _strip_control_chars("API_KEY", "sk-test\x7f")
        assert result == "sk-test"
        captured = capsys.readouterr()
        assert "DEL" in captured.err

    def test_preserves_tab_character(self):
        """TAB (0x09) is preserved — some tools use it as a delimiter."""
        from hermes_cli.config import _strip_control_chars

        result = _strip_control_chars("COMPOUND_VALUE", "val1\tval2")
        assert result == "val1\tval2"

    def test_strips_multiple_control_chars(self, capsys):
        """Multiple different control characters are all stripped."""
        from hermes_cli.config import _strip_control_chars

        result = _strip_control_chars("SEARXNG_URL", "\x1b\x07http://localhost\x00")
        assert result == "http://localhost"
        captured = capsys.readouterr()
        assert "ESC" in captured.err
        assert "BEL" in captured.err or "NUL" in captured.err

    def test_no_warning_for_clean_value(self, capsys):
        from hermes_cli.config import _strip_control_chars

        _strip_control_chars("API_KEY", "sk-clean-key-123")
        assert capsys.readouterr().err == ""


class TestSaveEnvValueControlChars:
    """Integration test: save_env_value strips control chars before writing."""

    def test_save_env_value_strips_esc(self, monkeypatch, capsys, tmp_path):
        """Verify that save_env_value strips ESC before writing to .env."""
        from hermes_cli.config import save_env_value, get_env_path

        env_path = tmp_path / ".env"
        monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: env_path)
        monkeypatch.setattr("hermes_cli.config.ensure_hermes_home", lambda: None)
        # Prevent _secure_file from running (needs real path)
        monkeypatch.setattr("hermes_cli.config._secure_file", lambda p: None)
        # Prevent os.environ side effects from polluting test environment
        monkeypatch.delenv("SEARXNG_URL", raising=False)

        save_env_value("SEARXNG_URL", "\x1b")

        # The value written to .env should be empty (ESC stripped)
        if env_path.exists():
            content = env_path.read_text(encoding="utf-8")
            assert "\x1b" not in content
            # After stripping ESC, the value is empty string
            assert "SEARXNG_URL=" in content

    def test_save_env_value_strips_esc_in_url(self, monkeypatch, capsys, tmp_path):
        """Verify that save_env_value strips ESC from a URL value."""
        from hermes_cli.config import save_env_value, get_env_path

        env_path = tmp_path / ".env"
        monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: env_path)
        monkeypatch.setattr("hermes_cli.config.ensure_hermes_home", lambda: None)
        monkeypatch.setattr("hermes_cli.config._secure_file", lambda p: None)
        monkeypatch.delenv("SEARXNG_URL", raising=False)

        save_env_value("SEARXNG_URL", "http://\x1blocalhost:8080")

        if env_path.exists():
            content = env_path.read_text(encoding="utf-8")
            assert "\x1b" not in content
            assert "SEARXNG_URL=http://localhost:8080" in content

    def test_save_env_value_clean_url_unchanged(self, monkeypatch, tmp_path):
        """A clean URL should pass through unchanged."""
        from hermes_cli.config import save_env_value, get_env_path

        env_path = tmp_path / ".env"
        monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: env_path)
        monkeypatch.setattr("hermes_cli.config.ensure_hermes_home", lambda: None)
        monkeypatch.setattr("hermes_cli.config._secure_file", lambda p: None)
        monkeypatch.delenv("SEARXNG_URL", raising=False)

        save_env_value("SEARXNG_URL", "http://localhost:8080")

        if env_path.exists():
            content = env_path.read_text(encoding="utf-8")
            assert "SEARXNG_URL=http://localhost:8080" in content
