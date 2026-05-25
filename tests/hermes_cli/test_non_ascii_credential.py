"""Tests for non-ASCII credential detection and sanitization.

Covers the fix for issue #6843 — API keys containing Unicode lookalike
characters (e.g. ʋ U+028B instead of v) cause UnicodeEncodeError when
httpx tries to encode the Authorization header as ASCII.
"""

import os
import sys
import tempfile

import pytest

from hermes_cli.config import _check_non_ascii_credential


class TestCheckNonAsciiCredential:
    """Tests for _check_non_ascii_credential()."""

    def test_ascii_key_unchanged(self):
        key = "sk-proj-" + "a" * 100
        result = _check_non_ascii_credential("TEST_API_KEY", key)
        assert result == key

    def test_strips_unicode_v_lookalike(self, capsys):
        """The exact scenario from issue #6843: ʋ instead of v."""
        key = "sk-proj-abc" + "ʋ" + "def"  # \u028b
        result = _check_non_ascii_credential("OPENROUTER_API_KEY", key)
        assert result == "sk-proj-abcdef"
        assert "ʋ" not in result
        # Should print a warning
        captured = capsys.readouterr()
        assert "non-ASCII" in captured.err

    def test_strips_multiple_non_ascii(self, capsys):
        key = "sk-proj-aʋbécd"
        result = _check_non_ascii_credential("OPENAI_API_KEY", key)
        assert result == "sk-proj-abcd"
        captured = capsys.readouterr()
        assert "U+028B" in captured.err  # reports the char

    def test_empty_key(self):
        result = _check_non_ascii_credential("TEST_KEY", "")
        assert result == ""

    def test_all_ascii_no_warning(self, capsys):
        result = _check_non_ascii_credential("KEY", "all-ascii-value-123")
        assert result == "all-ascii-value-123"
        captured = capsys.readouterr()
        assert captured.err == ""


class TestEnvLoaderSanitization:
    """Tests for _sanitize_loaded_credentials in env_loader."""

    def test_strips_non_ascii_from_api_key(self, monkeypatch):
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("OPENROUTER_API_KEY")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-proj-abcʋdef")
        _sanitize_loaded_credentials()
        assert os.environ["OPENROUTER_API_KEY"] == "sk-proj-abcdef"

    def test_strips_non_ascii_from_token(self, monkeypatch):
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("DISCORD_BOT_TOKEN")
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "tokénvalue")
        _sanitize_loaded_credentials()
        assert os.environ["DISCORD_BOT_TOKEN"] == "toknvalue"

    def test_ignores_non_credential_vars(self, monkeypatch):
        from hermes_cli.env_loader import _sanitize_loaded_credentials

        monkeypatch.setenv("MY_UNICODE_VAR", "héllo wörld")
        _sanitize_loaded_credentials()
        # Not a credential suffix — should be left alone
        assert os.environ["MY_UNICODE_VAR"] == "héllo wörld"

    def test_ascii_credentials_untouched(self, monkeypatch):
        from hermes_cli.env_loader import _sanitize_loaded_credentials

        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-allascii123")
        _sanitize_loaded_credentials()
        assert os.environ["OPENAI_API_KEY"] == "sk-proj-allascii123"

    def test_warns_to_stderr_when_stripping(self, monkeypatch, capsys):
        """Silent stripping masks bad keys as opaque provider 400s (see #6843 fallout).

        Users must be told when a copy-paste artifact was removed so they
        can re-copy the key if authentication fails.
        """
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("GOOGLE_API_KEY")
        monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy\u200babcdef")  # ZWSP mid-key
        _sanitize_loaded_credentials()
        assert os.environ["GOOGLE_API_KEY"] == "AIzaSyabcdef"

        captured = capsys.readouterr()
        assert "GOOGLE_API_KEY" in captured.err
        assert "U+200B" in captured.err
        assert "re-copy" in captured.err.lower()

    def test_warning_fires_only_once_per_key(self, monkeypatch, capsys):
        """Repeated loads (user env + project env) must not double-warn."""
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.discard("GEMINI_API_KEY")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza\u028bbad")
        _sanitize_loaded_credentials()
        first = capsys.readouterr().err

        monkeypatch.setenv("GEMINI_API_KEY", "AIza\u028bbad2")
        _sanitize_loaded_credentials()
        second = capsys.readouterr().err

        assert "GEMINI_API_KEY" in first
        assert second == ""  # no repeat warning

    def test_ascii_control_chars_not_stripped(self, monkeypatch, capsys):
        """ASCII control bytes (e.g. ESC 0x1B from terminal paste) are NOT non-ASCII.

        This is intentional — they're valid ASCII for HTTP headers even if the
        provider rejects them. Documents the scope of the sanitizer.
        """
        from hermes_cli.env_loader import _sanitize_loaded_credentials, _WARNED_KEYS

        _WARNED_KEYS.clear()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant\x1bapi-key")
        _sanitize_loaded_credentials()
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant\x1bapi-key"
        assert capsys.readouterr().err == ""
