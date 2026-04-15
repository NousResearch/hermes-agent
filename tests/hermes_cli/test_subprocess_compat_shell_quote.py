"""Tests for cross-platform shell_quote() in _subprocess_compat."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch, Mock

import pytest


@pytest.fixture
def compat_module():
    """Import (or re-import) the module fresh so IS_WINDOWS patching works."""
    import hermes_cli._subprocess_compat as mod
    importlib.reload(mod)
    return mod


class TestShellQuotePOSIX:
    """Test shell_quote() on POSIX (the real platform in CI)."""

    def test_simple_space(self, compat_module):
        result = compat_module.shell_quote("hello world")
        assert "hello world" in result
        assert result.startswith("'") or result.startswith('"')

    def test_no_space_unchanged(self, compat_module):
        assert compat_module.shell_quote("hello") == "hello"

    def test_semicolon_metachar(self, compat_module):
        result = compat_module.shell_quote("a; rm -rf /")
        assert "a; rm -rf /" in result
        assert not result.startswith("a;")

    def test_pipe_metachar(self, compat_module):
        result = compat_module.shell_quote("foo | bar")
        assert "foo | bar" in result
        assert not result.startswith("foo |")

    def test_ampersand_metachar(self, compat_module):
        result = compat_module.shell_quote("foo & bar")
        assert "foo & bar" in result
        assert not result.startswith("foo &")

    def test_single_quote_inside(self, compat_module):
        result = compat_module.shell_quote("it's fine")
        assert "it" in result and "fine" in result

    def test_empty_string(self, compat_module):
        assert compat_module.shell_quote("") == "''"


class TestShellQuoteWindows:
    """Test shell_quote() Windows branch via monkeypatch (no real Windows needed)."""

    def test_windows_uses_mslex(self, compat_module, monkeypatch):
        """When IS_WINDOWS=True and mslex is available, mslex.quote is used."""
        monkeypatch.setattr(compat_module, "IS_WINDOWS", True)

        fake_mslex = Mock()
        fake_mslex.quote = lambda s: f"[MSLEX]{s}[MSLEX]"

        with patch.dict(sys.modules, {"mslex": fake_mslex}):
            result = compat_module.shell_quote("hello world")
        assert result == "[MSLEX]hello world[MSLEX]"

    def test_windows_fallback_no_mslex(self, compat_module, monkeypatch):
        """When IS_WINDOWS=True and mslex is not installed, fallback to double-quote."""
        monkeypatch.setattr(compat_module, "IS_WINDOWS", True)

        # Setting mslex to None in sys.modules causes ImportError on import
        with patch.dict(sys.modules, {"mslex": None}):
            result = compat_module.shell_quote("hello world")

        assert result.startswith('"') and result.endswith('"')
        assert "hello world" in result

    def test_windows_fallback_escapes_embedded_quotes(self, compat_module, monkeypatch):
        """Fallback should escape embedded double quotes."""
        monkeypatch.setattr(compat_module, "IS_WINDOWS", True)

        with patch.dict(sys.modules, {"mslex": None}):
            result = compat_module.shell_quote('say "hi"')

        assert '\\"' in result
        assert result.startswith('"') and result.endswith('"')

    def test_windows_metachar_semicolon(self, compat_module, monkeypatch):
        """Windows quoting should handle semicolons (via mslex)."""
        monkeypatch.setattr(compat_module, "IS_WINDOWS", True)

        fake_mslex = Mock()
        fake_mslex.quote = lambda s: f"[Q]{s}[Q]"

        with patch.dict(sys.modules, {"mslex": fake_mslex}):
            result = compat_module.shell_quote("a; rm -rf \\")

        assert result == "[Q]a; rm -rf \\[Q]"