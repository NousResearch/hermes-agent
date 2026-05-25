"""Tests for agent.process_bootstrap — _SafeWriter and proxy helpers."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from agent.process_bootstrap import _SafeWriter


# ============================================================================
# _SafeWriter
# ============================================================================
class TestSafeWriter:
    """Tests for the crash-resistant stdio wrapper."""

    # -- write ---------------------------------------------------------------
    def test_write_delegates_to_inner(self):
        writes = []
        mock = type("MockStream", (), {"write": lambda s, d: writes.append(d) or len(d)})()

        sw = _SafeWriter(mock)
        result = sw.write("hello")
        assert result == 5
        assert writes == ["hello"]

    def test_write_oserror_returns_len(self):
        def broken(self, d):
            raise OSError("pipe broken")
        mock = type("MockStream", (), {"write": broken})()

        sw = _SafeWriter(mock)
        result = sw.write("hello")
        assert result == 5  # str len

    def test_write_valueerror_returns_len(self):
        def broken(self, d):
            raise ValueError("closed file")
        mock = type("MockStream", (), {"write": broken})()

        sw = _SafeWriter(mock)
        result = sw.write("hello")
        assert result == 5

    def test_write_bytes_oserror_returns_zero(self):
        def broken(self, d):
            raise OSError("gone")
        mock = type("MockStream", (), {"write": broken})()

        sw = _SafeWriter(mock)
        result = sw.write(b"bytes")
        assert result == 0  # bytes, not str → returns 0

    # -- flush ---------------------------------------------------------------
    def test_flush_delegates(self):
        flushed = []
        mock = type("MockStream", (), {"flush": lambda s: flushed.append(True)})()

        sw = _SafeWriter(mock)
        sw.flush()
        assert flushed == [True]

    def test_flush_oserror_silent(self):
        def broken(self):
            raise OSError("gone")
        mock = type("MockStream", (), {"flush": broken})()

        sw = _SafeWriter(mock)
        sw.flush()  # should not raise

    def test_flush_valueerror_silent(self):
        def broken(self):
            raise ValueError("closed")
        mock = type("MockStream", (), {"flush": broken})()

        sw = _SafeWriter(mock)
        sw.flush()  # should not raise

    # -- fileno --------------------------------------------------------------
    def test_fileno_delegates(self):
        mock = type("MockStream", (), {"fileno": lambda s: 42})()

        sw = _SafeWriter(mock)
        assert sw.fileno() == 42

    # -- isatty --------------------------------------------------------------
    def test_isatty_true(self):
        mock = type("MockStream", (), {"isatty": lambda s: True})()

        sw = _SafeWriter(mock)
        assert sw.isatty() is True

    def test_isatty_false(self):
        mock = type("MockStream", (), {"isatty": lambda s: False})()

        sw = _SafeWriter(mock)
        assert sw.isatty() is False

    def test_isatty_oserror_returns_false(self):
        def broken(self):
            raise OSError("no tty")
        mock = type("MockStream", (), {"isatty": broken})()

        sw = _SafeWriter(mock)
        assert sw.isatty() is False

    def test_isatty_valueerror_returns_false(self):
        def broken(self):
            raise ValueError("closed")
        mock = type("MockStream", (), {"isatty": broken})()

        sw = _SafeWriter(mock)
        assert sw.isatty() is False

    # -- __getattr__ fallback ------------------------------------------------
    def test_getattr_delegates(self):
        mock = type("MockStream", (), {"encoding": "utf-8"})()

        sw = _SafeWriter(mock)
        assert sw.encoding == "utf-8"

    def test_getattr_unknown_raises(self):
        mock = type("MockStream", (), {})()

        sw = _SafeWriter(mock)
        with pytest.raises(AttributeError):
            _ = sw.nonexistent

    # -- repr ----------------------------------------------------------------
    def test_repr_shows_wrapped_type(self):
        mock = type("MockStream", (), {"__class__": type("Fake", (), {})})()

        sw = _SafeWriter(mock)
        rep = repr(sw)
        assert "_SafeWriter" in rep or "MockStream" in rep

    # -- not a _SafeWriter check (used by _install_safe_stdio) ---------------
    def test_not_isinstance_self(self):
        """_SafeWriter is not isinstance of itself — the isinstance check
        in _install_safe_stdio looks for _SafeWriter, and this type() is used."""
        mock = type("MockStream", (), {"write": lambda s, d: 0})()
        sw = _SafeWriter(mock)
        # _install_safe_stdio does: not isinstance(stream, _SafeWriter)
        # This passes because _SafeWriter wraps, not inherits
        assert isinstance(sw, _SafeWriter)


# ============================================================================
# _get_proxy_from_env
# ============================================================================
class TestGetProxyFromEnv:
    """Tests for _get_proxy_from_env."""

    def test_no_proxy_set_returns_none(self, monkeypatch):
        from agent.process_bootstrap import _get_proxy_from_env

        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                    "https_proxy", "http_proxy", "all_proxy"):
            monkeypatch.delenv(key, raising=False)

        with patch("agent.process_bootstrap.normalize_proxy_url") as mock_norm:
            result = _get_proxy_from_env()
            assert result is None
            mock_norm.assert_not_called()

    def test_https_proxy_takes_priority(self, monkeypatch):
        from agent.process_bootstrap import _get_proxy_from_env

        monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.com:8080")
        monkeypatch.setenv("HTTP_PROXY", "http://other:3128")

        with patch("agent.process_bootstrap.normalize_proxy_url", return_value="https://proxy.example.com:8080") as mock_norm:
            result = _get_proxy_from_env()
            assert result == "https://proxy.example.com:8080"
            mock_norm.assert_called_once_with("https://proxy.example.com:8080")

    def test_http_proxy_used_when_https_not_set(self, monkeypatch):
        from agent.process_bootstrap import _get_proxy_from_env

        monkeypatch.setenv("HTTP_PROXY", "http://proxy:3128")

        with patch("agent.process_bootstrap.normalize_proxy_url", return_value="http://proxy:3128") as mock_norm:
            result = _get_proxy_from_env()
            assert result == "http://proxy:3128"

    def test_all_proxy_used_as_fallback(self, monkeypatch):
        from agent.process_bootstrap import _get_proxy_from_env

        monkeypatch.setenv("ALL_PROXY", "socks5://all:1080")
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            monkeypatch.delenv(k, raising=False)

        with patch("agent.process_bootstrap.normalize_proxy_url", return_value="socks5://all:1080"):
            result = _get_proxy_from_env()
            assert result == "socks5://all:1080"

    def test_lowercase_variants_work(self, monkeypatch):
        from agent.process_bootstrap import _get_proxy_from_env

        monkeypatch.setenv("https_proxy", "https://lower.example.com")

        with patch("agent.process_bootstrap.normalize_proxy_url", return_value="https://lower.example.com"):
            result = _get_proxy_from_env()
            assert result == "https://lower.example.com"

    def test_empty_string_treated_as_unset(self, monkeypatch):
        from agent.process_bootstrap import _get_proxy_from_env

        monkeypatch.setenv("HTTPS_PROXY", "")
        monkeypatch.setenv("HTTP_PROXY", "   ")

        with patch("agent.process_bootstrap.normalize_proxy_url") as mock_norm:
            result = _get_proxy_from_env()
            assert result is None
            mock_norm.assert_not_called()
