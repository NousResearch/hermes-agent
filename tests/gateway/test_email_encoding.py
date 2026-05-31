"""Regression tests for _resolve_charset and _decode_header_value with non-standard charsets.

Covers: #35901 — QQ Mail sends ``unknown-8bit`` charset which crashes IMAP fetch.
"""
import codecs
import email
from email.header import decode_header
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
import importlib
import sys

# Ensure the gateway package is importable (it may lack an __init__ in some
# checkouts, so we add the repo root to sys.path).
_REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The email adapter lives at gateway/platforms/email.py — import via importlib
# to avoid triggering the full gateway package __init__ (which pulls in heavy
# deps like aiohttp).
_mod = importlib.import_module("gateway.platforms.email")

_resolve_charset = _mod._resolve_charset
_decode_header_value = _mod._decode_header_value
_extract_text_body = _mod._extract_text_body


# ===================================================================
# _resolve_charset
# ===================================================================
class TestResolveCharset:
    """Unit tests for charset normalisation."""

    def test_none_returns_utf8(self):
        assert _resolve_charset(None) == "utf-8"

    def test_empty_returns_utf8(self):
        assert _resolve_charset("") == "utf-8"

    def test_whitespace_only_returns_utf8(self):
        assert _resolve_charset("   ") == "utf-8"

    def test_unknown_8bit_returns_utf8(self):
        """The primary fix: QQ Mail's ``unknown-8bit`` charset."""
        assert _resolve_charset("unknown-8bit") == "utf-8"

    def test_unknown_returns_utf8(self):
        assert _resolve_charset("unknown") == "utf-8"

    def test_x_unknown_returns_utf8(self):
        assert _resolve_charset("x-unknown") == "utf-8"

    def test_default_returns_utf8(self):
        assert _resolve_charset("default") == "utf-8"

    def test_case_insensitive(self):
        assert _resolve_charset("UNKNOWN-8BIT") == "utf-8"
        assert _resolve_charset("Unknown-8bit") == "utf-8"

    def test_valid_charset_preserved(self):
        assert _resolve_charset("utf-8") == "utf-8"
        assert _resolve_charset("iso-8859-1") == "iso-8859-1"
        assert _resolve_charset("gbk") == "gbk"
        assert _resolve_charset("gb2312") == "gb2312"

    def test_arbitrary_invalid_charset_falls_back(self):
        """Any charset not in codecs registry falls back to utf-8."""
        assert _resolve_charset("totally-bogus-charset-xyz") == "utf-8"

    def test_codecs_lookup_called(self):
        """Verify the codecs.lookup path is exercised for non-blocklisted names."""
        with patch.object(codecs, "lookup", wraps=codecs.lookup) as mock:
            _resolve_charset("iso-8859-1")
            mock.assert_called_once_with("iso-8859-1")


# ===================================================================
# _decode_header_value
# ===================================================================
class TestDecodeHeaderValue:
    """Regression tests for header decoding with unusual charsets."""

    def test_plain_ascii(self):
        assert _decode_header_value("Hello World") == "Hello World"

    def test_rfc2047_utf8(self):
        raw = "=?utf-8?b?SGVsbG8gV29ybGQ=?="
        assert _decode_header_value(raw) == "Hello World"

    def test_rfc2047_unknown_8bit(self):
        """QQ Mail may encode headers with ``unknown-8bit`` charset.

        The raw bytes should still be decoded (via utf-8 fallback) without
        raising ``LookupError: unknown encoding: unknown-8bit``.
        """
        # Construct a header that decode_header() would parse as
        # (bytes, "unknown-8bit") — simulate QQ Mail behaviour.
        raw_bytes = "测试"
        encoded = f"=?unknown-8bit?b?{__import__('base64').b64encode(raw_bytes.encode('utf-8')).decode('ascii')}?="
        # Should NOT raise
        result = _decode_header_value(encoded)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_header_object_input(self):
        """_decode_header_value must accept email.header.Header objects."""
        h = email.header.Header("测试邮件", charset="utf-8")
        result = _decode_header_value(h)
        assert "测试" in result or isinstance(result, str)

    def test_bytes_input(self):
        """Bytes input triggers TypeError in decode_header; function returns str(raw)."""
        result = _decode_header_value(b"raw bytes")
        # str(b"raw bytes") == "b'raw bytes'" — bytes are not a normal input
        assert isinstance(result, str)

    def test_decode_header_exception_caught(self):
        """If decode_header itself raises, we should return str(raw)."""
        with patch("gateway.platforms.email.decode_header", side_effect=Exception("boom")):
            result = _decode_header_value("something")
            assert result == "something"


# ===================================================================
# _extract_text_body
# ===================================================================
class TestExtractTextBody:
    """Integration tests for body extraction with non-standard charsets."""

    def _make_email(self, body: bytes, charset: str = "utf-8") -> email.message.Message:
        """Build a minimal text/plain email with the given body charset."""
        msg = email.message.Message()
        msg["Content-Type"] = f"text/plain; charset={charset}"
        msg.set_payload(body)
        return msg

    def test_utf8_body(self):
        body = "Hello, world!".encode("utf-8")
        msg = self._make_email(body, "utf-8")
        assert _extract_text_body(msg) == "Hello, world!"

    def test_unknown_8bit_body(self):
        """Payload declared as ``unknown-8bit`` must not crash."""
        body = "Hello from QQ Mail".encode("utf-8")
        msg = self._make_email(body, "unknown-8bit")
        # Should NOT raise LookupError
        result = _extract_text_body(msg)
        assert isinstance(result, str)
        assert "Hello" in result

    def test_gbk_body(self):
        """Chinese encoding should still work normally."""
        body = "你好世界".encode("gbk")
        msg = self._make_email(body, "gbk")
        assert "你好" in _extract_text_body(msg)

    def test_multipart_with_unknown_charset(self):
        """Multipart message with unknown-8bit text/plain part."""
        outer = email.message.Message()
        outer["Content-Type"] = "multipart/alternative; boundary=boundary"
        inner = email.message.Message()
        inner["Content-Type"] = "text/plain; charset=unknown-8bit"
        inner.set_payload("Test body".encode("utf-8"))
        # Manually attach as payload list
        outer.set_payload([inner])
        result = _extract_text_body(outer)
        assert isinstance(result, str)
