"""Email charset robustness: unknown/malformed charsets must never drop mail.

Regression tests for #35901 (QQ Mail ``unknown-8bit`` charset raising
``LookupError`` and aborting the IMAP fetch), #55381 (malformed RFC 2047
header crashing ``_decode_header_value``), and #55383 (unknown Content-Type
charset crashing ``_extract_text_body``). UIDs are marked seen before the
fetch, so any exception in per-message decoding permanently loses messages —
decoding must degrade, never raise.
"""

import email as email_lib
from email.message import EmailMessage

from plugins.platforms.email.adapter import (
    _decode_header_value,
    _extract_text_body,
    _safe_decode,
)


class TestSafeDecode:
    def test_unknown_8bit_charset_falls_back(self):
        # QQ Mail emits the RFC 1428 "unknown-8bit" placeholder (#35901).
        assert _safe_decode("你好".encode("utf-8"), "unknown-8bit") == "你好"

    def test_garbage_charset_label_never_raises(self):
        assert _safe_decode(b"Hello", "charset") == "Hello"
        assert _safe_decode(b"Hello", "not-a-codec-!!") == "Hello"

    def test_none_charset_defaults_to_utf8(self):
        assert _safe_decode("héllo".encode("utf-8"), None) == "héllo"

    def test_gb2312_label_decodes_gbk_extensions(self):
        # gb2312-labelled mail routinely contains GBK-only characters.
        assert _safe_decode("镕".encode("gb18030"), "gb2312") == "镕"

    def test_invalid_bytes_replace_not_raise(self):
        out = _safe_decode(b"ok \xff\xfe bad", "utf-8")
        assert "ok" in out and "bad" in out

    def test_latin1_last_resort(self):
        # A charset whose codec exists but whose bytes are invalid UTF-8
        # still returns text via errors="replace" — never an exception.
        assert _safe_decode(b"\x96\x97", "unknown-8bit") != ""


class TestDecodeHeaderValue:
    def test_unknown_8bit_encoded_word(self):
        assert _decode_header_value("=?unknown-8bit?B?SGVsbG8=?=") == "Hello"

    def test_malformed_charset_in_encoded_word(self):
        # decode_header parses this fine but "charset" is not a codec —
        # previously a LookupError aborted the whole fetch batch (#55381).
        out = _decode_header_value("=?charset?B?SGVsbG8=?=")
        assert isinstance(out, str)
        assert out  # degrades, never raises/empties

    def test_plain_header_passthrough(self):
        assert _decode_header_value("Just a subject") == "Just a subject"

    def test_qq_mail_gbk_subject(self):
        # =?gbk?B?...?= with GBK bytes; gbk→gb18030 alias covers extensions.
        import base64
        encoded = base64.b64encode("你好".encode("gbk")).decode()
        assert _decode_header_value(f"=?gbk?B?{encoded}?=") == "你好"


class TestExtractTextBodyCharsets:
    def _msg(self, body_bytes: bytes, charset_label: str):
        raw = (
            b"From: a@b.c\r\n"
            b"Subject: t\r\n"
            b"MIME-Version: 1.0\r\n"
            b'Content-Type: text/plain; charset="' + charset_label.encode() + b'"\r\n'
            b"Content-Transfer-Encoding: 8bit\r\n"
            b"\r\n" + body_bytes
        )
        return email_lib.message_from_bytes(raw)

    def test_unknown_charset_body_does_not_raise(self):
        # LookupError from an unknown Content-Type charset aborted the
        # fetch and dropped the message (#55383).
        msg = self._msg("正文内容".encode("utf-8"), "unknown-8bit")
        assert "正文内容" in _extract_text_body(msg)

    def test_bogus_charset_body_does_not_raise(self):
        msg = self._msg(b"plain ascii body", "x-mac-cyrillic-fake")
        assert "plain ascii body" in _extract_text_body(msg)

    def test_multipart_unknown_charset_part(self):
        outer = EmailMessage()
        outer["From"] = "a@b.c"
        outer["Subject"] = "t"
        outer.make_mixed()
        raw_part = (
            b'Content-Type: text/plain; charset="unknown-8bit"\r\n'
            b"Content-Transfer-Encoding: 8bit\r\n"
            b"\r\n"
            b"part body here"
        )
        part = email_lib.message_from_bytes(raw_part)
        outer.attach(part)
        assert "part body here" in _extract_text_body(outer)
