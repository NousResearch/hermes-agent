"""Outbound email bodies: HTML must arrive as HTML, not as visible markup.

``_new_reply`` attached every reply as a single ``text/plain`` part, so a reply carrying an HTML
document reached the mailbox as source code. The contract now:

- a body that OPENS with ``<!doctype html>`` / ``<html>`` (leading whitespace and any case allowed)
  is sent as ``multipart/alternative``: a readable ``text/plain`` fallback first, the untouched
  HTML second;
- every other body keeps the exact single ``text/plain`` part it always had — ordinary chat
  replies (including ones that merely contain a tag) must not change;
- headers, threading (``In-Reply-To``/``References``) and ``Message-ID`` are unaffected either way.

Both outbound paths are covered: ``_new_reply`` (``_send_email`` and ``_send_with_files``) and the
out-of-process ``_standalone_send``.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

HTML_BODY = (
    "<!DOCTYPE html>\n<html><head><style>h1{color:red}</style></head>"
    "<body><h1>Status</h1><p>Deploy &amp; ship</p><script>track()</script></body></html>"
)


def _make_adapter():
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True))


def _sent_message(body, *, to_addr="user@test.com", thread=None):
    """Run one ``_send_email`` against a mocked SMTP and return the message handed to ``send_message``."""
    adapter = _make_adapter()
    if thread:
        adapter._thread_context[to_addr] = thread
    with patch("smtplib.SMTP") as smtp_cls:
        mock_server = MagicMock()
        smtp_cls.return_value = mock_server
        adapter._send_email(to_addr, body, None)
        assert mock_server.send_message.called, "no message was sent"
        return mock_server.send_message.call_args[0][0]


def _reply_body_part(msg):
    """The body part of a reply: ``_new_reply`` wraps everything in ``multipart/mixed``."""
    assert msg.get_content_type() == "multipart/mixed"
    return msg.get_payload()[0]


class TestHtmlBody:
    def test_html_body_becomes_multipart_alternative(self):
        msg = _sent_message(HTML_BODY)
        alternative = _reply_body_part(msg)
        assert alternative.get_content_type() == "multipart/alternative"
        plain, html = alternative.get_payload()
        assert plain.get_content_type() == "text/plain"
        assert html.get_content_type() == "text/html"
        # The HTML part carries the original body byte-for-byte.
        assert html.get_payload(decode=True).decode("utf-8") == HTML_BODY
        # ...and the fallback is readable text, not markup.
        text = plain.get_payload(decode=True).decode("utf-8")
        assert "Status" in text
        assert "Deploy & ship" in text
        assert "<h1>" not in text and "<p>" not in text

    def test_html_fallback_omits_script_and_style_source(self):
        msg = _sent_message(HTML_BODY)
        text = _reply_body_part(msg).get_payload()[0].get_payload(decode=True).decode("utf-8")
        assert "track()" not in text
        assert "color:red" not in text

    def test_mixed_case_doctype_and_leading_whitespace_detected(self):
        bodies = (
            "<!DoCtYpE HtMl><p>x</p>",
            "\n   <HTML lang=\"en\"><body>x</body></HTML>",
            "<html><body>x</body>",
        )
        for body in bodies:
            assert _reply_body_part(_sent_message(body)).get_content_type() == "multipart/alternative", body

    def test_threading_headers_survive_an_html_body(self):
        adapter = _make_adapter()
        adapter._thread_context["user@test.com"] = {
            "subject": "Project question",
            "message_id": "<original@test.com>",
        }
        with patch("smtplib.SMTP") as smtp_cls:
            mock_server = MagicMock()
            smtp_cls.return_value = mock_server
            adapter._send_email("user@test.com", HTML_BODY, None)
            msg = mock_server.send_message.call_args[0][0]
        assert msg["Subject"] == "Re: Project question"
        assert msg["In-Reply-To"] == "<original@test.com>"
        assert msg["References"] == "<original@test.com>"
        assert msg["Message-ID"]
        assert msg["Date"]
        assert _reply_body_part(msg).get_content_type() == "multipart/alternative"


class TestPlainBodyUnchanged:
    def test_plain_body_stays_a_single_text_plain_part(self):
        body = "Here is the answer.\nSecond line."
        msg = _sent_message(body)
        part = _reply_body_part(msg)
        assert part.get_content_type() == "text/plain"
        assert not part.is_multipart()
        assert part.get_payload(decode=True).decode("utf-8") == body

    def test_body_containing_tags_is_not_treated_as_html(self):
        body = "Render <b>bold</b> and use <div>layout</div> — see <htmlcheat.xyz>."
        part = _reply_body_part(_sent_message(body))
        assert part.get_content_type() == "text/plain"
        assert part.get_payload(decode=True).decode("utf-8") == body

    def test_empty_body_stays_plain(self):
        part = _reply_body_part(_sent_message(""))
        assert part.get_content_type() == "text/plain"
        assert part.is_multipart() is False


class TestOtherSendPaths:
    def test_send_with_files_keeps_the_alternative_body(self):
        adapter = _make_adapter()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
            handle.write(b"attachment bytes")
            path = handle.name
        try:
            with patch("smtplib.SMTP") as smtp_cls:
                mock_server = MagicMock()
                smtp_cls.return_value = mock_server
                adapter._send_with_files("user@test.com", HTML_BODY, [(Path(path), "a.txt")], lenient=False)
                msg = mock_server.send_message.call_args[0][0]
        finally:
            os.unlink(path)
        assert msg.get_content_type() == "multipart/mixed"
        alternative, attachment = msg.get_payload()
        assert alternative.get_content_type() == "multipart/alternative"
        assert attachment.get_content_type() == "application/octet-stream"

    def test_standalone_send_delivers_html(self):
        from plugins.platforms.email.adapter import _standalone_send

        with patch.dict(os.environ, {"EMAIL_PASSWORD": "secret"}, clear=False):
            with patch("smtplib.SMTP") as smtp_cls:
                mock_server = MagicMock()
                smtp_cls.return_value = mock_server
                result = asyncio.run(_standalone_send(
                    SimpleNamespace(token=None, api_key=None,
                                    extra={"address": "hermes@test.com", "smtp_host": "smtp.test.com"}),
                    "user@test.com", HTML_BODY))
                msg = mock_server.send_message.call_args[0][0]
        assert result["success"] is True
        assert msg.get_content_type() == "multipart/alternative"
        plain, html = msg.get_payload()
        assert plain.get_content_type() == "text/plain"
        assert html.get_payload(decode=True).decode("utf-8") == HTML_BODY
        assert msg["To"] == "user@test.com"

    def test_standalone_send_keeps_plain_text_plain(self):
        from plugins.platforms.email.adapter import _standalone_send

        with patch.dict(os.environ, {"EMAIL_PASSWORD": "secret"}, clear=False):
            with patch("smtplib.SMTP") as smtp_cls:
                mock_server = MagicMock()
                smtp_cls.return_value = mock_server
                asyncio.run(_standalone_send(
                    SimpleNamespace(token=None, api_key=None,
                                    extra={"address": "hermes@test.com", "smtp_host": "smtp.test.com"}),
                    "user@test.com", "Plain hello"))
                msg = mock_server.send_message.call_args[0][0]
        assert msg.get_content_type() == "text/plain"
        assert msg.get_payload(decode=True).decode("utf-8") == "Plain hello"
