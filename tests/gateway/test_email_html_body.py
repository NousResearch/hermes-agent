"""Outgoing email carries a ``text/html`` alternative, and the plain part is unchanged.

The adapter sent ``text/plain`` only, so Markdown the agent writes for chat
(``**bold**``, ``## Headings``) arrived in a mail reader as literal asterisks and hashes.
Both send paths must now attach an HTML alternative, and a client that cannot render HTML
must still receive the byte-identical plain text it received before.
"""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Markdown the agent emits, plus untrusted markup that must not survive as live HTML.
_BODY = """## Bottom line

The **PJM** order stands, per [the filing](https://example.gov/pjm).

## Major stories

1. DOE issues an emergency order
2. FERC opens a docket

- Watch the Senate markup
- Track the <script>alert(1)</script> filing
"""

_SMTP_ENV = {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_SMTP_HOST": "smtp.test.com",
}


def _assert_alternative(test: unittest.TestCase, msg) -> None:
    """The contract both send paths share: an HTML alternative beside an unchanged plain part."""
    test.assertTrue(
        any(p.get_content_type() == "multipart/alternative" for p in msg.walk()),
        f"no multipart/alternative in {msg.get_content_type()}",
    )
    bodies = {
        part.get_content_type(): part.get_payload(decode=True).decode("utf-8")
        for part in msg.walk()
        if part.get_content_type() in ("text/plain", "text/html")
    }

    # Clients that cannot render HTML keep the exact text they got before the fix.
    test.assertEqual(bodies["text/plain"], _BODY)

    html = bodies["text/html"]
    test.assertIn("<h3", html)                     # "## " became a heading
    test.assertIn("<strong>PJM</strong>", html)    # "**PJM**" became bold
    test.assertNotIn("##", html)                   # no literal Markdown left behind
    test.assertNotIn("**", html)
    # Content is escaped, so a body can never inject live markup.
    test.assertNotIn("<script>", html)
    test.assertIn("&lt;script&gt;", html)


class TestReplyCarriesHtmlAlternative(unittest.TestCase):
    """The in-process send path (replies, attachments, images, documents)."""

    def _adapter(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter
        with patch.dict(os.environ, _SMTP_ENV):
            return EmailAdapter(PlatformConfig(enabled=True))

    def _send(self, adapter, fn, *args):
        with patch("smtplib.SMTP") as mock_smtp:
            server = MagicMock()
            mock_smtp.return_value = server
            fn(*args)
        return server.send_message.call_args[0][0]

    def test_reply_carries_html_alternative(self):
        adapter = self._adapter()
        sent = self._send(adapter, adapter._send_email, "user@test.com", _BODY, None)
        _assert_alternative(self, sent)

    def test_attachment_rides_beside_the_alternative(self):
        """An attachment is a sibling of the alternative, not a part inside it."""
        adapter = self._adapter()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"attachment bytes")
            path = f.name
        try:
            sent = self._send(adapter, adapter._send_email_with_attachment,
                              "user@test.com", _BODY, path)
        finally:
            os.unlink(path)

        self.assertEqual(sent.get_content_type(), "multipart/mixed")
        _assert_alternative(self, sent)
        self.assertTrue(any(
            "attachment" in str(p.get("Content-Disposition", "")) for p in sent.walk()
        ))


class TestStandaloneSenderCarriesHtmlAlternative(unittest.TestCase):
    """The out-of-process path cron delivery uses — the sibling call path."""

    def test_standalone_send_carries_html_alternative(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email import adapter as email_adapter

        with patch.dict(os.environ, _SMTP_ENV):
            with patch.object(email_adapter, "_open_smtp") as open_smtp:
                server = MagicMock()
                open_smtp.return_value = server
                result = asyncio.run(
                    email_adapter._standalone_send(PlatformConfig(enabled=True), "user@test.com", _BODY)
                )

        self.assertTrue(result.get("success"), result)
        sent = server.send_message.call_args[0][0]
        # Nothing to attach, so the alternative is the top level — not wrapped in a mixed part.
        self.assertEqual(sent.get_content_type(), "multipart/alternative")
        _assert_alternative(self, sent)
