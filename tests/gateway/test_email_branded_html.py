"""Regression tests for branded-HTML cron email output.

Reproduces the bug where cron jobs that produce a bare HTML report
(``<!DOCTYPE html>...</html>`` with a ``<title>`` element) arrive at the
recipient as a ``text/plain`` email with subject ``"Hermes Agent"`` and a
truncated body — instead of as a properly rendered HTML email whose
subject matches the ``<title>``.

These are the four class-level bugs found in production:

1. ``MIMEText(body, "plain", "utf-8")`` is hardcoded at every send site,
   so even a complete HTML document is shipped as ``text/plain``.
2. The default ``MAX_PLATFORM_OUTPUT = 4000`` gateway slice truncates the
   body mid-``<td>`` when ``splits_long_messages`` is False.
3. Subjects are hardcoded ``"Hermes Agent"`` (or read from a thread
   context that never gets populated for cron sends), so the subject
   never matches the document the user is about to read.
4. LLMs leak a "validation summary" prose preamble before the HTML
   despite explicit ``NO PREAMBLE`` prompt instructions — that preamble
   needs to be stripped before the MIME helper can recognise the body as
   HTML.

The tests pin the fixes by exercising the three helpers
(``_html_subtype``, ``_extract_title_from_html``, ``_strip_leading_prose``)
and by asserting on an end-to-end SMTP send through ``_standalone_send``
(no IMAP needed).
"""

import asyncio
import os
import unittest
from email import policy
from email.parser import BytesParser
from unittest.mock import MagicMock, patch

from plugins.platforms.email.adapter import (
    _extract_title_from_html,
    _html_subtype,
    _standalone_send,
    _strip_leading_prose,
    _subject_for_body,
)


class TestHtmlSubtypeHelper(unittest.TestCase):
    """``_html_subtype`` chooses ``html`` for complete HTML documents and
    ``plain`` for everything else."""

    def test_bare_html_document_is_html(self):
        body = (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head><title>Report</title></head>\n"
            "<body><p>Hello</p></body>\n"
            "</html>\n"
        )
        self.assertEqual(_html_subtype(body), "html")

    def test_lowercase_doctype_is_html(self):
        body = (
            "<!doctype html>\n<html><head><title>x</title></head><body></body></html>"
        )
        self.assertEqual(_html_subtype(body), "html")

    def test_leading_whitespace_is_stripped(self):
        body = "\n\n   <!DOCTYPE html>\n<html></html>"
        self.assertEqual(_html_subtype(body), "html")

    def test_prose_preamble_is_not_html(self):
        # The 2026-08-18 incident: agent wrote a prose validation summary
        # before the doctype. Without the preamble-strip helper, the MIME
        # detector sees the prose first and returns plain.
        body = (
            "This is a verification run for the patch. "
            "Here is the resulting report:\n\n"
            "<!DOCTYPE html>\n<html></html>"
        )
        self.assertEqual(_html_subtype(body), "plain")

    def test_literal_doctype_mention_in_prose_is_not_html(self):
        # A line that *talks about* <!DOCTYPE html> shouldn't be flagged.
        body = (
            "The cron job emits a complete document, e.g.\n\n"
            "<!DOCTYPE html>\n"
            "<html>\n"
            "  ...a styled report...\n"
            "</html>\n\n"
            "Some closing remarks about why this matters."
        )
        self.assertEqual(_html_subtype(body), "plain")

    def test_plain_text_reply_is_plain(self):
        self.assertEqual(_html_subtype("Thanks, will do."), "plain")

    def test_empty_body_is_plain(self):
        self.assertEqual(_html_subtype(""), "plain")


class TestExtractTitleFromHtml(unittest.TestCase):
    """``_extract_title_from_html`` returns the trimmed ``<title>`` content
    or ``None`` when no real ``<title>`` element exists."""

    def test_returns_inner_title(self):
        body = (
            "<!DOCTYPE html>\n<html><head>"
            "<title>Daily MFV Blog Content Capture Processor — August 2026"
            "</title></head></html>"
        )
        self.assertEqual(
            _extract_title_from_html(body),
            "Daily MFV Blog Content Capture Processor — August 2026",
        )

    def test_collapses_internal_whitespace(self):
        body = (
            "<html><head><title>Foo\n\n    Bar   Baz\n</title></head></html>"
        )
        self.assertEqual(_extract_title_from_html(body), "Foo Bar Baz")

    def test_returns_none_when_missing(self):
        self.assertIsNone(_extract_title_from_html("<html></html>"))
        self.assertIsNone(_extract_title_from_html("Plain text body"))
        self.assertIsNone(_extract_title_from_html(""))

    def test_literal_title_mention_in_prose_is_none(self):
        # A bare substring "title" or self-closing <title/> should not match;
        # we want a real <title>...</title> element pair to extract from.
        body = "Some prose. <title/> then more prose. <title>Real</title>"
        self.assertEqual(_extract_title_from_html(body), "Real")


class TestSubjectForBody(unittest.TestCase):
    """Branded reports use the exact HTML title; only plain replies use Re:."""

    def test_html_title_is_exact_and_never_prefixed_with_re(self):
        body = (
            "<!DOCTYPE html><html><head>"
            "<title>Daily GBrain Export — 2026-08-31</title>"
            "</head><body></body></html>"
        )
        subject = _subject_for_body(body, "Old Thread Subject")
        self.assertEqual(subject, "Daily GBrain Export — 2026-08-31")
        self.assertFalse(subject.startswith("Re:"))

    def test_plain_reply_adds_re_once(self):
        self.assertEqual(_subject_for_body("Plain reply", "Status"), "Re: Status")
        self.assertEqual(_subject_for_body("Plain reply", "Re: Status"), "Re: Status")


class TestStripLeadingProse(unittest.TestCase):
    """``_strip_leading_prose`` removes any prose the LLM leaked before
    ``<!DOCTYPE html>`` so the MIME detector can recognise the body."""

    def test_no_prose_is_noop(self):
        body = "<!DOCTYPE html>\n<html></html>"
        self.assertEqual(_strip_leading_prose(body), body)

    def test_strips_validation_summary(self):
        # Real pattern from the 2026-08-18 incident.
        prose = (
            "This is a verification run that Waseem manually triggered to "
            "test the email adapter/prompt patch.\n\n"
        )
        body = prose + "<!DOCTYPE html>\n<html></html>"
        self.assertEqual(
            _strip_leading_prose(body),
            "<!DOCTYPE html>\n<html></html>",
        )

    def test_strips_trailing_content_after_html(self):
        prose = "Validation summary line.\n\n"
        html = "<!DOCTYPE html>\n<html><body>Report</body></html>"
        body = prose + html + "\n```\nTo stop or manage this job..."
        self.assertEqual(_strip_leading_prose(body), html)

    def test_no_doctype_is_noop(self):
        body = "Plain text without any html"
        self.assertEqual(_strip_leading_prose(body), body)

    def test_empty_body_is_empty(self):
        self.assertEqual(_strip_leading_prose(""), "")


def _run(coro):
    """Helper to run async test helpers from sync test methods."""
    return asyncio.get_event_loop().run_until_complete(coro)


@patch.dict(os.environ, {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_SMTP_HOST": "smtp.test.com",
    "EMAIL_SMTP_PORT": "587",
}, clear=False)
class TestStandaloneSendHtml(unittest.TestCase):
    """``_standalone_send`` (the cron delivery fallback) must:

    * set ``Content-Type: text/html`` when the body is a complete HTML
      document,
    * use the document's ``<title>`` as the Subject (not "Hermes Agent"),
    * strip a leading prose preamble before either check,
    * deliver the full body via a single SMTP transaction (no slicing).
    """

    HTML_BODY = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head><title>Branded Report — August 2026</title></head>\n"
        "<body>\n"
        "  <h1>Report</h1>\n"
        "  <p>" + ("A " * 2000) + "</p>\n"
        "</body>\n"
        "</html>\n"
    )

    def _captured_message(self, body=None):
        """Run ``_standalone_send`` against a mocked SMTP server that
        records the assembled ``MIMEText`` bytes so we can parse it."""
        from gateway.config import PlatformConfig

        captured = {}

        class _FakeSMTP:
            def __init__(self, host, port, *a, **kw):
                self.host = host
                self.port = port

            def login(self, *a, **kw):
                return None

            def send_message(self, msg):
                # ``smtplib.SMTP.send_message`` writes serialised bytes
                # to the wire; capturing the live MIMEText here is enough
                # because BytesPolicy's ``as_bytes`` produces the same
                # payload that the real server would see.
                captured["msg"] = msg

            def sendmail(self, _from, _to, msg_bytes):
                captured.setdefault("msg_bytes", msg_bytes)

            def starttls(self, *a, **kw):
                return None

            def quit(self):
                return None

        if body is None:
            body = self.HTML_BODY

        pconfig = PlatformConfig(
            enabled=True,
            extra={
                "address": "hermes@test.com",
                "smtp_host": "smtp.test.com",
            },
        )

        with patch("smtplib.SMTP", _FakeSMTP):
            rc = _run(_standalone_send(
                pconfig,
                chat_id="user@test.com",
                message=body,
            ))

        self.assertTrue(
            rc.get("success"),
            msg=f"send returned {rc}",
        )
        # ``sendmail`` always uses \r\n line endings; normalise so the
        # parser doesn't complain about mid-message line endings.
        if "msg_bytes" in captured:
            msg_bytes = captured["msg_bytes"].replace(b"\r\n", b"\n")
        else:
            msg_bytes = captured["msg"].as_bytes(policy=policy.default)
            msg_bytes = msg_bytes.replace(b"\r\n", b"\n")
        return msg_bytes

    @staticmethod
    def _decode_body(msg):
        """Return the decoded body of the first text payload in ``msg``.

        Handles both ``Message`` (single-part) and ``MessageGroup`` (multipart)
        via ``msg.walk()`` so the test doesn't care whether ``_standalone_send``
        decided to wrap the body in a multipart container."""
        from email.message import Message

        for part in msg.walk():
            if isinstance(part, Message) and part.get_content_type() in (
                "text/plain",
                "text/html",
            ):
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    return payload.decode(part.get_content_charset() or "utf-8")
                return payload or ""
        return ""

    def test_standalone_send_uses_html_mime_for_html_body(self):
        msg_bytes = self._captured_message()
        parsed = BytesParser(policy=policy.default).parsebytes(msg_bytes)
        # Walk the tree looking for the text/html part; some send paths
        # wrap it in a multipart/alternative container.
        ctype = None
        for part in parsed.walk():
            if part.get_content_type() in ("text/plain", "text/html"):
                ctype = part.get_content_type()
                break
        self.assertEqual(ctype, "text/html")

    def test_standalone_send_subject_comes_from_title(self):
        msg_bytes = self._captured_message()
        parsed = BytesParser(policy=policy.default).parsebytes(msg_bytes)
        self.assertEqual(parsed["Subject"], "Branded Report — August 2026")

    def test_standalone_send_ships_full_body(self):
        # The body is large enough that the old ``MAX_PLATFORM_OUTPUT``
        # cap (4000) would have sliced it. If the adapter still respects
        # that cap (because ``splits_long_messages`` is False), the
        # decoded body will be shorter than ``len(self.HTML_BODY)``.
        msg_bytes = self._captured_message()
        parsed = BytesParser(policy=policy.default).parsebytes(msg_bytes)
        body = self._decode_body(parsed)
        self.assertIn("Branded Report — August 2026", body)
        # Whole-document telltale: the closing </html> must arrive.
        self.assertTrue(
            body.rstrip().endswith("</html>"),
            msg="body does not end with </html> — gateway slice truncated it",
        )

    def test_standalone_send_strips_leading_prose_before(self):
        preamble = (
            "This is a **verification run** that Waseem manually triggered "
            "to test the email adapter patch.\n\n"
        )
        msg_bytes = self._captured_message(body=preamble + self.HTML_BODY)
        parsed = BytesParser(policy=policy.default).parsebytes(msg_bytes)
        ctype = None
        for part in parsed.walk():
            if part.get_content_type() in ("text/plain", "text/html"):
                ctype = part.get_content_type()
                break
        self.assertEqual(ctype, "text/html")
        self.assertEqual(parsed["Subject"], "Branded Report — August 2026")
        body = self._decode_body(parsed)
        self.assertNotIn("verification run", body)


@patch.dict(os.environ, {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_SMTP_HOST": "smtp.test.com",
}, clear=False)
class TestSplitsLongMessagesFlag(unittest.TestCase):
    """``EmailAdapter.splits_long_messages`` must be True so the gateway
    delivery layer doesn't slice the body mid-``<td>``."""

    def test_splits_long_messages_is_true(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter
        adapter = EmailAdapter(PlatformConfig(enabled=True))
        self.assertTrue(adapter.splits_long_messages)


if __name__ == "__main__":
    unittest.main()
