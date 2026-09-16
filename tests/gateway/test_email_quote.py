"""Tests for quoting the inbound email under replies (platforms.email.quote_original / EMAIL_QUOTE_ORIGINAL)."""

import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_BASE_ENV = {
    "EMAIL_ADDRESS": "hermes@test.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_SMTP_HOST": "smtp.test.com",
}


def _make_adapter(extra=None, env=None):
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter

    with patch.dict(os.environ, {**_BASE_ENV, **(env or {})}):
        return EmailAdapter(PlatformConfig(enabled=True, extra=extra or {}))


def _msg_data(body="Hello Hermes,\nplease help.", message_id="<orig-1@test.com>", sender="user@test.com",
              date="Wed, 16 Sep 2026 15:53:29 +0200", name="Some User", subject="Question"):
    return {"uid": b"1", "sender_addr": sender, "sender_name": name, "subject": subject, "message_id": message_id,
            "in_reply_to": "", "body": body, "attachments": [], "date": date, "sender_authenticated": True,
            "auth_reason": "dmarc=pass"}


def _dispatch(adapter, msg_data):
    adapter.handle_message = AsyncMock()
    asyncio.run(adapter._dispatch_message(msg_data))


def _body_of(sent_msg):
    for part in sent_msg.walk():
        if part.get_content_type() == "text/plain":
            return part.get_payload(decode=True).decode("utf-8")
    return None


class _SmtpCapture:
    """Patches smtplib.SMTP; ``bodies`` holds the plain-text body of every sent message."""

    def __init__(self, fail_first=False):
        self.bodies, self._fail_first = [], fail_first

    def __enter__(self):
        server = MagicMock()

        def _send(msg):
            if self._fail_first:
                self._fail_first = False
                raise OSError("smtp down")
            self.bodies.append(_body_of(msg))
            self.last = msg

        server.send_message.side_effect = _send
        self._patcher = patch("smtplib.SMTP", return_value=server)
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        self._patcher.stop()


class TestBuildQuote(unittest.TestCase):
    def _quote(self, body, **kw):
        from plugins.platforms.email.adapter import _DEFAULT_QUOTE_HEADER, _build_quote

        args = {"date": "Wed, 16 Sep 2026 15:53:29 +0200", "name": "Some User", "address": "user@test.com",
                "header_fmt": _DEFAULT_QUOTE_HEADER, "max_chars": 10_000, **kw}
        return _build_quote(body, **args)

    def test_multiline_with_blank_and_already_quoted_lines(self):
        quote = self._quote("First line  \r\n\r\n> earlier quote\r\nlast\r\n\r\n")
        self.assertEqual(quote, "\n\nOn Wed, 16 Sep 2026 15:53 +0200, Some User <user@test.com> wrote:\n"
                                "> First line\n>\n>> earlier quote\n> last")

    def test_empty_body_yields_no_quote(self):
        self.assertEqual(self._quote("  \r\n \n"), "")

    def test_invalid_date_uses_raw_value(self):
        self.assertIn("On not a date, Some User", self._quote("hi", date="not a date"))

    def test_missing_date_is_left_out_of_header(self):
        self.assertTrue(self._quote("hi", date="").startswith("\n\nSome User <user@test.com> wrote:\n"))

    def test_name_falls_back_to_address(self):
        self.assertIn("user@test.com <user@test.com> wrote:", self._quote("hi", name=""))

    def test_truncates_at_line_end(self):
        quote = self._quote("aaaa\nbbbb\ncccc\ndddd", max_chars=len("> aaaa\n> bbbb\n> [...]"))
        self.assertTrue(quote.endswith("\n> aaaa\n> bbbb\n> [...]"))

    def test_broken_header_format_falls_back(self):
        self.assertIn("Some User <user@test.com> wrote:", self._quote("hi", header_fmt="{nope} wrote"))


class TestQuoteOriginal(unittest.TestCase):
    def setUp(self):
        self._env = patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"})
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def test_flag_off_body_unchanged(self):
        adapter = _make_adapter()
        _dispatch(adapter, _msg_data())
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", "The answer.", None)
        self.assertEqual(smtp.bodies, ["The answer."])
        self.assertEqual(adapter._original_by_msg_id, {})

    def test_flag_via_config(self):
        adapter = _make_adapter(extra={"quote_original": True})
        _dispatch(adapter, _msg_data())
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", "The answer.", None)
        self.assertEqual(smtp.bodies[0], "The answer.\n\nOn Wed, 16 Sep 2026 15:53 +0200, Some User <user@test.com> wrote:\n"
                                         "> Hello Hermes,\n> please help.")

    def test_flag_via_string_config(self):
        adapter = _make_adapter(extra={"quote_original": "true"})
        _dispatch(adapter, _msg_data())
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", "The answer.", None)
        self.assertIn("> please help.", smtp.bodies[0])

    def test_flag_via_env_without_config(self):
        adapter = _make_adapter(env={"EMAIL_QUOTE_ORIGINAL": "true"})
        self.assertTrue(adapter._quote_original)
        _dispatch(adapter, _msg_data())
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", "The answer.", None)
        self.assertIn("> Hello Hermes,", smtp.bodies[0])

    def test_env_wins_over_config(self):
        adapter = _make_adapter(extra={"quote_original": True}, env={"EMAIL_QUOTE_ORIGINAL": "false"})
        self.assertFalse(adapter._quote_original)

    def test_quote_max_chars(self):
        adapter = _make_adapter(extra={"quote_original": True, "quote_max_chars": len("> line 1\n> [...]")})
        _dispatch(adapter, _msg_data(body="line 1\nline 2\nline 3"))
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", "Answer", None)
        self.assertTrue(smtp.bodies[0].endswith("\n> line 1\n> [...]"))

    def test_quote_shrinks_to_max_message_length_and_reply_is_untouched(self):
        from plugins.platforms.email.adapter import MAX_MESSAGE_LENGTH

        adapter = _make_adapter(extra={"quote_original": True})
        _dispatch(adapter, _msg_data(body="\n".join(f"original line {i}" for i in range(2000))))
        reply = "r" * (MAX_MESSAGE_LENGTH - 500)
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", reply, None)
        body = smtp.bodies[0]
        self.assertTrue(body.startswith(reply))
        self.assertLessEqual(len(body), MAX_MESSAGE_LENGTH)
        self.assertTrue(body.endswith("> [...]"))

    def test_quote_dropped_when_reply_fills_the_limit(self):
        from plugins.platforms.email.adapter import MAX_MESSAGE_LENGTH

        adapter = _make_adapter(extra={"quote_original": True})
        _dispatch(adapter, _msg_data())
        reply = "r" * MAX_MESSAGE_LENGTH
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", reply, None)
        self.assertEqual(smtp.bodies[0], reply)

    def test_reply_to_selects_the_right_original(self):
        adapter = _make_adapter(extra={"quote_original": True})
        _dispatch(adapter, _msg_data(body="first mail", message_id="<a@test.com>"))
        _dispatch(adapter, _msg_data(body="second mail", message_id="<b@test.com>"))
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", "Reply to first", "<a@test.com>")
            adapter._send_email("user@test.com", "Reply to second", "<b@test.com>")
        self.assertIn("> first mail", smtp.bodies[0])
        self.assertNotIn("second mail", smtp.bodies[0])
        self.assertIn("> second mail", smtp.bodies[1])

    def test_unknown_reply_to_does_not_quote_another_mail(self):
        adapter = _make_adapter(extra={"quote_original": True})
        _dispatch(adapter, _msg_data(body="latest mail", message_id="<b@test.com>"))
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", "Reply", "<evicted@test.com>")
        self.assertEqual(smtp.bodies, ["Reply"])

    def test_mail_without_message_id_uses_sender_fallback(self):
        adapter = _make_adapter(extra={"quote_original": True})
        _dispatch(adapter, _msg_data(body="no id here", message_id=""))
        with _SmtpCapture() as smtp:
            adapter._send_email("user@test.com", "Reply", None)
        self.assertIn("> no id here", smtp.bodies[0])

    def test_only_first_send_per_inbound_mail_quotes(self):
        adapter = _make_adapter(extra={"quote_original": True})
        _dispatch(adapter, _msg_data())
        with _SmtpCapture() as smtp:
            asyncio.run(adapter.send("user@test.com", "part 1", "<orig-1@test.com>"))
            asyncio.run(adapter.send("user@test.com", "part 2", "<orig-1@test.com>"))
            asyncio.run(adapter.send("user@test.com", "part 3", None))
        self.assertIn("> Hello Hermes,", smtp.bodies[0])
        self.assertEqual(smtp.bodies[1:], ["part 2", "part 3"])

    def test_failed_send_is_quoted_again_on_retry(self):
        adapter = _make_adapter(extra={"quote_original": True})
        _dispatch(adapter, _msg_data())
        with _SmtpCapture(fail_first=True) as smtp:
            first = asyncio.run(adapter.send("user@test.com", "Answer", "<orig-1@test.com>"))
            second = asyncio.run(adapter.send("user@test.com", "Answer", "<orig-1@test.com>"))
        self.assertFalse(first.success)
        self.assertTrue(second.success)
        self.assertIn("> Hello Hermes,", smtp.bodies[0])

    def test_send_document_quotes(self):
        adapter = _make_adapter(extra={"quote_original": True})
        _dispatch(adapter, _msg_data())
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"doc")
            tmp_path = f.name
        try:
            with _SmtpCapture() as smtp:
                result = asyncio.run(adapter.send_document("user@test.com", tmp_path, "Here is the file"))
            self.assertTrue(result.success)
            self.assertIn("Here is the file\n\nOn ", smtp.bodies[0])
            self.assertIn("> please help.", smtp.bodies[0])
            self.assertTrue(any("attachment" in str(p.get("Content-Disposition", "")) for p in smtp.last.walk()))
        finally:
            os.unlink(tmp_path)

    def test_no_context_no_quote(self):
        adapter = _make_adapter(extra={"quote_original": True})
        with _SmtpCapture() as smtp:
            adapter._send_email("stranger@test.com", "Cron report", None)
        self.assertEqual(smtp.bodies, ["Cron report"])

    def test_standalone_send_never_quotes(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import _standalone_send

        server = MagicMock()
        with patch.dict(os.environ, {**_BASE_ENV, "EMAIL_QUOTE_ORIGINAL": "true"}), \
                patch("smtplib.SMTP", return_value=server):
            result = asyncio.run(_standalone_send(PlatformConfig(enabled=True, extra={"quote_original": True}),
                                                  "user@test.com", "Notification"))
        self.assertTrue(result["success"])
        self.assertEqual(server.send_message.call_args[0][0].get_payload(decode=True).decode(), "Notification")

    def test_rejected_sender_is_not_remembered(self):
        adapter = _make_adapter(extra={"quote_original": True})
        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "", "EMAIL_ALLOWED_USERS": "friend@test.com"}):
            _dispatch(adapter, _msg_data(sender="intruder@test.com"))
        adapter.handle_message.assert_not_called()
        with _SmtpCapture() as smtp:
            adapter._send_email("intruder@test.com", "Pairing code: 123456", None)
        self.assertEqual(smtp.bodies, ["Pairing code: 123456"])

    def test_lookup_is_bounded(self):
        from plugins.platforms.email.adapter import _QUOTE_LOOKUP_MAX

        adapter = _make_adapter(extra={"quote_original": True})
        adapter.handle_message = AsyncMock()
        for i in range(_QUOTE_LOOKUP_MAX + 50):
            asyncio.run(adapter._dispatch_message(_msg_data(body=f"mail {i}", message_id=f"<m{i}@test.com>")))
        self.assertEqual(len(adapter._original_by_msg_id), _QUOTE_LOOKUP_MAX)
        self.assertNotIn("<m0@test.com>", adapter._original_by_msg_id)
        self.assertIn(f"<m{_QUOTE_LOOKUP_MAX + 49}@test.com>", adapter._original_by_msg_id)


if __name__ == "__main__":
    unittest.main()
