"""Custom email subject via send metadata (cron ``subject_template``).

Kept in a dedicated file so the subject-template PR does not touch the shared
``test_email.py``. When ``metadata["subject"]`` is present the adapter starts a
fresh, non-threaded email with that verbatim subject; without it the normal
``Re:`` inbound-thread behavior is preserved.
"""

import asyncio
import os
import unittest
from unittest.mock import patch


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


class TestCustomSubject(unittest.TestCase):
    def test_new_reply_custom_subject_used_verbatim_without_threading(self):
        adapter = _make_adapter()
        adapter._thread_context["user@example.com"] = {"subject": "Old Thread", "message_id": "<old@test.com>"}
        msg, _msg_id, subject = adapter._new_reply(
            "user@example.com", "body", custom_subject="Daily Report 2026/09/18")
        self.assertEqual(subject, "Daily Report 2026/09/18")
        self.assertEqual(msg["Subject"], "Daily Report 2026/09/18")
        # A templated report starts a fresh conversation: the cached inbound thread id is not reused.
        self.assertIsNone(msg["In-Reply-To"])
        self.assertIsNone(msg["References"])

    def test_new_reply_without_custom_subject_threads_onto_context(self):
        adapter = _make_adapter()
        adapter._thread_context["user@example.com"] = {"subject": "Old Thread", "message_id": "<old@test.com>"}
        msg, _msg_id, subject = adapter._new_reply("user@example.com", "body", attach_empty_body=True)
        self.assertEqual(subject, "Re: Old Thread")
        self.assertEqual(msg["In-Reply-To"], "<old@test.com>")
        self.assertEqual(msg["References"], "<old@test.com>")

    def test_send_passes_metadata_subject(self):
        adapter = _make_adapter()
        captured = {}

        def _fake_send_email(to_addr, body, reply_to_msg_id=None, custom_subject=None):
            captured.update(to=to_addr, body=body, reply=reply_to_msg_id, subject=custom_subject)
            return "<msg-1@test.com>"

        adapter._send_email = _fake_send_email
        result = asyncio.run(adapter.send("user@example.com", "Hello", metadata={"subject": "Daily Report"}))
        self.assertTrue(result.success)
        self.assertEqual(captured["to"], "user@example.com")
        self.assertEqual(captured["body"], "Hello")
        self.assertEqual(captured["subject"], "Daily Report")

    def test_send_without_subject_metadata_passes_none(self):
        adapter = _make_adapter()
        captured = {}

        def _fake_send_email(to_addr, body, reply_to_msg_id=None, custom_subject=None):
            captured["subject"] = custom_subject
            return "<msg-1@test.com>"

        adapter._send_email = _fake_send_email
        result = asyncio.run(adapter.send("user@example.com", "Hello"))
        self.assertTrue(result.success)
        self.assertIsNone(captured["subject"])


if __name__ == "__main__":
    unittest.main()
