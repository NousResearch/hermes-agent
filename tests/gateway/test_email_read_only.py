"""Email read-only / no-auto-reply mode (#99876).

``platforms.email.extra.read_only: true`` lets a mailbox be used purely as an
inbound feed: IMAP polling / dispatch is
unchanged, but every outgoing send is suppressed before it reaches SMTP. A
suppressed send returns ``success=True`` so the gateway's delivery ledger
marks it delivered rather than retrying — the failure loop that disabling the
SMTP credential would cause. These tests pin that no SMTP helper is invoked in
read-only mode, that the default (unset) still sends, and that all three
adapter send entry points are covered.
"""
import asyncio
import logging
import os
import unittest
from unittest.mock import MagicMock, patch


def _make_adapter(*, extra=None, env=None):
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter

    with patch.dict(os.environ, env or {}, clear=False):
        return EmailAdapter(PlatformConfig(enabled=True, extra=extra or {}))


class TestEmailReadOnly(unittest.TestCase):
    def test_read_only_blocks_the_shared_smtp_delivery_boundary(self):
        """Even a future helper that reaches the shared SMTP sink cannot send."""
        adapter = _make_adapter(extra={"read_only": True})
        adapter._thread_context["user@example.com"] = {
            "subject": "Private report",
            "message_id": "<original@example.com>",
        }

        with patch("smtplib.SMTP") as smtp:
            result = adapter._send_email("user@example.com", "do not disclose")

        self.assertEqual(result, "read-only-suppressed")
        smtp.assert_not_called()

    def test_read_only_via_extra_suppresses_send(self):
        adapter = _make_adapter(extra={"read_only": True})
        adapter._send_email = MagicMock(name="_send_email")

        result = asyncio.run(adapter.send("user@example.com", "hi"))

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "read-only-suppressed")
        adapter._send_email.assert_not_called()

    def test_read_only_is_not_enabled_by_a_non_secret_env_var(self):
        adapter = _make_adapter(env={"EMAIL_READ_ONLY": "true"})

        self.assertFalse(adapter._read_only)

    def test_read_only_connects_with_imap_only_and_never_tests_smtp(self):
        """Inbound-only mode must not need a working SMTP endpoint to receive."""
        adapter = _make_adapter(extra={"read_only": True})
        adapter._address = "hermes@example.com"
        adapter._password = "mailbox-password"
        adapter._imap_host = "imap.example.com"
        adapter._smtp_host = ""
        adapter._connect_smtp = MagicMock(name="smtp_connection")
        imap = MagicMock()
        imap.uid.return_value = ("OK", [b""])

        async def connect_then_disconnect():
            with patch("imaplib.IMAP4_SSL", return_value=imap):
                self.assertTrue(await adapter.connect())
                await adapter.disconnect()

        asyncio.run(connect_then_disconnect())
        adapter._connect_smtp.assert_not_called()

    def test_default_is_not_read_only_and_sends(self):
        adapter = _make_adapter()  # no extra, no env -> read_only stays False
        self.assertFalse(adapter._read_only)
        adapter._send_email = MagicMock(return_value="<mid@localhost>")

        result = asyncio.run(adapter.send("user@example.com", "hi"))

        self.assertTrue(result.success)
        self.assertEqual(result.message_id, "<mid@localhost>")
        adapter._send_email.assert_called_once()

    def test_read_only_blocks_standalone_cron_and_shared_transport_smtp(self):
        """Every Email transport, including cron, stops before SMTP."""
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import _standalone_send

        pconfig = PlatformConfig(
            enabled=True,
            extra={
                "read_only": True,
                "address": "hermes@example.com",
                "smtp_host": "smtp.example.com",
            },
        )
        with patch.dict(os.environ, {"EMAIL_PASSWORD": "mailbox-password"}, clear=False), \
             patch("smtplib.SMTP") as smtp:
            result = asyncio.run(
                _standalone_send(pconfig, "report-recipient@example.com", "scheduled report")
            )

        self.assertTrue(result["success"])
        smtp.assert_not_called()

    def test_read_only_suppresses_document_and_image_batch(self):
        adapter = _make_adapter(extra={"read_only": True})
        adapter._send_email_with_attachment = MagicMock(name="doc")
        adapter._send_email_with_attachments = MagicMock(name="imgs")

        doc = asyncio.run(adapter.send_document("user@example.com", "/tmp/x.pdf"))
        self.assertTrue(doc.success)
        self.assertEqual(doc.message_id, "read-only-suppressed")
        adapter._send_email_with_attachment.assert_not_called()

        # send_multiple_images returns None; the point is no SMTP is attempted.
        asyncio.run(
            adapter.send_multiple_images(
                "user@example.com", [("file:///tmp/a.png", "")],
            )
        )
        adapter._send_email_with_attachments.assert_not_called()

    def test_read_only_suppresses_a_noisy_100_step_run_and_every_notice_kind(self):
        """The adapter is the hard egress boundary, not a display convention."""
        adapter = _make_adapter(extra={"read_only": True})
        adapter._send_email = MagicMock(name="smtp_delivery")

        for step in range(100):
            result = asyncio.run(
                adapter.send(
                    "user@example.com",
                    f"interim commentary {step}",
                    metadata={"session_id": "email-session-100"},
                )
            )
            self.assertTrue(result.success)

        for notice_kind in (
            "iteration-budget",
            "model-fallback",
            "approval-status",
            "attachment-follow-up",
            "stop-closeout",
            "delivery-ledger-retry",
            "final-response",
        ):
            result = asyncio.run(
                adapter.send(
                    "user@example.com",
                    f"{notice_kind} body",
                    metadata={"session_id": "email-session-100", "delivery_kind": notice_kind},
                )
            )
            self.assertTrue(result.success)

        adapter._send_email.assert_not_called()

    def test_suppression_audit_has_routing_metadata_but_no_body_or_secret(self):
        adapter = _make_adapter(extra={"read_only": True})
        adapter._thread_context["user@example.com"] = {
            "subject": "Weekly report",
            "message_id": "<original@example.com>",
        }

        with self.assertLogs("plugins.platforms.email.adapter", level=logging.INFO) as logs:
            asyncio.run(
                adapter.send(
                    "user@example.com",
                    "body which must never reach logs",
                    metadata={"session_id": "email-session-42", "token": "secret-value"},
                )
            )

        rendered = "\n".join(logs.output)
        self.assertIn("recipient=user@example.com", rendered)
        self.assertIn("subject=Weekly report", rendered)
        self.assertIn("session=email-session-42", rendered)
        self.assertNotIn("body which must never reach logs", rendered)
        self.assertNotIn("secret-value", rendered)


if __name__ == "__main__":
    unittest.main()
