"""Review-first outbound policy for the email adapter.

Covers the delivery disposition contract:

- Only an authenticated, allowlisted sender's reply goes out via SMTP.
- Every other reply is appended to the drafts mailbox and never touches SMTP.
- Trust is bound to the specific inbound message (Message-ID), not the sender.
- Gateway-internal sends (cron/notifications) send only to home/allowlisted
  addresses; everything anchor-less without provenance fails closed.
- Sent mail is archived to the Sent mailbox via IMAP APPEND (no Bcc-to-self).
"""

import email as email_lib
import os
import unittest
from unittest.mock import MagicMock, patch

ARMEN = "armenlsuny@gmail.com"
STRANGER = "someone@example.org"

_BASE_ENV = {
    "EMAIL_ADDRESS": "jordan@goodgravel.com",
    "EMAIL_PASSWORD": "secret",
    "EMAIL_IMAP_HOST": "imap.test.com",
    "EMAIL_IMAP_PORT": "993",
    "EMAIL_SMTP_HOST": "smtp.test.com",
    "EMAIL_SMTP_PORT": "587",
    "EMAIL_HOME_ADDRESS": ARMEN,
}


def _make_adapter(**extra):
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter

    merged = {
        "outbound_policy": "review_first",
        "auto_send_authenticated_senders": [ARMEN],
    }
    merged.update(extra)
    with patch.dict(os.environ, _BASE_ENV, clear=False):
        adapter = EmailAdapter(PlatformConfig(enabled=True, extra=merged))
    return adapter


def _mock_imap():
    imap = MagicMock()
    imap.capability.return_value = ("OK", [b"IMAP4rev1 UIDPLUS"])
    imap.append.return_value = ("OK", [b"APPEND completed"])
    return imap


class _DeliveryHarness(unittest.TestCase):
    """Common setup: adapter + patched SMTP/IMAP, with trust pre-recorded."""

    def _deliver(self, adapter, to_addr, *, reply_to=None, metadata=None,
                 body="Reply text", append_status="OK"):
        """Run _send_email under patched transports; return (smtp, imap, result)."""
        smtp = MagicMock()
        imap = _mock_imap()
        if append_status != "OK":
            imap.append.return_value = (append_status, [b"failed"])
        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=imap), \
             patch("smtplib.SMTP", return_value=smtp):
            result = adapter._send_email(to_addr, body, reply_to, metadata)
        return smtp, imap, result


class TestReviewFirstRouting(_DeliveryHarness):
    def test_authenticated_allowlisted_reply_sends_once(self):
        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<msg1@gmail.com>", True)
        smtp, imap, (msg_id, disposition) = self._deliver(
            adapter, ARMEN, reply_to="<msg1@gmail.com>"
        )
        self.assertEqual(disposition, "sent")
        smtp.send_message.assert_called_once()
        # Archived to Sent, never to Drafts.
        self.assertEqual(imap.append.call_count, 1)
        self.assertEqual(imap.append.call_args.args[0], "Sent")

    def test_allowlisted_but_unauthenticated_drafts(self):
        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<forged@gmail.com>", False)
        smtp, imap, (msg_id, disposition) = self._deliver(
            adapter, ARMEN, reply_to="<forged@gmail.com>"
        )
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()
        self.assertEqual(imap.append.call_args.args[0], "Drafts")

    def test_authenticated_non_allowlisted_drafts(self):
        adapter = _make_adapter()
        adapter._record_message_trust(STRANGER, "<s1@example.org>", True)
        smtp, imap, (msg_id, disposition) = self._deliver(
            adapter, STRANGER, reply_to="<s1@example.org>"
        )
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()

    def test_case_normalized_allowlist_match(self):
        adapter = _make_adapter(
            auto_send_authenticated_senders=["ArmenLSUNY@Gmail.COM"]
        )
        adapter._record_message_trust(ARMEN, "<msg2@gmail.com>", True)
        smtp, _, (_, disposition) = self._deliver(
            adapter, "Armenlsuny@gmail.com", reply_to="<msg2@gmail.com>"
        )
        self.assertEqual(disposition, "sent")
        smtp.send_message.assert_called_once()

    def test_empty_allowlist_drafts_everything(self):
        adapter = _make_adapter(auto_send_authenticated_senders=[])
        adapter._record_message_trust(ARMEN, "<msg3@gmail.com>", True)
        smtp, _, (_, disposition) = self._deliver(
            adapter, ARMEN, reply_to="<msg3@gmail.com>"
        )
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()

    def test_malformed_allowlist_fails_closed(self):
        adapter = _make_adapter(
            auto_send_authenticated_senders={"not": "a list"}
        )
        adapter._record_message_trust(ARMEN, "<msg4@gmail.com>", True)
        smtp, _, (_, disposition) = self._deliver(
            adapter, ARMEN, reply_to="<msg4@gmail.com>"
        )
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()

    def test_unknown_policy_value_fails_closed_to_review_first(self):
        adapter = _make_adapter(outbound_policy="reviewfirst-typo")
        self.assertEqual(adapter._outbound_policy, "review_first")

    def test_unknown_agent_initiated_sends_fails_closed_to_draft(self):
        adapter = _make_adapter(agent_initiated_sends="maybe")
        self.assertEqual(adapter._agent_initiated_sends, "draft")

    def test_draft_append_failure_raises_and_never_falls_back_to_smtp(self):
        adapter = _make_adapter()
        adapter._record_message_trust(STRANGER, "<s2@example.org>", True)
        smtp = MagicMock()
        imap = _mock_imap()
        imap.append.return_value = ("NO", [b"quota exceeded"])
        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=imap), \
             patch("smtplib.SMTP", return_value=smtp):
            with self.assertRaises(RuntimeError):
                adapter._send_email(STRANGER, "hi", "<s2@example.org>", None)
        smtp.send_message.assert_not_called()

    def test_trust_bound_to_message_not_sender(self):
        """A later failed-auth message must not downgrade an in-flight reply
        anchored to an earlier authenticated message — and vice versa."""
        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<real@gmail.com>", True)
        adapter._record_message_trust(ARMEN, "<forged@gmail.com>", False)
        # Thread context points at the newest (forged) message.
        adapter._thread_context[ARMEN] = {
            "subject": "s", "message_id": "<forged@gmail.com>",
        }
        smtp, _, (_, disposition) = self._deliver(
            adapter, ARMEN, reply_to="<real@gmail.com>"
        )
        self.assertEqual(disposition, "sent")
        smtp2, _, (_, disposition2) = self._deliver(
            adapter, ARMEN, reply_to="<forged@gmail.com>"
        )
        self.assertEqual(disposition2, "drafted")
        smtp2.send_message.assert_not_called()

    def test_anchor_without_trust_record_drafts(self):
        """Post-restart shape: a reply anchor with no in-memory record must
        draft — permission is never inferred from the address alone."""
        adapter = _make_adapter()
        smtp, _, (_, disposition) = self._deliver(
            adapter, ARMEN, reply_to="<lost-after-restart@gmail.com>"
        )
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()

    def test_metadata_anchor_resolves_like_explicit_reply_to(self):
        """Send paths without a reply_to parameter carry the anchor in metadata
        (stamped by the gateway's thread metadata) and resolve identically."""
        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<meta@gmail.com>", True)
        smtp, _, (_, disposition) = self._deliver(
            adapter, ARMEN, metadata={"reply_to_message_id": "<meta@gmail.com>"}
        )
        self.assertEqual(disposition, "sent")

    def test_thread_context_never_supplies_the_anchor(self):
        """Per-sender thread context tracks the sender's NEWEST message, so it must
        never stand in for a missing anchor: a trusted record reachable only via
        thread context drafts (fail-closed), never sends."""
        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<ctx@gmail.com>", True)
        adapter._thread_context[ARMEN] = {
            "subject": "s", "message_id": "<ctx@gmail.com>",
        }
        smtp, _, (_, disposition) = self._deliver(adapter, ARMEN)
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()

    def test_attachment_send_binds_to_explicit_anchor_not_ctx_race(self):
        """The wrongful-send race: thread context was overwritten by a NEWER trusted
        message while the turn replying to an untrusted one was in flight. The
        attachment path must bind to its own (metadata) anchor — drafting — and a
        metadata anchor to the trusted message must send."""
        import tempfile

        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<forged@gmail.com>", False)
        adapter._record_message_trust(ARMEN, "<real@gmail.com>", True)
        adapter._thread_context[ARMEN] = {
            "subject": "s", "message_id": "<real@gmail.com>",  # newest = trusted
        }
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG")
            path = f.name
        try:
            for anchor, expected in (
                ("<forged@gmail.com>", "drafted"),  # the in-flight forged turn
                ("<real@gmail.com>", "sent"),
                (None, "drafted"),                  # anchor-less: ctx grants nothing
            ):
                smtp = MagicMock()
                imap = _mock_imap()
                metadata = {"reply_to_message_id": anchor} if anchor else None
                with patch.dict(os.environ, _BASE_ENV, clear=False), \
                     patch("imaplib.IMAP4_SSL", return_value=imap), \
                     patch("smtplib.SMTP", return_value=smtp):
                    _, disposition = adapter._send_email_with_attachments(
                        ARMEN, "here you go", [path], metadata
                    )
                self.assertEqual(disposition, expected, f"anchor={anchor}")
        finally:
            os.unlink(path)

    def test_gateway_thread_metadata_carries_email_anchor(self):
        """The gateway side of the contract: _thread_metadata_for_source stamps the
        reply anchor into metadata for email sources."""
        from gateway.platforms.base import _thread_metadata_for_source
        from gateway.session import Platform as SessionPlatform, SessionSource

        src = SessionSource(platform=SessionPlatform.EMAIL, chat_id=ARMEN, chat_type="dm")
        meta = _thread_metadata_for_source(src, "<anchor@gmail.com>")
        self.assertEqual(meta.get("reply_to_message_id"), "<anchor@gmail.com>")

    def test_gateway_internal_send_to_allowlisted_home_sends(self):
        adapter = _make_adapter()
        smtp, _, (_, disposition) = self._deliver(
            adapter, ARMEN, metadata={"gateway_internal_send": True}
        )
        self.assertEqual(disposition, "sent")
        smtp.send_message.assert_called_once()

    def test_gateway_internal_send_to_stranger_drafts(self):
        adapter = _make_adapter()
        smtp, imap, (_, disposition) = self._deliver(
            adapter, STRANGER, metadata={"gateway_internal_send": True}
        )
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()

    def test_gateway_internal_send_to_non_allowlisted_home_drafts(self):
        """EMAIL_HOME_ADDRESS alone must not authorize transmission — a home
        address that is not on the auto-send allowlist gets drafts."""
        adapter = _make_adapter(
            auto_send_authenticated_senders=["someoneelse@x.com"]
        )
        smtp, imap, (_, disposition) = self._deliver(
            adapter, ARMEN, metadata={"gateway_internal_send": True}
        )
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()

    def test_image_send_preserves_security_metadata(self):
        """send_image must carry metadata (gateway_internal_send) through to
        the delivery decision like every other path."""
        import asyncio

        adapter = _make_adapter()
        smtp = MagicMock()
        imap = _mock_imap()
        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=imap), \
             patch("smtplib.SMTP", return_value=smtp):
            result = asyncio.run(adapter.send_image(
                ARMEN, "https://x/img.png", caption="chart",
                metadata={"gateway_internal_send": True},
            ))
        self.assertEqual(result.disposition, "sent")
        smtp.send_message.assert_called_once()

    def test_anchorless_without_provenance_drafts_by_default(self):
        adapter = _make_adapter()
        smtp, _, (_, disposition) = self._deliver(adapter, STRANGER)
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()

    def test_agent_initiated_toggle_does_not_reach_inprocess_sends(self):
        """agent_initiated_sends: send applies only to out-of-process
        standalone sends. Anchor-less in-process sends (e.g. redelivered
        obligations after a restart) draft unconditionally even with the
        toggle flipped."""
        adapter = _make_adapter(agent_initiated_sends="send")
        smtp, _, (_, disposition) = self._deliver(adapter, STRANGER)
        self.assertEqual(disposition, "drafted")
        smtp.send_message.assert_not_called()

    def test_direct_policy_preserves_legacy_send(self):
        adapter = _make_adapter(outbound_policy="")
        self.assertEqual(adapter._outbound_policy, "direct")
        smtp, imap, (_, disposition) = self._deliver(adapter, STRANGER)
        self.assertEqual(disposition, "sent")
        smtp.send_message.assert_called_once()
        # Legacy behavior: no IMAP traffic at all (save_sent defaults off).
        imap.append.assert_not_called()

    def test_all_send_paths_share_the_policy(self):
        import tempfile

        adapter = _make_adapter()
        adapter._record_message_trust(STRANGER, "<s3@example.org>", True)
        adapter._thread_context[STRANGER] = {
            "subject": "s", "message_id": "<s3@example.org>",
        }
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"attachment body")
            path = f.name
        try:
            for call in (
                lambda: adapter._send_email(STRANGER, "text", None, None),
                lambda: adapter._send_email_with_attachments(
                    STRANGER, "text", [path], None
                ),
                lambda: adapter._send_email_with_attachment(
                    STRANGER, "text", path, None, None, None
                ),
            ):
                smtp = MagicMock()
                imap = _mock_imap()
                with patch.dict(os.environ, _BASE_ENV, clear=False), \
                     patch("imaplib.IMAP4_SSL", return_value=imap), \
                     patch("smtplib.SMTP", return_value=smtp):
                    _, disposition = call()
                self.assertEqual(disposition, "drafted")
                smtp.send_message.assert_not_called()
                self.assertEqual(imap.append.call_args.args[0], "Drafts")
        finally:
            os.unlink(path)


class TestDraftFidelity(_DeliveryHarness):
    def test_draft_append_targets_configured_mailbox_with_draft_flag(self):
        adapter = _make_adapter(drafts_mailbox="INBOX.Drafts")
        smtp, imap, (_, disposition) = self._deliver(adapter, STRANGER)
        self.assertEqual(disposition, "drafted")
        mailbox, flags = imap.append.call_args.args[0], imap.append.call_args.args[1]
        self.assertEqual(mailbox, "INBOX.Drafts")
        self.assertIn(r"\Draft", flags)

    def test_draft_preserves_headers_body_and_threading(self):
        adapter = _make_adapter()
        adapter._thread_context[STRANGER] = {
            "subject": "Order inquiry", "message_id": "<orig@example.org>",
        }
        smtp, imap, (msg_id, disposition) = self._deliver(
            adapter, STRANGER, reply_to="<orig@example.org>", body="Proposed reply"
        )
        raw = imap.append.call_args.args[3]
        parsed = email_lib.message_from_bytes(raw)
        self.assertEqual(parsed["From"], "jordan@goodgravel.com")
        self.assertEqual(parsed["To"], STRANGER)
        self.assertEqual(parsed["Subject"], "Re: Order inquiry")
        self.assertEqual(parsed["In-Reply-To"], "<orig@example.org>")
        self.assertEqual(parsed["References"], "<orig@example.org>")
        self.assertEqual(parsed["Message-ID"], msg_id)
        self.assertTrue(parsed["Date"])
        self.assertIn(
            b"Proposed reply", parsed.get_payload(0).get_payload(decode=True)
        )

    def test_draft_preserves_attachments(self):
        import tempfile

        adapter = _make_adapter()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-fake")
            path = f.name
        try:
            smtp = MagicMock()
            imap = _mock_imap()
            with patch.dict(os.environ, _BASE_ENV, clear=False), \
                 patch("imaplib.IMAP4_SSL", return_value=imap), \
                 patch("smtplib.SMTP", return_value=smtp):
                _, disposition = adapter._send_email_with_attachment(
                    STRANGER, "see attached", path, "quote.pdf", None, None
                )
            self.assertEqual(disposition, "drafted")
            raw = imap.append.call_args.args[3]
            parsed = email_lib.message_from_bytes(raw)
            parts = [p for p in parsed.walk()
                     if "attachment" in str(p.get("Content-Disposition", ""))]
            self.assertEqual(len(parts), 1)
            self.assertIn("quote.pdf", parts[0]["Content-Disposition"])
            self.assertEqual(parts[0].get_payload(decode=True), b"%PDF-fake")
        finally:
            os.unlink(path)

    def test_send_result_reports_drafted(self):
        import asyncio

        adapter = _make_adapter()
        smtp = MagicMock()
        imap = _mock_imap()
        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=imap), \
             patch("smtplib.SMTP", return_value=smtp):
            result = asyncio.run(adapter.send(STRANGER, "hello"))
        self.assertTrue(result.success)
        self.assertEqual(result.disposition, "drafted")

    def test_send_result_reports_sent(self):
        import asyncio

        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<ok@gmail.com>", True)
        smtp = MagicMock()
        imap = _mock_imap()
        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=imap), \
             patch("smtplib.SMTP", return_value=smtp):
            result = asyncio.run(adapter.send(ARMEN, "hello", reply_to="<ok@gmail.com>"))
        self.assertTrue(result.success)
        self.assertEqual(result.disposition, "sent")


class TestSentArchival(_DeliveryHarness):
    def test_sent_mail_archived_once_with_seen_flag(self):
        adapter = _make_adapter(sent_mailbox="INBOX.Sent")
        adapter._record_message_trust(ARMEN, "<a@gmail.com>", True)
        smtp, imap, (_, disposition) = self._deliver(
            adapter, ARMEN, reply_to="<a@gmail.com>"
        )
        self.assertEqual(disposition, "sent")
        smtp.send_message.assert_called_once()
        self.assertEqual(imap.append.call_count, 1)
        mailbox, flags = imap.append.call_args.args[0], imap.append.call_args.args[1]
        self.assertEqual(mailbox, "INBOX.Sent")
        self.assertIn(r"\Seen", flags)

    def test_no_bcc_to_self_anywhere(self):
        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<b@gmail.com>", True)
        smtp, imap, _ = self._deliver(adapter, ARMEN, reply_to="<b@gmail.com>")
        sent_msg = smtp.send_message.call_args.args[0]
        self.assertIsNone(sent_msg["Bcc"])
        raw = imap.append.call_args.args[3]
        self.assertNotIn(b"Bcc:", raw)

    def test_archive_failure_still_reports_sent_without_smtp_retry(self):
        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<c@gmail.com>", True)
        smtp, imap, (_, disposition) = self._deliver(
            adapter, ARMEN, reply_to="<c@gmail.com>", append_status="NO"
        )
        self.assertEqual(disposition, "sent")
        smtp.send_message.assert_called_once()

    def test_save_sent_disabled_skips_archival(self):
        adapter = _make_adapter(save_sent=False)
        adapter._record_message_trust(ARMEN, "<d@gmail.com>", True)
        smtp, imap, (_, disposition) = self._deliver(
            adapter, ARMEN, reply_to="<d@gmail.com>"
        )
        self.assertEqual(disposition, "sent")
        imap.append.assert_not_called()

    def test_self_message_never_dispatched(self):
        """The poller's self-skip means archived copies can't loop back in."""
        import asyncio

        adapter = _make_adapter()
        handled = []
        adapter.handle_message = lambda event: handled.append(event)
        asyncio.run(adapter._dispatch_message({
            "sender_addr": "jordan@goodgravel.com",
            "sender_name": "Hermes",
            "subject": "Re: anything",
            "message_id": "<self@goodgravel.com>",
            "in_reply_to": "",
            "body": "archived copy",
            "attachments": [],
            "date": "",
            "sender_authenticated": False,
            "auth_reason": "",
        }))
        self.assertEqual(handled, [])


class TestStrictAuthservPinning(unittest.TestCase):
    """Review-first Authentication-Results evaluation must be pinned to an
    exact authserv-id AND position-verified: only headers above the first
    transport Received header — the region only the receiving server can
    write — may authenticate a sender. Every ambiguity fails closed."""

    PASS_AR = "purelymail.com; dmarc=pass header.from=gmail.com"
    RECEIVED = "from mx.google.com by mailserver.purelymail.com; Thu, 4 Sep 2026 10:00:00 +0000"

    def _verify(self, ordered_headers, *, authserv_id="purelymail.com",
                strict=True, sender=ARMEN):
        from email.mime.text import MIMEText

        from plugins.platforms.email.adapter import _verify_sender_authentication

        msg = MIMEText("body", "plain", "utf-8")
        msg["From"] = sender
        for name, value in ordered_headers:
            msg[name] = value
        return _verify_sender_authentication(
            msg, sender, authserv_id=authserv_id, strict=strict
        )

    def test_missing_authserv_pin_fails_closed(self):
        ok, reason = self._verify(
            [("Authentication-Results", self.PASS_AR)], authserv_id=""
        )
        self.assertFalse(ok)
        self.assertIn("authserv_id", reason)

    def test_genuine_header_above_received_chain_authenticates(self):
        ok, reason = self._verify([
            ("Authentication-Results", self.PASS_AR),
            ("Received", self.RECEIVED),
        ])
        self.assertTrue(ok)
        self.assertEqual(reason, "dmarc=pass")

    def test_sole_forged_header_claiming_exact_pin_is_rejected(self):
        """The reviewer-reproduced attack: the receiving server stamped no
        header, and the ONLY Authentication-Results claims the exact pinned
        authserv-id — but it travels in the attacker's header block, below
        the transport Received chain, so it must not authenticate."""
        ok, reason = self._verify([
            ("Received", self.RECEIVED),
            ("Authentication-Results", self.PASS_AR),  # attacker-authored
        ])
        self.assertFalse(ok)
        self.assertIn("above the transport Received chain", reason)

    def test_forged_pin_below_received_cannot_shadow_genuine_above(self):
        """A forged exact-pin duplicate below the Received chain is ignored
        rather than making the genuine header ambiguous (no draft-DoS)."""
        ok, reason = self._verify([
            ("Authentication-Results", self.PASS_AR),        # server-stamped
            ("Received", self.RECEIVED),
            ("Authentication-Results", self.PASS_AR),        # attacker copy
        ])
        self.assertTrue(ok)
        self.assertEqual(reason, "dmarc=pass")

    def test_forged_header_with_wrong_authserv_id_rejected(self):
        ok, reason = self._verify(
            [("Authentication-Results", "evil.example; dmarc=pass header.from=gmail.com")]
        )
        self.assertFalse(ok)
        # Sanity: the same forged header IS trusted by the legacy unpinned
        # path — which is exactly why review_first requires the pin.
        ok_legacy, _ = self._verify(
            [("Authentication-Results", "evil.example; dmarc=pass header.from=gmail.com")],
            authserv_id="", strict=False,
        )
        self.assertTrue(ok_legacy)

    def test_duplicated_pinned_headers_above_received_are_ambiguous(self):
        ok, reason = self._verify([
            ("Authentication-Results", self.PASS_AR),
            ("Authentication-Results", self.PASS_AR),
            ("Received", self.RECEIVED),
        ])
        self.assertFalse(ok)
        self.assertIn("ambiguous", reason)

    def test_subdomain_authserv_id_is_not_exact_and_fails_strict(self):
        """Strict pinning requires the literal stamped token — organizational
        alignment is a legacy-only convenience."""
        ok, reason = self._verify(
            [("Authentication-Results",
              "mailserver.purelymail.com; dmarc=pass header.from=gmail.com")]
        )
        self.assertFalse(ok)
        ok_legacy, _ = self._verify(
            [("Authentication-Results",
              "mailserver.purelymail.com; dmarc=pass header.from=gmail.com")],
            strict=False,
        )
        self.assertTrue(ok_legacy)

    def test_malformed_headers_do_not_match_pin(self):
        ok, _ = self._verify([
            ("Authentication-Results", "; dmarc=pass header.from=gmail.com"),
            ("Authentication-Results", "dmarc=pass"),
            ("Received", self.RECEIVED),
        ])
        self.assertFalse(ok)

    def test_exact_pin_without_received_boundary_fails_closed(self):
        """A message with NO Received header offers no transport boundary
        to verify header provenance against — even an exact-pin passing
        header must not authenticate it (every SMTP-delivered message
        carries at least the receiving server's own Received line)."""
        ok, reason = self._verify([
            ("Authentication-Results", self.PASS_AR),
        ])
        self.assertFalse(ok)
        self.assertIn("no transport Received header", reason)

    def test_pinned_but_failed_dmarc_still_fails(self):
        ok, _ = self._verify([
            ("Authentication-Results",
             "purelymail.com; dmarc=fail header.from=gmail.com"),
            ("Received", self.RECEIVED),
        ])
        self.assertFalse(ok)

    def test_dkim_pass_with_only_header_from_fails_strict(self):
        """header.from is the attacker-visible identity, not a signing identity — strict
        DKIM accepts only header.d (identity semantics per #56608, enforced here because
        the verdict becomes SMTP send authority)."""
        ok, _ = self._verify([
            ("Authentication-Results", "purelymail.com; dkim=pass header.from=gmail.com"),
            ("Received", self.RECEIVED),
        ])
        self.assertFalse(ok)

    def test_aligned_scoped_dkim_authenticates_strict(self):
        ok, reason = self._verify([
            ("Authentication-Results", "purelymail.com; dkim=pass header.d=gmail.com"),
            ("Received", self.RECEIVED),
        ])
        self.assertTrue(ok)
        self.assertEqual(reason, "dkim=pass aligned")

    def test_cross_method_property_leak_fails_strict(self):
        """A header.d recorded in the SPF clause proves nothing about DKIM: strict
        properties are scoped to the clause that produced them (per #56608)."""
        ok, _ = self._verify([
            ("Authentication-Results",
             "purelymail.com; spf=pass smtp.mailfrom=evil.example header.d=gmail.com; dkim=pass"),
            ("Received", self.RECEIVED),
        ])
        self.assertFalse(ok)

    def test_duplicate_method_clauses_are_ambiguous_strict(self):
        """Two dmarc clauses inside one server-stamped value cannot be disambiguated —
        that method fails closed even when both claim pass."""
        ok, _ = self._verify([
            ("Authentication-Results",
             "purelymail.com; dmarc=pass header.from=gmail.com; dmarc=pass header.from=gmail.com"),
            ("Received", self.RECEIVED),
        ])
        self.assertFalse(ok)

    def test_dmarc_pass_for_misaligned_domain_never_authenticates(self):
        """DMARC vouches only for the identity it evaluated (header.from). A
        legitimately-passing message for evil.example must not authenticate a
        gmail.com From: — in strict OR legacy mode."""
        headers = [
            ("Authentication-Results",
             "purelymail.com; dmarc=pass header.from=evil.example"),
            ("Received", self.RECEIVED),
        ]
        ok, _ = self._verify(headers)
        self.assertFalse(ok)
        ok_legacy, _ = self._verify(headers, strict=False)
        self.assertFalse(ok_legacy)

    def test_dmarc_pass_without_recorded_header_from_fails_strict_only(self):
        """Strict mode requires the evaluated identity to be recorded and aligned;
        legacy keeps accepting servers that omit header.from."""
        headers = [
            ("Authentication-Results", "purelymail.com; dmarc=pass"),
            ("Received", self.RECEIVED),
        ]
        ok, _ = self._verify(headers)
        self.assertFalse(ok)
        ok_legacy, _ = self._verify(headers, strict=False)
        self.assertTrue(ok_legacy)

    def test_display_name_spoof_parses_to_real_sender(self):
        """'"x <victim>" <attacker>' must extract the attacker's real address —
        parseaddr semantics, not first-angle-bracket regex."""
        from plugins.platforms.email.adapter import _extract_email_address

        self.assertEqual(
            _extract_email_address(f'"attacker <{ARMEN}>" <evil@evil.example>'),
            "evil@evil.example",
        )
        self.assertEqual(_extract_email_address(f"Armen <{ARMEN}>"), ARMEN)
        self.assertEqual(_extract_email_address(ARMEN), ARMEN)


class TestUntrustedAbuseLimits(unittest.TestCase):
    """Rate limits and duplicate suppression fire at dispatch, before any
    model turn is started for an untrusted sender."""

    def _dispatch_env(self):
        env = dict(_BASE_ENV)
        env["EMAIL_ALLOW_ALL_USERS"] = "true"
        return env

    def _msg(self, sender, message_id, authenticated=False):
        return {
            "sender_addr": sender,
            "sender_name": "Sender",
            "subject": "Hello",
            "message_id": message_id,
            "in_reply_to": "",
            "body": "hi",
            "attachments": [],
            "date": "",
            "sender_authenticated": authenticated,
            "auth_reason": "",
        }

    def _run_dispatch(self, adapter, messages):
        import asyncio

        handled = []

        async def handle(event):
            handled.append(event)

        # Stub handle_message directly: the base-class implementation spawns background session
        # tasks, and this class tests the dispatch gate, not the session machinery.
        adapter.handle_message = handle
        with patch.dict(os.environ, self._dispatch_env(), clear=False):
            for m in messages:
                asyncio.run(adapter._dispatch_message(m))
        return handled

    def test_duplicate_message_id_dispatched_once(self):
        adapter = _make_adapter()
        handled = self._run_dispatch(adapter, [
            self._msg(STRANGER, "<dup@example.org>"),
            self._msg(STRANGER, "<dup@example.org>"),
        ])
        self.assertEqual(len(handled), 1)

    def test_per_sender_hourly_limit(self):
        adapter = _make_adapter(untrusted_draft_limit_per_sender_hour=2)
        handled = self._run_dispatch(adapter, [
            self._msg(STRANGER, f"<m{i}@example.org>") for i in range(5)
        ])
        self.assertEqual(len(handled), 2)

    def test_global_hourly_limit(self):
        adapter = _make_adapter(untrusted_draft_limit_global_hour=3)
        senders = [f"user{i}@example{i}.org" for i in range(6)]
        handled = self._run_dispatch(adapter, [
            self._msg(s, f"<g{i}@example.org>") for i, s in enumerate(senders)
        ])
        self.assertEqual(len(handled), 3)

    def test_trusted_sender_not_rate_limited(self):
        adapter = _make_adapter(
            untrusted_draft_limit_per_sender_hour=1,
            untrusted_draft_limit_global_hour=1,
        )
        handled = self._run_dispatch(adapter, [
            self._msg(ARMEN, f"<a{i}@gmail.com>", authenticated=True)
            for i in range(4)
        ])
        self.assertEqual(len(handled), 4)

    def test_oversized_inbound_body_truncated(self):
        adapter = _make_adapter()
        msg = self._msg(STRANGER, "<big@example.org>")
        msg["body"] = "x" * 120_000
        handled = self._run_dispatch(adapter, [msg])
        self.assertEqual(len(handled), 1)
        self.assertLess(len(handled[0].text), 60_000)
        self.assertIn("truncated by gateway", handled[0].text)

    def test_malformed_limit_config_falls_back_to_default(self):
        adapter = _make_adapter(untrusted_draft_limit_per_sender_hour="lots")
        self.assertEqual(adapter._untrusted_draft_limit_per_sender_hour, 4)
        adapter = _make_adapter(untrusted_draft_limit_global_hour=-5)
        self.assertEqual(adapter._untrusted_draft_limit_global_hour, 20)


class TestProvenanceMinting(unittest.TestCase):
    """gateway_internal_send is send AUTHORITY under review_first, so only gateway-OWNED
    outbound work (home-channel lifecycle broadcasts, DeliveryRouter delivery) may mint it.
    The shared status helper — which also wraps turn-local heartbeats/status/progress —
    must never mint it (reviewer blocker on PR #103977)."""

    def test_shared_status_helper_never_mints_email_provenance(self):
        from gateway.run import _non_conversational_metadata
        from gateway.session import Platform as SessionPlatform

        meta = _non_conversational_metadata({"thread_id": "x"}, platform=SessionPlatform.EMAIL)
        self.assertNotIn("gateway_internal_send", meta or {})
        self.assertIsNone(_non_conversational_metadata(None, platform=SessionPlatform.EMAIL))

    def test_lifecycle_helper_mints_for_email_only(self):
        from gateway.run import _gateway_internal_send_metadata
        from gateway.session import Platform as SessionPlatform

        meta = _gateway_internal_send_metadata(None, platform=SessionPlatform.EMAIL)
        self.assertIs(meta.get("gateway_internal_send"), True)
        self.assertIsNone(_gateway_internal_send_metadata(None, platform=SessionPlatform.TELEGRAM))

    def test_interim_sends_suppressed_under_review_first(self):
        """A mid-turn heartbeat/status send (marked _interim_send) produces neither SMTP
        traffic nor a draft — even fully anchored and trusted, its authority is the
        in-flight turn's and email is not a streaming surface."""
        import asyncio

        adapter = _make_adapter()
        adapter._record_message_trust(ARMEN, "<hb@gmail.com>", True)
        smtp = MagicMock()
        imap = _mock_imap()
        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=imap), \
             patch("smtplib.SMTP", return_value=smtp):
            result = asyncio.run(adapter.send(
                ARMEN, "still on it",
                metadata={"_interim_send": True, "reply_to_message_id": "<hb@gmail.com>"},
            ))
        self.assertTrue(result.success)
        self.assertIsNone(result.disposition)
        smtp.send_message.assert_not_called()
        imap.append.assert_not_called()

    def test_interim_sends_unchanged_under_direct_policy(self):
        import asyncio

        adapter = _make_adapter(outbound_policy="")
        smtp = MagicMock()
        imap = _mock_imap()
        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=imap), \
             patch("smtplib.SMTP", return_value=smtp):
            result = asyncio.run(adapter.send(STRANGER, "progress", metadata={"_interim_send": True}))
        self.assertTrue(result.success)
        self.assertEqual(result.disposition, "sent")
        smtp.send_message.assert_called_once()


class TestYamlConfigBridge(unittest.TestCase):
    """``platforms.email.<behavior key>`` (the documented top-level shape) must reach
    ``PlatformConfig.extra`` via the plugin's ``apply_yaml_config_fn`` hook — without the
    bridge the review-first block would silently configure nothing."""

    def test_behavior_keys_seed_extra(self):
        from plugins.platforms.email.adapter import _apply_yaml_config

        seeded = _apply_yaml_config({}, {
            "outbound_policy": "review_first",
            "auto_send_authenticated_senders": [ARMEN],
            "authserv_id": "purelymail.com",
            "skip_attachments": True,
            "agent_initiated_sends": "draft",
            "address": "jordan@goodgravel.com",  # connection key, not bridged here
            "unrelated": "x",
        })
        self.assertEqual(seeded["outbound_policy"], "review_first")
        self.assertEqual(seeded["auto_send_authenticated_senders"], [ARMEN])
        self.assertEqual(seeded["authserv_id"], "purelymail.com")
        self.assertTrue(seeded["skip_attachments"])
        self.assertEqual(seeded["agent_initiated_sends"], "draft")
        self.assertNotIn("unrelated", seeded)
        self.assertNotIn("address", seeded)

    def test_no_behavior_keys_returns_none(self):
        from plugins.platforms.email.adapter import _apply_yaml_config

        self.assertIsNone(_apply_yaml_config({}, {"address": "a@b.c"}))

    def test_registry_wires_the_hook(self):
        from plugins.platforms.email import adapter as email_mod

        captured = {}

        class _Ctx:
            def register_platform(self, **kwargs):
                captured.update(kwargs)

        email_mod.register(_Ctx())
        self.assertIs(captured.get("apply_yaml_config_fn"), email_mod._apply_yaml_config)


class TestStandaloneSendPolicy(unittest.TestCase):
    def _run(self, chat_id, **extra):
        import asyncio

        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import _standalone_send

        merged = {
            "outbound_policy": "review_first",
            "auto_send_authenticated_senders": [ARMEN],
        }
        merged.update(extra)
        pconfig = PlatformConfig(enabled=True, extra=merged)
        smtp = MagicMock()
        imap = _mock_imap()
        with patch.dict(os.environ, _BASE_ENV, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=imap), \
             patch("smtplib.SMTP", return_value=smtp), \
             patch("smtplib.SMTP_SSL", return_value=smtp):
            result = asyncio.run(_standalone_send(pconfig, chat_id, "hello"))
        return smtp, imap, result

    def test_standalone_to_stranger_drafts(self):
        smtp, imap, result = self._run(STRANGER)
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("disposition"), "drafted")
        smtp.send_message.assert_not_called()
        self.assertEqual(imap.append.call_args.args[0], "Drafts")

    def test_standalone_lacks_provenance_so_even_home_drafts(self):
        """A standalone send is out-of-process and carries no gateway-authored
        provenance; recipient membership (allowlist or home) cannot establish
        trusted origin, so it drafts."""
        smtp, imap, result = self._run(ARMEN)
        self.assertEqual(result.get("disposition"), "drafted")
        smtp.send_message.assert_not_called()

    def test_standalone_send_toggle(self):
        smtp, imap, result = self._run(STRANGER, agent_initiated_sends="send")
        self.assertEqual(result.get("disposition"), "sent")
        smtp.send_message.assert_called_once()

    def test_standalone_direct_policy_unchanged(self):
        smtp, imap, result = self._run(STRANGER, outbound_policy="")
        self.assertEqual(result.get("disposition"), "sent")
        smtp.send_message.assert_called_once()
        imap.append.assert_not_called()


if __name__ == "__main__":
    unittest.main()
