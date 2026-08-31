"""Per-thread session isolation for the email adapter.

Every email from one sender used to collapse into a single session keyed on the
sender address alone, and outbound reply threading was sourced from a single
per-sender slot holding "the last message received from this address". Two
unrelated conversations with the same person therefore shared one context, and a
cron job created in one thread delivered its result into whichever thread with
that sender had been touched most recently.

The fix keys both the session and the outbound reply context on Gmail's
server-computed ``X-GM-THRID``. These tests mock only the IMAP/SMTP boundary —
never adapter internals — and assert on observable outcomes.
"""

import os
import unittest
import uuid
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch


def _make_adapter(address="hermes@test.com"):
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": address,
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True))


def _raw_email(sender="user@test.com", subject="Hello", message_id=None):
    msg = MIMEText("Test body", "plain", "utf-8")
    msg["From"] = sender
    msg["Subject"] = subject
    msg["Message-ID"] = message_id or f"<{uuid.uuid4().hex[:8]}@test.com>"
    return msg.as_bytes()


def _is_thrid_fetch(args):
    return any("X-GM-THRID" in str(a) for a in args)


class TestThreadIdFetch(unittest.TestCase):
    """X-GM-THRID is fetched over the existing IMAP connection and parsed."""

    def _fetch(self, thrid_response):
        adapter = _make_adapter()
        raw = _raw_email()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                if _is_thrid_fetch(args):
                    return thrid_response
                return ("OK", [(b"1", raw)])
            return ("NO", [])

        mock_imap = MagicMock()

        mock_imap.capabilities = ("IMAP4REV1", "X-GM-EXT-1")
        mock_imap.uid.side_effect = uid_handler
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            return adapter, adapter._fetch_new_messages(lambda candidate: True)

    def test_well_formed_response_yields_thread_id(self):
        _, results = self._fetch(
            ("OK", [b"1 (X-GM-THRID 1795604436429434944 UID 1)"])
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["thread_id"], "1795604436429434944")

    def test_error_status_is_a_failure(self):
        """A non-OK FETCH leaves the message unprocessed and UNSEEN."""
        adapter, results = self._fetch(("NO", [b"nope"]))
        self.assertEqual(results, [])
        self.assertNotIn(b"1", adapter._seen_uids)

    def test_empty_response_is_a_failure(self):
        adapter, results = self._fetch(("OK", []))
        self.assertEqual(results, [])
        self.assertNotIn(b"1", adapter._seen_uids)

    def test_response_without_thrid_is_a_failure(self):
        """A well-formed FETCH carrying no X-GM-THRID is treated as a failure."""
        adapter, results = self._fetch(("OK", [b"1 (UID 1)"]))
        self.assertEqual(results, [])
        self.assertNotIn(b"1", adapter._seen_uids)

    def test_thread_id_fetched_before_rfc822(self):
        """The RFC822 fetch sets \\Seen, so the thread lookup must precede it.

        Otherwise a message whose thread never resolves has already lost the
        UNSEEN flag that makes a retry possible.
        """
        adapter = _make_adapter()
        specs = []

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                specs.append(str(args[-1]))
                if _is_thrid_fetch(args):
                    return ("OK", [b"1 (X-GM-THRID 42 UID 1)"])
                return ("OK", [(b"1", _raw_email())])
            return ("NO", [])

        mock_imap = MagicMock()

        mock_imap.capabilities = ("IMAP4REV1", "X-GM-EXT-1")
        mock_imap.uid.side_effect = uid_handler
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            adapter._fetch_new_messages(lambda candidate: True)

        # main's fetch does a bounded header preflight first, so assert the
        # ordering that matters: the thread lookup precedes the RFC822 fetch.
        thrid_at = next(i for i, s in enumerate(specs) if "X-GM-THRID" in s)
        rfc822_at = next(i for i, s in enumerate(specs) if "RFC822" in s)
        self.assertLess(thrid_at, rfc822_at)


class TestThreadLookupRetryAndGiveUp(unittest.TestCase):
    """Five strikes: retry, then notify the sender and drop."""

    def _adapter_that_always_fails_thrid(self):
        adapter = _make_adapter()
        raw = _raw_email(sender="user@test.com")

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"7"])
            if command == "fetch":
                if _is_thrid_fetch(args):
                    return ("NO", [])
                if "HEADER.FIELDS" in str(args[-1]):
                    return ("OK", [(b"7 (BODY[HEADER.FIELDS (FROM)]",
                                    b"From: user@test.com\r\n\r\n")])
                return ("OK", [(b"7", raw)])
            if command == "store":
                return ("OK", [b"7"])
            return ("NO", [])

        mock_imap = MagicMock()

        mock_imap.capabilities = ("IMAP4REV1", "X-GM-EXT-1")
        mock_imap.uid.side_effect = uid_handler
        return adapter, mock_imap

    def test_stays_unseen_through_first_four_failures(self):
        adapter, mock_imap = self._adapter_that_always_fails_thrid()

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
                patch("smtplib.SMTP") as mock_smtp:
            for cycle in range(1, 5):
                results = adapter._fetch_new_messages(lambda candidate: True)
                self.assertEqual(results, [], f"cycle {cycle} should yield nothing")
                self.assertNotIn(
                    b"7", adapter._seen_uids,
                    f"UID must stay retryable after failure {cycle}",
                )
                self.assertEqual(adapter._thrid_failures[b"7"], cycle)

            # No notification and no \Seen store while still retrying.
            mock_smtp.assert_not_called()
            store_calls = [
                c for c in mock_imap.uid.call_args_list if c[0][0] == "store"
            ]
            self.assertEqual(store_calls, [])

    def test_fifth_failure_notifies_sender_marks_seen_and_drops(self):
        adapter, mock_imap = self._adapter_that_always_fails_thrid()

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
                patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            with self.assertLogs(
                "plugins.platforms.email.adapter", level="ERROR"
            ) as logs:
                for _ in range(5):
                    results = adapter._fetch_new_messages(lambda candidate: True)

            self.assertEqual(results, [])

            # Sender was notified.
            self.assertTrue(mock_server.send_message.called)
            sent = mock_server.send_message.call_args[0][0]
            self.assertEqual(sent["To"], "user@test.com")
            self.assertNotIn("Re:", sent["Subject"])

            # Marked seen on the server so it is not retried forever.
            store_calls = [
                c for c in mock_imap.uid.call_args_list if c[0][0] == "store"
            ]
            self.assertEqual(len(store_calls), 1)
            self.assertEqual(store_calls[0][0][1], b"7")
            self.assertIn("Seen", str(store_calls[0][0][3]))

            # Dropped locally too.
            self.assertIn(b"7", adapter._seen_uids)
            self.assertNotIn(b"7", adapter._thrid_failures)

        # Logged under a grep-able prefix, without the message body.
        from plugins.platforms.email.adapter import (
            THREAD_LOOKUP_FAILURE_LOG_PREFIX,
        )
        failure_logs = [
            r for r in logs.output if THREAD_LOOKUP_FAILURE_LOG_PREFIX in r
        ]
        self.assertTrue(failure_logs)
        self.assertIn("user@test.com", failure_logs[0])
        self.assertNotIn("Test body", "\n".join(logs.output))

    def test_success_after_transient_failure_clears_strikes(self):
        """A recovered lookup must not carry its strikes forward."""
        adapter = _make_adapter()
        raw = _raw_email()
        state = {"fail": True}

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"3"])
            if command == "fetch":
                if _is_thrid_fetch(args):
                    if state["fail"]:
                        return ("NO", [])
                    return ("OK", [b"3 (X-GM-THRID 55 UID 3)"])
                return ("OK", [(b"3", raw)])
            return ("NO", [])

        mock_imap = MagicMock()

        mock_imap.capabilities = ("IMAP4REV1", "X-GM-EXT-1")
        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            self.assertEqual(adapter._fetch_new_messages(lambda candidate: True), [])
            self.assertEqual(adapter._thrid_failures[b"3"], 1)

            state["fail"] = False
            results = adapter._fetch_new_messages(lambda candidate: True)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["thread_id"], "55")
        self.assertNotIn(b"3", adapter._thrid_failures)


class TestSessionKeyIsolation(unittest.TestCase):
    """The core regression test: one sender, two threads, two sessions."""

    def test_distinct_threads_produce_distinct_session_keys(self):
        from gateway.session import build_session_key
        from gateway.config import Platform
        from gateway.platforms.base import SessionSource

        def source_for(thread_id):
            return SessionSource(
                platform=Platform.EMAIL,
                chat_id="user@test.com",
                chat_type="dm",
                user_id="user@test.com",
                thread_id=thread_id,
            )

        key_a = build_session_key(source_for("111"))
        key_b = build_session_key(source_for("222"))

        self.assertNotEqual(key_a, key_b)
        self.assertIn("111", key_a)
        self.assertIn("222", key_b)

    def test_dispatch_populates_thread_id_on_the_session_source(self):
        """thread_id must reach SessionSource — the rest of the pipeline
        (build_session_key -> HERMES_SESSION_THREAD_ID -> cron origin) already
        handles it when present."""
        import asyncio

        adapter = _make_adapter()
        captured = []

        async def capture(event):
            captured.append(event)

        adapter.handle_message = capture

        msg_data = {
            "uid": b"1",
            "thread_id": "1795604436429434944",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Quarterly report",
            "message_id": "<a@test.com>",
            "in_reply_to": "",
            "body": "Hi",
            "attachments": [],
            "date": "",
            "sender_authenticated": True,
        }

        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(adapter._dispatch_message(msg_data))

        self.assertEqual(len(captured), 1)
        self.assertEqual(
            captured[0].source.thread_id, "1795604436429434944"
        )

    def test_two_threads_from_one_sender_do_not_share_a_session(self):
        """End-to-end over the IMAP boundary: two concurrently open threads."""
        import asyncio
        from gateway.session import build_session_key

        adapter = _make_adapter()
        # main authenticates the From: domain before dispatch; this fake IMAP
        # cannot serve Authentication-Results, so trust the header here.
        adapter._require_authenticated_sender = False
        sources = []

        async def capture(event):
            sources.append(event.source)

        adapter.handle_message = capture

        bodies = {
            b"1": _raw_email(subject="Invoice", message_id="<one@test.com>"),
            b"2": _raw_email(subject="Lunch", message_id="<two@test.com>"),
        }
        thrids = {b"1": b"1000", b"2": b"2000"}

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1 2"])
            if command == "fetch":
                uid = args[0]
                if _is_thrid_fetch(args):
                    return ("OK", [b"%s (X-GM-THRID %s UID %s)"
                                   % (uid, thrids[uid], uid)])
                return ("OK", [(uid, bodies[uid])])
            return ("NO", [])

        mock_imap = MagicMock()

        mock_imap.capabilities = ("IMAP4REV1", "X-GM-EXT-1")
        mock_imap.uid.side_effect = uid_handler

        async def drive():
            with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
                await adapter._check_inbox()

        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(drive())

        self.assertEqual(len(sources), 2)
        keys = {build_session_key(s) for s in sources}
        self.assertEqual(len(keys), 2, f"threads shared a session: {keys}")


class TestOutboundReplyThreading(unittest.TestCase):
    """In-Reply-To/References must name the thread being replied to."""

    def _adapter_with_two_threads(self):
        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1000", "Invoice", "<one@test.com>"
        )
        adapter._remember_thread_context(
            "user@test.com", "2000", "Lunch", "<two@test.com>"
        )
        return adapter

    def _send(self, adapter, thread_id):
        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            adapter._send_email(
                "user@test.com", "Reply body", None, thread_id=thread_id
            )
            return mock_server.send_message.call_args[0][0]

    def test_each_thread_gets_its_own_reply_anchor(self):
        adapter = self._adapter_with_two_threads()

        older = self._send(adapter, "1000")
        self.assertEqual(older["Subject"], "Re: Invoice")
        self.assertEqual(older["In-Reply-To"], "<one@test.com>")
        self.assertEqual(older["References"], "<one@test.com>")

        newer = self._send(adapter, "2000")
        self.assertEqual(newer["Subject"], "Re: Lunch")
        self.assertEqual(newer["In-Reply-To"], "<two@test.com>")

    def test_reply_to_older_thread_is_not_captured_by_the_newer_one(self):
        """The exact regression: the most-recently-touched thread used to win."""
        adapter = self._adapter_with_two_threads()
        sent = self._send(adapter, "1000")
        self.assertNotEqual(sent["In-Reply-To"], "<two@test.com>")

    def test_unknown_thread_does_not_borrow_another_threads_anchor(self):
        adapter = self._adapter_with_two_threads()
        sent = self._send(adapter, "9999")
        self.assertIsNone(sent["In-Reply-To"])
        self.assertEqual(sent["Subject"], "Re: Hermes Agent")

    def test_send_reads_thread_id_from_delivery_metadata(self):
        """The gateway delivery router passes thread_id in send metadata."""
        import asyncio

        adapter = self._adapter_with_two_threads()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            result = asyncio.run(
                adapter.send(
                    "user@test.com",
                    "Cron output",
                    metadata={"thread_id": "1000"},
                )
            )

        self.assertTrue(result.success)
        sent = mock_server.send_message.call_args[0][0]
        self.assertEqual(sent["In-Reply-To"], "<one@test.com>")
        self.assertEqual(sent["Subject"], "Re: Invoice")

    def test_send_without_thread_id_falls_back_to_most_recent_thread(self):
        """A send that names no thread (standalone/home-channel) still threads."""
        import asyncio

        adapter = self._adapter_with_two_threads()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            asyncio.run(adapter.send("user@test.com", "Notice"))

        sent = mock_server.send_message.call_args[0][0]
        self.assertEqual(sent["In-Reply-To"], "<two@test.com>")

    def test_dispatch_does_not_overwrite_a_sibling_thread(self):
        """A new message in thread A must leave thread B's anchor intact."""
        import asyncio

        adapter = self._adapter_with_two_threads()

        async def noop(event):
            pass

        adapter.handle_message = noop

        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(adapter._dispatch_message({
                "uid": b"9",
                "thread_id": "1000",
                "sender_addr": "user@test.com",
                "sender_name": "User",
                "subject": "Re: Invoice",
                "message_id": "<one-followup@test.com>",
                "in_reply_to": "<one@test.com>",
                "body": "ping",
                "attachments": [],
                "date": "",
                "sender_authenticated": True,
            }))

        self.assertEqual(
            adapter._thread_context[("user@test.com", "1000")]["message_id"],
            "<one-followup@test.com>",
        )
        self.assertEqual(
            adapter._thread_context[("user@test.com", "2000")]["message_id"],
            "<two@test.com>",
        )


class TestCronOriginCapture(unittest.TestCase):
    """An email-originated cron job records its thread and delivers back to it."""

    def test_origin_captures_thread_id_for_an_email_session(self):
        from tools.cronjob_tools import _origin_from_env

        with patch.dict(os.environ, {
            "HERMES_SESSION_PLATFORM": "email",
            "HERMES_SESSION_CHAT_ID": "user@test.com",
            "HERMES_SESSION_THREAD_ID": "1795604436429434944",
        }):
            origin = _origin_from_env()

        self.assertIsNotNone(origin)
        self.assertEqual(origin["platform"], "email")
        self.assertIsNotNone(origin["thread_id"])
        self.assertEqual(origin["thread_id"], "1795604436429434944")

    def test_origin_delivery_nests_under_the_originating_thread(self):
        """deliver="origin" routes through the delivery router, whose metadata
        thread_id must select that thread's reply anchor."""
        import asyncio
        from gateway.delivery import DeliveryRouter, DeliveryTarget
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1000", "Invoice", "<one@test.com>"
        )
        # A later message in a different thread with the same sender — the
        # slot that used to capture this delivery.
        adapter._remember_thread_context(
            "user@test.com", "2000", "Lunch", "<two@test.com>"
        )

        config = GatewayConfig()
        config.platforms[Platform.EMAIL] = PlatformConfig(enabled=True)
        router = DeliveryRouter(config, {Platform.EMAIL: adapter})

        target = DeliveryTarget(
            platform=Platform.EMAIL,
            chat_id="user@test.com",
            thread_id="1000",
            is_explicit=True,
        )

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            asyncio.run(
                router._deliver_to_platform(target, "Job finished.", {})
            )

        sent = mock_server.send_message.call_args[0][0]
        self.assertEqual(sent["In-Reply-To"], "<one@test.com>")
        self.assertEqual(sent["References"], "<one@test.com>")
        self.assertEqual(sent["Subject"], "Re: Invoice")


class TestGmailCapabilityProbe(unittest.TestCase):
    """X-GM-EXT-1 gates the thread-id path so non-Gmail servers keep working."""

    def _run(self, capabilities, thrid_ok=True):
        adapter = _make_adapter()
        raw = _raw_email(sender="user@test.com")

        mock_imap = MagicMock()

        mock_imap.capabilities = ("IMAP4REV1", "X-GM-EXT-1")
        mock_imap.capabilities = capabilities

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                if _is_thrid_fetch(args):
                    if not thrid_ok:
                        import imaplib
                        raise imaplib.IMAP4.error("BAD Invalid system flag")
                    return ("OK", [b"1 (X-GM-THRID 4242 UID 1)"])
                if "HEADER.FIELDS" in str(args[-1]):
                    return ("OK", [(b"1 (BODY[HEADER.FIELDS (FROM)]",
                                    b"From: user@test.com\r\n\r\n")])
                return ("OK", [(b"1", raw)])
            return ("OK", [b"1"])

        mock_imap.uid.side_effect = uid_handler
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
                patch("smtplib.SMTP") as mock_smtp:
            results = adapter._fetch_new_messages(lambda candidate: True)
        return adapter, results, mock_smtp

    def test_gmail_server_gets_thread_ids(self):
        adapter, results, _ = self._run(("IMAP4REV1", "X-GM-EXT-1"))
        self.assertTrue(adapter._has_gmail_ext)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["thread_id"], "4242")

    def test_non_gmail_server_still_delivers_mail(self):
        """The regression this probe exists to prevent: without it, a server
        that cannot answer X-GM-THRID loses every inbound message to the
        retry-and-drop path."""
        adapter, results, mock_smtp = self._run(("IMAP4REV1",), thrid_ok=False)

        self.assertFalse(adapter._has_gmail_ext)
        self.assertEqual(len(results), 1, "message must still be delivered")
        self.assertIsNone(results[0]["thread_id"])
        # No strike accumulated and no apology email sent.
        self.assertEqual(adapter._thrid_failures, {})
        mock_smtp.assert_not_called()

    def test_non_gmail_falls_back_to_sender_keyed_session(self):
        """thread_id=None reproduces the previous one-session-per-sender key."""
        from gateway.session import build_session_key
        from gateway.config import Platform
        from gateway.platforms.base import SessionSource

        key = build_session_key(SessionSource(
            platform=Platform.EMAIL, chat_id="user@test.com",
            chat_type="dm", user_id="user@test.com", thread_id=None,
        ))
        self.assertEqual(key, "agent:main:email:dm:user@test.com")

    def test_probe_failure_fails_safe(self):
        """An unusable capability response must not enable the THRID path."""
        adapter = _make_adapter()
        mock_imap = MagicMock()
        mock_imap.capabilities = ()
        mock_imap.capability.side_effect = Exception("connection reset")
        self.assertFalse(adapter._probe_gmail_ext(mock_imap))

    def test_capability_queried_when_attribute_absent(self):
        adapter = _make_adapter()
        mock_imap = MagicMock()
        mock_imap.capabilities = None
        mock_imap.capability.return_value = ("OK", [b"IMAP4REV1 X-GM-EXT-1 UIDPLUS"])
        self.assertTrue(adapter._probe_gmail_ext(mock_imap))


class TestRFC5322Compliance(unittest.TestCase):
    """Outgoing headers follow RFC 5322 §3.6.4 (Identification Fields)."""

    def _send(self, adapter, thread_id, reply_to=None):
        with patch("smtplib.SMTP") as mock_smtp:
            server = MagicMock()
            mock_smtp.return_value = server
            adapter._send_email("user@test.com", "Body", reply_to,
                                thread_id=thread_id)
            return server.send_message.call_args[0][0]

    def test_references_carries_the_full_ancestor_chain(self):
        """§3.6.4: References = parent's References + parent's Message-ID.

        Emitting only the parent's Message-ID loses the ancestry a client
        needs to nest a reply in a long thread.
        """
        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1", "Deep thread", "<p3@test.com>",
            references="<root@test.com> <p1@test.com> <p2@test.com>",
        )
        msg = self._send(adapter, "1")
        self.assertEqual(msg["In-Reply-To"], "<p3@test.com>")
        self.assertEqual(
            msg["References"],
            "<root@test.com> <p1@test.com> <p2@test.com> <p3@test.com>",
        )

    def test_references_without_prior_chain_is_just_the_parent(self):
        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1", "New thread", "<only@test.com>"
        )
        msg = self._send(adapter, "1")
        self.assertEqual(msg["References"], "<only@test.com>")

    def test_references_deduplicates(self):
        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1", "S", "<p1@test.com>",
            references="<root@test.com> <p1@test.com>",
        )
        msg = self._send(adapter, "1")
        self.assertEqual(msg["References"], "<root@test.com> <p1@test.com>")

    def test_references_chain_is_capped_keeping_root_and_recent(self):
        from plugins.platforms.email.adapter import MAX_REFERENCES
        chain = " ".join(f"<a{i}@test.com>" for i in range(50))
        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1", "S", "<parent@test.com>", references=chain
        )
        msg = self._send(adapter, "1")
        refs = msg["References"].split()
        self.assertEqual(len(refs), MAX_REFERENCES)
        self.assertEqual(refs[0], "<a0@test.com>", "thread root must survive")
        self.assertEqual(refs[-1], "<parent@test.com>")

    def test_header_injecting_message_id_is_sanitized(self):
        """A CRLF-bearing inbound Message-ID must not reach the header setter.

        Python raises HeaderParseError on such a value, which would make every
        reply in that thread fail — so the value is trimmed to its msg-id.
        """
        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1", "S",
            "<ok@test.com>\r\nBcc: attacker@evil.com\r\nX-Injected: yes",
        )
        msg = self._send(adapter, "1")
        raw = msg.as_bytes()
        self.assertEqual(msg["In-Reply-To"], "<ok@test.com>")
        self.assertNotIn(b"Bcc:", raw)
        self.assertNotIn(b"X-Injected", raw)

    def test_malformed_message_id_is_dropped_not_echoed(self):
        """§3.6.4 requires msg-id = '<' addr-spec '>'; a bare token is not one."""
        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1", "S", "not-a-valid-msgid"
        )
        msg = self._send(adapter, "1")
        self.assertIsNone(msg["In-Reply-To"])
        self.assertIsNone(msg["References"])

    def test_generated_message_id_is_well_formed(self):
        adapter = _make_adapter()
        msg = self._send(adapter, None)
        mid = msg["Message-ID"]
        self.assertTrue(mid.startswith("<") and mid.endswith(">"))
        self.assertIn("@", mid)
        self.assertNotIn(" ", mid)

    def test_generated_message_ids_are_unique(self):
        adapter = _make_adapter()
        ids = {self._send(adapter, None)["Message-ID"] for _ in range(25)}
        self.assertEqual(len(ids), 25)

    def test_non_ascii_subject_is_rfc2047_encoded(self):
        """§2.2: header field bodies are US-ASCII; non-ASCII needs RFC 2047."""
        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1", "Uber Cafe \u4f60\u597d", "<p@test.com>"
        )
        raw = self._send(adapter, "1").as_bytes()
        subject_lines = [l for l in raw.split(b"\n")
                         if l.lower().startswith(b"subject")]
        self.assertTrue(subject_lines)
        for line in subject_lines:
            self.assertTrue(all(b < 128 for b in line),
                            "Subject must not contain raw 8-bit bytes")

    def test_body_lines_within_rfc5322_limit(self):
        """§2.1.1: no line may exceed 998 characters."""
        adapter = _make_adapter()
        adapter._remember_thread_context(
            "user@test.com", "1", "S", "<p@test.com>"
        )
        with patch("smtplib.SMTP") as mock_smtp:
            server = MagicMock()
            mock_smtp.return_value = server
            adapter._send_email("user@test.com", "x" * 50000, None, thread_id="1")
            msg = server.send_message.call_args[0][0]
        for line in msg.as_bytes().split(b"\n"):
            self.assertLessEqual(len(line.rstrip(b"\r")), 998)

    def test_inbound_references_captured_at_ingestion(self):
        adapter = _make_adapter()
        from email.mime.text import MIMEText as _MT
        m = _MT("hi")
        m["From"] = "user@test.com"
        m["Subject"] = "S"
        m["Message-ID"] = "<child@test.com>"
        m["References"] = "<root@test.com> <mid@test.com>"

        mock_imap = MagicMock()

        mock_imap.capabilities = ("IMAP4REV1", "X-GM-EXT-1")
        mock_imap.capabilities = ("X-GM-EXT-1",)

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                if _is_thrid_fetch(args):
                    return ("OK", [b"1 (X-GM-THRID 9 UID 1)"])
                return ("OK", [(b"1", m.as_bytes())])
            return ("OK", [b"1"])

        mock_imap.uid.side_effect = uid_handler
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(lambda candidate: True)

        self.assertEqual(results[0]["references"],
                         "<root@test.com> <mid@test.com>")


class TestNotificationVolume(unittest.TestCase):
    """Guard for #27804's second half: email must not emit progress chatter.

    The isolation half is fixed by per-thread sessions; the 100-200 status
    emails per request are prevented by the email platform's registered
    ``display_tier``. main reads that tier from the plugin registry, so pin the
    declaration here and the five surfaces it disables.
    """

    OPTIONAL = {
        "tool_progress": "off",
        "interim_assistant_messages": False,
        "long_running_notifications": False,
        "streaming": False,
        "busy_ack_detail": False,
    }

    def test_email_registers_the_minimal_display_tier(self):
        from plugins.platforms.email import adapter as email_adapter

        captured = {}

        class _Ctx:
            def register_platform(self, **kwargs):
                captured.update(kwargs)

        email_adapter.register(_Ctx())
        self.assertEqual(captured.get("display_tier"), "minimal")

    def test_minimal_tier_switches_every_progress_surface_off(self):
        from gateway.display_config import _TIERS

        tier = _TIERS["minimal"]
        for setting, want in self.OPTIONAL.items():
            with self.subTest(setting=setting):
                self.assertEqual(
                    tier.get(setting), want,
                    f"email must default {setting}={want!r} — every progress "
                    "event costs a separate email (#27804)",
                )

    def test_operator_can_still_opt_in(self):
        """The defaults are defaults, not a hard block."""
        from gateway.display_config import resolve_display_setting

        cfg = {"display": {"platforms": {"email": {"tool_progress": "all"}}}}
        self.assertEqual(
            resolve_display_setting(cfg, "email", "tool_progress", None), "all"
        )

    def test_typing_indicator_is_a_noop(self):
        import asyncio
        adapter = _make_adapter()
        asyncio.run(adapter.send_typing("user@test.com"))


if __name__ == "__main__":
    unittest.main()
