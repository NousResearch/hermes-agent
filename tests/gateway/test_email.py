"""Tests for the Email gateway platform adapter.

Covers:
1. Platform enum exists with correct value
2. Config loading from env vars via _apply_env_overrides
3. Adapter init and config parsing
4. Helper functions (header decoding, body extraction, address extraction, HTML stripping)
5. Authorization integration (platform in allowlist maps)
6. Send message tool routing (platform in platform_map)
7. check_email_requirements function
8. Attachment extraction and caching
9. Message dispatch and threading
"""

import os
import unittest
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from unittest.mock import patch, MagicMock, AsyncMock, ANY

from gateway.platforms.base import SendResult


class TestConfigEnvOverrides(unittest.TestCase):
    """Verify email config is loaded from environment variables."""


    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_HOME_ADDRESS": "user@test.com",
    }, clear=False)
    def test_email_home_channel_loaded(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)
        home = config.platforms[Platform.EMAIL].home_channel
        self.assertIsNotNone(home)
        self.assertEqual(home.chat_id, "user@test.com")


class TestCheckRequirements(unittest.TestCase):
    """Verify check_email_requirements function."""

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "a@b.com",
        "EMAIL_PASSWORD": "pw",
        "EMAIL_IMAP_HOST": "imap.b.com",
        "EMAIL_SMTP_HOST": "smtp.b.com",
    }, clear=False)
    def test_requirements_met(self):
        from plugins.platforms.email.adapter import check_email_requirements
        self.assertTrue(check_email_requirements())


class TestHelperFunctions(unittest.TestCase):
    """Test email parsing helper functions."""


    def test_decode_header_encoded(self):
        from plugins.platforms.email.adapter import _decode_header_value
        # RFC 2047 encoded subject
        encoded = "=?utf-8?B?TWVyaGFiYQ==?="  # "Merhaba" in base64
        result = _decode_header_value(encoded)
        self.assertEqual(result, "Merhaba")

    def test_extract_email_address_with_name(self):
        from plugins.platforms.email.adapter import _extract_email_address
        self.assertEqual(
            _extract_email_address("John Doe <john@example.com>"),
            "john@example.com"
        )


    def test_strip_html_basic(self):
        from plugins.platforms.email.adapter import _strip_html
        html = "<p>Hello <b>world</b></p>"
        result = _strip_html(html)
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertNotIn("<p>", result)
        self.assertNotIn("<b>", result)


class TestExtractTextBody(unittest.TestCase):
    """Test email body extraction from different message formats."""

    def test_plain_text_body(self):
        from plugins.platforms.email.adapter import _extract_text_body
        msg = MIMEText("Hello, this is a test.", "plain", "utf-8")
        result = _extract_text_body(msg)
        self.assertEqual(result, "Hello, this is a test.")


    def test_multipart_prefers_plain(self):
        from plugins.platforms.email.adapter import _extract_text_body
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("<p>HTML version</p>", "html", "utf-8"))
        msg.attach(MIMEText("Plain version", "plain", "utf-8"))
        result = _extract_text_body(msg)
        self.assertEqual(result, "Plain version")


class TestExtractAttachments(unittest.TestCase):
    """Test attachment extraction and caching."""

    def test_no_attachments(self):
        from plugins.platforms.email.adapter import _extract_attachments
        msg = MIMEText("No attachments here.", "plain", "utf-8")
        result = _extract_attachments(msg)
        self.assertEqual(result, [])


class TestDispatchMessage(unittest.TestCase):
    """Test email message dispatch logic."""

    def setUp(self):
        # These tests exercise dispatch mechanics (subject formatting,
        # attachment typing, source building), not the authorization gate.
        # The adapter now fails closed at dispatch when no allowlist / allow-all
        # is configured (SECURITY.md 2.6), so opt into allow-all here to keep
        # exercising the dispatch path. Auth-contract tests below override this.
        self._prev_allow_all = os.environ.get("EMAIL_ALLOW_ALL_USERS")
        os.environ["EMAIL_ALLOW_ALL_USERS"] = "true"

    def tearDown(self):
        if self._prev_allow_all is None:
            os.environ.pop("EMAIL_ALLOW_ALL_USERS", None)
        else:
            os.environ["EMAIL_ALLOW_ALL_USERS"] = self._prev_allow_all

    def _make_adapter(self):
        """Create an EmailAdapter with mocked env vars."""
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_IMAP_PORT": "993",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_SMTP_PORT": "587",
            "EMAIL_POLL_INTERVAL": "15",
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter

    def test_self_message_filtered(self):
        """Messages from the agent's own address should be skipped."""
        import asyncio
        adapter = self._make_adapter()
        adapter._message_handler = MagicMock()

        msg_data = {
            "uid": b"1",
            "sender_addr": "hermes@test.com",
            "sender_name": "Hermes",
            "subject": "Test",
            "message_id": "<msg1@test.com>",
            "in_reply_to": "",
            "body": "Self message",
            "attachments": [],
            "date": "",
        }

        asyncio.run(adapter._dispatch_message(msg_data))
        adapter._message_handler.assert_not_called()

    def test_subject_included_in_text(self):
        """Subject should be prepended to body for non-reply emails."""
        import asyncio
        adapter = self._make_adapter()
        captured_events = []

        async def mock_handler(event):
            captured_events.append(event)
            return None

        adapter._message_handler = mock_handler
        # Override handle_message to capture the event directly
        original_handle = adapter.handle_message

        async def capture_handle(event):
            captured_events.append(event)

        adapter.handle_message = capture_handle

        msg_data = {
            "uid": b"2",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Help with Python",
            "message_id": "<msg2@test.com>",
            "in_reply_to": "",
            "body": "How do I use lists?",
            "attachments": [],
            "date": "",
        }

        asyncio.run(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertIn("[Subject: Help with Python]", captured_events[0].text)
        self.assertIn("How do I use lists?", captured_events[0].text)

    def test_reply_subject_not_duplicated(self):
        """Re: subjects should not be prepended to body."""
        import asyncio
        adapter = self._make_adapter()
        captured_events = []

        async def capture_handle(event):
            captured_events.append(event)

        adapter.handle_message = capture_handle

        msg_data = {
            "uid": b"3",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Re: Help with Python",
            "message_id": "<msg3@test.com>",
            "in_reply_to": "<msg2@test.com>",
            "body": "Thanks for the help!",
            "attachments": [],
            "date": "",
        }

        asyncio.run(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertNotIn("[Subject:", captured_events[0].text)
        self.assertEqual(captured_events[0].text, "Thanks for the help!")


    def test_image_attachment_sets_photo_type(self):
        """Email with image attachment should set message type to PHOTO."""
        import asyncio
        from gateway.platforms.base import MessageType
        adapter = self._make_adapter()
        captured_events = []

        async def capture_handle(event):
            captured_events.append(event)

        adapter.handle_message = capture_handle

        msg_data = {
            "uid": b"5",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Re: photo",
            "message_id": "<msg5@test.com>",
            "in_reply_to": "",
            "body": "Check this photo",
            "attachments": [{"path": "/tmp/img.jpg", "filename": "img.jpg", "type": "image", "media_type": "image/jpeg"}],
            "date": "",
        }

        asyncio.run(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertEqual(captured_events[0].message_type, MessageType.PHOTO)
        self.assertEqual(captured_events[0].media_urls, ["/tmp/img.jpg"])


    def test_empty_allowlist_denies_without_optin(self):
        """No allowlist and no allow-all opt-in → adapter fails closed (2.6)."""
        import asyncio
        with patch.dict(os.environ, {}, clear=False):
            # No allowlist, and explicitly no allow-all opt-in.
            for k in ("EMAIL_ALLOWED_USERS", "EMAIL_ALLOW_ALL_USERS",
                      "GATEWAY_ALLOW_ALL_USERS"):
                os.environ.pop(k, None)

            adapter = self._make_adapter()
            adapter._message_handler = MagicMock()

            msg_data = {
                "uid": b"101",
                "sender_addr": "anyone@test.com",
                "sender_name": "Anyone",
                "subject": "Hey",
                "message_id": "<any@test.com>",
                "in_reply_to": "",
                "body": "Hi",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))
            # Fail closed: an unset allowlist without allow-all drops the sender.
            adapter._message_handler.assert_not_called()


    def test_unauthenticated_allowed_with_allow_all(self):
        """EMAIL_ALLOW_ALL_USERS=true makes sender identity moot — gate skipped.

        With allow-all and no restrictive allowlist, an unauthenticated sender
        is forwarded: the operator has explicitly chosen to accept anyone.
        """
        import asyncio
        with patch.dict(os.environ, {
            "EMAIL_ALLOW_ALL_USERS": "true",
        }):
            os.environ.pop("EMAIL_ALLOWED_USERS", None)
            os.environ.pop("GATEWAY_ALLOWED_USERS", None)
            adapter = self._make_adapter()
            captured = []

            async def capture_handle(event):
                captured.append(event)

            adapter.handle_message = capture_handle

            msg_data = {
                "uid": b"203",
                "sender_addr": "stranger@elsewhere.com",
                "sender_name": "Stranger",
                "subject": "Hi",
                "message_id": "<s@elsewhere.com>",
                "in_reply_to": "",
                "body": "Hello",
                "attachments": [],
                "date": "",
                "sender_authenticated": False,
                "auth_reason": "no Authentication-Results header",
            }

            asyncio.run(adapter._dispatch_message(msg_data))
            self.assertEqual(len(captured), 1)


class TestThreadContext(unittest.TestCase):
    """Test email reply threading logic."""

    def setUp(self):
        # Thread-context storage is a dispatch-mechanics test, not an auth test.
        # The adapter fails closed at dispatch without allow-all (SECURITY.md 2.6),
        # so opt into allow-all to keep exercising the threading path.
        self._prev_allow_all = os.environ.get("EMAIL_ALLOW_ALL_USERS")
        os.environ["EMAIL_ALLOW_ALL_USERS"] = "true"

    def tearDown(self):
        if self._prev_allow_all is None:
            os.environ.pop("EMAIL_ALLOW_ALL_USERS", None)
        else:
            os.environ["EMAIL_ALLOW_ALL_USERS"] = self._prev_allow_all

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter


    def test_reply_uses_re_prefix(self):
        """Reply subject should have Re: prefix."""
        adapter = self._make_adapter()
        adapter._thread_context["user@test.com"] = {
            "subject": "Project question",
            "message_id": "<original@test.com>",
        }

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            adapter._send_email("user@test.com", "Here is the answer.", None)

            # Check the sent message
            send_call = mock_server.send_message.call_args[0][0]
            self.assertEqual(send_call["Subject"], "Re: Project question")
            self.assertEqual(send_call["In-Reply-To"], "<original@test.com>")
            self.assertEqual(send_call["References"], "<original@test.com>")
            self.assertIn("Date", send_call)


class TestSendMethods(unittest.TestCase):
    """Test email send methods."""

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter


    def test_send_document_with_attachment(self):
        """send_document should send email with file attachment."""
        import asyncio
        import tempfile
        adapter = self._make_adapter()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test document content")
            tmp_path = f.name

        try:
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value = mock_server

                result = asyncio.run(
                    adapter.send_document("user@test.com", tmp_path, "Here is the file")
                )

                self.assertTrue(result.success)
                mock_server.send_message.assert_called_once()
                sent_msg = mock_server.send_message.call_args[0][0]
                # Should be multipart with attachment
                parts = list(sent_msg.walk())
                has_attachment = any(
                    "attachment" in str(p.get("Content-Disposition", ""))
                    for p in parts
                )
                self.assertTrue(has_attachment)
        finally:
            os.unlink(tmp_path)


    def test_get_chat_info(self):
        """get_chat_info should return email address as chat info."""
        import asyncio
        adapter = self._make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Test", "message_id": "<m@t>"}

        info = asyncio.run(
            adapter.get_chat_info("user@test.com")
        )

        self.assertEqual(info["name"], "user@test.com")
        self.assertEqual(info["type"], "dm")
        self.assertEqual(info["subject"], "Test")


class TestConnectDisconnect(unittest.TestCase):
    """Test IMAP/SMTP connection lifecycle."""

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter

    def test_connect_success(self):
        """Successful IMAP + SMTP connection returns True."""
        import asyncio
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b"1 2 3"])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            result = asyncio.run(adapter.connect())

            self.assertTrue(result)
            self.assertTrue(adapter._running)
            # Should have skipped existing messages
            self.assertEqual(len(adapter._seen_uids), 3)
            # Cleanup
            adapter._running = False
            if adapter._poll_task:
                adapter._poll_task.cancel()


class TestFetchNewMessages(unittest.TestCase):
    """Test IMAP message fetching logic."""

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter

    def test_fetch_skips_seen_uids(self):
        """Already-seen UIDs should not be fetched again."""
        adapter = self._make_adapter()
        adapter._seen_uids = {b"1", b"2"}

        raw_email = MIMEText("Hello", "plain", "utf-8")
        raw_email["From"] = "user@test.com"
        raw_email["Subject"] = "Test"
        raw_email["Message-ID"] = "<msg@test.com>"

        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1 2 3"])
            if command == "fetch":
                return ("OK", [(b"3", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        # Only UID 3 should be fetched (1 and 2 already seen)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sender_addr"], "user@test.com")
        self.assertIn(b"3", adapter._seen_uids)


class TestPollLoop(unittest.TestCase):
    """Test the async polling loop."""

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_POLL_INTERVAL": "1",
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter

    def test_check_inbox_dispatches_messages(self):
        """_check_inbox should fetch and dispatch new messages."""
        import asyncio
        adapter = self._make_adapter()
        dispatched = []

        async def mock_dispatch(msg_data):
            dispatched.append(msg_data)

        adapter._dispatch_message = mock_dispatch

        raw_email = MIMEText("Test body", "plain", "utf-8")
        raw_email["From"] = "sender@test.com"
        raw_email["Subject"] = "Inbox Test"
        raw_email["Message-ID"] = "<inbox@test.com>"

        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            asyncio.run(adapter._check_inbox())

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0]["subject"], "Inbox Test")

    def test_check_inbox_notifies_fatal_error_on_fetch_failure(self):
        """A failed IMAP check must surface through the fatal-error hook so
        the gateway's reconnect/backoff machinery learns email is unhealthy
        instead of silently treating the failed check as an empty inbox
        (#80016)."""
        import asyncio
        adapter = self._make_adapter()
        notified = []

        async def mock_fatal_handler(adapter):
            notified.append(adapter)

        adapter.set_fatal_error_handler(mock_fatal_handler)

        mock_imap = MagicMock()
        mock_imap.login.side_effect = Exception("read operation timed out")

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            asyncio.run(adapter._check_inbox())

        self.assertEqual(len(notified), 1)
        self.assertEqual(adapter.fatal_error_code, "email_imap_fetch_failed")
        self.assertTrue(adapter.fatal_error_retryable)
        self.assertIn("read operation timed out", adapter.fatal_error_message)

    def test_partial_batch_dispatched_before_escalation(self):
        """A mid-batch IMAP failure must dispatch the messages already
        fetched BEFORE escalating — dropping them would lose mail, since
        their UIDs are marked seen (#80032 review)."""
        import asyncio
        adapter = self._make_adapter()
        dispatched, notified = [], []

        async def mock_dispatch(msg_data):
            dispatched.append(msg_data)

        async def mock_fatal_handler(a):
            notified.append(a)

        adapter._dispatch_message = mock_dispatch
        adapter.set_fatal_error_handler(mock_fatal_handler)

        raw_email = MIMEText("Body", "plain", "utf-8")
        raw_email["From"] = "sender@test.com"
        raw_email["Subject"] = "First of batch"
        raw_email["Message-ID"] = "<batch1@test.com>"

        mock_imap = MagicMock()
        fetches = []

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1 2"])
            if command == "fetch":
                fetches.append(args)
                if len(fetches) == 1:
                    return ("OK", [(b"1", raw_email.as_bytes())])
                raise OSError("connection dropped mid-batch")
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            asyncio.run(adapter._check_inbox())

        # The successfully fetched message was dispatched, not dropped.
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0]["subject"], "First of batch")
        # The failure still escalated through the fatal-error hook.
        self.assertEqual(len(notified), 1)
        self.assertEqual(adapter.fatal_error_code, "email_imap_fetch_failed")

    def test_mid_batch_failure_leaves_unfetched_uids_eligible(self):
        """UIDs are marked seen only after their fetch returns — a
        connection failure mid-batch must leave the remaining UIDs eligible
        for the next poll instead of permanently skipping them."""
        adapter = self._make_adapter()

        raw_email = MIMEText("Body", "plain", "utf-8")
        raw_email["From"] = "sender@test.com"
        raw_email["Subject"] = "ok"
        raw_email["Message-ID"] = "<ok@test.com>"

        mock_imap = MagicMock()
        fetches = []

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1 2 3"])
            if command == "fetch":
                fetches.append(args)
                if len(fetches) == 1:
                    return ("OK", [(b"1", raw_email.as_bytes())])
                raise OSError("connection dropped")
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertIn(b"1", adapter._seen_uids)     # fetched → seen
        self.assertNotIn(b"2", adapter._seen_uids)  # fetch raised → retry next poll
        self.assertNotIn(b"3", adapter._seen_uids)  # never reached → retry next poll
        self.assertTrue(adapter._last_fetch_failed)

    def test_poison_message_skipped_once_without_escalation(self):
        """A message whose processing raises is marked seen and skipped —
        it must not abort the batch, escalate to a reconnect, or be
        retried forever (#80032 review)."""
        adapter = self._make_adapter()

        good_email = MIMEText("Body", "plain", "utf-8")
        good_email["From"] = "sender@test.com"
        good_email["Subject"] = "good"
        good_email["Message-ID"] = "<good@test.com>"

        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1 2"])
            if command == "fetch":
                uid = args[0]
                if uid == b"1":
                    return ("OK", [(b"1", b"poison")])
                return ("OK", [(b"2", good_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), patch(
            "plugins.platforms.email.adapter.EmailAdapter._parse_fetched_message",
            side_effect=[ValueError("unparseable"), {"subject": "good"}],
        ):
            results = adapter._fetch_new_messages()

        # Poison message consumed (seen, skipped); good message survived.
        self.assertEqual(len(results), 1)
        self.assertIn(b"1", adapter._seen_uids)
        self.assertIn(b"2", adapter._seen_uids)
        self.assertFalse(adapter._last_fetch_failed)


class TestReconnectSeenUidsRestore(unittest.TestCase):
    """connect(is_reconnect=True) must not re-mark the whole mailbox seen."""

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter

    def setUp(self):
        from plugins.platforms.email.adapter import EmailAdapter
        EmailAdapter._seen_uids_snapshot.clear()

    tearDown = setUp

    def _run_connect(self, adapter, mailbox_uids, *, is_reconnect):
        import asyncio

        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [mailbox_uids])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        smtp = MagicMock()

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), patch.object(
            adapter, "_connect_smtp", return_value=smtp
        ):
            return asyncio.run(adapter.connect(is_reconnect=is_reconnect))

    def test_reconnect_restores_snapshot_instead_of_marking_all_seen(self):
        # First adapter connects with UIDs 1-2 in the mailbox.
        first = self._make_adapter()
        self.assertTrue(self._run_connect(first, b"1 2", is_reconnect=False))
        self.assertEqual(first._seen_uids, {b"1", b"2"})
        import asyncio
        asyncio.run(first.disconnect())

        # Outage: UID 3 arrives. The reconnect watcher builds a FRESH adapter
        # and connects with is_reconnect=True.
        second = self._make_adapter()
        self.assertTrue(self._run_connect(second, b"1 2 3", is_reconnect=True))
        # Baseline restored from the snapshot — UID 3 stays eligible.
        self.assertEqual(second._seen_uids, {b"1", b"2"})
        asyncio.run(second.disconnect())

    def test_first_connect_still_marks_all_seen(self):
        adapter = self._make_adapter()
        self.assertTrue(self._run_connect(adapter, b"7 8 9", is_reconnect=False))
        self.assertEqual(adapter._seen_uids, {b"7", b"8", b"9"})
        import asyncio
        asyncio.run(adapter.disconnect())

    def test_reconnect_without_snapshot_falls_back_to_mark_all_seen(self):
        # e.g. gateway restarted: no in-process snapshot exists.
        adapter = self._make_adapter()
        self.assertTrue(self._run_connect(adapter, b"4 5", is_reconnect=True))
        self.assertEqual(adapter._seen_uids, {b"4", b"5"})
        import asyncio
        asyncio.run(adapter.disconnect())


class TestSendEmailStandalone(unittest.TestCase):
    """Test the standalone _send_email function in send_message_tool."""

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_SMTP_PORT": "587",
    })
    def test_send_email_tool_success(self):
        """_send_email should use verified STARTTLS when sending."""
        import asyncio
        import ssl
        from plugins.platforms.email.adapter import _standalone_send as _email_send
        from types import SimpleNamespace
        async def _send_email(extra, chat_id, message):
            return await _email_send(SimpleNamespace(token=None, api_key=None, extra=extra or {}), chat_id, message)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            result = asyncio.run(
                _send_email({"address": "hermes@test.com", "smtp_host": "smtp.test.com"}, "user@test.com", "Hello")
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["platform"], "email")
            _, kwargs = mock_server.starttls.call_args
            self.assertIsInstance(kwargs["context"], ssl.SSLContext)
            send_call = mock_server.send_message.call_args[0][0]
            self.assertEqual(send_call["Subject"], "Hermes Agent")
            self.assertIn("Date", send_call)
            self.assertEqual(send_call["To"], "user@test.com")
            self.assertEqual(send_call["From"], "hermes@test.com")


class TestSmtpConnectionCleanup(unittest.TestCase):
    """Verify SMTP connections are closed even when send_message raises."""

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_SMTP_PORT": "587",
    }, clear=False)
    def _make_adapter(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter
        return EmailAdapter(PlatformConfig(enabled=True))


    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_SMTP_PORT": "587",
    }, clear=False)
    def test_smtp_close_called_when_quit_also_fails(self):
        """If both send_message() and quit() fail, close() is the fallback."""
        adapter = self._make_adapter()
        mock_smtp = MagicMock()
        mock_smtp.send_message.side_effect = Exception("send failed")
        mock_smtp.quit.side_effect = Exception("quit failed")

        with patch("smtplib.SMTP", return_value=mock_smtp):
            with self.assertRaises(Exception):
                adapter._send_email("user@test.com", "Hello")

        mock_smtp.close.assert_called_once()


class TestImapConnectionCleanup(unittest.TestCase):
    """Verify IMAP connections are closed even when fetch raises."""

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_IMAP_PORT": "993",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }, clear=False)
    def _make_adapter(self):
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter
        return EmailAdapter(PlatformConfig(enabled=True))

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_IMAP_PORT": "993",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }, clear=False)
    def test_imap_logout_called_on_uid_fetch_failure(self):
        """IMAP logout() must be called even when uid fetch raises."""
        adapter = self._make_adapter()
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                raise Exception("fetch failed")
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(results, [])
        mock_imap.logout.assert_called_once()


class TestImapIdExtensionForNetEase(unittest.TestCase):
    """Regression for #22271: 163/NetEase mailbox requires the RFC 2971
    IMAP ID command after LOGIN, otherwise it returns ``BYE Unsafe Login``
    on every UID SEARCH.  We send ID best-effort after every login so that
    163 works while non-supporting servers stay unaffected.
    """

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@163.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.163.com",
            "EMAIL_SMTP_HOST": "smtp.163.com",
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter

    def test_connect_sends_imap_id_after_login(self):
        """connect() must call xatom('ID', ...) after LOGIN for 163 support."""
        import asyncio
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.connect())
            adapter._running = False
            if adapter._poll_task:
                adapter._poll_task.cancel()

        id_calls = [c for c in mock_imap.xatom.call_args_list if c.args and c.args[0] == "ID"]
        self.assertTrue(
            id_calls,
            "EmailAdapter.connect() must call imap.xatom('ID', ...) after "
            "LOGIN so 163/NetEase mailbox does not return 'Unsafe Login'.",
        )
        payload = id_calls[0].args[1]
        self.assertIn("hermes-agent", payload)

        names = [c[0] for c in mock_imap.method_calls]
        self.assertIn("login", names)
        self.assertLess(names.index("login"), names.index("xatom"))


class TestConnectSmtp(unittest.TestCase):
    """Test _connect_smtp() helper: protocol selection and IPv6 fallback."""

    def _make_adapter(self, port="587"):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_SMTP_PORT": port,
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            return EmailAdapter(PlatformConfig(enabled=True))


    def test_ipv6_timeout_falls_back_to_ipv4(self):
        """When default connection times out, retry with an IPv4-only SMTP path."""
        import socket as _socket
        import plugins.platforms.email.adapter as email_mod

        adapter = self._make_adapter("587")

        with patch("smtplib.SMTP", side_effect=_socket.timeout("timed out")), \
             patch.object(email_mod, "_IPv4SMTP") as mock_ipv4_smtp:
            mock_server = MagicMock()
            mock_ipv4_smtp.return_value = mock_server

            result = adapter._connect_smtp()

            self.assertIs(result, mock_server)
            mock_ipv4_smtp.assert_called_once_with("smtp.test.com", 587, timeout=30)
            mock_server.starttls.assert_called_once()

    def test_port_465_ipv6_fallback(self):
        """Port 465 IPv6 timeout falls back to IPv4 with SMTP_SSL."""
        import socket as _socket
        import plugins.platforms.email.adapter as email_mod

        adapter = self._make_adapter("465")

        with patch("smtplib.SMTP_SSL", side_effect=_socket.timeout("timed out")), \
             patch.object(email_mod, "_IPv4SMTP_SSL") as mock_ipv4_smtp_ssl:
            mock_server = MagicMock()
            mock_ipv4_smtp_ssl.return_value = mock_server

            result = adapter._connect_smtp()

            self.assertIs(result, mock_server)
            mock_ipv4_smtp_ssl.assert_called_once_with(
                "smtp.test.com", 465, timeout=30, context=ANY,
            )


class TestConnectionConfigResolution(unittest.TestCase):
    """Host/address resolution and pre-connect validation (#49736)."""


    def test_connect_aborts_without_attempting_imap_when_host_missing(self):
        """A missing host returns False without the cryptic DNS error, and marks
        the failure non-retryable so the gateway stops reconnecting (#40715)."""
        import asyncio
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }, clear=False):
            adapter = EmailAdapter(PlatformConfig(enabled=True))

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            result = asyncio.run(adapter.connect())

        self.assertFalse(result)
        mock_imap.assert_not_called()
        # The OOM fix (#40715): a blank host must NOT leave the platform in the
        # retryable reconnect loop — it is a permanent config error.
        self.assertTrue(adapter.has_fatal_error)
        self.assertEqual(adapter.fatal_error_code, "email_missing_configuration")
        self.assertFalse(adapter.fatal_error_retryable)
        self.assertIn("EMAIL_IMAP_HOST", adapter.fatal_error_message or "")

    def test_blank_present_env_vars_are_not_required(self):
        """Blank/whitespace EMAIL_* values must read as missing (#40715) — an
        abandoned setup with empty keys must not enable the platform."""
        from plugins.platforms.email.adapter import check_email_requirements
        for blank in ("", "   ", "\n"):
            with patch.dict(os.environ, {
                "EMAIL_ADDRESS": blank, "EMAIL_PASSWORD": blank,
                "EMAIL_IMAP_HOST": blank, "EMAIL_SMTP_HOST": blank,
            }, clear=False):
                self.assertFalse(check_email_requirements())


class TestSenderAuthentication(unittest.TestCase):
    """Verify _verify_sender_authentication parses Authentication-Results
    correctly and resists From: spoofing (GHSA-rxqh-5572-8m77)."""

    def _msg(self, from_addr, auth_results=None):
        """Build an email.message.Message with the given From: and
        zero or more Authentication-Results headers (first = topmost/trusted)."""
        msg = MIMEText("body")
        msg["From"] = from_addr
        for ar in auth_results or []:
            msg["Authentication-Results"] = ar
        return msg

    def _verify(self, from_addr, auth_results=None, authserv_id=""):
        from plugins.platforms.email.adapter import (
            _verify_sender_authentication,
            _extract_email_address,
        )
        msg = self._msg(from_addr, auth_results)
        addr = _extract_email_address(from_addr)
        return _verify_sender_authentication(msg, addr, authserv_id=authserv_id)

    def test_dmarc_pass_authenticates(self):
        ok, reason = self._verify(
            "Admin <admin@example.com>",
            ["mx.google.com; dmarc=pass header.from=example.com; spf=pass"],
        )
        self.assertTrue(ok, reason)


    def test_dkim_pass_aligned_authenticates(self):
        ok, reason = self._verify(
            "admin@example.com",
            ["mx.google.com; dkim=pass header.d=example.com"],
        )
        self.assertTrue(ok, reason)

    def test_spf_pass_misaligned_rejected(self):
        # SPF passes for the envelope domain, but it doesn't match From: domain.
        ok, reason = self._verify(
            "admin@example.com",
            ["mx.google.com; spf=pass smtp.mailfrom=bounce@evil.com"],
        )
        self.assertFalse(ok, reason)


    def test_injected_header_below_trusted_does_not_authenticate(self):
        """An attacker-injected Authentication-Results sorts BELOW the receiving
        server's. With authserv-id pinning, only the trusted (first) header is
        consulted, so a forged 'dmarc=pass' lower in the stack is ignored."""
        ok, reason = self._verify(
            "admin@example.com",
            [
                # Trusted: stamped by our server, real verdict = fail
                "mx.ourserver.com; dmarc=fail header.from=example.com",
                # Forged by attacker, claims pass
                "mx.ourserver.com; dmarc=pass header.from=example.com",
            ],
            authserv_id="mx.ourserver.com",
        )
        self.assertFalse(ok, reason)


class TestFolderLifecycle(unittest.TestCase):
    """Tests for the INBOX → Working → Done two-stage folder lifecycle."""

    # ------------------------------------------------------------------
    # Helper: build a minimal EmailAdapter with controlled folder settings
    # ------------------------------------------------------------------

    def _make_adapter(self, extra=None, env=None):
        """Build an EmailAdapter.

        ``extra`` maps to ``config.yaml`` ``platforms.email`` (working_folder /
        done_folder live here now). ``env`` adds/overrides environment variables
        (e.g. EMAIL_ALLOWED_USERS, which the dispatch path still reads from env).
        """
        from gateway.config import PlatformConfig
        base_env = {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_IMAP_PORT": "993",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "EMAIL_SMTP_PORT": "587",
            "EMAIL_POLL_INTERVAL": "15",
        }
        if env:
            base_env.update(env)
        with patch.dict(os.environ, base_env, clear=True):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True, extra=extra or {}))
        return adapter

    def _make_raw_email(self, sender="user@test.com", subject="Test", message_id="<msg@test.com>"):
        """Build a minimal RFC822 message as bytes."""
        msg = MIMEText("Hello Hermes", "plain", "utf-8")
        msg["From"] = sender
        msg["Subject"] = subject
        msg["Message-ID"] = message_id
        return msg.as_bytes()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    def test_imap_move_uses_uid_move_when_available(self):
        """_imap_move should call UID MOVE and NOT fall back to COPY when MOVE works."""
        from plugins.platforms.email.adapter import EmailAdapter

        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b"1"])

        result = EmailAdapter._imap_move(mock_imap, b"42", "Hermes_Working")

        self.assertTrue(result)
        # First uid call must be MOVE
        first_call = mock_imap.uid.call_args_list[0]
        self.assertEqual(first_call.args[0], "MOVE")
        # COPY must NOT have been called
        copy_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "COPY"]
        self.assertEqual(len(copy_calls), 0)

    @staticmethod
    def _uidplus_imap(uid_handler):
        """A mock IMAP handle that advertises UIDPLUS (needed for the fallback)."""
        mock_imap = MagicMock()
        mock_imap.capabilities = ("IMAP4REV1", "UIDPLUS")
        mock_imap.uid.side_effect = uid_handler
        return mock_imap

    def test_imap_move_falls_back_to_copy_expunge(self):
        """_imap_move should fall back to COPY+STORE+EXPUNGE when MOVE returns NO."""
        from plugins.platforms.email.adapter import EmailAdapter

        def uid_handler(command, *args):
            if command == "MOVE":
                return ("NO", [b"MOVE not supported"])
            if command == "COPY":
                return ("OK", [b"1"])
            return ("OK", [b""])

        mock_imap = self._uidplus_imap(uid_handler)

        result = EmailAdapter._imap_move(mock_imap, b"42", "Hermes_Done")

        self.assertTrue(result)
        commands = [c.args[0] for c in mock_imap.uid.call_args_list]
        self.assertIn("COPY", commands)
        self.assertIn("STORE", commands)
        self.assertIn("EXPUNGE", commands)
        # UID EXPUNGE only — a bare EXPUNGE would hit the whole folder
        mock_imap.expunge.assert_not_called()

    def test_imap_move_falls_back_when_move_raises(self):
        """_imap_move should fall back to COPY+EXPUNGE when MOVE raises an exception."""
        from plugins.platforms.email.adapter import EmailAdapter

        def uid_handler(command, *args):
            if command == "MOVE":
                raise Exception("command unknown: UID MOVE")
            if command == "COPY":
                return ("OK", [b"1"])
            return ("OK", [b""])

        mock_imap = self._uidplus_imap(uid_handler)

        result = EmailAdapter._imap_move(mock_imap, b"7", "Hermes_Done")

        self.assertTrue(result)
        commands = [c.args[0] for c in mock_imap.uid.call_args_list]
        self.assertIn("COPY", commands)

    def test_imap_move_refuses_fallback_without_uidplus(self):
        """Without MOVE and without UIDPLUS, nothing is copied, deleted or expunged.

        A global EXPUNGE would permanently remove every \\Deleted message in
        the folder, including mail another client flagged. Bail out instead.
        """
        from plugins.platforms.email.adapter import EmailAdapter

        def uid_handler(command, *args):
            if command == "MOVE":
                return ("NO", [b"MOVE not supported"])
            return ("OK", [b"1"])

        mock_imap = MagicMock()
        mock_imap.capabilities = ("IMAP4REV1",)   # no UIDPLUS
        mock_imap.uid.side_effect = uid_handler

        result = EmailAdapter._imap_move(mock_imap, b"42", "Hermes_Done")

        self.assertFalse(result)
        commands = [c.args[0] for c in mock_imap.uid.call_args_list]
        self.assertNotIn("COPY", commands)
        self.assertNotIn("STORE", commands)
        mock_imap.expunge.assert_not_called()

    def test_imap_move_survives_uid_expunge_failure(self):
        """A refused UID EXPUNGE still counts as moved — and never escalates to EXPUNGE."""
        from plugins.platforms.email.adapter import EmailAdapter

        def uid_handler(command, *args):
            if command == "MOVE":
                return ("NO", [b"MOVE not supported"])
            if command == "COPY":
                return ("OK", [b"1"])
            if command == "EXPUNGE":
                raise Exception("UID EXPUNGE refused")
            return ("OK", [b""])

        mock_imap = self._uidplus_imap(uid_handler)

        result = EmailAdapter._imap_move(mock_imap, b"42", "Hermes_Done")

        # The copy reached the destination; only the source cleanup failed.
        self.assertTrue(result)
        mock_imap.expunge.assert_not_called()

    def test_search_message_id_quotes_the_literal(self):
        """The attacker-controlled Message-ID is passed as ONE quoted IMAP literal."""
        from plugins.platforms.email.adapter import EmailAdapter

        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b"5"])

        uids = EmailAdapter._search_message_id(mock_imap, '<a" OR ALL@evil>')

        self.assertEqual(uids, [b"5"])
        # Quoted, with the embedded quote escaped — not injected as extra keys
        self.assertEqual(
            mock_imap.uid.call_args.args,
            ("SEARCH", None, "HEADER", "Message-ID", '"<a\\" OR ALL@evil>"'),
        )

    def test_search_message_id_refuses_control_characters(self):
        """A Message-ID with CR/LF or non-ASCII is refused, not escaped."""
        from plugins.platforms.email.adapter import EmailAdapter

        for bad in ("<a\r\nA001 SELECT INBOX@x>", "<a\x00@x>", "<ä@x>"):
            with self.subTest(message_id=bad):
                mock_imap = MagicMock()
                self.assertEqual(EmailAdapter._search_message_id(mock_imap, bad), [])
                mock_imap.uid.assert_not_called()

    def test_ensure_folder_swallows_errors(self):
        """_ensure_folder must not propagate exceptions from imap.create()."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.create.side_effect = Exception("unexpected server error")

        # Should not raise
        adapter._ensure_folder(mock_imap, "Hermes_Working")
        mock_imap.create.assert_called_once_with("Hermes_Working")

    def test_ensure_folder_warns_once_per_folder(self):
        """A CREATE failure warns the operator once, then drops to debug."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.create.side_effect = Exception("permission denied")

        with self.assertLogs("plugins.platforms.email.adapter", level="WARNING") as logs:
            adapter._ensure_folder(mock_imap, "Hermes_Working")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("Hermes_Working", logs.output[0])

        # Second attempt (e.g. next reconnect) must not repeat the warning
        with self.assertLogs("plugins.platforms.email.adapter", level="DEBUG") as logs:
            adapter._ensure_folder(mock_imap, "Hermes_Working")
        self.assertFalse([line for line in logs.output if line.startswith("WARNING")])

    def test_ensure_folder_no_warning_when_folder_exists(self):
        """A NO response (already exists) is the normal path and stays quiet."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.create.return_value = ("NO", [b"Mailbox already exists"])

        with self.assertLogs("plugins.platforms.email.adapter", level="DEBUG") as logs:
            adapter._ensure_folder(mock_imap, "Hermes_Done")
        self.assertFalse([line for line in logs.output if line.startswith("WARNING")])

    def test_ensure_folder_skips_empty_name(self):
        """_ensure_folder must do nothing when name is empty."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        adapter._ensure_folder(mock_imap, "")
        mock_imap.create.assert_not_called()

    # ------------------------------------------------------------------
    # Finalize connection reuse
    # ------------------------------------------------------------------

    def test_finalize_reuses_one_connection(self):
        """Finalizing a batch must not reconnect per message."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        mock_imap = MagicMock()
        mock_imap.capabilities = ("IMAP4REV1", "UIDPLUS")
        mock_imap.noop.return_value = ("OK", [b"NOOP completed"])

        def uid_handler(command, *args):
            if command == "SEARCH":
                return ("OK", [b"3"])
            return ("OK", [b"1"])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap) as ctor:
            for i in range(3):
                adapter._finalize_message(f"<m{i}@test.com>", "Hermes_Working")

        self.assertEqual(ctor.call_count, 1, "expected one IMAP connection for the batch")
        # Subsequent calls probe the cached handle instead of reconnecting
        self.assertEqual(mock_imap.noop.call_count, 2)

    def test_finalize_reconnects_after_a_dropped_connection(self):
        """A connection the server dropped mid-batch is reopened, not lost."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        dead = MagicMock()
        dead.capabilities = ("IMAP4REV1", "UIDPLUS")
        dead.noop.return_value = ("OK", [b"NOOP completed"])
        dead.select.side_effect = Exception("socket error: EOF")

        live = MagicMock()
        live.capabilities = ("IMAP4REV1", "UIDPLUS")
        live.noop.return_value = ("OK", [b"NOOP completed"])
        live.uid.side_effect = lambda command, *a: (
            ("OK", [b"3"]) if command == "SEARCH" else ("OK", [b"1"])
        )

        with patch("imaplib.IMAP4_SSL", side_effect=[dead, live]) as ctor:
            adapter._finalize_message("<m@test.com>", "Hermes_Working")

        self.assertEqual(ctor.call_count, 2)
        live.select.assert_called_once_with("Hermes_Working")

    def test_disconnect_closes_finalize_connection(self):
        """disconnect() releases the cached handle so no socket is left behind."""
        import asyncio

        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        mock_imap = MagicMock()
        adapter._finalize_conn = mock_imap

        asyncio.run(adapter.disconnect())

        self.assertIsNone(adapter._finalize_conn)
        mock_imap.logout.assert_called_once()

    # ------------------------------------------------------------------
    # _fetch_new_messages: Working-folder move
    # ------------------------------------------------------------------

    def test_fetch_moves_to_working_when_configured(self):
        """_fetch_new_messages should MOVE mail to Hermes_Working and set source_folder."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        raw = self._make_raw_email()
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"10"])
            if command == "fetch":
                return ("OK", [(b"10", raw)])
            if command == "MOVE":
                return ("OK", [b"1"])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "Hermes_Working")

        # Verify a MOVE to Hermes_Working was issued
        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "MOVE"]
        self.assertTrue(move_calls, "Expected UID MOVE to be called")
        self.assertEqual(move_calls[0].args[2], "Hermes_Working")

    def test_fetch_skips_move_when_working_empty(self):
        """_fetch_new_messages should NOT issue any MOVE when working_folder is empty."""
        adapter = self._make_adapter({
            "working_folder": "",
            "done_folder": "Hermes_Done",
        })

        raw = self._make_raw_email()
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"11"])
            if command == "fetch":
                return ("OK", [(b"11", raw)])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "INBOX")

        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] in ("MOVE", "COPY")]
        self.assertEqual(len(move_calls), 0, "No MOVE/COPY should be issued when working_folder is empty")

    def test_fetch_source_folder_inbox_when_move_fails(self):
        """source_folder should remain 'INBOX' when the MOVE to Working fails."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        raw = self._make_raw_email()
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"12"])
            if command == "fetch":
                return ("OK", [(b"12", raw)])
            if command == "MOVE":
                return ("NO", [b"failed"])
            if command == "COPY":
                return ("NO", [b"failed"])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "INBOX")

    def test_fetch_skips_move_when_done_empty(self):
        """done_folder='' disables ALL moves, even when working_folder is set."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "",
        })

        raw = self._make_raw_email()
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"13"])
            if command == "fetch":
                return ("OK", [(b"13", raw)])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "INBOX")
        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] in ("MOVE", "COPY")]
        self.assertEqual(len(move_calls), 0, "No move should happen when done_folder is empty")

    def test_fetch_skips_move_when_no_message_id(self):
        """Mail with no Message-ID must NOT be moved to Working (it could not be
        re-located there for the final MOVE → Done) — it stays in INBOX."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        raw = self._make_raw_email(message_id="")
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"14"])
            if command == "fetch":
                return ("OK", [(b"14", raw)])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_folder"], "INBOX")
        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] in ("MOVE", "COPY")]
        self.assertEqual(len(move_calls), 0, "No Working move without a Message-ID")

    # ------------------------------------------------------------------
    # _finalize_message
    # ------------------------------------------------------------------

    def test_finalize_noop_when_done_empty(self):
        """_finalize_message must do nothing when done_folder is empty."""
        adapter = self._make_adapter({"done_folder": ""})

        with patch("imaplib.IMAP4_SSL") as mock_cls:
            adapter._finalize_message("<msg@test.com>", "Hermes_Working")
            mock_cls.assert_not_called()

    def test_finalize_noop_when_already_in_done(self):
        """_finalize_message must do nothing when source_folder == _done_folder."""
        adapter = self._make_adapter({"done_folder": "Hermes_Done"})

        with patch("imaplib.IMAP4_SSL") as mock_cls:
            adapter._finalize_message("<msg@test.com>", "Hermes_Done")
            mock_cls.assert_not_called()

    def test_finalize_moves_to_done(self):
        """_finalize_message should open IMAP, SEARCH by Message-ID, and MOVE to Done."""
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "SEARCH":
                return ("OK", [b"42"])
            if command == "MOVE":
                return ("OK", [b"1"])
            return ("OK", [b""])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            adapter._finalize_message("<msg@test.com>", "Hermes_Working")

        mock_imap.select.assert_called_once_with("Hermes_Working")
        search_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "SEARCH"]
        self.assertTrue(search_calls)
        move_calls = [c for c in mock_imap.uid.call_args_list if c.args[0] == "MOVE"]
        self.assertTrue(move_calls)
        self.assertEqual(move_calls[0].args[2], "Hermes_Done")

    # ------------------------------------------------------------------
    # Full dispatch flow
    # ------------------------------------------------------------------

    def test_finalize_still_runs_when_handle_message_raises(self):
        """_finalize_message must be called even if handle_message raises."""
        import asyncio
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        finalize_called = []

        def fake_finalize(message_id, source_folder):
            finalize_called.append((message_id, source_folder))

        adapter._finalize_message = fake_finalize

        async def raising_handler(event):
            raise RuntimeError("agent crashed")

        adapter.handle_message = raising_handler

        msg_data = {
            "uid": b"50",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Crash test",
            "message_id": "<crash@test.com>",
            "in_reply_to": "",
            "body": "Will crash",
            "attachments": [],
            "date": "",
            "source_folder": "Hermes_Working",
        }

        # user@test.com must clear the allowlist gate to reach handle_message
        # (upstream default-denies when EMAIL_ALLOWED_USERS is unset), so the
        # finalize-on-raise invariant is actually exercised. Env is read at
        # dispatch time, so patch around the call (not just at construction).
        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            with self.assertRaises(RuntimeError):
                asyncio.run(adapter._dispatch_message(msg_data))

        self.assertEqual(len(finalize_called), 1)
        self.assertEqual(finalize_called[0][0], "<crash@test.com>")
        self.assertEqual(finalize_called[0][1], "Hermes_Working")

    def test_finalize_moves_to_done_after_dispatch(self):
        """Full dispatch flow: after handle_message returns, mail is moved to Done."""
        import asyncio
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        finalize_calls = []

        def fake_finalize(message_id, source_folder):
            finalize_calls.append((message_id, source_folder))

        adapter._finalize_message = fake_finalize

        async def noop_handler(event):
            pass

        adapter.handle_message = noop_handler

        msg_data = {
            "uid": b"60",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Done test",
            "message_id": "<done@test.com>",
            "in_reply_to": "",
            "body": "Process me",
            "attachments": [],
            "date": "",
            "source_folder": "Hermes_Working",
        }

        # Authorize the sender so it reaches handle_message (upstream default-
        # denies an unset allowlist); env is read at dispatch time.
        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(adapter._dispatch_message(msg_data))

        self.assertEqual(len(finalize_calls), 1)
        self.assertEqual(finalize_calls[0][0], "<done@test.com>")
        self.assertEqual(finalize_calls[0][1], "Hermes_Working")

    def test_dispatch_uses_inbox_fallback_when_no_source_folder(self):
        """_dispatch_message should fall back to 'INBOX' if source_folder is absent."""
        import asyncio
        adapter = self._make_adapter({
            "working_folder": "Hermes_Working",
            "done_folder": "Hermes_Done",
        })

        finalize_calls = []

        def fake_finalize(message_id, source_folder):
            finalize_calls.append(source_folder)

        adapter._finalize_message = fake_finalize

        async def noop_handler(event):
            pass

        adapter.handle_message = noop_handler

        # Omit source_folder — simulates a pre-patch message dict
        msg_data = {
            "uid": b"70",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Fallback test",
            "message_id": "<fallback@test.com>",
            "in_reply_to": "",
            "body": "No source_folder key",
            "attachments": [],
            "date": "",
        }

        # Authorize the sender so it reaches handle_message (upstream default-
        # denies an unset allowlist); env is read at dispatch time.
        with patch.dict(os.environ, {"EMAIL_ALLOW_ALL_USERS": "true"}):
            asyncio.run(adapter._dispatch_message(msg_data))

        self.assertEqual(finalize_calls, ["INBOX"])

    # ------------------------------------------------------------------
    # Early-drop paths must still finalize (no mail stranded in Working)
    # ------------------------------------------------------------------

    def _dropping_adapter(self):
        """Adapter whose handle_message records calls and finalize records folders."""
        adapter = self._make_adapter(
            {"working_folder": "Hermes_Working", "done_folder": "Hermes_Done"},
        )
        adapter._handled = []
        adapter._finalized = []

        async def recording_handler(event):
            adapter._handled.append(event)

        def recording_finalize(message_id, source_folder):
            adapter._finalized.append((message_id, source_folder))

        adapter.handle_message = recording_handler
        adapter._finalize_message = recording_finalize
        return adapter

    def _drop_msg(self, sender):
        return {
            "uid": b"80",
            "sender_addr": sender,
            "sender_name": "Someone",
            "subject": "Hi",
            "message_id": "<drop@test.com>",
            "in_reply_to": "",
            "body": "body",
            "attachments": [],
            "date": "",
            "source_folder": "Hermes_Working",
        }

    def test_dispatch_finalizes_dropped_self_message(self):
        """A self-message is dropped (handler not called) but still finalized → Done."""
        import asyncio
        adapter = self._dropping_adapter()

        asyncio.run(adapter._dispatch_message(self._drop_msg("hermes@test.com")))

        self.assertEqual(adapter._handled, [], "self-message must not be handled")
        self.assertEqual(adapter._finalized, [("<drop@test.com>", "Hermes_Working")],
                         "dropped self-message must still be finalized out of Working")

    def test_dispatch_finalizes_non_allowlisted_sender(self):
        """A non-allowlisted sender is dropped but still finalized → Done."""
        import asyncio
        adapter = self._dropping_adapter()

        # EMAIL_ALLOWED_USERS is read from env at dispatch time, so patch it
        # around the dispatch call (not just at construction).
        with patch.dict(os.environ, {"EMAIL_ALLOWED_USERS": "boss@test.com"}):
            asyncio.run(adapter._dispatch_message(self._drop_msg("stranger@test.com")))

        self.assertEqual(adapter._handled, [], "non-allowlisted sender must not be handled")
        self.assertEqual(adapter._finalized, [("<drop@test.com>", "Hermes_Working")],
                         "dropped non-allowlisted mail must still be finalized out of Working")


if __name__ == "__main__":
    unittest.main()
