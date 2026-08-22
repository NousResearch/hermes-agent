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
from unittest.mock import patch, MagicMock, AsyncMock, ANY, call

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
        self.assertEqual(results[0]["_imap_uid"], b"3")
        self.assertNotIn(b"3", adapter._seen_uids)


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

    def test_check_inbox_times_out_stalled_executor_fetch(self):
        """A wedged IMAP worker must enter reconnect instead of pinning the poll task."""
        import asyncio
        import threading
        import time

        from plugins.platforms.email import adapter as email_adapter

        adapter = self._make_adapter()
        fetch_started = threading.Event()
        release_fetch = threading.Event()
        notified = []
        aborted = []

        def stalled_fetch(dispatch_callback=None):
            fetch_started.set()
            release_fetch.wait(timeout=5.0)
            return []

        async def mock_fatal_handler(failed_adapter):
            notified.append(failed_adapter)

        def abort_fetch():
            aborted.append(True)
            release_fetch.set()

        async def exercise():
            started = time.monotonic()
            await adapter._check_inbox()
            return time.monotonic() - started

        adapter._fetch_new_messages = stalled_fetch
        adapter._abort_active_imap = abort_fetch
        adapter.set_fatal_error_handler(mock_fatal_handler)

        with patch.object(email_adapter, "IMAP_FETCH_WATCHDOG_TIMEOUT", 0.5):
            elapsed = asyncio.run(exercise())

        self.assertTrue(fetch_started.is_set())
        self.assertLess(elapsed, 2.0)
        self.assertEqual(notified, [adapter])
        self.assertEqual(aborted, [True])
        self.assertEqual(adapter.fatal_error_code, "email_imap_fetch_timeout")
        self.assertTrue(adapter.fatal_error_retryable)
        self.assertIn("no progress for 0.5s", adapter.fatal_error_message)

    def test_progressing_fetch_can_exceed_watchdog_total_duration(self):
        """The watchdog measures inactivity, not total healthy batch time."""
        import asyncio
        import time

        from plugins.platforms.email import adapter as email_adapter

        adapter = self._make_adapter()
        notified = []

        def progressing_fetch(dispatch_callback=None):
            for _ in range(5):
                time.sleep(0.15)
                adapter._record_fetch_progress()
            return []

        async def mock_fatal_handler(failed_adapter):
            notified.append(failed_adapter)

        adapter._fetch_new_messages = progressing_fetch
        adapter.set_fatal_error_handler(mock_fatal_handler)

        with patch.object(email_adapter, "IMAP_FETCH_WATCHDOG_TIMEOUT", 0.5):
            asyncio.run(adapter._check_inbox())

        self.assertEqual(notified, [])

    def test_stalled_dispatch_releases_fetch_gate(self):
        """A wedged dispatch must not leave reconnect blocked indefinitely."""
        import asyncio
        import time

        from plugins.platforms.email import adapter as email_adapter

        adapter = self._make_adapter()
        dispatch_cancelled = asyncio.Event()
        notified = []
        raw_email = MIMEText("Body", "plain", "utf-8")
        raw_email["From"] = "sender@test.com"
        raw_email["Subject"] = "Stalled dispatch"
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        async def stalled_dispatch(msg_data):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                dispatch_cancelled.set()
                raise

        async def mock_fatal_handler(failed_adapter):
            notified.append(failed_adapter)

        async def exercise():
            await adapter._check_inbox()
            deadline = time.monotonic() + 2.0
            while adapter._address in adapter._active_fetches:
                if time.monotonic() >= deadline:
                    self.fail("dispatch timeout did not release the fetch gate")
                await asyncio.sleep(0.01)

        mock_imap.uid.side_effect = uid_handler
        adapter._dispatch_message = stalled_dispatch
        adapter.set_fatal_error_handler(mock_fatal_handler)

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), patch.object(
            email_adapter, "IMAP_FETCH_WATCHDOG_TIMEOUT", 0.5
        ):
            asyncio.run(exercise())

        self.assertTrue(dispatch_cancelled.is_set())
        self.assertEqual(notified, [adapter])
        self.assertEqual(adapter.fatal_error_code, "email_message_dispatch_timeout")
        self.assertTrue(adapter.fatal_error_retryable)
        self.assertNotIn(b"1", adapter._seen_uids)

    def test_dispatch_timeout_boundary_success_is_committed_once(self):
        """A dispatch completing during timeout handling must still commit its UID."""
        import asyncio

        adapter = self._make_adapter()
        notified = []
        raw_email = MIMEText("Body", "plain", "utf-8")
        raw_email["From"] = "sender@test.com"
        raw_email["Subject"] = "Boundary success"
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        class BoundaryFuture:
            def __init__(self):
                self.result_calls = 0

            def result(self, timeout=None):
                self.result_calls += 1
                if self.result_calls == 1:
                    raise TimeoutError()
                return None

            def done(self):
                return True

        boundary_future = BoundaryFuture()

        def complete_at_boundary(coro, loop):
            coro.close()
            return boundary_future

        async def mock_fatal_handler(failed_adapter):
            notified.append(failed_adapter)

        mock_imap.uid.side_effect = uid_handler
        adapter.set_fatal_error_handler(mock_fatal_handler)

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), patch(
            "asyncio.run_coroutine_threadsafe", side_effect=complete_at_boundary
        ):
            asyncio.run(adapter._check_inbox())

        self.assertEqual(boundary_future.result_calls, 2)
        self.assertEqual(notified, [])
        self.assertIn(b"1", adapter._seen_uids)
        self.assertIn(
            call("store", b"1", "+FLAGS", "(\\Seen)"),
            mock_imap.uid.call_args_list,
        )

    def test_dispatch_failure_is_not_reported_as_imap_failure(self):
        """A downstream dispatch exception must retain its own failure classification."""
        import asyncio

        adapter = self._make_adapter()
        notified = []
        raw_email = MIMEText("Body", "plain", "utf-8")
        raw_email["From"] = "sender@test.com"
        raw_email["Subject"] = "Dispatch failure"
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        async def failed_dispatch(msg_data):
            raise RuntimeError("downstream unavailable")

        async def mock_fatal_handler(failed_adapter):
            notified.append(failed_adapter)

        mock_imap.uid.side_effect = uid_handler
        adapter._dispatch_message = failed_dispatch
        adapter.set_fatal_error_handler(mock_fatal_handler)

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            asyncio.run(adapter._check_inbox())

        self.assertEqual(notified, [adapter])
        self.assertEqual(adapter.fatal_error_code, "email_message_dispatch_failed")
        self.assertIn("downstream unavailable", adapter.fatal_error_message)
        self.assertFalse(adapter._last_fetch_failed)
        self.assertNotIn(b"1", adapter._seen_uids)

    def test_timeout_keeps_completed_message_eligible_for_reconnect(self):
        """A later stalled UID must not consume an earlier undispatched UID."""
        import asyncio
        import threading

        from plugins.platforms.email import adapter as email_adapter

        adapter = self._make_adapter()
        release_fetch = threading.Event()
        dispatched = []
        raw_email = MIMEText("Body", "plain", "utf-8")
        raw_email["From"] = "sender@test.com"
        raw_email["Subject"] = "First before stall"

        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1 2"])
            if command == "fetch" and args[0] == b"1":
                return ("OK", [(b"1", raw_email.as_bytes())])
            if command == "fetch":
                release_fetch.wait(timeout=5.0)
                raise OSError("aborted")
            return ("NO", [])

        def shutdown():
            release_fetch.set()

        async def dispatch(msg_data):
            dispatched.append(msg_data)

        mock_imap.uid.side_effect = uid_handler
        mock_imap.shutdown.side_effect = shutdown
        adapter._dispatch_message = dispatch

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), patch.object(
            email_adapter, "IMAP_FETCH_WATCHDOG_TIMEOUT", 0.5
        ):
            asyncio.run(adapter._check_inbox())

        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0]["subject"], "First before stall")
        self.assertIn(b"1", adapter._seen_uids)
        fetch_calls = [
            call for call in mock_imap.uid.call_args_list
            if call.args[0] == "fetch"
        ]
        self.assertEqual(fetch_calls[0].args[2], "(BODY.PEEK[])")
        self.assertIn(
            call("store", b"1", "+FLAGS", "(\\Seen)"),
            mock_imap.uid.call_args_list,
        )

    def test_reconnect_waits_for_previous_fetch_worker_to_exit(self):
        """A replacement adapter must not overlap an abandoned account fetch."""
        import asyncio

        adapter = self._make_adapter()
        token = object()
        with adapter._active_fetches_lock:
            adapter._active_fetches[adapter._address] = token
        try:
            connected = asyncio.run(adapter.connect(is_reconnect=True))
        finally:
            with adapter._active_fetches_lock:
                if adapter._active_fetches.get(adapter._address) is token:
                    adapter._active_fetches.pop(adapter._address)

        self.assertFalse(connected)
        self.assertEqual(
            adapter.fatal_error_code, "email_imap_fetch_still_stopping"
        )
        self.assertTrue(adapter.fatal_error_retryable)

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
        self.assertEqual(results[0]["_imap_uid"], b"1")
        self.assertNotIn(b"1", adapter._seen_uids)  # dispatch has not committed it
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
        self.assertEqual(results[0]["_imap_uid"], b"2")
        self.assertNotIn(b"2", adapter._seen_uids)
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


if __name__ == "__main__":
    unittest.main()
