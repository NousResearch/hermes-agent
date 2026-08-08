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


class TestUidCursorFile(unittest.TestCase):
    """The persisted resume point itself (plugins/platforms/email/uid_cursor.py)."""

    def _path(self, tmp):
        from pathlib import Path
        return Path(tmp) / "email_uid_cursor_hermes_test.com.json"

    def _cursor(self, tmp):
        from plugins.platforms.email.uid_cursor import EmailUidCursor
        return EmailUidCursor("hermes@test.com", path=self._path(tmp))

    def test_missing_file_means_no_resume_point(self):
        """A mailbox that was never tracked must re-baseline, not resume from 0."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cursor = self._cursor(tmp)
            self.assertIsNone(cursor.resume_from("42"))
            self.assertEqual(cursor.uid, 0)

    def test_baseline_round_trips_across_instances(self):
        """A restart reads back the UID the previous run stopped at."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._cursor(tmp).baseline("42", 40)
            self.assertEqual(self._cursor(tmp).resume_from("42"), 40)

    def test_other_uidvalidity_generation_is_not_resumed(self):
        """UIDs are only comparable inside one UIDVALIDITY generation."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._cursor(tmp).baseline("42", 40)
            self.assertIsNone(self._cursor(tmp).resume_from("99"))

    def test_advance_is_monotonic(self):
        """An out-of-order UID must never drag the cursor backwards."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cursor = self._cursor(tmp)
            cursor.baseline("42", 40)
            cursor.advance(50)
            cursor.advance(20)
            cursor.flush()
            self.assertEqual(self._cursor(tmp).resume_from("42"), 50)

    def test_corrupt_file_degrades_to_no_resume_point(self):
        """A truncated write must re-baseline rather than raise on startup."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._path(tmp).write_text("{not json", encoding="utf-8")
            self.assertIsNone(self._cursor(tmp).resume_from("42"))

    def test_baseline_at_zero_is_a_resume_point(self):
        """An empty mailbox baselines at UID 0; that cursor must still resume.

        Treating a stored 0 as "no cursor" would re-baseline past mail that
        arrived during the next outage — the #80925 hole again.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._cursor(tmp).baseline("42", 0)
            self.assertEqual(self._cursor(tmp).resume_from("42"), 0)

    def test_unwritable_path_does_not_raise(self):
        """Persistence is best-effort: polling must not break on a bad path."""
        import tempfile
        from pathlib import Path
        from plugins.platforms.email.uid_cursor import EmailUidCursor
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            cursor = EmailUidCursor("hermes@test.com", path=blocker / "cursor.json")
            cursor.baseline("42", 40)          # must not raise
            cursor.advance(41)
            cursor.flush()                     # must not raise
            self.assertEqual(cursor.uid, 41)

    def test_path_is_per_mailbox_address(self):
        """Two addresses under one profile must not share a cursor file.

        Profile multiplexing can resolve two email adapters to the same home; a
        shared file would make each mailbox invalidate the other's cursor on
        every connect and silently re-baseline.
        """
        import tempfile
        from pathlib import Path
        from plugins.platforms.email.uid_cursor import cursor_path
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HERMES_HOME": tmp}):
                one = cursor_path("one@test.com")
                two = cursor_path("two@test.com")
            self.assertNotEqual(one, two)
            self.assertEqual(one.parent, Path(tmp) / "gateway")


class TestResumeAfterDowntime(unittest.TestCase):
    """platforms.email.resume_after_downtime — the #80925 downtime backlog."""

    ADDRESS = "hermes@test.com"

    def _make_adapter(self, home, resume=True):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": self.ADDRESS,
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
            "HERMES_HOME": home,
        }):
            from plugins.platforms.email.adapter import EmailAdapter
            extra = {"resume_after_downtime": True} if resume else {}
            return EmailAdapter(PlatformConfig(enabled=True, extra=extra))

    def _cursor_path(self, home):
        from plugins.platforms.email.uid_cursor import cursor_path
        with patch.dict(os.environ, {"HERMES_HOME": home}):
            return cursor_path(self.ADDRESS)

    def _seed_cursor(self, home, uidvalidity, uid):
        """Write the cursor a previous run would have left behind."""
        from plugins.platforms.email.uid_cursor import EmailUidCursor
        path = self._cursor_path(home)
        EmailUidCursor(self.ADDRESS, path=path).baseline(uidvalidity, uid)
        return path

    def _email(self, subject):
        msg = MIMEText("body", "plain", "utf-8")
        msg["From"] = "user@test.com"
        msg["Subject"] = subject
        msg["Message-ID"] = f"<{subject}@test.com>"
        return msg

    def _mock_imap(self, uidvalidity=b"42", search=b"", messages=None):
        messages = messages or {}
        mock_imap = MagicMock()
        mock_imap.response.return_value = ("UIDVALIDITY", [uidvalidity])

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [search])
            if command == "fetch":
                uid = args[0]
                if uid in messages:
                    return ("OK", [(uid, messages[uid].as_bytes())])
                return ("NO", [])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        return mock_imap

    def _connect(self, adapter, mock_imap):
        """Run connect() against a mocked IMAP/SMTP, with the poll loop stubbed.

        connect() starts the poll task, and asyncio.run() gives it one step
        during shutdown cancellation — enough for run_in_executor to submit a
        real _fetch_new_messages to a thread. That thread outlives the patch
        context (racing the assertions, or worse, dialing the real IMAP host),
        so the loop itself is stubbed out instead of cancelled after the fact.
        """
        import asyncio
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("smtplib.SMTP", return_value=MagicMock()), \
             patch.object(type(adapter), "_poll_loop", new=AsyncMock()):
            result = asyncio.run(adapter.connect())
        adapter._running = False
        if adapter._poll_task:
            adapter._poll_task.cancel()
        return result

    def _search_args(self, mock_imap):
        """Arguments of the IMAP UID SEARCH the adapter issued."""
        for call in mock_imap.uid.call_args_list:
            if call.args and call.args[0] == "search":
                return call.args
        return ()

    def test_first_enable_baselines_and_answers_nothing(self):
        """Turning the option on must not answer mail already in the INBOX."""
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            adapter = self._make_adapter(home)
            mock_imap = self._mock_imap(search=b"1 2 3")

            self.assertTrue(self._connect(adapter, mock_imap))

            self.assertEqual(adapter._uid_cursor.uid, 3)
            # The startup skip set is what swallowed downtime mail; the resume
            # path must not populate it at all.
            self.assertEqual(adapter._seen_uids, set())
            stored = json.loads(self._cursor_path(home).read_text(encoding="utf-8"))
            self.assertEqual(stored["uid"], 3)
            self.assertEqual(stored["uidvalidity"], "42")

    def test_downtime_mail_is_answered_after_restart(self):
        """The reported bug: mail that arrived while down is dispatched."""
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            path = self._seed_cursor(home, "42", 3)
            adapter = self._make_adapter(home)
            mock_imap = self._mock_imap(
                search=b"4 5",
                messages={b"4": self._email("Downtime one"),
                          b"5": self._email("Downtime two")},
            )

            with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
                results = adapter._fetch_new_messages()

            self.assertEqual([r["subject"] for r in results],
                             ["Downtime one", "Downtime two"])
            # Searched the UID range above the cursor, not UNSEEN.
            self.assertEqual(self._search_args(mock_imap),
                             ("search", None, "UID", "4:*"))
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["uid"], 5
            )

    def test_answered_mail_is_not_reanswered_after_restart(self):
        """A UID at or below the cursor must never be dispatched again.

        RFC 3501 makes an ``n:*`` range always include the mailbox's highest UID
        even when n is above it, so the server hands back already-answered mail
        on every poll; the cursor filter is what stops the reply loop.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            self._seed_cursor(home, "42", 5)
            adapter = self._make_adapter(home)
            mock_imap = self._mock_imap(
                search=b"5", messages={b"5": self._email("Already answered")}
            )

            with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
                results = adapter._fetch_new_messages()

            self.assertEqual(results, [])
            self.assertEqual(adapter._uid_cursor.uid, 5)

    def test_empty_mailbox_baseline_still_resumes(self):
        """A stored cursor of UID 0 must resume, not trigger a re-baseline.

        The baseline of a mailbox that was empty when the option kicked in is
        0. Re-baselining on the next start would set the cursor past mail that
        arrived during the outage — the #80925 hole again.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            self._seed_cursor(home, "42", 0)
            adapter = self._make_adapter(home)
            mock_imap = self._mock_imap(
                search=b"1 2",
                messages={b"1": self._email("Outage one"),
                          b"2": self._email("Outage two")},
            )

            self.assertTrue(self._connect(adapter, mock_imap))
            # Resumed at 0; a re-baseline here would move the cursor to 2 and
            # swallow both waiting messages.
            self.assertEqual(adapter._uid_cursor.uid, 0)

            with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
                results = adapter._fetch_new_messages()

            self.assertEqual(len(results), 2)
            self.assertEqual(self._search_args(mock_imap),
                             ("search", None, "UID", "1:*"))

    def test_mail_a_human_already_read_is_still_answered(self):
        """The server-side \\Seen flag must not act as the queue.

        A person opening the mailbox in a mail client sets \\Seen without the
        agent having answered anything, which hides that mail from an UNSEEN
        search forever. The resume path must not ask for UNSEEN.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            self._seed_cursor(home, "42", 3)
            adapter = self._make_adapter(home)
            mock_imap = self._mock_imap(
                search=b"4", messages={b"4": self._email("Read on a phone")}
            )

            with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
                results = adapter._fetch_new_messages()

            self.assertEqual(len(results), 1)
            self.assertNotIn("UNSEEN", self._search_args(mock_imap))

    def test_uidvalidity_change_rebaselines_without_replay(self):
        """A recreated mailbox must not be answered from UID 1."""
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            path = self._seed_cursor(home, "42", 3)
            adapter = self._make_adapter(home)
            mock_imap = self._mock_imap(uidvalidity=b"99", search=b"1 2 3 4 5")

            self.assertTrue(self._connect(adapter, mock_imap))

            self.assertEqual(adapter._uid_cursor.uid, 5)
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored["uidvalidity"], "99")
            self.assertEqual(stored["uid"], 5)

    def test_corrupt_cursor_file_rebaselines_without_replay(self):
        """An unreadable cursor must degrade to today's behavior, not crash."""
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            path = self._seed_cursor(home, "42", 3)
            path.write_text("{ truncated", encoding="utf-8")
            adapter = self._make_adapter(home)
            mock_imap = self._mock_imap(search=b"1 2 3 4")

            self.assertTrue(self._connect(adapter, mock_imap))

            self.assertEqual(adapter._uid_cursor.uid, 4)

    def test_missing_uidvalidity_does_not_resume(self):
        """Without a generation to pin UIDs to, re-baseline rather than guess."""
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            self._seed_cursor(home, "42", 3)
            adapter = self._make_adapter(home)
            mock_imap = self._mock_imap(search=b"1 2 3 4 5")
            mock_imap.response.return_value = ("UIDVALIDITY", [None])

            self.assertTrue(self._connect(adapter, mock_imap))

            self.assertEqual(adapter._uid_cursor.uid, 5)

    def test_failed_baseline_search_disables_resume_for_the_run(self):
        """A non-OK baseline search must not record baseline 0.

        Baseline 0 also means "empty mailbox", so recording it would make the
        first poll search UID 1:* and answer the whole INBOX. The run must
        degrade to the option-off UNSEEN path and write no cursor file.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            adapter = self._make_adapter(home)
            mock_imap = self._mock_imap(search=b"")
            ok_handler = mock_imap.uid.side_effect

            def uid_handler(command, *args):
                if command == "search" and args == (None, "ALL"):
                    return ("NO", [])
                return ok_handler(command, *args)

            mock_imap.uid.side_effect = uid_handler

            self.assertTrue(self._connect(adapter, mock_imap))

            self.assertIsNone(adapter._uid_cursor)
            self.assertFalse(self._cursor_path(home).exists())

            fetch_imap = self._mock_imap(search=b"")
            with patch("imaplib.IMAP4_SSL", return_value=fetch_imap):
                adapter._fetch_new_messages()
            self.assertEqual(self._search_args(fetch_imap),
                             ("search", None, "UNSEEN"))

    def test_default_keeps_todays_behavior(self):
        """With the option off nothing changes and no state file is written."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as home:
            adapter = self._make_adapter(home, resume=False)
            self.assertIsNone(adapter._uid_cursor)

            mock_imap = self._mock_imap(search=b"1 2 3")
            self.assertTrue(self._connect(adapter, mock_imap))
            self.assertEqual(len(adapter._seen_uids), 3)

            fetch_imap = self._mock_imap(search=b"")
            with patch("imaplib.IMAP4_SSL", return_value=fetch_imap):
                adapter._fetch_new_messages()
            self.assertEqual(self._search_args(fetch_imap),
                             ("search", None, "UNSEEN"))
            self.assertFalse((Path(home) / "gateway").exists())

    def test_default_path_still_swallows_mail_present_at_startup(self):
        """The #80925 behavior itself, pinned on the default path.

        Mail sitting in the INBOX at connect() is marked seen and never
        dispatched, however unread it is. This is the bug the opt-in path fixes
        — see test_downtime_mail_is_answered_after_restart for the contrast —
        and it must stay exactly this way when the option is off.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            adapter = self._make_adapter(home, resume=False)
            self._connect(adapter, self._mock_imap(search=b"1 2 3"))

            # UID 3 arrived while the gateway was down and is still UNSEEN.
            fetch_imap = self._mock_imap(
                search=b"3", messages={b"3": self._email("Sent during downtime")}
            )
            with patch("imaplib.IMAP4_SSL", return_value=fetch_imap):
                results = adapter._fetch_new_messages()

            self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
