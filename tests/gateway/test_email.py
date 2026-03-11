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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

from gateway.platforms.base import SendResult


class TestPlatformEnum(unittest.TestCase):
    """Verify EMAIL is in the Platform enum."""

    def test_email_in_platform_enum(self):
        from gateway.config import Platform
        self.assertEqual(Platform.EMAIL.value, "email")


class TestConfigEnvOverrides(unittest.TestCase):
    """Verify email config is loaded from environment variables."""

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }, clear=False)
    def test_email_config_loaded_from_env(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)
        self.assertIn(Platform.EMAIL, config.platforms)
        self.assertTrue(config.platforms[Platform.EMAIL].enabled)
        self.assertEqual(config.platforms[Platform.EMAIL].extra["address"], "hermes@test.com")

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

    @patch.dict(os.environ, {}, clear=True)
    def test_email_not_loaded_without_env(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)
        self.assertNotIn(Platform.EMAIL, config.platforms)

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }, clear=False)
    def test_email_in_connected_platforms(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)
        connected = config.get_connected_platforms()
        self.assertIn(Platform.EMAIL, connected)


class TestCheckRequirements(unittest.TestCase):
    """Verify check_email_requirements function."""

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "a@b.com",
        "EMAIL_PASSWORD": "pw",
        "EMAIL_IMAP_HOST": "imap.b.com",
        "EMAIL_SMTP_HOST": "smtp.b.com",
    }, clear=False)
    def test_requirements_met(self):
        from gateway.platforms.email import check_email_requirements
        self.assertTrue(check_email_requirements())

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "a@b.com",
    }, clear=True)
    def test_requirements_not_met(self):
        from gateway.platforms.email import check_email_requirements
        self.assertFalse(check_email_requirements())

    @patch.dict(os.environ, {}, clear=True)
    def test_requirements_empty_env(self):
        from gateway.platforms.email import check_email_requirements
        self.assertFalse(check_email_requirements())


class TestHelperFunctions(unittest.TestCase):
    """Test email parsing helper functions."""

    def test_decode_header_plain(self):
        from gateway.platforms.email import _decode_header_value
        self.assertEqual(_decode_header_value("Hello World"), "Hello World")

    def test_decode_header_encoded(self):
        from gateway.platforms.email import _decode_header_value
        # RFC 2047 encoded subject
        encoded = "=?utf-8?B?TWVyaGFiYQ==?="  # "Merhaba" in base64
        result = _decode_header_value(encoded)
        self.assertEqual(result, "Merhaba")

    def test_extract_email_address_with_name(self):
        from gateway.platforms.email import _extract_email_address
        self.assertEqual(
            _extract_email_address("John Doe <john@example.com>"),
            "john@example.com"
        )

    def test_extract_email_address_bare(self):
        from gateway.platforms.email import _extract_email_address
        self.assertEqual(
            _extract_email_address("john@example.com"),
            "john@example.com"
        )

    def test_extract_email_address_uppercase(self):
        from gateway.platforms.email import _extract_email_address
        self.assertEqual(
            _extract_email_address("John@Example.COM"),
            "john@example.com"
        )

    def test_strip_html_basic(self):
        from gateway.platforms.email import _strip_html
        html = "<p>Hello <b>world</b></p>"
        result = _strip_html(html)
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertNotIn("<p>", result)
        self.assertNotIn("<b>", result)

    def test_strip_html_br_tags(self):
        from gateway.platforms.email import _strip_html
        html = "Line 1<br>Line 2<br/>Line 3"
        result = _strip_html(html)
        self.assertIn("Line 1", result)
        self.assertIn("Line 2", result)

    def test_strip_html_entities(self):
        from gateway.platforms.email import _strip_html
        html = "a &amp; b &lt; c &gt; d"
        result = _strip_html(html)
        self.assertIn("a & b", result)


class TestExtractTextBody(unittest.TestCase):
    """Test email body extraction from different message formats."""

    def test_plain_text_body(self):
        from gateway.platforms.email import _extract_text_body
        msg = MIMEText("Hello, this is a test.", "plain", "utf-8")
        result = _extract_text_body(msg)
        self.assertEqual(result, "Hello, this is a test.")

    def test_html_body_fallback(self):
        from gateway.platforms.email import _extract_text_body
        msg = MIMEText("<p>Hello from HTML</p>", "html", "utf-8")
        result = _extract_text_body(msg)
        self.assertIn("Hello from HTML", result)
        self.assertNotIn("<p>", result)

    def test_multipart_prefers_plain(self):
        from gateway.platforms.email import _extract_text_body
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("<p>HTML version</p>", "html", "utf-8"))
        msg.attach(MIMEText("Plain version", "plain", "utf-8"))
        result = _extract_text_body(msg)
        self.assertEqual(result, "Plain version")

    def test_multipart_html_only(self):
        from gateway.platforms.email import _extract_text_body
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("<p>Only HTML</p>", "html", "utf-8"))
        result = _extract_text_body(msg)
        self.assertIn("Only HTML", result)

    def test_empty_body(self):
        from gateway.platforms.email import _extract_text_body
        msg = MIMEText("", "plain", "utf-8")
        result = _extract_text_body(msg)
        self.assertEqual(result, "")


class TestExtractAttachments(unittest.TestCase):
    """Test attachment extraction and caching."""

    def test_no_attachments(self):
        from gateway.platforms.email import _extract_attachments
        msg = MIMEText("No attachments here.", "plain", "utf-8")
        result = _extract_attachments(msg)
        self.assertEqual(result, [])

    @patch("gateway.platforms.email.cache_document_from_bytes")
    def test_document_attachment(self, mock_cache):
        from gateway.platforms.email import _extract_attachments
        mock_cache.return_value = "/tmp/cached_doc.pdf"

        msg = MIMEMultipart()
        msg.attach(MIMEText("See attached.", "plain", "utf-8"))

        part = MIMEBase("application", "pdf")
        part.set_payload(b"%PDF-1.4 fake pdf content")
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=report.pdf")
        msg.attach(part)

        result = _extract_attachments(msg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "document")
        self.assertEqual(result[0]["filename"], "report.pdf")
        mock_cache.assert_called_once()

    @patch("gateway.platforms.email.cache_image_from_bytes")
    def test_image_attachment(self, mock_cache):
        from gateway.platforms.email import _extract_attachments
        mock_cache.return_value = "/tmp/cached_img.jpg"

        msg = MIMEMultipart()
        msg.attach(MIMEText("See photo.", "plain", "utf-8"))

        part = MIMEBase("image", "jpeg")
        part.set_payload(b"\xff\xd8\xff\xe0 fake jpg")
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=photo.jpg")
        msg.attach(part)

        result = _extract_attachments(msg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "image")
        mock_cache.assert_called_once()


class TestAuthorizationMaps(unittest.TestCase):
    """Verify email is in authorization maps in gateway/run.py."""

    def test_email_in_adapter_factory(self):
        """Email adapter creation branch should exist."""
        import gateway.run
        import inspect
        source = inspect.getsource(gateway.run.GatewayRunner._create_adapter)
        self.assertIn("Platform.EMAIL", source)

    def test_email_in_allowed_users_map(self):
        """EMAIL_ALLOWED_USERS should be in platform_env_map."""
        import gateway.run
        import inspect
        source = inspect.getsource(gateway.run.GatewayRunner._is_user_authorized)
        self.assertIn("EMAIL_ALLOWED_USERS", source)

    def test_email_in_allow_all_map(self):
        """EMAIL_ALLOW_ALL_USERS should be in platform_allow_all_map."""
        import gateway.run
        import inspect
        source = inspect.getsource(gateway.run.GatewayRunner._is_user_authorized)
        self.assertIn("EMAIL_ALLOW_ALL_USERS", source)


class TestSendMessageToolRouting(unittest.TestCase):
    """Verify email routing in send_message_tool."""

    def test_email_in_platform_map(self):
        import tools.send_message_tool as smt
        import inspect
        source = inspect.getsource(smt._handle_send)
        self.assertIn('"email"', source)

    def test_send_to_platform_has_email_branch(self):
        import tools.send_message_tool as smt
        import inspect
        source = inspect.getsource(smt._send_to_platform)
        self.assertIn("Platform.EMAIL", source)


class TestCronDelivery(unittest.TestCase):
    """Verify email in cron scheduler platform_map."""

    def test_email_in_cron_platform_map(self):
        import cron.scheduler
        import inspect
        source = inspect.getsource(cron.scheduler)
        self.assertIn('"email"', source)


class TestToolset(unittest.TestCase):
    """Verify email toolset is registered."""

    def test_email_toolset_exists(self):
        from toolsets import TOOLSETS
        self.assertIn("hermes-email", TOOLSETS)

    def test_email_in_gateway_toolset(self):
        from toolsets import TOOLSETS
        includes = TOOLSETS["hermes-gateway"]["includes"]
        self.assertIn("hermes-email", includes)


class TestPlatformHints(unittest.TestCase):
    """Verify email platform hint is registered."""

    def test_email_in_platform_hints(self):
        from agent.prompt_builder import PLATFORM_HINTS
        self.assertIn("email", PLATFORM_HINTS)
        self.assertIn("email", PLATFORM_HINTS["email"].lower())


class TestChannelDirectory(unittest.TestCase):
    """Verify email in channel directory session-based discovery."""

    def test_email_in_session_discovery(self):
        import gateway.channel_directory
        import inspect
        source = inspect.getsource(gateway.channel_directory.build_channel_directory)
        self.assertIn('"email"', source)


class TestGatewaySetup(unittest.TestCase):
    """Verify email in gateway setup wizard."""

    def test_email_in_platforms_list(self):
        from hermes_cli.gateway import _PLATFORMS
        keys = [p["key"] for p in _PLATFORMS]
        self.assertIn("email", keys)

    def test_email_has_setup_vars(self):
        from hermes_cli.gateway import _PLATFORMS
        email_platform = next(p for p in _PLATFORMS if p["key"] == "email")
        var_names = [v["name"] for v in email_platform["vars"]]
        self.assertIn("EMAIL_ADDRESS", var_names)
        self.assertIn("EMAIL_PASSWORD", var_names)
        self.assertIn("EMAIL_IMAP_HOST", var_names)
        self.assertIn("EMAIL_SMTP_HOST", var_names)


class TestEnvExample(unittest.TestCase):
    """Verify .env.example has email config."""

    def test_env_example_has_email_vars(self):
        env_path = Path(__file__).resolve().parents[2] / ".env.example"
        content = env_path.read_text()
        self.assertIn("EMAIL_ADDRESS", content)
        self.assertIn("EMAIL_PASSWORD", content)
        self.assertIn("EMAIL_IMAP_HOST", content)
        self.assertIn("EMAIL_SMTP_HOST", content)


class TestDispatchMessage(unittest.TestCase):
    """Test email message dispatch logic."""

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
            from gateway.platforms.email import EmailAdapter
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

        asyncio.get_event_loop().run_until_complete(adapter._dispatch_message(msg_data))
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

        asyncio.get_event_loop().run_until_complete(adapter._dispatch_message(msg_data))
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

        asyncio.get_event_loop().run_until_complete(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertNotIn("[Subject:", captured_events[0].text)
        self.assertEqual(captured_events[0].text, "Thanks for the help!")

    def test_empty_body_handled(self):
        """Email with no body should dispatch '(empty email)'."""
        import asyncio
        adapter = self._make_adapter()
        captured_events = []

        async def capture_handle(event):
            captured_events.append(event)

        adapter.handle_message = capture_handle

        msg_data = {
            "uid": b"4",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Re: test",
            "message_id": "<msg4@test.com>",
            "in_reply_to": "",
            "body": "",
            "attachments": [],
            "date": "",
        }

        asyncio.get_event_loop().run_until_complete(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertIn("(empty email)", captured_events[0].text)

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

        asyncio.get_event_loop().run_until_complete(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertEqual(captured_events[0].message_type, MessageType.PHOTO)
        self.assertEqual(captured_events[0].media_urls, ["/tmp/img.jpg"])

    def test_source_built_correctly(self):
        """Session source should have correct chat_id and user info."""
        import asyncio
        adapter = self._make_adapter()
        captured_events = []

        async def capture_handle(event):
            captured_events.append(event)

        adapter.handle_message = capture_handle

        msg_data = {
            "uid": b"6",
            "sender_addr": "john@example.com",
            "sender_name": "John Doe",
            "subject": "Re: hi",
            "message_id": "<msg6@test.com>",
            "in_reply_to": "",
            "body": "Hello",
            "attachments": [],
            "date": "",
        }

        asyncio.get_event_loop().run_until_complete(adapter._dispatch_message(msg_data))
        event = captured_events[0]
        self.assertEqual(event.source.chat_id, "john@example.com")
        self.assertEqual(event.source.user_id, "john@example.com")
        self.assertEqual(event.source.user_name, "John Doe")
        self.assertEqual(event.source.chat_type, "dm")


class TestThreadContext(unittest.TestCase):
    """Test email reply threading logic."""

    def _make_adapter(self):
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com",
            "EMAIL_SMTP_HOST": "smtp.test.com",
        }):
            from gateway.platforms.email import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter

    def test_thread_context_stored_after_dispatch(self):
        """After dispatching a message, thread context should be stored."""
        import asyncio
        adapter = self._make_adapter()

        async def noop_handle(event):
            pass

        adapter.handle_message = noop_handle

        msg_data = {
            "uid": b"10",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Project question",
            "message_id": "<original@test.com>",
            "in_reply_to": "",
            "body": "Hello",
            "attachments": [],
            "date": "",
        }

        asyncio.get_event_loop().run_until_complete(adapter._dispatch_message(msg_data))
        ctx = adapter._thread_context.get("user@test.com")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["subject"], "Project question")
        self.assertEqual(ctx["message_id"], "<original@test.com>")

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

    def test_reply_does_not_double_re(self):
        """If subject already has Re:, don't add another."""
        adapter = self._make_adapter()
        adapter._thread_context["user@test.com"] = {
            "subject": "Re: Project question",
            "message_id": "<reply@test.com>",
        }

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            adapter._send_email("user@test.com", "Follow up.", None)

            send_call = mock_server.send_message.call_args[0][0]
            self.assertEqual(send_call["Subject"], "Re: Project question")
            self.assertFalse(send_call["Subject"].startswith("Re: Re:"))

    def test_no_thread_context_uses_default_subject(self):
        """Without thread context, subject should be 'Re: Hermes Agent'."""
        adapter = self._make_adapter()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            adapter._send_email("newuser@test.com", "Hello!", None)

            send_call = mock_server.send_message.call_args[0][0]
            self.assertEqual(send_call["Subject"], "Re: Hermes Agent")


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
            from gateway.platforms.email import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter

    def test_send_calls_smtp(self):
        """send() should use SMTP to deliver email."""
        import asyncio
        adapter = self._make_adapter()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            result = asyncio.get_event_loop().run_until_complete(
                adapter.send("user@test.com", "Hello from Hermes!")
            )

            self.assertTrue(result.success)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("hermes@test.com", "secret")
            mock_server.send_message.assert_called_once()
            mock_server.quit.assert_called_once()

    def test_send_failure_returns_error(self):
        """SMTP failure should return SendResult with error."""
        import asyncio
        adapter = self._make_adapter()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.side_effect = Exception("Connection refused")

            result = asyncio.get_event_loop().run_until_complete(
                adapter.send("user@test.com", "Hello")
            )

            self.assertFalse(result.success)
            self.assertIn("Connection refused", result.error)

    def test_send_image_includes_url(self):
        """send_image should include image URL in email body."""
        import asyncio
        from unittest.mock import AsyncMock
        adapter = self._make_adapter()

        adapter.send = AsyncMock(return_value=SendResult(success=True))

        asyncio.get_event_loop().run_until_complete(
            adapter.send_image("user@test.com", "https://img.com/photo.jpg", "My photo")
        )

        call_args = adapter.send.call_args
        body = call_args[0][1]
        self.assertIn("https://img.com/photo.jpg", body)
        self.assertIn("My photo", body)

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

                result = asyncio.get_event_loop().run_until_complete(
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

    def test_send_typing_is_noop(self):
        """send_typing should do nothing for email."""
        import asyncio
        adapter = self._make_adapter()
        # Should not raise
        asyncio.get_event_loop().run_until_complete(adapter.send_typing("user@test.com"))

    def test_get_chat_info(self):
        """get_chat_info should return email address as chat info."""
        import asyncio
        adapter = self._make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Test", "message_id": "<m@t>"}

        info = asyncio.get_event_loop().run_until_complete(
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
            from gateway.platforms.email import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        return adapter

    def test_connect_success(self):
        """Successful IMAP + SMTP connection returns True."""
        import asyncio
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.search.return_value = ("OK", [b"1 2 3"])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            result = asyncio.get_event_loop().run_until_complete(adapter.connect())

            self.assertTrue(result)
            self.assertTrue(adapter._running)
            # Should have skipped existing messages
            self.assertEqual(len(adapter._seen_uids), 3)
            # Cleanup
            adapter._running = False
            if adapter._poll_task:
                adapter._poll_task.cancel()

    def test_connect_imap_failure(self):
        """IMAP connection failure returns False."""
        import asyncio
        adapter = self._make_adapter()

        with patch("imaplib.IMAP4_SSL", side_effect=Exception("IMAP down")):
            result = asyncio.get_event_loop().run_until_complete(adapter.connect())
            self.assertFalse(result)
            self.assertFalse(adapter._running)

    def test_connect_smtp_failure(self):
        """SMTP connection failure returns False."""
        import asyncio
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.search.return_value = ("OK", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("smtplib.SMTP", side_effect=Exception("SMTP down")):
            result = asyncio.get_event_loop().run_until_complete(adapter.connect())
            self.assertFalse(result)

    def test_disconnect_cancels_poll(self):
        """disconnect() should cancel the polling task."""
        import asyncio
        adapter = self._make_adapter()
        adapter._running = True
        adapter._poll_task = asyncio.ensure_future(asyncio.sleep(100))

        asyncio.get_event_loop().run_until_complete(adapter.disconnect())

        self.assertFalse(adapter._running)
        self.assertIsNone(adapter._poll_task)


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
            from gateway.platforms.email import EmailAdapter
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
        mock_imap.search.return_value = ("OK", [b"1 2 3"])
        mock_imap.fetch.return_value = ("OK", [(b"3", raw_email.as_bytes())])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        # Only UID 3 should be fetched (1 and 2 already seen)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sender_addr"], "user@test.com")
        self.assertIn(b"3", adapter._seen_uids)

    def test_fetch_no_unseen_messages(self):
        """No unseen messages returns empty list."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.search.return_value = ("OK", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(results, [])

    def test_fetch_handles_imap_error(self):
        """IMAP errors should be caught and return empty list."""
        adapter = self._make_adapter()

        with patch("imaplib.IMAP4_SSL", side_effect=Exception("Network error")):
            results = adapter._fetch_new_messages()

        self.assertEqual(results, [])

    def test_fetch_extracts_sender_name(self):
        """Sender name should be extracted from 'Name <addr>' format."""
        adapter = self._make_adapter()

        raw_email = MIMEText("Hello", "plain", "utf-8")
        raw_email["From"] = '"John Doe" <john@test.com>'
        raw_email["Subject"] = "Test"
        raw_email["Message-ID"] = "<msg@test.com>"

        mock_imap = MagicMock()
        mock_imap.search.return_value = ("OK", [b"1"])
        mock_imap.fetch.return_value = ("OK", [(b"1", raw_email.as_bytes())])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sender_addr"], "john@test.com")
        self.assertEqual(results[0]["sender_name"], "John Doe")


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
            from gateway.platforms.email import EmailAdapter
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
        mock_imap.search.return_value = ("OK", [b"1"])
        mock_imap.fetch.return_value = ("OK", [(b"1", raw_email.as_bytes())])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            asyncio.get_event_loop().run_until_complete(adapter._check_inbox())

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
        """_send_email should use SMTP to send."""
        import asyncio
        from tools.send_message_tool import _send_email

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            result = asyncio.get_event_loop().run_until_complete(
                _send_email({"address": "hermes@test.com", "smtp_host": "smtp.test.com"}, "user@test.com", "Hello")
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["platform"], "email")

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    })
    def test_send_email_tool_failure(self):
        """SMTP failure should return error dict."""
        import asyncio
        from tools.send_message_tool import _send_email

        with patch("smtplib.SMTP", side_effect=Exception("SMTP error")):
            result = asyncio.get_event_loop().run_until_complete(
                _send_email({"address": "hermes@test.com", "smtp_host": "smtp.test.com"}, "user@test.com", "Hello")
            )

            self.assertIn("error", result)
            self.assertIn("SMTP error", result["error"])

    @patch.dict(os.environ, {}, clear=True)
    def test_send_email_tool_not_configured(self):
        """Missing config should return error."""
        import asyncio
        from tools.send_message_tool import _send_email

        result = asyncio.get_event_loop().run_until_complete(
            _send_email({}, "user@test.com", "Hello")
        )

        self.assertIn("error", result)
        self.assertIn("not configured", result["error"])


if __name__ == "__main__":
    unittest.main()
