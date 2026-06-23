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

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "a@b.com",
    }, clear=True)
    def test_requirements_not_met(self):
        from plugins.platforms.email.adapter import check_email_requirements
        self.assertFalse(check_email_requirements())

    @patch.dict(os.environ, {}, clear=True)
    def test_requirements_empty_env(self):
        from plugins.platforms.email.adapter import check_email_requirements
        self.assertFalse(check_email_requirements())


class TestHelperFunctions(unittest.TestCase):
    """Test email parsing helper functions."""

    def test_decode_header_plain(self):
        from plugins.platforms.email.adapter import _decode_header_value
        self.assertEqual(_decode_header_value("Hello World"), "Hello World")

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

    def test_extract_email_address_bare(self):
        from plugins.platforms.email.adapter import _extract_email_address
        self.assertEqual(
            _extract_email_address("john@example.com"),
            "john@example.com"
        )

    def test_extract_email_address_uppercase(self):
        from plugins.platforms.email.adapter import _extract_email_address
        self.assertEqual(
            _extract_email_address("John@Example.COM"),
            "john@example.com"
        )

    def test_authenticated_sender_prefers_return_path(self):
        from plugins.platforms.email.adapter import _authenticated_sender_address

        msg = MIMEText("hello", "plain", "utf-8")
        msg["From"] = "Allowed User <allowed@example.com>"
        msg["Return-Path"] = "<bounce@example.net>"

        self.assertEqual(_authenticated_sender_address(msg), "bounce@example.net")

    def test_authenticated_sender_falls_back_to_from(self):
        from plugins.platforms.email.adapter import _authenticated_sender_address

        msg = MIMEText("hello", "plain", "utf-8")
        msg["From"] = "Allowed User <allowed@example.com>"

        self.assertEqual(_authenticated_sender_address(msg), "allowed@example.com")

    def test_dmarc_aligned_from_passed(self):
        from plugins.platforms.email.adapter import _dmarc_aligned_from_passed

        msg = MIMEText("hello", "plain", "utf-8")
        msg["From"] = "Allowed User <allowed@example.com>"
        msg["Authentication-Results"] = (
            "mx.example.net; spf=pass smtp.mailfrom=bounces@relay.example.net; "
            "dkim=pass header.d=example.com; dmarc=pass header.from=example.com"
        )

        with patch.dict(os.environ, {"EMAIL_TRUSTED_AUTHSERV_ID": "mx.example.net"}):
            self.assertTrue(_dmarc_aligned_from_passed(msg, "allowed@example.com"))

    def test_dmarc_aligned_from_rejects_wrong_domain(self):
        from plugins.platforms.email.adapter import _dmarc_aligned_from_passed

        msg = MIMEText("hello", "plain", "utf-8")
        msg["From"] = "Allowed User <allowed@example.com>"
        msg["Authentication-Results"] = (
            "mx.example.net; spf=pass smtp.mailfrom=attacker@evil.com; "
            "dkim=pass header.d=evil.com; dmarc=pass header.from=evil.com"
        )

        with patch.dict(os.environ, {"EMAIL_TRUSTED_AUTHSERV_ID": "mx.example.net"}):
            self.assertFalse(_dmarc_aligned_from_passed(msg, "allowed@example.com"))

    def test_dmarc_alignment_requires_trusted_authserv_id(self):
        """A forged topmost Authentication-Results must never satisfy DMARC
        alignment when no trusted authserv-id is configured."""
        from plugins.platforms.email.adapter import _dmarc_aligned_from_passed

        msg = MIMEText("hello", "plain", "utf-8")
        msg["From"] = "Allowed User <allowed@example.com>"
        # Attacker-supplied topmost header claiming a passing aligned result.
        msg["Authentication-Results"] = (
            "attacker-chosen; spf=pass smtp.mailfrom=attacker@evil.com; "
            "dmarc=pass header.from=example.com"
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMAIL_TRUSTED_AUTHSERV_ID", None)
            self.assertFalse(_dmarc_aligned_from_passed(msg, "allowed@example.com"))

    def test_dmarc_alignment_ignores_untrusted_authserv_id(self):
        """An aligned dmarc=pass stamped by an authserv-id we do not trust is
        ignored, even though it is the topmost Authentication-Results header."""
        from plugins.platforms.email.adapter import _dmarc_aligned_from_passed

        msg = MIMEText("hello", "plain", "utf-8")
        msg["From"] = "Allowed User <allowed@example.com>"
        # Forged result from an authserv-id outside the operator's trust anchor.
        msg["Authentication-Results"] = (
            "forged.example; spf=pass smtp.mailfrom=evil.com; "
            "dkim=pass header.d=example.com; dmarc=pass header.from=example.com"
        )

        with patch.dict(os.environ, {"EMAIL_TRUSTED_AUTHSERV_ID": "mx.example.net"}):
            self.assertFalse(_dmarc_aligned_from_passed(msg, "allowed@example.com"))

    def test_strip_html_basic(self):
        from plugins.platforms.email.adapter import _strip_html
        html = "<p>Hello <b>world</b></p>"
        result = _strip_html(html)
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertNotIn("<p>", result)
        self.assertNotIn("<b>", result)

    def test_strip_html_br_tags(self):
        from plugins.platforms.email.adapter import _strip_html
        html = "Line 1<br>Line 2<br/>Line 3"
        result = _strip_html(html)
        self.assertIn("Line 1", result)
        self.assertIn("Line 2", result)

    def test_strip_html_entities(self):
        from plugins.platforms.email.adapter import _strip_html
        html = "a &amp; b &lt; c &gt; d"
        result = _strip_html(html)
        self.assertIn("a & b", result)


class TestExtractTextBody(unittest.TestCase):
    """Test email body extraction from different message formats."""

    def test_plain_text_body(self):
        from plugins.platforms.email.adapter import _extract_text_body
        msg = MIMEText("Hello, this is a test.", "plain", "utf-8")
        result = _extract_text_body(msg)
        self.assertEqual(result, "Hello, this is a test.")

    def test_html_body_fallback(self):
        from plugins.platforms.email.adapter import _extract_text_body
        msg = MIMEText("<p>Hello from HTML</p>", "html", "utf-8")
        result = _extract_text_body(msg)
        self.assertIn("Hello from HTML", result)
        self.assertNotIn("<p>", result)

    def test_multipart_prefers_plain(self):
        from plugins.platforms.email.adapter import _extract_text_body
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("<p>HTML version</p>", "html", "utf-8"))
        msg.attach(MIMEText("Plain version", "plain", "utf-8"))
        result = _extract_text_body(msg)
        self.assertEqual(result, "Plain version")

    def test_multipart_html_only(self):
        from plugins.platforms.email.adapter import _extract_text_body
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("<p>Only HTML</p>", "html", "utf-8"))
        result = _extract_text_body(msg)
        self.assertIn("Only HTML", result)

    def test_empty_body(self):
        from plugins.platforms.email.adapter import _extract_text_body
        msg = MIMEText("", "plain", "utf-8")
        result = _extract_text_body(msg)
        self.assertEqual(result, "")


class TestExtractAttachments(unittest.TestCase):
    """Test attachment extraction and caching."""

    def test_no_attachments(self):
        from plugins.platforms.email.adapter import _extract_attachments
        msg = MIMEText("No attachments here.", "plain", "utf-8")
        result = _extract_attachments(msg)
        self.assertEqual(result, [])

    @patch("plugins.platforms.email.adapter.cache_document_from_bytes")
    def test_document_attachment(self, mock_cache):
        from plugins.platforms.email.adapter import _extract_attachments
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

    @patch("plugins.platforms.email.adapter.cache_image_from_bytes")
    def test_image_attachment(self, mock_cache):
        from plugins.platforms.email.adapter import _extract_attachments
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

        asyncio.run(adapter._dispatch_message(msg_data))
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

        asyncio.run(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertEqual(captured_events[0].message_type, MessageType.PHOTO)
        self.assertEqual(captured_events[0].media_urls, ["/tmp/img.jpg"])

    def test_document_attachment_sets_document_type(self):
        """Email with a document attachment must set DOCUMENT so run.py injects file context."""
        import asyncio
        from gateway.platforms.base import MessageType
        adapter = self._make_adapter()
        captured_events = []

        async def capture_handle(event):
            captured_events.append(event)

        adapter.handle_message = capture_handle

        msg_data = {
            "uid": b"6",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Re: report",
            "message_id": "<msg6@test.com>",
            "in_reply_to": "",
            "body": "See attached",
            "attachments": [{"path": "/tmp/report.pdf", "filename": "report.pdf", "type": "document", "media_type": "application/pdf"}],
            "date": "",
        }

        asyncio.run(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertEqual(captured_events[0].message_type, MessageType.DOCUMENT)
        self.assertEqual(captured_events[0].media_urls, ["/tmp/report.pdf"])

    def test_mixed_image_and_document_prefers_document(self):
        """DOCUMENT wins for mixed attachments — image handling keys off per-path
        mime types, but document injection gates strictly on MessageType.DOCUMENT."""
        import asyncio
        from gateway.platforms.base import MessageType
        adapter = self._make_adapter()
        captured_events = []

        async def capture_handle(event):
            captured_events.append(event)

        adapter.handle_message = capture_handle

        msg_data = {
            "uid": b"7",
            "sender_addr": "user@test.com",
            "sender_name": "User",
            "subject": "Re: both",
            "message_id": "<msg7@test.com>",
            "in_reply_to": "",
            "body": "Photo and PDF",
            "attachments": [
                {"path": "/tmp/img.jpg", "filename": "img.jpg", "type": "image", "media_type": "image/jpeg"},
                {"path": "/tmp/report.pdf", "filename": "report.pdf", "type": "document", "media_type": "application/pdf"},
            ],
            "date": "",
        }

        asyncio.run(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertEqual(captured_events[0].message_type, MessageType.DOCUMENT)
        self.assertEqual(len(captured_events[0].media_urls), 2)

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

        asyncio.run(adapter._dispatch_message(msg_data))
        event = captured_events[0]
        self.assertEqual(event.source.chat_id, "john@example.com")
        self.assertEqual(event.source.user_id, "john@example.com")
        self.assertEqual(event.source.user_name, "John Doe")
        self.assertEqual(event.source.chat_type, "dm")

    def test_non_allowlisted_sender_dropped(self):
        """Senders not in EMAIL_ALLOWED_USERS should be dropped before dispatch."""
        import asyncio
        with patch.dict(os.environ, {
            "EMAIL_ALLOWED_USERS": "hermes@test.com,admin@test.com",
        }):
            adapter = self._make_adapter()
            adapter._message_handler = MagicMock()

            msg_data = {
                "uid": b"99",
                "sender_addr": "outsider@evil.com",
                "sender_name": "Spammer",
                "subject": "Buy now!!!",
                "message_id": "<spam@evil.com>",
                "in_reply_to": "",
                "body": "Cheap meds",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))
            # Handler should NOT be called for non-allowlisted sender
            adapter._message_handler.assert_not_called()
            # Thread context should NOT be created
            self.assertNotIn("outsider@evil.com", adapter._thread_context)

    def test_allowlisted_sender_proceeds(self):
        """Senders in EMAIL_ALLOWED_USERS should proceed to dispatch normally."""
        import asyncio
        with patch.dict(os.environ, {
            "EMAIL_ALLOWED_USERS": "hermes@test.com,admin@test.com",
        }):
            adapter = self._make_adapter()
            captured_events = []

            async def mock_handler(event):
                captured_events.append(event)
                return None

            adapter._message_handler = mock_handler

            msg_data = {
                "uid": b"100",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Important",
                "message_id": "<msg@test.com>",
                "in_reply_to": "",
                "body": "Hello",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))
            self.assertEqual(len(captured_events), 1)
            self.assertEqual(captured_events[0].source.chat_id, "admin@test.com")

    def test_allowlist_uses_authenticated_sender_not_spoofed_from(self):
        """A spoofed From must not pass allowlist when Return-Path differs."""
        import asyncio
        with patch.dict(os.environ, {
            "EMAIL_ALLOWED_USERS": "admin@test.com",
        }):
            adapter = self._make_adapter()
            adapter.handle_message = AsyncMock()

            msg_data = {
                "uid": b"102",
                "sender_addr": "admin@test.com",
                "auth_sender_addr": "attacker@evil.com",
                "sender_name": "Admin",
                "subject": "Spoof",
                "message_id": "<spoof@evil.com>",
                "in_reply_to": "",
                "body": "run this",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))
            adapter.handle_message.assert_not_called()
            self.assertNotIn("admin@test.com", adapter._thread_context)

    def test_allowlist_accepts_dmarc_aligned_relay_sender(self):
        """Relay Return-Path can differ when the visible From passed DMARC."""
        import asyncio
        with patch.dict(os.environ, {
            "EMAIL_ALLOWED_USERS": "admin@test.com",
        }):
            adapter = self._make_adapter()
            captured_events = []

            async def capture_handle(event):
                captured_events.append(event)

            adapter.handle_message = capture_handle

            msg_data = {
                "uid": b"104",
                "sender_addr": "admin@test.com",
                "auth_sender_addr": "bounces@relay.example.net",
                "auth_from_aligned": True,
                "sender_name": "Admin",
                "subject": "Via relay",
                "message_id": "<relay@test.com>",
                "in_reply_to": "",
                "body": "hello",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))
            self.assertEqual(len(captured_events), 1)
            self.assertEqual(captured_events[0].source.user_id, "admin@test.com")

    def test_allowlist_rejects_dmarc_aligned_non_allowlisted_from(self):
        """DMARC alignment does not grant access to non-allowlisted senders."""
        import asyncio
        with patch.dict(os.environ, {
            "EMAIL_ALLOWED_USERS": "admin@test.com",
        }):
            adapter = self._make_adapter()
            adapter.handle_message = AsyncMock()

            msg_data = {
                "uid": b"105",
                "sender_addr": "outsider@example.com",
                "auth_sender_addr": "bounces@relay.example.net",
                "auth_from_aligned": True,
                "sender_name": "Outsider",
                "subject": "Via relay",
                "message_id": "<relay-outsider@test.com>",
                "in_reply_to": "",
                "body": "hello",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))
            adapter.handle_message.assert_not_called()
            self.assertNotIn("outsider@example.com", adapter._thread_context)

    def test_allowlist_accepts_matching_authenticated_sender(self):
        """Matching From and Return-Path preserve the normal allowlist path."""
        import asyncio
        with patch.dict(os.environ, {
            "EMAIL_ALLOWED_USERS": "admin@test.com",
        }):
            adapter = self._make_adapter()
            captured_events = []

            async def capture_handle(event):
                captured_events.append(event)

            adapter.handle_message = capture_handle

            msg_data = {
                "uid": b"103",
                "sender_addr": "admin@test.com",
                "auth_sender_addr": "admin@test.com",
                "auth_from_aligned": False,
                "sender_name": "Admin",
                "subject": "Re: normal",
                "message_id": "<normal@test.com>",
                "in_reply_to": "",
                "body": "hello",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))
            self.assertEqual(len(captured_events), 1)
            self.assertEqual(captured_events[0].source.user_id, "admin@test.com")

    def test_empty_allowlist_allows_all(self):
        """When EMAIL_ALLOWED_USERS is not set, all senders should proceed."""
        import asyncio
        with patch.dict(os.environ, {}, clear=False):
            # Ensure EMAIL_ALLOWED_USERS is not in the env
            if "EMAIL_ALLOWED_USERS" in os.environ:
                del os.environ["EMAIL_ALLOWED_USERS"]

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
            # Handler should be called when no allowlist is configured
            adapter._message_handler.assert_called()


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
            from plugins.platforms.email.adapter import EmailAdapter
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

        asyncio.run(adapter._dispatch_message(msg_data))
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
            self.assertIn("Date", send_call)

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

    def test_send_calls_smtp(self):
        """send() should use SMTP to deliver email."""
        import asyncio
        adapter = self._make_adapter()

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            result = asyncio.run(
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

            result = asyncio.run(
                adapter.send("user@test.com", "Hello")
            )

            self.assertFalse(result.success)
            self.assertIn("Connection refused", result.error)

    def test_send_image_includes_url(self):
        """send_image should include image URL in email body."""
        import asyncio
        adapter = self._make_adapter()

        adapter.send = AsyncMock(return_value=SendResult(success=True))

        asyncio.run(
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

    def test_send_typing_is_noop(self):
        """send_typing should do nothing for email."""
        import asyncio
        adapter = self._make_adapter()
        # Should not raise
        asyncio.run(adapter.send_typing("user@test.com"))

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

    def test_connect_imap_failure(self):
        """IMAP connection failure returns False."""
        import asyncio
        adapter = self._make_adapter()

        with patch("imaplib.IMAP4_SSL", side_effect=Exception("IMAP down")):
            result = asyncio.run(adapter.connect())
            self.assertFalse(result)
            self.assertFalse(adapter._running)

    def test_connect_smtp_failure(self):
        """SMTP connection failure returns False."""
        import asyncio
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("smtplib.SMTP", side_effect=Exception("SMTP down")):
            result = asyncio.run(adapter.connect())
            self.assertFalse(result)

    def test_disconnect_cancels_poll(self):
        """disconnect() should cancel the polling task."""
        import asyncio
        adapter = self._make_adapter()
        adapter._running = True

        async def _exercise_disconnect():
            adapter._poll_task = asyncio.create_task(asyncio.sleep(100))
            await adapter.disconnect()

        asyncio.run(_exercise_disconnect())

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

    def test_fetch_no_unseen_messages(self):
        """No unseen messages returns empty list."""
        adapter = self._make_adapter()

        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b""])

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

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

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

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    })
    def test_send_email_tool_failure(self):
        """SMTP failure should return error dict."""
        import asyncio
        from plugins.platforms.email.adapter import _standalone_send as _email_send
        from types import SimpleNamespace
        async def _send_email(extra, chat_id, message):
            return await _email_send(SimpleNamespace(token=None, api_key=None, extra=extra or {}), chat_id, message)

        with patch("smtplib.SMTP", side_effect=Exception("SMTP error")):
            result = asyncio.run(
                _send_email({"address": "hermes@test.com", "smtp_host": "smtp.test.com"}, "user@test.com", "Hello")
            )

            self.assertIn("error", result)
            self.assertIn("SMTP error", result["error"])

    @patch.dict(os.environ, {}, clear=True)
    def test_send_email_tool_not_configured(self):
        """Missing config should return error."""
        import asyncio
        from plugins.platforms.email.adapter import _standalone_send as _email_send
        from types import SimpleNamespace
        async def _send_email(extra, chat_id, message):
            return await _email_send(SimpleNamespace(token=None, api_key=None, extra=extra or {}), chat_id, message)

        result = asyncio.run(
            _send_email({}, "user@test.com", "Hello")
        )

        self.assertIn("error", result)
        self.assertIn("not configured", result["error"])


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
    def test_smtp_quit_called_on_send_message_failure(self):
        """SMTP quit() must be called even when send_message() raises."""
        adapter = self._make_adapter()
        mock_smtp = MagicMock()
        mock_smtp.send_message.side_effect = Exception("send failed")

        with patch("smtplib.SMTP", return_value=mock_smtp):
            with self.assertRaises(Exception):
                adapter._send_email("user@test.com", "Hello")

        mock_smtp.quit.assert_called_once()

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

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_IMAP_PORT": "993",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }, clear=False)
    def test_imap_logout_called_on_early_return(self):
        """IMAP logout() must be called even when returning early (no unseen)."""
        adapter = self._make_adapter()
        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b""])

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

    def test_fetch_new_messages_sends_imap_id_after_login(self):
        """_fetch_new_messages must also send ID — it opens its own IMAP session."""
        adapter = self._make_adapter()
        mock_imap = MagicMock()
        mock_imap.uid.return_value = ("OK", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            adapter._fetch_new_messages()

        id_calls = [c for c in mock_imap.xatom.call_args_list if c.args and c.args[0] == "ID"]
        self.assertTrue(
            id_calls,
            "_fetch_new_messages() must call imap.xatom('ID', ...) after "
            "LOGIN — the polling path opens a fresh IMAP connection.",
        )

    def test_send_imap_id_swallows_errors_for_non_supporting_servers(self):
        """Servers that reject ID must not break the connection."""
        from plugins.platforms.email.adapter import _send_imap_id

        mock_imap = MagicMock()
        mock_imap.xatom.side_effect = Exception("BAD command unknown: ID")

        _send_imap_id(mock_imap)
        mock_imap.xatom.assert_called_once()


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

    def test_port_587_uses_smtp_with_starttls(self):
        """Port 587 should use smtplib.SMTP + STARTTLS."""
        adapter = self._make_adapter("587")

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("smtplib.SMTP_SSL") as mock_smtp_ssl:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server

            result = adapter._connect_smtp()

            mock_smtp.assert_called_once()
            mock_smtp_ssl.assert_not_called()
            mock_server.starttls.assert_called_once()
            self.assertIs(result, mock_server)

    def test_port_465_uses_smtp_ssl(self):
        """Port 465 should use smtplib.SMTP_SSL (implicit TLS)."""
        adapter = self._make_adapter("465")

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("smtplib.SMTP_SSL") as mock_smtp_ssl:
            mock_server = MagicMock()
            mock_smtp_ssl.return_value = mock_server

            result = adapter._connect_smtp()

            mock_smtp_ssl.assert_called_once()
            mock_smtp.assert_not_called()
            self.assertIs(result, mock_server)

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

    def test_tls_verification_error_does_not_retry_ipv4(self):
        """Certificate failures are security errors, not IPv6 reachability failures."""
        import ssl as _ssl
        import plugins.platforms.email.adapter as email_mod

        adapter = self._make_adapter("465")

        with patch("smtplib.SMTP_SSL", side_effect=_ssl.SSLError("cert verify failed")), \
             patch.object(email_mod, "_IPv4SMTP_SSL") as mock_ipv4_smtp_ssl:
            with self.assertRaises(_ssl.SSLError):
                adapter._connect_smtp()

            mock_ipv4_smtp_ssl.assert_not_called()

    def test_ipv4_connection_does_not_mutate_global_resolver(self):
        """IPv4 fallback must not monkeypatch process-global socket state."""
        import socket as _socket
        from plugins.platforms.email.adapter import _create_ipv4_connection

        original_getaddrinfo = _socket.getaddrinfo
        fake_sock = MagicMock()

        with patch(
            "socket.getaddrinfo",
            return_value=[(_socket.AF_INET, _socket.SOCK_STREAM, 6, "", ("192.0.2.1", 587))],
        ) as mock_getaddrinfo, patch("socket.socket", return_value=fake_sock):
            result = _create_ipv4_connection("smtp.test.com", 587, 30)

        self.assertIs(result, fake_sock)
        mock_getaddrinfo.assert_called_once_with(
            "smtp.test.com", 587, _socket.AF_INET, _socket.SOCK_STREAM,
        )
        self.assertIs(_socket.getaddrinfo, original_getaddrinfo)


class TestConnectionConfigResolution(unittest.TestCase):
    """Host/address resolution and pre-connect validation (#49736)."""

    def test_host_and_address_whitespace_stripped(self):
        """A stray space/newline must not reach IMAP4_SSL as part of the host.

        Whitespace in the host produced the misleading
        ``[Errno 8] nodename nor servname`` (unresolvable name) instead of a
        successful connection.
        """
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "  hermes@test.com\n",
            "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": " imap.test.com ",
            "EMAIL_SMTP_HOST": "smtp.test.com\n",
        }, clear=False):
            adapter = EmailAdapter(PlatformConfig(enabled=True))
        self.assertEqual(adapter._imap_host, "imap.test.com")
        self.assertEqual(adapter._smtp_host, "smtp.test.com")
        self.assertEqual(adapter._address, "hermes@test.com")

    def test_falls_back_to_platform_config_extra(self):
        """When env vars are absent, settings come from PlatformConfig.extra —
        the same dict gateway.config populates and `hermes config show` reads."""
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import EmailAdapter
        cfg = PlatformConfig(enabled=True)
        cfg.extra.update({
            "address": "hermes@test.com",
            "imap_host": "imap.test.com",
            "smtp_host": "smtp.test.com",
        })
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "", "EMAIL_IMAP_HOST": "", "EMAIL_SMTP_HOST": "",
            "EMAIL_PASSWORD": "secret",
        }, clear=False):
            adapter = EmailAdapter(cfg)
        self.assertEqual(adapter._imap_host, "imap.test.com")
        self.assertEqual(adapter._smtp_host, "smtp.test.com")
        self.assertEqual(adapter._address, "hermes@test.com")

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

    def test_all_settings_present_satisfies_requirements(self):
        """The connected check passes only when all four settings are non-blank."""
        from plugins.platforms.email.adapter import check_email_requirements
        with patch.dict(os.environ, {
            "EMAIL_ADDRESS": "hermes@test.com", "EMAIL_PASSWORD": "secret",
            "EMAIL_IMAP_HOST": "imap.test.com", "EMAIL_SMTP_HOST": "smtp.test.com",
        }, clear=False):
            self.assertTrue(check_email_requirements())


if __name__ == "__main__":
    unittest.main()
