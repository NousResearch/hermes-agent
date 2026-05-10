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
import json
import re
import stat
import tempfile
import time
import unittest
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

from gateway.platforms.base import SendResult


def _write_cached_test_attachment(kind, filename, data=b"cached test attachment"):
    from hermes_constants import get_hermes_home
    from gateway.platforms import base

    # base.py is imported at module import time for SendResult, before the
    # test autouse fixture redirects HERMES_HOME. Rebind the cache roots here
    # so cleanup tests do not write to the developer's real ~/.hermes cache.
    base.DOCUMENT_CACHE_DIR = get_hermes_home() / "cache" / "documents"
    base.IMAGE_CACHE_DIR = get_hermes_home() / "cache" / "images"

    cache_dir = {
        "document": base.get_document_cache_dir,
        "image": base.get_image_cache_dir,
    }[kind]()
    path = cache_dir / filename
    path.write_bytes(data)
    return path


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

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_AUTH_MODE": "challenge",
        "EMAIL_CHALLENGE_TTL_SECONDS": "300",
        "EMAIL_CHALLENGE_STORE": "/tmp/email_challenges.json",
    }, clear=False)
    def test_email_challenge_config_loaded_from_env(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)
        extra = config.platforms[Platform.EMAIL].extra
        self.assertEqual(extra["auth_mode"], "challenge")
        self.assertEqual(extra["challenge_ttl_seconds"], "300")
        self.assertEqual(extra["challenge_store"], "/tmp/email_challenges.json")

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

    def test_total_attachment_cap_deletes_previously_cached_files(self):
        """If a later attachment exceeds the total cap, earlier cached files are removed."""
        from gateway.platforms import base
        from gateway.platforms.email import _extract_attachments

        with tempfile.TemporaryDirectory() as tmpdir:
            base.DOCUMENT_CACHE_DIR = Path(tmpdir) / "cache" / "documents"
            msg = MIMEMultipart()
            msg.attach(MIMEText("See attached.", "plain", "utf-8"))

            first = MIMEBase("application", "pdf")
            first.set_payload(b"small")
            encoders.encode_base64(first)
            first.add_header("Content-Disposition", "attachment; filename=first.pdf")
            msg.attach(first)

            second = MIMEBase("application", "pdf")
            second.set_payload(b"large payload")
            encoders.encode_base64(second)
            second.add_header("Content-Disposition", "attachment; filename=second.pdf")
            msg.attach(second)

            with self.assertRaises(ValueError):
                _extract_attachments(
                    msg,
                    max_attachment_bytes=100,
                    max_total_attachment_bytes=10,
                )

            cache_dir = base.get_document_cache_dir()
            self.assertEqual(list(cache_dir.iterdir()), [])

    def test_attachment_count_cap_deletes_previously_cached_files(self):
        """If a later attachment exceeds the count cap, earlier cached files are removed."""
        from gateway.platforms import base
        from gateway.platforms.email import _extract_attachments

        with tempfile.TemporaryDirectory() as tmpdir:
            base.DOCUMENT_CACHE_DIR = Path(tmpdir) / "cache" / "documents"
            msg = MIMEMultipart()
            msg.attach(MIMEText("See attached.", "plain", "utf-8"))
            for name in ("first.pdf", "second.pdf"):
                part = MIMEBase("application", "pdf")
                part.set_payload(b"small")
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={name}")
                msg.attach(part)

            with self.assertRaises(ValueError):
                _extract_attachments(msg, max_attachments=1)

            cache_dir = base.get_document_cache_dir()
            self.assertEqual(list(cache_dir.iterdir()), [])

    def test_cache_oserror_deletes_previously_cached_files(self):
        """If a later cache write fails, earlier cached files are removed."""
        from gateway.platforms import base
        from gateway.platforms.email import _extract_attachments

        with tempfile.TemporaryDirectory() as tmpdir:
            base.DOCUMENT_CACHE_DIR = Path(tmpdir) / "cache" / "documents"
            cached_first = base.get_document_cache_dir() / "doc_first.pdf"
            msg = MIMEMultipart()
            msg.attach(MIMEText("See attached.", "plain", "utf-8"))
            for name in ("first.pdf", "second.pdf"):
                part = MIMEBase("application", "pdf")
                part.set_payload(b"small")
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={name}")
                msg.attach(part)

            def cache_document(payload, filename):
                if filename == "first.pdf":
                    cached_first.write_bytes(payload)
                    return str(cached_first)
                raise OSError("disk full")

            with patch("gateway.platforms.email.cache_document_from_bytes", side_effect=cache_document):
                with self.assertRaises(OSError):
                    _extract_attachments(msg)

            self.assertFalse(cached_first.exists())


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

    def _extract_confirmation_code(self, sent_body):
        match = re.search(r"/confirm\s+(\S+)", sent_body)
        self.assertIsNotNone(match)
        return match.group(1)

    def test_email_challenge_store_is_cached_per_adapter_instance(self):
        """Challenge store path/TTL are resolved once for each adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            first_store = Path(tmpdir) / "first.json"
            second_store = Path(tmpdir) / "second.json"
            with patch.dict(os.environ, {
                "EMAIL_CHALLENGE_STORE": str(first_store),
                "EMAIL_CHALLENGE_TTL_SECONDS": "123",
            }, clear=False):
                adapter = self._make_adapter()

            first = adapter._email_challenge_store()
            with patch.dict(os.environ, {
                "EMAIL_CHALLENGE_STORE": str(second_store),
                "EMAIL_CHALLENGE_TTL_SECONDS": "456",
            }, clear=False):
                second = adapter._email_challenge_store()

            self.assertIs(first, second)
            self.assertEqual(first.path, first_store)
            self.assertEqual(first.ttl_seconds, 123)

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

    def test_self_message_drop_deletes_cached_attachment(self):
        """Self-sent messages must not leave challenge-owned cached files behind."""
        import asyncio

        attachment_path = _write_cached_test_attachment(
            "document", "doc_self-sent-drop.pdf", b"self-sent attachment"
        )
        adapter = self._make_adapter()
        adapter._message_handler = MagicMock()

        msg_data = {
            "uid": b"1",
            "sender_addr": "hermes@test.com",
            "sender_name": "Hermes",
            "subject": "Self attachment",
            "message_id": "<self-attachment@test.com>",
            "in_reply_to": "",
            "body": "Self message",
            "attachments": [{
                "type": "document",
                "media_type": "application/pdf",
                "path": str(attachment_path),
            }],
            "date": "",
        }

        asyncio.run(adapter._dispatch_message(msg_data))

        adapter._message_handler.assert_not_called()
        self.assertFalse(attachment_path.exists())

    def test_automated_sender_drop_deletes_cached_attachment(self):
        """Automated dispatch drops must delete cached files before returning."""
        import asyncio

        attachment_path = _write_cached_test_attachment(
            "document", "doc_automated-drop.pdf", b"automated attachment"
        )
        adapter = self._make_adapter()
        adapter._message_handler = MagicMock()
        adapter.send = AsyncMock()

        msg_data = {
            "uid": b"2",
            "sender_addr": "no-reply@test.com",
            "sender_name": "No Reply",
            "subject": "Automated attachment",
            "message_id": "<automated-attachment@test.com>",
            "in_reply_to": "",
            "body": "Automated message",
            "attachments": [{
                "type": "document",
                "media_type": "application/pdf",
                "path": str(attachment_path),
            }],
            "date": "",
        }

        asyncio.run(adapter._dispatch_message(msg_data))

        adapter._message_handler.assert_not_called()
        adapter.send.assert_not_called()
        self.assertFalse(attachment_path.exists())

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
        with patch.dict(os.environ, {"GATEWAY_ALLOW_ALL_USERS": "true"}, clear=False):
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
                "attachments": [
                    {"path": "/tmp/img.jpg", "filename": "img.jpg", "type": "image", "media_type": "image/jpeg"}
                ],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))
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

    def test_direct_mode_star_allowlist_is_literal_not_wildcard(self):
        """EMAIL_ALLOWED_USERS=* should preserve direct-mode literal matching behavior."""
        import asyncio
        with patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "direct",
            "EMAIL_ALLOWED_USERS": "*",
        }):
            adapter = self._make_adapter()
            adapter._message_handler = MagicMock()

            msg_data = {
                "uid": b"98",
                "sender_addr": "outsider@test.com",
                "sender_name": "Outsider",
                "subject": "Literal star",
                "message_id": "<literal-star@test.com>",
                "in_reply_to": "",
                "body": "This should not pass direct adapter allowlist",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter._message_handler.assert_not_called()
            self.assertNotIn("outsider@test.com", adapter._thread_context)

    def test_direct_mode_global_allowlist_miss_drops_cached_attachment_before_central_auth(self):
        """Direct-mode global misses may reach central auth but must not retain cached attachments."""
        import asyncio

        with patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "direct",
            "EMAIL_ALLOWED_USERS": "",
            "GATEWAY_ALLOWED_USERS": "admin@test.com",
        }):
            attachment_path = _write_cached_test_attachment(
                "document", "doc_direct-global-rejected.pdf", b"direct global rejected"
            )
            adapter = self._make_adapter()
            captured_events = []

            async def capture_handle(event):
                captured_events.append(event)

            adapter.handle_message = capture_handle

            msg_data = {
                "uid": b"97",
                "sender_addr": "outsider@test.com",
                "sender_name": "Outsider",
                "subject": "Global allowlist miss",
                "message_id": "<direct-global-miss@test.com>",
                "in_reply_to": "",
                "body": "This should not reach the agent",
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(attachment_path),
                }],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            self.assertEqual(len(captured_events), 1)
            self.assertEqual(captured_events[0].source.user_id, "outsider@test.com")
            self.assertEqual(captured_events[0].media_urls, [])
            self.assertIn("outsider@test.com", adapter._thread_context)
            self.assertFalse(attachment_path.exists())

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

    def test_challenge_mode_allowlisted_sender_is_challenged_not_dispatched(self):
        """Challenge mode stores allowlisted email bodies without invoking the agent."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
            "EMAIL_CHALLENGE_TTL_SECONDS": "900",
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = AsyncMock()

            msg_data = {
                "uid": b"200",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Run report",
                "message_id": "<challenge@test.com>",
                "in_reply_to": "",
                "body": "Please summarize revenue.",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_awaited_once()
            sent_body = adapter.send.await_args.args[1]
            self.assertIn("pending confirmation", sent_body)
            self.assertIn("/confirm ", sent_body)
            self.assertIn("Run report", sent_body)
            self.assertNotIn("Please summarize revenue.", sent_body)

            store_path = Path(os.environ["EMAIL_CHALLENGE_STORE"])
            stored = json.loads(store_path.read_text())
            self.assertEqual(len(stored["challenges"]), 1)
            self.assertEqual(stored["challenges"][0]["sender"], "admin@test.com")
            self.assertEqual(stored["challenges"][0]["event"]["body"], "Please summarize revenue.")
            self.assertNotIn("body", stored["challenges"][0])
            self.assertNotIn(self._extract_confirmation_code(sent_body), store_path.read_text())
            self.assertEqual(stat.S_IMODE(store_path.stat().st_mode), 0o600)

    def test_challenge_confirm_dispatches_original_once_and_replay_is_rejected(self):
        """A valid confirmation dispatches the stored original exactly once."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
            "EMAIL_CHALLENGE_TTL_SECONDS": "900",
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            captured_events = []

            async def capture_handle(event):
                captured_events.append(event)

            adapter.handle_message = capture_handle
            original = {
                "uid": b"201",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Run report",
                "message_id": "<original@test.com>",
                "in_reply_to": "",
                "body": "Please summarize revenue.",
                "attachments": [],
                "date": "",
            }
            asyncio.run(adapter._dispatch_message(original))
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])

            confirm = {
                "uid": b"202",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Re: Run report",
                "message_id": "<confirm@test.com>",
                "in_reply_to": "<challenge-reply@test.com>",
                "body": f"/confirm {code}",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(confirm))
            asyncio.run(adapter._dispatch_message(confirm))

            self.assertEqual(len(captured_events), 1)
            self.assertIn("[Subject: Run report]", captured_events[0].text)
            self.assertIn("Please summarize revenue.", captured_events[0].text)
            self.assertEqual(captured_events[0].message_id, "<original@test.com>")
            self.assertGreaterEqual(adapter.send.await_count, 2)
            replay_response = adapter.send.await_args.args[1]
            self.assertIn("invalid or already used", replay_response.lower())

    def test_challenge_confirmed_email_allowed_user_passes_real_gateway_auth(self):
        """Confirmed originals flow through GatewayRunner's central authorization gate."""
        import asyncio
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import GatewayRunner

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))

            runner = object.__new__(GatewayRunner)
            runner.config = GatewayConfig(platforms={Platform.EMAIL: PlatformConfig(enabled=True)})
            runner.adapters = {Platform.EMAIL: adapter}
            runner.pairing_store = MagicMock()
            runner.pairing_store.is_approved.return_value = False
            runner.pairing_store._is_rate_limited.return_value = False
            runner._running_agents = {}
            runner._running_agents_ts = {}
            runner._update_prompt_pending = {}
            runner._draining = False
            runner._busy_input_mode = "interrupt"
            runner._handle_message_with_agent = AsyncMock(return_value="agent accepted")
            adapter.handle_message = runner._handle_message

            original = {
                "uid": b"234",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Central auth",
                "message_id": "<central-auth@test.com>",
                "in_reply_to": "",
                "body": "This should reach the gateway after confirmation.",
                "attachments": [],
                "date": "",
            }
            asyncio.run(adapter._dispatch_message(original))
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])

            asyncio.run(adapter._dispatch_message(dict(
                original,
                uid=b"235",
                subject="Re: Central auth",
                message_id="<confirm-central-auth@test.com>",
                body=f"/confirm {code}",
            )))

            runner.pairing_store.is_approved.assert_called()
            runner._handle_message_with_agent.assert_awaited_once()
            event = runner._handle_message_with_agent.await_args.args[0]
            self.assertEqual(event.source.platform, Platform.EMAIL)
            self.assertEqual(event.source.user_id, "admin@test.com")
            self.assertIn("This should reach the gateway", event.text)

    def test_challenge_confirm_from_different_sender_does_not_dispatch(self):
        """Confirmation codes are bound to the original From sender."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com,other@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            attachment_path = Path(tmpdir) / "cache" / "documents" / "doc_sensitive-expired.pdf"
            attachment_path.parent.mkdir(parents=True)
            attachment_path.write_bytes(b"sensitive document")
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()
            original = {
                "uid": b"203",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Sensitive",
                "message_id": "<sensitive@test.com>",
                "in_reply_to": "",
                "body": "Secret task",
                "attachments": [],
                "date": "",
            }
            asyncio.run(adapter._dispatch_message(original))
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])

            confirm = dict(
                original,
                uid=b"204",
                sender_addr="other@test.com",
                sender_name="Other",
                body=f"/confirm {code}",
            )
            asyncio.run(adapter._dispatch_message(confirm))

            adapter.handle_message.assert_not_awaited()
            self.assertIn("invalid or already used", adapter.send.await_args.args[1].lower())
            self.assertNotIn("does not match", adapter.send.await_args.args[1].lower())

    def test_expired_challenge_confirm_does_not_dispatch(self):
        """Expired challenge codes are rejected without invoking the agent."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
            "EMAIL_CHALLENGE_TTL_SECONDS": "1",
        }):
            attachment_path = Path(tmpdir) / "cache" / "documents" / "doc_sensitive-expired.pdf"
            attachment_path.parent.mkdir(parents=True)
            attachment_path.write_bytes(b"sensitive document")
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()
            original = {
                "uid": b"205",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Old task",
                "message_id": "<old@test.com>",
                "in_reply_to": "",
                "body": "Do old thing",
                "attachments": [],
                "date": "",
            }
            asyncio.run(adapter._dispatch_message(original))
            store_path = Path(os.environ["EMAIL_CHALLENGE_STORE"])
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])
            stored = json.loads(store_path.read_text())
            stored["challenges"][0]["created_at"] = 0
            store_path.write_text(json.dumps(stored))

            confirm = dict(
                original,
                uid=b"206",
                subject="Re: Old task",
                message_id="<confirm-old@test.com>",
                body=f"/confirm {code}",
            )
            asyncio.run(adapter._dispatch_message(confirm))

            adapter.handle_message.assert_not_awaited()
            self.assertIn("expired", adapter.send.await_args.args[1].lower())

    def test_challenge_confirm_accepts_quoted_reply_body(self):
        """Confirmation parsing accepts normal replies with quoted original text."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            attachment_path = Path(tmpdir) / "cache" / "images" / "img_sensitive-success.png"
            attachment_path.parent.mkdir(parents=True)
            attachment_path.write_bytes(b"sensitive image")
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            captured_events = []

            async def capture_handle(event):
                captured_events.append(event)

            adapter.handle_message = capture_handle
            original = {
                "uid": b"208",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Quoted reply",
                "message_id": "<quoted@test.com>",
                "in_reply_to": "",
                "body": "Run this after confirmation",
                "attachments": [],
                "date": "",
            }
            asyncio.run(adapter._dispatch_message(original))
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])
            confirm = dict(
                original,
                uid=b"209",
                subject="Re: Quoted reply",
                message_id="<confirm-quoted@test.com>",
                body=f"/confirm {code}\n\nOn Sat, Hermes wrote:\n> Reply with: /confirm {code}",
            )

            asyncio.run(adapter._dispatch_message(confirm))

            self.assertEqual(len(captured_events), 1)
            self.assertIn("Run this after confirmation", captured_events[0].text)

    def test_challenge_mode_wildcard_allowlist_is_challenged(self):
        """Wildcard allowlists work in challenge mode instead of early-dropping senders."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "*",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()
            msg_data = {
                "uid": b"210",
                "sender_addr": "anyone@test.com",
                "sender_name": "Anyone",
                "subject": "Wildcard",
                "message_id": "<wildcard@test.com>",
                "in_reply_to": "",
                "body": "Please confirm wildcard",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_awaited_once()
            stored = json.loads(Path(os.environ["EMAIL_CHALLENGE_STORE"]).read_text())
            self.assertEqual(stored["challenges"][0]["sender"], "anyone@test.com")

    def test_challenge_global_pending_cap_sends_busy_reply_and_deletes_rejected_attachment(self):
        """A full challenge store tells authorized senders to retry and deletes rejected files."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "*",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }), patch("gateway.email_challenge.MAX_PENDING_CHALLENGES_TOTAL", 2):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()
            rejected_attachment_path = _write_cached_test_attachment(
                "document", "doc_rejected-cap.pdf", b"rejected cap attachment"
            )

            for index, sender in enumerate(["one@test.com", "two@test.com", "three@test.com"], start=1):
                attachments = []
                if index == 3:
                    attachments = [{
                        "type": "document",
                        "media_type": "application/pdf",
                        "path": str(rejected_attachment_path),
                    }]
                asyncio.run(adapter._dispatch_message({
                    "uid": str(300 + index).encode(),
                    "sender_addr": sender,
                    "sender_name": sender,
                    "subject": f"Request {index}",
                    "message_id": f"<request-{index}@test.com>",
                    "in_reply_to": "",
                    "body": f"Please run request {index}",
                    "attachments": attachments,
                    "date": "",
                }))

            adapter._message_handler.assert_not_called()
            self.assertEqual(adapter.send.await_count, 3)
            self.assertIn("busy", adapter.send.await_args.args[1].lower())
            self.assertFalse(rejected_attachment_path.exists())
            stored = json.loads(Path(os.environ["EMAIL_CHALLENGE_STORE"]).read_text())
            self.assertEqual(len(stored["challenges"]), 2)
            self.assertEqual({entry["sender"] for entry in stored["challenges"]}, {"one@test.com", "two@test.com"})

    def test_challenge_rejects_oversized_body_without_storing_or_retaining_attachment(self):
        """Oversized bodies are rejected instead of truncating what the agent would see."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            attachment_path = _write_cached_test_attachment(
                "document", "doc_too-large.pdf", b"too large attachment"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()

            asyncio.run(adapter._dispatch_message({
                "uid": b"312",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Too large",
                "message_id": "<too-large@test.com>",
                "in_reply_to": "",
                "body": "x" * (50_000 + 1),
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(attachment_path),
                }],
                "date": "",
            }))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_awaited_once()
            self.assertIn("too large", adapter.send.await_args.args[1].lower())
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())
            self.assertFalse(attachment_path.exists())

    def test_challenge_rejects_too_many_attachments_without_storing(self):
        """Challenge storage caps attachment lists and deletes rejected cached files."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }), patch("gateway.platforms.email.MAX_EMAIL_CHALLENGE_ATTACHMENTS", 1):
            first = _write_cached_test_attachment("document", "doc_too-many-1.pdf", b"first")
            second = _write_cached_test_attachment("document", "doc_too-many-2.pdf", b"second")
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()

            asyncio.run(adapter._dispatch_message({
                "uid": b"313",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Too many attachments",
                "message_id": "<too-many-attachments@test.com>",
                "in_reply_to": "",
                "body": "Please process these attachments",
                "attachments": [
                    {"type": "document", "media_type": "application/pdf", "path": str(first)},
                    {"type": "document", "media_type": "application/pdf", "path": str(second)},
                ],
                "date": "",
            }))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_awaited_once()
            self.assertIn("too many attachments", adapter.send.await_args.args[1].lower())
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())

    def test_challenge_rejects_oversized_attachment_metadata_without_storing(self):
        """Challenge storage caps serialized event size, including attachment metadata."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }), patch("gateway.platforms.email.MAX_EMAIL_CHALLENGE_EVENT_BYTES", 256):
            attachment_path = _write_cached_test_attachment(
                "document", "doc_oversized-metadata.pdf", b"oversized metadata"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()

            asyncio.run(adapter._dispatch_message({
                "uid": b"314",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Oversized metadata",
                "message_id": "<oversized-metadata@test.com>",
                "in_reply_to": "",
                "body": "Small body",
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "filename": "x" * 2048,
                    "path": str(attachment_path),
                }],
                "date": "",
            }))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_awaited_once()
            self.assertIn("too large", adapter.send.await_args.args[1].lower())
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())
            self.assertFalse(attachment_path.exists())

    def test_challenge_rejects_oversized_attachment_payload_without_storing(self):
        """Challenge fetch byte caps reject oversized cached payloads before storage."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()

            asyncio.run(adapter._dispatch_message({
                "uid": b"315",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Oversized attachment payload",
                "message_id": "<oversized-attachment-payload@test.com>",
                "in_reply_to": "",
                "body": "Small body",
                "attachments": [],
                "date": "",
                "_email_challenge_attachment_error": "too_large",
            }))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_awaited_once()
            self.assertIn("attachment", adapter.send.await_args.args[1].lower())
            self.assertIn("too large", adapter.send.await_args.args[1].lower())
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())

    def test_challenge_global_pending_cap_ignores_expired_entries(self):
        """Expired challenges are cleaned up before enforcing the global pending cap."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "*",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
            "EMAIL_CHALLENGE_TTL_SECONDS": "1",
        }), patch("gateway.email_challenge.MAX_PENDING_CHALLENGES_TOTAL", 1):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            store_path = Path(os.environ["EMAIL_CHALLENGE_STORE"])

            asyncio.run(adapter._dispatch_message({
                "uid": b"310",
                "sender_addr": "old@test.com",
                "sender_name": "Old",
                "subject": "Old",
                "message_id": "<old-cap@test.com>",
                "in_reply_to": "",
                "body": "Old request",
                "attachments": [],
                "date": "",
            }))
            stored = json.loads(store_path.read_text())
            stored["challenges"][0]["created_at"] = 0
            store_path.write_text(json.dumps(stored))

            asyncio.run(adapter._dispatch_message({
                "uid": b"311",
                "sender_addr": "new@test.com",
                "sender_name": "New",
                "subject": "New",
                "message_id": "<new-cap@test.com>",
                "in_reply_to": "",
                "body": "New request",
                "attachments": [],
                "date": "",
            }))

            self.assertEqual(adapter.send.await_count, 2)
            stored = json.loads(store_path.read_text())
            self.assertEqual(len(stored["challenges"]), 1)
            self.assertEqual(stored["challenges"][0]["sender"], "new@test.com")

    def test_challenge_mode_email_allow_all_still_challenges(self):
        """Challenge mode must not bypass confirmation when EMAIL_ALLOW_ALL_USERS authorizes sender."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOW_ALL_USERS": "true",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }, clear=False):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()
            msg_data = {
                "uid": b"215",
                "sender_addr": "allowall@test.com",
                "sender_name": "Allow All",
                "subject": "Allow all",
                "message_id": "<allow-all@test.com>",
                "in_reply_to": "",
                "body": "This still needs confirmation",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_awaited_once()
            stored = json.loads(Path(os.environ["EMAIL_CHALLENGE_STORE"]).read_text())
            self.assertEqual(stored["challenges"][0]["sender"], "allowall@test.com")

    def test_challenge_mode_global_allowlist_still_challenges(self):
        """Challenge mode must not bypass confirmation for GATEWAY_ALLOWED_USERS-authorized email."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "GATEWAY_ALLOWED_USERS": "global@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }, clear=False):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()
            msg_data = {
                "uid": b"216",
                "sender_addr": "global@test.com",
                "sender_name": "Global",
                "subject": "Global allowlist",
                "message_id": "<global@test.com>",
                "in_reply_to": "",
                "body": "This also needs confirmation",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_awaited_once()
            stored = json.loads(Path(os.environ["EMAIL_CHALLENGE_STORE"]).read_text())
            self.assertEqual(stored["challenges"][0]["sender"], "global@test.com")

    def test_challenge_mode_config_allow_from_without_env_allowlist_is_not_challenged(self):
        """Challenge mode does not challenge config-only allow_from senders.

        The adapter challenge gate intentionally mirrors the central gateway auth paths
        it can validate before storing untrusted bodies. config.yaml allow_from is a
        direct-mode adapter filter, not a central challenge-mode authorization source.
        """
        import asyncio
        from gateway.config import PlatformConfig
        from gateway.platforms.email import EmailAdapter

        env = {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_CHALLENGE_STORE": "",
            "EMAIL_ALLOWED_USERS": "",
            "EMAIL_ALLOW_ALL_USERS": "",
            "GATEWAY_ALLOWED_USERS": "",
            "GATEWAY_ALLOW_ALL_USERS": "",
        }
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, env, clear=False):
            os.environ["EMAIL_CHALLENGE_STORE"] = str(Path(tmpdir) / "challenges.json")
            adapter = EmailAdapter(PlatformConfig(enabled=True, extra={"allow_from": ["config@test.com"]}))
            adapter._address = "hermes@test.com"
            adapter._password = "secret"
            adapter._imap_host = "imap.test.com"
            adapter._smtp_host = "smtp.test.com"
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()
            msg_data = {
                "uid": b"217",
                "sender_addr": "config@test.com",
                "sender_name": "Config",
                "subject": "Config allowlist",
                "message_id": "<config@test.com>",
                "in_reply_to": "",
                "body": "Config-only authorization must not be retained",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_not_awaited()
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())

    def test_challenge_confirm_scrubs_store_but_preserves_original_attachment_file(self):
        """Successful confirmation scrubs retained metadata without deleting files the agent may read."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            attachment_path = _write_cached_test_attachment(
                "image", "img_sensitive-success.png", b"sensitive image"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            captured_events = []

            async def capture_handle(event):
                captured_events.append(event)

            adapter.handle_message = capture_handle
            original = {
                "uid": b"218",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Retain success",
                "message_id": "<retain-success@test.com>",
                "in_reply_to": "",
                "body": "Sensitive success body",
                "attachments": [{
                    "type": "image",
                    "media_type": "image/png",
                    "path": str(attachment_path),
                }],
                "date": "",
            }
            asyncio.run(adapter._dispatch_message(original))
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])
            confirm_attachment_path = _write_cached_test_attachment(
                "document", "doc_confirm-success.pdf", b"confirmation attachment"
            )
            confirm = dict(
                original,
                uid=b"219",
                message_id="<confirm-retain-success@test.com>",
                body=f"/confirm {code}",
                attachments=[{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(confirm_attachment_path),
                }],
            )

            asyncio.run(adapter._dispatch_message(confirm))

            self.assertEqual(len(captured_events), 1)
            self.assertIn("Sensitive success body", captured_events[0].text)
            self.assertEqual(captured_events[0].media_urls, [str(attachment_path)])
            self.assertTrue(attachment_path.exists())
            self.assertFalse(confirm_attachment_path.exists())
            store_text = Path(os.environ["EMAIL_CHALLENGE_STORE"]).read_text()
            stored = json.loads(store_text)
            self.assertEqual(len(stored["challenges"]), 1)
            self.assertTrue(stored["challenges"][0]["used"])
            self.assertNotIn("event", stored["challenges"][0])
            self.assertNotIn("subject", stored["challenges"][0])
            self.assertNotIn("message_id", stored["challenges"][0])
            self.assertNotIn("Sensitive success body", store_text)
            self.assertNotIn("Retain success", store_text)
            self.assertNotIn("<retain-success@test.com>", store_text)
            self.assertNotIn("img_sensitive-success.png", store_text)

    def test_challenge_send_failure_removes_pending_original_and_attachment(self):
        """If the challenge code cannot be delivered, the retained original is removed."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            attachment_path = _write_cached_test_attachment(
                "document", "doc_send-failure.pdf", b"failed challenge attachment"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=False, error="smtp down"))
            adapter.handle_message = AsyncMock()
            original = {
                "uid": b"222",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Undelivered challenge",
                "message_id": "<send-failure@test.com>",
                "in_reply_to": "",
                "body": "Sensitive body without delivered code",
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(attachment_path),
                }],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(original))

            adapter.handle_message.assert_not_awaited()
            adapter.send.assert_awaited_once()
            self.assertFalse(attachment_path.exists())
            store_text = Path(os.environ["EMAIL_CHALLENGE_STORE"]).read_text()
            self.assertEqual(json.loads(store_text)["challenges"], [])
            self.assertNotIn("Sensitive body without delivered code", store_text)
            self.assertNotIn("doc_send-failure.pdf", store_text)

    def test_challenge_create_store_failure_is_contained_and_deletes_attachment(self):
        """Create-time store I/O failures fail closed without dispatching the original."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }), patch("gateway.platforms.email.EmailChallengeStore.create", side_effect=OSError("store down")):
            attachment_path = _write_cached_test_attachment(
                "document", "doc_create-failure.pdf", b"create failure"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()

            asyncio.run(adapter._dispatch_message({
                "uid": b"226",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Create failure",
                "message_id": "<create-failure@test.com>",
                "in_reply_to": "",
                "body": "Sensitive body should not dispatch",
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(attachment_path),
                }],
                "date": "",
            }))

            adapter.handle_message.assert_not_awaited()
            adapter.send.assert_awaited_once()
            self.assertIn("temporarily", adapter.send.await_args.args[1].lower())
            self.assertFalse(attachment_path.exists())

    def test_challenge_confirm_store_failure_is_contained_and_deletes_confirmation_attachment(self):
        """Confirm-time store I/O failures do not dispatch or retain confirmation files."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }), patch("gateway.platforms.email.EmailChallengeStore.confirm", side_effect=OSError("store down")):
            attachment_path = _write_cached_test_attachment(
                "image", "img_confirm-store-failure.png", b"confirm failure"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()

            asyncio.run(adapter._dispatch_message({
                "uid": b"227",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Re: Confirm failure",
                "message_id": "<confirm-store-failure@test.com>",
                "in_reply_to": "",
                "body": "/confirm broken-code",
                "attachments": [{
                    "type": "image",
                    "media_type": "image/png",
                    "path": str(attachment_path),
                }],
                "date": "",
            }))

            adapter.handle_message.assert_not_awaited()
            adapter.send.assert_awaited_once()
            self.assertIn("temporarily", adapter.send.await_args.args[1].lower())
            self.assertFalse(attachment_path.exists())

    def test_challenge_send_failure_remove_error_is_contained_and_deletes_attachment(self):
        """Remove failures after challenge-send failure are logged and cleaned best-effort."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }), patch("gateway.platforms.email.EmailChallengeStore.remove", side_effect=OSError("remove failed")):
            attachment_path = _write_cached_test_attachment(
                "document", "doc_remove-failure.pdf", b"remove failure"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=False, error="smtp down"))
            adapter.handle_message = AsyncMock()

            asyncio.run(adapter._dispatch_message({
                "uid": b"228",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Remove failure",
                "message_id": "<remove-failure@test.com>",
                "in_reply_to": "",
                "body": "Sensitive body with failed send",
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(attachment_path),
                }],
                "date": "",
            }))

            adapter.handle_message.assert_not_awaited()
            adapter.send.assert_awaited_once()
            self.assertFalse(attachment_path.exists())

    def test_challenge_send_failure_remove_false_deletes_attachment_fallback(self):
        """A false remove result after send failure still deletes retained original files."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }), patch("gateway.platforms.email.EmailChallengeStore.remove", return_value=False):
            attachment_path = _write_cached_test_attachment(
                "document", "doc_remove-false.pdf", b"remove returned false"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=False, error="smtp down"))
            adapter.handle_message = AsyncMock()

            asyncio.run(adapter._dispatch_message({
                "uid": b"230",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Remove false",
                "message_id": "<remove-false@test.com>",
                "in_reply_to": "",
                "body": "Sensitive body with failed cleanup",
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(attachment_path),
                }],
                "date": "",
            }))

            adapter.handle_message.assert_not_awaited()
            adapter.send.assert_awaited_once()
            self.assertFalse(attachment_path.exists())

    def test_invalid_challenge_confirmation_deletes_confirmation_attachment(self):
        """Attachments cached from invalid /confirm replies are not retained."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            attachment_path = _write_cached_test_attachment(
                "image", "img_invalid-confirm.png", b"invalid confirm image"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()
            confirm = {
                "uid": b"223",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Re: Unknown challenge",
                "message_id": "<invalid-confirm@test.com>",
                "in_reply_to": "",
                "body": "/confirm missing-code",
                "attachments": [{
                    "type": "image",
                    "media_type": "image/png",
                    "path": str(attachment_path),
                }],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(confirm))

            adapter.handle_message.assert_not_awaited()
            self.assertIn("invalid", adapter.send.await_args.args[1].lower())
            self.assertFalse(attachment_path.exists())

    def test_overlong_challenge_confirmation_rejected_without_store_lookup(self):
        """Very large /confirm tokens are rejected before hashing or store lookup."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }), patch("gateway.platforms.email.EmailChallengeStore.confirm") as confirm:
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()
            msg_data = {
                "uid": b"231",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Re: Overlong confirm",
                "message_id": "<overlong-confirm@test.com>",
                "in_reply_to": "",
                "body": "/confirm " + ("x" * 4096),
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter.handle_message.assert_not_awaited()
            adapter.send.assert_awaited_once()
            self.assertIn("invalid", adapter.send.await_args.args[1].lower())
            confirm.assert_not_called()

    def test_expired_challenge_confirm_scrubs_stored_event(self):
        """Expired confirmation rejects and immediately scrubs retained body/attachments."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
            "EMAIL_CHALLENGE_TTL_SECONDS": "1",
        }):
            attachment_path = _write_cached_test_attachment(
                "document", "doc_sensitive-expired.pdf", b"sensitive document"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()
            original = {
                "uid": b"220",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Retain expiry",
                "message_id": "<retain-expiry@test.com>",
                "in_reply_to": "",
                "body": "Sensitive expired body",
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(attachment_path),
                }],
                "date": "",
            }
            self.assertTrue(attachment_path.exists())
            asyncio.run(adapter._dispatch_message(original))
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])
            store_path = Path(os.environ["EMAIL_CHALLENGE_STORE"])
            stored = json.loads(store_path.read_text())
            stored["challenges"][0]["created_at"] = 0
            store_path.write_text(json.dumps(stored))
            confirm_attachment_path = _write_cached_test_attachment(
                "image", "img_confirm-expired.png", b"expired confirmation image"
            )
            confirm = dict(
                original,
                uid=b"221",
                message_id="<confirm-retain-expiry@test.com>",
                body=f"/confirm {code}",
                attachments=[{
                    "type": "image",
                    "media_type": "image/png",
                    "path": str(confirm_attachment_path),
                }],
            )

            asyncio.run(adapter._dispatch_message(confirm))

            adapter.handle_message.assert_not_awaited()
            self.assertIn("expired", adapter.send.await_args.args[1].lower())
            self.assertFalse(attachment_path.exists())
            self.assertFalse(confirm_attachment_path.exists())
            store_text = store_path.read_text()
            stored = json.loads(store_text)
            self.assertEqual(len(stored["challenges"]), 1)
            self.assertTrue(stored["challenges"][0]["used"])
            self.assertNotIn("event", stored["challenges"][0])
            self.assertNotIn("subject", stored["challenges"][0])
            self.assertNotIn("message_id", stored["challenges"][0])
            self.assertNotIn("Sensitive expired body", store_text)
            self.assertNotIn("Retain expiry", store_text)
            self.assertNotIn("<retain-expiry@test.com>", store_text)
            self.assertNotIn("doc_sensitive-expired.pdf", store_text)

    def test_expired_challenge_confirm_keeps_cleanup_tombstone_after_unlink_failure(self):
        """Expired confirm scrubs sensitive metadata but keeps retry data when unlink fails."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
            "EMAIL_CHALLENGE_TTL_SECONDS": "1",
        }):
            attachment_path = _write_cached_test_attachment(
                "document", "doc_confirm-unlink-failure.pdf", b"confirm unlink failure"
            )
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()
            original = {
                "uid": b"232",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Confirm unlink failure",
                "message_id": "<confirm-unlink-failure@test.com>",
                "in_reply_to": "",
                "body": "Sensitive body with unlink failure",
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(attachment_path),
                }],
                "date": "",
            }
            asyncio.run(adapter._dispatch_message(original))
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])
            store_path = Path(os.environ["EMAIL_CHALLENGE_STORE"])
            stored = json.loads(store_path.read_text())
            stored["challenges"][0]["created_at"] = 0
            store_path.write_text(json.dumps(stored))

            with patch("pathlib.Path.unlink", side_effect=OSError("permission denied")):
                asyncio.run(adapter._dispatch_message(dict(original, uid=b"233", body=f"/confirm {code}")))

            self.assertTrue(attachment_path.exists())
            store_text = store_path.read_text()
            stored = json.loads(store_text)
            self.assertEqual(len(stored["challenges"]), 1)
            self.assertTrue(stored["challenges"][0]["used"])
            self.assertNotIn("event", stored["challenges"][0])
            self.assertNotIn("subject", stored["challenges"][0])
            self.assertNotIn("message_id", stored["challenges"][0])
            self.assertNotIn("Sensitive body with unlink failure", store_text)
            self.assertEqual(stored.get("cleanup_pending", [])[0]["attachments"][0]["path"], str(attachment_path))

            adapter._email_challenge_store().cleanup_expired()
            self.assertFalse(attachment_path.exists())
            self.assertEqual(json.loads(store_path.read_text()).get("cleanup_pending", []), [])

    def test_challenge_mode_mixed_case_allowlist_does_not_pre_auth_lowercase_sender(self):
        """Challenge pre-auth mirrors central gateway's case-sensitive allowlist matching."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "Admin@Test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()
            msg_data = {
                "uid": b"224",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Case mismatch",
                "message_id": "<case-mismatch@test.com>",
                "in_reply_to": "",
                "body": "This should not be retained before central auth would allow it",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_not_awaited()
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())

    def test_challenge_confirmation_preserves_attachments(self):
        """Confirmed challenge dispatches the original attachment metadata."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            captured_events = []

            async def capture_handle(event):
                captured_events.append(event)

            adapter.handle_message = capture_handle
            original = {
                "uid": b"211",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Photo",
                "message_id": "<photo@test.com>",
                "in_reply_to": "",
                "body": "See attached",
                "attachments": [{
                    "type": "image",
                    "media_type": "image/png",
                    "path": "/tmp/hermes-photo.png",
                }],
                "date": "",
            }
            asyncio.run(adapter._dispatch_message(original))
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])
            confirm = dict(original, uid=b"212", message_id="<confirm-photo@test.com>", body=f"/confirm {code}")

            asyncio.run(adapter._dispatch_message(confirm))

            self.assertEqual(len(captured_events), 1)
            self.assertEqual(captured_events[0].media_urls, ["/tmp/hermes-photo.png"])
            self.assertEqual(captured_events[0].media_types, ["image/png"])

    def test_malformed_challenge_created_at_does_not_crash(self):
        """Malformed store timestamps are rejected instead of crashing dispatch."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()
            original = {
                "uid": b"213",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Bad timestamp",
                "message_id": "<bad-time@test.com>",
                "in_reply_to": "",
                "body": "This should not crash",
                "attachments": [],
                "date": "",
            }
            asyncio.run(adapter._dispatch_message(original))
            code = self._extract_confirmation_code(adapter.send.await_args.args[1])
            store_path = Path(os.environ["EMAIL_CHALLENGE_STORE"])
            stored = json.loads(store_path.read_text())
            stored["challenges"][0]["created_at"] = "not-a-float"
            store_path.write_text(json.dumps(stored))
            confirm = dict(original, uid=b"214", message_id="<confirm-bad-time@test.com>", body=f"/confirm {code}")

            asyncio.run(adapter._dispatch_message(confirm))

            adapter.handle_message.assert_not_awaited()
            self.assertIn("invalid or already used", adapter.send.await_args.args[1].lower())
            self.assertEqual(json.loads(store_path.read_text())["challenges"], [])

    def test_challenge_mode_unknown_sender_uses_existing_drop_path(self):
        """Unknown senders are not challenged as if they were allowlisted."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()
            msg_data = {
                "uid": b"207",
                "sender_addr": "outsider@test.com",
                "sender_name": "Outsider",
                "subject": "Hello",
                "message_id": "<outsider@test.com>",
                "in_reply_to": "",
                "body": "Let me in",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_not_awaited()
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())

    def test_challenge_email_allowlist_bare_local_part_does_not_match_any_domain(self):
        """EMAIL_ALLOWED_USERS=admin must not challenge admin@arbitrary-domain."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin",
            "GATEWAY_ALLOWED_USERS": "",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }, clear=False):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()

            asyncio.run(adapter._dispatch_message({
                "uid": b"229",
                "sender_addr": "admin@evil.example",
                "sender_name": "Admin Evil",
                "subject": "Local part bypass",
                "message_id": "<local-part-bypass@test.com>",
                "in_reply_to": "",
                "body": "This must not be retained or challenged.",
                "attachments": [],
                "date": "",
            }))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_not_awaited()
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())

    def test_invalid_email_auth_mode_fails_closed_to_challenge(self):
        """Security-sensitive EMAIL_AUTH_MODE typos must not fall back to direct dispatch."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challlenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }):
            adapter = self._make_adapter()
            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter._message_handler = MagicMock()
            msg_data = {
                "uid": b"225",
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Typo auth mode",
                "message_id": "<typo-auth-mode@test.com>",
                "in_reply_to": "",
                "body": "This must still require confirmation.",
                "attachments": [],
                "date": "",
            }

            asyncio.run(adapter._dispatch_message(msg_data))

            adapter._message_handler.assert_not_called()
            adapter.send.assert_awaited_once()
            self.assertIn("pending confirmation", adapter.send.await_args.args[1])
            stored = json.loads(Path(os.environ["EMAIL_CHALLENGE_STORE"]).read_text())
            self.assertEqual(stored["challenges"][0]["sender"], "admin@test.com")

    def test_check_inbox_passively_cleans_expired_challenge_attachments(self):
        """Polling cleanup removes expired retained challenge data without a new create."""
        import asyncio
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
            "EMAIL_CHALLENGE_TTL_SECONDS": "1",
        }):
            attachment_path = _write_cached_test_attachment(
                "image", "img_passive-expiry.png", b"passive expiry image"
            )
            store_path = Path(os.environ["EMAIL_CHALLENGE_STORE"])
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=1)
            store.create("admin@test.com", "Passive", "<passive@test.com>", {
                "sender_addr": "admin@test.com",
                "sender_name": "Admin",
                "subject": "Passive",
                "message_id": "<passive@test.com>",
                "in_reply_to": "",
                "body": "Sensitive passive body",
                "attachments": [{
                    "type": "image",
                    "media_type": "image/png",
                    "path": str(attachment_path),
                }],
                "date": "",
                "_email_challenge_confirmed": True,
            })
            stored = json.loads(store_path.read_text())
            stored["challenges"][0]["created_at"] = 0
            store_path.write_text(json.dumps(stored))

            adapter = self._make_adapter()
            adapter._fetch_new_messages = MagicMock(return_value=[])
            awaitable = adapter._check_inbox()
            asyncio.run(awaitable)

            self.assertFalse(attachment_path.exists())
            self.assertEqual(json.loads(store_path.read_text())["challenges"], [])


class TestEmailChallengeStore(unittest.TestCase):
    """Focused tests for challenge-store retention cleanup."""

    def test_store_read_oserror_propagates_for_mutating_operations(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store_path.write_text('{"challenges": []}', encoding="utf-8")
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)

            with patch("pathlib.Path.read_text", side_effect=OSError("read failed")):
                with self.assertRaises(OSError):
                    store.create("admin@test.com", "Subject", "<msg@test.com>", {"attachments": []})
                with self.assertRaises(OSError):
                    store.confirm("admin@test.com", "code")
                with self.assertRaises(OSError):
                    store.remove("admin@test.com", "code")
                with self.assertRaises(OSError):
                    store.cleanup_expired()

    def test_attachment_cleanup_refuses_similar_cache_path_outside_hermes_roots(self):
        from gateway.email_challenge import cleanup_challenge_cached_attachments

        with tempfile.TemporaryDirectory() as tmpdir:
            outside_attachment = Path(tmpdir) / "cache" / "documents" / "doc_outside-root.pdf"
            outside_attachment.parent.mkdir(parents=True)
            outside_attachment.write_bytes(b"outside")

            cleanup_challenge_cached_attachments({
                "attachments": [{
                    "type": "document",
                    "media_type": "application/pdf",
                    "path": str(outside_attachment),
                }]
            })

            self.assertTrue(outside_attachment.exists())

    def test_sender_trim_deletes_trimmed_challenge_attachment(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir, patch("gateway.email_challenge.MAX_PENDING_CHALLENGES_PER_SENDER", 1):
            old_attachment = _write_cached_test_attachment("image", "img_trimmed-old.png", b"old")
            new_attachment = _write_cached_test_attachment("image", "img_trimmed-new.png", b"new")
            store = EmailChallengeStore(path=str(Path(tmpdir) / "challenges.json"), ttl_seconds=900)

            store.create("admin@test.com", "Old", "<old@test.com>", {
                "body": "old body",
                "attachments": [{"path": str(old_attachment), "type": "image", "media_type": "image/png"}],
            })
            store.create("admin@test.com", "New", "<new@test.com>", {
                "body": "new body",
                "attachments": [{"path": str(new_attachment), "type": "image", "media_type": "image/png"}],
            })

            self.assertFalse(old_attachment.exists())
            self.assertTrue(new_attachment.exists())

    def test_cleanup_expired_does_not_save_when_no_entries_change(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = EmailChallengeStore(path=str(Path(tmpdir) / "challenges.json"), ttl_seconds=900)
            store.create("admin@test.com", "Fresh", "<fresh@test.com>", {
                "body": "fresh body",
                "attachments": [],
            })

            with patch.object(store, "_save", wraps=store._save) as save:
                store.cleanup_expired()

            save.assert_not_called()

    def test_load_only_confirm_hardens_existing_store_permissions(self):
        from gateway.email_challenge import EmailChallengeStore

        if os.name == "nt":
            self.skipTest("POSIX chmod bits are not meaningful on Windows")

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store_path.write_text('{"challenges": []}', encoding="utf-8")
            os.chmod(store_path, 0o644)
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)

            self.assertEqual(store.confirm("admin@test.com", "missing"), ("not_found", None))

            self.assertEqual(stat.S_IMODE(store_path.stat().st_mode), 0o600)

    def test_pending_cached_attachment_byte_quota_rejects_and_deletes_new_attachment(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "gateway.email_challenge.MAX_PENDING_CHALLENGE_CACHED_ATTACHMENT_BYTES",
            10,
        ):
            first = _write_cached_test_attachment("document", "doc_quota-first.pdf", b"123456")
            second = _write_cached_test_attachment("document", "doc_quota-second.pdf", b"abcdef")
            store = EmailChallengeStore(path=str(Path(tmpdir) / "challenges.json"), ttl_seconds=900)

            first_code = store.create("one@test.com", "First", "<first@test.com>", {
                "body": "first",
                "attachments": [{"path": str(first), "type": "document", "media_type": "application/pdf"}],
            })
            second_code = store.create("two@test.com", "Second", "<second@test.com>", {
                "body": "second",
                "attachments": [{"path": str(second), "type": "document", "media_type": "application/pdf"}],
            })

            self.assertIsNotNone(first_code)
            self.assertIsNone(second_code)
            self.assertTrue(first.exists())
            self.assertFalse(second.exists())
            stored = json.loads((Path(tmpdir) / "challenges.json").read_text())
            self.assertEqual(len(stored["challenges"]), 1)
            self.assertEqual(stored["challenges"][0]["sender"], "one@test.com")

    def test_cleanup_expired_retries_failed_attachment_unlinks(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            attachment_path = _write_cached_test_attachment(
                "document", "doc_cleanup-retry.pdf", b"cleanup retry"
            )
            store_path = Path(tmpdir) / "challenges.json"
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=1)
            store.create("admin@test.com", "Cleanup retry", "<cleanup-retry@test.com>", {
                "body": "sensitive cleanup retry body",
                "attachments": [{
                    "path": str(attachment_path),
                    "type": "document",
                    "media_type": "application/pdf",
                }],
            })
            stored = json.loads(store_path.read_text())
            stored["challenges"][0]["created_at"] = 0
            store_path.write_text(json.dumps(stored))

            with patch("pathlib.Path.unlink", side_effect=OSError("permission denied")):
                store.cleanup_expired()

            self.assertTrue(attachment_path.exists())
            stored = json.loads(store_path.read_text())
            self.assertEqual(stored["challenges"], [])
            self.assertEqual(stored.get("cleanup_pending", [])[0]["attachments"][0]["path"], str(attachment_path))
            self.assertNotIn("sensitive cleanup retry body", store_path.read_text())

            store.cleanup_expired()
            self.assertFalse(attachment_path.exists())
            self.assertEqual(json.loads(store_path.read_text()).get("cleanup_pending", []), [])

    def test_cleanup_pending_tombstone_removed_when_file_already_gone(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            attachment_path = _write_cached_test_attachment(
                "document", "doc_cleanup-already-gone.pdf", b"already gone"
            )
            attachment_path.unlink()
            store_path = Path(tmpdir) / "challenges.json"
            store_path.write_text(json.dumps({
                "challenges": [],
                "cleanup_pending": [{
                    "attachments": [{
                        "path": str(attachment_path),
                        "type": "document",
                        "media_type": "application/pdf",
                    }],
                }],
            }))
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)

            store.cleanup_expired()

            self.assertNotIn("cleanup_pending", json.loads(store_path.read_text()))

    def test_corrupt_store_file_is_removed_instead_of_retaining_plaintext(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store_path.write_text('{"challenges":[{"body":"secret body"}', encoding="utf-8")
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)

            self.assertEqual(store.confirm("admin@test.com", "missing"), ("not_found", None))

            self.assertFalse(store_path.exists())

    def test_invalid_store_shape_is_removed_instead_of_retaining_plaintext(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store_path.write_text('{"body":"secret body","challenges":{}}', encoding="utf-8")
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)

            self.assertEqual(store.confirm("admin@test.com", "missing"), ("not_found", None))

            self.assertFalse(store_path.exists())

    def test_partially_valid_store_with_invalid_challenge_entry_is_sanitized(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store_path.write_text('{"challenges":["secret body"]}', encoding="utf-8")
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)

            self.assertEqual(store.confirm("admin@test.com", "missing"), ("not_found", None))

            self.assertNotIn("secret body", store_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(store_path.read_text(encoding="utf-8")), {"challenges": []})

    def test_partially_valid_store_with_extra_sensitive_key_is_sanitized(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store_path.write_text(
                '{"body":"secret body","challenges":[]}',
                encoding="utf-8",
            )
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)

            self.assertEqual(store.cleanup_expired(), None)

            self.assertNotIn("secret body", store_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(store_path.read_text(encoding="utf-8")), {"challenges": []})

    def test_fresh_partial_dict_challenge_body_is_sanitized_on_confirm_miss(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store_path.write_text(json.dumps({
                "challenges": [{
                    "created_at": time.time(),
                    "used": False,
                    "body": "secret body",
                }],
            }), encoding="utf-8")
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)

            self.assertEqual(store.confirm("admin@test.com", "missing"), ("not_found", None))

            store_text = store_path.read_text(encoding="utf-8")
            self.assertNotIn("secret body", store_text)
            self.assertEqual(json.loads(store_text), {"challenges": []})

    def test_fresh_partial_dict_challenge_sensitive_fields_are_sanitized_on_cleanup(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store_path.write_text(json.dumps({
                "challenges": [{
                    "created_at": time.time(),
                    "used": False,
                    "event": {"body": "secret event body"},
                    "attachments": [{"path": "/tmp/secret.pdf"}],
                    "subject": "secret subject",
                    "message_id": "<secret@test.com>",
                }],
            }), encoding="utf-8")
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)

            self.assertEqual(store.cleanup_expired(), None)

            store_text = store_path.read_text(encoding="utf-8")
            self.assertNotIn("secret event body", store_text)
            self.assertNotIn("secret subject", store_text)
            self.assertNotIn("<secret@test.com>", store_text)
            self.assertNotIn("secret.pdf", store_text)
            self.assertEqual(json.loads(store_text), {"challenges": []})

    def test_valid_created_challenge_still_confirms_after_store_sanitization(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)
            code = store.create("admin@test.com", "Valid subject", "<valid@test.com>", {
                "body": "valid body",
                "attachments": [],
            })

            result, event = EmailChallengeStore(path=str(store_path), ttl_seconds=900).confirm(
                "admin@test.com",
                code,
            )

            self.assertEqual(result, "ok")
            self.assertEqual(event["body"], "valid body")

    def test_missing_created_at_challenge_is_dropped_during_sanitization(self):
        from gateway.email_challenge import EmailChallengeStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "challenges.json"
            store = EmailChallengeStore(path=str(store_path), ttl_seconds=900)
            code = store.create("admin@test.com", "Missing timestamp", "<missing-time@test.com>", {
                "body": "sensitive body",
                "attachments": [],
            })
            stored = json.loads(store_path.read_text())
            stored["challenges"][0].pop("created_at")
            store_path.write_text(json.dumps(stored), encoding="utf-8")

            self.assertEqual(store.confirm("admin@test.com", code), ("not_found", None))

            store_text = store_path.read_text(encoding="utf-8")
            self.assertNotIn("sensitive body", store_text)
            self.assertEqual(json.loads(store_text), {"challenges": []})


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
        from unittest.mock import AsyncMock
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
            from gateway.platforms.email import EmailAdapter
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

    def test_challenge_fetch_skips_unauthorized_sender_before_attachment_cache(self):
        """Challenge-mode unauthorized senders are dropped before caching attachments."""
        adapter = self._make_adapter()
        raw_email = MIMEMultipart()
        raw_email["From"] = "outsider@test.com"
        raw_email["Subject"] = "Unauthorized attachment"
        raw_email["Message-ID"] = "<unauthorized-attachment@test.com>"
        raw_email.attach(MIMEText("Do not retain me", "plain", "utf-8"))
        part = MIMEBase("application", "pdf")
        part.set_payload(b"sensitive pdf")
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=secret.pdf")
        raw_email.attach(part)
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        with patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "GATEWAY_ALLOWED_USERS": "",
            "EMAIL_ALLOW_ALL_USERS": "",
            "GATEWAY_ALLOW_ALL_USERS": "",
        }, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email.cache_document_from_bytes") as mock_cache:
            results = adapter._fetch_new_messages()

        self.assertEqual(results, [])
        mock_cache.assert_not_called()

    def test_challenge_fetch_oversized_message_skips_attachment_cache(self):
        """Oversized challenge-mode messages keep body for rejection but do not cache attachments."""
        adapter = self._make_adapter()
        raw_email = MIMEMultipart()
        raw_email["From"] = "admin@test.com"
        raw_email["Subject"] = "Oversized attachment"
        raw_email["Message-ID"] = "<oversized-attachment@test.com>"
        raw_email.attach(MIMEText("x" * (50_000 + 1), "plain", "utf-8"))
        part = MIMEBase("application", "pdf")
        part.set_payload(b"oversized pdf")
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=oversized.pdf")
        raw_email.attach(part)
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        with patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
        }, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email.cache_document_from_bytes") as mock_cache:
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["attachments"], [])
        self.assertGreater(len(results[0]["body"]), 50_000)
        mock_cache.assert_not_called()

    def test_challenge_fetch_oversized_attachment_payload_skips_cache(self):
        """Challenge-mode oversized attachment payloads are rejected before caching."""
        adapter = self._make_adapter()
        raw_email = MIMEMultipart()
        raw_email["From"] = "admin@test.com"
        raw_email["Subject"] = "Oversized attachment payload"
        raw_email["Message-ID"] = "<oversized-attachment-payload@test.com>"
        raw_email.attach(MIMEText("Body is small", "plain", "utf-8"))
        part = MIMEBase("application", "pdf")
        part.set_payload(b"x" * 64)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=oversized-payload.pdf")
        raw_email.attach(part)
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        with patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
        }, clear=False), \
             patch("gateway.platforms.email.MAX_EMAIL_CHALLENGE_ATTACHMENT_BYTES", 16), \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email.cache_document_from_bytes") as mock_cache:
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["attachments"], [])
        self.assertEqual(results[0].get("_email_challenge_attachment_error"), "too_large")
        mock_cache.assert_not_called()

    def test_challenge_fetch_total_attachment_cap_deletes_partial_cache_and_stores_no_event(self):
        """Total-byte failures remove earlier cached files and dispatch stores no challenge."""
        import asyncio
        from gateway.platforms import base

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }, clear=False):
            base.DOCUMENT_CACHE_DIR = Path(tmpdir) / "cache" / "documents"
            adapter = self._make_adapter()
            raw_email = MIMEMultipart()
            raw_email["From"] = "admin@test.com"
            raw_email["Subject"] = "Partial cache leak"
            raw_email["Message-ID"] = "<partial-cache-leak@test.com>"
            raw_email.attach(MIMEText("Body is small", "plain", "utf-8"))
            first = MIMEBase("application", "pdf")
            first.set_payload(b"small")
            encoders.encode_base64(first)
            first.add_header("Content-Disposition", "attachment; filename=first.pdf")
            raw_email.attach(first)
            second = MIMEBase("application", "pdf")
            second.set_payload(b"large payload")
            encoders.encode_base64(second)
            second.add_header("Content-Disposition", "attachment; filename=second.pdf")
            raw_email.attach(second)
            mock_imap = MagicMock()

            def uid_handler(command, *args):
                if command == "search":
                    return ("OK", [b"1"])
                if command == "fetch":
                    return ("OK", [(b"1", raw_email.as_bytes())])
                return ("NO", [])

            mock_imap.uid.side_effect = uid_handler
            with patch("gateway.platforms.email.MAX_EMAIL_CHALLENGE_TOTAL_ATTACHMENT_BYTES", 10), \
                 patch("imaplib.IMAP4_SSL", return_value=mock_imap):
                results = adapter._fetch_new_messages()

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["attachments"], [])
            self.assertEqual(results[0].get("_email_challenge_attachment_error"), "too_large")
            self.assertEqual(list(base.get_document_cache_dir().iterdir()), [])

            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()
            asyncio.run(adapter._dispatch_message(results[0]))

            adapter.handle_message.assert_not_awaited()
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())

    def test_challenge_fetch_attachment_count_cap_deletes_partial_cache_and_reports_too_many(self):
        """Fetch-time count failures remove cached files and use the too-many rejection path."""
        import asyncio
        from gateway.platforms import base

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "challenge",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
            "EMAIL_CHALLENGE_STORE": str(Path(tmpdir) / "challenges.json"),
        }, clear=False):
            base.DOCUMENT_CACHE_DIR = Path(tmpdir) / "cache" / "documents"
            adapter = self._make_adapter()
            raw_email = MIMEMultipart()
            raw_email["From"] = "admin@test.com"
            raw_email["Subject"] = "Too many fetch attachments"
            raw_email["Message-ID"] = "<too-many-fetch-attachments@test.com>"
            raw_email.attach(MIMEText("Body is small", "plain", "utf-8"))
            for name in ("first.pdf", "second.pdf"):
                part = MIMEBase("application", "pdf")
                part.set_payload(b"small")
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={name}")
                raw_email.attach(part)
            mock_imap = MagicMock()

            def uid_handler(command, *args):
                if command == "search":
                    return ("OK", [b"1"])
                if command == "fetch":
                    return ("OK", [(b"1", raw_email.as_bytes())])
                return ("NO", [])

            mock_imap.uid.side_effect = uid_handler
            with patch("gateway.platforms.email.MAX_EMAIL_CHALLENGE_ATTACHMENTS", 1), \
                 patch("imaplib.IMAP4_SSL", return_value=mock_imap):
                results = adapter._fetch_new_messages()

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["attachments"], [])
            self.assertEqual(results[0].get("_email_challenge_attachment_error"), "too_many")
            self.assertEqual(list(base.get_document_cache_dir().iterdir()), [])

            adapter.send = AsyncMock(return_value=SendResult(success=True))
            adapter.handle_message = AsyncMock()
            asyncio.run(adapter._dispatch_message(results[0]))

            adapter.handle_message.assert_not_awaited()
            self.assertIn("too many attachments", adapter.send.await_args.args[1].lower())
            self.assertFalse(Path(os.environ["EMAIL_CHALLENGE_STORE"]).exists())

    def test_direct_fetch_skips_non_allowlisted_sender_before_attachment_cache(self):
        """Direct-mode EMAIL_ALLOWED_USERS rejection does not retain cached attachments."""
        adapter = self._make_adapter()
        raw_email = MIMEMultipart()
        raw_email["From"] = "outsider@test.com"
        raw_email["Subject"] = "Direct unauthorized attachment"
        raw_email["Message-ID"] = "<direct-unauthorized-attachment@test.com>"
        raw_email.attach(MIMEText("Do not retain me", "plain", "utf-8"))
        part = MIMEBase("application", "pdf")
        part.set_payload(b"direct sensitive pdf")
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=direct-secret.pdf")
        raw_email.attach(part)
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        with patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "direct",
            "EMAIL_ALLOWED_USERS": "admin@test.com",
        }, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email.cache_document_from_bytes") as mock_cache:
            results = adapter._fetch_new_messages()

        self.assertEqual(results, [])
        mock_cache.assert_not_called()

    def test_direct_fetch_global_allowlist_match_caches_attachment(self):
        """Direct-mode global allowlist matches can cache attachments for processing."""
        adapter = self._make_adapter()
        raw_email = MIMEMultipart()
        raw_email["From"] = "admin@test.com"
        raw_email["Subject"] = "Global authorized attachment"
        raw_email["Message-ID"] = "<global-authorized-attachment@test.com>"
        raw_email.attach(MIMEText("Retain me", "plain", "utf-8"))
        part = MIMEBase("application", "pdf")
        part.set_payload(b"global sensitive pdf")
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=global-secret.pdf")
        raw_email.attach(part)
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        with patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "direct",
            "EMAIL_ALLOWED_USERS": "",
            "GATEWAY_ALLOWED_USERS": "admin@test.com",
        }, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email.cache_document_from_bytes") as mock_cache:
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sender_addr"], "admin@test.com")
        mock_cache.assert_called_once()

    def test_direct_fetch_global_allowlist_miss_does_not_cache_attachment(self):
        """Direct-mode global misses can reach central auth only without retained attachments."""
        adapter = self._make_adapter()
        raw_email = MIMEMultipart()
        raw_email["From"] = "outsider@test.com"
        raw_email["Subject"] = "Global unauthorized attachment"
        raw_email["Message-ID"] = "<global-unauthorized-attachment@test.com>"
        raw_email.attach(MIMEText("Central auth may still decide pairing", "plain", "utf-8"))
        part = MIMEBase("application", "pdf")
        part.set_payload(b"global sensitive pdf")
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=global-secret.pdf")
        raw_email.attach(part)
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"1"])
            if command == "fetch":
                return ("OK", [(b"1", raw_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        with patch.dict(os.environ, {
            "EMAIL_AUTH_MODE": "direct",
            "EMAIL_ALLOWED_USERS": "",
            "GATEWAY_ALLOWED_USERS": "admin@test.com",
        }, clear=False), \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("gateway.platforms.email.cache_document_from_bytes") as mock_cache:
            results = adapter._fetch_new_messages()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["sender_addr"], "outsider@test.com")
        self.assertEqual(results[0]["attachments"], [])
        mock_cache.assert_not_called()


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

    def test_check_inbox_cleanup_failure_still_fetches_messages(self):
        """Passive challenge cleanup is best-effort and must not block polling."""
        import asyncio
        adapter = self._make_adapter()
        adapter._cleanup_expired_email_challenges = MagicMock(side_effect=OSError("cleanup failed"))
        adapter._fetch_new_messages = MagicMock(return_value=[])

        with patch.dict(os.environ, {"EMAIL_AUTH_MODE": "challenge"}, clear=False):
            asyncio.run(adapter._check_inbox())

        adapter._fetch_new_messages.assert_called_once()


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
        from tools.send_message_tool import _send_email

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
        from tools.send_message_tool import _send_email

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
        from tools.send_message_tool import _send_email

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
        from gateway.platforms.email import EmailAdapter
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
        from gateway.platforms.email import EmailAdapter
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
            from gateway.platforms.email import EmailAdapter
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
        from gateway.platforms.email import _send_imap_id

        mock_imap = MagicMock()
        mock_imap.xatom.side_effect = Exception("BAD command unknown: ID")

        _send_imap_id(mock_imap)
        mock_imap.xatom.assert_called_once()


if __name__ == "__main__":
    unittest.main()
