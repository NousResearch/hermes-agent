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
from typing import Optional
from unittest.mock import patch, MagicMock, AsyncMock

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
        # After PR A, _thread_context is keyed by (sender, message_id).
        ctx = adapter._thread_context.get(("user@test.com", "<original@test.com>"))
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["subject"], "Project question")
        self.assertEqual(ctx["message_id"], "<original@test.com>")
        self.assertIn("last_seen_ts", ctx)

    def test_reply_uses_re_prefix(self):
        """Reply subject should have Re: prefix."""
        import time
        adapter = self._make_adapter()
        adapter._thread_context[("user@test.com", "<original@test.com>")] = {
            "subject": "Project question",
            "message_id": "<original@test.com>",
            "last_seen_ts": time.time(),
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
        import time
        adapter = self._make_adapter()
        adapter._thread_context[("user@test.com", "<reply@test.com>")] = {
            "subject": "Re: Project question",
            "message_id": "<reply@test.com>",
            "last_seen_ts": time.time(),
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

    def test_concurrent_threads_no_clobber(self):
        """Two concurrent inbound emails from the same sender must keep both
        thread contexts distinct so each reply threads to the correct
        Message-ID. Pre-PR-A this clobbered the first thread with the second."""
        import asyncio
        adapter = self._make_adapter()

        async def noop_handle(event):
            pass

        adapter.handle_message = noop_handle

        base = {
            "sender_addr": "alice@example.com",
            "sender_name": "Alice",
            "in_reply_to": "",
            "body": "x",
            "attachments": [],
            "date": "",
        }
        asyncio.run(adapter._dispatch_message({
            **base, "uid": b"1", "subject": "Topic A", "message_id": "<a@x>",
        }))
        asyncio.run(adapter._dispatch_message({
            **base, "uid": b"2", "subject": "Topic B", "message_id": "<b@x>",
        }))

        # Both entries must coexist under distinct tuple keys.
        self.assertEqual(len(adapter._thread_context), 2)
        self.assertIn(("alice@example.com", "<a@x>"), adapter._thread_context)
        self.assertIn(("alice@example.com", "<b@x>"), adapter._thread_context)

        # Explicit thread_id lookup returns the exact entry for each.
        ctx_a = adapter._lookup_thread_context("alice@example.com", "<a@x>")
        ctx_b = adapter._lookup_thread_context("alice@example.com", "<b@x>")
        self.assertEqual(ctx_a["subject"], "Topic A")
        self.assertEqual(ctx_b["subject"], "Topic B")

    def test_thread_context_trim_bounds_memory(self):
        """When _thread_context exceeds the cap, the trimmer keeps only the
        most-recent entries (by insertion order) up to ~half the cap."""
        import time
        adapter = self._make_adapter()
        adapter._thread_context_max = 10  # shrink cap for a fast test

        for i in range(adapter._thread_context_max + 1):
            adapter._thread_context[("alice@example.com", f"<m{i}@x>")] = {
                "subject": f"s{i}",
                "message_id": f"<m{i}@x>",
                "last_seen_ts": time.time() + i,
            }
        adapter._trim_thread_context()

        self.assertLessEqual(len(adapter._thread_context), adapter._thread_context_max)
        # Most-recent entry survives (inserted last).
        self.assertIn(
            ("alice@example.com", f"<m{adapter._thread_context_max}@x>"),
            adapter._thread_context,
        )
        # Oldest entry was dropped.
        self.assertNotIn(("alice@example.com", "<m0@x>"), adapter._thread_context)

    def test_lookup_helper_falls_back_to_latest_when_no_thread_id(self):
        """Without thread_id, the helper returns the most-recent entry from
        this sender. This preserves today's 'reply with sender's latest
        subject' behavior for callers not yet plumbing thread_id."""
        import time
        adapter = self._make_adapter()
        now = time.time()
        adapter._thread_context[("alice@example.com", "<old@x>")] = {
            "subject": "Old",
            "message_id": "<old@x>",
            "last_seen_ts": now - 100,
        }
        adapter._thread_context[("alice@example.com", "<new@x>")] = {
            "subject": "New",
            "message_id": "<new@x>",
            "last_seen_ts": now,
        }
        # Decoy: different sender with a newer timestamp must NOT match.
        adapter._thread_context[("bob@example.com", "<decoy@x>")] = {
            "subject": "Decoy",
            "message_id": "<decoy@x>",
            "last_seen_ts": now + 100,
        }

        ctx = adapter._lookup_thread_context("alice@example.com")
        self.assertEqual(ctx["subject"], "New")

    def test_lookup_helper_returns_empty_for_unknown_sender(self):
        """Helper returns {} when no entry matches."""
        adapter = self._make_adapter()
        self.assertEqual(adapter._lookup_thread_context("nobody@example.com"), {})


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
        import time
        adapter = self._make_adapter()
        adapter._thread_context[("user@test.com", "<m@t>")] = {
            "subject": "Test",
            "message_id": "<m@t>",
            "last_seen_ts": time.time(),
        }

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


class TestSessionKeying(unittest.TestCase):
    """Tests for per-thread email session keying via Gmail's X-GM-THRID.

    PR B opt-in via `platforms.email.extra.session_keying: gmail_thread_id`.
    Default `sender` mode preserves the pre-PR-B single-session-per-sender
    behavior.
    """

    _ENV = {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.gmail.com",
        "EMAIL_SMTP_HOST": "smtp.gmail.com",
    }

    def _make_adapter(self, session_keying: Optional[str] = None):
        from gateway.config import PlatformConfig
        extra = {}
        if session_keying is not None:
            extra["session_keying"] = session_keying
        with patch.dict(os.environ, self._ENV):
            from gateway.platforms.email import EmailAdapter
            return EmailAdapter(PlatformConfig(enabled=True, extra=extra))

    @staticmethod
    def _mock_fetch_response(uid: bytes, thrid: Optional[str], body: bytes):
        """Build the payload that `imap.uid('fetch', ...)` returns for Gmail.

        Real Gmail response when fetching (RFC822 X-GM-THRID) looks like:
            ('OK', [(b'5 (UID 5 X-GM-THRID 1234567890 RFC822 {123}', <body>), b')'])
        """
        header = f"{uid.decode()} (UID {uid.decode()}".encode()
        if thrid is not None:
            header += f" X-GM-THRID {thrid}".encode()
        header += b" RFC822 {" + str(len(body)).encode() + b"}"
        return ("OK", [(header, body), b")"])

    @staticmethod
    def _raw_email(sender: str = "alice@example.com",
                   subject: str = "Hello",
                   message_id: str = "<a@x>") -> bytes:
        msg = MIMEText("body", "plain", "utf-8")
        msg["From"] = sender
        msg["Subject"] = subject
        msg["Message-ID"] = message_id
        return msg.as_bytes()

    # ---------------- default / sender-mode behavior ----------------

    def test_sender_mode_is_default(self):
        adapter = self._make_adapter()
        self.assertEqual(adapter._session_keying, "sender")

    def test_sender_mode_produces_no_thread_id(self):
        """In sender mode, source.thread_id stays None and the session key
        reduces to the chat_id — preserves pre-PR-B behavior."""
        import asyncio
        adapter = self._make_adapter(session_keying="sender")
        captured = []

        async def capture(event):
            captured.append(event)
        adapter.handle_message = capture

        asyncio.run(adapter._dispatch_message({
            "uid": b"1",
            "sender_addr": "alice@example.com",
            "sender_name": "Alice",
            "subject": "Hello",
            "message_id": "<a@x>",
            "in_reply_to": "",
            "body": "Body",
            "attachments": [],
            "date": "",
            "gmail_thread_id": None,
        }))
        self.assertEqual(len(captured), 1)
        self.assertIsNone(captured[0].source.thread_id)

    def test_unknown_mode_falls_back_to_sender(self):
        adapter = self._make_adapter(session_keying="bogus_mode")
        self.assertEqual(adapter._session_keying, "sender")

    # ---------------- gmail_thread_id mode behavior ----------------

    def test_gmail_mode_assigns_gthr_prefixed_thread_id(self):
        """With THRID present, thread_id is `gthr-<digits>` — the prefix
        makes thread IDs distinguishable from message-id fallbacks in logs."""
        import asyncio
        adapter = self._make_adapter(session_keying="gmail_thread_id")
        captured = []

        async def capture(event):
            captured.append(event)
        adapter.handle_message = capture

        asyncio.run(adapter._dispatch_message({
            "uid": b"1",
            "sender_addr": "alice@example.com",
            "sender_name": "Alice",
            "subject": "Hello",
            "message_id": "<a@x>",
            "in_reply_to": "",
            "body": "Body",
            "attachments": [],
            "date": "",
            "gmail_thread_id": "1234567890123456789",
        }))
        self.assertEqual(captured[0].source.thread_id, "gthr-1234567890123456789")

    def test_gmail_mode_without_thrid_falls_back_to_message_id_hash(self):
        """When THRID is absent (edge case — extension somehow off), treat
        this message as its own thread root via a stable hash of its own
        Message-ID."""
        import asyncio
        adapter = self._make_adapter(session_keying="gmail_thread_id")
        captured = []

        async def capture(event):
            captured.append(event)
        adapter.handle_message = capture

        asyncio.run(adapter._dispatch_message({
            "uid": b"1",
            "sender_addr": "alice@example.com",
            "sender_name": "Alice",
            "subject": "Hello",
            "message_id": "<a@x>",
            "in_reply_to": "",
            "body": "Body",
            "attachments": [],
            "date": "",
            "gmail_thread_id": None,
        }))
        tid = captured[0].source.thread_id
        self.assertIsNotNone(tid)
        self.assertTrue(tid.startswith("mid-"))

    def test_gmail_mode_same_thrid_produces_same_thread_id(self):
        """Multiple inbound messages with the same THRID must resolve to the
        same session key — that's the point of the feature."""
        import asyncio
        adapter = self._make_adapter(session_keying="gmail_thread_id")
        captured = []

        async def capture(event):
            captured.append(event)
        adapter.handle_message = capture

        for i, mid in enumerate(["<a@x>", "<b@x>", "<c@x>"]):
            asyncio.run(adapter._dispatch_message({
                "uid": str(i).encode(),
                "sender_addr": "alice@example.com",
                "sender_name": "Alice",
                "subject": "Thread topic",
                "message_id": mid,
                "in_reply_to": "",
                "body": "Body",
                "attachments": [],
                "date": "",
                "gmail_thread_id": "42",
            }))
        tids = {e.source.thread_id for e in captured}
        self.assertEqual(tids, {"gthr-42"})

    def test_gmail_mode_distinct_thrids_produce_distinct_thread_ids(self):
        import asyncio
        adapter = self._make_adapter(session_keying="gmail_thread_id")
        captured = []

        async def capture(event):
            captured.append(event)
        adapter.handle_message = capture

        for uid, mid, thrid in [
            (b"1", "<a@x>", "11"),
            (b"2", "<b@x>", "22"),
        ]:
            asyncio.run(adapter._dispatch_message({
                "uid": uid,
                "sender_addr": "alice@example.com",
                "sender_name": "Alice",
                "subject": "Hello",
                "message_id": mid,
                "in_reply_to": "",
                "body": "Body",
                "attachments": [],
                "date": "",
                "gmail_thread_id": thrid,
            }))
        tids = [e.source.thread_id for e in captured]
        self.assertEqual(tids, ["gthr-11", "gthr-22"])

    # ---------------- capability probe ----------------

    def test_capability_probe_degrades_when_extension_missing(self):
        """When the server doesn't advertise X-GM-EXT-1, the adapter must
        fall back to sender mode rather than silently misroute sessions."""
        import asyncio
        adapter = self._make_adapter(session_keying="gmail_thread_id")
        self.assertEqual(adapter._session_keying, "gmail_thread_id")

        mock_imap = MagicMock()
        # Server without X-GM-EXT-1 — vanilla IMAP4REV1 only.
        mock_imap.capability.return_value = ("OK", [b"IMAP4REV1 STARTTLS AUTH=PLAIN"])
        mock_imap.uid.return_value = ("OK", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.connect())

        self.assertFalse(adapter._has_gmail_ext)
        self.assertEqual(adapter._session_keying, "sender")

        # Clean up the poll task spun up by connect()
        adapter._running = False
        if adapter._poll_task:
            adapter._poll_task.cancel()

    def test_capability_probe_keeps_mode_when_extension_present(self):
        import asyncio
        adapter = self._make_adapter(session_keying="gmail_thread_id")

        mock_imap = MagicMock()
        mock_imap.capability.return_value = ("OK", [b"IMAP4REV1 X-GM-EXT-1 AUTH=PLAIN"])
        mock_imap.uid.return_value = ("OK", [b""])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.connect())

        self.assertTrue(adapter._has_gmail_ext)
        self.assertEqual(adapter._session_keying, "gmail_thread_id")
        adapter._running = False
        if adapter._poll_task:
            adapter._poll_task.cancel()

    # ---------------- THRID parser ----------------

    def test_parse_gm_thrid_extracts_decimal(self):
        from gateway.platforms.email import EmailAdapter
        response = (b"5 (UID 5 X-GM-THRID 1234567890123456789 RFC822 {123}", b"body")
        self.assertEqual(
            EmailAdapter._parse_gm_thrid(response),
            "1234567890123456789",
        )

    def test_parse_gm_thrid_returns_none_when_absent(self):
        from gateway.platforms.email import EmailAdapter
        response = (b"5 (UID 5 RFC822 {123}", b"body")
        self.assertIsNone(EmailAdapter._parse_gm_thrid(response))

    def test_parse_gm_thrid_returns_none_for_malformed_input(self):
        from gateway.platforms.email import EmailAdapter
        self.assertIsNone(EmailAdapter._parse_gm_thrid(None))
        self.assertIsNone(EmailAdapter._parse_gm_thrid("not a tuple"))

    # ---------------- fetch wires THRID through ----------------

    def test_fetch_populates_gmail_thread_id_when_gmail_mode_active(self):
        adapter = self._make_adapter(session_keying="gmail_thread_id")
        adapter._has_gmail_ext = True  # bypass capability probe for unit test

        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"5"])
            if command == "fetch":
                # Verify we asked for X-GM-THRID in addition to RFC822.
                self.assertIn("X-GM-THRID", args[1])
                return self._mock_fetch_response(
                    b"5", "1234567890123456789", self._raw_email(),
                )
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["gmail_thread_id"], "1234567890123456789")

    def test_fetch_skips_extended_items_in_sender_mode(self):
        """In sender mode, the FETCH call must stay on (RFC822) so we don't
        send extra request bytes servers without the Gmail extension would
        reject."""
        adapter = self._make_adapter(session_keying="sender")

        mock_imap = MagicMock()
        seen_items = []

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"5"])
            if command == "fetch":
                seen_items.append(args[1])
                return ("OK", [(b"5 (UID 5", self._raw_email()), b")"])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            adapter._fetch_new_messages()
        self.assertEqual(seen_items, ["(RFC822)"])

    # ---------------- metadata plumbing through send_image / send_document ----------------

    def test_send_image_passes_metadata_through_to_send(self):
        """Regression: pre-PR-B send_image dropped metadata, so image replies
        in gmail_thread_id mode would fall back to the latest-subject heuristic
        instead of threading into the exact thread."""
        import asyncio
        adapter = self._make_adapter(session_keying="gmail_thread_id")
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="<x>"))
        asyncio.run(adapter.send_image(
            "alice@example.com",
            "https://example.com/i.png",
            caption="hi",
            metadata={"thread_id": "gthr-42"},
        ))
        _, kwargs = adapter.send.call_args
        self.assertEqual(kwargs.get("metadata"), {"thread_id": "gthr-42"})

    def test_send_document_forwards_thread_id_to_attachment_sender(self):
        import asyncio
        import tempfile
        adapter = self._make_adapter(session_keying="gmail_thread_id")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"content")
            path = f.name

        with patch.object(adapter, "_send_email_with_attachment",
                          return_value="<x>") as spy:
            asyncio.run(adapter.send_document(
                "alice@example.com",
                path,
                caption="docs",
                metadata={"thread_id": "gthr-99"},
            ))
        # Signature: _send_email_with_attachment(to_addr, body, file_path,
        #                                         file_name, thread_id)
        called_args = spy.call_args[0]
        self.assertEqual(called_args[4], "gthr-99")


if __name__ == "__main__":
    unittest.main()
