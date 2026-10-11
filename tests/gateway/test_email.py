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
from unittest.mock import patch, MagicMock, ANY



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
        # Unquoted forms strict parseaddr rejects still resolve to their single bracketed address.
        for raw in ("john@example.com <john@example.com>", "Doe, John <John@example.com>",
                    "Doe, John (CEO) <John@example.com>"):
            self.assertEqual(_extract_email_address(raw), "john@example.com", raw)
        # ...but never when the display part could hold another mailbox, group or comment.
        for raw in ("a@x.com, b@y.com", "Group: a@x.com, b@y.com;", "<>", "a@x.test <john@example.com>",
                    "attacker@evil.test, <victim@x>", "attacker@evil.test,\r\n <victim@x>",
                    "attacker@evil.test (c) <victim@x>", "attacker@evil.test; <victim@x>",
                    "Grp: attacker@evil.test; <victim@x>", "undisclosed-recipients:; <victim@x>",
                    "John", 'a\\"b <victim@x>',
                    # over _MAX_FROM_LEN (uncapped parseaddr would return a@example.com)
                    "x" * 3000 + " <a@example.com>",
                    # >=500 nested comments make stdlib parseaddr raise RecursionError
                    "Doe, " + "(" * 500 + ")" * 500 + " <v@example.com>"):
            self.assertEqual(_extract_email_address(raw), "", raw)
        # A From with no usable address is dropped at parse time, before dispatch.
        from plugins.platforms.email.adapter import EmailAdapter
        self.assertIsNone(EmailAdapter._parse_fetched_message(
            object.__new__(EmailAdapter), b"2", b"From: a@x.com, b@y.com\r\nSubject: x\r\n\r\nbody"))

    def test_extract_email_address_ignores_angle_brackets_in_display_name(self):
        from plugins.platforms.email.adapter import _extract_email_address
        self.assertEqual(
            _extract_email_address('"Victim <victim@example.com>" <attacker@evil.test>'),
            "attacker@evil.test",
        )

    def test_extract_email_address_unfolds_display_name_before_parsing(self):
        from plugins.platforms.email.adapter import _extract_email_address
        self.assertEqual(
            _extract_email_address(
                '"Some Very Long Display Name That Exceeds The Line\r\n Limit" <real@example.com>'
            ),
            "real@example.com",
        )
        # Ordinary forms must keep resolving the same way.
        self.assertEqual(_extract_email_address("Plain <user@example.com>"), "user@example.com")
        self.assertEqual(_extract_email_address("bare@example.com"), "bare@example.com")


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

        adapter._message_handler = mock_handler
        # Override handle_message to capture the event directly

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
            "sender_authenticated": True,
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
            "sender_authenticated": True,
        }

        asyncio.run(adapter._dispatch_message(msg_data))
        self.assertEqual(len(captured_events), 1)
        self.assertNotIn("[Subject:", captured_events[0].text)
        self.assertEqual(captured_events[0].text, "Thanks for the help!")


    def test_image_attachment_sets_photo_type(self):
        """Email with image attachment should set message type to PHOTO."""
        import asyncio
        from gateway.platforms.event import MessageType
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
            "sender_authenticated": True,
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


class TestDispatchDefersToGatewayAuthorization(unittest.TestCase):
    """The pre-dispatch gate must not drop mail the gateway would authorize (GATEWAY_ALLOWED_USERS,
    an approved pairing) or answer itself (an explicit pair/decline unauthorized_dm_behavior)."""

    STRANGER = "stranger@example.com"

    def setUp(self):
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for key in ("EMAIL_ALLOWED_USERS", "EMAIL_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS",
                    "GATEWAY_ALLOW_ALL_USERS", "EMAIL_TRUST_FROM_HEADER"):
            os.environ.pop(key, None)

    def tearDown(self):
        self._env.stop()

    def _reached_gateway(self, *, extra=None, env=None, paired=False, authenticated=True, auth_reason=None):
        """Dispatch one mail from STRANGER with the real GatewayRunner auth callback wired, as startup does;
        return the events handed to the gateway. Each call gets its own pairing store."""
        import asyncio
        import tempfile
        from pathlib import Path
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.pairing import PairingStore
        from gateway.run import GatewayRunner
        from plugins.platforms.email.adapter import _NO_AUTH_RESULTS_REASON, EmailAdapter
        with tempfile.TemporaryDirectory() as pairing_dir, \
                patch("gateway.pairing.PAIRING_DIR", Path(pairing_dir)), \
                patch.dict(os.environ, {"EMAIL_ADDRESS": "hermes@test.com", "EMAIL_PASSWORD": "secret",
                                        "EMAIL_IMAP_HOST": "imap.test.com", "EMAIL_SMTP_HOST": "smtp.test.com",
                                        **(env or {})}):
            adapter = EmailAdapter(PlatformConfig(enabled=True, extra=dict(extra or {})))
            runner = object.__new__(GatewayRunner)
            runner.config = GatewayConfig(platforms={Platform.EMAIL: adapter.config})
            runner.adapters = {Platform.EMAIL: adapter}
            runner.pairing_store = PairingStore()
            adapter.set_authorization_check(runner._make_adapter_auth_check(Platform.EMAIL))
            if paired:
                code = runner.pairing_store.generate_code("email", self.STRANGER, "Stranger")
                self.assertIsNotNone(runner.pairing_store.approve_code("email", code))
            captured = []

            async def capture(event):
                captured.append(event)

            adapter.handle_message = capture
            asyncio.run(adapter._dispatch_message({
                "uid": b"301", "sender_addr": self.STRANGER, "sender_name": "Stranger", "subject": "Hello",
                "message_id": "<m301@example.com>", "in_reply_to": "", "body": "Hi there", "attachments": [],
                "date": "", "sender_authenticated": authenticated,
                "auth_reason": auth_reason or ("dmarc=pass" if authenticated else _NO_AUTH_RESULTS_REASON)}))
        return captured

    def test_only_a_missing_auth_results_header_warns_with_the_opt_out_hint(self):
        """A granted sender's mail with no Authentication-Results suggests a server that never stamps it, so the drop
        warns with the opt-out hint; no stamp from the pinned authserv-id warns to check authserv_id; a listed sender's
        failing verdict warns without a hint; forged stranger mail under open access stays at debug. A missing
        authserv_id pin warns once per account from connect(), including when the account first comes up via a
        reconnect after a failed initial connect."""
        import asyncio
        from gateway.config import PlatformConfig
        from plugins.platforms.email.adapter import _MISSING_AUTHSERV_REASON, _UNTRUSTED_AUTHSERV_REASON, EmailAdapter

        adapter_log = "plugins.platforms.email.adapter"
        with self.assertLogs(adapter_log, level="WARNING") as logs:
            self.assertEqual(self._reached_gateway(authenticated=False, env={"EMAIL_ALLOW_ALL_USERS": "true"}), [])
        self.assertIn("require_authenticated_sender: false", logs.output[0])
        with self.assertLogs(adapter_log, level="WARNING") as logs:
            self.assertEqual(self._reached_gateway(authenticated=False, auth_reason=_UNTRUSTED_AUTHSERV_REASON,
                                                   env={"EMAIL_ALLOWED_USERS": self.STRANGER}), [])
        self.assertIn("authserv_id", logs.output[0])
        self.assertNotIn("require_authenticated_sender", logs.output[0])
        # No pin drops every message of the account, so connect() names the fix (pin, or the explicit opt-out) once
        # per account: the first successful connect warns even when it is a reconnect after a failed startup, later
        # reconnects do not, each unpinned mailbox warns once, pinned/opted-out stay quiet.
        EmailAdapter._missing_pin_warned.clear()

        def connect(address, extra=None, is_reconnect=False, fail=False):
            with patch.dict(os.environ, {"EMAIL_ADDRESS": address, "EMAIL_PASSWORD": "secret",
                                         "EMAIL_IMAP_HOST": "imap.test.com", "EMAIL_SMTP_HOST": "smtp.test.com"}):
                adapter = EmailAdapter(PlatformConfig(enabled=True, extra=extra or {}))
            imap = MagicMock()
            imap.uid.return_value = ("OK", [b""])
            imap_patch = patch("imaplib.IMAP4_SSL", side_effect=OSError("down")) if fail else patch("imaplib.IMAP4_SSL", return_value=imap)
            with imap_patch, patch.object(adapter, "_connect_smtp"):
                self.assertEqual(asyncio.run(adapter.connect(is_reconnect=is_reconnect)), not fail)

        with self.assertLogs(adapter_log, level="WARNING") as logs:
            connect("one@test.com", fail=True)
            connect("one@test.com", is_reconnect=True)
            connect("one@test.com", is_reconnect=True)
            connect("two@test.com")
            connect("pinned@test.com", {"authserv_id": "mx.ourserver.com"})
            connect("optout@test.com", {"require_authenticated_sender": False})
        logs.output[:] = [line for line in logs.output if _MISSING_AUTHSERV_REASON in line]
        self.assertEqual([line.split(": ")[0] for line in logs.output],
                         ["WARNING:plugins.platforms.email.adapter:[Email] one@test.com",
                          "WARNING:plugins.platforms.email.adapter:[Email] two@test.com"], logs.output)
        self.assertIn(_MISSING_AUTHSERV_REASON, logs.output[1])
        self.assertIn("EMAIL_AUTHSERV_ID", logs.output[1])
        self.assertIn("EMAIL_TRUST_FROM_HEADER=true", logs.output[1])
        with self.assertNoLogs(adapter_log, level="WARNING"):
            self.assertEqual(self._reached_gateway(authenticated=False, auth_reason="dmarc=fail",
                                                   env={"EMAIL_ALLOW_ALL_USERS": "true"}), [])
        # A listed contact's failing mail (broken DKIM, a forwarder) is worth seeing, but the opt-out is wrong advice.
        with self.assertLogs(adapter_log, level="WARNING") as logs:
            self.assertEqual(self._reached_gateway(authenticated=False, auth_reason="dmarc=fail",
                                                   env={"EMAIL_ALLOWED_USERS": self.STRANGER}), [])
        self.assertIn("dmarc=fail", logs.output[0])
        self.assertNotIn("require_authenticated_sender", logs.output[0])

    def test_mail_the_gateway_admits_or_answers_reaches_it(self):
        cases = {
            "pair opt-in": {"extra": {"unauthorized_dm_behavior": "pair"}},
            "decline opt-in": {"extra": {"unauthorized_dm_behavior": "decline"}},
            "GATEWAY_ALLOWED_USERS": {"env": {"GATEWAY_ALLOWED_USERS": self.STRANGER}},
            "EMAIL_ALLOWED_USERS JSON list literal": {"env": {"EMAIL_ALLOWED_USERS": f'["{self.STRANGER}"]'}},
            "approved pairing": {"paired": True},
            # Open access admits any sender whose From: authenticates, or any From: once the operator opts out.
            "allow-all, authenticated From": {"env": {"EMAIL_ALLOW_ALL_USERS": "true"}},
            "allow-all, EMAIL_TRUST_FROM_HEADER=true, unauthenticated From": {
                "authenticated": False, "env": {"EMAIL_ALLOW_ALL_USERS": "true", "EMAIL_TRUST_FROM_HEADER": "true"}},
            "allow-all, require_authenticated_sender: false, unauthenticated From": {
                "authenticated": False, "env": {"EMAIL_ALLOW_ALL_USERS": "true"},
                "extra": {"require_authenticated_sender": False}},
        }
        for label, kwargs in cases.items():
            with self.subTest(label):
                self.assertEqual(len(self._reached_gateway(**kwargs)), 1)

    def test_mail_the_gateway_would_ignore_or_that_forges_from_is_dropped(self):
        cases = {
            "default ignore": {},
            "pair opt-in, unauthenticated From": {"extra": {"unauthorized_dm_behavior": "pair"}, "authenticated": False},
            "approved pairing, unauthenticated From": {"paired": True, "authenticated": False},
            # Open access admits any sender, not any From:: a forged one would land in that address's session.
            "EMAIL_ALLOW_ALL_USERS, unauthenticated From": {"authenticated": False, "env": {"EMAIL_ALLOW_ALL_USERS": "true"}},
            "GATEWAY_ALLOW_ALL_USERS, unauthenticated From": {
                "authenticated": False, "env": {"GATEWAY_ALLOW_ALL_USERS": "true"}},
            # Open access grants a stranger nothing beside a list, so a pairing code must not go to a forged From:.
            "pair opt-in, allow-all beside EMAIL list, unauthenticated From": {
                "extra": {"unauthorized_dm_behavior": "pair"}, "authenticated": False,
                "env": {"GATEWAY_ALLOW_ALL_USERS": "true", "EMAIL_ALLOWED_USERS": "boss@example.com"}},
            # GATEWAY_ALLOW_ALL_USERS is inert beside a list, so a listed address still has to authenticate its From:.
            "listed sender, GATEWAY allow-all beside the list, unauthenticated From": {
                "authenticated": False, "env": {"GATEWAY_ALLOW_ALL_USERS": "true", "EMAIL_ALLOWED_USERS": self.STRANGER}},
            "pair opt-in, allow-all beside GATEWAY list, unauthenticated From": {
                "extra": {"unauthorized_dm_behavior": "pair"}, "authenticated": False,
                "env": {"GATEWAY_ALLOW_ALL_USERS": "true", "GATEWAY_ALLOWED_USERS": "boss@example.com"}},
            # A bare entry (a chat username, say) names one principal, never stranger@<any domain>: the
            # domain is the sender's to choose, so such mail is dropped rather than admitted or paired.
            "GATEWAY_ALLOWED_USERS bare entry": {"env": {"GATEWAY_ALLOWED_USERS": "stranger"}},
            "EMAIL_ALLOWED_USERS bare entry, JSON list literal": {"env": {"EMAIL_ALLOWED_USERS": '["stranger"]'}},
            "bare entry, pair opt-in": {"env": {"GATEWAY_ALLOWED_USERS": "stranger"},
                                        "extra": {"unauthorized_dm_behavior": "pair"}},
        }
        for label, kwargs in cases.items():
            with self.subTest(label):
                self.assertEqual(self._reached_gateway(**kwargs), [])


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


    def test_send_document_threads_on_explicit_reply_to(self):
        """An explicit reply_to wins over the cached thread context for attachment sends (#10131)."""
        import asyncio
        import tempfile
        adapter = self._make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Old", "message_id": "<cached@test.com>"}
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"doc")
            tmp_path = f.name
        try:
            with patch("smtplib.SMTP") as mock_smtp:
                mock_server = MagicMock()
                mock_smtp.return_value = mock_server
                result = asyncio.run(adapter.send_document("user@test.com", tmp_path, reply_to="<explicit@test.com>"))
                self.assertTrue(result.success)
                sent_msg = mock_server.send_message.call_args[0][0]
                self.assertEqual(sent_msg["In-Reply-To"], "<explicit@test.com>")
                self.assertEqual(sent_msg["References"], "<explicit@test.com>")
        finally:
            os.unlink(tmp_path)



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
            results = adapter._fetch_new_messages(lambda _c: True)

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
        # _check_inbox gates every UID on _sender_accepted before the RFC822 fetch; authorize the
        # unauthenticated test senders so these tests exercise the fetch/dispatch loop itself.
        adapter._is_sender_authorized = lambda *a, **k: True
        adapter._require_authenticated_sender = False
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

    def test_rejected_sender_is_not_full_fetched_before_authorization(self):
        """A header-only, bounded preflight rejects a forged lower auth result
        before IMAP returns the MIME body or an attachment is decoded."""
        import asyncio
        from plugins.platforms.email.adapter import _MAX_PREAUTH_HEADER_BYTES

        adapter = self._make_adapter()
        adapter._require_authenticated_sender = True
        adapter._authserv_id = "mx.trusted.test"
        dispatched, fetch_specs = [], []

        async def capture(msg_data):
            dispatched.append(msg_data)

        adapter._dispatch_message = capture
        forged_headers = (
            b"From: Owner <owner@example.com>\r\n"
            b"Subject: forged lower result\r\n"
            b"Message-ID: <forged@example.com>\r\n"
            b"Authentication-Results: edge.receiver.test; dmarc=fail header.from=example.com\r\n"
            b"Authentication-Results: mx.trusted.test; dmarc=pass header.from=example.com\r\n"
            b"Content-Type: multipart/mixed; boundary=x\r\n\r\n"
        )
        trusted_headers = (
            b"From: Owner <owner@example.com>\r\n"
            b"Subject: trusted result\r\n"
            b"Message-ID: <trusted@example.com>\r\n"
            b"Authentication-Results: mx.trusted.test; dmarc=pass header.from=example.com\r\n"
            b"Content-Type: text/plain\r\n\r\n"
        )
        oversized_headers = (
            b"From: Owner <owner@example.com>\r\nAuthentication-Results: mx.trusted.test; "
            + b"x" * _MAX_PREAUTH_HEADER_BYTES
        )
        headers = {b"9": forged_headers, b"10": trusted_headers, b"11": oversized_headers}
        full_messages = {
            b"9": forged_headers + (
                b"--x\r\nContent-Disposition: attachment; filename=payload.bin\r\n\r\n"
                + b"x" * 200_000 + b"\r\n--x--\r\n"
            ),
            b"10": trusted_headers + b"authorized body",
            b"11": oversized_headers + b"unauthorized body",
        }
        mock_imap = MagicMock()

        def uid_handler(command, *args):
            if command == "search":
                return ("OK", [b"9 10 11"])
            if command == "fetch":
                uid, spec = args
                fetch_specs.append((uid, spec))
                payload = headers[uid] if "BODY.PEEK[HEADER.FIELDS" in spec else full_messages[uid]
                return ("OK", [(uid, payload)])
            if command == "store":
                return ("OK", [b""])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler
        gates, fetch = [], adapter._fetch_new_messages
        adapter._fetch_new_messages = lambda gate: (gates.append(gate), fetch(gate))[1]
        with patch.dict(os.environ, {
            "EMAIL_ALLOWED_USERS": "owner@example.com",
            "EMAIL_ALLOW_ALL_USERS": "",
            "GATEWAY_ALLOWED_USERS": "",
            "GATEWAY_ALLOW_ALL_USERS": "",
        }, clear=False), patch("imaplib.IMAP4_SSL", return_value=mock_imap), patch(
            "plugins.platforms.email.adapter._extract_attachments", return_value=[],
        ) as extract_attachments:
            asyncio.run(adapter._check_inbox())

        self.assertEqual([message["message_id"] for message in dispatched], ["<trusted@example.com>"])
        self.assertNotIn((b"9", "(RFC822)"), fetch_specs)
        self.assertNotIn((b"11", "(RFC822)"), fetch_specs)
        self.assertIn((b"10", "(RFC822)"), fetch_specs)
        header_specs = [spec for _, spec in fetch_specs if "BODY.PEEK[HEADER.FIELDS" in spec]
        self.assertEqual(len(header_specs), 3)
        self.assertTrue(all(f"<0.{_MAX_PREAUTH_HEADER_BYTES + 1}>" in spec for spec in header_specs))
        extract_attachments.assert_called_once()
        self.assertTrue({b"9", b"10", b"11"}.issubset(adapter._seen_uids))
        # Once the poll's loop is closed (stop race) the gate fails closed instead of raising.
        self.assertFalse(gates[0](dispatched[0]))

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
                if args[0] == b"1":
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
                if args[0] == b"1":
                    return ("OK", [(b"1", raw_email.as_bytes())])
                raise OSError("connection dropped")
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            results = adapter._fetch_new_messages(lambda _c: True)

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
                if "BODY.PEEK" in args[1]:
                    return ("OK", [(uid, good_email.as_bytes())])
                if uid == b"1":
                    return ("OK", [(b"1", b"poison")])
                return ("OK", [(b"2", good_email.as_bytes())])
            return ("NO", [])

        mock_imap.uid.side_effect = uid_handler

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap), patch(
            "plugins.platforms.email.adapter.EmailAdapter._parse_fetched_message",
            side_effect=[ValueError("unparseable"), {"subject": "good"}],
        ):
            results = adapter._fetch_new_messages(lambda _c: True)

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
            from agent.i18n import t
            self.assertEqual(send_call["Subject"], t("platform.email.standalone_subject"))
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
            results = adapter._fetch_new_messages(lambda _c: True)

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
        mock_imap.capabilities = ("IMAP4REV1", "ID", "UIDPLUS")
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

        with patch("smtplib.SMTP", side_effect=TimeoutError("timed out")), \
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

        with patch("smtplib.SMTP_SSL", side_effect=TimeoutError("timed out")), \
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

    def _verify(self, from_addr, auth_results=None, authserv_id="mx.ourserver.com"):
        """Rows stamp the pinned receiver's id, so a verdict is judged on its content, not on a pin mismatch."""
        from plugins.platforms.email.adapter import (
            _verify_sender_authentication,
            _extract_email_address,
        )
        msg = self._msg(from_addr, auth_results)
        addr = _extract_email_address(from_addr)
        return _verify_sender_authentication(msg, addr, authserv_id=authserv_id)

    def test_auth_results_verdicts(self):
        ok, reason = self._verify(
            "Admin <admin@example.com>",
            ["mx.ourserver.com; dmarc=pass header.from=example.com; spf=pass"],
        )
        self.assertTrue(ok, reason)
        # A dmarc=pass issued for another domain must not vouch for this From,
        # even when a later dkim clause carries an aligned header.from.
        # Verdict and header.from are read from the one dmarc clause, with (comments) stripped first.
        for ar in ("mx.ourserver.com; dmarc=pass header.from=evil.test",
                   "mx.ourserver.com; dmarc=pass header.from=evil.test; dkim=pass header.d=x.test header.from=example.com",
                   "mx.ourserver.com; dmarc=pass (p=none; sp=none) header.from=evil.test",
                   "mx.ourserver.com; dmarc=fail header.from=example.com; dmarc=pass header.from=evil.test",
                   # every header.from in the dmarc clause must align, not just one
                   "mx.ourserver.com; dmarc=pass header.from=evil.test header.from=example.com",
                   # ';' inside quoted-strings / nested comments must not split or smuggle a dmarc clause
                   'mx.ourserver.com; spf=pass smtp.mailfrom="x;dmarc=pass header.from=example.com x"@evil.test; '
                   "dmarc=fail header.from=example.com",
                   "mx.ourserver.com; dmarc=pass (a (b) ; header.from=example.com) header.from=evil.test",
                   'mx.ourserver.com; dmarc=pass reason="a;b" header.from=evil.test',
                   "mx.ourserver.com; dmarc=pass a) ; header.from=evil.test",  # stray ')' is unbalanced
                   "mx.ourserver.com; dmarc=pass header.from=example.com; dmarc=pass header.from=evil.test",
                   "mx.ourserver.com; dmarc=pass (a ; header.from=evil.test",
                   r'mx.ourserver.com; spf=pass smtp.mailfrom="x\\";dmarc=pass header.from=example.com;x="y"; '
                   "dmarc=fail header.from=example.com",
                   # spf/dkim verdicts and domains come only from their own clause, never quoted text or comments
                   'mx.ourserver.com; spf=fail smtp.mailfrom="x spf=pass smtp.mailfrom=example.com "@evil.test; '
                   "dmarc=fail header.from=example.com",
                   "mx.ourserver.com; spf=fail (spf=pass) smtp.mailfrom=a@example.com",
                   "mx.ourserver.com; spf=fail smtp.mailfrom=a.spf=pass@example.com; dmarc=fail header.from=example.com",
                   'mx.ourserver.com; dkim=pass header.d=evil.test header.i="x header.d=example.com y"@evil.test',
                   "mx.ourserver.com; spf=pass smtp.mailfrom=example.com; spf=fail smtp.mailfrom=evil.test",
                   "mx.ourserver.com; spf=fail smtp.mailfrom=evil.test; spf=pass smtp.mailfrom=example.com",
                   "mx.ourserver.com; dkim=pass header.d=evil.test; dkim=fail header.d=example.com",
                   'mx.ourserver.com; dkim=pass header.i="x header.d=example.com"@evil.test',
                   # an escaped quote keeps the quoted-string open, so no dmarc clause is smuggled out of it
                   r'mx.ourserver.com; spf=fail smtp.mailfrom="a\";dmarc=pass header.from=example.com;x=\""@evil.test'):
            ok, reason = self._verify("Admin <admin@example.com>", [ar])
            self.assertFalse(ok, ar)
        # Real MTA headers (multi-signature DKIM, comments, quoted values) keep authenticating.
        for ar in ("mx.ourserver.com; arc=pass (dmarc=fail header.from=evil.test); dmarc=pass header.from=example.com",
                   'mx.ourserver.com; dmarc=pass reason="a;b" header.from="example.com"',
                   'mx.ourserver.com; dmarc=pass reason="header.from=evil.test" header.from=example.com',
                   "mx.ourserver.com; dkim=pass header.i=@example.com header.s=s1 header.b=AbC; spf=pass (google.com: "
                   "domain of admin@example.com designates 1.2.3.4 as permitted sender) smtp.mailfrom=admin@example.com; "
                   "dmarc=pass (p=REJECT sp=REJECT dis=NONE) header.from=example.com",
                   "mx.ourserver.com; spf=pass (sender IP is 1.2.3.4) smtp.mailfrom=example.com; dkim=pass (signature was verified) "
                   "header.d=example.com;dmarc=pass action=none header.from=example.com;compauth=pass reason=100",
                   "mx.ourserver.com; dmarc=pass (p=none dis=none) header.from=example.com",
                   'mx.ourserver.com; dkim=pass (2048-bit key; unprotected) header.d=example.com header.i=@example.com '
                   'header.b="AbC+/1"; spf=pass smtp.mailfrom=example.com',
                   "mx.ourserver.com; dkim=pass (1024-bit key) header.d=esp.test header.i=@esp.test; "
                   "dkim=pass (2048-bit key) header.d=example.com header.i=@example.com; spf=softfail "
                   "smtp.mailfrom=bounce@esp.test"):
            ok, reason = self._verify("Admin <admin@example.com>", [ar])
            self.assertTrue(ok, (ar, reason))


    def test_dkim_pass_aligned_authenticates(self):
        ok, reason = self._verify(
            "admin@example.com",
            ["mx.ourserver.com; dkim=pass header.d=example.com"],
        )
        self.assertTrue(ok, reason)

    def test_spf_pass_misaligned_rejected(self):
        # SPF passes for the envelope domain, but it doesn't match From: domain.
        ok, reason = self._verify(
            "admin@example.com",
            ["mx.ourserver.com; spf=pass smtp.mailfrom=bounce@evil.com"],
        )
        self.assertFalse(ok, reason)


    def test_only_topmost_exact_authserv_id_is_trusted(self):
        """Never search below the authoritative field or relax an authserv-id pin.
        The subdomain and comment-smuggled rows carry dmarc=pass, so only the exact pin rejects them."""
        from plugins.platforms.email.adapter import _MISSING_AUTHSERV_REASON

        forged_lower = "mx.ourserver.com; dmarc=pass header.from=example.com"
        for topmost in (
            "mx.ourserver.com; dmarc=fail header.from=example.com",
            "edge.receiver.test; dmarc=fail header.from=example.com",
            "child.mx.ourserver.com; dmarc=pass header.from=example.com",
            "(mx.ourserver.com) edge.receiver.test; dmarc=pass header.from=example.com",
        ):
            with self.subTest(topmost=topmost):
                ok, reason = self._verify(
                    "admin@example.com", [topmost, forged_lower],
                    authserv_id="mx.ourserver.com",
                )
                self.assertFalse(ok, reason)
        # No pin at all: the topmost header may be one the sender wrote (a self-hosted MTA that stamps nothing).
        ok, reason = self._verify("owner@allowed.example", ["attacker.self; dmarc=pass header.from=allowed.example"],
                                  authserv_id="")
        self.assertEqual((ok, reason), (False, _MISSING_AUTHSERV_REASON))

        # Matching remains case-insensitive and accepts RFC 8601 CFWS comments,
        # including a semicolon inside a nested comment.
        ok, reason = self._verify(
            "admin@example.com",
            ["MX.OURSERVER.COM (receiver (a;b)); dmarc=pass header.from=example.com"],
            authserv_id="mx.ourserver.com",
        )
        self.assertTrue(ok, reason)


def test_oversized_cron_output_is_delivered_as_one_whole_email():
    """No 4000-char truncation footer pointing at a file on the gateway host: the router hands
    the whole cron payload to the email adapter, which sends it as a single message."""
    import asyncio
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.delivery import DeliveryRouter

    with patch.dict(os.environ, {"EMAIL_ADDRESS": "hermes@test.com", "EMAIL_PASSWORD": "secret",
                                 "EMAIL_IMAP_HOST": "imap.test.com", "EMAIL_SMTP_HOST": "smtp.test.com"}):
        from plugins.platforms.email.adapter import EmailAdapter
        adapter = EmailAdapter(PlatformConfig(enabled=True))
    sent = []
    smtp = MagicMock()
    smtp.send_message.side_effect = lambda msg: sent.append(msg.get_payload()[0].get_payload(decode=True).decode())
    adapter._connect_smtp = lambda: smtp
    content = "\n\n".join(f"line {i} " + "x" * 200 for i in range(60))
    payload = DeliveryRouter(GatewayConfig())._cap_oversized_output(adapter, content, "job")
    result = asyncio.run(adapter.send("user@test.com", payload))

    assert result.success
    assert sent == [content]


class TestSentFolderAppend(unittest.TestCase):
    """Tests for the ``platforms.email.sent_folder`` / _append_to_sent feature."""

    _BASE_ENV = {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_IMAP_PORT": "993",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_SMTP_PORT": "587",
    }

    def _make_adapter(self, extra=None):
        """Build an EmailAdapter; ``extra`` maps to config.yaml ``platforms.email``."""
        from gateway.config import PlatformConfig
        with patch.dict(os.environ, self._BASE_ENV, clear=False):
            from plugins.platforms.email.adapter import EmailAdapter
            adapter = EmailAdapter(PlatformConfig(enabled=True, extra=extra or {}))
        return adapter

    def setUp(self):
        """Drop the process-wide CREATE bookkeeping so tests stay independent."""
        from plugins.platforms.email import adapter as adapter_mod
        adapter_mod._SENT_FOLDER_CREATE_ATTEMPTED.clear()

    @staticmethod
    def _ok_imap():
        """A mock IMAP connection whose APPEND reports success."""
        mock_imap = MagicMock()
        mock_imap.append.return_value = ("OK", [b"[APPENDUID 1 2] APPEND completed"])
        return mock_imap

    def test_append_to_sent_called_on_send(self):
        """With sent_folder=Sent, send() must APPEND the message to IMAP."""
        import asyncio
        adapter = self._make_adapter({"sent_folder": "Sent"})

        mock_imap = self._ok_imap()

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_smtp.return_value = MagicMock()

            result = asyncio.run(adapter.send("user@test.com", "Hello!"))

        self.assertTrue(result.success)
        # APPEND must have been called once
        mock_imap.append.assert_called_once()
        folder_arg = mock_imap.append.call_args[0][0]
        self.assertEqual(folder_arg, "Sent")
        # The payload (4th positional arg) must be bytes
        payload_arg = mock_imap.append.call_args[0][3]
        self.assertIsInstance(payload_arg, bytes)

    def test_append_to_sent_disabled_when_empty(self):
        """With sent_folder='', no IMAP connection must be opened for APPEND."""
        import asyncio
        adapter = self._make_adapter({"sent_folder": ""})

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL") as mock_imap_cls:
            mock_smtp.return_value = MagicMock()

            result = asyncio.run(adapter.send("user@test.com", "Hello!"))

        self.assertTrue(result.success)
        # No IMAP connection should have been opened for APPEND
        mock_imap_cls.assert_not_called()

    def test_append_default_folder_is_sent(self):
        """When sent_folder is unconfigured, it defaults to "Sent"."""
        import asyncio
        adapter = self._make_adapter()  # no extra config

        mock_imap = self._ok_imap()

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_smtp.return_value = MagicMock()

            result = asyncio.run(adapter.send("user@test.com", "Hello!"))

        self.assertTrue(result.success)
        mock_imap.append.assert_called_once()
        self.assertEqual(mock_imap.append.call_args[0][0], "Sent")

    def test_append_failure_does_not_break_send(self):
        """If IMAP APPEND raises, send() must still return success."""
        import asyncio
        adapter = self._make_adapter({"sent_folder": "Sent"})

        mock_imap = MagicMock()
        mock_imap.append.side_effect = Exception("IMAP quota exceeded")

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_smtp.return_value = MagicMock()

            result = asyncio.run(adapter.send("user@test.com", "Hello!"))

        self.assertTrue(result.success)

    def test_append_non_ok_status_warns_and_does_not_break_send(self):
        """A ``NO``/``BAD`` APPEND status must warn (not silently 'succeed')."""
        import asyncio
        adapter = self._make_adapter({"sent_folder": "Sent"})

        mock_imap = MagicMock()
        # imaplib returns NO/BAD without raising on a rejected APPEND.
        mock_imap.append.return_value = ("NO", [b"Over quota"])

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap), \
             patch("plugins.platforms.email.adapter.logger") as mock_logger:
            mock_smtp.return_value = MagicMock()

            result = asyncio.run(adapter.send("user@test.com", "Hello!"))

        # SMTP send still succeeded — the Sent mirror is best-effort.
        self.assertTrue(result.success)
        mock_imap.append.assert_called_once()
        # The non-OK status must have produced a warning.
        self.assertTrue(mock_logger.warning.called)
        warned = " ".join(str(c.args) for c in mock_logger.warning.call_args_list)
        self.assertIn("NO", warned)

    def test_append_to_sent_for_send_document(self):
        """send_document() must also APPEND the message to the Sent folder."""
        import asyncio
        import tempfile
        adapter = self._make_adapter({"sent_folder": "Sent"})

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Attachment content")
            tmp_path = f.name

        mock_imap = self._ok_imap()

        try:
            with patch("smtplib.SMTP") as mock_smtp, \
                 patch("imaplib.IMAP4_SSL", return_value=mock_imap):
                mock_smtp.return_value = MagicMock()

                result = asyncio.run(
                    adapter.send_document("user@test.com", tmp_path, "See attached")
                )
        finally:
            os.unlink(tmp_path)

        self.assertTrue(result.success)
        mock_imap.append.assert_called_once()
        folder_arg = mock_imap.append.call_args[0][0]
        self.assertEqual(folder_arg, "Sent")
        payload_arg = mock_imap.append.call_args[0][3]
        self.assertIsInstance(payload_arg, bytes)

    # ------------------------------------------------------------------
    # CREATE is issued once, not per outbound mail
    # ------------------------------------------------------------------

    def test_sent_folder_created_once_across_sends(self):
        """CREATE runs on the first send only; later sends just APPEND."""
        import asyncio
        adapter = self._make_adapter({"sent_folder": "Sent"})

        mock_imap = self._ok_imap()

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_smtp.return_value = MagicMock()
            for _ in range(3):
                asyncio.run(adapter.send("user@test.com", "Hello!"))

        self.assertEqual(mock_imap.append.call_count, 3)
        mock_imap.create.assert_called_once_with("Sent")

    def test_sent_folder_create_failure_warns_once(self):
        """A rejected CREATE is surfaced once instead of being swallowed."""
        import asyncio
        adapter = self._make_adapter({"sent_folder": "Sent"})

        mock_imap = self._ok_imap()
        mock_imap.create.side_effect = Exception("permission denied")

        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_smtp.return_value = MagicMock()
            with self.assertLogs("plugins.platforms.email.adapter", level="WARNING") as logs:
                for _ in range(3):
                    asyncio.run(adapter.send("user@test.com", "Hello!"))

        create_warnings = [line for line in logs.output if "CREATE Sent folder" in line]
        self.assertEqual(len(create_warnings), 1, logs.output)
        # The sends themselves are unaffected — APPEND is still attempted
        self.assertEqual(mock_imap.append.call_count, 3)

    # ------------------------------------------------------------------
    # A message that was never sent must never be archived
    # ------------------------------------------------------------------

    def test_no_append_when_smtp_send_fails(self):
        """A failed SMTP send must not leave a copy in Sent.

        The SMTP block is ``try/finally`` (the inner ``except`` only guards
        ``quit()``), so a send failure propagates before the APPEND is
        reached. Pinned by a test so a later refactor to ``try/except``
        cannot silently start archiving unsent mail.
        """
        import asyncio
        import tempfile
        adapter = self._make_adapter({"sent_folder": "Sent"})

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Attachment content")
            tmp_path = f.name

        def _failing_smtp(*args, **kwargs):
            smtp = MagicMock()
            smtp.send_message.side_effect = Exception("550 mailbox unavailable")
            # quit() also fails, so the inner handler's close() path runs too
            smtp.quit.side_effect = Exception("connection already closed")
            return smtp

        try:
            for label, coro_factory in (
                ("send", lambda: adapter.send("user@test.com", "Hello!")),
                (
                    "send_document",
                    lambda: adapter.send_document("user@test.com", tmp_path, "See attached"),
                ),
                (
                    "send_multiple_images",
                    lambda: adapter.send_multiple_images(
                        "user@test.com", [{"path": tmp_path, "caption": "x"}]
                    ),
                ),
            ):
                with self.subTest(path=label):
                    with patch("smtplib.SMTP", side_effect=_failing_smtp), \
                         patch("imaplib.IMAP4_SSL") as mock_imap_cls:
                        asyncio.run(coro_factory())
                    mock_imap_cls.assert_not_called()
        finally:
            os.unlink(tmp_path)


class TestStandaloneSentFolderAppend(unittest.TestCase):
    """The out-of-process ``_standalone_send`` must mirror to the Sent folder
    exactly like the live adapter — it shares the same archival helper.

    Covers the four out-of-process SMTP call sites (cron delivery, ``hermes
    send``, and any send made with no live adapter present).
    """

    _BASE_ENV = {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_SMTP_PORT": "587",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_IMAP_PORT": "993",
    }

    def setUp(self):
        """Drop the process-wide CREATE bookkeeping so tests stay independent."""
        from plugins.platforms.email import adapter as adapter_mod
        adapter_mod._SENT_FOLDER_CREATE_ATTEMPTED.clear()

    def _send(self, extra):
        """Invoke _standalone_send with a SimpleNamespace pconfig."""
        import asyncio
        from types import SimpleNamespace
        from plugins.platforms.email.adapter import _standalone_send
        pconfig = SimpleNamespace(token=None, api_key=None, extra=extra or {})
        return asyncio.run(_standalone_send(pconfig, "user@test.com", "Hello"))

    @staticmethod
    def _ok_imap():
        mock_imap = MagicMock()
        mock_imap.append.return_value = ("OK", [b"[APPENDUID 1 2] APPEND completed"])
        return mock_imap

    @patch.dict(os.environ, _BASE_ENV, clear=False)
    def test_standalone_appends_to_sent_on_success(self):
        """A successful standalone send must APPEND the message to IMAP."""
        mock_imap = self._ok_imap()
        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_smtp.return_value = MagicMock()
            result = self._send({"sent_folder": "Sent"})

        self.assertTrue(result["success"])
        mock_imap.append.assert_called_once()
        self.assertEqual(mock_imap.append.call_args[0][0], "Sent")
        self.assertIsInstance(mock_imap.append.call_args[0][3], bytes)

    @patch.dict(os.environ, _BASE_ENV, clear=False)
    def test_standalone_default_folder_is_sent(self):
        """Unconfigured sent_folder defaults to "Sent" on the standalone path."""
        mock_imap = self._ok_imap()
        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_smtp.return_value = MagicMock()
            result = self._send({})

        self.assertTrue(result["success"])
        mock_imap.append.assert_called_once()
        self.assertEqual(mock_imap.append.call_args[0][0], "Sent")

    @patch.dict(os.environ, _BASE_ENV, clear=False)
    def test_standalone_append_disabled_when_empty(self):
        """sent_folder='' opts out — no IMAP connection is opened."""
        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL") as mock_imap_cls:
            mock_smtp.return_value = MagicMock()
            result = self._send({"sent_folder": ""})

        self.assertTrue(result["success"])
        mock_imap_cls.assert_not_called()

    @patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_SMTP_PORT": "587",
    }, clear=True)
    def test_standalone_append_skipped_without_imap_host(self):
        """No IMAP host configured (SMTP-only standalone) — APPEND no-ops."""
        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL") as mock_imap_cls:
            mock_smtp.return_value = MagicMock()
            result = self._send({"sent_folder": "Sent"})

        self.assertTrue(result["success"])
        mock_imap_cls.assert_not_called()

    @patch.dict(os.environ, _BASE_ENV, clear=False)
    def test_standalone_append_failure_does_not_break_send(self):
        """An APPEND failure must not turn a completed SMTP send into an error."""
        mock_imap = MagicMock()
        mock_imap.append.side_effect = Exception("IMAP quota exceeded")
        with patch("smtplib.SMTP") as mock_smtp, \
             patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            mock_smtp.return_value = MagicMock()
            result = self._send({"sent_folder": "Sent"})

        self.assertTrue(result["success"])
        self.assertNotIn("error", result)

    @patch.dict(os.environ, _BASE_ENV, clear=False)
    def test_standalone_send_runs_off_the_event_loop(self):
        """Neither the SMTP nor the IMAP socket work may run on the loop thread.

        Both are blocking calls with multi-second timeouts; an unreachable
        server would otherwise stall every other task in the process.
        """
        import threading

        main_thread = threading.get_ident()
        seen = {}

        def _record_smtp(*args, **kwargs):
            seen["smtp"] = threading.get_ident()
            return MagicMock()

        def _record_imap(*args, **kwargs):
            seen["imap"] = threading.get_ident()
            return self._ok_imap()

        with patch("smtplib.SMTP", side_effect=_record_smtp), \
             patch("imaplib.IMAP4_SSL", side_effect=_record_imap):
            result = self._send({"sent_folder": "Sent"})

        self.assertTrue(result["success"])
        self.assertNotEqual(seen.get("smtp"), main_thread, "SMTP ran on the event loop")
        self.assertNotEqual(seen.get("imap"), main_thread, "IMAP ran on the event loop")

    @patch.dict(os.environ, _BASE_ENV, clear=False)
    def test_standalone_send_reports_smtp_failure(self):
        """A send that fails inside the executor still surfaces as an error dict."""
        def _failing_smtp(*args, **kwargs):
            smtp = MagicMock()
            smtp.send_message.side_effect = Exception("550 mailbox unavailable")
            return smtp

        with patch("smtplib.SMTP", side_effect=_failing_smtp), \
             patch("imaplib.IMAP4_SSL") as mock_imap_cls:
            result = self._send({"sent_folder": "Sent"})

        self.assertNotIn("success", result)
        self.assertIn("error", result)
        # Nothing was sent, so nothing may be archived
        mock_imap_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
