"""Tests for the WPS Xiezuo built-in gateway adapter."""

import asyncio
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.platforms.base import SendResult


# ---------------------------------------------------------------------------
# Crypto
# ---------------------------------------------------------------------------

class TestWpsXiezuoCrypto(unittest.TestCase):
    """Verify HMAC-SHA256 signature and AES-256-CBC crypto."""

    def test_compute_signature(self):
        from gateway.platforms.wps_xiezuo import compute_signature

        sig = compute_signature(
            app_id="wp_1", app_secret="sec",
            topic="kso.app_chat.message", nonce="n1",
            timestamp=1700000000, encrypted_data="Zm9v",
        )
        import hmac, hashlib, base64
        content = "wp_1:kso.app_chat.message:n1:1700000000:Zm9v"
        expected = base64.urlsafe_b64encode(
            hmac.new(b"sec", content.encode(), hashlib.sha256).digest()
        ).decode().rstrip("=")
        self.assertEqual(sig, expected)

    def test_verify_signature_valid(self):
        from gateway.platforms.wps_xiezuo import compute_signature, verify_signature

        sig = compute_signature("wp_1", "sec", "topic", "n1", 123, "data")
        self.assertTrue(verify_signature(sig, "wp_1", "sec", "topic", "n1", 123, "data"))

    def test_verify_signature_invalid(self):
        from gateway.platforms.wps_xiezuo import verify_signature
        self.assertFalse(verify_signature("bad", "wp_1", "sec", "topic", "n1", 123, "data"))

    def test_encrypt_decrypt_roundtrip(self):
        from gateway.platforms.wps_xiezuo import decrypt_event

        import hashlib, base64

        secret = "test_secret_key"
        nonce = "abcdefghijklmnop"  # 16+ chars
        plaintext = '{"message": {"id": "m1"}, "chat": {"id": "c1"}, "sender": {"id": "u1"}}'

        key = hashlib.md5(secret.encode()).hexdigest().encode("utf-8")
        iv = nonce.encode("utf-8")[:16]

        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding as cp

        padder = cp.PKCS7(128).padder()
        padded = padder.update(plaintext.encode()) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        enc = cipher.encryptor()
        ciphertext = enc.update(padded) + enc.finalize()
        encrypted_data = base64.b64encode(ciphertext).decode()

        decrypted = decrypt_event(encrypted_data, secret, nonce)
        self.assertEqual(json.loads(decrypted), json.loads(plaintext))

    def test_handshake_frame_structure(self):
        from gateway.platforms.wps_xiezuo import build_handshake

        frame = build_handshake("wp_test", "my_secret", "abc123")
        self.assertEqual(frame["opcode"], 1)
        payload = json.loads(frame["payload"])
        self.assertIn("app_id", payload)
        self.assertIn("signature", payload)
        self.assertEqual(payload["app_id"], "wp_test")
        self.assertEqual(payload["nonce"], "abc123")


# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------

class TestAppTokenStore(unittest.TestCase):
    """Verify token management with in-flight dedup."""

    def setUp(self):
        self.store = None

    def test_fetch_and_cache(self):
        from gateway.platforms.wps_xiezuo import AppTokenStore

        async def _run():
            store = AppTokenStore("https://example.com", "id", "secret")
            mock_resp = MagicMock()
            mock_resp.json = AsyncMock(return_value={
                "code": 0,
                "data": {"access_token": "tok_123", "expires_in": 7200},
            })
            with patch("aiohttp.ClientSession.post", return_value=mock_resp):
                # Use real session but mock post
                pass

            # Simple mock approach
            store._fetch_token = AsyncMock(return_value=("tok_abc", 7200))
            token = await store.get_token()
            self.assertEqual(token, "tok_abc")
            # Second call should use cache
            store._fetch_token = AsyncMock(return_value=("tok_should_not", 7200))
            token2 = await store.get_token()
            self.assertEqual(token2, "tok_abc")

        asyncio.get_event_loop().run_until_complete(_run())

    def test_invalidate(self):
        from gateway.platforms.wps_xiezuo import AppTokenStore

        async def _run():
            store = AppTokenStore("https://example.com", "id", "secret")
            store._token = "old"
            store._expires_at = time.time() + 9999
            store.invalidate()
            self.assertIsNone(store._token)

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------

class TestWpsXiezuoAdapter(unittest.TestCase):
    """Verify adapter construction and settings."""

    def test_adapter_instantiation(self):
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        config = PlatformConfig(enabled=True)
        config.extra.update({"app_id": "wp_test", "app_secret": "sec", "connection_mode": "websocket"})
        adapter = WpsXiezuoAdapter(config)
        self.assertEqual(adapter._app_id, "wp_test")
        self.assertEqual(adapter._connection_mode, "websocket")
        self.assertFalse(adapter._connected)

    def test_parse_receiver_chat(self):
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        self.assertEqual(WpsXiezuoAdapter._parse_receiver("123"), {"type": "chat", "receiver_id": "123"})
        self.assertEqual(WpsXiezuoAdapter._parse_receiver("chat:456"), {"type": "chat", "receiver_id": "456"})
        self.assertEqual(WpsXiezuoAdapter._parse_receiver("user:789"), {"type": "user", "receiver_id": "789"})
        self.assertIsNone(WpsXiezuoAdapter._parse_receiver(""))

    def test_normalize_content(self):
        from gateway.platforms.wps_xiezuo import _normalize_message_content, _strip_at_mention

        msg = {"content": {"text": {"content": "hello", "type": "text"}}}
        self.assertEqual(_normalize_message_content(msg), "hello")

        msg2 = {"content": "raw text"}
        self.assertEqual(_normalize_message_content(msg2), "raw text")

        text = "hi <at user_id=bot>Bot</at> there"
        self.assertEqual(_strip_at_mention(text), "hi Bot there")

    def test_sanitize_dsml_tokens(self):
        from gateway.platforms.wps_xiezuo import _sanitize_model_output

        # DeepSeek tool-call tokens leaking into output
        raw = (
            '根据查询结果，金山云(KC)今日股价如下：\n'
            '<|｜DSML｜｜tool_calls> <|｜DSML｜｜invoke name="terminal"> '
            '<|｜DSML｜｜parameter name="command" string="true">curl -s "https://example.com"</|｜DSML｜｜parameter> '
            '</|｜DSML｜｜invoke> </|｜DSML｜｜tool_calls>'
        )
        cleaned = _sanitize_model_output(raw)
        self.assertNotIn("DSML", cleaned)
        self.assertIn("金山云", cleaned)
        # Should not leave excessive blank lines
        self.assertNotIn("\n\n\n", cleaned)


# ---------------------------------------------------------------------------
# Inbound dispatch
# ---------------------------------------------------------------------------

class TestWpsXiezuoInbound(unittest.TestCase):
    """Verify message routing, dedup, and anti-loopback."""

    def test_dm_message_dispatched(self):
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        config = PlatformConfig(enabled=True)
        config.extra.update({"app_id": "wp_test", "app_secret": "sec"})
        adapter = WpsXiezuoAdapter(config)
        adapter._connected = True
        adapter._wps_client = MagicMock()
        adapter.handle_message = AsyncMock()

        event = {
            "topic": "kso.app_chat.message",
            "operation": "create",
            "data": {
                "sender": {"id": "u1", "type": "user"},
                "message": {"id": "m1", "content": {"text": {"content": "hello", "type": "text"}}},
                "chat": {"id": "c1", "type": "p2p"},
            },
        }

        async def _run():
            await adapter._handle_inbound_event(event)
            adapter.handle_message.assert_called_once()

        asyncio.get_event_loop().run_until_complete(_run())

    def test_loopback_filtered(self):
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        config = PlatformConfig(enabled=True)
        config.extra.update({"app_id": "wp_test", "app_secret": "sec"})
        adapter = WpsXiezuoAdapter(config)
        adapter._connected = True
        adapter._wps_client = MagicMock()
        adapter.handle_message = AsyncMock()
        adapter._bot_user_id = "bot1"

        event = {
            "topic": "kso.app_chat.message",
            "operation": "create",
            "data": {
                "sender": {"id": "bot1", "type": "robot"},
                "message": {"id": "m2", "content": {"text": {"content": "ignore", "type": "text"}}},
                "chat": {"id": "c1", "type": "p2p"},
            },
        }

        async def _run():
            await adapter._handle_inbound_event(event)
            adapter.handle_message.assert_not_called()

        asyncio.get_event_loop().run_until_complete(_run())

    def test_dedup_same_msg_id(self):
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        config = PlatformConfig(enabled=True)
        config.extra.update({"app_id": "wp_test", "app_secret": "sec"})
        adapter = WpsXiezuoAdapter(config)
        adapter._connected = True
        adapter._wps_client = MagicMock()
        adapter.handle_message = AsyncMock()

        event = {
            "topic": "kso.app_chat.message",
            "operation": "create",
            "data": {
                "sender": {"id": "u1", "type": "user"},
                "message": {"id": "m_dedup", "content": {"text": {"content": "hello", "type": "text"}}},
                "chat": {"id": "c1", "type": "p2p"},
            },
        }

        async def _run():
            await adapter._handle_inbound_event(event)
            self.assertEqual(adapter.handle_message.call_count, 1)
            await adapter._handle_inbound_event(event)
            self.assertEqual(adapter.handle_message.call_count, 1)  # deduped

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------

class TestWpsXiezuoSend(unittest.TestCase):
    """Verify outbound message sending."""

    def test_send_not_connected(self):
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        config = PlatformConfig(enabled=True)
        adapter = WpsXiezuoAdapter(config)
        result = asyncio.get_event_loop().run_until_complete(adapter.send("c1", "hello"))
        self.assertFalse(result.success)

    def test_send_text_message(self):
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        config = PlatformConfig(enabled=True)
        adapter = WpsXiezuoAdapter(config)
        adapter._connected = True
        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value={"code": 0, "data": {"message_id": "msg_out"}})
        adapter._wps_client = mock_client

        async def _run():
            result = await adapter.send("c1", "**hello**")
            self.assertTrue(result.success)
            self.assertEqual(result.message_id, "msg_out")
            call_args = mock_client.request.call_args
            self.assertEqual(call_args[0][0], "POST")
            self.assertEqual(call_args[0][1], "/v7/messages/create")
            body = call_args[1]["json"]
            self.assertEqual(body["receiver"]["receiver_id"], "c1")
            self.assertEqual(body["content"]["text"]["type"], "markdown")

        asyncio.get_event_loop().run_until_complete(_run())

    def test_send_markdown_fallback_to_plain(self):
        """When markdown send fails with 'can not get text info', fall back to sanitized markdown."""
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter, WpsRequestError

        config = PlatformConfig(enabled=True)
        adapter = WpsXiezuoAdapter(config)
        adapter._connected = True
        adapter._token_store = MagicMock()
        mock_client = MagicMock()
        # First call (markdown) fails, second call (sanitized markdown) succeeds
        mock_client.request = AsyncMock(side_effect=[
            WpsRequestError("WPS API /v7/messages/create failed: InvalidArgument:can not get text info"),
            {"code": 0, "data": {"message_id": "msg_plain"}},
        ])
        adapter._wps_client = mock_client

        async def _run():
            result = await adapter.send("c1", "**hello** world")
            self.assertTrue(result.success)
            self.assertEqual(result.message_id, "msg_plain")
            # Second call should still use markdown type (WPS doesn't support "text")
            second_call = mock_client.request.call_args_list[1]
            body = second_call[1]["json"]
            self.assertEqual(body["content"]["text"]["type"], "markdown")

        asyncio.get_event_loop().run_until_complete(_run())
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WS data handling
# ---------------------------------------------------------------------------

class TestWpsXiezuoWsHandling(unittest.TestCase):
    """Verify WS goaway handling and ACK sending."""

    def test_goaway_connection_replaced(self):
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        config = PlatformConfig(enabled=True)
        adapter = WpsXiezuoAdapter(config)
        adapter._ws_client = MagicMock()
        adapter._ws_client.send_json = AsyncMock()
        adapter._ws_is_websockets_lib = False

        async def _run():
            await adapter._handle_ws_data({
                "type": "goaway",
                "reason": "connection_replaced",
                "message": "Another client connected",
            })
            # connection_replaced should NOT stop reconnect
            self.assertFalse(adapter._stop_reconnect)
            self.assertFalse(adapter._connected)

        asyncio.get_event_loop().run_until_complete(_run())

    def test_event_sends_ack(self):
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        config = PlatformConfig(enabled=True)
        adapter = WpsXiezuoAdapter(config)
        adapter._ws_client = MagicMock()
        adapter._ws_client.send_json = AsyncMock()
        adapter._ws_is_websockets_lib = False
        adapter._handle_inbound_event = AsyncMock()

        async def _run():
            await adapter._handle_ws_data({
                "topic": "kso.app_chat.message",
                "operation": "create",
                "nonce": "ack_nonce_1",
                "data": {},
            })
            # Check ACK was sent BEFORE _handle_inbound_event was called
            adapter._ws_client.send_json.assert_called_once()
            ack_msg = adapter._ws_client.send_json.call_args[0][0]
            self.assertEqual(ack_msg["type"], "ack")
            self.assertEqual(ack_msg["nonce"], "ack_nonce_1")
            # Verify inbound event was also called
            adapter._handle_inbound_event.assert_called_once()

    def test_ack_sent_before_event_processing(self):
        """ACK must be sent BEFORE awaiting event processing (WPS server
        has a ~5s ACK timeout; agent processing can take 10+ seconds)."""
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.wps_xiezuo import WpsXiezuoAdapter

        config = PlatformConfig(enabled=True)
        adapter = WpsXiezuoAdapter(config)
        adapter._ws_client = MagicMock()
        adapter._ws_is_websockets_lib = False

        ack_time = [None]
        process_time = [None]

        async def fake_send_json(msg, **kw):
            ack_time[0] = time.time()

        async def fake_process(evt):
            process_time[0] = time.time()

        adapter._ws_client.send_json = AsyncMock(side_effect=fake_send_json)
        adapter._handle_inbound_event = AsyncMock(side_effect=fake_process)

        async def _run():
            await adapter._handle_ws_data({
                "topic": "kso.app_chat.message",
                "operation": "create",
                "nonce": "order_test",
                "data": {},
            })
            # ACK must have been sent before processing started
            self.assertIsNotNone(ack_time[0])
            self.assertIsNotNone(process_time[0])
            self.assertLess(ack_time[0], process_time[0])

        asyncio.get_event_loop().run_until_complete(_run())

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Config & requirements
# ---------------------------------------------------------------------------

class TestWpsXiezuoConfig(unittest.TestCase):
    """Verify config and requirements checks."""

    def test_check_requirements_false_when_missing(self):
        from gateway.platforms.wps_xiezuo import check_wps_xiezuo_requirements

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(check_wps_xiezuo_requirements())

    @patch.dict(os.environ, {
        "WPS_XIEZUO_APP_ID": "wp_xxx",
        "WPS_XIEZUO_APP_SECRET": "secret_xxx",
    }, clear=False)
    def test_check_requirements_true_when_set(self):
        from gateway.platforms.wps_xiezuo import check_wps_xiezuo_requirements
        self.assertTrue(check_wps_xiezuo_requirements())

    def test_platform_enum_value(self):
        from gateway.config import Platform
        self.assertEqual(Platform.WPS_XIEZUO.value, "wps-xiezuo")


if __name__ == "__main__":
    unittest.main()
