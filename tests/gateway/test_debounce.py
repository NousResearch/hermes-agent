"""Tests for gateway inbound message debounce."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gateway.config import GatewayConfig, load_gateway_config
from gateway.run import GatewayRunner


class TestDebounceConfig:
    def test_default_debounce_ms_is_zero(self):
        cfg = GatewayConfig()
        assert cfg.inbound_debounce_ms == 0

    def test_debounce_ms_from_dict(self):
        cfg = GatewayConfig.from_dict({"inbound_debounce_ms": 3000})
        assert cfg.inbound_debounce_ms == 3000

    def test_debounce_ms_to_dict(self):
        cfg = GatewayConfig(inbound_debounce_ms=5000)
        d = cfg.to_dict()
        assert d["inbound_debounce_ms"] == 5000

    def test_debounce_ms_roundtrip(self):
        cfg = GatewayConfig(inbound_debounce_ms=1500)
        restored = GatewayConfig.from_dict(cfg.to_dict())
        assert restored.inbound_debounce_ms == 1500

    def test_load_gateway_config_debounce_from_yaml(self, tmp_path, monkeypatch):
        import yaml
        config_path = tmp_path / "config.yaml"
        config_path.write_text("inbound_debounce_ms: 2000\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Patch get_hermes_home to return tmp_path
        import gateway.config as gc
        monkeypatch.setattr(gc, "get_hermes_home", lambda: tmp_path)
        cfg = load_gateway_config()
        assert cfg.inbound_debounce_ms == 2000


class TestDebounceBuffering:
    def _make_runner(self, debounce_ms=500):
        cfg = GatewayConfig(inbound_debounce_ms=debounce_ms)
        with patch("gateway.run.load_gateway_config", return_value=cfg), \
             patch("gateway.run.SessionStore"), \
             patch("gateway.run.DeliveryRouter"), \
             patch("gateway.pairing.PairingStore"), \
             patch("hermes_state.SessionDB", side_effect=Exception("no db")):
            runner = GatewayRunner(config=cfg)
        return runner

    def _make_event(self, text="hello", msg_type=None):
        from gateway.platforms.base import MessageEvent, MessageType, SessionSource, Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")
        return MessageEvent(
            text=text,
            message_type=msg_type or MessageType.TEXT,
            source=source,
            message_id="msg1",
        )

    @pytest.mark.asyncio
    async def test_debounce_disabled_by_default(self):
        runner = self._make_runner(debounce_ms=0)
        event = self._make_event()
        runner.handle_message = AsyncMock(return_value="response")
        # When disabled, messages go through immediately (no buffering)
        assert runner._debounce_tasks == {}
        assert runner._debounce_buffers == {}

    @pytest.mark.asyncio
    async def test_debounce_buffers_message(self):
        runner = self._make_runner(debounce_ms=500)
        event = self._make_event()
        session_key = "telegram:123:u1"

        # Simulate debounce path directly
        runner._debounce_buffers.setdefault(session_key, []).append(event)
        assert len(runner._debounce_buffers[session_key]) == 1

    @pytest.mark.asyncio
    async def test_flush_merges_messages(self):
        runner = self._make_runner(debounce_ms=500)
        from gateway.platforms.base import MessageEvent, MessageType, SessionSource, Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")
        session_key = "telegram:123:u1"

        e1 = MessageEvent(text="part 1", message_type=MessageType.TEXT, source=source, message_id="1")
        e2 = MessageEvent(text="part 2", message_type=MessageType.TEXT, source=source, message_id="2")

        runner._debounce_buffers[session_key] = [e1, e2]
        runner.handle_message = AsyncMock(return_value=None)

        await runner._flush_debounce_buffer(session_key)

        runner.handle_message.assert_called_once()
        merged_event = runner.handle_message.call_args[0][0]
        assert "part 1" in merged_event.text
        assert "part 2" in merged_event.text

    @pytest.mark.asyncio
    async def test_flush_single_message_unchanged(self):
        runner = self._make_runner(debounce_ms=500)
        from gateway.platforms.base import MessageEvent, MessageType, SessionSource, Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")
        session_key = "telegram:123:u1"

        e1 = MessageEvent(text="single message", message_type=MessageType.TEXT, source=source, message_id="1")
        runner._debounce_buffers[session_key] = [e1]
        runner.handle_message = AsyncMock(return_value=None)

        await runner._flush_debounce_buffer(session_key)

        runner.handle_message.assert_called_once()
        sent_event = runner.handle_message.call_args[0][0]
        assert sent_event.text == "single message"

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self):
        runner = self._make_runner(debounce_ms=500)
        from gateway.platforms.base import MessageEvent, MessageType, SessionSource, Platform
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")
        session_key = "telegram:123:u1"

        e1 = MessageEvent(text="msg", message_type=MessageType.TEXT, source=source, message_id="1")
        runner._debounce_buffers[session_key] = [e1]
        runner.handle_message = AsyncMock(return_value=None)

        await runner._flush_debounce_buffer(session_key)

        assert session_key not in runner._debounce_buffers
        assert session_key not in runner._debounce_tasks
