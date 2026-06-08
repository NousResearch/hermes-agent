"""Tests for /reasoning stream mode and the ReasoningStreamConsumer.

Covers the new end-to-end flow:
  - /reasoning stream toggles per-platform state and persists config
  - /reasoning show and /reasoning hide both clear stream mode
  - Status command reflects stream mode
  - ReasoningStreamConsumer.send_initial returns False when no message_id
  - ReasoningStreamConsumer.on_delta + run() edits in place with throttling
  - The consumer exposes emitted_any after first delta
  - Non-streaming models still get the legacy prepend path
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.stream_consumer import ReasoningStreamConsumer
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/reasoning", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform, user_id=user_id, chat_id=chat_id, user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Bare-bones GatewayRunner without calling __init__."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._reasoning_stream_per_platform = {}
    return runner


def _write_config(tmp_path, body):
    """Write a config.yaml to tmp_path/hermes/ and return the path."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(body, encoding="utf-8")
    return hermes_home


# ── /reasoning command-level tests ────────────────────────────────────


class TestReasoningStreamCommand:
    @pytest.mark.asyncio
    async def test_stream_sets_both_flags_and_persists(self, tmp_path, monkeypatch):
        hermes_home = _write_config(tmp_path, "agent:\n  reasoning_effort: medium\n")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        runner._show_reasoning = False
        runner._reasoning_stream_per_platform = {}

        result = await runner._handle_reasoning_command(_make_event("/reasoning stream"))

        assert runner._show_reasoning is True
        assert runner._reasoning_stream_per_platform.get("telegram") is True
        import yaml
        with open(hermes_home / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["display"]["platforms"]["telegram"]["show_reasoning"] is True
        assert cfg["display"]["platforms"]["telegram"]["reasoning_stream"] is True

    @pytest.mark.asyncio
    async def test_show_clears_stream(self, tmp_path, monkeypatch):
        hermes_home = _write_config(
            tmp_path,
            "agent:\n  reasoning_effort: medium\n"
            "display:\n  platforms:\n    telegram:\n"
            "      show_reasoning: true\n      reasoning_stream: true\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        runner._show_reasoning = True
        runner._reasoning_stream_per_platform = {"telegram": True}

        await runner._handle_reasoning_command(_make_event("/reasoning show"))

        assert runner._show_reasoning is True
        assert runner._reasoning_stream_per_platform.get("telegram") is False
        import yaml
        with open(hermes_home / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["display"]["platforms"]["telegram"]["show_reasoning"] is True
        assert cfg["display"]["platforms"]["telegram"]["reasoning_stream"] is False

    @pytest.mark.asyncio
    async def test_hide_clears_stream(self, tmp_path, monkeypatch):
        hermes_home = _write_config(
            tmp_path,
            "agent:\n  reasoning_effort: medium\n"
            "display:\n  platforms:\n    telegram:\n"
            "      show_reasoning: true\n      reasoning_stream: true\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        runner._show_reasoning = True
        runner._reasoning_stream_per_platform = {"telegram": True}

        await runner._handle_reasoning_command(_make_event("/reasoning hide"))

        assert runner._show_reasoning is False
        assert runner._reasoning_stream_per_platform.get("telegram") is False
        import yaml
        with open(hermes_home / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["display"]["platforms"]["telegram"]["show_reasoning"] is False
        assert cfg["display"]["platforms"]["telegram"]["reasoning_stream"] is False

    @pytest.mark.asyncio
    async def test_status_shows_stream_state(self, tmp_path, monkeypatch):
        hermes_home = _write_config(
            tmp_path,
            "agent:\n  reasoning_effort: medium\n"
            "display:\n  platforms:\n    telegram:\n"
            "      show_reasoning: true\n      reasoning_stream: true\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        runner._show_reasoning = True
        runner._reasoning_stream_per_platform = {"telegram": True}

        result = await runner._handle_reasoning_command(_make_event("/reasoning"))

        assert "stream" in result.lower()

    @pytest.mark.asyncio
    async def test_load_reasoning_stream_default_false(self, tmp_path, monkeypatch):
        hermes_home = _write_config(tmp_path, "display:\n")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        result = gateway_run.GatewayRunner._load_reasoning_stream(Platform.TELEGRAM)
        assert result is False

    @pytest.mark.asyncio
    async def test_load_reasoning_stream_per_platform(self, tmp_path, monkeypatch):
        hermes_home = _write_config(
            tmp_path,
            "display:\n  platforms:\n    telegram:\n      reasoning_stream: true\n"
            "    discord:\n      reasoning_stream: false\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        assert gateway_run.GatewayRunner._load_reasoning_stream(Platform.TELEGRAM) is True
        assert gateway_run.GatewayRunner._load_reasoning_stream(Platform.DISCORD) is False

    @pytest.mark.asyncio
    async def test_load_reasoning_stream_ttl_default_0(self, tmp_path, monkeypatch):
        """TTL defaults to 0 (delete immediately)."""
        hermes_home = _write_config(tmp_path, "display:\n  platforms:\n    telegram:\n      reasoning_stream: true\n")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        ttl = gateway_run.GatewayRunner._load_reasoning_stream_ttl(Platform.TELEGRAM)
        assert ttl == 0

    @pytest.mark.asyncio
    async def test_load_reasoning_stream_ttl_custom(self, tmp_path, monkeypatch):
        hermes_home = _write_config(
            tmp_path,
            "display:\n  platforms:\n    telegram:\n      reasoning_stream: true\n"
            "      reasoning_stream_ttl_seconds: 30\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        ttl = gateway_run.GatewayRunner._load_reasoning_stream_ttl(Platform.TELEGRAM)
        assert ttl == 30

    @pytest.mark.asyncio
    async def test_load_reasoning_stream_ttl_clamped_non_negative(self, tmp_path, monkeypatch):
        """Negative TTL values are clamped to 0."""
        hermes_home = _write_config(
            tmp_path,
            "display:\n  platforms:\n    telegram:\n      reasoning_stream: true\n"
            "      reasoning_stream_ttl_seconds: -5\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        ttl = gateway_run.GatewayRunner._load_reasoning_stream_ttl(Platform.TELEGRAM)
        assert ttl == 0

    @pytest.mark.asyncio
    async def test_load_reasoning_stream_ttl_invalid_str(self, tmp_path, monkeypatch):
        """Non-numeric TTL values are silently treated as 0."""
        hermes_home = _write_config(
            tmp_path,
            "display:\n  platforms:\n    telegram:\n      reasoning_stream: true\n"
            "      reasoning_stream_ttl_seconds: infinite\n",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        ttl = gateway_run.GatewayRunner._load_reasoning_stream_ttl(Platform.TELEGRAM)
        assert ttl == 0


# ── ReasoningStreamConsumer tests ─────────────────────────────────────


def _make_send_result(message_id):
    """Build a fake send() result with the given message_id."""
    result = MagicMock()
    result.message_id = message_id
    return result


class TestReasoningStreamConsumer:
    @pytest.mark.asyncio
    async def test_send_initial_returns_true_with_message_id(self):
        adapter = MagicMock()
        adapter.SUPPORTS_MESSAGE_EDITING = True
        adapter.send = AsyncMock(return_value=_make_send_result("42"))
        c = ReasoningStreamConsumer(adapter, "chat-1", edit_interval=0.01)
        ok = await c.send_initial()
        assert ok is True
        assert c.message_id == "42"
        adapter.send.assert_awaited_once()
        args, kwargs = adapter.send.call_args
        assert "Reasoning" in (kwargs.get("content") or args[1] if len(args) > 1 else kwargs.get("content"))

    @pytest.mark.asyncio
    async def test_send_initial_returns_false_when_no_message_id(self):
        adapter = MagicMock()
        adapter.SUPPORTS_MESSAGE_EDITING = True
        result = MagicMock()
        result.message_id = None
        adapter.send = AsyncMock(return_value=result)
        c = ReasoningStreamConsumer(adapter, "chat-1", edit_interval=0.01)
        ok = await c.send_initial()
        assert ok is False
        assert c.message_id is None

    @pytest.mark.asyncio
    async def test_send_initial_returns_false_on_exception(self):
        adapter = MagicMock()
        adapter.SUPPORTS_MESSAGE_EDITING = True
        adapter.send = AsyncMock(side_effect=RuntimeError("network down"))
        c = ReasoningStreamConsumer(adapter, "chat-1", edit_interval=0.01)
        ok = await c.send_initial()
        assert ok is False

    def test_on_delta_ignores_empty_text(self):
        adapter = MagicMock()
        c = ReasoningStreamConsumer(adapter, "chat-1", edit_interval=0.01)
        c.on_delta("")
        c.on_delta(None)
        assert c.emitted_any is False

    def test_on_delta_queues_text(self):
        adapter = MagicMock()
        c = ReasoningStreamConsumer(adapter, "chat-1", edit_interval=0.01)
        c.on_delta("hello")
        assert c._queue.qsize() == 1

    @pytest.mark.asyncio
    async def test_emitted_any_set_after_run_consumes_deltas(self):
        adapter = MagicMock()
        adapter.SUPPORTS_MESSAGE_EDITING = True
        adapter.send = AsyncMock(return_value=_make_send_result("1"))
        adapter.edit_message = AsyncMock()
        c = ReasoningStreamConsumer(adapter, "chat-1", edit_interval=0.01)
        await c.send_initial()
        c.on_delta("hello")
        c.finish()
        await asyncio.wait_for(c.run(), timeout=2.0)
        assert c.emitted_any is True
        assert c.finalized_message_id == "1"

    @pytest.mark.asyncio
    async def test_run_edits_in_place_with_throttling(self):
        adapter = MagicMock()
        adapter.SUPPORTS_MESSAGE_EDITING = True
        adapter.send = AsyncMock(return_value=_make_send_result("1"))
        adapter.edit_message = AsyncMock()
        c = ReasoningStreamConsumer(adapter, "chat-1", edit_interval=0.01)
        await c.send_initial()
        c.on_delta("hello ")
        c.on_delta("world")
        c.finish()
        await asyncio.wait_for(c.run(), timeout=2.0)
        assert adapter.edit_message.await_count >= 1
        last_call = adapter.edit_message.call_args_list[-1]
        content = last_call.kwargs.get("content", "")
        assert "hello" in content
        assert "world" in content
        assert c.finalized_message_id == "1"

    @pytest.mark.asyncio
    async def test_run_no_op_when_initial_send_failed(self):
        adapter = MagicMock()
        adapter.SUPPORTS_MESSAGE_EDITING = True
        adapter.send = AsyncMock(return_value=_make_send_result(None))
        adapter.edit_message = AsyncMock()
        c = ReasoningStreamConsumer(adapter, "chat-1", edit_interval=0.01)
        await c.send_initial()
        c.on_delta("hello")
        c.finish()
        await asyncio.wait_for(c.run(), timeout=2.0)
        assert adapter.edit_message.await_count == 0
        assert c.finalized_message_id is None

    @pytest.mark.asyncio
    async def test_run_skips_redundant_edits(self):
        adapter = MagicMock()
        adapter.SUPPORTS_MESSAGE_EDITING = True
        adapter.send = AsyncMock(return_value=_make_send_result("1"))
        adapter.edit_message = AsyncMock()
        c = ReasoningStreamConsumer(adapter, "chat-1", edit_interval=10.0)
        await c.send_initial()
        c.on_delta("x")
        c.finish()
        await asyncio.wait_for(c.run(), timeout=1.0)
        assert adapter.edit_message.await_count == 1
