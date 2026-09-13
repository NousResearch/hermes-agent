"""Tests for WhatsApp passive (read-only) mode.

When ``whatsapp.passive_mode`` / ``WHATSAPP_PASSIVE_MODE`` is enabled, inbound
messages are appended to the session transcript as observed context
(``observed: True``) but never dispatched to the agent — no replies, no model
spend. Outbound paths (cron deliveries, ``hermes send``) are unaffected.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_adapter(passive_mode: bool):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._passive_mode = passive_mode
    adapter._bridge_port = 19876
    adapter._bridge_process = None
    adapter._bridge_log_fh = None
    adapter._running = False
    adapter._http_session = None
    adapter._shutting_down = False
    return adapter


def _text_event(text="note to self", message_id="wa-msg-1"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=MagicMock(),
        message_id=message_id,
    )


class TestObservePassive:
    @pytest.mark.asyncio
    async def test_appends_observed_entry_without_agent_dispatch(self):
        adapter = _make_adapter(passive_mode=True)
        adapter.handle_message = AsyncMock()
        session_entry = MagicMock()
        session_entry.session_id = "sess-1"
        store = MagicMock()
        store.get_or_create_session.return_value = session_entry
        adapter._session_store = store

        event = _text_event()
        await adapter._observe_passive(event)

        store.get_or_create_session.assert_called_once_with(event.source)
        (session_id, entry), _ = store.append_to_transcript.call_args
        assert session_id == "sess-1"
        assert entry["role"] == "user"
        assert entry["content"] == "note to self"
        assert entry["observed"] is True
        assert entry["message_id"] == "wa-msg-1"
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_media_only_event_preserves_capture_context(self):
        """A photo/voice note without text must not persist as an empty row.

        Regression for media-only events: ``_build_message_event()`` puts
        cached media paths in ``media_urls``/``media_types`` and inbound
        details in ``metadata`` — the observed entry must retain all of them.
        """
        adapter = _make_adapter(passive_mode=True)
        session_entry = MagicMock()
        session_entry.session_id = "sess-1"
        store = MagicMock()
        store.get_or_create_session.return_value = session_entry
        adapter._session_store = store

        event = MessageEvent(
            text="",
            message_type=MessageType.PHOTO,
            source=MagicMock(),
            message_id="wa-img-1",
            media_urls=["/tmp/wa-cache/photo.jpg"],
            media_types=["image/jpeg"],
            metadata={"sender_name": "Samuel"},
        )
        await adapter._observe_passive(event)

        (_, entry), _ = store.append_to_transcript.call_args
        assert entry["content"] == "[image/jpeg: /tmp/wa-cache/photo.jpg]"
        assert entry["media_urls"] == ["/tmp/wa-cache/photo.jpg"]
        assert entry["media_types"] == ["image/jpeg"]
        assert entry["metadata"] == {"sender_name": "Samuel"}
        assert entry["observed"] is True

    @pytest.mark.asyncio
    async def test_media_with_caption_keeps_both(self):
        adapter = _make_adapter(passive_mode=True)
        session_entry = MagicMock()
        session_entry.session_id = "sess-1"
        store = MagicMock()
        store.get_or_create_session.return_value = session_entry
        adapter._session_store = store

        event = MessageEvent(
            text="the whiteboard from today",
            message_type=MessageType.PHOTO,
            source=MagicMock(),
            media_urls=["/tmp/wa-cache/board.jpg"],
            media_types=["image/jpeg"],
        )
        await adapter._observe_passive(event)

        (_, entry), _ = store.append_to_transcript.call_args
        assert entry["content"] == (
            "[image/jpeg: /tmp/wa-cache/board.jpg] the whiteboard from today"
        )

    @pytest.mark.asyncio
    async def test_truly_empty_event_is_not_persisted(self):
        adapter = _make_adapter(passive_mode=True)
        store = MagicMock()
        adapter._session_store = store

        event = MessageEvent(text="", source=MagicMock())
        await adapter._observe_passive(event)

        store.append_to_transcript.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_session_store_is_a_noop(self):
        adapter = _make_adapter(passive_mode=True)
        # No _session_store attribute at all — must not raise.
        await adapter._observe_passive(_text_event())

    @pytest.mark.asyncio
    async def test_store_errors_are_swallowed(self):
        adapter = _make_adapter(passive_mode=True)
        store = MagicMock()
        store.get_or_create_session.side_effect = RuntimeError("db locked")
        adapter._session_store = store
        # Must not raise — poll loop keeps running.
        await adapter._observe_passive(_text_event())


class TestPollDispatch:
    def _run_one_poll(self, adapter, event):
        """Drive a single _poll_messages iteration delivering one message."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=[{"body": "hi"}])
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=_AsyncCM(mock_resp))
        adapter._http_session = mock_session
        adapter._running = True
        adapter._build_message_event = AsyncMock(return_value=event)

        def _stop(*args, **kwargs):
            adapter._running = False

        return _stop

    @pytest.mark.asyncio
    async def test_passive_mode_observes_instead_of_dispatching(self):
        adapter = _make_adapter(passive_mode=True)
        event = _text_event()
        stop = self._run_one_poll(adapter, event)
        adapter._observe_passive = AsyncMock(side_effect=stop)
        adapter._enqueue_text_event = MagicMock()
        adapter.handle_message = AsyncMock()

        with patch(
            "plugins.platforms.whatsapp.adapter.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await adapter._poll_messages()

        adapter._observe_passive.assert_awaited_once_with(event)
        adapter._enqueue_text_event.assert_not_called()
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_active_mode_dispatch_unchanged(self):
        adapter = _make_adapter(passive_mode=False)
        event = _text_event()
        stop = self._run_one_poll(adapter, event)
        adapter._observe_passive = AsyncMock()
        adapter._enqueue_text_event = MagicMock(side_effect=stop)
        adapter.handle_message = AsyncMock()

        with patch(
            "plugins.platforms.whatsapp.adapter.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await adapter._poll_messages()

        adapter._enqueue_text_event.assert_called_once_with(event)
        adapter._observe_passive.assert_not_awaited()


class TestYamlConfigBridge:
    def test_passive_mode_yaml_key_sets_env(self, monkeypatch):
        from plugins.platforms.whatsapp.adapter import _apply_yaml_config

        monkeypatch.delenv("WHATSAPP_PASSIVE_MODE", raising=False)
        _apply_yaml_config({}, {"passive_mode": True})
        assert os.environ["WHATSAPP_PASSIVE_MODE"] == "true"
        monkeypatch.delenv("WHATSAPP_PASSIVE_MODE", raising=False)

    def test_env_var_takes_precedence_over_yaml(self, monkeypatch):
        from plugins.platforms.whatsapp.adapter import _apply_yaml_config

        monkeypatch.setenv("WHATSAPP_PASSIVE_MODE", "false")
        _apply_yaml_config({}, {"passive_mode": True})
        assert os.environ["WHATSAPP_PASSIVE_MODE"] == "false"


class TestLoaderEndToEnd:
    """Verify the configured value through the production dispatch path.

    ``load_gateway_config()`` (gateway/config.py) discovers platform plugins
    and runs their ``apply_yaml_config_fn`` hooks — this is the path a real
    gateway start takes, unlike calling ``_apply_yaml_config`` directly.
    """

    def test_passive_mode_bridged_by_load_gateway_config(
        self, tmp_path, monkeypatch
    ):
        from gateway.config import load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "whatsapp:\n"
            "  enabled: false\n"
            "  passive_mode: true\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("WHATSAPP_PASSIVE_MODE", raising=False)

        try:
            load_gateway_config()
            assert os.environ.get("WHATSAPP_PASSIVE_MODE") == "true"
        finally:
            os.environ.pop("WHATSAPP_PASSIVE_MODE", None)
