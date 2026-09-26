"""Behavior tests for the official personal WhatsApp Agent Platform plugin."""

import asyncio
import json
import time
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

wa = load_plugin_adapter("whatsapp_agent")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("WHATSAPP_AGENT_TOKEN", "test-only-token")
    return wa.WhatsAppAgentAdapter(PlatformConfig(enabled=True))


def test_plugin_is_discoverable_and_creator_is_authorized_upstream(monkeypatch, tmp_path):
    a = adapter(monkeypatch, tmp_path)
    assert Platform("whatsapp_agent").value == "whatsapp_agent"
    assert a.authorization_is_upstream is True
    assert wa._env_enablement() == {}


def test_cursor_and_recipient_survive_restart(monkeypatch, tmp_path):
    a = adapter(monkeypatch, tmp_path)
    a._offset = 1287
    a._last_recipient = "user:42"
    a._save_state()
    state = json.loads(a._state_file.read_text())
    assert state["offset"] == 1287
    assert state["recipient"] == "user:42"
    assert state["fresh_after"] == a._fresh_after
    assert a._state_file.stat().st_mode & 0o777 == 0o600
    b = adapter(monkeypatch, tmp_path)
    b._load_state()
    assert b._offset == 1287
    assert b._last_recipient == "user:42"


def test_incoming_text_preserves_platform_identity(monkeypatch, tmp_path):
    a = adapter(monkeypatch, tmp_path)
    a.handle_message = AsyncMock()
    run(a._receive({
        "from": "user:42", "id": "wamid.1", "timestamp": str(int(time.time())),
        "type": "text", "text": {"body": "hello"},
    }))
    event = a.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.message_id == "wamid.1"
    assert event.source.chat_id == "user:42"
    assert event.source.user_id == "user:42"
    assert a._last_recipient == "user:42"


def test_outbound_uses_creator_id_and_official_payload(monkeypatch, tmp_path):
    a = adapter(monkeypatch, tmp_path)
    class Response:
        status_code = 200
        def json(self):
            return {"messages": [{"id": "wamid.reply"}]}
    a._client = type("Client", (), {"post": AsyncMock(return_value=Response())})()
    result = run(a.send("user:42", "hello"))
    assert result.success
    args, kwargs = a._client.post.await_args
    assert args == ("/messages",)
    assert kwargs["json"] == {
        "messaging_product": "whatsapp", "to": "user:42",
        "type": "text", "text": {"body": "hello"},
    }


def test_old_backlog_is_not_replayed_after_cursor_advances(monkeypatch, tmp_path):
    a = adapter(monkeypatch, tmp_path)
    a._offset = 50
    a.handle_message = AsyncMock()
    run(a._receive({
        "from": "user:42", "id": "wamid.old", "timestamp": "1736844652",
        "type": "text", "text": {"body": "old message"},
    }))
    a.handle_message.assert_not_awaited()
