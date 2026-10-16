"""Weixin cross-delivery session-not-ready suppression and alert convergence."""

import asyncio
from unittest.mock import AsyncMock

from gateway.config import PlatformConfig
from gateway.platforms.base import classify_send_error
from gateway.platforms.weixin import WeixinAdapter
from gateway.platforms.weixin_session_backoff import SessionBackoffRegistry


def _adapter(tmp_path, monkeypatch, **extra):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    values = {"account_id": "account", "session_backoff_base_seconds": "0.1",
              "session_backoff_max_seconds": "0.3", **extra}
    adapter = WeixinAdapter(PlatformConfig(enabled=True, token="token", extra=values))
    adapter._send_session = object()
    return adapter


def _fail_send(adapter, chat_id="chat"):
    if not isinstance(adapter._send_text_chunk, AsyncMock):
        adapter._send_text_chunk = AsyncMock()
    adapter._send_text_chunk.side_effect = RuntimeError(
        "iLink sendmessage session not ready: ret=-2 errcode=-2 errmsg=prepare failed")
    return asyncio.run(adapter.send(chat_id, "hello"))


def test_send_error_classifier_recognizes_session_not_ready_without_rate_limit_collision():
    assert classify_send_error(None, error_text=(
        "iLink sendmessage session not ready: ret=-2 errcode=-2 errmsg=prepare failed"
    )) == "session_not_ready"
    assert classify_send_error(None, error_text="HTTP 429 rate limit; retry after 3") == "rate_limited"
    assert classify_send_error(None, error_text="unexpected payload") == "unknown"


def test_registry_backoff_doubles_then_reaches_bounded_long_window(tmp_path):
    registry = SessionBackoffRegistry(str(tmp_path), base_seconds=30, max_seconds=1800)
    assert registry.record_failure("account", "chat", "first", now=100, threshold=3) is False
    assert registry.should_suppress("account", "chat", now=100)[1] == 30
    assert registry.record_failure("account", "chat", "second", now=130, threshold=3) is False
    assert registry.should_suppress("account", "chat", now=130)[1] == 60
    assert registry.record_failure("account", "chat", "third", now=190, threshold=3) is True
    assert registry.should_suppress("account", "chat", now=190)[1] == 1800
    assert registry.should_suppress("account", "other", now=190)[0] is False


def test_registry_persists_and_restores_state(tmp_path):
    registry = SessionBackoffRegistry(str(tmp_path))
    registry.record_failure("account", "chat", "not ready", now=100)
    restored = SessionBackoffRegistry(str(tmp_path))
    assert restored.should_suppress("account", "chat", now=100)[0] is True
    assert restored.should_suppress("account", "chat", now=130)[0] is False
    assert restored.clear("account", "chat") is True
    assert restored.should_suppress("account", "chat", now=100)[0] is False


def test_adapter_backoff_short_circuits_same_target_and_isolates_other_target(tmp_path, monkeypatch, caplog):
    adapter = _adapter(tmp_path, monkeypatch)
    result = _fail_send(adapter)
    assert result.error_kind == "session_not_ready"
    assert adapter._send_text_chunk.await_count == 1

    suppressed = _fail_send(adapter)
    assert suppressed.success is False
    assert suppressed.error_kind == "session_not_ready"
    assert "backoff active" in suppressed.error
    assert adapter._send_text_chunk.await_count == 1

    other = _fail_send(adapter, chat_id="other")
    assert other.error_kind == "session_not_ready"
    assert adapter._send_text_chunk.await_count == 2
    assert "other" in str(adapter._send_text_chunk.await_args.kwargs["chat_id"])


def test_adapter_reprobes_after_ttl_and_success_clears_state(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch)
    _fail_send(adapter)
    asyncio.run(asyncio.sleep(0.11))
    adapter._send_text_chunk = AsyncMock(return_value=None)
    result = asyncio.run(adapter.send("chat", "hello"))
    assert result.success is True
    assert adapter._send_text_chunk.await_count == 1
    failure = _fail_send(adapter)
    assert failure.error_kind == "session_not_ready"
    assert adapter._send_text_chunk.await_count == 2


def test_adapter_alert_threshold_converges_to_one_critical(tmp_path, monkeypatch, caplog):
    adapter = _adapter(tmp_path, monkeypatch)
    with caplog.at_level("CRITICAL", logger="gateway.platforms.weixin"):
        _fail_send(adapter)
        asyncio.run(asyncio.sleep(0.11))
        _fail_send(adapter)
        asyncio.run(asyncio.sleep(0.21))
        _fail_send(adapter)
    critical = [record for record in caplog.records if record.levelname == "CRITICAL"]
    assert len(critical) == 1
    assert "chat" in critical[0].getMessage()
    assert "user must send the bot a message first" in critical[0].getMessage()


def test_adapter_backoff_can_be_disabled(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch, session_backoff_enabled=False)
    _fail_send(adapter)
    _fail_send(adapter)
    assert adapter._send_text_chunk.await_count == 2


def test_successful_normal_send_does_not_write_backoff_state(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch)
    adapter._send_text_chunk = AsyncMock(return_value=None)
    result = asyncio.run(adapter.send("chat", "hello"))
    assert result.success is True
    assert not (tmp_path / "gateway" / "weixin_session_backoff.json").exists()


class _BadResponse:
    status = 200

    async def json(self, content_type=None):
        return {"error": "ignored"}


class _BadPost:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return _BadResponse()

    async def __aexit__(self, *args):
        return False


class _BadSession:
    def __init__(self, *args, **kwargs):
        self.post_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        self.post_calls += 1
        return _BadPost()


def test_webhook_alert_rejects_http_200_with_wrong_body_and_retries(tmp_path, monkeypatch, caplog):
    adapter = _adapter(tmp_path, monkeypatch, weixin_alert_webhook_url="https://alerts.example/hook")
    session = _BadSession()
    monkeypatch.setattr("gateway.platforms.weixin.aiohttp.ClientSession", lambda *a, **kw: session)
    with caplog.at_level("WARNING", logger="gateway.platforms.weixin"):
        asyncio.run(adapter._send_session_alert("chat", 3, "not ready"))
    assert session.post_calls == 3
    assert any("alert abandoned" in record.getMessage() for record in caplog.records)
