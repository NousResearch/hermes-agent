"""WhatsApp adapter failure surfaces that must reach the reconnect watcher (#137831):
an adopted bridge dying mid-poll, and the connect() transient paths that used to return
a bare False with no named error."""
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.run import _profile_runtime_scope
from plugins.platforms.whatsapp import adapter as module
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


class _RespCM:
    """Minimal async-CM standing in for ``self._bridge_req(...)`` yielding a 200 with no messages."""

    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


def _ok_response():
    resp = Mock()
    resp.status = 200
    resp.json = AsyncMock(return_value=[])
    return _RespCM(resp)


def _adapter_under(home):
    with _profile_runtime_scope(home, hydrate_secrets=False):
        return WhatsAppAdapter(PlatformConfig())


@pytest.mark.asyncio
async def test_adopted_bridge_poll_streak_flags_retryable(tmp_path, monkeypatch):
    """An adopted bridge holds no Popen handle, so its death surfaces only as failed polls; the streak
    bound must flag a retryable fatal (the adapter may not stay 'connected' forever) so the watcher
    rebuilds instead of wedging the profile deaf (#137831)."""
    home = tmp_path / "a"
    home.mkdir()
    adapter = _adapter_under(home)
    adapter._running = True
    adapter._http_session = object()  # adopted: _bridge_process stays None
    req = Mock(side_effect=ConnectionError("refused"))
    monkeypatch.setattr(adapter, "_bridge_req", req)
    monkeypatch.setattr(adapter, "_report_bridge_exit", AsyncMock(return_value=False))
    notify = AsyncMock()
    monkeypatch.setattr(adapter, "_notify_fatal_error", notify)

    async def _no_sleep(_secs=0):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", _no_sleep)
    await adapter._poll_messages()
    assert adapter.has_fatal_error
    assert adapter.fatal_error_retryable
    assert adapter.fatal_error_code == "whatsapp_bridge_unreachable"
    assert str(adapter._bridge_port) in adapter.fatal_error_message
    notify.assert_awaited_once()
    assert req.call_count == module._POLL_FAILURES_BEFORE_FATAL


@pytest.mark.asyncio
async def test_poll_streak_resets_on_a_healthy_poll(tmp_path, monkeypatch):
    """Failures either side of one healthy poll must not add up: a transient dropout shorter than the
    bound leaves the adapter running with no fatal error."""
    home = tmp_path / "b"
    home.mkdir()
    adapter = _adapter_under(home)
    adapter._running = True
    adapter._http_session = object()
    state = {"calls": 0}

    def _req(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 5 or state["calls"] == 10:
            if state["calls"] == 10:
                adapter._running = False  # a clean tail: the loop exits without a fatal
            return _ok_response()
        raise ConnectionError("refused")

    monkeypatch.setattr(adapter, "_bridge_req", _req)
    monkeypatch.setattr(adapter, "_report_bridge_exit", AsyncMock(return_value=False))
    notify = AsyncMock()
    monkeypatch.setattr(adapter, "_notify_fatal_error", notify)

    async def _no_sleep(_secs=0):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", _no_sleep)
    await adapter._poll_messages()
    assert not adapter.has_fatal_error
    notify.assert_not_awaited()
    assert state["calls"] == 10


@pytest.mark.asyncio
async def test_session_lock_failure_is_named_retryable(tmp_path, monkeypatch):
    """The unserve/re-serve race: the dying adapter still holds the whatsapp-session lock. connect()
    must name the failure and mark it retryable, not return a bare False the status cannot classify."""
    home = tmp_path / "c"
    home.mkdir()
    adapter = _adapter_under(home)
    adapter._runtime_status_platform_key = "c:whatsapp"
    monkeypatch.setattr(module, "find_node_executable", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(adapter, "_preflight", lambda: True)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a: False)
    assert not await adapter.connect()
    assert adapter.has_fatal_error
    assert adapter.fatal_error_retryable
    assert adapter.fatal_error_code == "whatsapp_session_lock_busy"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_spawn_exception_is_named_retryable(tmp_path, monkeypatch):
    """A bridge spawn that raises (node gone mid-run, Popen error) is transient: name it retryable so
    the reconnect watcher keeps trying instead of seeing an unclassified connect failure."""
    home = tmp_path / "d"
    home.mkdir()
    adapter = _adapter_under(home)
    monkeypatch.setattr(module, "find_node_executable", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(adapter, "_preflight", lambda: True)
    monkeypatch.setattr(adapter, "_ensure_bridge_deps", lambda path: True)
    monkeypatch.setattr(adapter, "_reuse_running_bridge", AsyncMock(return_value=False))
    monkeypatch.setattr(module, "_kill_stale_bridge_by_pidfile", lambda *a: None)
    monkeypatch.setattr(module, "_kill_port_process", lambda *a: None)

    async def _no_sleep(_secs=0):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(module.subprocess, "Popen", Mock(side_effect=RuntimeError("spawn denied")))
    assert not await adapter.connect()
    assert adapter.has_fatal_error
    assert adapter.fatal_error_retryable
    assert adapter.fatal_error_code == "whatsapp_bridge_start_failed"
    await adapter.disconnect()
