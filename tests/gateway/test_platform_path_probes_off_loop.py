"""Platform media path probes must run off the event-loop thread.

A blocking ``os.path.exists`` / ``Path.exists`` / ``Path.is_file`` / ``Path.expanduser``
inside an async send path stalls the whole gateway event loop whenever the path is slow:
a hung network mount, a spun-down disk, a file on a failing share. Every adapter here
probes a media path before it touches its client, so the probe is reachable without a
live connection.
"""

import threading
from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult


def _recorder(monkeypatch, owner, name, original=None):
    """Patch ``owner.name`` to record the thread each call runs on."""
    real = original or getattr(owner, name)
    calls = []

    def _tracking(*args, **kwargs):
        calls.append(threading.current_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, _tracking)
    return calls


@pytest.mark.asyncio
async def test_feishu_send_image_file_probes_path_off_loop(monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()  # send_image_file returns early without one
    calls = _recorder(monkeypatch, __import__("os").path, "exists", lambda _p: False)
    loop_thread = threading.current_thread()

    result = await adapter.send_image_file("chat-1", "/missing/image.png")

    assert result.success is False
    assert calls, "the existence probe never ran"
    assert all(thread is not loop_thread for thread in calls)


@pytest.mark.asyncio
async def test_google_chat_send_file_probes_path_off_loop(monkeypatch):
    import os

    from plugins.platforms.google_chat.adapter import GoogleChatAdapter

    cfg = PlatformConfig(enabled=True)
    cfg.extra.update({"project_id": "p", "subscription_name": "s", "service_account_json": "/tmp/sa.json"})
    adapter = GoogleChatAdapter(cfg)
    calls = _recorder(monkeypatch, os.path, "exists", lambda _p: False)
    loop_thread = threading.current_thread()

    result = await adapter._send_file("spaces/AAA", "/missing/upload.bin", None, None)

    assert result.success is False
    assert calls
    assert all(thread is not loop_thread for thread in calls)


@pytest.mark.asyncio
async def test_mattermost_send_local_file_probes_path_off_loop(monkeypatch):
    from plugins.platforms.mattermost.adapter import MattermostAdapter

    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="mm-token", extra={"url": "https://mm.example"}))
    calls = _recorder(monkeypatch, Path, "exists", lambda _self: False)
    loop_thread = threading.current_thread()

    result = await adapter._send_local_file("chan-1", "/missing/report.pdf", None, None)

    # A missing local file is skipped, not an error (pre-existing behaviour).
    assert result.success is True
    assert calls
    assert all(thread is not loop_thread for thread in calls)


@pytest.mark.asyncio
async def test_simplex_send_document_probes_path_off_loop(monkeypatch):
    from plugins.platforms.simplex.adapter import SimplexAdapter

    adapter = SimplexAdapter(PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"}))
    calls = _recorder(monkeypatch, Path, "exists", lambda _self: False)
    loop_thread = threading.current_thread()

    result = await adapter.send_document("corr-1", "/missing/file.pdf")

    assert result.success is False
    assert result.error == "File not found"
    assert calls
    assert all(thread is not loop_thread for thread in calls)


@pytest.mark.asyncio
async def test_buzz_send_image_expand_and_probe_off_loop(monkeypatch):
    from plugins.platforms.buzz import adapter as buzz_mod

    adapter = buzz_mod.BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay"}))
    expand_calls = _recorder(monkeypatch, Path, "expanduser")
    probe_calls = _recorder(monkeypatch, Path, "is_file", lambda _self: False)

    async def _stub_send(*_args, **_kwargs):
        return SendResult(success=True, message_id="m-1")

    monkeypatch.setattr(adapter, "send", _stub_send)
    loop_thread = threading.current_thread()

    await adapter.send_image("chan-1", "~/missing/photo.png", caption="hi")

    assert expand_calls and probe_calls
    assert all(thread is not loop_thread for thread in expand_calls + probe_calls)


@pytest.mark.asyncio
async def test_line_send_image_file_runs_media_check_off_loop(monkeypatch):
    from plugins.platforms.line.adapter import LineAdapter

    adapter = LineAdapter(PlatformConfig(enabled=True, extra={
        "channel_access_token": "tok", "channel_secret": "secret", "port": 1}))
    check_calls = []

    def _fake_check(kind, path):
        check_calls.append(threading.current_thread())
        return None, "Media file not found"

    monkeypatch.setattr(adapter, "_check_media_file", _fake_check)
    loop_thread = threading.current_thread()

    result = await adapter.send_image_file("user-1", "/missing/image.png")

    # The error path returns the message string (pre-existing shape in this adapter).
    assert result == "Media file not found"
    assert check_calls
    assert all(thread is not loop_thread for thread in check_calls)
