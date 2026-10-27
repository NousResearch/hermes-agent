"""Actual warning callback and media producer beside an armed native Slack stream."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agent.status_output import StatusOutputMixin
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from gateway.session import SessionSource
from gateway.config import Platform
from tests.gateway.test_slack_native_streaming import _make_adapter, _open_streams, META


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
async def test_warning_and_media_failure_do_not_seal_requested_final(tmp_path, monkeypatch, setting):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    cfg = {} if setting is None else {"display": {"suppress_warning_notifications": setting}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))
    adapter, client = _make_adapter()
    first = await adapter.send_draft("D1", 7, "Requested", metadata=META)
    assert first.success
    source = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm", thread_id=META["thread_id"])
    ctx = TurnContext(source=source, user_config=cfg, _run_still_current=lambda: True,
        _status_adapter=adapter, _status_chat_id="D1", _status_thread_metadata=dict(META))
    turn = TurnRunner(SimpleNamespace(), ctx)
    loop = asyncio.get_running_loop()
    scheduled = []
    def schedule(coro, *args):
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        scheduled.append(future)
        return future
    turn._schedule = schedule
    agent = StatusOutputMixin()
    agent.suppress_status_output = True
    agent.status_callback = turn._status_callback_sync
    await asyncio.to_thread(agent._warn_uncompressed_context_overflow, 200, 100)
    await asyncio.gather(*(asyncio.wrap_future(f) for f in scheduled))
    assert bool(agent._last_ctx_overflow_warn)
    assert client.chat_postMessage.await_count == (0 if setting is True else 1)
    assert client.chat_stopStream.await_count == 0
    # Actual upload producer; only Slack SDK rejects the local file.
    client.files_upload_v2 = AsyncMock(side_effect=RuntimeError("fixture upload failure"))
    adapter._client_for = lambda *args: client
    adapter._dm_target = AsyncMock(return_value="D1")
    document = tmp_path / "fixture.pdf"
    document.write_bytes(b"fixture document")
    media = await adapter.send_document("D1", str(document), metadata={**META, "_interim_send": True})
    assert media.success is not (setting is True)  # legacy text fallback receipt
    assert client.chat_postMessage.await_count == (0 if setting is True else 2)
    assert client.chat_stopStream.await_count == 0
    assert _open_streams(adapter, "D1")  # keyed per (team, chat, thread)
    await adapter.send_draft("D1", 7, "Requested final", metadata=META)
    result = await adapter.send("D1", "Requested final", metadata=META)
    assert result.success
    assert client.chat_startStream.await_count == 1
    assert client.chat_stopStream.await_count == 1
    assert not _open_streams(adapter, "D1")
    assert client.chat_postMessage.await_count == (0 if setting is True else 2)
