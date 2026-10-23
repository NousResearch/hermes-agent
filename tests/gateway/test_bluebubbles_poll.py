"""REST recovery contracts using the real adapter and a local query transport."""

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest
import httpx

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter


class _Request:
    query = {"password": "secret"}
    headers = {}

    def __init__(self, record):
        self.record = record

    async def read(self):
        return json.dumps({"type": "new-message", "data": self.record}).encode()


def _record(guid, date, text="hello", **fields):
    return {"guid": guid, "dateCreated": date, "text": text,
            "chats": [{"guid": "iMessage;-;+15550001111"}],
            "handle": {"address": "+15550001111"}, **fields}


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    result = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://localhost:1234", "password": "secret", "send_read_receipts": False,
    }))
    result.handle_message = AsyncMock()
    return result


def _poller(adapter, tmp_path):
    from gateway.platforms.bluebubbles_poll import BlueBubblesPoller
    adapter._poller = BlueBubblesPoller(adapter, interval=15, state_path=tmp_path / "poll.json")
    return adapter._poller


def _server(adapter, records):
    async def query(path, body):
        assert path == "/api/v1/message/query"
        matches = [r for r in records if body.get("after", 0) <= r["dateCreated"]
                   <= body.get("before", float("inf"))]
        matches.sort(key=lambda r: r["dateCreated"], reverse=body["sort"] == "DESC")
        return {"data": matches[body["offset"]:body["offset"] + body["limit"]]}
    adapter._api_post = AsyncMock(side_effect=query)


@pytest.mark.asyncio
async def test_poll_recovers_miss_without_replaying_initial_overlap(adapter, tmp_path):
    now = int(time.time() * 1000)
    records = [_record("old-1", now - 3000), _record("old-2", now - 2000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    adapter.handle_message.assert_not_called()
    records.append(_record("missed", now - 1000, "recover this"))
    await poller._tick()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.call_args.args[0]
    assert (event.text, event.message_id, event.source.chat_id,
            event.source.user_id, event.source.chat_type) == (
                "recover this", "missed", "iMessage;-;+15550001111", "+15550001111", "dm")
    assert json.loads((tmp_path / "poll.json").read_text())["cursor_ms"] >= now - 1000


@pytest.mark.asyncio
@pytest.mark.parametrize("webhook_first", [True, False])
async def test_webhook_poll_race_and_edit(adapter, tmp_path, webhook_first):
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    message = _record("race", now - 1000)
    records.append(message)
    if webhook_first:
        await adapter._handle_webhook(_Request(message))
    await poller._tick()
    if not webhook_first:
        await adapter._handle_webhook(_Request(message))
    adapter.handle_message.assert_awaited_once()
    records[-1] = {**message, "text": "edited"}
    await poller._tick()
    assert [call.args[0].text for call in adapter.handle_message.call_args_list] == ["hello", "edited"]


@pytest.mark.asyncio
@pytest.mark.parametrize("poll_phase", ["idle", "query", "unseeded", "seed_query"])
async def test_webhook_handoff_survives_restart_before_poll(adapter, tmp_path, monkeypatch, poll_phase):
    from gateway.platforms import bluebubbles_poll as polling
    now = int(time.time() * 1000)
    monkeypatch.setattr(polling, "_now_ms", lambda: now)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    if poll_phase in {"idle", "query"}:
        await poller._tick()
    query_started, release_query = asyncio.Event(), asyncio.Event()
    query = adapter._api_post.side_effect

    async def delayed_query(path, body):
        response = await query(path, body)
        query_started.set()
        await release_query.wait()
        return response

    pending_poll = None
    if poll_phase in {"query", "seed_query"}:
        adapter._api_post.side_effect = delayed_query
        pending_poll = asyncio.create_task(poller._tick())
        await asyncio.wait_for(query_started.wait(), 5)
    message = _record("webhook", now + 2000)
    records.extend([_record("missed", now + 1000), message])
    monkeypatch.setattr(polling, "_now_ms", lambda: now + 3000)
    try:
        assert (await adapter._handle_webhook(_Request(message))).status == 200
        adapter.handle_message.assert_awaited_once()
    finally:
        release_query.set()
        if pending_poll:
            await asyncio.wait_for(pending_poll, 5)

    # A new adapter has no in-memory dedup; only the persisted handoff can prevent replay.
    restarted = BlueBubblesAdapter(adapter.config)
    restarted.handle_message = AsyncMock()
    _server(restarted, records)
    await _poller(restarted, tmp_path)._tick()
    assert [call.args[0].message_id for call in restarted.handle_message.call_args_list] == ["missed"]
    records[-1] = {**message, "text": "edited"}
    await restarted._poller._tick()
    assert [call.args[0].text for call in restarted.handle_message.call_args_list] == ["hello", "edited"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_webhook_handoff_retries_save_without_redispatch(adapter, tmp_path, monkeypatch, cancel):
    from gateway.platforms import bluebubbles_poll as polling
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    await _poller(adapter, tmp_path)._tick()
    started, release = asyncio.Event(), asyncio.Event()
    to_thread = polling._to_thread

    async def interrupted_save(fn, *args):
        if not started.is_set():
            started.set()
            await release.wait()
            if not cancel:
                raise OSError("disk unavailable")
        return await to_thread(fn, *args)

    monkeypatch.setattr(polling, "_to_thread", interrupted_save)
    message = _record("webhook", now - 1000)
    records.append(message)
    webhook = asyncio.create_task(adapter._handle_webhook(_Request(message)))
    await asyncio.wait_for(started.wait(), 5)
    if cancel:
        webhook.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError if cancel else OSError):
        await webhook
    assert (await adapter._handle_webhook(_Request(message))).status == 200
    adapter.handle_message.assert_awaited_once()
    restarted = BlueBubblesAdapter(adapter.config)
    restarted.handle_message = AsyncMock()
    _server(restarted, records)
    await _poller(restarted, tmp_path)._tick()
    restarted.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_restart_late_arrivals_and_idle_dedup_expiry(adapter, tmp_path):
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    records.extend([_record("new", now - 2000), _record("own", now - 1000, isFromMe=True)])
    await poller._tick()
    assert poller.state.cursor_ms == now - 1000
    records.append(_record("late", now - 2000))
    await poller._tick()
    assert adapter.handle_message.await_count == 2
    adapter._dedup.clear()  # A restart/TTL expiration must not replay the still-queried overlap.
    restarted = _poller(adapter, tmp_path)
    await restarted._tick()
    adapter._dedup.clear()
    await restarted._tick()
    assert adapter.handle_message.await_count == 2
    records.append(_record("after-restart", now - 500))
    await restarted._tick()
    assert adapter.handle_message.call_args.args[0].message_id == "after-restart"
    assert adapter.handle_message.await_count == 3


@pytest.mark.asyncio
async def test_quiet_chat_uses_last_success_for_restart_window(adapter, tmp_path, monkeypatch):
    from gateway.platforms import bluebubbles_poll as polling
    now = int(time.time() * 1000)
    monkeypatch.setattr(polling, "_now_ms", lambda: now)
    records = [_record("old", now - polling.CATCHUP_MS * 3)]
    _server(adapter, records)
    await _poller(adapter, tmp_path)._tick()
    records.append(_record("recent", now - 1000))
    await _poller(adapter, tmp_path)._tick()
    adapter.handle_message.assert_awaited_once()
    monkeypatch.setattr(polling, "_now_ms", lambda: now + polling.CATCHUP_MS + 1)
    records.append(_record("stale", now + 1000))
    restarted = _poller(adapter, tmp_path)
    await restarted._tick()
    await restarted._tick()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_capped_pagination_resumes_across_restart_and_timestamp_ties(adapter, tmp_path, monkeypatch):
    from gateway.platforms import bluebubbles_poll as polling
    monkeypatch.setattr(polling, "PAGE_LIMIT", 2)
    monkeypatch.setattr(polling, "MAX_PAGES", 2)
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    records.extend(_record(str(i), now - 1000) for i in range(9))
    await poller._tick()
    assert adapter.handle_message.await_count == 3
    assert poller.state.offset == 4
    restarted = _poller(adapter, tmp_path)
    await restarted._tick()
    await restarted._tick()
    assert restarted.state.offset == 0
    assert [c.args[0].message_id for c in adapter.handle_message.call_args_list] == [str(i) for i in range(9)]
    records.append(_record("later", now - 500))
    for _ in range(3):
        await restarted._tick()
    assert adapter.handle_message.await_count == 10
    assert adapter.handle_message.call_args.args[0].message_id == "later"


@pytest.mark.asyncio
async def test_startup_delay_empty_server_and_future_record(adapter, tmp_path, monkeypatch):
    from gateway.platforms import bluebubbles_poll as polling
    now = int(time.time() * 1000)
    monkeypatch.setattr(polling, "_now_ms", lambda: now)
    records = [_record("future", now + 100_000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    records.append(_record("during-startup", now + 1000))
    monkeypatch.setattr(polling, "_now_ms", lambda: now + 15_000)
    await poller._tick()
    adapter.handle_message.assert_not_called()
    assert poller.state.cursor_ms == now
    await poller._tick()
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.call_args.args[0].message_id == "during-startup"


@pytest.mark.asyncio
async def test_persist_happens_after_handoff_and_failed_save_retries(adapter, tmp_path, monkeypatch):
    from gateway.platforms import bluebubbles_poll as polling
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    original = polling.BlueBubblesPollState.save
    order = []

    async def handle(event):
        order.append("handoff")

    def save(state, path):
        order.append("save")
        if order.count("save") == 1:
            raise OSError("disk unavailable")
        original(state, path)

    adapter.handle_message.side_effect = handle
    monkeypatch.setattr(polling.BlueBubblesPollState, "save", save)
    records.append(_record("miss", now - 1000))
    with pytest.raises(OSError):
        await poller._tick()
    assert poller.state.cursor_ms == now - 5000
    await poller._tick()
    assert order == ["handoff", "save", "save"]
    assert poller.state.cursor_ms == now - 1000


@pytest.mark.asyncio
async def test_cancellation_waits_for_persistence_before_restart(adapter, tmp_path, monkeypatch):
    from gateway.platforms import bluebubbles_poll as polling
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_write(fn, *args):
        started.set()
        await release.wait()
        fn(*args)

    monkeypatch.setattr(polling, "_to_thread", delayed_write)
    records.append(_record("miss", now - 1000))
    task = asyncio.create_task(poller._tick())
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _poller(adapter, tmp_path)._tick()
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_loop_retries_query_error_without_leaking_password(adapter, tmp_path, monkeypatch, caplog):
    from gateway.platforms import bluebubbles_poll as polling
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    records.append(_record("miss", now - 1000))
    query = adapter._api_post.side_effect
    calls = 0
    delays = []

    async def sometimes_fail(path, body):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("http://server?password=do-not-log-this")
        return await query(path, body)

    async def sleep(delay):
        delays.append(delay)
        if len(delays) == 2:
            assert poller.state.cursor_ms == now - 5000
        if len(delays) == 3:
            adapter._running = False

    adapter._api_post.side_effect = sometimes_fail
    monkeypatch.setattr(polling, "_sleep", sleep)
    adapter._running = True
    await poller.run()
    assert delays == [15, 30, 15]
    adapter.handle_message.assert_awaited_once()
    assert "do-not-log-this" not in caplog.text


@pytest.mark.asyncio
async def test_polled_filters_run_before_attachment_download(adapter, tmp_path, monkeypatch):
    from gateway.platforms.event import MessageType
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    adapter.require_mention = True
    attachment = {"guid": "image", "mimeType": "image/png"}
    download = AsyncMock(return_value="cached-image.png")
    monkeypatch.setattr(adapter, "_download_attachment", download)
    group = {"chats": [{"guid": "iMessage;+;group"}], "attachments": [attachment]}
    records.extend([
        _record("unmentioned", now - 4000, **group),
        _record("reaction", now - 3000, associatedMessageType=2000, **group),
        _record("from-me", now - 2000, isFromMe=True, **group),
        _record("mentioned", now - 1000, "Hermes: see this", **group),
    ])
    await poller._tick()
    adapter.handle_message.assert_awaited_once()
    download.assert_awaited_once()
    event = adapter.handle_message.call_args.args[0]
    assert event.source.chat_id == "iMessage;+;group"
    assert event.source.chat_type == "group"
    assert event.text == "see this"
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls == ["cached-image.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_poll_recovers_after_inflight_webhook_attachment_failure(adapter, tmp_path, monkeypatch, cancel):
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    started, release = asyncio.Event(), asyncio.Event()
    collect = adapter._collect_attachments
    attempts = 0

    async def interrupted(record):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            started.set()
            await release.wait()
            raise OSError("transient")
        return await collect(record)

    monkeypatch.setattr(adapter, "_collect_attachments", interrupted)
    message = _record("miss", now - 1000)
    records.append(message)
    webhook = asyncio.create_task(adapter._handle_webhook(_Request(message)))
    await asyncio.wait_for(started.wait(), 5)
    poll = asyncio.create_task(poller._tick())
    if cancel:
        webhook.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError if cancel else OSError):
        await webhook
    await asyncio.wait_for(poll, 5)
    adapter.handle_message.assert_awaited_once()
    assert poller.state.cursor_ms == now - 1000


@pytest.mark.asyncio
async def test_corrupt_state_reseeds_and_bad_record_does_not_poison_page(adapter, tmp_path):
    now = int(time.time() * 1000)
    (tmp_path / "poll.json").write_text('{"seen": ["broken"]}')
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    adapter._api_post = AsyncMock(return_value={"data": [
        _record("bad", None), _record("good", now - 1000),
    ]})
    await poller._tick()
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.call_args.args[0].message_id == "good"


@pytest.mark.asyncio
async def test_attachment_retry_does_not_lose_older_restart_backlog(adapter, tmp_path, monkeypatch):
    now = int(time.time() * 1000)
    records = [_record("old", now - 240_000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    records.extend([
        _record("photo", now - 120_000, "", attachments=[{"guid": "attachment", "mimeType": "image/png"}]),
        _record("newer", now - 1000),
    ])
    monkeypatch.setattr(adapter, "_download_attachment", AsyncMock(side_effect=[None, "photo.png"]))
    await poller._tick()
    adapter.handle_message.assert_not_called()
    assert poller.state.cursor_ms == now - 240_000
    restarted = _poller(adapter, tmp_path)
    await restarted._tick()
    await restarted._tick()
    assert [c.args[0].message_id for c in adapter.handle_message.call_args_list] == ["photo", "newer"]
    assert adapter.handle_message.call_args_list[0].args[0].media_urls == ["photo.png"]


@pytest.mark.asyncio
async def test_failed_later_page_preserves_earlier_checkpoint(adapter, tmp_path, monkeypatch):
    from gateway.platforms import bluebubbles_poll as polling
    monkeypatch.setattr(polling, "PAGE_LIMIT", 2)
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    records.extend([_record("first", now - 2000), _record("second", now - 1000)])
    query = adapter._api_post.side_effect

    async def broken_later_page(path, body):
        if body["offset"]:
            raise OSError("temporary transport failure")
        return await query(path, body)

    adapter._api_post.side_effect = broken_later_page
    with pytest.raises(OSError):
        await poller._tick()
    assert poller.state.offset == 2
    adapter._dedup.clear()
    adapter._api_post.side_effect = query
    await _poller(adapter, tmp_path)._tick()
    assert [c.args[0].message_id for c in adapter.handle_message.call_args_list] == ["first", "second"]


@pytest.mark.asyncio
async def test_query_uses_real_authenticated_rest_transport(adapter, tmp_path):
    now = int(time.time() * 1000)
    calls = []

    async def transport(request):
        assert request.method == "POST"
        assert request.url.path == "/api/v1/message/query"
        assert request.url.params["password"] == "secret"
        body = json.loads(request.content)
        calls.append(body)
        data = [_record("old", now - 5000)] if body["sort"] == "DESC" else [_record("miss", now - 1000)]
        return httpx.Response(200, json={"status": 200, "data": data})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        adapter.client = client
        poller = _poller(adapter, tmp_path)
        await poller._tick()
        await poller._tick()
    adapter.handle_message.assert_awaited_once()
    assert calls[1]["after"] == now - 5000 - 60_000
    assert "attachment" in calls[1]["with"]


@pytest.mark.asyncio
async def test_permanent_attachment_failure_does_not_block_later_messages(adapter, tmp_path, monkeypatch, caplog):
    now = int(time.time() * 1000)
    records = [_record("old", now - 5000)]
    _server(adapter, records)
    poller = _poller(adapter, tmp_path)
    await poller._tick()
    records.extend([
        _record("photo", now - 2000, "", attachments=[{"guid": "gone", "mimeType": "image/png"}]),
        _record("text", now - 1000),
    ])
    download = AsyncMock(return_value=None)
    monkeypatch.setattr(adapter, "_download_attachment", download)
    for _ in range(4):
        # Retry accounting must survive a process restart, too.
        await _poller(adapter, tmp_path)._tick()
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.call_args.args[0].message_id == "text"
    assert "skipping unavailable attachment" in caplog.text


@pytest.mark.asyncio
async def test_attachment_http_error_does_not_log_password(adapter, caplog):
    adapter.password = "password-that-must-stay-private"
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404))) as client:
        adapter.client = client
        assert await adapter._download_attachment("gone", {"mimeType": "image/png"}) is None
    assert "failed to download attachment" in caplog.text
    assert adapter.password not in caplog.text
