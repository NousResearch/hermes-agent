"""WhatsApp typing presence must follow foreground processing ownership."""

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


CHAT_ID = "15551234567"
CHAT_JID = f"{CHAT_ID}@s.whatsapp.net"
OTHER_CHAT_ID = "15557654321"


class _AsyncResponse:
    def __init__(
        self,
        *,
        status=200,
        payload=None,
        entered=None,
        release=None,
    ):
        self.status = status
        self._payload = {"success": True} if payload is None else payload
        self._entered = entered
        self._release = release

    async def __aenter__(self):
        if self._entered is not None:
            self._entered.set()
        if self._release is not None:
            await self._release.wait()
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def json(self):
        return self._payload


class _LostComposingAckResponse(_AsyncResponse):
    """Model presence applied by the bridge before its acknowledgement is lost."""

    def __init__(self, *, wait_for_cancellation):
        super().__init__()
        self.json_entered = asyncio.Event()
        self._wait_for_cancellation = wait_for_cancellation

    async def json(self):
        self.json_entered.set()
        if self._wait_for_cancellation:
            await asyncio.Event().wait()
        raise ConnectionResetError("typing acknowledgement lost")


class _RecordingHTTPSession:
    """Record bridge calls while keeping the transport boundary fake."""

    def __init__(self, responses=None):
        self.requests = []
        self._responses = list(responses or [])
        self.closed = False

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        if self._responses:
            return self._responses.pop(0)
        return _AsyncResponse()

    @property
    def presence_states(self):
        return [
            kwargs["json"].get("state", "composing")
            for url, kwargs in self.requests
            if url.endswith("/typing")
        ]


def _make_adapter(http_session=None):
    adapter = WhatsAppAdapter(
        PlatformConfig(enabled=True, extra={"session_name": "typing-test"})
    )
    adapter._running = True
    adapter._http_session = http_session or _RecordingHTTPSession()
    return adapter


async def _wait_for_presence_count(adapter, count):
    async def _poll():
        while len(adapter._http_session.presence_states) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(_poll(), timeout=1)


@pytest.mark.asyncio
async def test_single_foreground_typing_loop_pauses_once_only_after_completion():
    adapter = _make_adapter()
    completed = asyncio.Event()
    task = asyncio.create_task(
        adapter._keep_typing(CHAT_ID, interval=0.01, stop_event=completed)
    )

    await _wait_for_presence_count(adapter, 1)
    assert adapter._http_session.presence_states == ["composing"]

    completed.set()
    await asyncio.wait_for(task, timeout=1)

    assert adapter._http_session.presence_states[-1] == "paused"
    assert adapter._http_session.presence_states.count("paused") == 1

    # Base cleanup invokes stop repeatedly; an already-paused chat stays quiet.
    await adapter.stop_typing(CHAT_ID)
    await adapter.stop_typing(CHAT_ID)
    assert adapter._http_session.presence_states.count("paused") == 1


@pytest.mark.asyncio
async def test_same_chat_stays_composing_until_last_foreground_owner_finishes():
    adapter = _make_adapter()
    first = asyncio.create_task(adapter._keep_typing(CHAT_ID, interval=60))
    await _wait_for_presence_count(adapter, 1)
    second = asyncio.create_task(adapter._keep_typing(CHAT_ID, interval=60))
    await _wait_for_presence_count(adapter, 2)

    first.cancel()
    await asyncio.wait_for(first, timeout=1)
    assert "paused" not in adapter._http_session.presence_states

    second.cancel()
    await asyncio.wait_for(second, timeout=1)
    assert adapter._http_session.presence_states[-1] == "paused"
    assert adapter._http_session.presence_states.count("paused") == 1


@pytest.mark.asyncio
async def test_stop_typing_during_foreground_ownership_does_not_pause():
    adapter = _make_adapter()
    task = asyncio.create_task(adapter._keep_typing(CHAT_ID, interval=60))
    await _wait_for_presence_count(adapter, 1)

    await adapter.stop_typing(CHAT_ID)
    assert adapter._http_session.presence_states == ["composing"]

    task.cancel()
    await asyncio.wait_for(task, timeout=1)
    assert adapter._http_session.presence_states == ["composing", "paused"]


@pytest.mark.asyncio
async def test_stop_typing_prunes_done_leaked_owner_but_preserves_live_owner():
    session = _RecordingHTTPSession()
    adapter = _make_adapter(session)
    chat_key, _ = adapter._typing_presence_state(CHAT_ID)
    release_owner = asyncio.Event()
    owner = asyncio.create_task(release_owner.wait())
    adapter._typing_refresh_owners[chat_key] = {owner}

    await adapter.stop_typing(CHAT_ID)
    assert session.presence_states == []
    assert adapter._typing_refresh_owners[chat_key] == {owner}

    release_owner.set()
    await asyncio.wait_for(owner, timeout=1)
    assert owner.done()
    assert adapter._typing_refresh_owners[chat_key] == {owner}

    await adapter.stop_typing(CHAT_ID)

    assert session.presence_states == ["paused"]
    assert chat_key not in adapter._typing_refresh_owners


@pytest.mark.asyncio
async def test_slow_typing_request_for_one_chat_does_not_block_another_chat():
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    session = _RecordingHTTPSession(
        [
            _AsyncResponse(entered=first_entered, release=release_first),
            _AsyncResponse(entered=second_entered),
        ]
    )
    adapter = _make_adapter(session)

    first = asyncio.create_task(adapter.send_typing(CHAT_ID))
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    second = asyncio.create_task(adapter.send_typing(OTHER_CHAT_ID))

    try:
        await asyncio.wait_for(second_entered.wait(), timeout=0.5)
    finally:
        release_first.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

    assert session.presence_states == ["composing", "composing"]


@pytest.mark.asyncio
async def test_unsuccessful_pause_acknowledgement_is_not_cached():
    session = _RecordingHTTPSession(
        [
            _AsyncResponse(status=200, payload={"success": False}),
            _AsyncResponse(status=200, payload={"success": True}),
        ]
    )
    adapter = _make_adapter(session)

    await adapter.stop_typing(CHAT_ID)
    await adapter.stop_typing(CHAT_ID)
    await adapter.stop_typing(CHAT_ID)

    assert session.presence_states == ["paused", "paused"]


@pytest.mark.asyncio
async def test_non_success_http_pause_response_is_not_cached():
    session = _RecordingHTTPSession(
        [
            _AsyncResponse(status=503, payload={"error": "Not connected"}),
            _AsyncResponse(status=200, payload={"success": True}),
        ]
    )
    adapter = _make_adapter(session)

    await adapter.stop_typing(CHAT_ID)
    await adapter.stop_typing(CHAT_ID)
    await adapter.stop_typing(CHAT_ID)

    assert session.presence_states == ["paused", "paused"]


@pytest.mark.asyncio
async def test_bare_and_jid_chat_ids_share_foreground_ownership():
    adapter = _make_adapter()
    task = asyncio.create_task(adapter._keep_typing(CHAT_ID, interval=60))
    await _wait_for_presence_count(adapter, 1)

    await adapter.stop_typing(CHAT_JID)
    assert adapter._http_session.presence_states == ["composing"]

    task.cancel()
    await asyncio.wait_for(task, timeout=1)
    assert adapter._http_session.presence_states == ["composing", "paused"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_loss", ["connection_reset", "cancellation"])
async def test_applied_composing_with_lost_ack_invalidates_cached_pause(ack_loss):
    lost_ack = _LostComposingAckResponse(
        wait_for_cancellation=ack_loss == "cancellation"
    )
    session = _RecordingHTTPSession([_AsyncResponse(), lost_ack, _AsyncResponse()])
    adapter = _make_adapter(session)

    await adapter.stop_typing(CHAT_ID)
    assert session.presence_states == ["paused"]

    composing = asyncio.create_task(adapter.send_typing(CHAT_ID))
    await asyncio.wait_for(lost_ack.json_entered.wait(), timeout=1)
    if ack_loss == "cancellation":
        composing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await composing
    else:
        await composing

    await adapter.stop_typing(CHAT_ID)

    assert session.presence_states == ["paused", "composing", "paused"]