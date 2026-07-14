"""Integration coverage for polling progress against the installed PTB runtime."""

import asyncio

import pytest
pytest.importorskip("telegram", reason="python-telegram-bot not installed")
from telegram.error import Conflict, TelegramError
from telegram.request import BaseRequest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class _GeneralRequest(BaseRequest):
    @property
    def read_timeout(self):
        return 10

    async def initialize(self):
        return None

    async def shutdown(self):
        return None

    async def do_request(self, url, method, request_data=None, **_kwargs):
        if url.endswith("/getMe"):
            return (
                200,
                b'{"ok":true,"result":{"id":1,"is_bot":true,'
                b'"first_name":"Test","username":"test_bot"}}',
            )
        return 200, b'{"ok":true,"result":true}'


class _GetUpdatesRequest(BaseRequest):
    def __init__(self):
        self.initial_conflict_sent = False
        self.replacement_enabled = False
        self.replacement_progress_sent = False
        self.cleanup_calls = 0
        self.block = asyncio.Event()

    @property
    def read_timeout(self):
        return 10

    async def initialize(self):
        return None

    async def shutdown(self):
        return None

    async def do_request(self, url, method, request_data=None, **_kwargs):
        parameters = request_data.parameters if request_data is not None else {}
        timeout = parameters.get("timeout")
        timeout_seconds = (
            timeout.total_seconds() if hasattr(timeout, "total_seconds") else timeout
        )
        if timeout_seconds == 0:
            self.cleanup_calls += 1
            return 200, b'{"ok":true,"result":[]}'
        if not self.initial_conflict_sent:
            self.initial_conflict_sent = True
            return (
                409,
                b'{"ok":false,"error_code":409,'
                b'"description":"Conflict: another getUpdates request"}',
            )
        if self.replacement_enabled and not self.replacement_progress_sent:
            self.replacement_progress_sent = True
            return 200, b'{"ok":true,"result":[]}'
        await self.block.wait()
        return 200, b'{"ok":true,"result":[]}'


class _EnvelopeRequest(BaseRequest):
    def __init__(self, payload):
        self.payload = payload

    @property
    def read_timeout(self):
        return 10

    async def initialize(self):
        return None

    async def shutdown(self):
        return None

    async def do_request(self, url, method, request_data=None, **_kwargs):
        return 200, self.payload


class _SlottedEnvelopeRequest(BaseRequest):
    """A getUpdates request with no instance ``__dict__``.

    Reproduces PTB's real HTTPXRequest shape on Python 3.13, where every
    class in the MRO defines ``__slots__`` and instances therefore reject an
    instance-attribute ``do_request`` monkey-patch as "read-only" (#64482).
    ``__slots__`` names the payload so the double needs no ``__dict__``.
    """

    __slots__ = ("_payload",)

    def __init__(self, payload):
        self._payload = payload

    @property
    def read_timeout(self):
        return 10

    async def initialize(self):
        return None

    async def shutdown(self):
        return None

    async def do_request(self, url, method, request_data=None, **_kwargs):
        return 200, self._payload


async def _cancel_task(task):
    if task is None or task.done():
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_real_base_request_invalid_200_body_cannot_record_progress():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 4
    adapter._polling_conflict_count = 3
    request = adapter._instrument_polling_request(_EnvelopeRequest(b"not-json"))
    context_token = tg_adapter._POLLING_GENERATION_CONTEXT.set(generation)

    try:
        with pytest.raises(TelegramError, match="Invalid server response"):
            await request.post("https://api.telegram.org/bot-token/getUpdates")
    finally:
        tg_adapter._POLLING_GENERATION_CONTEXT.reset(context_token)

    assert not progress.is_set()
    assert adapter._polling_network_error_count == 4
    assert adapter._polling_conflict_count == 3
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_real_base_request_bom_rejected_by_ptb_cannot_record_progress():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 4
    adapter._polling_conflict_count = 3
    request = adapter._instrument_polling_request(
        _EnvelopeRequest(b'\xef\xbb\xbf{"ok":true,"result":[]}')
    )
    context_token = tg_adapter._POLLING_GENERATION_CONTEXT.set(generation)

    try:
        with pytest.raises(TelegramError, match="Invalid server response"):
            await request.post("https://api.telegram.org/bot-token/getUpdates")
    finally:
        tg_adapter._POLLING_GENERATION_CONTEXT.reset(context_token)

    assert not progress.is_set()
    assert adapter._polling_network_error_count == 4
    assert adapter._polling_conflict_count == 3
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_real_base_request_ptb_replacement_decode_records_progress():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 4
    adapter._polling_conflict_count = 3
    request = adapter._instrument_polling_request(
        _EnvelopeRequest(b'{"ok":true,"result":[],"note":"\xff"}')
    )
    context_token = tg_adapter._POLLING_GENERATION_CONTEXT.set(generation)

    try:
        result = await request.post("https://api.telegram.org/bot-token/getUpdates")
    finally:
        tg_adapter._POLLING_GENERATION_CONTEXT.reset(context_token)

    assert result == []
    assert progress.is_set()
    assert adapter._polling_network_error_count == 0
    assert adapter._polling_conflict_count == 0
    assert adapter._send_path_degraded is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "missing_result"),
    [
        (b'{"ok":false,"result":[]}', False),
        (b'{"ok":true}', True),
    ],
)
async def test_real_base_request_unsuccessful_200_envelope_cannot_record_progress(
    payload, missing_result
):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 4
    adapter._polling_conflict_count = 3
    request = adapter._instrument_polling_request(_EnvelopeRequest(payload))
    context_token = tg_adapter._POLLING_GENERATION_CONTEXT.set(generation)

    try:
        if missing_result:
            with pytest.raises(KeyError, match="result"):
                await request.post("https://api.telegram.org/bot-token/getUpdates")
        else:
            assert await request.post(
                "https://api.telegram.org/bot-token/getUpdates"
            ) == []
    finally:
        tg_adapter._POLLING_GENERATION_CONTEXT.reset(context_token)

    assert not progress.is_set()
    assert adapter._polling_network_error_count == 4
    assert adapter._polling_conflict_count == 3
    assert adapter._send_path_degraded is True


@pytest.mark.asyncio
async def test_real_base_request_valid_success_envelope_records_progress():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 4
    adapter._polling_conflict_count = 3
    request = adapter._instrument_polling_request(
        _EnvelopeRequest(b'{"ok":true,"result":[]}')
    )
    context_token = tg_adapter._POLLING_GENERATION_CONTEXT.set(generation)

    try:
        result = await request.post(
            "https://api.telegram.org/bot-token/getUpdates"
        )
    finally:
        tg_adapter._POLLING_GENERATION_CONTEXT.reset(context_token)

    assert result == []
    assert progress.is_set()
    assert adapter._polling_network_error_count == 0
    assert adapter._polling_conflict_count == 0
    assert adapter._send_path_degraded is False


@pytest.mark.asyncio
async def test_slotted_request_instrumented_without_read_only_error():
    """Instrumenting a __slots__ request must not raise, and must still record.

    Regression for #64482: PTB's HTTPXRequest carries no instance ``__dict__``
    on Python 3.13, so the old ``request.do_request = wrapper`` monkey-patch
    raised ``AttributeError: ... 'do_request' is read-only`` and broke every
    Telegram connect. The subclass re-tag must instrument such an instance and
    still observe getUpdates progress.
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))
    generation, progress = adapter._begin_polling_generation()
    adapter._polling_network_error_count = 4
    adapter._polling_conflict_count = 3

    request = _SlottedEnvelopeRequest(b'{"ok":true,"result":[]}')
    # Guard the premise: the slotted instance rejects the old monkey-patch,
    # exactly as PTB's HTTPXRequest does on Python 3.13.
    with pytest.raises(AttributeError, match="read-only"):
        request.do_request = lambda *a, **k: None

    instrumented = adapter._instrument_polling_request(request)
    context_token = tg_adapter._POLLING_GENERATION_CONTEXT.set(generation)
    try:
        result = await instrumented.post(
            "https://api.telegram.org/bot-token/getUpdates"
        )
    finally:
        tg_adapter._POLLING_GENERATION_CONTEXT.reset(context_token)

    assert result == []
    assert progress.is_set()
    assert adapter._polling_network_error_count == 0
    assert adapter._polling_conflict_count == 0
    assert adapter._send_path_degraded is False


@pytest.mark.asyncio
async def test_real_ptb_stop_cleanup_cannot_heal_recovery_generation():
    assert tg_adapter.TELEGRAM_AVAILABLE is True
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:test-token"))
    polling_request = _GetUpdatesRequest()
    app = (
        tg_adapter.Application.builder()
        .token("123456:test-token")
        .request(_GeneralRequest())
        .get_updates_request(adapter._instrument_polling_request(polling_request))
        .build()
    )
    adapter._app = app
    adapter._polling_network_error_count = 4
    adapter._polling_conflict_count = 3
    callback_called = asyncio.Event()
    recovery_task = None

    async def stop_for_recovery():
        await app.updater.stop()

    def schedule_recovery(error):
        nonlocal recovery_task
        assert isinstance(error, Conflict)
        recovery_task = asyncio.create_task(stop_for_recovery())
        callback_called.set()

    await app.initialize()
    try:
        await adapter._start_polling_once(
            app,
            drop_pending_updates=False,
            error_callback=schedule_recovery,
        )
        generation = adapter._polling_generation
        progress = adapter._polling_progress_event
        await asyncio.wait_for(callback_called.wait(), timeout=2)
        await asyncio.wait_for(recovery_task, timeout=3)

        assert polling_request.cleanup_calls == 1
        assert not progress.is_set()
        assert adapter._polling_network_error_count == 4
        assert adapter._polling_conflict_count == 3
        assert adapter._send_path_degraded is True

        polling_request.replacement_enabled = True
        await adapter._start_polling_once(
            app,
            drop_pending_updates=False,
            error_callback=schedule_recovery,
        )
        replacement_generation = adapter._polling_generation
        replacement_progress = adapter._polling_progress_event
        await asyncio.wait_for(replacement_progress.wait(), timeout=2)

        assert replacement_generation == generation + 1
        assert adapter._polling_network_error_count == 0
        assert adapter._polling_conflict_count == 0
        assert adapter._send_path_degraded is False
    finally:
        polling_request.block.set()
        if app.updater.running:
            await app.updater.stop()
        await _cancel_task(adapter._polling_progress_verifier_task)
        await app.shutdown()
