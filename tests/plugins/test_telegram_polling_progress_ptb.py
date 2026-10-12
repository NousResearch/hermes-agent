"""Integration coverage for polling progress against the installed PTB runtime."""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
pytest.importorskip("telegram", reason="python-telegram-bot not installed")
from telegram import Update
from telegram.error import Conflict, TelegramError
from telegram.ext import (
    Application,
    ApplicationHandlerStop,
    MessageHandler,
    TypeHandler,
    filters,
)
from telegram.request import BaseRequest

from gateway.config import PlatformConfig
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
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


# ===========================================================================
# REAL PTB ingress across a real transient-OSError rebuild (offline)
# ===========================================================================
# Lives here, NOT under tests/gateway/: that package's conftest installs MagicMocks for
# telegram/telegram.ext/telegram.request, so a "real PTB" test is impossible there.

_BOUND_CHAT = -1002000000001
_OTHER_CHAT = -1002000000002
_OWNER = 4242
_STRANGER = 9999


class _OfflineRequest(BaseRequest):
    """The ONLY permitted Bot API surface: initialize, shutdown, read_timeout, getMe.

    The first ``getMe`` raises OSError once -- a real transient network failure, which is what
    makes ``_initialize_app_with_retries`` take its rebuild branch. Any other endpoint is a hard
    failure, so this test cannot silently reach the network.
    """

    def __init__(self, fail_first: bool = True) -> None:
        super().__init__()
        self._fail_first, self.calls = fail_first, []

    @property
    def read_timeout(self):
        return 1.0

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def do_request(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        self.calls.append(endpoint)
        if endpoint != "getMe":
            raise AssertionError(f"forbidden offline Bot API call: {endpoint}")
        if self._fail_first:
            self._fail_first = False
            raise OSError("transient getMe failure")
        return 200, json.dumps({
            "ok": True,
            "result": {"id": 1, "is_bot": True, "first_name": "T", "username": "t_bot"},
        }).encode()


def _room_ingress_factory(delivered, stopped):
    """The real ingress shape: bound-chat-only, owner-only, ApplicationHandlerStop in ``finally``."""
    def wire(application, adapter):
        async def incoming(update, context):
            try:
                message, sender = update.effective_message, update.effective_user
                if (message and sender and not sender.is_bot and sender.id == _OWNER
                        and message.chat_id == _BOUND_CHAT and message.text):
                    delivered.append(message.message_id)
                else:
                    stopped.append(message.message_id if message else None)
            finally:
                raise ApplicationHandlerStop
        application.add_handler(MessageHandler(filters.Chat(_BOUND_CHAT), incoming), group=-50)
    return wire


def _ingress_update(bot, *, message_id: int, chat_id: int, user_id: int, text: str = "hi"):
    return Update.de_json({
        "update_id": message_id,
        "message": {
            "message_id": message_id, "date": 1700000000,
            "chat": {"id": chat_id, "type": "supergroup"},
            "from": {"id": user_id, "is_bot": False, "first_name": "F"},
            "text": text,
        },
    }, bot)


class TestRealPluginIngressSurvivesInitRetry:
    """Generic rebuild + plugin dispatch, not full core routing.

    ``_register_handlers`` is a one-line sentinel on purpose: what is under test is that a rebuilt
    Application still dispatches plugin-owned traffic to the plugin (group -50) ahead of core
    (group 99), which is exactly what the retry branch was dropping.
    """

    def test_registered_ingress_still_receives_bound_chat_input_after_a_real_rebuild(
            self, tmp_path, monkeypatch):
        # No real profile discovery: both are set before the adapter is constructed.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
        delivered, stopped, core_fallback = [], [], []
        mgr = PluginManager()
        ctx = PluginContext(
            manifest=PluginManifest(
                name="shared-room-ingress", version="0.1.0", description="test"),
            manager=mgr)
        ctx.register_platform_handler("telegram", _room_ingress_factory(delivered, stopped))

        request = _OfflineRequest(fail_first=True)
        builder = (Application.builder().token("123456:TEST")
                   .request(request).get_updates_request(_OfflineRequest(fail_first=False)))
        adapter._app = builder.build()
        adapter._bot = adapter._app.bot
        original_app = adapter._app

        async def core(update, context):
            core_fallback.append(getattr(update.effective_message, "message_id", None))

        async def drive():
            try:
                with patch("hermes_cli.plugins.get_plugin_manager", return_value=mgr), \
                        patch.object(type(adapter), "_register_handlers",
                                     side_effect=lambda native: native.add_handler(
                                         TypeHandler(Update, core), group=99)):
                    # Wire the ORIGINAL app the way connect() does, so the rebuild is on trial.
                    adapter._wire_plugin_handlers(adapter._app)
                    adapter._register_handlers(adapter._app)
                    await adapter._initialize_app_with_retries(builder)
                app = adapter._app
                bot = app.bot
                # An initialized Application dispatches without pollers or background jobs.
                await app.process_update(
                    _ingress_update(bot, message_id=1, chat_id=_BOUND_CHAT, user_id=_OWNER))
                await app.process_update(
                    _ingress_update(bot, message_id=2, chat_id=_BOUND_CHAT, user_id=_STRANGER))
                await app.process_update(
                    _ingress_update(bot, message_id=3, chat_id=_OTHER_CHAT, user_id=_OWNER))
            finally:
                await adapter._app.shutdown()

        asyncio.run(drive())

        assert adapter._app is not original_app, "the transient OSError did not rebuild the app"
        assert request.calls == ["getMe", "getMe"], request.calls   # one failure, one success
        assert delivered == [1], "the bound-chat authorized input did not reach the plugin once"
        assert stopped == [2], "the bound-chat unauthorized input was not stopped by the plugin"
        assert core_fallback == [3], (
            "bound-chat input leaked to the core fallback, or the unrelated chat did not fall through")
