"""An oversized-file note never costs the message: the event the adapter dispatches (single file or
album) passes the real native admission, with the caption and every per-item note in its text."""

import weakref
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership
from gateway.session_authorities import active_authority
from gateway.session_ingress import admit_message
from gateway.session_ingress_context import native_callback, register_transport_home
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_documents import _make_document, _make_message
from tests.waiting import async_wait_until

TOO_BIG = 500 * 1024 * 1024


def _update(msg):
    return SimpleNamespace(update_id=1, message=msg, effective_message=msg, channel_post=None,
                           business_message=None, edited_business_message=None)


def _zip(caption=None, group=None):
    msg = _make_message(document=_make_document("report.zip", "application/zip", file_size=TOO_BIG),
                        caption=caption, media_group_id=group)
    # A real update always carries a date (the envelope stamps it) and no reply unless it is one.
    msg.date, msg.reply_to_message, msg.quote, msg.external_reply = datetime.now(UTC), None, None, None
    return msg


async def _dispatched(items):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={"media_group_wait_seconds": 0.05}))
    adapter.handle_message = AsyncMock()
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    for item in items:
        await adapter._handle_media_message(_update(item), MagicMock())
    await async_wait_until(lambda: adapter.handle_message.await_count, timeout=5)
    adapter.handle_message.assert_awaited_once()
    return adapter.handle_message.await_args.args[0]


async def _admit(tmp_path, monkeypatch, event):
    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_agent import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    config = GatewayConfig()
    config._runtime_profile_homes = (("default", root),)
    runner = GatewayRunner(config)
    process_ownership.reserve([root])
    try:
        await initialize_gateway_runtime(runner)
        register_transport_home(runner, None, root)
        transport = object.__new__(TelegramAdapter)
        runner.adapters = {Platform.TELEGRAM: transport}
        runner._is_user_authorized_for_source = lambda *a, **k: True
        event.source._transport_adapter_ref = weakref.ref(transport)
        with native_callback(runner, event, root, None):
            return await admit_message(active_authority(runner), event)
    finally:
        for authority in list(getattr(runner, "session_authorities", []) or []):
            authority.db.close()
        process_ownership.release(root)


@pytest.mark.asyncio
@pytest.mark.parametrize("items", [lambda: [_zip("here it is")],
                                   lambda: [_zip("here it is", "mg"), _zip(None, "mg")]], ids=["single", "album"])
async def test_oversized_file_message_passes_native_admission(tmp_path, monkeypatch, items):
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "docs")
    built = items()
    event = await _dispatched(built)
    assert event.text.count("here it is") == 1 and event.text.count("too large") == len(built), event.text

    reply = await _admit(tmp_path, monkeypatch, event)

    assert not (reply and "refused" in reply), reply
