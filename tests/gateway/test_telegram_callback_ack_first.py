"""A Telegram button tap is acknowledged fast, even while an agent turn is running."""

import asyncio
import contextlib
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _query(data: str, user_id: str = "777"):
    q = AsyncMock()
    q.data = data
    q.message = MagicMock()
    q.message.chat_id = 12345
    q.message.text = "Pick"
    q.from_user = MagicMock()
    q.from_user.id = user_id
    q.from_user.first_name = "Tester"
    q.answer = AsyncMock()
    q.edit_message_text = AsyncMock()
    return q

@pytest.mark.asyncio
async def test_tap_is_acknowledged_before_slow_resolution():
    """The ack reaches Telegram before the (possibly slow) resolve, so the button never spins."""
    from tools import clarify_gateway as cm
    adapter = _adapter()
    cm.register("cid1", "sk1", "Pick", ["a", "b"])
    adapter._clarify_state["cid1"] = "sk1"
    order = []
    q = _query("cl:cid1:0")
    q.answer = AsyncMock(side_effect=lambda **kw: order.append(("answer", time.monotonic())))

    async def slow_resolve(*_a, **_k):
        await asyncio.sleep(0.3)
        order.append(("resolved", time.monotonic()))
        return True

    update = MagicMock(); update.callback_query = q
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False), \
         patch("tools.clarify_gateway.resolve_gateway_clarify", side_effect=slow_resolve):
        await adapter._handle_callback_query(update, MagicMock())
    kinds = [k for k, _ in order]
    assert kinds[0] == "answer", kinds
    assert q.answer.await_count >= 1

@pytest.mark.asyncio
async def test_unauthorized_tap_still_gets_exactly_one_answer():
    """Pre-ack must not double-answer: a refused tap gets the denial, not an empty ack first."""
    from gateway.platforms.base import unauthorized_action_notice
    adapter = _adapter()
    adapter._clarify_state["cidX"] = "skX"
    q = _query("cl:cidX:0", user_id="999")
    update = MagicMock(); update.callback_query = q
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "777"}, clear=False):
        await adapter._handle_callback_query(update, MagicMock())
    assert q.answer.await_count == 1
    assert q.answer.call_args[1]["text"] == unauthorized_action_notice("telegram")

@pytest.mark.asyncio
async def test_tap_during_running_text_turn_is_answered_within_one_second():
    """A tap that arrives while a text turn holds the dispatcher must still be answered fast."""
    # The gateway conftest mocks ``telegram``; this test needs the real dispatcher.
    import importlib, sys
    for k in [k for k in sys.modules if k == "telegram" or k.startswith("telegram.")]:
        del sys.modules[k]
    real_ext = importlib.import_module("telegram.ext")
    Application, CallbackQueryHandler, MessageHandler, filters = (
        real_ext.Application, real_ext.CallbackQueryHandler, real_ext.MessageHandler, real_ext.filters)
    # Re-import the admission module so TelegramApplication subclasses the REAL Application.
    adm = importlib.reload(importlib.import_module("plugins.platforms.telegram.update_admission"))
    TelegramApplication = adm.TelegramApplication
    # Same for the adapter: its module-level handler classes were bound to the mock at import.
    adapter_mod = importlib.reload(importlib.import_module("plugins.platforms.telegram.adapter"))
    real_tg = importlib.import_module("telegram")

    def real_tg_user(uid):
        return real_tg.User(id=uid, is_bot=True, first_name="bot")
    # PLACEHOLDER_T3
    adapter = adapter_mod.TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = AsyncMock()
    app = Application.builder().token("1:x").application_class(
        TelegramApplication, {"adapter": adapter}).updater(None).concurrent_updates(8).build()
    # The fetcher names its tasks after ``bot.id``; no network, so seed the cached get_me() user
    # and mark the app initialised without running Application.initialize (which would hit the API).
    app.bot._bot_user = real_tg_user(1)
    app._initialized = True
    adapter._app = app
    ack_at = {}

    async def slow_text(update, context):
        await asyncio.sleep(2.0)

    async def tap(update, context):
        ack_at["t"] = time.monotonic()

    # Use the adapter's OWN registration (that is what ships); then swap the callbacks so the
    # test measures dispatch timing, not the handlers' bodies.
    adapter._register_handlers(app)
    for group in app.handlers.values():
        for h in group:
            name = type(h).__name__
            if name == "CallbackQueryHandler":
                h.callback = tap
            elif name == "MessageHandler" and getattr(h, "callback", None) is adapter._handle_text_message:
                h.callback = slow_text
    for h in app.handlers.get(0, []):
        if getattr(h, "callback", None) in (tap, slow_text):
            continue
        if type(h).__name__ == "MessageHandler":
            h.callback = slow_text
    # PLACEHOLDER_T3B
    import datetime
    CallbackQuery, Chat, Message, Update, User = (
        real_tg.CallbackQuery, real_tg.Chat, real_tg.Message, real_tg.Update, real_tg.User)
    u = User(id=777, is_bot=False, first_name="T"); c = Chat(id=12345, type="private")
    now = datetime.datetime.now(datetime.timezone.utc)
    text_upd = Update(update_id=1, message=Message(message_id=1, date=now, chat=c, from_user=u, text="hi"))
    cq = CallbackQuery(id="q1", from_user=u, chat_instance="ci", data="cl:a:0",
                       message=Message(message_id=2, date=now, chat=c, from_user=u))
    tap_upd = Update(update_id=2, callback_query=cq)
    t0 = time.monotonic()
    # Drive PTB's own fetcher loop (private, but that is where the serialisation lives).
    fetcher = asyncio.create_task(app._Application__update_fetcher())
    await app.update_queue.put(text_upd)
    await asyncio.sleep(0.05)
    await app.update_queue.put(tap_upd)
    await asyncio.sleep(2.5)
    fetcher.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await fetcher
    assert "t" in ack_at, "tap never dispatched"
    assert ack_at["t"] - t0 < 1.0, f"tap handled after {ack_at['t'] - t0:.2f}s (behind the text turn)"


def test_build_ptb_requests_enables_bounded_concurrent_updates():
    """The adapter's PTB builder must not keep its serial (1 update) default."""
    import inspect
    assert "builder = builder.concurrent_updates(8)" in inspect.getsource(TelegramAdapter.connect)


