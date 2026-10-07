"""Behavior tests for requester-only Telegram group controls (ephemeral messages).

Covers the privacy lifecycle only: receiver mismatch, same ephemeral id scoped
across two chats, fail-closed on a missing requester, no public fallback on any
failure (uniform decline recognized by ``gateway.relay.egress.declined_send``),
and the actual edit/delete lifecycle. No forwarded-dict wire-shape assertions.
Adapter/gateway integration is parent-owned.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from gateway.relay.egress import declined_send  # noqa: E402
from plugins.platforms.telegram.telegram_private_controls import (  # noqa: E402
    EPH_PREFIX,
    PrivateControlError,
    PrivateQuery,
    TelegramPrivateControlsMixin,
    _PrivateRecord,
    parse_private_handle,
)


class _Bot:
    def __init__(self, send_result=None, send_error=None, api_error=None, api_result=None):
        self.send_calls = 0
        self.send_kwargs = None
        self.api_calls = []
        self.api_errors = []  # per-call: popped FIFO, else api_error
        self.api_result = api_result  # raw dict a real do_api_request returns
        self._send_result, self._send_error, self._api_error = send_result, send_error, api_error

    async def send_message(self, **kwargs):
        self.send_calls += 1
        self.send_kwargs = kwargs
        if self._send_error is not None:
            raise self._send_error
        return self._send_result

    async def do_api_request(self, method, api_kwargs=None):
        self.api_calls.append((method, api_kwargs))
        if self.api_errors:
            raise self.api_errors.pop(0)
        if self._api_error is not None:
            raise self._api_error
        return self.api_result if self.api_result is not None else True


def _adapter(bot=None, private_controls=True):
    cls = type("A", (TelegramPrivateControlsMixin, object), {"name": "telegram"})
    adapter = object.__new__(cls)
    adapter._private_controls = private_controls
    adapter._bot = bot if bot is not None else _Bot()
    return adapter


def _eph_msg(chat_id=-100, receiver=111, eph=7):
    return SimpleNamespace(
        message_id=0, chat_id=chat_id, chat=SimpleNamespace(id=chat_id),
        api_kwargs={"receiver_user": {"id": receiver}, "ephemeral_message_id": eph})


def _query(user_id, message):
    return SimpleNamespace(
        id="cbq-1", data="ea:once:1", from_user=SimpleNamespace(id=user_id, first_name="A"),
        message=message, answer=AsyncMock())


def _meta(**kw):
    base = {"telegram_requester_user_id": "111", "telegram_chat_type": "group",
            "telegram_callback_query_id": "cbq-1"}
    base.update(kw)
    return base


# -- gating -----------------------------------------------------------------


def test_private_control_requested_groups_forums_only():
    adapter = _adapter()
    ok = {c: adapter.private_control_requested(-100, _meta(telegram_chat_type=c))
          for c in ("group", "supergroup", "forum")}
    no = {c: adapter.private_control_requested(-100, _meta(telegram_chat_type=c))
          for c in ("dm", "private", "channel", "")}
    assert all(ok.values()) and not any(no.values())


def test_requested_is_requester_independent_and_flag_gated():
    adapter = _adapter()
    assert adapter.private_control_requested(-100, _meta(telegram_requester_user_id="")) is True
    assert _adapter(private_controls=False).private_control_requested(-100, _meta()) is False


# -- fail-closed sends -------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_requester_fails_closed_no_send():
    bot = _Bot(send_result=_eph_msg())
    adapter = _adapter(bot=bot)
    result = await adapter.send_private_control_prompt({"chat_id": -100, "text": "x"}, _meta(telegram_requester_user_id=""))
    assert result.success is False and declined_send(result) is True
    assert bot.send_calls == 0  # nothing sent, nothing leaked


@pytest.mark.asyncio
async def test_send_failures_are_uniform_declines_never_public_fallback():
    """API rejection, transient timeout, and a 200-without-ephemeral-id are all
    the SAME privacy outcome: decline, no retry, no second send, no public post."""
    for bot in (_Bot(send_error=RuntimeError("Bad Request: not a chat admin")),
                _Bot(send_error=TimeoutError()),
                _Bot(send_result=SimpleNamespace(message_id=55, chat_id=-100))):
        adapter = _adapter(bot=bot)
        result = await adapter.send_private_control_prompt({"chat_id": -100, "text": "x"}, _meta())
        assert result.success is False and result.retryable is False
        assert declined_send(result) is True  # recognized by the gateway classifier
        assert bot.send_calls <= 1 and bot.api_calls == []


@pytest.mark.asyncio
async def test_send_private_control_raises_on_failure_never_none():
    adapter = _adapter(bot=_Bot(send_error=RuntimeError("boom")))
    with pytest.raises(PrivateControlError):
        await adapter._send_private_control({"chat_id": -100, "text": "x"}, _meta())

# -- lifecycle: handle, registry scoping, edit/delete -------------------------


@pytest.mark.asyncio
async def test_handle_is_opaque_and_feeds_on_sent():
    bot = _Bot(send_result=_eph_msg())
    adapter = _adapter(bot=bot)
    seen = {}
    result = await adapter.send_private_control_prompt(
        {"chat_id": -100, "text": "x"}, _meta(), on_sent=lambda m: seen.update(id=m.message_id))
    assert result.success is True
    assert seen["id"] == f"{EPH_PREFIX}111:7" and result.message_id == seen["id"]


@pytest.mark.asyncio
async def test_same_ephemeral_id_in_two_chats_stays_scoped():
    bot = _Bot(send_result=_eph_msg())
    adapter = _adapter(bot=bot)
    for chat in (-100, -200):
        await adapter._send_private_control({"chat_id": chat, "text": "x"}, _meta())
    for chat in (-100, -200):
        record = adapter.resolve_private_control(f"{EPH_PREFIX}111:7", chat_id=chat)
        assert record is not None and record["chat_id"] == chat
        ok = await adapter.edit_ephemeral_control_text(f"{EPH_PREFIX}111:7", "✓", chat_id=chat)
        assert ok is True
    chats = {payload["chat_id"] for method, payload in bot.api_calls}
    assert chats == {-100, -200}  # both scoped, no cross-talk


def test_edit_delete_never_target_regular_message_id():
    adapter = _adapter()
    for obj in ("55", SimpleNamespace(message_id=55, chat_id=-100, api_kwargs={})):
        assert adapter.resolve_private_control(obj, chat_id=-100) is None
    assert parse_private_handle(f"{EPH_PREFIX}111:0") is None  # message_id 0 placeholder


@pytest.mark.asyncio
async def test_edit_and_delete_use_ephemeral_endpoints_receiver_separate():
    bot = _Bot(send_result=_eph_msg())
    adapter = _adapter(bot=bot)
    await adapter._send_private_control({"chat_id": -100, "text": "x"}, _meta())
    assert await adapter.edit_ephemeral_control_text(f"{EPH_PREFIX}111:7", "✓ Approved", chat_id=-100) is True
    method, payload = bot.api_calls[0]
    assert method == "editEphemeralMessageText"
    assert payload["receiver_user_id"] == 111 and payload["chat_id"] == -100 and "message_id" not in payload
    assert await adapter.delete_ephemeral_control(f"{EPH_PREFIX}111:7", chat_id=-100) is True
    assert bot.api_calls[1][0] == "deleteEphemeralMessage"


@pytest.mark.asyncio
async def test_edit_failure_degrades_to_nothing_never_public():
    bot = _Bot(send_result=_eph_msg(), api_error=RuntimeError("Bad Request"))
    adapter = _adapter(bot=bot)
    await adapter._send_private_control({"chat_id": -100, "text": "x"}, _meta())
    assert await adapter.edit_ephemeral_control_text(f"{EPH_PREFIX}111:7", "✓", chat_id=-100) is False
    assert bot.send_calls == 1  # only the original ephemeral send


# -- inbound: recognition, receiver gate, facade ------------------------------


@pytest.mark.asyncio
async def test_wrap_recognizes_ephemeral_ignores_regular_and_gates():
    adapter = _adapter()
    q = _query(111, _eph_msg())
    wrapped = adapter.wrap_private_control_query(q)
    assert isinstance(wrapped, PrivateQuery)
    assert adapter.wrap_private_control_query(_query(111, SimpleNamespace(message_id=55, chat_id=-100, api_kwargs={}))) is None
    # None (regular public control) passes — the allowlist decides; receiver passes silently.
    assert await adapter.gate_private_control_query(None) is True
    assert await adapter.gate_private_control_query(wrapped) is True
    q.answer.assert_not_called()


@pytest.mark.asyncio
async def test_facade_edit_by_receiver_refused_without_endpoint_call():
    bot = _Bot()
    adapter = _adapter(bot=bot)
    wrapped = adapter.wrap_private_control_query(_query(222, _eph_msg()))
    assert await wrapped.edit_message_text("Approved") is False
    assert await wrapped.delete() is False
    assert bot.api_calls == []  # non-receiver: no edit, no delete, no public path


@pytest.mark.asyncio
async def test_facade_edit_by_receiver_routes_to_private_endpoint():
    bot = _Bot()
    adapter = _adapter(bot=bot)
    wrapped = adapter.wrap_private_control_query(_query(111, _eph_msg()))
    assert await wrapped.edit_message_text("✓ Approved") is True
    method, payload = bot.api_calls[0]
    assert method == "editEphemeralMessageText"
    assert (payload["chat_id"], payload["receiver_user_id"], payload["ephemeral_message_id"]) == (-100, 111, 7)
    assert bot.send_calls == 0  # never a public sendMessage


# -- rich embedding: flag gating and format selection -------------------------


def _rich_adapter(bot=None, *, enabled=True, helper=None, classifier=None):
    """Adapter with the rich-control surface the module negotiates with.

    ``helper`` defaults to the real one so assertions check actual payload
    SHAPE (InputRichMessage with tg-button-row buttons), not a mock echo.
    """
    cls = type("R", (TelegramPrivateControlsMixin, object), {"name": "telegram"})
    adapter = object.__new__(cls)
    adapter._private_controls = True
    adapter._bot = bot if bot is not None else _Bot(send_result=_eph_msg())
    adapter._rich_controls_enabled = lambda: enabled
    from plugins.platforms.telegram.telegram_rich_controls import rich_control_payload
    adapter._rich_control_payload = helper or rich_control_payload
    if classifier is not None:
        adapter._is_rich_fallback_error = classifier
    return adapter


@pytest.mark.asyncio
async def test_send_rich_flag_off_keeps_plain_ephemeral_send():
    """rich_controls OFF → no sendRichMessage, no rich fields on the plain send."""
    bot = _Bot(send_result=_eph_msg())
    adapter = _rich_adapter(bot, enabled=False)
    await adapter._send_private_control(
        {"chat_id": -100, "text": "Approve?", "reply_markup": {"inline_keyboard": [[{"text": "Yes", "callback_data": "ea:1"}]]}},
        _meta())
    assert bot.send_calls == 1 and bot.api_calls == []  # plain sendMessage only
    assert bot.send_kwargs["api_kwargs"]["ephemeral_message_parameters"]["receiver_user_id"] == 111
    assert "rich_message" not in bot.send_kwargs


@pytest.mark.asyncio
async def test_send_rich_flag_on_uses_sendrichmessage_with_ephemeral_parameters():
    """rich_controls ON → ONE sendRichMessage carrying ephemeral_message_parameters
    and native tg-button-row buttons; never a public sendMessage after it.
    The runtime return here is a raw dict — what ``do_api_request`` gives
    without ``return_type`` (PTB 22.8)."""
    bot = _Bot(api_result={"message_id": 0, "chat_id": -100,
                           "api_kwargs": {"receiver_user": {"id": 111}, "ephemeral_message_id": 7}})
    adapter = _rich_adapter(bot)
    handle = await adapter._send_private_control(
        {"chat_id": -100, "text": "Approve?", "parse_mode": "MarkdownV2",
         "reply_markup": {"inline_keyboard": [[{"text": "Approve", "callback_data": "ea:yes:1"}]]}},
        _meta())
    assert bot.send_calls == 0  # no public sendMessage
    method, payload = bot.api_calls[0]
    assert method == "sendRichMessage"
    assert payload["ephemeral_message_parameters"]["receiver_user_id"] == 111
    assert payload["ephemeral_message_parameters"]["callback_query_id"] == "cbq-1"
    assert "<tg-button" in payload["rich_message"]["html"] and "Approve?" in payload["rich_message"]["html"]
    for absent in ("text", "parse_mode", "reply_markup"):  # no contradictory content fields
        assert absent not in payload
    assert handle.message_id == f"{EPH_PREFIX}111:7"
    assert len(bot.api_calls) == 1  # exactly one API call


@pytest.mark.asyncio
async def test_send_rich_permanent_rejection_falls_back_to_plain_ephemeral():
    """Permanent (capability) sendRichMessage rejection → exactly one plain
    ephemeral sendMessage; a transient rejection must NOT re-send."""
    bot = _Bot(send_result=_eph_msg())
    bot.api_errors.append(RuntimeError("Bad Request: unsupported method sendRichMessage"))
    adapter = _rich_adapter(bot, classifier=lambda exc: "unsupported" in str(exc).lower())
    handle = await adapter._send_private_control({"chat_id": -100, "text": "Approve?"}, _meta())
    assert bot.send_calls == 1  # the plain fallback send
    assert bot.api_calls[0][0] == "sendRichMessage"
    assert handle.message_id == f"{EPH_PREFIX}111:7"

    # Transient: the rich request may have landed — fail closed, never re-send.
    bot2 = _Bot(send_result=_eph_msg())
    bot2.api_errors.append(TimeoutError("peer closed"))
    adapter2 = _rich_adapter(bot2, classifier=lambda exc: False)
    with pytest.raises(PrivateControlError):
        await adapter2._send_private_control({"chat_id": -100, "text": "Approve?"}, _meta())
    assert bot2.send_calls == 0  # no second, possibly duplicate send


@pytest.mark.asyncio
async def test_send_rich_success_without_ephemeral_id_fails_closed():
    """A sendRichMessage 200 without an ephemeral id is unmanageable — decline,
    never fall through to a second send (the rich message may be delivered)."""
    bot = _Bot(api_result={"message_id": 0, "chat_id": -100, "api_kwargs": {}})
    adapter = _rich_adapter(bot)
    with pytest.raises(PrivateControlError):
        await adapter._send_private_control({"chat_id": -100, "text": "x"}, _meta())
    assert bot.send_calls == 0


@pytest.mark.asyncio
async def test_send_preserves_caller_api_kwargs_non_destructive():
    """A caller's own api_kwargs keys survive the ephemeral merge."""
    bot = _Bot(send_result=_eph_msg())
    adapter = _rich_adapter(bot, enabled=False)
    await adapter._send_private_control(
        {"chat_id": -100, "text": "x", "api_kwargs": {"business_connection_id": "bc-1"}}, _meta())
    merged = bot.send_kwargs["api_kwargs"]
    assert merged["business_connection_id"] == "bc-1"
    assert merged["ephemeral_message_parameters"]["receiver_user_id"] == 111


@pytest.mark.asyncio
async def test_facade_rich_edit_exclusive_rich_or_text_never_both():
    """Rich edit payload carries rich_message ALONE; flag off keeps the plain
    text/parse_mode/reply_markup edit; both never ride one payload."""
    bot = _Bot()
    adapter = _rich_adapter(bot)
    wrapped = adapter.wrap_private_control_query(_query(111, _eph_msg()))
    markup = {"inline_keyboard": [[{"text": "Approve", "callback_data": "ea:yes:1"}]]}
    assert await wrapped.edit_message_text("✓ Approved", parse_mode="MarkdownV2", reply_markup=markup) is True
    method, payload = bot.api_calls[0]
    assert method == "editEphemeralMessageText"
    assert "<tg-button" in payload["rich_message"]["html"]
    for absent in ("text", "parse_mode", "reply_markup"):  # exclusive content form
        assert absent not in payload

    plain_bot = _Bot()
    plain_adapter = _rich_adapter(plain_bot, enabled=False)
    plain = plain_adapter.wrap_private_control_query(_query(111, _eph_msg()))
    assert await plain.edit_message_text("✓", parse_mode="MarkdownV2", reply_markup=markup) is True
    _, plain_payload = plain_bot.api_calls[0]
    assert plain_payload["text"] == "✓" and plain_payload["parse_mode"] == "MarkdownV2"
    assert plain_payload["reply_markup"] == markup and "rich_message" not in plain_payload


@pytest.mark.asyncio
async def test_facade_rich_edit_transient_failure_never_races_plain_edit():
    """A transient rich-edit failure returns False with NO plain edit attempt;
    a permanent one retries the SAME message as plain text."""
    bot = _Bot()
    bot.api_errors.append(TimeoutError("peer closed"))
    adapter = _rich_adapter(bot, classifier=lambda exc: False)
    wrapped = adapter.wrap_private_control_query(_query(111, _eph_msg()))
    assert await wrapped.edit_message_text("✓ Approved") is False
    assert len(bot.api_calls) == 1  # only the rich attempt — no racing plain edit

    perm_bot = _Bot()
    perm_bot.api_errors.append(RuntimeError("Bad Request: rich not supported"))
    perm_adapter = _rich_adapter(perm_bot, classifier=lambda exc: "rich not supported" in str(exc))
    perm = perm_adapter.wrap_private_control_query(_query(111, _eph_msg()))
    assert await perm.edit_message_text("✓ Approved") is True
    assert perm_bot.api_calls[0][1]["rich_message"] is not None
    assert perm_bot.api_calls[1][1]["text"] == "✓ Approved"  # the plain retry, same target
    assert perm_bot.api_calls[1][0] == "editEphemeralMessageText"


@pytest.mark.asyncio
async def test_facade_rich_flag_off_never_uses_rich_helper():
    """Flag off → the rich helper is never even consulted."""
    calls = {"n": 0}

    def helper(text, parse_mode=None, reply_markup=None):
        calls["n"] += 1
        return {"html": "<b>should not happen</b>"}

    bot = _Bot()
    adapter = _rich_adapter(bot, enabled=False, helper=helper)
    wrapped = adapter.wrap_private_control_query(_query(111, _eph_msg()))
    assert await wrapped.edit_message_text("✓") is True
    assert calls["n"] == 0 and "rich_message" not in bot.api_calls[0][1]


# -- lifecycle: every Telegram call is deadline-bounded ------------------------


class _HangingBot:
    """Bot whose every Telegram call never completes."""

    def __init__(self):
        self.send_calls = 0
        self.api_calls = 0

    async def send_message(self, **kwargs):
        self.send_calls += 1
        await asyncio.Event().wait()  # never set

    async def do_api_request(self, method, api_kwargs=None):
        self.api_calls += 1
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_send_edit_and_delete_are_all_deadline_bounded(monkeypatch):
    """A wedged Telegram call surfaces as a controlled failure within the
    deadline instead of hanging forever — send declines, edit/delete return False."""
    import plugins.platforms.telegram.telegram_private_controls as _mod
    monkeypatch.setattr(_mod, "_TEXT_SEND_DEADLINE", 0.05)  # real 60s would be absurd
    bot = _HangingBot()
    adapter = _rich_adapter(bot, enabled=False)
    result = await adapter.send_private_control_prompt({"chat_id": -100, "text": "x"}, _meta())
    assert result.success is False and declined_send(result) is True
    assert bot.send_calls == 1

    # edit + delete against a valid record: bounded → False, never hang.
    adapter._private_records()[(-100, 111, 7)] = _PrivateRecord(-100, 111, 7)
    handle = f"{EPH_PREFIX}111:7"
    assert await adapter.edit_ephemeral_control_text(handle, "✓", chat_id=-100) is False
    assert await adapter.delete_ephemeral_control(handle, chat_id=-100) is False
    assert bot.api_calls == 2
