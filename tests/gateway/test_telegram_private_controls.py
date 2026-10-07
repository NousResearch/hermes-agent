"""Behavior tests for requester-only Telegram group controls (ephemeral messages).

Covers the privacy lifecycle only: receiver mismatch, same ephemeral id scoped
across two chats, fail-closed on a missing requester, no public fallback on any
failure (uniform decline recognized by ``gateway.relay.egress.declined_send``),
and the actual edit/delete lifecycle. No forwarded-dict wire-shape assertions.
Adapter/gateway integration is parent-owned.
"""

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
    parse_private_handle,
)


class _Bot:
    def __init__(self, send_result=None, send_error=None, api_error=None):
        self.send_calls = 0
        self.api_calls = []
        self._send_result, self._send_error, self._api_error = send_result, send_error, api_error

    async def send_message(self, **kwargs):
        self.send_calls += 1
        if self._send_error is not None:
            raise self._send_error
        return self._send_result

    async def do_api_request(self, method, api_kwargs=None):
        self.api_calls.append((method, api_kwargs))
        if self._api_error is not None:
            raise self._api_error
        return True


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
