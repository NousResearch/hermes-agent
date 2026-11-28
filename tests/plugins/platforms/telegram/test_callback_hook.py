"""``telegram:callback`` — a gateway hook for third-party inline buttons (#153232).

Integrations that send their own inline buttons used to have no supported way to receive the
press: any ``callback_data`` prefix outside the built-in tables ended at "unhandled callback
query", and the only workaround was patching ``_handle_callback_query`` (which upstream
refactors keep breaking). The adapter now emits a collect-style hook before the built-in
dispatch and stops when a handler claims the tap.

The tests pin the three things a claim must guarantee: zero cost when nobody subscribes, a
fail-closed sender gate before any third-party code, and built-ins untouched unless claimed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import CALLBACK_HOOK_EVENT, TelegramAdapter


class _FakeRegistry:
    """Stand-in for HookRegistry/ProfileHookRegistries with the two methods the adapter uses."""

    def __init__(self, results=None, handlers=True, boom=None):
        self.results = list(results or [])
        self._handlers = handlers
        self._boom = boom
        self.contexts: list[dict] = []
        self.events: list[str] = []

    def has_handlers(self, event_type: str) -> bool:
        return self._handlers

    async def emit_collect(self, event_type: str, context=None):
        if self._boom is not None:
            raise self._boom
        self.events.append(event_type)
        self.contexts.append(context or {})
        return self.results


def _make_adapter(registry=None, **extra) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra=extra))
    adapter._bot = AsyncMock()
    if registry is not None:
        adapter._callback_hook_registry = registry
    return adapter


def _query(data: str = "deploy:42", user_id: int = 99):
    query = AsyncMock(
        data=data,
        from_user=SimpleNamespace(id=user_id, first_name="Alice"),
        message=MagicMock(chat_id=1, chat_type="private", message_id=777),
    )
    return query


@pytest.fixture(autouse=True)
def _allow_all(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")


# ---------------------------------------------------------------------------
# no subscriber: nothing changes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_handlers_is_a_no_op_and_never_gates():
    adapter = _make_adapter(_FakeRegistry(handlers=False))
    query = _query()
    gate = AsyncMock(return_value=True)
    adapter._callback_authorized = gate

    claimed = await adapter._emit_callback_hook(query, query.data, adapter._callback_ctx(query))

    assert claimed is False
    gate.assert_not_awaited()          # no gate cost without a listener
    adapter._bot.answer_callback_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_unhandled_custom_data_still_reaches_the_builtin_ack():
    adapter = _make_adapter(_FakeRegistry(results=[]))
    query = _query("custom:x")

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    query.answer.assert_awaited_once()  # the existing "no handler owns it" ack


# ---------------------------------------------------------------------------
# fail-closed authorization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unadmitted_sender_never_reaches_the_hook(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123")
    registry = _FakeRegistry(results=[True])
    adapter = _make_adapter(registry)
    query = _query("deploy:42", user_id=999)

    claimed = await adapter._emit_callback_hook(query, query.data, adapter._callback_ctx(query))

    assert claimed is True                       # dispatch stops
    assert registry.contexts == []               # third-party code never ran
    query.answer.assert_awaited_once()           # denial toast, same as a built-in refusal


# ---------------------------------------------------------------------------
# claim semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_true_claim_stops_dispatch_and_carries_the_full_context():
    registry = _FakeRegistry(results=[True])
    adapter = _make_adapter(registry)
    query = _query("deploy:42")

    claimed = await adapter._emit_callback_hook(query, query.data, adapter._callback_ctx(query))

    assert claimed is True
    assert registry.events == [CALLBACK_HOOK_EVENT]
    ctx = registry.contexts[0]
    assert ctx["platform"] == "telegram"
    assert ctx["data"] == "deploy:42"
    assert ctx["chat_id"] == 1
    assert ctx["message_id"] == 777
    assert ctx["user_id"] == "99"
    assert ctx["user_name"] == "Alice"
    assert ctx["query"] is query
    assert ctx["adapter"] is adapter
    assert ctx["bot"] is adapter._bot


@pytest.mark.asyncio
async def test_dict_claim_is_honoured():
    adapter = _make_adapter(_FakeRegistry(results=[{"handled": True}]))
    query = _query()

    assert await adapter._emit_callback_hook(query, query.data, adapter._callback_ctx(query)) is True


@pytest.mark.asyncio
async def test_none_result_falls_through_to_the_builtins():
    adapter = _make_adapter(_FakeRegistry(results=[None, {}]))
    query = _query()

    assert await adapter._emit_callback_hook(query, query.data, adapter._callback_ctx(query)) is False


@pytest.mark.asyncio
async def test_claiming_hook_shadows_a_builtin_prefix():
    registry = _FakeRegistry(results=[True])
    adapter = _make_adapter(registry)
    adapter._handle_exec_approval_callback = AsyncMock()
    query = _query("ea:once:4")

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    adapter._handle_exec_approval_callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_builtin_prefix_still_dispatches_without_a_claim():
    registry = _FakeRegistry(results=[None])
    adapter = _make_adapter(registry)
    adapter._handle_exec_approval_callback = AsyncMock()
    query = _query("ea:once:4")

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    adapter._handle_exec_approval_callback.assert_awaited_once()


# ---------------------------------------------------------------------------
# failures never eat a tap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raising_handler_falls_through_to_the_builtins():
    adapter = _make_adapter(_FakeRegistry(boom=RuntimeError("boom")))
    adapter._handle_exec_approval_callback = AsyncMock()
    query = _query("ea:once:4")

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    adapter._handle_exec_approval_callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_broken_registry_never_breaks_the_tap():
    adapter = _make_adapter()

    def _explode():
        raise RuntimeError("hooks dir unreadable")

    adapter._hook_registry = _explode
    adapter._handle_exec_approval_callback = AsyncMock()
    query = _query("ea:once:4")

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    adapter._handle_exec_approval_callback.assert_awaited_once()
