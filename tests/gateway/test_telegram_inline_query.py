"""Tests for the Telegram adapter's inline-query dispatch wrapper.

`_handle_inline_query` is a thin gate over the router: it honours the
`inline.enabled` config flag and forwards enabled queries to the router. The
router's own behaviour (matching, discovery) is covered in
test_telegram_inline_router.py.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(inline_cfg=None, router=None):
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    extra = {} if inline_cfg is None else {"inline": inline_cfg}
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._inline_router = router
    return adapter


def _update(query="hello"):
    iq = SimpleNamespace(
        query=query,
        from_user=SimpleNamespace(id=42),
        answer=AsyncMock(),
    )
    return SimpleNamespace(inline_query=iq), iq


@pytest.mark.asyncio
async def test_inline_disabled_skips_dispatch_and_answer():
    """inline.enabled: false silences inline handling — no dispatch, no answer."""
    router = SimpleNamespace(dispatch=AsyncMock())
    adapter = _make_adapter(inline_cfg={"enabled": False}, router=router)
    update, iq = _update()

    await adapter._handle_inline_query(update, None)

    router.dispatch.assert_not_awaited()
    iq.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_inline_enabled_dispatches_and_passes_results_through():
    router = SimpleNamespace(dispatch=AsyncMock(return_value=["R1"]))
    adapter = _make_adapter(inline_cfg={"enabled": True}, router=router)
    update, iq = _update("draw a cat")

    await adapter._handle_inline_query(update, None)

    router.dispatch.assert_awaited_once_with(42, "draw a cat")
    iq.answer.assert_awaited_once()
    assert iq.answer.await_args.args[0] == ["R1"]


@pytest.mark.asyncio
async def test_inline_enabled_by_default_when_unconfigured():
    """No inline config → enabled (the handler stays active unless explicitly off)."""
    router = SimpleNamespace(dispatch=AsyncMock(return_value=[]))
    adapter = _make_adapter(inline_cfg=None, router=router)
    update, _ = _update("x")

    await adapter._handle_inline_query(update, None)

    router.dispatch.assert_awaited_once()
