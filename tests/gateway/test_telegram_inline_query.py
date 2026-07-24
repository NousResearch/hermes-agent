"""Tests for the Telegram adapter's inline-query dispatch wrapper.

`_handle_inline_query` is a thin gate over the router: it honours the
`inline.enabled` config flag and forwards enabled queries to the router. The
router's own behaviour (matching, discovery) is covered in
test_telegram_inline_router.py.
"""
from types import SimpleNamespace
from typing import Any
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


@pytest.mark.asyncio
async def test_inline_end_to_end_with_stub_executor(tmp_path):
    """Full path with a stand-in executor (real executors are user-space): a matching
    query routes through a real router to a registered executor and its results reach
    answer()."""
    import yaml
    from gateway.platforms.telegram_inline_router import InlineExecutor, TelegramInlineRouter

    registry = tmp_path / "inline_tools.yaml"
    registry.write_text(yaml.safe_dump({"version": 1, "tools": [
        {"id": "echo", "executor": "echo", "enabled": True, "match": [{"type": "search"}]},
    ]}))

    class _EchoExecutor(InlineExecutor):
        async def execute(self, user_id, query):
            return [{"type": "article", "id": "echo", "title": query}]

    router = TelegramInlineRouter(bot=None, registry_path=str(registry))
    router.register_executor("echo", lambda cfg, bot: _EchoExecutor())
    adapter = _make_adapter(inline_cfg={"enabled": True}, router=router)
    update, iq = _update("hello world")

    await adapter._handle_inline_query(update, None)

    iq.answer.assert_awaited_once()
    assert iq.answer.await_args.args[0] == [{"type": "article", "id": "echo", "title": "hello world"}]


@pytest.mark.asyncio
async def test_inline_dispatch_deadline_answers_empty(monkeypatch):
    """A dispatch exceeding RESPONSE_DEADLINE is cancelled and answered with [] — and
    because the deadline is applied around dispatch(), it holds even when an executor
    layer replaces dispatch."""
    import asyncio
    monkeypatch.setattr("gateway.platforms.telegram_inline_router.RESPONSE_DEADLINE", 0.05)

    async def _slow_dispatch(user_id, query):
        await asyncio.sleep(1.0)
        return ["late"]

    router = SimpleNamespace(dispatch=_slow_dispatch)
    adapter = _make_adapter(inline_cfg={"enabled": True}, router=router)
    update, iq = _update("x")

    await adapter._handle_inline_query(update, None)

    iq.answer.assert_awaited_once()
    assert iq.answer.await_args.args[0] == []


def test_check_telegram_requirements_rebinds_inline_query_handler(monkeypatch):
    """Regression: check_telegram_requirements()'s lazy-install rebind path
    left InlineQueryHandler pinned at the module's import-failure fallback
    (``Any``), because it was missing from both the `global` declaration and
    the re-import/rebind block. Any adapter constructed after a successful
    lazy install (python-telegram-bot missing at first import, then
    installed on demand) would call ``InlineQueryHandler(self._handle_inline_
    query)`` in ``connect()`` and get ``Any(...)`` — not callable — silently
    swallowed by that call site's broad ``except Exception`` and logged as
    "inline-query dispatch not enabled" with no other symptom.

    Exercises the real rebind mechanism rather than asserting on a canned
    sentinel: whatever ``telegram.ext.InlineQueryHandler`` currently
    resolves to in this process (the real PTB class, or another test's
    session-wide mock, depending on import order) is what must come out
    the other end — proving the rebind assignment itself fires, not just
    that some hardcoded value matches."""
    import plugins.platforms.telegram.adapter as adapter_mod
    from telegram.ext import InlineQueryHandler as _RealInlineQueryHandler

    monkeypatch.setattr(adapter_mod, "TELEGRAM_AVAILABLE", False)
    # Simulate the state left behind by a failed first import.
    monkeypatch.setattr(adapter_mod, "InlineQueryHandler", Any)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda feature, prompt=False: None)

    result = adapter_mod.check_telegram_requirements()

    assert result is True
    assert adapter_mod.InlineQueryHandler is _RealInlineQueryHandler


def test_missing_inline_query_handler_does_not_break_availability(monkeypatch):
    """Regression: InlineQueryHandler used to be bundled into the SAME
    `from telegram.ext import (...)` statement as CallbackQueryHandler,
    MessageHandler, ContextTypes, and filters. Any environment whose
    telegram.ext doesn't define InlineQueryHandler -- an older PTB release
    predating inline-mode support, or (as several pre-existing test files
    turned out to do) a minimal test double that stubs only the handlers it
    exercises -- raised ImportError for the WHOLE statement, so
    check_telegram_requirements() returned False and left ParseMode, filters,
    etc. all pinned at their None/Any import-failure fallbacks. That broke
    real Telegram sends entirely ('NoneType' object has no attribute
    'MARKDOWN_V2'), not just inline-query dispatch.

    InlineQueryHandler must be its own try/except (mirroring the existing
    LinkPreviewOptions optional-import a few lines above), so its absence
    only disables the inline-query bonus feature -- core Telegram
    functionality (ParseMode, filters, CallbackQueryHandler, ...) must still
    become available.
    """
    import importlib
    import types

    import plugins.platforms.telegram.adapter as adapter_mod

    real_telegram_ext = importlib.import_module("telegram.ext")

    class _ExtWithoutInlineQueryHandler(types.ModuleType):
        def __getattr__(self, name):
            if name == "InlineQueryHandler":
                raise AttributeError(name)
            return getattr(real_telegram_ext, name)

    stub_ext = _ExtWithoutInlineQueryHandler("telegram.ext")
    monkeypatch.setitem(__import__("sys").modules, "telegram.ext", stub_ext)
    monkeypatch.setattr(adapter_mod, "TELEGRAM_AVAILABLE", False)
    monkeypatch.setattr(adapter_mod, "ParseMode", None)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda feature, prompt=False: None)

    result = adapter_mod.check_telegram_requirements()

    assert result is True
    assert adapter_mod.ParseMode is not None
    assert adapter_mod.ParseMode.MARKDOWN_V2
