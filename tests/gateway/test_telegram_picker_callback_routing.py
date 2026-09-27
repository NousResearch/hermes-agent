"""Routing guard: every /model-picker callback must reach the picker handler.

Bug class this prevents (real report, 2026-09-27: "Search ve son kullanılan
butonları açılmıyor"):

The flat /model menu grew two buttons — 🔎 Search (``ms``) and
🕘 Son kullanılanlar (``mr``) — and ``_handle_model_picker_callback`` implements
both flows completely. But the dispatcher in ``_handle_callback_query`` matched
only the *older* prefix tuple, so a tap on either button fell through to the
"unknown callback" tail: no answer popup, no re-render, nothing happened. The
feature looked broken while every flow-level test passed, because those tests
called ``_handle_model_picker_callback`` directly and never went through the
dispatcher.

Two guards here:
1. ``test_every_handled_callback_is_routed`` — scans the picker handler's source
   for every callback value it handles and asserts a dispatcher prefix tuple
   covers all of them. Adding a button without registering its prefix fails here
   instead of failing silently in Telegram.
2. ``test_search_and_recent_taps_reach_the_picker`` — drives the real dispatcher
   with ``ms`` / ``mr`` (and their sub-steps) and asserts the picker handler runs.
"""
from __future__ import annotations

import asyncio
import inspect
import re

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _handled_callback_values() -> set[str]:
    """Callback values handled inside _handle_model_picker_callback."""
    src = inspect.getsource(TelegramAdapter._handle_model_picker_callback)
    values = set(re.findall(r'data\s*==\s*"([^"]+)"', src))
    for group in re.findall(r"data\.startswith\(([^)]*)\)", src):
        values.update(re.findall(r'"([^"]+)"', group))
    assert values, "handler source scan found no callbacks — the regex drifted"
    return values


def _dispatcher_prefix_tuples() -> list[tuple]:
    """Prefix tuples the callback dispatcher routes on, in source order."""
    src = inspect.getsource(TelegramAdapter._handle_callback_query)
    tuples = re.findall(r"\(\(([^)]*)\)\s*,", src)
    return [tuple(re.findall(r'"([^"]+)"', group)) for group in tuples]


def test_every_handled_callback_is_routed():
    routed = _dispatcher_prefix_tuples()
    assert routed, "no dispatcher prefix tuples found — the regex drifted"
    unrouted = sorted(
        value for value in _handled_callback_values()
        if not any(value.startswith(prefixes) for prefixes in routed)
    )
    assert not unrouted, (
        "these callbacks are handled by _handle_model_picker_callback but no dispatcher "
        f"prefix tuple in _handle_callback_query routes them: {unrouted} — register the "
        "prefix there, otherwise the button is dead in Telegram (silent drop)."
    )


def test_menu_buttons_search_and_recent_are_routed():
    routed = _dispatcher_prefix_tuples()
    for cb in ("ms", "mr"):
        assert any(cb.startswith(prefixes) for prefixes in routed), cb


class _FakeUser:
    first_name = "tester"
    id = 1


class _FakeMessage:
    chat_id = 4242
    message_thread_id = None
    chat = None


class _FakeQuery:
    def __init__(self, data: str) -> None:
        self.data = data
        self.message = _FakeMessage()
        self.from_user = _FakeUser()

    async def answer(self, *a, **kw):  # pragma: no cover - not expected here
        raise AssertionError(f"unexpected query.answer for {self.data!r}")


class _FakeUpdate:
    def __init__(self, data: str) -> None:
        self.callback_query = _FakeQuery(data)


def _dispatch(monkeypatch, data: str) -> list[tuple[str, str]]:
    """Run the real dispatcher; return the ``(callback_data, chat_id)`` picks it routed."""
    adapter = object.__new__(TelegramAdapter)
    adapter._accept_update = lambda: None  # type: ignore[assignment]
    seen: list[tuple[str, str]] = []

    async def _recorder(query, cb_data, chat_id):
        seen.append((cb_data, chat_id))

    async def _authorized(query, cb, denial_text):
        return True

    monkeypatch.setattr(adapter, "_callback_authorized", _authorized, raising=False)
    monkeypatch.setattr(adapter, "_handle_model_picker_callback", _recorder, raising=False)
    asyncio.run(adapter._handle_callback_query(_FakeUpdate(data), None))  # type: ignore[arg-type]
    return seen


@pytest.mark.parametrize("cb", ["ms", "mr", "msv:1", "msel:0", "msc:0", "mrv:1", "mrsel:0", "mrc:0"])
def test_search_and_recent_taps_reach_the_picker(monkeypatch, cb):
    assert _dispatch(monkeypatch, cb) == [(cb, "4242")], f"tap {cb!r} was dropped"
