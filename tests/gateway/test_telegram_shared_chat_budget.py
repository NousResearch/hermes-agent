"""Shared per-chat Telegram outbound budget (issue #107612).

Telegram meters send/edit/typing/delete against ONE envelope per chat.
These tests pin the sum-ceiling (not send-only or typing-only) and must
fail on main before the shared budget exists.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id


WINDOW_SECS = 60.0
PRIVATE_CEILING = 60
GROUP_CEILING = 20
PRIVATE_GAP = WINDOW_SECS / PRIVATE_CEILING
GROUP_GAP = WINDOW_SECS / GROUP_CEILING
API_KINDS = frozenset({"send_message", "send_chat_action", "edit_message_text", "delete_message"})


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class _FloodError(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


class _RecordingBot:
    """Minimal Bot stand-in that records outbound calls against the fake clock."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self.calls: list[tuple[str, object, float]] = []
        self._send_n = 0
        self._flood_once: float | None = None
        self.get_chat = AsyncMock(side_effect=AssertionError("get_chat must not classify chat type"))

    def flood_next_send(self, retry_after: float) -> None:
        self._flood_once = float(retry_after)

    def _record(self, kind: str, chat_id: object) -> None:
        self.calls.append((kind, chat_id, self._clock()))

    async def send_message(self, **kwargs):
        self._record("send_message", kwargs.get("chat_id"))
        if self._flood_once is not None:
            wait = self._flood_once
            self._flood_once = None
            raise _FloodError(wait)
        self._send_n += 1
        return MagicMock(message_id=self._send_n)

    async def send_chat_action(self, **kwargs):
        self._record("send_chat_action", kwargs.get("chat_id"))

    async def edit_message_text(self, **kwargs):
        self._record("edit_message_text", kwargs.get("chat_id"))

    async def delete_message(self, **kwargs):
        self._record("delete_message", kwargs.get("chat_id"))


def _wire_clock(adapter: TelegramAdapter, clock: _FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _sleep(dt: float) -> None:
        clock.advance(dt)

    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", _sleep)
    budget = getattr(adapter, "_chat_budget", None)
    if budget is not None:
        budget.clock = clock


def _make_adapter(clock: _FakeClock, monkeypatch: pytest.MonkeyPatch) -> tuple[TelegramAdapter, _RecordingBot]:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._rich_send_disabled = True
    adapter._rich_messages_enabled = False
    bot = _RecordingBot(clock)
    adapter._bot = bot
    _wire_clock(adapter, clock, monkeypatch)
    return adapter, bot


def _api_in_window(
    bot: _RecordingBot, chat_id: str, start: float, window: float = WINDOW_SECS,
) -> list[tuple[str, object, float]]:
    key = str(normalize_telegram_chat_id(chat_id))
    matched = []
    for kind, raw_id, ts in bot.calls:
        if kind not in API_KINDS:
            continue
        if str(normalize_telegram_chat_id(raw_id)) != key:
            continue
        if start <= ts < start + window:
            matched.append((kind, raw_id, ts))
    return matched


def _kinds(calls: list[tuple[str, object, float]]) -> set[str]:
    return {c[0] for c in calls}


async def _mixed_burst(adapter: TelegramAdapter, chat_id: str, n: int) -> None:
    """Fire send + typing + periodic edit/delete so the sum — not one type — is tested."""
    for i in range(n):
        await adapter.send_typing(chat_id)
        await adapter.send(chat_id, f"msg-{i}")
        if i % 8 == 0:
            await adapter.edit_message(chat_id, "1", f"edit-{i}", finalize=True)
            await adapter.delete_message(chat_id, "1")


@pytest.mark.asyncio
async def test_sum_ceiling_private(monkeypatch):
    clock = _FakeClock()
    adapter, bot = _make_adapter(clock, monkeypatch)
    start = clock()
    await _mixed_burst(adapter, "123", 40)

    window = _api_in_window(bot, "123", start)
    assert _kinds(window) - {"send_message"}
    assert _kinds(window) - {"send_chat_action"}
    assert len(window) <= PRIVATE_CEILING


@pytest.mark.asyncio
async def test_sum_ceiling_group(monkeypatch):
    clock = _FakeClock()
    adapter, bot = _make_adapter(clock, monkeypatch)
    start = clock()
    await _mixed_burst(adapter, "-100123", 20)

    window = _api_in_window(bot, "-100123", start)
    assert _kinds(window) - {"send_message"}
    assert _kinds(window) - {"send_chat_action"}
    assert len(window) <= GROUP_CEILING


@pytest.mark.asyncio
async def test_shared_across_concurrent_senders(monkeypatch):
    clock = _FakeClock()
    adapter, bot = _make_adapter(clock, monkeypatch)
    chat_id = "123"
    start = clock()

    async def _worker() -> None:
        await _mixed_burst(adapter, chat_id, 25)

    await asyncio.gather(_worker(), _worker())
    window = _api_in_window(bot, chat_id, start)
    assert _kinds(window) - {"send_message"}
    assert _kinds(window) - {"send_chat_action"}
    assert len(window) <= PRIVATE_CEILING


@pytest.mark.asyncio
async def test_independent_chats(monkeypatch):
    clock = _FakeClock()
    adapter, bot = _make_adapter(clock, monkeypatch)
    start = clock()
    await _mixed_burst(adapter, "111", 40)
    before_b = len(_api_in_window(bot, "222", start))
    await adapter.send_typing("222")
    result = await adapter.send("222", "other-chat")
    assert result.success is True
    b_calls = _api_in_window(bot, "222", start)
    assert len(b_calls) > before_b
    assert any(k == "send_message" for k, _, _ in b_calls)


@pytest.mark.asyncio
async def test_retry_after_widens_on_real_send_path(monkeypatch):
    clock = _FakeClock()
    adapter, bot = _make_adapter(clock, monkeypatch)
    bot.flood_next_send(2.0)

    result = await adapter.send("123", "hello")
    assert result.success is True

    actions_before = sum(1 for k, _, _ in bot.calls if k == "send_chat_action")
    t_before = clock()
    await adapter.send_typing("123")
    later = await adapter.send("123", "after-widen")

    delayed = (clock() - t_before) >= 2.0 - 1e-9
    shed = sum(1 for k, _, _ in bot.calls if k == "send_chat_action") == actions_before
    assert later.success is True
    assert shed or delayed


@pytest.mark.asyncio
async def test_widen_expires(monkeypatch):
    clock = _FakeClock()
    adapter, bot = _make_adapter(clock, monkeypatch)
    bot.flood_next_send(2.0)
    first = await adapter.send("123", "hello")
    assert first.success is True

    clock.advance(WINDOW_SECS + 0.1)
    t0 = clock()
    assert (await adapter.send("123", "a")).success is True
    assert (await adapter.send("123", "b")).success is True
    elapsed = clock() - t0
    assert elapsed == pytest.approx(PRIVATE_GAP, abs=0.05)


@pytest.mark.asyncio
async def test_typing_shed_not_queued(monkeypatch):
    clock = _FakeClock()
    adapter, bot = _make_adapter(clock, monkeypatch)

    await adapter.send_typing("123")
    await adapter.send("123", "first")
    actions_after_fill = sum(1 for k, _, _ in bot.calls if k == "send_chat_action")

    sleep_calls = {"n": 0}

    async def _guarded_sleep(dt: float) -> None:
        sleep_calls["n"] += 1
        clock.advance(dt)

    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", _guarded_sleep)
    await adapter.send_typing("123")
    assert sum(1 for k, _, _ in bot.calls if k == "send_chat_action") == actions_after_fill
    assert sleep_calls["n"] == 0

    clock.advance(PRIVATE_GAP)
    result = await adapter.send("123", "final")
    assert result.success is True
    assert result.message_id is not None


@pytest.mark.asyncio
async def test_send_long_flood_fails_closed_without_inline_sleep(monkeypatch):
    """CONTROL: 5827s retry_after still fail-closed, no inline sleep (#91969)."""
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._rich_send_disabled = True
    adapter._bot.send_message = AsyncMock(side_effect=_FloodError(5827.0))
    sleep = AsyncMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", sleep)

    result = await adapter.send("123", "hello")

    assert result.success is False
    assert result.error == "flood_control:5827.0"
    assert result.retry_after == 5827.0
    assert result.retryable is False
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_username_no_getchat(monkeypatch):
    clock = _FakeClock()
    adapter, bot = _make_adapter(clock, monkeypatch)
    chat_id = "@somechannel"
    start = clock()
    await _mixed_burst(adapter, chat_id, 20)

    bot.get_chat.assert_not_called()
    window = _api_in_window(bot, chat_id, start)
    assert len(window) <= GROUP_CEILING
