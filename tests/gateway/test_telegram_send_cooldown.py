"""TelegramAdapter per-chat send-cooldown gate.

Cross-cutting rate-limiter for outbound ``send()`` calls. Telegram's
Bot API enforces ~1 msg/sec/chat in private DMs and a global ~30
msg/sec budget across all chats for bots. A single user turn
historically fans out 3-5 sends from independent code paths
(status callbacks, progress bubbles, streaming previews, final
answer) and they were fired in parallel — a burst pattern that
crosses Telegram's threshold and triggers a flood-control penalty
that escalates into multi-thousand-second back-offs.

The fix is a per-chat minimum-gap gate in ``send()`` so the
cumulative send rate stays under the threshold regardless of how
many components fire concurrently. These tests pin the gate's
behaviour:

  - successful sends stamp the cooldown so the next send to the
    same chat waits for ``_send_cooldown_seconds``
  - a different chat has an independent cooldown
  - the wait is bounded by ``_send_cooldown_max_wait`` so a
    7000-second Telegram penalty doesn't stall the chat path
  - a send that arrives just after the cooldown expires goes
    through without an extra delay
"""
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    mod.error.RetryAfter = type(
        "RetryAfter",
        (Exception,),
        {"__init__": lambda self, retry_after=1: setattr(self, "retry_after", retry_after)},
    )
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(
        return_value=MagicMock(message_id=42),
    )
    return adapter


@pytest.mark.asyncio
async def test_first_send_stamps_cooldown_for_same_chat(monkeypatch):
    """A successful send records a cooldown timestamp so the next send to
    the same chat can be gated by ``send()``."""
    adapter = _make_adapter()
    adapter._send_cooldown_seconds = 1.1
    adapter._send_cooldown_max_wait = 5.0

    fake_now = {"t": 1000.0}
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.time.monotonic",
        lambda: fake_now["t"],
    )

    result = await adapter.send("123", "hello")

    assert result.success is True
    # Stamped roughly +1.1s from the start of the call. Allow a small
    # floating-point slack because the gate reads monotonic() multiple
    # times during the request.
    stamped = adapter._send_cooldown_until["123"]
    assert 1000.0 <= stamped <= 1001.5


@pytest.mark.asyncio
async def test_second_send_within_window_blocks(monkeypatch):
    """A second send to the same chat within the cooldown window waits for
    the gate before issuing its Bot API call. We assert via the elapsed
    ``time.monotonic()`` — by advancing a fake clock while the gate sleeps
    we can verify the wait happened without sleeping real time."""
    adapter = _make_adapter()
    adapter._send_cooldown_seconds = 1.1
    adapter._send_cooldown_max_wait = 5.0

    fake_now = {"t": 1000.0}

    def fake_monotonic():
        return fake_now["t"]

    # Capture the real sleep and let it advance our fake clock — this
    # mirrors how the actual gate waits via asyncio.sleep().
    real_sleep = time.sleep

    def fake_sleep(seconds):
        real_sleep(seconds)  # tiny real delay
        fake_now["t"] += seconds

    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.time.monotonic",
        fake_monotonic,
    )
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.sleep",
        AsyncMock(side_effect=fake_sleep),
    )

    await adapter.send("123", "first")
    fake_now["t"] = 1000.3  # within cooldown
    await adapter.send("123", "second")

    # Two actual API calls (one per send), and the second one was
    # delayed by the cooldown duration.
    assert adapter._bot.send_message.await_count == 2
    # The cooldown after the second send should reflect the advanced
    # clock — first stamp at ~1001.1, gate waited ~0.8s, second stamp
    # at ~1002.0 (give or take the scheduling jitter).
    stamped = adapter._send_cooldown_until["123"]
    assert stamped >= 1001.5


@pytest.mark.asyncio
async def test_second_send_after_window_passes_immediately(monkeypatch):
    """A second send that arrives after the cooldown has expired must NOT
    wait — only the gate's check is cheap, but we still verify the
    Bot API was called."""
    adapter = _make_adapter()
    adapter._send_cooldown_seconds = 1.1
    adapter._send_cooldown_max_wait = 5.0

    fake_now = {"t": 1000.0}
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.time.monotonic",
        lambda: fake_now["t"],
    )
    sleep_calls = []
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.sleep",
        AsyncMock(side_effect=lambda s: sleep_calls.append(s)),
    )

    await adapter.send("123", "first")
    # Jump well past the cooldown window.
    fake_now["t"] = 1050.0
    await adapter.send("123", "second")

    assert adapter._bot.send_message.await_count == 2
    # The gate should NOT have slept — the cooldown was already past.
    assert sleep_calls == []


@pytest.mark.asyncio
async def test_independent_cooldowns_per_chat(monkeypatch):
    """Two different chats have independent cooldowns; a send to chat B
    must not block on chat A's cooldown and vice versa."""
    adapter = _make_adapter()
    adapter._send_cooldown_seconds = 5.0
    adapter._send_cooldown_max_wait = 10.0

    fake_now = {"t": 1000.0}
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.time.monotonic",
        lambda: fake_now["t"],
    )
    sleep_calls = []
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.sleep",
        AsyncMock(side_effect=lambda s: sleep_calls.append(s)),
    )

    # First send to chat A stamps A's cooldown.
    await adapter.send("AAA", "to A")
    # Immediate send to chat B should NOT be blocked by A's cooldown.
    await adapter.send("BBB", "to B")

    assert adapter._send_cooldown_until["AAA"] >= 1005.0
    assert adapter._send_cooldown_until["BBB"] >= 1005.0
    # The B-send must not have slept behind A's gate.
    assert sleep_calls == []


@pytest.mark.asyncio
async def test_oversized_wait_returns_retryable_error(monkeypatch):
    """If the gate's wait would exceed ``_send_cooldown_max_wait`` (e.g.
    because Telegram imposed a multi-thousand-second penalty) the gate
    must NOT block the caller indefinitely. Instead it returns a
    retryable error so upstream retries can back off too."""
    adapter = _make_adapter()
    adapter._send_cooldown_seconds = 1.1
    adapter._send_cooldown_max_wait = 5.0

    # Pretend chat A has a 7000-second flood penalty still running.
    adapter._send_cooldown_until["999"] = time.monotonic() + 7000.0

    sleep_calls = []
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.asyncio.sleep",
        AsyncMock(side_effect=lambda s: sleep_calls.append(s)),
    )

    result = await adapter.send("999", "hello")

    # Gate must NOT have slept — the wait is too long.
    assert sleep_calls == []
    # And must have returned a retryable error tagged with the wait time
    # so the caller can decide what to do.
    assert result.success is False
    assert result.retryable is True
    assert "flood_control" in (result.error or "")
    # No Bot API call should have been made.
    assert adapter._bot.send_message.await_count == 0
