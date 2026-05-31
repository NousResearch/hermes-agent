"""Per-profile token bucket: a chatty profile can't starve the others."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.rate_limit import TokenBucket, ProfileRateLimiter
from gateway.session import SessionSource
from gateway.config import Platform


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_bucket_allows_up_to_capacity_then_throttles():
    clk = Clock()
    bucket = TokenBucket(capacity=3, refill_per_sec=1, clock=clk)
    assert [bucket.allow() for _ in range(3)] == [True, True, True]
    assert bucket.allow() is False


def test_bucket_refills_over_time():
    clk = Clock()
    bucket = TokenBucket(capacity=2, refill_per_sec=1, clock=clk)
    assert bucket.allow() and bucket.allow()
    assert bucket.allow() is False
    clk.t += 1.0
    assert bucket.allow() is True


def test_capacity_is_not_exceeded_by_refill():
    clk = Clock()
    bucket = TokenBucket(capacity=2, refill_per_sec=10, clock=clk)
    clk.t += 100
    assert [bucket.allow() for _ in range(3)] == [True, True, False]


def test_per_profile_isolation():
    clk = Clock()
    limiter = ProfileRateLimiter(capacity=2, refill_per_sec=1, clock=clk)
    assert limiter.allow("a") and limiter.allow("a")
    assert limiter.allow("a") is False  # A exhausted
    assert limiter.allow("b") is True  # B has its own budget
    assert limiter.allow("b") is True


@pytest.mark.asyncio
async def test_throttled_profile_is_not_dispatched():
    r = object.__new__(gateway_run.GatewayRunner)
    adapter = MagicMock()
    adapter.send = AsyncMock()
    r.adapters = {Platform.TELEGRAM: adapter}
    r._profile_rate_limiter = ProfileRateLimiter(capacity=1, refill_per_sec=0)
    r._dispatch_to_worker = AsyncMock()

    ev = MagicMock()
    ev.routed_profile = "coder"
    ev.source = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", user_id="u1")

    assert await r._maybe_dispatch_routed(ev, ev.source) is True  # 1st: allowed
    r._dispatch_to_worker.assert_awaited_once()
    assert await r._maybe_dispatch_routed(ev, ev.source) is True  # 2nd: throttled
    r._dispatch_to_worker.assert_awaited_once()  # still once — not dispatched again
    assert "busy" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_control_commands_bypass_rate_limit():
    """/reset must not be throttled — it carries no model cost and stranding a
    stuck conversation behind the bucket would be the worst time to throttle."""
    from gateway.platforms.base import MessageEvent, MessageType

    r = object.__new__(gateway_run.GatewayRunner)
    adapter = MagicMock()
    adapter.send = AsyncMock()
    r.adapters = {Platform.TELEGRAM: adapter}
    r._profile_rate_limiter = ProfileRateLimiter(capacity=0, refill_per_sec=0)  # everything throttled
    r._reset_routed_worker = AsyncMock()
    r._dispatch_to_worker = AsyncMock()

    src = SessionSource(platform=Platform.TELEGRAM, chat_id="100", chat_type="group", user_id="u1")
    ev = MessageEvent(text="/reset", message_type=MessageType.TEXT, source=src)
    ev.routed_profile = "coder"

    assert await r._maybe_dispatch_routed(ev, src) is True
    r._reset_routed_worker.assert_awaited_once()  # reset ran despite an exhausted bucket
    assert all("busy" not in c.args[1] for c in adapter.send.await_args_list)
