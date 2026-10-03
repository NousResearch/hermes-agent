"""Outbound throttling for the Telegram Bot API.

Why this exists
---------------
Telegram documents three send limits for bots (core.telegram.org/bots/faq,
"My bot is hitting limits, how do I avoid this?"):

* **~1 message per second in a single chat.** Short bursts are tolerated,
  then the API answers 429 ``Flood control exceeded``.
* **20 messages per minute in a group.**
* **~30 messages per second overall** (bulk/broadcast ceiling).

python-telegram-bot ships :class:`telegram.ext.AIORateLimiter`, but it only
enforces the *overall* 30/s bucket plus the *group* 20/min bucket, and the
group bucket is keyed on a negative (group/channel) ``chat_id``. A private
DM has a **positive** ``chat_id``, so in a DM the stock limiter enforces
nothing but 30/s -- exactly 30x over the limit that actually bans you. It
also requires the optional ``aiolimiter`` dependency, which is not installed
in Hermes environments.

Hermes makes that gap acute: every Telegram topic in Sam's DM is the *same*
``chat_id``, so N concurrent topic sessions, each streaming edits and
splitting long replies into 4096-char chunks, all spend one shared 1/s
budget. On 2026-08-24 that produced a ``retry_after`` of 5072s (85 minutes)
on chat 1335137548, and every subsequent call made inside the ban extended
it further (peaks past 6600s).

This module adds the missing per-chat bucket, keeps the documented group and
overall buckets, and needs no third-party dependency. Requests queue instead
of being refused, so a paced send is slower but never banned.

Design notes
------------
* ``getUpdates`` is never throttled -- PTB excludes it before calling us.
* Editing counts against the same per-chat budget as sending, and every
  request routed through ``ExtBot`` (send, edit, reaction, typing, ...)
  passes through :meth:`process_request`, so streaming edits are paced too.
* A token bucket, not a fixed window: capacity ``burst`` lets a short flurry
  through (Telegram tolerates that) and then hard-paces at ``rate``.
* On a 429 we still honour ``retry_after``: short waits are absorbed here
  (the caller never sees them), long ones propagate so the adapter's
  fail-closed breaker and failover bot own the decision.
* Bounded memory: idle per-chat buckets are dropped once they are full.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, Dict, Optional, Union

try:  # pragma: no cover - exercised implicitly wherever PTB is installed
    from telegram.error import RetryAfter as _RetryAfter
    from telegram.ext import BaseRateLimiter as _BaseRateLimiter

    PTB_RATE_LIMITER_AVAILABLE = True
except Exception:  # pragma: no cover - PTB missing/older
    _RetryAfter = None  # type: ignore[assignment]
    _BaseRateLimiter = object  # type: ignore[assignment,misc]
    PTB_RATE_LIMITER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Telegram's documented ceilings. Kept as literals rather than reading
# telegram.constants.FloodLimit so the module still imports on PTB builds
# that predate those constants.
DEFAULT_PER_CHAT_RATE = 1.0        # messages/second in one chat
DEFAULT_PER_CHAT_BURST = 3.0       # tolerated short burst before pacing
DEFAULT_GROUP_RATE = 20.0          # messages/minute in a group
DEFAULT_GROUP_PERIOD = 60.0
DEFAULT_OVERALL_RATE = 30.0        # messages/second bot-wide
DEFAULT_OVERALL_PERIOD = 1.0

# Longest 429 this limiter will absorb itself. Above this the exception is
# re-raised so the adapter's flood breaker + failover bot take over rather
# than a coroutine being pinned for over an hour (#91969).
DEFAULT_MAX_ABSORBED_RETRY_AFTER = 5.0

# Never queue behind more than this. A caller waiting longer than the value
# of the message is better served by failing to the failover path.
DEFAULT_MAX_QUEUE_WAIT = 30.0


class TokenBucket:
    """Async token bucket. ``rate`` tokens/second, holding at most ``burst``."""

    __slots__ = ("_rate", "_burst", "_tokens", "_updated", "_lock")

    def __init__(self, rate: float, burst: float) -> None:
        self._rate = float(rate)
        self._burst = max(float(burst), 1.0)
        self._tokens = self._burst
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def is_idle(self) -> bool:
        """True when the bucket has refilled completely (safe to evict)."""
        return self._peek() >= self._burst

    def _peek(self) -> float:
        now = time.monotonic()
        return min(self._burst, self._tokens + (now - self._updated) * self._rate)

    def wait_time(self) -> float:
        """Seconds until one token is available. 0.0 when ready now."""
        available = self._peek()
        if available >= 1.0:
            return 0.0
        if self._rate <= 0:
            return float("inf")
        return (1.0 - available) / self._rate

    async def acquire(self, max_wait: Optional[float] = None) -> bool:
        """Consume one token, sleeping if needed.

        Returns False when the required wait exceeds ``max_wait`` (nothing is
        consumed in that case), True once a token has been taken.
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self._burst, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                if self._rate <= 0:
                    return False
                need = (1.0 - self._tokens) / self._rate
                if max_wait is not None and need > max_wait:
                    return False
                await asyncio.sleep(need)


class HermesTelegramRateLimiter(_BaseRateLimiter):  # type: ignore[misc,valid-type]
    """Per-chat + per-group + overall throttle for the Telegram Bot API.

    Unlike :class:`telegram.ext.AIORateLimiter` this enforces the documented
    ~1 msg/s **per chat** limit, which is the one that bans a DM bot, and it
    has no third-party dependency.
    """

    def __init__(
        self,
        per_chat_rate: float = DEFAULT_PER_CHAT_RATE,
        per_chat_burst: float = DEFAULT_PER_CHAT_BURST,
        group_rate: float = DEFAULT_GROUP_RATE,
        group_period: float = DEFAULT_GROUP_PERIOD,
        overall_rate: float = DEFAULT_OVERALL_RATE,
        overall_period: float = DEFAULT_OVERALL_PERIOD,
        max_absorbed_retry_after: float = DEFAULT_MAX_ABSORBED_RETRY_AFTER,
        max_queue_wait: float = DEFAULT_MAX_QUEUE_WAIT,
        max_chat_buckets: int = 512,
    ) -> None:
        self._per_chat_rate = float(per_chat_rate)
        self._per_chat_burst = float(per_chat_burst)
        self._group_rate = float(group_rate)
        self._group_period = float(group_period)
        self._max_absorbed_retry_after = float(max_absorbed_retry_after)
        self._max_queue_wait = float(max_queue_wait)
        self._max_chat_buckets = int(max_chat_buckets)

        self._overall: Optional[TokenBucket] = (
            TokenBucket(overall_rate / overall_period, overall_rate)
            if overall_rate > 0 and overall_period > 0
            else None
        )
        self._chat_buckets: Dict[str, TokenBucket] = {}
        self._group_buckets: Dict[str, TokenBucket] = {}

        # Cleared while a 429 is being served, so every other request parks
        # instead of hammering the API and extending the ban.
        self._open = asyncio.Event()
        self._open.set()

        # Observability: what the throttle actually did, for a status probe.
        self.stats: Dict[str, Union[int, float]] = {
            "requests": 0,
            "throttled": 0,
            "throttled_seconds": 0.0,
            "refused_queue_too_long": 0,
            "retry_after_absorbed": 0,
            "retry_after_propagated": 0,
        }

    async def initialize(self) -> None:
        """No resources to set up (PTB calls this on start)."""

    async def shutdown(self) -> None:
        """No resources to tear down (PTB calls this on stop)."""

    # -- internals -----------------------------------------------------

    def _evict_if_needed(self, buckets: Dict[str, TokenBucket], keep: str) -> None:
        if len(buckets) <= self._max_chat_buckets:
            return
        for key, bucket in list(buckets.items()):
            if key != keep and bucket.is_idle:
                del buckets[key]

    def _chat_bucket(self, key: str) -> TokenBucket:
        bucket = self._chat_buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(self._per_chat_rate, self._per_chat_burst)
            self._chat_buckets[key] = bucket
            self._evict_if_needed(self._chat_buckets, key)
        return bucket

    def _group_bucket(self, key: str) -> TokenBucket:
        bucket = self._group_buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(self._group_rate / self._group_period, self._group_rate)
            self._group_buckets[key] = bucket
            self._evict_if_needed(self._group_buckets, key)
        return bucket

    @staticmethod
    def _is_group(chat_id: Any) -> bool:
        """Groups/channels: negative numeric id, or an @username string."""
        try:
            return int(chat_id) < 0
        except (TypeError, ValueError):
            return isinstance(chat_id, str) and bool(chat_id.strip())

    def snapshot(self) -> Dict[str, Any]:
        """Current throttle state, for a health probe or a status line."""
        return {
            "per_chat_rate": self._per_chat_rate,
            "per_chat_burst": self._per_chat_burst,
            "group_rate_per_min": self._group_rate,
            "overall_rate_per_sec": (
                self._overall._rate if self._overall else 0.0  # noqa: SLF001
            ),
            "tracked_chats": len(self._chat_buckets),
            "tracked_groups": len(self._group_buckets),
            "open": self._open.is_set(),
            "stats": dict(self.stats),
        }

    # -- PTB entry point -----------------------------------------------

    async def process_request(
        self,
        callback: Callable[..., Coroutine[Any, Any, Any]],
        args: Any,
        kwargs: Dict[str, Any],
        endpoint: str,
        data: Dict[str, Any],
        rate_limit_args: Optional[int] = None,
    ) -> Any:
        """Pace one Bot API call. ``getUpdates`` never reaches this method."""
        self.stats["requests"] = int(self.stats["requests"]) + 1
        chat_id = (data or {}).get("chat_id")

        # Calls with no chat (getMe, setMyCommands, ...) only take the
        # bot-wide bucket: they cannot trip a per-chat flood ban.
        if chat_id is None:
            await self._open.wait()
            if self._overall is not None:
                await self._overall.acquire()
            return await callback(*args, **kwargs)

        # Paid broadcasts have their own 1000/s ceiling; do not pace them
        # against the free per-chat limit.
        if (data or {}).get("allow_paid_broadcast"):
            await self._open.wait()
            return await callback(*args, **kwargs)

        key = str(chat_id)
        started = time.monotonic()
        await self._open.wait()

        chat_bucket = self._chat_bucket(key)
        expected = chat_bucket.wait_time()
        if expected > self._max_queue_wait:
            self.stats["refused_queue_too_long"] = (
                int(self.stats["refused_queue_too_long"]) + 1
            )
            logger.warning(
                "[telegram-throttle] chat %s backlog %.1fs exceeds max_queue_wait "
                "%.1fs; refusing locally so the caller can fail over",
                key,
                expected,
                self._max_queue_wait,
            )
            if _RetryAfter is not None:
                raise _RetryAfter(int(expected) + 1)
            raise RuntimeError(f"telegram_throttle_backlog:{expected:.0f}s")

        if not await chat_bucket.acquire(max_wait=self._max_queue_wait):
            self.stats["refused_queue_too_long"] = (
                int(self.stats["refused_queue_too_long"]) + 1
            )
            if _RetryAfter is not None:
                raise _RetryAfter(int(self._max_queue_wait) + 1)
            raise RuntimeError("telegram_throttle_backlog")

        if self._is_group(chat_id) and self._group_rate > 0:
            await self._group_bucket(key).acquire()
        if self._overall is not None:
            await self._overall.acquire()

        waited = time.monotonic() - started
        if waited > 0.05:
            self.stats["throttled"] = int(self.stats["throttled"]) + 1
            self.stats["throttled_seconds"] = (
                float(self.stats["throttled_seconds"]) + waited
            )
            logger.debug(
                "[telegram-throttle] paced %s for chat %s by %.2fs",
                endpoint,
                key,
                waited,
            )

        try:
            return await callback(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - inspected, then re-raised
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is None and _RetryAfter is not None:
                inner = getattr(exc, "_retry_after", None)
                if inner is not None:
                    with contextlib.suppress(Exception):
                        retry_after = inner.total_seconds()
            if retry_after is None:
                raise

            wait = float(retry_after)
            if wait <= self._max_absorbed_retry_after:
                # Short penalty: hold every other request, wait it out, retry
                # once. Callers never see a blip this small.
                self.stats["retry_after_absorbed"] = (
                    int(self.stats["retry_after_absorbed"]) + 1
                )
                logger.warning(
                    "[telegram-throttle] absorbing %.1fs flood wait for chat %s "
                    "(all sends paused)",
                    wait,
                    key,
                )
                self._open.clear()
                try:
                    await asyncio.sleep(wait + 0.1)
                finally:
                    self._open.set()
                return await callback(*args, **kwargs)

            # Long ban: park every other request for its duration so we stop
            # extending it, and let the adapter's breaker/failover decide.
            self.stats["retry_after_propagated"] = (
                int(self.stats["retry_after_propagated"]) + 1
            )
            logger.warning(
                "[telegram-throttle] flood wait %.0fs for chat %s exceeds the "
                "absorb cap %.0fs; parking sends and propagating",
                wait,
                key,
                self._max_absorbed_retry_after,
            )
            if self._open.is_set():
                self._open.clear()
                asyncio.get_running_loop().call_later(wait, self._open.set)
            raise


def build_rate_limiter_from_config(extra: Optional[Dict[str, Any]]) -> Optional[Any]:
    """Build the limiter from ``platforms.telegram.extra.rate_limit``.

    Enabled by default: shipping this off would leave the exact defect it
    fixes in place. Set ``rate_limit.enabled: false`` to opt out.
    """
    if not PTB_RATE_LIMITER_AVAILABLE:
        logger.warning(
            "[telegram-throttle] PTB BaseRateLimiter unavailable; outbound "
            "throttling disabled (flood bans possible)"
        )
        return None

    cfg = dict((extra or {}).get("rate_limit") or {})
    if cfg.get("enabled") is False:
        logger.warning(
            "[telegram-throttle] disabled by config; Telegram flood bans are "
            "possible on bursty chats"
        )
        return None

    def _f(name: str, default: float) -> float:
        try:
            return float(cfg.get(name, default))
        except (TypeError, ValueError):
            return default

    limiter = HermesTelegramRateLimiter(
        per_chat_rate=_f("per_chat_rate", DEFAULT_PER_CHAT_RATE),
        per_chat_burst=_f("per_chat_burst", DEFAULT_PER_CHAT_BURST),
        group_rate=_f("group_rate", DEFAULT_GROUP_RATE),
        group_period=_f("group_period", DEFAULT_GROUP_PERIOD),
        overall_rate=_f("overall_rate", DEFAULT_OVERALL_RATE),
        overall_period=_f("overall_period", DEFAULT_OVERALL_PERIOD),
        max_absorbed_retry_after=_f(
            "max_absorbed_retry_after", DEFAULT_MAX_ABSORBED_RETRY_AFTER
        ),
        max_queue_wait=_f("max_queue_wait", DEFAULT_MAX_QUEUE_WAIT),
    )
    logger.info(
        "[telegram-throttle] active: %.2f msg/s per chat (burst %.0f), "
        "%.0f/min per group, %.0f/s overall",
        limiter._per_chat_rate,  # noqa: SLF001
        limiter._per_chat_burst,  # noqa: SLF001
        limiter._group_rate,  # noqa: SLF001
        _f("overall_rate", DEFAULT_OVERALL_RATE),
    )
    return limiter
