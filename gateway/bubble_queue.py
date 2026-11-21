"""5-State Lifecycle Queue and Dispatcher for Hermes Agent Streaming Bubbles (Phase 2.3 Hardened).

Features:
- 5-State Bubble Lifecycle: QUEUED -> SENDING -> SENT | FAILED | UNKNOWN_DELIVERY | CANCELLED.
- Non-preemptible Network Task Lifecycle via asyncio.shield with Strong Task Ownership.
- Robust In-Flight Task Tracking: send_task is tracked until completion across timeouts, cancellations, and worker shutdown.
- Late ACK Reconciliation: Late ACKs update original BubbleRecord without redispatching.
- UNKNOWN_DELIVERY Protection: Never redispatched once in unknown state; transitions to SENT upon late ACK.
- Error Boundary & Circuit Breaking: 2 consecutive hard failures trip circuit breaker, halting queue safely.
- Global Cross-Turn Pacing: 400ms idle interval enforced across all turns for the same chat_id.
- Monotonic Clock Precision: Prevents wall-clock skew/NTP jumping from corrupting timing calculations.
- Zero Unhandled Task Exceptions: Every spawned network task has guaranteed exception consumption.
- Dual Access Compatibility: sent_count, unknown_delivery_count, failed_count, cancelled_count support both property (count > 0) and method (count()) access patterns.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

logger = logging.getLogger("gateway.bubble_queue")


class BubbleDeliveryState(Enum):
    QUEUED = "queued"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    UNKNOWN_DELIVERY = "unknown_delivery"
    CANCELLED = "cancelled"


# Alias for compatibility
DeliveryStatus = BubbleDeliveryState


class QueueState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    CIRCUIT_BROKEN = "circuit_broken"
    CANCELLED = "cancelled"
    FINISHED = "finished"


class StreamingQueueError(Exception):
    """Base exception for streaming queue errors."""


class CircuitBreakerTrippedError(StreamingQueueError):
    """Raised when consecutive dispatch failures breach the threshold."""


class CallableInt(int):
    """An int subclass that is also callable, supporting both `obj.count > 0` and `obj.count()`."""
    def __call__(self) -> int:
        return int(self)


class BubbleRecord:
    """Tracks complete lifecycle, timestamps, and attribution for an individual bubble."""

    def __init__(self, index: int, text: str) -> None:
        self.index = index
        self.text = text
        self.state: BubbleDeliveryState = BubbleDeliveryState.QUEUED
        self.created_at: float = time.monotonic()
        self.send_started_at: Optional[float] = None
        self.send_finished_at: Optional[float] = None
        self.message_id: Optional[str] = None
        self.error: Optional[str] = None
        self.late_ack_received: bool = False

    @property
    def status(self) -> BubbleDeliveryState:
        return self.state

    @status.setter
    def status(self, val: BubbleDeliveryState) -> None:
        self.state = val

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "state": self.state.value,
            "created_at": self.created_at,
            "send_started_at": self.send_started_at,
            "send_finished_at": self.send_finished_at,
            "message_id": self.message_id,
            "error": self.error,
            "late_ack_received": self.late_ack_received,
        }


# Alias BubbleItem to BubbleRecord for compatibility
BubbleItem = BubbleRecord


class ChatPacer:
    """Enforces minimum idle spacing per chat_id across turns using monotonic clock,
    with cognitive adaptive pacing based on bubble length, semantic tone, and burst dynamics."""

    def __init__(self, chat_id: str, min_interval_s: float = 0.850, adaptive: bool = True) -> None:
        self.chat_id = chat_id
        self.min_interval_s = min_interval_s
        self.adaptive = adaptive
        self._last_send_finish_time: float = 0.0
        self._last_text: Optional[str] = None
        self._last_bubble_interval: float = min_interval_s
        self._lock = asyncio.Lock()

    @property
    def last_send_finish_time(self) -> float:
        return self._last_send_finish_time

    @property
    def current_interval_s(self) -> float:
        return self._last_bubble_interval

    def compute_interval(
        self,
        last_text: Optional[str] = None,
        next_text: Optional[str] = None,
    ) -> float:
        """Compute organic, human-like transmission delay between bubbles."""
        if not self.adaptive:
            return self.min_interval_s

        last_clean = (last_text or "").strip()
        next_clean = (next_text or "").strip()

        # 1. Tapback reaction: ultra-fast transition (double-tap gesture on iOS)
        if last_clean.startswith("TAPBACK:") or next_clean.startswith("TAPBACK:"):
            return max(0.350, random.uniform(0.380, 0.500))

        # 2. Standalone URL cards: allow iOS to fetch OpenGraph preview
        if last_clean.startswith(("http://", "https://")):
            return random.uniform(1.200, 1.600)

        # 3. Media cards (image/cheat sheet): visual scan gaze pause
        if last_clean.startswith(("IMAGE:", "FILE:", "POLL:")):
            return random.uniform(2.500, 3.500)

        # 4. Double-texting burst detection (连击追打短句):
        # Must be strictly very short text (<= 6 chars) to qualify as a rapid burst
        is_burst = (
            0 < len(next_clean) <= 6
            and (
                len(next_clean) <= 4
                or next_clean.startswith(("而且", "不过", "话说", "哈哈", "对啦", "另外", "好滴", "确实", "对吧", "等等", "真的", "绝了"))
                or next_clean in {"笑死", "好滴", "确实", "对吧", "等等", "真的", "绝了", "回头聊", "可以啊", "收到"}
            )
        )
        if is_burst:
            burst_delay = random.uniform(0.950, 1.350)
            return max(self.min_interval_s, burst_delay)

        # 5. Non-linear ergonomic reading + typing model
        # Reading allowance of last bubble: ~0.60s base + ~0.045s per char for first 20 chars, then 0.020s per char
        last_len = len(last_clean)
        reading_time = 0.600 + min(20, last_len) * 0.045 + max(0, last_len - 20) * 0.020

        # Punctuation cadence: thinking pause on questions, trailing off on ellipses
        if last_clean.endswith(("？", "?")):
            reading_time += random.uniform(0.350, 0.600)  # hesitation / thought pause
        elif last_clean.endswith(("...", "…", "~")):
            reading_time += random.uniform(0.250, 0.450)
        elif last_clean.endswith(("！", "!")):
            reading_time += random.uniform(0.150, 0.300)

        # Next bubble typing preparation (simulated finger tapping cadence: ~3-4 chars per sec)
        next_len = len(next_clean)
        if next_len > 0:
            typing_prep = 0.500 + min(20, next_len) * 0.050 + max(0, next_len - 20) * 0.025
        else:
            typing_prep = 0.0

        total_delay = reading_time + typing_prep

        # 6. Natural Gaussian micro-jitter (+-10% to +-15%)
        jitter = random.gauss(1.0, 0.08)
        jitter = max(0.85, min(1.20, jitter))
        total_delay *= jitter

        # Bounded between min_interval_s and 4.5s cap
        return max(self.min_interval_s, min(4.500, total_delay))

    def update_finish_time(self, ts: Optional[float] = None, last_text: Optional[str] = None) -> None:
        self._last_send_finish_time = ts if ts is not None else time.monotonic()
        self._last_text = last_text
        self._last_bubble_interval = self.compute_interval(last_text=last_text)

    async def wait_pacing(
        self,
        cancel_event: Optional[asyncio.Event] = None,
        next_text: Optional[str] = None,
        on_typing: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> bool:
        """Wait until current required dynamic interval has elapsed since last_send_finish_time."""
        async with self._lock:
            # Re-evaluate interval now that next_text is known
            if next_text is not None and self.adaptive:
                self._last_bubble_interval = self.compute_interval(
                    last_text=self._last_text, next_text=next_text
                )

            now = time.monotonic()
            elapsed = now - self._last_send_finish_time
            remaining = self._last_bubble_interval - elapsed

            if remaining > 0:
                # If pacing delay is substantial (> 0.45s), pulse typing indicator
                if remaining > 0.450 and on_typing is not None and not (cancel_event and cancel_event.is_set()):
                    try:
                        asyncio.create_task(on_typing())
                    except Exception:
                        pass

                if cancel_event is not None:
                    try:
                        await asyncio.wait_for(cancel_event.wait(), timeout=remaining)
                        return False
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(remaining)
            return True


class ChatPacerRegistry:
    """Adapter/Process-wide registry for shared ChatPacer instances."""
    _pacers: Dict[str, ChatPacer] = {}
    _lock = asyncio.Lock()

    @classmethod
    def get_pacer(cls, chat_id: str, min_interval_s: float = 0.850) -> ChatPacer:
        if chat_id not in cls._pacers:
            cls._pacers[chat_id] = ChatPacer(chat_id, min_interval_s=min_interval_s)
        return cls._pacers[chat_id]

    @classmethod
    def reset(cls) -> None:
        cls._pacers.clear()


_STREAM_END_SENTINEL = object()
_SHUTDOWN_SENTINEL = object()


class StreamingBubbleQueue:
    """FIFO Queue and Dispatcher with 5-state lifecycle and zero-leak cancellation."""

    def __init__(
        self,
        chat_id: str,
        send_func: Callable[..., Awaitable[Any]],
        maxsize: int = 20,
        min_interval_s: float = 0.850,
        in_flight_timeout_s: float = 12.0,
        pacer: Optional[ChatPacer] = None,
        circuit_breaker_threshold: int = 3,
        typing_func: Optional[Callable[..., Awaitable[Any]]] = None,
    ) -> None:
        self.chat_id = chat_id
        self.send_func = send_func
        self.maxsize = maxsize
        self.min_interval_s = min_interval_s
        self.in_flight_timeout_s = in_flight_timeout_s
        self.pacer = pacer or ChatPacerRegistry.get_pacer(chat_id, min_interval_s=min_interval_s)
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.typing_func = typing_func

        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self._records: List[BubbleRecord] = []
        self._state: QueueState = QueueState.IDLE

        self._cancel_event = asyncio.Event()
        self._finished_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None

        # Task Tracking Mechanisms (Phase 2.3 Hardened)
        self._in_flight_tasks: Set[asyncio.Task] = set()
        self._task_to_record: Dict[asyncio.Task, BubbleRecord] = {}
        self._current_in_flight_task: Optional[asyncio.Task] = None

        self._cancel_reason: Optional[str] = None
        self._circuit_consecutive_failures: int = 0
        self._circuit_breaker_tripped: bool = False
        self._circuit_breaker_reason: Optional[str] = None

    @property
    def records(self) -> List[BubbleRecord]:
        return list(self._records)

    @property
    def bubbles_history(self) -> List[BubbleRecord]:
        return list(self._records)

    @property
    def in_flight_task_count(self) -> int:
        return len(self._in_flight_tasks)

    @property
    def state(self) -> QueueState:
        return self._state

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_breaker_tripped

    @property
    def consecutive_failures(self) -> int:
        return self._circuit_consecutive_failures

    @property
    def sent_count(self) -> CallableInt:
        return CallableInt(sum(1 for r in self._records if r.state == BubbleDeliveryState.SENT))

    @property
    def failed_count(self) -> CallableInt:
        return CallableInt(sum(1 for r in self._records if r.state == BubbleDeliveryState.FAILED))

    @property
    def unknown_delivery_count(self) -> CallableInt:
        return CallableInt(sum(1 for r in self._records if r.state == BubbleDeliveryState.UNKNOWN_DELIVERY))

    @property
    def unknown_count(self) -> CallableInt:
        return self.unknown_delivery_count

    @property
    def cancelled_count(self) -> CallableInt:
        return CallableInt(sum(1 for r in self._records if r.state == BubbleDeliveryState.CANCELLED))

    @property
    def delivered_message_ids(self) -> List[str]:
        return [r.message_id for r in self._records if r.state == BubbleDeliveryState.SENT and r.message_id]

    def has_in_flight_task(self) -> bool:
        return len(self._in_flight_tasks) > 0 or (
            self._current_in_flight_task is not None and not self._current_in_flight_task.done()
        )

    def start(self) -> None:
        if self._state == QueueState.IDLE:
            self._state = QueueState.RUNNING
            self._finished_event.clear()
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def enqueue(self, bubble_text: str, timeout: Optional[float] = 10.0) -> bool:
        """Enqueue a bubble text with backpressure support. Returns False if dropped/tripped."""
        if self._state in (QueueState.CIRCUIT_BROKEN, QueueState.CANCELLED):
            record = BubbleRecord(index=len(self._records), text=bubble_text)
            record.state = BubbleDeliveryState.CANCELLED
            record.error = f"Queue not accepting items in {self._state.name} state"
            self._records.append(record)
            if self._state == QueueState.CIRCUIT_BROKEN:
                raise CircuitBreakerTrippedError(f"Circuit breaker tripped: {self._circuit_breaker_reason}")
            return False

        record = BubbleRecord(index=len(self._records), text=bubble_text)
        self._records.append(record)
        self.start()

        try:
            if timeout is not None:
                await asyncio.wait_for(self._queue.put(record), timeout=timeout)
            else:
                await self._queue.put(record)
            return True
        except asyncio.TimeoutError:
            record.state = BubbleDeliveryState.FAILED
            record.error = "Backpressure timeout: queue remained full"
            return False

    async def finish_stream(self) -> None:
        """Signal end of stream; worker will process all remaining items and terminate."""
        if self._state == QueueState.RUNNING:
            await self._queue.put(_STREAM_END_SENTINEL)

    async def wait_finished(self, timeout: Optional[float] = 30.0) -> bool:
        """Wait until worker finishes processing all items and in-flight tasks settle."""
        try:
            if timeout is not None:
                await asyncio.wait_for(self._finished_event.wait(), timeout=timeout)
            else:
                await self._finished_event.wait()
            if self._worker_task and not self._worker_task.done():
                await self._worker_task
            await self.wait_in_flight(timeout=5.0)
            return True
        except asyncio.TimeoutError:
            return False

    async def cancel(self, reason: str = "user_cancelled") -> None:
        """Cancel turn immediately: drops queued items, safeguards in-flight send, and stops worker."""
        self._cancel_reason = reason
        self._state = QueueState.CANCELLED
        self._cancel_event.set()

        # 1. Drain and cancel unstarted items in queue
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if isinstance(item, BubbleRecord):
                    item.state = BubbleDeliveryState.CANCELLED
                    item.error = f"Cancelled before sending: {reason}"
                self._queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

        # 2. Wake up worker if it was waiting on empty queue
        try:
            self._queue.put_nowait(_SHUTDOWN_SENTINEL)
        except (asyncio.QueueFull, Exception):
            pass

        # 3. Wait for worker task to settle gracefully without leaking
        if self._worker_task and not self._worker_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._worker_task), timeout=self.in_flight_timeout_s + 0.5)
            except (asyncio.TimeoutError, Exception):
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except (asyncio.CancelledError, Exception):
                    pass

        self._finished_event.set()

    async def wait_in_flight(self, timeout: Optional[float] = 5.0) -> None:
        """Wait for all tracked in-flight tasks to settle (e.g. at gateway shutdown or turn end)."""
        tasks = [t for t in self._in_flight_tasks if not t.done()]
        if not tasks:
            return
        try:
            if timeout is not None:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
            else:
                await asyncio.gather(*tasks, return_exceptions=True)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    async def close(self, drain: bool = True, wait_in_flight_timeout: float = 5.0) -> None:
        """Alias/convenience wrapper matching standard queue close pattern."""
        if drain and self._state == QueueState.RUNNING:
            await self.finish_stream()
            await self.wait_finished(timeout=wait_in_flight_timeout)
        else:
            await self.cancel(reason="queue_closed")
            await self.wait_in_flight(timeout=wait_in_flight_timeout)

    def _trip_circuit_breaker(self, reason: str) -> None:
        """Trip circuit breaker, cancel remaining queue items, and stop worker."""
        self._circuit_breaker_tripped = True
        self._circuit_breaker_reason = reason
        self._state = QueueState.CIRCUIT_BROKEN

        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if isinstance(item, BubbleRecord):
                    item.state = BubbleDeliveryState.CANCELLED
                    item.error = f"Circuit breaker tripped: {reason}"
                self._queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

        logger.error(f"[CircuitBreaker] Tripped for chat {self.chat_id}: {reason}")
        self._cancel_event.set()
        self._finished_event.set()

    def _handle_failure(self, error: str) -> None:
        self._circuit_consecutive_failures += 1
        if self._circuit_consecutive_failures >= self.circuit_breaker_threshold:
            self._trip_circuit_breaker(f"Consecutive failures: {self._circuit_consecutive_failures}")

    async def _invoke_send(self, text: str) -> Dict[str, Any]:
        """Adapt send_func regardless of signature or return type (SendResult / dict)."""
        try:
            sig = inspect.signature(self.send_func)
            params = list(sig.parameters.keys())
            if len(params) == 1:
                res = await self.send_func(text)
            elif len(params) >= 2:
                if params[0] in ('self', 'adapter') or params[0] == 'chat_id':
                    res = await self.send_func(self.chat_id, text)
                else:
                    res = await self.send_func(text=text, chat_id=self.chat_id)
            else:
                res = await self.send_func(text)
        except (TypeError, ValueError):
            try:
                res = await self.send_func(self.chat_id, text)
            except TypeError:
                res = await self.send_func(text)

        if isinstance(res, dict):
            return res
        msg_id = getattr(res, "id", None) or getattr(res, "message_id", None)
        success = getattr(res, "success", True)
        return {
            "ok": bool(success),
            "id": msg_id,
            "message_id": msg_id,
            "error": getattr(res, "error", None) or ("Send failed" if not success else None),
        }

    def _register_in_flight_task(self, task: asyncio.Task, record: BubbleRecord) -> None:
        """Track an active network task and attach a robust completion callback."""
        self._in_flight_tasks.add(task)
        self._task_to_record[task] = record
        self._current_in_flight_task = task

        def _on_done(t: asyncio.Task) -> None:
            self._in_flight_tasks.discard(t)
            registered_record = self._task_to_record.pop(t, None)
            if self._current_in_flight_task is t:
                self._current_in_flight_task = None

            if t.cancelled():
                if registered_record and registered_record.state == BubbleDeliveryState.SENDING:
                    registered_record.state = BubbleDeliveryState.CANCELLED
                return

            exc = t.exception()
            if exc is not None:
                logger.warning(
                    f"Network task {t.get_name()} finished with error: {exc}",
                    exc_info=exc,
                )
                if registered_record and registered_record.state in (
                    BubbleDeliveryState.SENDING,
                    BubbleDeliveryState.QUEUED,
                ):
                    registered_record.state = BubbleDeliveryState.FAILED
                    registered_record.error = str(exc)
                    self._handle_failure(str(exc))
                return

            try:
                res = t.result()
                if registered_record:
                    if registered_record.state == BubbleDeliveryState.UNKNOWN_DELIVERY:
                        registered_record.late_ack_received = True
                        registered_record.state = BubbleDeliveryState.SENT
                        msg_id = res.get("message_id") or res.get("id") if isinstance(res, dict) else None
                        if msg_id:
                            registered_record.message_id = str(msg_id)
                        logger.info(
                            f"[Late ACK] Bubble {registered_record.index} in UNKNOWN_DELIVERY "
                            f"reconciled to SENT with msg_id={registered_record.message_id}"
                        )
            except Exception as e:
                logger.error(f"Error inspecting completed network task result: {e}")

        task.add_done_callback(_on_done)

    async def _trigger_typing(self) -> None:
        """Trigger typing pulse through registered typing_func during pacing pause."""
        if self.typing_func:
            try:
                await self.typing_func()
            except Exception:
                pass

    async def _worker_loop(self) -> None:
        """Dedicated sequential consumer loop dispatching bubbles with pacing."""
        while True:
            if self._cancel_event.is_set():
                break

            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break

            if item is _STREAM_END_SENTINEL or item is _SHUTDOWN_SENTINEL:
                self._queue.task_done()
                break

            if isinstance(item, BubbleRecord):
                record = item

                # 1. Pacing phase: shared cross-turn dynamic human cadence constraint
                pacing_ok = await self.pacer.wait_pacing(
                    cancel_event=self._cancel_event,
                    next_text=record.text,
                    on_typing=self._trigger_typing if self.typing_func else None,
                )
                if not pacing_ok or self._cancel_event.is_set():
                    record.state = BubbleDeliveryState.CANCELLED
                    record.error = f"Cancelled during pacing sleep: {self._cancel_reason}"
                    self._queue.task_done()
                    break

                # 2. Sending phase: network transmission with strong task tracking
                record.state = BubbleDeliveryState.SENDING
                record.send_started_at = time.monotonic()

                send_task = asyncio.create_task(
                    self._invoke_send(record.text),
                    name=f"send-{self.chat_id}-b{record.index}",
                )
                self._register_in_flight_task(send_task, record)

                try:
                    send_timeout = 30.0 if (record.text.startswith("IMAGE:") or record.text.startswith("FILE:") or record.text.startswith("POLL:")) else self.in_flight_timeout_s
                    res = await asyncio.wait_for(asyncio.shield(send_task), timeout=send_timeout)
                    record.send_finished_at = time.monotonic()
                    self.pacer.update_finish_time(record.send_finished_at, last_text=record.text)

                    ok = res.get("ok", False) if isinstance(res, dict) else True
                    msg_id = res.get("message_id") or res.get("id") if isinstance(res, dict) else None
                    if ok:
                        record.state = BubbleDeliveryState.SENT
                        if msg_id:
                            record.message_id = str(msg_id)
                        self._circuit_consecutive_failures = 0
                    else:
                        record.state = BubbleDeliveryState.FAILED
                        record.error = res.get("error", "Send returned failure") if isinstance(res, dict) else "Send failed"
                        self._handle_failure(record.error)
                except asyncio.TimeoutError:
                    record.send_finished_at = time.monotonic()
                    self.pacer.update_finish_time(record.send_finished_at, last_text=record.text)
                    record.state = BubbleDeliveryState.UNKNOWN_DELIVERY
                    record.error = f"In-flight delivery unconfirmed: Timeout after {send_timeout}s"
                    self._handle_failure("Network ambiguity: TimeoutError")
                except asyncio.CancelledError:
                    record.send_finished_at = time.monotonic()
                    self.pacer.update_finish_time(record.send_finished_at, last_text=record.text)
                    record.state = BubbleDeliveryState.UNKNOWN_DELIVERY
                    record.error = "In-flight delivery unconfirmed: CancelledError"
                    self._trip_circuit_breaker("Network ambiguity: CancelledError")
                    raise
                except Exception as e:
                    record.send_finished_at = time.monotonic()
                    self.pacer.update_finish_time(record.send_finished_at)
                    record.state = BubbleDeliveryState.UNKNOWN_DELIVERY
                    record.error = f"Network invocation exception: {e}"
                    self._handle_failure(str(e))
                finally:
                    self._queue.task_done()

        if self._state == QueueState.RUNNING:
            self._state = QueueState.FINISHED

        self._finished_event.set()
