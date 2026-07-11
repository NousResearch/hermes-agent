"""Built-in mem0 rerank arm normalization and fail-loud incident state."""

from __future__ import annotations

from collections import deque
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

RERANK_OFF = "off"
RERANK_BUILTIN = "builtin"
_ALERT_TARGET = "discord:1480528231286181948"
_ROLLBACK = "set mem0.json rerank=off, then Apollo safe-restarts the gateway"
_STATE_LOCKS: Dict[str, threading.Lock] = {}
_STATE_LOCKS_GUARD = threading.Lock()
_MANAGERS: Dict[str, "RerankIncidentManager"] = {}
_MANAGERS_LOCK = threading.Lock()


def normalize_rerank_arm(value: Any) -> str:
    """Return the only enabled arm; legacy booleans/enums fail closed to off."""
    return RERANK_BUILTIN if str(value or "").strip().lower() == RERANK_BUILTIN else RERANK_OFF


def _state_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _STATE_LOCKS_GUARD:
        return _STATE_LOCKS.setdefault(key, threading.Lock())


def _default_state() -> Dict[str, Any]:
    return {
        "active": False,
        "consecutive_failures": 0,
        "failure_class": None,
        "observed_arm": None,
        "pending_page": None,
        "pending_message": None,
    }


def _send_page(message: str) -> None:
    """Deliver through Hermes' sender and require its documented positive ack."""
    from hermes_cli.send_cmd import _load_hermes_env
    from tools.send_message_tool import send_message_tool

    _load_hermes_env()
    raw = send_message_tool({
        "action": "send",
        "target": _ALERT_TARGET,
        "message": message,
    })
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError) as exc:
        raise RuntimeError("page sender did not return a positive success acknowledgement") from exc
    if not isinstance(parsed, dict) or parsed.get("success") is not True:
        detail = parsed.get("error") if isinstance(parsed, dict) else None
        raise RuntimeError(
            str(detail) if detail else "page sender did not return a positive success acknowledgement"
        )


class RerankIncidentManager:
    """Atomic, edge-triggered incident detector shared by gateway request threads."""

    def __init__(
        self,
        *,
        state_path: Path,
        latency_budget_ms: float,
        failure_threshold: int = 3,
        alert_fn: Optional[Callable[[str], None]] = None,
        window_size: int = 100,
        queue_size: int = 256,
        delivery_timeout_s: float = 10.0,
        retry_backoff_s: float = 5.0,
    ):
        self._path = Path(state_path).expanduser()
        self._latency_budget_ms = max(0.0, float(latency_budget_ms))
        self._failure_threshold = max(1, int(failure_threshold))
        self._alert = alert_fn or _send_page
        self._window_size = max(1, int(window_size))
        self._lock = _state_lock(self._path)
        self._queue_size = max(1, int(queue_size))
        self._observations = deque()
        self._condition = threading.Condition()
        self._owner_thread: Optional[threading.Thread] = None
        self._owner_thread_id: Optional[int] = None
        self._owner_busy = False
        self._observer_error_count = 0
        self._observer_drop_count = 0
        self._persistence_error_count = 0
        self._overflow_pending = 0
        self._retry_at: Optional[float] = None
        self._dirty_state = False
        self._delivery_timeout_s = max(0.001, float(delivery_timeout_s))
        self._retry_backoff_s = max(0.001, float(retry_backoff_s))
        self._delivery_attempt: Optional[Dict[str, Any]] = None
        with self._lock:
            self._state = self._read()
        self._latencies: list[float] = []
        if self._state.get("pending_page"):
            with self._condition:
                self._retry_at = time.monotonic()
                self._ensure_owner_locked()
                self._condition.notify()

    def _read(self) -> Dict[str, Any]:
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return _default_state()
            state = _default_state()
            state.update(value)
            state.pop("latencies_ms", None)
            if not state.get("active"):
                state["consecutive_failures"] = 0
                state["failure_class"] = None
                state["observed_arm"] = None
            return state
        except (OSError, ValueError, TypeError):
            return _default_state()

    def _write(self, state: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self._path.name}.", dir=self._path.parent)
        fd_owned = True
        try:
            handle = None
            try:
                handle = os.fdopen(fd, "w", encoding="utf-8")
                fd_owned = False  # fdopen owns the descriptor from here
            finally:
                if fd_owned:
                    os.close(fd)
            with handle:
                json.dump(state, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self._path)
            directory_fd = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def _safe_write(self, state: Dict[str, Any]) -> bool:
        try:
            self._write(state)
            self._dirty_state = False
            return True
        except Exception as exc:
            self._dirty_state = True
            self._persistence_error_count += 1
            logger.error("mem0 rerank incident-state persistence failed: %s", exc)
            self._schedule_retry()
            return False

    @staticmethod
    def _p95(values: list[float]) -> float:
        ordered = sorted(values)
        index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]

    @property
    def owner_thread_id(self) -> Optional[int]:
        with self._condition:
            return self._owner_thread_id

    @property
    def observer_error_count(self) -> int:
        with self._condition:
            return self._observer_error_count

    @property
    def observer_drop_count(self) -> int:
        with self._condition:
            return self._observer_drop_count

    @property
    def persistence_error_count(self) -> int:
        with self._lock:
            return self._persistence_error_count

    def _schedule_retry(self) -> None:
        with self._condition:
            retry_at = time.monotonic() + self._retry_backoff_s
            if self._retry_at is None or retry_at < self._retry_at:
                self._retry_at = retry_at
            self._ensure_owner_locked()
            self._condition.notify()

    def _ensure_owner_locked(self) -> None:
        if self._owner_thread is not None:
            return
        self._owner_thread = threading.Thread(
            target=self._dispatch_loop,
            name="mem0-rerank-incident",
            daemon=True,
        )
        self._owner_thread.start()

    def observe(self, metadata: Dict[str, Any]) -> None:
        """Append one observation; recall never waits for persistence or paging."""
        try:
            observation = dict(metadata or {})
            with self._condition:
                self._ensure_owner_locked()
                if len(self._observations) >= self._queue_size:
                    self._observations.popleft()
                    self._observer_drop_count += 1
                    self._overflow_pending += 1
                self._observations.append(observation)
                self._condition.notify()
        except Exception as exc:
            with self._condition:
                self._observer_error_count += 1
            logger.error("mem0 rerank observer rejected metadata: %s", exc)

    def submit(self, metadata: Dict[str, Any]) -> None:
        """Compatibility alias for callers from the pre-flusher implementation."""
        self.observe(metadata)

    def _page_message(
        self,
        page_kind: str,
        *,
        failure_class: Optional[str] = None,
        effective_arm: str = RERANK_OFF,
        p95: Optional[float] = None,
    ) -> str:
        if page_kind == "recovery":
            return "MEM0 RERANK RECOVERED: arm=builtin; detector healthy; configured arm unchanged."
        detail = (
            f" p95_ms={p95:.2f} budget_ms={self._latency_budget_ms:.2f}"
            if failure_class == "LATENCY-BREACH" and p95 is not None else ""
        )
        return (
            "MEM0 RERANK INCIDENT: arm=builtin "
            f"failure={failure_class or 'failure'} observed_arm={effective_arm}.{detail} "
            f"Configured arm remains enabled; manual rollback: {_ROLLBACK}."
        )

    def _delivery_worker(self, message: str, attempt: Dict[str, Any]) -> None:
        try:
            self._alert(message)
        except BaseException as exc:
            attempt["error"] = exc
        finally:
            attempt["done"].set()
            # Retry is only armed on failure. A successful delivery is settled by
            # the waiting _deliver_pending (or, after a caller timeout, by the
            # retry that caller already armed) — an unconditional re-arm here
            # races _clear_retry_if_clean and leaves a spurious wakeup pending.
            if attempt.get("error") is not None:
                self._schedule_retry()

    def _clear_retry_if_clean(self) -> None:
        with self._lock:
            clean = not self._dirty_state and not self._state.get("pending_page")
        if clean:
            with self._condition:
                self._retry_at = None
                self._condition.notify_all()

    def _deliver_pending(self) -> None:
        with self._lock:
            page_kind = self._state.get("pending_page")
            if not page_kind:
                if (
                    self._delivery_attempt is not None
                    and self._delivery_attempt["done"].is_set()
                ):
                    self._delivery_attempt = None
                return
            message = self._state.get("pending_message") or self._page_message(
                page_kind,
                failure_class=self._state.get("failure_class"),
                effective_arm=self._state.get("observed_arm") or RERANK_OFF,
            )
        attempt = self._delivery_attempt
        if attempt is not None and not attempt["done"].is_set():
            self._schedule_retry()
            return
        if attempt is not None:
            self._delivery_attempt = None
            if attempt.get("error") is not None:
                logger.error(
                    "mem0 rerank page delivery failed (retry armed): %s",
                    attempt["error"],
                )
                self._schedule_retry()
                return
            if attempt.get("message") == message:
                with self._lock:
                    if (
                        self._state.get("pending_page") == page_kind
                        and self._state.get("pending_message") in (None, message)
                    ):
                        self._state["pending_page"] = None
                        self._state["pending_message"] = None
                        self._safe_write(self._state)
                self._clear_retry_if_clean()
                return
        attempt = {
            "message": message,
            "done": threading.Event(),
            "error": None,
        }
        self._delivery_attempt = attempt
        threading.Thread(
            target=self._delivery_worker,
            args=(message, attempt),
            name="mem0-rerank-page-send",
            daemon=True,
        ).start()
        if not attempt["done"].wait(self._delivery_timeout_s):
            logger.error(
                "mem0 rerank page delivery timed out after %.3fs (retry armed)",
                self._delivery_timeout_s,
            )
            self._schedule_retry()
            return
        self._deliver_pending()

    def _retry_deferred(self) -> None:
        with self._lock:
            durable = not self._dirty_state or self._safe_write(self._state)
        if durable:
            self._deliver_pending()
        self._clear_retry_if_clean()

    def _dispatch_loop(self) -> None:
        with self._condition:
            self._owner_thread_id = threading.get_ident()
            self._condition.notify_all()
        while True:
            with self._condition:
                retry_due = self._retry_at is not None and self._retry_at <= time.monotonic()
                while not self._observations and not self._overflow_pending and not retry_due:
                    self._owner_busy = False
                    self._condition.notify_all()
                    timeout = (
                        max(0.0, self._retry_at - time.monotonic())
                        if self._retry_at is not None else None
                    )
                    self._condition.wait(timeout=timeout)
                    retry_due = self._retry_at is not None and self._retry_at <= time.monotonic()
                self._owner_busy = True
                overflow_count = min(self._overflow_pending, self._failure_threshold)
                self._overflow_pending = 0
                metadata = self._observations.popleft() if self._observations else None
                if retry_due:
                    self._retry_at = None
            try:
                for _ in range(overflow_count):
                    self._observe_one({
                        "status": "failure",
                        "failure_class": "observer_queue_overflow",
                        "effective_arm": RERANK_OFF,
                    })
                if metadata is not None:
                    self._observe_one(metadata)
                if retry_due:
                    self._retry_deferred()
            except Exception:
                with self._condition:
                    self._observer_error_count += 1
                logger.exception("mem0 rerank incident dispatcher failed")
            finally:
                with self._condition:
                    self._owner_busy = False
                    self._condition.notify_all()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Test/closeout seam: wait until the dispatcher has handled queued work."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while (
                self._observations
                or self._overflow_pending
                or self._owner_busy
                or self._retry_at is not None
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def _observe_one(self, metadata: Dict[str, Any]) -> None:
        """Record one rerank result and emit only onset/recovery transitions."""
        transition = None
        page_kind = None
        message = None
        failure_class = None
        p95 = None
        status = str((metadata or {}).get("status") or "failure").strip().lower()
        effective_arm = str((metadata or {}).get("effective_arm") or RERANK_OFF).strip().lower()

        with self._lock:
            state = self._state
            latencies = self._latencies
            persist = False
            durable = not self._dirty_state
            latency = (metadata or {}).get("latency_ms")
            if status == "success":
                if effective_arm != RERANK_BUILTIN:
                    status = "failure"
                    failure_class = "unexpected_effective_arm"
                else:
                    try:
                        if latency is None:
                            raise ValueError("latency is required")
                        latency_value = float(latency)
                        if not math.isfinite(latency_value) or latency_value < 0:
                            raise ValueError("latency must be finite and non-negative")
                        latency = latency_value
                        latencies.append(latency_value)
                    except (TypeError, ValueError):
                        status = "failure"
                        failure_class = "invalid_latency_metadata"
            latencies = latencies[-self._window_size:]
            self._latencies = latencies

            if status == "success" and latencies and self._latency_budget_ms > 0:
                # Early-window p95 (small n) degenerates toward max() — intentional:
                # conservative in the fail-loud direction, and an incident still
                # requires `failure_threshold` consecutive breaches of a budget
                # sized in seconds. Recovery from LATENCY-BREACH conversely
                # requires a FULL ring (see below) — asymmetry is by design.
                p95 = self._p95(latencies)
                if p95 > self._latency_budget_ms:
                    status = "failure"
                    failure_class = "LATENCY-BREACH"

            if status == "success":
                if state.get("active") and state.get("pending_page") == "onset":
                    # Do not let recovery overwrite an undelivered onset. Retry onset
                    # first; the next successful observation can transition recovery.
                    page_kind = "onset"
                    failure_class = state.get("failure_class")
                    effective_arm = state.get("observed_arm") or effective_arm
                elif (
                    state.get("active")
                    and state.get("failure_class") == "LATENCY-BREACH"
                    and len(latencies) < self._window_size
                ):
                    # A replacement process must rebuild a full ring before it can
                    # clear a host-persisted latency incident.
                    state["consecutive_failures"] = 0
                else:
                    state["consecutive_failures"] = 0
                    state["failure_class"] = None
                    state["observed_arm"] = None
                    if state.get("active"):
                        state["active"] = False
                        transition = "recovery"
            else:
                if state.get("pending_page") == "recovery":
                    # A renewed failure invalidates an undelivered recovery.
                    # Never acknowledge stale health while the detector is failing.
                    state["pending_page"] = None
                    state["pending_message"] = None
                    persist = True
                failure_class = failure_class or str(
                    (metadata or {}).get("failure_class") or status or "failure"
                )
                state["consecutive_failures"] = int(state.get("consecutive_failures") or 0) + 1
                state["failure_class"] = failure_class
                state["observed_arm"] = effective_arm
                if (
                    not state.get("active")
                    and state["consecutive_failures"] >= self._failure_threshold
                ):
                    state["active"] = True
                    transition = "onset"
            if transition is not None:
                message = self._page_message(
                    transition,
                    failure_class=failure_class,
                    effective_arm=effective_arm,
                    p95=p95,
                )
                state["pending_page"] = transition
                state["pending_message"] = message
                page_kind = transition
                persist = True
            elif (
                state.get("pending_page") == "onset" and state.get("active")
            ) or (
                state.get("pending_page") == "recovery" and not state.get("active")
            ):
                page_kind = state.get("pending_page")
                message = state.get("pending_message")
            self._state = state
            if persist:
                durable = self._safe_write(state)

        if transition is not None and message is not None:
            logger.warning(message)
        if page_kind is not None and durable:
            self._deliver_pending()
        elif page_kind is not None:
            self._schedule_retry()
        else:
            logger.debug(
                "mem0 rerank observation status=%s failure=%s latency_ms=%s",
                status, failure_class, latency,
            )


def get_rerank_incident_manager(
    *, state_path: Path, latency_budget_ms: float, failure_threshold: int = 3
) -> RerankIncidentManager:
    """Return the process-wide owner for one host-persisted state file."""
    key = str(Path(state_path).expanduser().resolve())
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(key)
        if manager is None:
            manager = RerankIncidentManager(
                state_path=Path(key),
                latency_budget_ms=latency_budget_ms,
                failure_threshold=failure_threshold,
            )
            _MANAGERS[key] = manager
        elif (
            manager._latency_budget_ms != max(0.0, float(latency_budget_ms))
            or manager._failure_threshold != max(1, int(failure_threshold))
        ):
            raise RuntimeError(
                "mem0 rerank incident manager already initialized with different config; "
                "gateway restart required"
            )
        return manager
