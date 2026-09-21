"""
Real-time Resource Monitor for Hermes-Agent.

Collects CPU, memory, token-usage, and latency metrics via a background thread
and exposes an immutable snapshot for observability.

Design principles (OS/observability):
- All state is observable: current metrics, peak values, rates, and thresholds
- Background collection is non-blocking — emitting threads are never stalled
- Integrates with EventBus for metrics events and structured logging for alerts
- Backward compatible: works without psutil or EventBus (degrades gracefully)

Usage:
    from agent.hermes.resource_monitor import ResourceMonitor, get_resource_monitor

    monitor = get_resource_monitor(session_id="sess-abc")
    monitor.start()

    # Inspect live state:
    snapshot = monitor.snapshot()
    print(f"CPU: {snapshot.cpu_percent:.1f}%, Mem: {snapshot.memory_mb:.1f}MB")

    monitor.stop()
"""

from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# ── psutil (optional) ──────────────────────────────────────────────────────────

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

# ── Internal imports ───────────────────────────────────────────────────────────

from agent.hermes.analytics import Event, EventType

logger: Optional[object] = None


def _get_logger():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ResourceSnapshot:
    """
    Observable snapshot of current resource state.

    All numeric fields are deterministic — no internal locks required to read.
    """
    # Timing
    timestamp: float = field(default_factory=time.time)   # epoch seconds
    elapsed_seconds: float = 0.0                          # session elapsed time

    # CPU
    cpu_percent: float = 0.0            # current CPU usage (0.0–100.0)
    cpu_percent_avg: float = 0.0        # rolling average since start

    # Memory (Python/process)
    memory_mb: float = 0.0             # current RSS in MB
    memory_mb_avg: float = 0.0         # rolling average
    memory_peak_mb: float = 0.0        # peak RSS seen

    # System memory (available only with psutil)
    system_memory_percent: float = 0.0  # system-wide mem pressure (0.0–100.0)

    # Token usage (cumulative)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Token rates (tokens per second, rolling window)
    tokens_per_second: float = 0.0
    input_tokens_per_second: float = 0.0
    output_tokens_per_second: float = 0.0

    # Latency (seconds)
    last_llm_latency_seconds: float = 0.0
    avg_llm_latency_seconds: float = 0.0
    max_llm_latency_seconds: float = 0.0
    last_tool_latency_seconds: float = 0.0
    avg_tool_latency_seconds: float = 0.0

    # Counters
    llm_call_count: int = 0
    tool_call_count: int = 0
    error_count: int = 0

    # Derived labels
    @property
    def token_rate_label(self) -> str:
        return f"{self.tokens_per_second:.1f} tok/s"

    @property
    def memory_label(self) -> str:
        return f"{self.memory_mb:.1f}MB (peak {self.memory_peak_mb:.1f}MB)"

    @property
    def cpu_label(self) -> str:
        return f"{self.cpu_percent:.1f}% (avg {self.cpu_percent_avg:.1f}%)"

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
            "cpu_percent": round(self.cpu_percent, 2),
            "cpu_percent_avg": round(self.cpu_percent_avg, 2),
            "memory_mb": round(self.memory_mb, 1),
            "memory_mb_avg": round(self.memory_mb_avg, 1),
            "memory_peak_mb": round(self.memory_peak_mb, 1),
            "system_memory_percent": round(self.system_memory_percent, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "input_tokens_per_second": round(self.input_tokens_per_second, 2),
            "output_tokens_per_second": round(self.output_tokens_per_second, 2),
            "last_llm_latency_seconds": round(self.last_llm_latency_seconds, 3),
            "avg_llm_latency_seconds": round(self.avg_llm_latency_seconds, 3),
            "max_llm_latency_seconds": round(self.max_llm_latency_seconds, 3),
            "last_tool_latency_seconds": round(self.last_tool_latency_seconds, 3),
            "avg_tool_latency_seconds": round(self.avg_tool_latency_seconds, 3),
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "error_count": self.error_count,
        }


# ──────────────────────────────────────────────────────────────────────────────
# ResourceMonitor
# ──────────────────────────────────────────────────────────────────────────────

class ResourceMonitor:
    """
    Real-time resource metrics collector with background thread.

    Collects CPU, memory, and derived metrics (token rate, latency averages).
    Backed by a background thread that samples at *sample_interval* seconds.
    All state is thread-safe via a single lock.

    Usage::

        monitor = ResourceMonitor(sample_interval=1.0)
        monitor.start()

        # ... agent runs ...

        snapshot = monitor.snapshot()
        print(snapshot.cpu_label)
        print(snapshot.token_rate_label)

        monitor.stop()

    The monitor also subscribes to the EventBus (if provided) to receive
    llm.response and tool.result events for token counts and latency.
    """

    def __init__(
        self,
        *,
        sample_interval: float = 1.0,
        event_bus: Optional[object] = None,
        session_id: str = "",
        on_metrics: Optional[Callable[[ResourceSnapshot], None]] = None,
    ):
        """
        Initialize ResourceMonitor.

        Args:
            sample_interval: Seconds between background samples (default: 1.0)
            event_bus: Optional EventBus to subscribe to for token/latency events
            session_id: Session ID for logging correlation
            on_metrics: Optional callback invoked after each sample
        """
        self._sample_interval = sample_interval
        self._event_bus = event_bus
        self._session_id = session_id
        self._on_metrics = on_metrics

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Token accumulators
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cache_read_tokens: int = 0
        self._cache_write_tokens: int = 0

        # Latency accumulators (for rolling average)
        self._llm_latencies: list[float] = []
        self._tool_latencies: list[float] = []
        self._max_llm_latency: float = 0.0
        self._max_tool_latency: float = 0.0

        # Counters
        self._llm_call_count: int = 0
        self._tool_call_count: int = 0
        self._error_count: int = 0

        # CPU/memory rolling state
        self._cpu_samples: list[float] = []
        self._mem_samples: list[float] = []
        self._peak_memory_mb: float = 0.0
        self._sample_count: int = 0
        self._cpu_sum: float = 0.0
        self._mem_sum: float = 0.0

        # Token rate window: (timestamp, input_tokens, output_tokens)
        self._token_window: list[tuple[float, int, int]] = []
        self._start_time: float = 0.0

        # EventBus subscription handle (for cleanup)
        self._bus_handler_refs: list = []

        # Subscribe to EventBus if provided
        if self._event_bus is not None:
            self._subscribe_to_event_bus()

    # ── EventBus integration ────────────────────────────────────────────────

    def _subscribe_to_event_bus(self) -> None:
        """Subscribe to EventBus for llm.response and tool.result events."""
        try:
            bus = self._event_bus

            def on_llm_response(event: Event) -> None:
                self._record_llm_event(event)

            def on_tool_result(event: Event) -> None:
                self._record_tool_event(event)

            def on_error(event: Event) -> None:
                self._record_error(event)

            bus.subscribe(EventType.LLM_RESPONSE, on_llm_response)
            bus.subscribe(EventType.TOOL_RESULT, on_tool_result)
            bus.subscribe(EventType.ERROR, on_error)

            self._bus_handler_refs = [on_llm_response, on_tool_result, on_error]
        except Exception as exc:
            _get_logger().warning("ResourceMonitor: failed to subscribe to EventBus: %s", exc)

    def _record_llm_event(self, event: Event) -> None:
        """Handle llm.response: accumulate tokens and latency."""
        with self._lock:
            payload = event.payload or {}

            # Tokens
            usage = payload.get("usage", {}) or {}
            self._input_tokens += usage.get("input_tokens", 0)
            self._output_tokens += usage.get("output_tokens", 0)
            self._cache_read_tokens += usage.get("cache_read_tokens", 0)
            self._cache_write_tokens += usage.get("cache_write_tokens", 0)
            self._llm_call_count += 1

            # Latency
            latency = payload.get("latency_seconds") or payload.get("duration_seconds") or 0.0
            if latency > 0:
                self._llm_latencies.append(latency)
                if len(self._llm_latencies) > 100:
                    self._llm_latencies = self._llm_latencies[-100:]
                if latency > self._max_llm_latency:
                    self._max_llm_latency = latency

    def _record_tool_event(self, event: Event) -> None:
        """Handle tool.result: accumulate tool call count and latency."""
        with self._lock:
            self._tool_call_count += 1

            payload = event.payload or {}
            latency = payload.get("latency_seconds") or payload.get("duration_seconds") or 0.0
            if latency > 0:
                self._tool_latencies.append(latency)
                if len(self._tool_latencies) > 100:
                    self._tool_latencies = self._tool_latencies[-100:]
                if latency > self._max_tool_latency:
                    self._max_tool_latency = latency

    def _record_error(self, event: Event) -> None:
        """Handle error events."""
        with self._lock:
            self._error_count += 1

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background collection thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()
            self._start_time = time.time()
            self._thread = threading.Thread(target=self._run, name="ResourceMonitor", daemon=True)
            self._thread.start()
        _get_logger().debug("ResourceMonitor started (interval=%.1fs)", self._sample_interval)

    def stop(self) -> None:
        """Stop background collection thread."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Unsubscribe from EventBus
        if self._event_bus is not None and self._bus_handler_refs:
            try:
                for handler in self._bus_handler_refs:
                    self._event_bus.unsubscribe(EventType.LLM_RESPONSE, handler)
                    self._event_bus.unsubscribe(EventType.TOOL_RESULT, handler)
                    self._event_bus.unsubscribe(EventType.ERROR, handler)
            except Exception:
                pass

        _get_logger().debug("ResourceMonitor stopped")

    # ── Background loop ──────────────────────────────────────────────────────

    def _run(self) -> None:
        """Background collection loop (runs in dedicated thread)."""
        while not self._stop_event.is_set():
            self._collect_sample()
            self._emit_metrics_event()
            if self._on_metrics:
                snap = self.snapshot()
                try:
                    self._on_metrics(snap)
                except Exception:
                    pass
            self._stop_event.wait(timeout=self._sample_interval)

    def _collect_sample(self) -> None:
        """Collect a single CPU/memory sample. Thread-safe."""
        with self._lock:
            now = time.time()
            elapsed = now - self._start_time if self._start_time else 0.0
            self._sample_count += 1

            # CPU
            cpu = 0.0
            system_mem_pct = 0.0
            if _PSUTIL_AVAILABLE:
                try:
                    cpu = psutil.cpu_percent(interval=None)
                    sys_mem = psutil.virtual_memory()
                    system_mem_pct = sys_mem.percent
                except Exception:
                    pass

            self._cpu_samples.append(cpu)
            if len(self._cpu_samples) > 60:
                self._cpu_samples = self._cpu_samples[-60:]
            self._cpu_sum += cpu

            # Memory
            mem_mb = 0.0
            if _PSUTIL_AVAILABLE:
                try:
                    proc = psutil.Process(os.getpid())
                    mem_mb = proc.memory_info().rss / (1024 * 1024)
                except Exception:
                    pass

            self._mem_samples.append(mem_mb)
            if len(self._mem_samples) > 60:
                self._mem_samples = self._mem_samples[-60:]
            self._mem_sum += mem_mb
            if mem_mb > self._peak_memory_mb:
                self._peak_memory_mb = mem_mb

            # Token rate (tokens in last N seconds)
            window_seconds = 10.0
            cutoff = now - window_seconds
            self._token_window = [(t, i, o) for t, i, o in self._token_window if t >= cutoff]
            self._token_window.append((now, self._input_tokens, self._output_tokens))

            # Emit a metrics event to the EventBus if available
            self._last_now = now
            self._last_elapsed = elapsed

    def _emit_metrics_event(self) -> None:
        """Emit a metrics snapshot event to the EventBus."""
        if self._event_bus is None:
            return
        try:
            snap = self.snapshot()
            self._event_bus.emit_event(
                EventType.METRICS_SAMPLE if hasattr(EventType, "METRICS_SAMPLE") else "metrics.sample",
                snap.to_dict(),
                session_id=self._session_id,
            )
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def record_tokens(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """
        Manually record token usage (for callers not using EventBus).

        Thread-safe.
        """
        with self._lock:
            self._input_tokens += max(0, input_tokens)
            self._output_tokens += max(0, output_tokens)
            self._cache_read_tokens += max(0, cache_read_tokens)
            self._cache_write_tokens += max(0, cache_write_tokens)

    def record_llm_latency(self, latency_seconds: float) -> None:
        """Manually record an LLM call latency (seconds). Thread-safe."""
        with self._lock:
            self._llm_call_count += 1
            if latency_seconds > 0:
                self._llm_latencies.append(latency_seconds)
                if len(self._llm_latencies) > 100:
                    self._llm_latencies = self._llm_latencies[-100:]
                if latency_seconds > self._max_llm_latency:
                    self._max_llm_latency = latency_seconds

    def record_tool_latency(self, latency_seconds: float) -> None:
        """Manually record a tool call latency (seconds). Thread-safe."""
        with self._lock:
            self._tool_call_count += 1
            if latency_seconds > 0:
                self._tool_latencies.append(latency_seconds)
                if len(self._tool_latencies) > 100:
                    self._tool_latencies = self._tool_latencies[-100:]

    def record_error(self) -> None:
        """Manually record an error. Thread-safe."""
        with self._lock:
            self._error_count += 1

    def snapshot(self) -> ResourceSnapshot:
        """
        Return an immutable snapshot of current resource state.

        Thread-safe.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._start_time if self._start_time else 0.0

            # Token rate (tokens per second over last 10s window)
            if len(self._token_window) >= 2:
                oldest_t, oldest_i, oldest_o = self._token_window[0]
                newest_t, newest_i, newest_o = self._token_window[-1]
                duration = newest_t - oldest_t
                if duration > 0.1:
                    delta_in = newest_i - oldest_i
                    delta_out = newest_o - oldest_o
                    delta_total = delta_in + delta_out
                    tps = delta_total / duration
                    ips = delta_in / duration
                    ops = delta_out / duration
                else:
                    tps = ips = ops = 0.0
            else:
                tps = ips = ops = 0.0

            # CPU/memory averages
            n = self._sample_count or 1
            cpu_avg = self._cpu_sum / n
            mem_avg = self._mem_sum / n if self._mem_samples else 0.0

            # System memory
            sys_mem_pct = 0.0
            if _PSUTIL_AVAILABLE and self._mem_samples:
                try:
                    sys_mem_pct = psutil.virtual_memory().percent
                except Exception:
                    pass

            # Latency averages
            avg_llm = sum(self._llm_latencies) / len(self._llm_latencies) if self._llm_latencies else 0.0
            avg_tool = sum(self._tool_latencies) / len(self._tool_latencies) if self._tool_latencies else 0.0
            last_llm = self._llm_latencies[-1] if self._llm_latencies else 0.0
            last_tool = self._tool_latencies[-1] if self._tool_latencies else 0.0

            return ResourceSnapshot(
                timestamp=now,
                elapsed_seconds=elapsed,
                cpu_percent=self._cpu_samples[-1] if self._cpu_samples else 0.0,
                cpu_percent_avg=round(cpu_avg, 2),
                memory_mb=round(self._mem_samples[-1], 1) if self._mem_samples else 0.0,
                memory_mb_avg=round(mem_avg, 1),
                memory_peak_mb=round(self._peak_memory_mb, 1),
                system_memory_percent=round(sys_mem_pct, 2),
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                total_tokens=self._input_tokens + self._output_tokens,
                cache_read_tokens=self._cache_read_tokens,
                cache_write_tokens=self._cache_write_tokens,
                tokens_per_second=round(tps, 2),
                input_tokens_per_second=round(ips, 2),
                output_tokens_per_second=round(ops, 2),
                last_llm_latency_seconds=last_llm,
                avg_llm_latency_seconds=round(avg_llm, 3),
                max_llm_latency_seconds=round(self._max_llm_latency, 3),
                last_tool_latency_seconds=last_tool,
                avg_tool_latency_seconds=round(avg_tool, 3),
                llm_call_count=self._llm_call_count,
                tool_call_count=self._tool_call_count,
                error_count=self._error_count,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton (optional — callers can also instantiate directly)
# ──────────────────────────────────────────────────────────────────────────────

_resource_monitors: dict[str, ResourceMonitor] = {}
_monitors_lock = threading.Lock()


def get_resource_monitor(
    session_id: str,
    *,
    sample_interval: float = 1.0,
    event_bus: Optional[object] = None,
) -> ResourceMonitor:
    """
    Get (or create) a ResourceMonitor singleton for *session_id*.

    Thread-safe. Subsequent calls with the same session_id return the same instance.
    """
    with _monitors_lock:
        if session_id not in _resource_monitors:
            _resource_monitors[session_id] = ResourceMonitor(
                sample_interval=sample_interval,
                event_bus=event_bus,
                session_id=session_id,
            )
        return _resource_monitors[session_id]


def stop_all_monitors() -> None:
    """Stop all active ResourceMonitor instances."""
    with _monitors_lock:
        for monitor in list(_resource_monitors.values()):
            monitor.stop()
        _resource_monitors.clear()
