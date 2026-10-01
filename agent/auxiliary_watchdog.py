"""Stream watchdog, forward-progress hooks, and cancellation plumbing for auxiliary calls.

Sharded from ``agent/auxiliary_client.py`` (Part of #125186): the Codex Responses stream
watchdog (`_CodexStreamGuard`), the thread-local forward-progress hooks streamed aux wires
tick (`aux_progress_hook`, `_notify_aux_*`), the waiting host's absolute stream deadline
(`aux_stream_deadline`), the interrupt-protection scope shared by every wire
(`aux_interrupt_protection`), the task-scoped no-progress timeout reader, and the Anthropic
per-event progress hook.

Every moved name stays importable from ``agent.auxiliary_client`` (module-level re-export),
so callers and monkeypatch seams keep the facade binding. Symbols that still live in the
facade (cache eviction, total-ceiling budget, aux task config) are late-imported inside the
functions that need them — the facade never gains a module-level import back to here.
"""

import contextlib
import logging
import threading
import time
from typing import Any, Callable, Optional

from agent.codex_runtime import _codex_event_has_content

logger = logging.getLogger(__name__)

# Interrupt protection for atomic aux tasks: a compression summary killed by an ordinary
# gateway interrupt degrades to a static marker, so a thread-local flag marks such calls
# protected. Explicit host cancel (Ctrl+C, /stop) still overrides it, timeouts still fire.
# ── Interrupt protection for atomic auxiliary tasks ────────────────────── Some auxiliary tasks must NOT be
# aborted mid-flight by a gateway interrupt (e.g. an incoming user message while the agent is busy). Context
# compression is the prime case: if the summary LLM call is interrupted part-way, compression falls back to
# a static "summary unavailable" marker and the real handoff is lost (#23975). A thread-local flag lets such
# a task mark its in-flight LLM call as interrupt-protected; the Codex Responses stream's cancellation check
# honors it. TIMEOUTS still fire (a hung call must die), and all OTHER aux tasks (vision, web_extract,
# title_generation, …) remain freely interruptible.
_aux_interrupt_protection = threading.local()


class AuxiliaryExplicitCancellation(BaseException):
    """Frozen signal that an auxiliary attempt was explicitly hard-cancelled. ``BaseException`` so broad
    ``except Exception`` retry/fallback code never treats a host stop as a transport failure; ``cause``
    is immutable class data so nothing re-queries a mutable host Event after the transport unwound."""
    cause = "explicit_host_cancel"

    def __init__(self) -> None:
        super().__init__("auxiliary request explicitly cancelled by host")


def _aux_interrupt_protected() -> bool:
    return bool(getattr(_aux_interrupt_protection, "active", False))


def _aux_interrupt_cancel_requested() -> bool:
    """Return whether an explicit host cancel overrides aux protection."""
    check = _capture_aux_cancel_check()
    return _captured_aux_cancel_requested(check) if check is not None else False


@contextlib.contextmanager
def aux_interrupt_protection(active: bool = True, cancel_check=None, cancel_event=None):
    """Mark this thread's aux LLM call interrupt-protected (re-entrant-safe). ``cancel_check`` /
    ``cancel_event`` keep an explicit host hard-cancel path (Event preferred); nested scopes inherit both."""
    prev = getattr(_aux_interrupt_protection, "active", False)
    prev_cancel_check = getattr(_aux_interrupt_protection, "cancel_check", None)
    prev_cancel_event = getattr(_aux_interrupt_protection, "cancel_event", None)
    _aux_interrupt_protection.active = active
    if callable(cancel_check):
        _aux_interrupt_protection.cancel_check = cancel_check
    if cancel_event is not None and callable(getattr(cancel_event, "is_set", None)):
        _aux_interrupt_protection.cancel_event = cancel_event
    try:
        yield
    finally:
        _aux_interrupt_protection.active = prev
        _aux_interrupt_protection.cancel_check = prev_cancel_check
        _aux_interrupt_protection.cancel_event = prev_cancel_event


def _capture_aux_cancel_check() -> Optional[Callable[[], Any]]:
    """Capture the current explicit-cancel source on the owning request thread."""
    is_set = getattr(getattr(_aux_interrupt_protection, "cancel_event", None), "is_set", None)
    if callable(is_set):
        return is_set
    # Return the callable itself so attempt-local decision objects keep begin_timeout_cleanup().
    check = getattr(_aux_interrupt_protection, "cancel_check", None)
    return check if callable(check) else None


def _captured_aux_cancel_requested(cancel_check: Callable[[], Any]) -> bool:
    """Read a request-thread cancellation source without leaking its failures."""
    try:
        return bool(cancel_check())
    except Exception:
        logger.debug("captured aux cancel check failed", exc_info=True)
        return False



# Forward-progress hooks for streamed aux calls: a fixed host deadline kills a SLOW model
# streaming a big summary as hard as a HUNG one, so wire consumers tick the progress hook only
# for non-empty payloads and the host extends its deadline while tokens move. Thread-local:
# the call and its stream consumption run on the installing thread.
_aux_progress = threading.local()
_aux_dispatch = threading.local()
_aux_provider_response = threading.local()
# Absolute monotonic deadline of the waiting HOST. The stream's own ceiling
# (_aux_stream_total_ceiling, >= the host's and started later) would otherwise leave an
# orphaned stream still billing after every host-ceiling timeout.
# Absolute wall-clock deadline (time.monotonic) of the HOST waiting for this auxiliary call, when it has one
# (#99692). Liveness alone is not enough: a host also stops waiting at its own total ceiling, and the
# streamed consumer below bounds itself only by _aux_stream_total_ceiling() — a budget derived from the aux
# request timeout, which is >= the host ceiling for every configured value AND starts counting later. So the
# stream that outlives its abandoned host is not an edge case; it is the guaranteed outcome of every
# total-ceiling timeout.
_aux_stream_deadline = threading.local()


def _tick_hook(local: threading.local, label: str) -> None:
    """Call the thread-local hook installed on ``local``, if any. Never raises."""
    hook = getattr(local, "hook", None)
    if hook is None:
        return
    try:
        hook()
    except Exception:
        logger.debug("aux %s hook failed", label, exc_info=True)


def _notify_aux_progress() -> None:
    """Tick the installed forward-progress hook, if any."""
    _tick_hook(_aux_progress, "progress")


def _notify_aux_dispatch() -> None:
    """Record an actual provider dispatch without claiming response progress."""
    _tick_hook(_aux_dispatch, "dispatch")


def _notify_aux_timing_response() -> None:
    """Record a content-free frame (keepalive/empty delta): counts toward
    ``time_to_first_progress_ms`` but must not reset a compression inactivity fence."""
    _tick_hook(_aux_provider_response, "provider response")


def _notify_aux_provider_response() -> None:
    """Record a provider response/chunk, then preserve the liveness signal."""
    _notify_aux_timing_response()
    _notify_aux_progress()


def _aux_progress_active() -> bool:
    return getattr(_aux_progress, "hook", None) is not None



def _field(obj: Any, key: str, default: Any = None) -> Any:
    """Field access for wire objects that may be dicts or SDK/SimpleNamespace objects."""
    val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return default if val is None else val


def _anthropic_event_has_content(event: Any) -> bool:
    """Whether an Anthropic stream event carries a non-empty payload."""
    event_type = _field(event, "type")
    if event_type == "content_block_delta":
        delta = _field(event, "delta")
        return any(bool(_field(delta, f)) for f in ("text", "thinking", "partial_json", "signature", "citation"))
    if event_type == "content_block_start":
        block = _field(event, "content_block")
        return _field(block, "type") == "tool_use" and any(bool(_field(block, f)) for f in ("id", "name"))
    return False


def _anthropic_aux_stream_event_hook() -> Callable[[Any], None]:
    """Per-event callback for the Anthropic aux wire: progress only for substantive payloads
    (keepalives must not keep a stalled summary alive), stop at the host deadline or explicit
    cancel. The ``TimeoutError`` text must say "timed out" so ``_is_timeout_error`` classifies it."""
    host_deadline = _current_aux_stream_deadline()
    started = time.monotonic()

    def _on_event(event: Any) -> None:
        if _anthropic_event_has_content(event):
            _notify_aux_provider_response()
        else:
            _notify_aux_timing_response()
        if _aux_interrupt_cancel_requested():
            raise AuxiliaryExplicitCancellation()
        if host_deadline is not None and time.monotonic() >= host_deadline:
            raise TimeoutError(
                "Anthropic auxiliary stream timed out at the host compression "
                f"deadline after {time.monotonic() - started:.0f}s (the caller already stopped waiting)")

    return _on_event


# A dead stream fails at the no-progress window (first token AND between tokens); a live
# stream re-arms per event, bounded by _aux_stream_total_ceiling().
_AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS = 60.0


@contextlib.contextmanager
def _aux_thread_local_hook(local: threading.local, hook):
    """Install one thread-local hook, restoring the prior on exit (non-callable = passthrough)."""
    previous = getattr(local, "hook", None)
    local.hook = hook if callable(hook) else previous
    try:
        yield
    finally:
        local.hook = previous


@contextlib.contextmanager
def aux_progress_hook(hook):
    """Install *hook* as the current thread's aux forward-progress callback (None = passthrough)."""
    with _aux_thread_local_hook(_aux_progress, hook):
        yield


def _current_aux_stream_deadline() -> Optional[float]:
    """The waiting host's absolute monotonic deadline, if one is installed."""
    return getattr(_aux_stream_deadline, "value", None)


@contextlib.contextmanager
def aux_stream_deadline(deadline: Optional[float]):
    """Publish the host's absolute ``time.monotonic()`` deadline to the stream consumer.

    ``None`` is a passthrough; re-entrant-safe. Host->worker return leg of the progress hook:
    without it the isolated provider daemon streams to its own ceiling after the host stopped
    waiting, billing a summary the commit fence refuses.

    ``8207862212`` releases the compression OWNER when the fence is cancelled, but the isolated provider
    daemon (:func:`_run_protected_sync_provider_call`) that holds the socket keeps streaming to its own
    ``_aux_stream_total_ceiling`` budget — >= the host's ceiling by construction — billing an abandoned
    summary the commit fence is already guaranteed to refuse, and stacking one fresh orphan per turn on a
    session that compression never managed to shrink. See #99692.
    """
    previous = getattr(_aux_stream_deadline, "value", None)
    _aux_stream_deadline.value = deadline if isinstance(deadline, (int, float)) else previous
    try:
        yield
    finally:
        _aux_stream_deadline.value = previous



def _close_quietly(target: Any, failure_note: Optional[str]) -> None:
    """Call ``target.close()`` if present; a failure is debug-logged under ``failure_note`` (silent when None)."""
    close = getattr(target, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            if failure_note:
                logger.debug("Codex auxiliary: %s", failure_note, exc_info=True)


def _get_task_no_progress_timeout(task: str) -> Optional[float]:
    """``auxiliary.<task>.no_progress_timeout`` from config, or None when unset/invalid
    (the Codex stream guard then keeps its built-in ``_AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS``
    default). Lets an operator widen the substantive-progress window independently of the
    overall request timeout — see #108104."""
    if not task:
        return None
    from agent.auxiliary_client import _get_auxiliary_task_config
    raw = _get_auxiliary_task_config(task).get("no_progress_timeout")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        value = 0.0
    if isinstance(raw, bool) or value <= 0:
        # Fail clearly: a typo here silently leaving the 60s default is exactly the
        # "why did my 600s request abort after 60s" confusion the key exists to remove.
        logger.warning(
            "auxiliary.%s.no_progress_timeout=%r is not a positive number of seconds; "
            "using the built-in %.0fs default", task, raw, _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS)
        return None
    return value


def _stream_no_progress_timeout_seconds() -> float:
    """Read the default through the facade so its established patch seam remains live."""
    try:
        from agent import auxiliary_client
        value = auxiliary_client._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS
    except (ImportError, AttributeError):
        value = _AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS
    return float(value)


class _CodexStreamGuard:
    """Progress-aware deadline + FD-safe timeout watchdog for one Codex aux stream attempt.

    (1) The first substantive payload must arrive within ``no_progress_timeout`` or we fail fast into
    the caller's retry/fallback chain (a dead or keepalive-only zombie must not hold the budget);
    (2) each substantive event re-arms that window (keepalive/lifecycle frames do NOT, mirroring
    commit-fence gating) so a live stream is never killed by an absolute total; (3) a hard ceiling
    from ``_aux_stream_total_ceiling`` still terminates a pathological drip.
    """

    def __init__(
        self, client: Any, total_timeout: Optional[float],
        no_progress_timeout: Optional[float] = None,
    ):
        self._client = client
        self.total_timeout = total_timeout
        self._start = time.monotonic()
        # Task-scoped override (auxiliary.<task>.no_progress_timeout, #108104); falls back to the
        # built-in default when unset or not a positive number.
        if isinstance(no_progress_timeout, (int, float)) and no_progress_timeout > 0:
            self.no_progress_timeout = float(no_progress_timeout)
        else:
            self.no_progress_timeout = _stream_no_progress_timeout_seconds()
        # Progress-aware stream deadlines (supersedes the old single absolute kill at ``total_timeout``).
        # Three regimes: 1. First token: the stream must produce its first substantive payload within
        # ``no_progress_timeout`` (60s default) or we fail fast and let the caller's normal retry/fallback
        # chain run — a dead (or keepalive-only zombie) Codex stream no longer holds the full 300s
        # compression budget before falling back (masoria report, Aug 2026: 3 stacked 300s waits -> 20+ min
        # stuck on "Summarizing"). 2. Streaming: every substantive event re-arms the deadline by
        # ``no_progress_timeout`` — a live stream is never killed by an absolute total, so a long reasoning
        # summary that is actually producing tokens completes instead of timing out at 300s and falling back
        # (#54915's original complaint, fixed properly). Keepalive/lifecycle frames do NOT re-arm, mirroring
        # the commit-fence progress gating (#96707). 3. Hard ceiling: an absolute backstop from
        # ``_aux_stream_total_ceiling`` (max(600s, 4x configured timeout) — the same bound the streamed
        # chat.completions path uses) so a pathological one-token-per-59s drip still terminates.
        if total_timeout is not None:
            self.no_progress_timeout = min(self.no_progress_timeout, float(total_timeout))
        from agent.auxiliary_client import _aux_stream_total_ceiling
        self.hard_deadline = self._start + _aux_stream_total_ceiling(total_timeout)
        # The waiting host's absolute deadline clamps the ceiling so the watchdog Timer severs
        # the socket the instant the host stops waiting — a stream blocked between events
        # can't be stopped by a per-event check.
        host_deadline = _current_aux_stream_deadline()
        if isinstance(host_deadline, (int, float)) and host_deadline < self.hard_deadline:
            self.hard_deadline = float(host_deadline)
        self._deadline_lock = threading.Lock()
        self._progress_deadline = self._start + self.no_progress_timeout
        self.saw_content = threading.Event()
        self.timed_out = threading.Event()
        # Set only when the timeout WON (not when the owner hard-cancelled first): tells the
        # owner's ``finally`` the shared client's FDs still need a real close.
        self.timeout_release_pending = threading.Event()
        self.stream_finished = threading.Event()
        self._timer = None
        # The owner may return on hard cancel while this attempt is still blocked in the SDK
        # stream. Timer threads don't inherit the worker's thread-local protection state, so
        # freeze the hard-cancel source before creating the timer.
        self._protected_cancel_check = _capture_aux_cancel_check() if _aux_interrupt_protected() else None
        self._attempt_stream_lock = threading.Lock()
        self._attempt_stream: Any = None
        # The request-driving thread owns the transport FDs — see _close_client_on_timeout.
        self._owner_tid = threading.get_ident()

    def effective_deadline(self) -> float:
        with self._deadline_lock:
            return min(self.hard_deadline, self._progress_deadline)

    def cancel_requested(self) -> bool:
        """True when the frozen hard-cancel source says the owner already cancelled."""
        check = self._protected_cancel_check
        return callable(check) and _captured_aux_cancel_requested(check)

    def adopt_stream(self, stream: Any) -> None:
        with self._attempt_stream_lock:
            self._attempt_stream = stream

    def release_stream(self, stream: Any) -> None:
        """Owner-side: close the attempt stream silently and forget it."""
        _close_quietly(stream, None)
        with self._attempt_stream_lock:
            self._attempt_stream = None

    def close_attempt_stream(self, failure_note: str) -> None:
        """Closes only this attempt's stream — never the process-shared client."""
        with self._attempt_stream_lock:
            stream = self._attempt_stream
        _close_quietly(stream, failure_note)

    def record_progress(self) -> None:
        """Substantive payload re-arms the no-progress window; the hard ceiling never moves."""
        with self._deadline_lock:
            self._progress_deadline = time.monotonic() + self.no_progress_timeout

    def timeout_message(self) -> str:
        elapsed = time.monotonic() - self._start
        if time.monotonic() >= self.hard_deadline:
            return f"Codex auxiliary Responses stream exceeded {self.hard_deadline - self._start:.1f}s hard ceiling"
        if not self.saw_content.is_set():
            return (
                "Codex auxiliary Responses stream produced no output "
                f"within {float(self.no_progress_timeout):.1f}s (no-progress timeout, {elapsed:.1f}s elapsed)")
        return (
            "Codex auxiliary Responses stream stalled: no new output "
            f"for {float(self.no_progress_timeout):.1f}s ({elapsed:.1f}s elapsed)")

    def _close_client_on_timeout(self) -> None:
        begin_timeout_cleanup = getattr(self._protected_cancel_check, "begin_timeout_cleanup", None)
        if callable(begin_timeout_cleanup):
            timeout_won = bool(begin_timeout_cleanup())
        else:
            timeout_won = not self.cancel_requested()
        # Publish transport timeout only after the attempt-local decision is fixed, so owner
        # polling cannot observe completion in between.
        self.timed_out.set()
        if not timeout_won:
            # Owner already hard-cancelled. The OpenAI client is process-shared, so never
            # close/evict it here; wake only this attempt's stream if responses.create()
            # returned one, else rely on the bounded SDK timeout.
            self.close_attempt_stream("cancelled attempt stream close during timeout failed")
            return
        # FD-ownership contract: only the thread driving the request may ``close()`` this
        # client's FDs. From a stranger thread (the watchdog Timer) only ``shutdown()`` is
        # FD-safe — ``close()`` releases the raw TLS fd while the owner's OpenSSL BIO still
        # caches it, the kernel recycles it (e.g. into a SQLite handle), and the owner's TLS
        # flush corrupts that file. The owner does the real close in its ``finally``.
        # This callback has two callers — ``_check_cancelled`` on the owning thread, and the daemon watchdog
        # ``threading.Timer``, which is a stranger thread. The owning thread performs the real close in the
        # ``finally`` below, which is where the FD release belongs. See #70773.
        self.timeout_release_pending.set()
        if threading.get_ident() == self._owner_tid:
            _close_quietly(self._client, "client close during timeout failed")
        else:
            try:
                from agent.agent_runtime_helpers import force_close_tcp_sockets
                shutdown_count = force_close_tcp_sockets(self._client)
                logger.info(
                    "Codex auxiliary client aborted (timeout, tcp_force_closed=%d, "
                    "deferred_close=stranger_thread)", shutdown_count)
            except Exception:
                logger.debug("Codex auxiliary: client abort during timeout failed", exc_info=True)
            # Socket shutdown only wakes a reader on a REAL transport; the owner may be blocked
            # inside the SDK's event stream (or a socketless test double). Closing the
            # attempt-owned stream releases it without touching shared FDs.
            self.close_attempt_stream("attempt stream close during stranger-thread timeout failed")
        # The aux client cache wraps this same client; drop the entry so the next aux call
        # doesn't reuse the dead transport and fail fast.
        try:
            # After we close the httpx transport above, the cache must drop that entry — otherwise the next
            # auxiliary call (compression retry, memory flush, etc.) reuses the dead client and fails fast
            # with a connection error. See issue #23432.
            from agent.auxiliary_client import _evict_cached_client_instance
            _evict_cached_client_instance(self._client)
        except Exception:
            logger.debug("Codex auxiliary: cache eviction on timeout failed", exc_info=True)

    def check_cancelled(self) -> None:
        if self.total_timeout is not None and time.monotonic() >= self.effective_deadline():
            if not self.timed_out.is_set():
                self._close_client_on_timeout()
            raise TimeoutError(self.timeout_message())
        try:
            from tools.interrupt import is_interrupted
            # Protected atomic aux tasks (compression) must not abort on a mid-flight gateway
            # interrupt (degraded fallback marker); explicit host cancel has its own exception.
            if _aux_interrupt_cancel_requested():
                raise AuxiliaryExplicitCancellation()
            # Explicit host cancellation has its own frozen exception; timeouts above still fire and other
            # aux tasks remain interruptible. See #23975.
            if is_interrupted() and not _aux_interrupt_protected():
                raise InterruptedError("Codex auxiliary Responses stream interrupted")
        except InterruptedError:
            raise
        except Exception:
            # Interrupt state is best-effort UX; never a new failure mode.
            pass

    def _watchdog_fire(self) -> None:
        # Re-armable: if progress moved the deadline forward, reschedule instead of killing a
        # live stream.
        remaining = self.effective_deadline() - time.monotonic()
        if remaining > 0:
            if not (self.timed_out.is_set() or self.stream_finished.is_set()):
                self._arm_timer(remaining)
            return
        self._close_client_on_timeout()

    def _arm_timer(self, delay: float) -> None:
        self._timer = t = threading.Timer(delay, self._watchdog_fire)
        t.daemon = True
        t.start()

    def start(self) -> None:
        """Arm the watchdog (when a total timeout exists) and run the first cancel check."""
        if self.total_timeout:
            self._arm_timer(max(self.effective_deadline() - time.monotonic(), 0.0))
        self.check_cancelled()

    def on_event(self, _event: Any) -> None:
        # TTFP telemetry records every frame, but forward progress (compression commit fence,
        # no-progress window) counts only substantive payloads — keepalives must not re-arm,
        # so a zombie stream dies at the same window as a dead connection.
        # #93650: keep bulk wire-format payload out of the SDK's GIL-holding request transform on auxiliary
        # calls too.
        if _codex_event_has_content(_event):
            self.record_progress()
            self.saw_content.set()
            _notify_aux_provider_response()
        else:
            _notify_aux_timing_response()
        self.check_cancelled()

    def finish(self) -> None:
        """Owner ``finally``: stop the watchdog and release FDs a stranger-thread timeout only shut down."""
        self.stream_finished.set()
        if self._timer is not None:
            self._timer.cancel()
        # Gated on timeout_release_pending, NOT timed_out: after a hard-cancel the shared
        # client must stay usable for other sessions.
        if self.timeout_release_pending.is_set():
            _close_quietly(self._client, "owner-thread close after timeout failed")


