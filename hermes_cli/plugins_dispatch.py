"""Plugin hook / middleware / event-bus / system-prompt-section dispatch.

Mixed into :class:`hermes_cli.plugins.PluginManager`. ``_resolve_hook_callback_timeout`` stays on
the origin (tests patch it there) and is looked up lazily.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import hmac
import inspect
import logging
import queue
import re
import threading
import time
import types
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Mapping, Optional, Set, Union

from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION

logger = logging.getLogger("hermes_cli.plugins")

# Allowlist of agent-turn hot-path hooks bounded by plugins.hook_callback_timeout (fail-open:
# abandon without join — joining reintroduced a shutdown hang). Unlisted hooks run synchronously.
# Intentionally unbounded: on_session_finalize/reset (last-chance flush — abandon can lose state);
# subagent_start (observer); pre_gateway_dispatch (policy gate — neither fail mode is acceptable);
# pre/post_approval_* (approval UX has its own timeout); kanban_* (own heartbeat/stale reclaim).
# The goal is to stop a hung Python plugin callback from wedging the conversation loop (#76821) without
# joining the worker (avoids the #6622 ThreadPoolExecutor shutdown hang). Hooks not listed below run
# synchronously to completion. (on_session_start/end stay bounded — they sit on the common session-boundary
# path.) - subagent_start — observer only; blocking delegation belongs in pre_tool_call. Lower frequency
# than tool/LLM hooks. Abandoning is unsafe either way (fail-open skips auth-like checks; fail-closed can
# drop legitimate messages). Prefer finish-or-exception fallthrough. - pre_approval_request /
# post_approval_response — observers only (cannot veto); the approval UX already has its own timeout; not on
# the tool loop hot path. - kanban_task_* — fire after the board DB commit, observers only, in
# dispatcher/worker processes; kanban has its own heartbeat/stale reclaim. Abandon-without-join also leaves
# a daemon thread that may still mutate shared state — safer for value-returning observers than for
# gates/flushes.
_HOOK_TIMEOUT_BOUNDED_HOOKS: Set[str] = {
    "post_tool_call", "transform_terminal_output", "transform_tool_result", "transform_llm_output",
    "pre_llm_call", "post_llm_call", "pre_api_request", "post_api_request", "api_request_error",
    "pre_verify", "on_session_start", "on_session_end",
}

# Policy hooks: timeout / still-running must fail closed (block the tool).
_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS: Set[str] = {"pre_tool_call"}
# Documented parent-thread serialization contract — never run on a timeout worker (hooks.md).
_HOOK_CALLER_THREAD_HOOKS: Set[str] = {"subagent_stop"}
# After a timeout, suppress the same callback this long so a hung hook cannot pile up threads.
_HOOK_TIMEOUT_SUPPRESSION_SECONDS = 60.0
_PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE = "pre_tool_call plugin callback timed out or is still running"

_HOOK_CALLBACK_TELEMETRY_SCHEMA = "hermes-plugin-callback-telemetry/v1"
_HOOK_CALLBACK_TELEMETRY_MAX_EVENTS = 256
_HOOK_CALLBACK_TELEMETRY_OUTCOMES = frozenset({
    "timed_out",
    "suppression_window",
    "callback_abandoned",
    "same_identity_running",
    "worker_start_failed",
    "callback_exception",
})
_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS = {
    "hook": 128,
    "plugin": 128,
    "callback": 192,
    "exception_type": 128,
    "exception_message": 512,
}
_HOOK_CALLBACK_CORRELATION_INPUT_CHARS = 512
_HOOK_FAILURE_DEDUPE_MAX = 256
_MAX_TELEMETRY_MILLISECONDS = (1 << 63) - 1

# System-prompt sections are tightly bounded: they become high-trust prompt bytes charged every turn.
SYSTEM_PROMPT_SECTION_POSITIONS = frozenset({"after_memory"})
DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTION_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTIONS = 32
MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS = 8_000
_SYSTEM_PROMPT_SECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SYSTEM_PROMPT_SECTION_HEADING_PREFIX = "## Plugin Context: "
PLUGIN_SECTIONS_START = "<!-- hermes-plugin-sections:start -->"
PLUGIN_SECTIONS_END = "<!-- hermes-plugin-sections:end -->"


def is_valid_system_prompt_section_id(value: Any) -> bool:
    """Return whether *value* is a stable, heading-safe section identifier."""
    return isinstance(value, str) and bool(_SYSTEM_PROMPT_SECTION_ID_RE.fullmatch(value))


def format_system_prompt_section(section_id: str, content: str) -> str:
    """Render an auditable, length-framed block recoverable from the full prompt."""
    return (
        f"{_SYSTEM_PROMPT_SECTION_HEADING_PREFIX}{section_id}\n"
        f"<!-- hermes-plugin-section-chars:{len(content)} -->\n\n{content}")


def format_system_prompt_sections(sections: list) -> str:
    """Render the canonical container used for persistence recovery."""
    if not sections:
        return ""
    blocks = [format_system_prompt_section(item.id, item.content) for item in sections]
    return f"{PLUGIN_SECTIONS_START}\n" + "\n\n".join(blocks) + f"\n{PLUGIN_SECTIONS_END}"


# Reserved event namespace prefix — only core may publish ``hermes:<event>``.
HERMES_EVENT_NAMESPACE = "hermes"
# Event recursion depth cap (subscribers may emit); over-deep emits are dropped with a warning.
_EVENT_EMIT_DEPTH_CAP = 8
# Max queued + running events per manager generation; emit never waits — a full budget drops.
_EVENT_PENDING_CAP = 64
_EVENT_WORKER_STOP = object()


@dataclass(frozen=True)
class PluginSystemPromptSection:
    """A plugin-owned section rendered once for each new session."""

    id: str
    content: Union[str, Callable[[Mapping[str, Any]], str]]
    position: str
    max_chars: int
    plugin: str


@dataclass(frozen=True)
class RenderedPluginSystemPromptSection:
    """Validated prompt bytes frozen on the owning AIAgent."""

    id: str
    content: str
    position: str
    plugin: str


@dataclass(frozen=True)
class _EventSubscription:
    """Host-owned subscription ledger entry."""

    owner: str
    callback: Callable


@dataclass(frozen=True)
class _QueuedPluginEvent:
    """Immutable dispatch envelope consumed by the event worker."""

    event: str
    payload: Dict[str, Any]
    subscriptions: tuple[_EventSubscription, ...]
    depth: int
    generation: int
    # The emitter's contextvars: the single worker thread serves every profile, so each delivery
    # runs under the profile scope the emit happened in (#118538).
    context: contextvars.Context


# Hook callback timeout (non-blocking abandon). Default cap per Python hook callback; overridden by
# ``plugins.hook_callback_timeout``. Shell hooks enforce their own subprocess timeout.
_HOOK_CALLBACK_TIMEOUT_SECS = 30.0
_MAX_HOOK_CALLBACK_TIMEOUT_SECS = 600.0
_HOOK_SKIPPED = object()  # returned by _run_hook_callback_bounded on skip/timeout


def _hook_call_correlation(kwargs: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """Trusted field name and opaque logical identity for one bounded hook call.

    Tool calls are the narrowest grain, followed by turns and sessions. Deliberately do
    not use ``api_request_id`` here: one provider request can contain several tool calls.
    """
    for field in ("tool_call_id", "turn_id", "session_id"):
        value = kwargs.get(field)
        if isinstance(value, str) and value:
            return field, value
    return "none", None


def _hook_call_identity(kwargs: Dict[str, Any]) -> Optional[str]:
    """Opaque logical identity used by the healthy in-flight gate."""
    return _hook_call_correlation(kwargs)[1]


def _printable_collapsed_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value))
    printable = "".join(char if char.isprintable() else " " for char in normalized)
    return " ".join(printable.split())


def _canonical_telemetry_text(value: Any, *, max_chars: int, fallback: str) -> str:
    """Return printable, whitespace-collapsed, forcibly redacted bounded text."""
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(
            _printable_collapsed_text(value),
            force=True,
            redact_url_credentials=True,
        )
        return _printable_collapsed_text(redacted)[:max_chars] or fallback
    except Exception:
        return fallback[:max_chars]


def _canonical_correlation_bytes(field: str, value: str) -> bytes:
    """Canonical, bounded HMAC input. The returned bytes are never retained."""
    collapsed = _printable_collapsed_text(value)[:_HOOK_CALLBACK_CORRELATION_INPUT_CHARS]
    return f"{field}:{collapsed}".encode("utf-8", errors="replace")


def _bounded_milliseconds(value: float) -> int:
    try:
        milliseconds = int(max(0.0, float(value)) * 1_000)
    except (TypeError, ValueError, OverflowError):
        return 0
    return min(milliseconds, _MAX_TELEMETRY_MILLISECONDS)


def _hook_uses_callback_timeout(hook_name: str, timeout: float) -> bool:
    """Whether *hook_name* should run under the non-blocking timeout path."""
    if timeout <= 0 or hook_name in _HOOK_CALLER_THREAD_HOOKS:
        return False
    return hook_name in _HOOK_TIMEOUT_BOUNDED_HOOKS or hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS


class PluginDispatchMixin:
    if TYPE_CHECKING:
        _hook_callback_telemetry_secret: bytes
        _hook_callback_owners: Dict[int, list[tuple[Callable, str, object]]]
        _hook_callback_telemetry_lock: threading.Lock
        _hook_callback_telemetry_sequence: int
        _hook_callback_telemetry_events: Deque[Mapping[str, Any]]
        _hook_failures_reported: Dict[tuple, None]

    def _record_hook_callback_telemetry(
        self,
        *,
        outcome: str,
        hook_name: str,
        callback: Callable,
        kwargs: Dict[str, Any],
        timeout: Optional[float] = None,
        elapsed: Optional[float] = None,
        exception: Optional[Exception] = None,
    ) -> Optional[Mapping[str, Any]]:
        """Append one privacy-bounded diagnostic event and return its immutable copy.

        This method is called only through ``_record_hook_callback_telemetry_safely`` by
        callback dispatch. Keeping the builder separate makes fault isolation testable.
        """
        if outcome not in _HOOK_CALLBACK_TELEMETRY_OUTCOMES:
            raise ValueError(f"unsupported plugin callback telemetry outcome: {outcome!r}")
        identity_field, identity = _hook_call_correlation(kwargs)
        correlation_digest = "none"
        if identity is not None:
            correlation_digest = hmac.new(
                self._hook_callback_telemetry_secret,
                _canonical_correlation_bytes(identity_field, identity),
                hashlib.sha256,
            ).hexdigest()[:32]

        callback_module = getattr(callback, "__module__", "")
        callback_name = getattr(callback, "__qualname__", None)
        if callback_name is None:
            callback_name = getattr(callback, "__name__", type(callback).__qualname__)
        callback_label = f"{callback_module}.{callback_name}" if callback_module else callback_name
        owner_entries = self._hook_callback_owners.get(id(callback), ())
        owner = (
            owner_entries[-1][1]
            if owner_entries and owner_entries[-1][0] is callback
            else "core/unowned"
        )

        event: Dict[str, Any] = {
            "schema_version": _HOOK_CALLBACK_TELEMETRY_SCHEMA,
            "observed_at_monotonic_ms": _bounded_milliseconds(time.monotonic()),
            "outcome": outcome,
            "hook": _canonical_telemetry_text(
                hook_name,
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["hook"],
                fallback="unknown",
            ),
            "plugin": _canonical_telemetry_text(
                owner,
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["plugin"],
                fallback="core/unowned",
            ),
            "callback": _canonical_telemetry_text(
                callback_label,
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["callback"],
                fallback="unknown",
            ),
            "identity_field": identity_field,
            "correlation_digest": correlation_digest,
        }
        if timeout is not None:
            event["timeout_ms"] = _bounded_milliseconds(timeout)
        if elapsed is not None:
            event["elapsed_ms"] = _bounded_milliseconds(elapsed)
        if exception is not None:
            event["exception_type"] = _canonical_telemetry_text(
                type(exception).__name__,
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["exception_type"],
                fallback="Exception",
            )
            event["exception_message"] = _canonical_telemetry_text(
                exception,
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["exception_message"],
                fallback="redacted",
            )
        with self._hook_callback_telemetry_lock:
            self._hook_callback_telemetry_sequence += 1
            event["sequence"] = self._hook_callback_telemetry_sequence
            stored = types.MappingProxyType(event.copy())
            self._hook_callback_telemetry_events.append(stored)
        return types.MappingProxyType(event.copy())

    def _record_hook_callback_telemetry_safely(self, **event: Any) -> Optional[Mapping[str, Any]]:
        """Best-effort wrapper: telemetry can never change callback behavior."""
        try:
            return self._record_hook_callback_telemetry(**event)
        except BaseException:
            try:
                logger.debug("Plugin callback telemetry recording failed", exc_info=True)
            except BaseException:
                pass
            return None

    @staticmethod
    def _safe_hook_callback_labels(
        hook_name: str, callback: Callable
    ) -> tuple[str, str]:
        """Return canonical labels safe for callback diagnostics."""
        callback_module = getattr(callback, "__module__", "")
        callback_name = getattr(callback, "__qualname__", None)
        if callback_name is None:
            callback_name = getattr(callback, "__name__", type(callback).__qualname__)
        callback_label = (
            f"{callback_module}.{callback_name}" if callback_module else callback_name
        )
        return (
            _canonical_telemetry_text(
                hook_name,
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["hook"],
                fallback="unknown",
            ),
            _canonical_telemetry_text(
                callback_label,
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["callback"],
                fallback="unknown",
            ),
        )

    def get_hook_callback_telemetry(self, limit: int = _HOOK_CALLBACK_TELEMETRY_MAX_EVENTS) -> tuple:
        """Return an immutable newest-tail snapshot with a bounded caller limit."""
        try:
            bounded_limit = max(0, min(int(limit), _HOOK_CALLBACK_TELEMETRY_MAX_EVENTS))
        except (TypeError, ValueError, OverflowError):
            bounded_limit = _HOOK_CALLBACK_TELEMETRY_MAX_EVENTS
        if bounded_limit == 0:
            return ()
        with self._hook_callback_telemetry_lock:
            tail = tuple(self._hook_callback_telemetry_events)[-bounded_limit:]
            return tuple(types.MappingProxyType(dict(event)) for event in tail)

    @staticmethod
    def _invoke_hook_callback(callback: Callable, payload: Dict[str, Any]) -> Any:
        """Invoke a hook while withholding additive fields from narrow legacy callbacks.

        An ``async def`` callback returns a coroutine; resolve it the way plugin slash commands
        are (loop-safe), otherwise the bare coroutine object is appended to the results and the
        plugin's body never runs (#12449).
        """
        from hermes_cli.plugins import resolve_plugin_command_result
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            return resolve_plugin_command_result(callback(**payload))  # no introspectable signature
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return resolve_plugin_command_result(callback(**payload))
        keyword_kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        return resolve_plugin_command_result(callback(**{
            name: value for name, value in payload.items()
            if name in parameters and parameters[name].kind in keyword_kinds
        }))

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Call all callbacks for *hook_name*; return their non-``None`` results.

        Payloads evolve additively: ``**kwargs`` callbacks get everything, narrow signatures only
        what they declare. Each callback is isolated. Bounded hooks and ``pre_tool_call`` run under
        ``plugins.hook_callback_timeout`` (worker abandoned, never joined); ``pre_tool_call`` fails
        closed with a block directive, others skip. ``_HOOK_CALLER_THREAD_HOOKS`` always run on the
        caller thread. ``pre_llm_call`` may return ``{"context": "..."}`` (or a str) to inject.
        """
        from hermes_cli.plugins import _resolve_hook_callback_timeout
        # Gateway platform events define event-local envelopes; a bus-wide version here would turn
        # unrelated adapter payloads into one monolithic compatibility contract.
        if hook_name != "gateway_platform_event":
            kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        results: List[Any] = []
        timeout = _resolve_hook_callback_timeout()
        use_timeout = _hook_uses_callback_timeout(hook_name, timeout)
        fail_closed = hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
        for cb in self._hooks.get(hook_name, []):
            try:
                if use_timeout:
                    ret = self._run_hook_callback_bounded(hook_name, cb, kwargs, timeout)
                    if ret is _HOOK_SKIPPED:
                        if fail_closed:  # policy hook: fail closed with a block directive
                            results.append({"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE})
                        continue
                else:
                    ret = self._invoke_hook_callback(cb, kwargs)
                if ret is not None:
                    results.append(ret)
            except (Exception, SystemExit) as exc:
                self._report_hook_failure(hook_name, cb, kwargs, exc)
        return results

    def _report_hook_failure(
        self, hook_name: str, cb: Callable, kwargs: Dict[str, Any], exc: BaseException, *, surface: str = "Hook"
    ) -> None:
        """Report one safe WARNING per distinct failure and record callback exceptions.

        Hook exception text is untrusted plugin output. Canonical redaction happens before
        both the bounded dedupe key and every emitted log message.
        """
        callback_name = _canonical_telemetry_text(
            getattr(cb, "__name__", type(cb).__qualname__),
            max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["callback"],
            fallback="unknown",
        )
        safe_message = _canonical_telemetry_text(
            exc,
            max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["exception_message"],
            fallback="redacted",
        )
        if surface == "Hook":
            self._record_hook_callback_telemetry_safely(
                outcome="callback_exception",
                hook_name=hook_name,
                callback=cb,
                kwargs=kwargs,
                exception=exc,
            )
        key = (
            _canonical_telemetry_text(
                hook_name,
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["hook"],
                fallback="unknown",
            ),
            _canonical_telemetry_text(
                getattr(cb, "__module__", ""),
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["callback"],
                fallback="unknown",
            ),
            _canonical_telemetry_text(
                getattr(cb, "__qualname__", callback_name),
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["callback"],
                fallback=callback_name,
            ),
            type(exc).__name__,
            safe_message[:200],
        )
        if key in self._hook_failures_reported:
            logger.debug(
                "%s '%s' callback %s raised again: %s",
                surface,
                key[0],
                callback_name,
                safe_message,
            )
            return
        self._hook_failures_reported[key] = None
        while len(self._hook_failures_reported) > _HOOK_FAILURE_DEDUPE_MAX:
            self._hook_failures_reported.pop(next(iter(self._hook_failures_reported)))
        logger.warning(
            "%s '%s' callback %s raised: %s (%s provides: %s; identical failures are logged at DEBUG from now on)",
            surface,
            key[0],
            callback_name,
            safe_message,
            surface.lower(),
            ", ".join(sorted(kwargs)) or "no fields",
        )

    def _run_hook_callback_bounded(
        self, hook_name: str, cb: Callable, kwargs: Dict[str, Any], timeout: float
    ) -> Any:
        """Run one callback under the existing non-blocking timeout contract."""
        callback_name = _canonical_telemetry_text(
            getattr(cb, "__name__", type(cb).__qualname__),
            max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["callback"],
            fallback="unknown",
        )
        suppression_key = (hook_name, id(cb))
        gate_key = (*suppression_key, _hook_call_identity(kwargs))
        token = object()
        skip_outcome: Optional[str] = None
        now = time.monotonic()
        with self._hook_timeout_lock:
            suppressed_until = self._hook_timeout_suppressed_until.get(suppression_key)
            abandoned = bool(self._hook_abandoned.get(suppression_key))
            if abandoned:
                skip_outcome = "callback_abandoned"
            elif suppressed_until is not None and suppressed_until > now:
                skip_outcome = "suppression_window"
            elif gate_key in self._hook_running_callbacks:
                skip_outcome = "same_identity_running"
            if skip_outcome is None:
                if suppressed_until is not None:
                    self._hook_timeout_suppressed_until.pop(suppression_key, None)
                self._hook_running_callbacks[gate_key] = token

        if skip_outcome is not None:
            event = self._record_hook_callback_telemetry_safely(
                outcome=skip_outcome,
                hook_name=hook_name,
                callback=cb,
                kwargs=kwargs,
                timeout=timeout,
            )
            safe_hook, safe_callback = (
                (event["hook"], event["callback"])
                if event is not None
                else self._safe_hook_callback_labels(hook_name, cb)
            )
            logger.warning(
                "Hook '%s' callback %s skipped: %s",
                safe_hook,
                safe_callback,
                skip_outcome,
            )
            return _HOOK_SKIPPED

        context = contextvars.copy_context()
        done = threading.Event()
        callback_outcome: Dict[str, Any] = {}
        failure: Dict[str, BaseException] = {}
        started_at = time.monotonic()

        def _release_token() -> None:
            with self._hook_timeout_lock:
                if self._hook_running_callbacks.get(gate_key) is token:
                    self._hook_running_callbacks.pop(gate_key, None)
                    abandoned_gates = self._hook_abandoned.get(suppression_key)
                    if abandoned_gates is not None:
                        abandoned_gates.discard(gate_key)
                        if not abandoned_gates:
                            self._hook_abandoned.pop(suppression_key, None)

        def _runner() -> None:
            try:
                callback_outcome["value"] = context.run(
                    self._invoke_hook_callback, cb, kwargs
                )
            except BaseException as exc:
                failure["exc"] = exc
            finally:
                _release_token()
                done.set()

        thread = threading.Thread(
            target=_runner,
            name=f"hermes-hook-{callback_name}"[:40],
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError as exc:
            _release_token()
            event = self._record_hook_callback_telemetry_safely(
                outcome="worker_start_failed",
                hook_name=hook_name,
                callback=cb,
                kwargs=kwargs,
                timeout=timeout,
                elapsed=time.monotonic() - started_at,
            )
            safe_hook, safe_callback = (
                (event["hook"], event["callback"])
                if event is not None
                else self._safe_hook_callback_labels(hook_name, cb)
            )
            safe_message = _canonical_telemetry_text(
                exc,
                max_chars=_HOOK_CALLBACK_TELEMETRY_LABEL_LIMITS["exception_message"],
                fallback="redacted",
            )
            logger.warning(
                "Hook '%s' callback %s worker failed to start: %s — skipping",
                safe_hook,
                safe_callback,
                safe_message,
            )
            return _HOOK_SKIPPED

        if not done.wait(timeout=timeout):
            with self._hook_timeout_lock:
                self._hook_timeout_suppressed_until[suppression_key] = (
                    time.monotonic() + self._hook_timeout_suppression_seconds
                )
                if self._hook_running_callbacks.get(gate_key) is token:
                    self._hook_abandoned.setdefault(suppression_key, set()).add(gate_key)
            event = self._record_hook_callback_telemetry_safely(
                outcome="timed_out",
                hook_name=hook_name,
                callback=cb,
                kwargs=kwargs,
                timeout=timeout,
                elapsed=time.monotonic() - started_at,
            )
            safe_hook, safe_callback = (
                (event["hook"], event["callback"])
                if event is not None
                else self._safe_hook_callback_labels(hook_name, cb)
            )
            logger.warning(
                "Hook '%s' callback %s timed out after %gs — skipping",
                safe_hook,
                safe_callback,
                timeout,
            )
            return _HOOK_SKIPPED
        if "exc" in failure:
            raise failure["exc"]
        return callback_outcome.get("value")

    def _subscribe_event(self, owner: str, event: str, callback: Callable) -> None:
        """Add an owner-tagged event subscription in registration order."""
        if not callable(callback):
            raise TypeError("Event subscriber callback must be callable")
        with self._event_lock:
            self._subscriptions.setdefault(event, []).append(_EventSubscription(owner, callback))

    def _remove_plugin_subscriptions(self, owner: str) -> int:
        """Remove every subscription owned by *owner*; return the count. Queued envelopes re-check
        membership per callback, so this also cancels already-snapshotted deliveries.

        TODO(#64229): when the central plugin ownership ledger / registration handles land, route this
        owner-tagged bookkeeping through that ledger so per-plugin unload cancels event subscriptions
        alongside every other registration surface. This method is the integration seam.
        """
        removed = 0
        with self._event_lock:
            for event in list(self._subscriptions):
                entries = self._subscriptions[event]
                retained = [entry for entry in entries if entry.owner != owner]
                removed += len(entries) - len(retained)
                if retained:
                    self._subscriptions[event] = retained
                else:
                    del self._subscriptions[event]
        return removed

    def _ensure_event_worker_locked(self) -> None:
        worker = self._event_worker
        if worker is not None and worker.is_alive():
            return
        worker = threading.Thread(
            target=self._event_worker_loop, args=(self._event_queue,), name="hermes-plugin-events",
            daemon=True,
        )
        self._event_worker = worker
        worker.start()

    def _event_worker_loop(self, dispatch_queue: queue.Queue[Any]) -> None:
        while True:
            item = dispatch_queue.get()
            try:
                if item is _EVENT_WORKER_STOP:
                    return
                self._deliver_event(item)
            finally:
                if item is not _EVENT_WORKER_STOP:
                    self._mark_event_done(item.generation)
                dispatch_queue.task_done()

    def _mark_event_done(self, generation: int) -> None:
        with self._event_idle:
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending > 0:
                self._event_pending_by_generation[generation] = pending - 1
            self._event_idle.notify_all()

    def _deliver_event(self, item: _QueuedPluginEvent) -> None:
        """Deliver one queued event on the host-owned worker thread."""
        from hermes_cli.plugins import resolve_plugin_command_result
        with self._event_lock:
            if item.generation != self._event_generation:
                return
        previous_depth = getattr(self._emit_depth, "value", 0)
        self._emit_depth.value = item.depth
        try:
            for subscription in item.subscriptions:
                with self._event_lock:
                    if item.generation != self._event_generation:
                        break
                    # Owner unload may have removed this entry after the event was queued.
                    if not any(cur is subscription for cur in self._subscriptions.get(item.event, [])):
                        continue
                callback = subscription.callback
                try:
                    # Fresh deep copy per subscriber: no callback can mutate what the next sees.
                    resolve_plugin_command_result(
                        item.context.copy().run(callback, **copy.deepcopy(item.payload)))
                except (Exception, SystemExit) as exc:
                    # A subscriber that fails identically on every emit is reported once (#111922).
                    self._report_hook_failure(item.event, callback, item.payload, exc, surface="Event")
        finally:
            self._emit_depth.value = previous_depth

    def _wait_for_event_dispatch(self, timeout: float = 2.0) -> bool:
        """Wait for the current event generation to become idle (test helper)."""
        with self._event_idle:
            generation = self._event_generation
            return self._event_idle.wait_for(
                lambda: self._event_pending_by_generation.get(generation, 0) == 0, timeout=timeout)

    def _dispatch_event(self, event: str, payload: Dict[str, Any]) -> int:
        """Queue *event* without blocking; return the subscriber count scheduled. Pending work is
        bounded per generation so a blocking subscriber costs one worker and later emits drop."""
        depth = getattr(self._emit_depth, "value", 0)
        if depth >= _EVENT_EMIT_DEPTH_CAP:
            logger.warning(
                "Event bus recursion cap (%d) exceeded while dispatching '%s' "
                "— dropping this emit to prevent an infinite loop", _EVENT_EMIT_DEPTH_CAP, event)
            return 0
        budget_msg = "Event bus pending budget (%d) exhausted while dispatching '%s' — dropping this emit"
        with self._event_lock:
            subscriptions = tuple(self._subscriptions.get(event, []))
            if not subscriptions:
                return 0
            generation = self._event_generation
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending >= _EVENT_PENDING_CAP:
                logger.warning(budget_msg, _EVENT_PENDING_CAP, event)
                return 0
            item = _QueuedPluginEvent(
                event=event, payload=dict(payload), subscriptions=subscriptions, depth=depth + 1,
                generation=generation, context=contextvars.copy_context())
            try:
                self._event_queue.put_nowait(item)
            except queue.Full:
                logger.warning(budget_msg, _EVENT_PENDING_CAP, event)
                return 0
            self._event_pending_by_generation[generation] = pending + 1
            self._ensure_event_worker_locked()
            return len(subscriptions)

    def has_hook(self, hook_name: str) -> bool:
        """Return True when at least one callback is registered for a hook."""
        return bool(self._hooks.get(hook_name))

    def iter_hook_callbacks(self, hook_name: str) -> tuple[Callable, ...]:
        """Return a stable snapshot of callbacks registered for a hook."""
        return tuple(self._hooks.get(hook_name, ()))

    def render_system_prompt_sections(
        self, session_info: Mapping[str, Any]
    ) -> List[RenderedPluginSystemPromptSection]:
        """Render all registered sections deterministically and fail open."""
        frozen_info = types.MappingProxyType(dict(session_info))
        rendered: List[RenderedPluginSystemPromptSection] = []
        total_chars = len(PLUGIN_SECTIONS_START) + len(PLUGIN_SECTIONS_END) + 2
        for _section_id, section in sorted(self._system_prompt_sections.items()):
            if len(rendered) >= MAX_SYSTEM_PROMPT_SECTIONS:
                logger.warning(
                    "Plugin system prompt section %s exceeded the section-count "
                    "budget (%d) and was skipped", section.id, MAX_SYSTEM_PROMPT_SECTIONS)
                continue
            text = self._render_prompt_section_text(section, frozen_info)
            if text is None:
                continue
            rendered_chars = len(format_system_prompt_section(section.id, text))
            if rendered:
                rendered_chars += 2  # canonical ``\n\n`` separator
            if total_chars + rendered_chars > MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS:
                logger.warning(
                    "Plugin system prompt section %s (%s) exceeded the aggregate "
                    "session budget (%d chars) and was skipped", section.id, section.plugin,
                    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS)
                continue
            rendered.append(
                RenderedPluginSystemPromptSection(
                    id=section.id, content=text, position=section.position, plugin=section.plugin))
            total_chars += rendered_chars
            logger.info(
                "Session plugin prompt section: id=%s plugin=%s position=%s chars=%d", section.id,
                section.plugin, section.position, len(text))
        return rendered

    @staticmethod
    def _render_prompt_section_text(
        section: PluginSystemPromptSection, frozen_info: Mapping[str, Any]
    ) -> Optional[str]:
        """Evaluate one section; return its stripped text or None (with a warning) when skipped."""
        def _skip(detail: str, *args: Any) -> None:
            logger.warning(
                "Plugin system prompt section %s (%s) " + detail, section.id, section.plugin, *args)

        try:
            value = section.content(frozen_info) if callable(section.content) else section.content
        except (Exception, SystemExit) as exc:
            _skip("raised and was skipped: %s", exc)
            return None
        if not isinstance(value, str):
            _skip("returned %s, not str; skipped", type(value).__name__)
            return None
        text = value.strip()
        if not text:
            return None
        if PLUGIN_SECTIONS_START in text or PLUGIN_SECTIONS_END in text:
            _skip("contained a reserved persistence marker and was skipped")
            return None
        if len(text) > section.max_chars:
            _skip("exceeded max_chars (%d > %d) and was skipped", len(text), section.max_chars)
            return None
        return text

    def has_middleware(self, kind: str) -> bool:
        """Return True when at least one callback is registered for middleware."""
        return bool(self._middleware.get(kind))

    def invoke_middleware(self, kind: str, **kwargs: Any) -> List[Any]:
        """Call middleware callbacks for *kind* (each isolated); return non-``None`` results."""
        results: List[Any] = []
        for cb in self._middleware.get(kind, []):
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except (Exception, SystemExit) as exc:
                # Runs once per tool call like a hook, so a mis-declared callback floods identically.
                self._report_hook_failure(kind, cb, kwargs, exc, surface="Middleware")
        return results
