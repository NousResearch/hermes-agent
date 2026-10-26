"""Opt-in prepared compaction (``compression.prepare_ahead``, default off).

When the provider-reported prompt size comes within ``PREPARE_BAND_RATIO`` of
the compaction trigger, the turn loop (before a tool batch runs, and at turn
end) starts ONE background pass. The pass plans the same window
``ContextCompressor.compress`` would plan and runs the existing summariser on a
copy of the compressor and of that window's messages. The live transcript and
compressor are untouched, so the prompt cache is too.

At the next automatic threshold compaction ``compress`` takes the completed
summary if the messages it covers are still the same messages: same session,
same head boundary, a boundary no later than the current tail cut, and the same
message structure (roles, tool call and result ids) up to that boundary, each
message vouched for by its durable ``message_uid``. The text of a persisted
message is not compared: plugins rewrite text in place, and a summary written
before such a rewrite still summarises the same messages. Only a message that
has no uid on one side (caller-supplied history that is never persisted) is
compared by text, since nothing else can tell it from a different message. On a hit the summary replaces
``messages[start:candidate_end]`` and every message after the candidate
boundary, however many arrived since the pass started, stays verbatim in the
tail. On a miss compaction runs inline exactly as without the flag.

When no completed summary exists but a running pass fits, compaction waits for
that pass instead of starting the same summary over: the pass's streamed
progress and local prompt reading feed the waiting compaction's hooks, so the
host's idle watchdog and deadline extension see a live summary exactly as they
would inline, and the wait ends at the host's deadline (the compression aux
timeout without one), on cancellation, or when the pass is discarded. A pass
that compaction or a session boundary supersedes is cancelled: its summary
stream is closed instead of being billed to completion for nothing.

State is per compressor and in memory only; every session boundary (reset,
end, or a rebind to another session id) discards it. Manual, focus,
overflow-recovery and memory-provider-context compactions always summarise
fresh; the first compaction that needs provider context stops further passes
for the agent, since every later one would need it too. A failed pass backs off
further passes but never touches the live compressor's cooldown or failure
state; a pass that recovered on the main model hands that fallback to the live
compressor on a splice, as an inline summary would.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Start a pass once the real prompt count is within this fraction of the
# compaction trigger (a band below ``threshold_tokens``).
PREPARE_BAND_RATIO = 0.12
# Minimum pause after a failed pass (the summariser's own cooldown applies when longer).
_FAILED_PASS_BACKOFF_SECONDS = 60.0
# Background passes running across the process (a cancelled pass frees its slot as it unwinds; its
# provider stream closes on the aux cancel path, normally within one chunk).
_slots = threading.BoundedSemaphore(2)
# How often a compaction waiting on a running pass re-checks cancellation and the deadline.
_WAIT_POLL_SECONDS = 0.5
# Compressor state a summary call leaves behind besides the summary: the main-model fallback after a
# failed summary model and the warning it raises (the next compaction tries the summary model again,
# inline or prepared alike). A splice adopts the pass's.
_SUMMARY_CALL_STATE = (
    "summary_model", "_last_aux_model_failure_error", "_last_aux_model_failure_model", "_last_aux_resolved_model",
)
# Attempt telemetry the summary call fills; a splice copies the pass's values onto the attempt.
_SUMMARY_CALL_TELEMETRY = (
    "aux_prompt_tokens", "aux_output_reservation", "aux_provider", "aux_model", "effective_aux_context", "fit_margin",
    "aux_call_duration_ms", "queue_wait_ms", "prompt_build_ms", "time_to_first_progress_ms", "summary_generation_ms",
    "summary_input_chars", "summary_input_sampled_chars", "summary_input_omitted_chars", "summary_input_record_count",
    "summary_input_sampled_record_count", "summary_input_elided_record_count", "fallback_used", "failure_class",
)

# Per message: (structure, durable uid or None, text digest).
_Identity = tuple[str, Optional[str], str]
# Why a compaction stopped waiting on a running pass; the pass then records it as its cancel reason.
_WAIT_CANCEL_REASONS = frozenset({"wait_timeout", "host_cancelled"})


class _Miss(str):
    """A splice-miss reason: the log text, plus the closed code opt-in shared metrics count."""

    code: str

    def __new__(cls, code: str, text: str) -> _Miss:
        miss = super().__new__(cls, text)
        miss.code = code
        return miss


def _record(event: str, **dims: Any) -> None:
    """Count one event in opt-in shared metrics (a no-op when off; never raises)."""
    from hermes_cli.observability.shared_metrics_prepared_compaction import record_prepared_compaction

    record_prepared_compaction(event, **dims)


@dataclass(frozen=True)
class _Window:
    session_id: str
    compress_start: int
    compress_end: int
    identities: tuple[_Identity, ...]  # one per message of ``messages[:compress_end]``


@dataclass(frozen=True)
class PreparedSummary(_Window):
    summary: str  # prefixed handoff text, exactly as ``_generate_summary`` returns it
    body: str  # bare body the next compaction extends iteratively
    call_state: dict[str, Any] = field(default_factory=dict, compare=False)  # _SUMMARY_CALL_STATE after the pass
    telemetry: dict[str, Any] = field(default_factory=dict, compare=False)  # _SUMMARY_CALL_TELEMETRY of the pass
    waited: Optional[float] = field(default=None, compare=False)  # seconds the taking compaction waited on the pass


class _PendingPass:
    """The one running pass allowed to publish: its planned window, completion and cancel events, and
    the progress and prompt-reading hooks of a compaction waiting on it."""

    def __init__(self, window: _Window) -> None:
        self.window = window
        self.done = threading.Event()
        # Set once nothing can use the pass any more; its summary stream is closed (the aux hard-cancel path).
        self.cancelled = threading.Event()
        self.cancel_reason = "superseded"  # set by whoever cancels first; the pass records it
        self.on_progress: Optional[Callable[[], None]] = None
        self.on_reading: Optional[Callable[[float], None]] = None

    def cancel(self, reason: str) -> None:
        if not self.cancelled.is_set():
            self.cancel_reason = reason
        self.cancelled.set()

    def touch(self) -> None:
        hook = self.on_progress
        if hook is not None:
            hook()

    def read(self, seconds: float) -> None:
        hook = self.on_reading
        if hook is not None:
            hook(seconds)


def _message_identity(msg: Any) -> _Identity:
    """Role, tool call ids/names and tool result ids; the durable ``message_uid`` when the message has
    been persisted; and a digest of its text for messages no uid vouches for. Tool arguments and result
    bodies are left out: the no-LLM prune rewrites them in place."""
    if not isinstance(msg, dict):
        return "?", None, ""
    from agent.memory_manager import sanitize_context
    from agent.message_content import flatten_message_text
    from agent.message_metadata import message_uid_or_none

    role = str(msg.get("role") or "")
    uid = message_uid_or_none(msg)
    if role == "tool":
        return f"tool\x1f{msg.get('tool_call_id') or ''}", uid, ""
    # Normalised the way a SessionDB round trip normalises it.
    text = sanitize_context(flatten_message_text(msg.get("content"))).strip()
    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).hexdigest()
    parts = [role]
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            parts.append(f"{tc.get('id', '')}:{fn.get('name', '') if isinstance(fn, dict) else ''}")
        else:
            parts.append(f"{getattr(tc, 'id', '')}:{getattr(getattr(tc, 'function', None), 'name', '')}")
    return "\x1f".join(parts), uid, digest


def prefix_identities(messages: list[dict[str, Any]], end: int) -> tuple[_Identity, ...]:
    """Identity of each message in ``messages[:end]``."""
    return tuple(_message_identity(msg) for msg in messages[:end])


def _mismatch(
    window: _Window, messages: list[dict[str, Any]], session_id: str, start: int, end: int,
) -> Optional[_Miss]:
    """Why *window* cannot splice into the planned window ``[start, end)`` of ``messages``, or None."""
    if window.session_id != session_id:
        return _Miss("other_session", "another session")
    if window.compress_start != start:
        return _Miss("head_moved", f"head moved ({window.compress_start} -> {start})")
    if window.compress_end > end:
        return _Miss("inside_tail", f"prepared boundary {window.compress_end} is inside the protected tail (cut {end})")
    if window.compress_end > len(messages):
        return _Miss("transcript_shorter", "transcript shorter than the prepared window")
    for idx, ((structure, uid, digest), msg) in enumerate(zip(window.identities, messages)):
        now_structure, now_uid, now_digest = _message_identity(msg)
        if now_structure != structure:
            return _Miss("shape_changed", f"message {idx} changed shape")
        if uid and now_uid:
            if uid != now_uid:
                return _Miss("rewound", f"message {idx} is a different message (rewound or branched)")
        elif digest != now_digest:
            return _Miss("text_changed", f"message {idx} changed and has no durable id to vouch for it")
    return None


def _worker_copy(compressor: Any) -> Any:
    """A compressor copy whose writes stay off the live one: no durable row, private attempt
    telemetry, no host cancel consult (those belong to a foreground attempt), and the per-call
    fallback fields cleared as a compaction attempt clears them."""
    worker = copy.copy(compressor)
    worker._session_db = None
    worker._compression_cancelled_check = None
    worker._last_aux_model_failure_error = worker._last_aux_model_failure_model = None
    worker._begin_compression_telemetry(current_tokens=None)
    return worker


def _blocked_for_platform(agent: Any, origin: str) -> bool:
    """A turn-end pass in a session that ends with its turn (a cron run, a delegated child, a one-shot) is never
    used."""
    if origin != "turn-end":
        return False
    if getattr(agent, "platform", None) in ("cron", "subagent") or getattr(agent, "_delegate_depth", 0) > 0:
        return True
    from agent.oneshot_footprint import agent_is_oneshot

    return agent_is_oneshot(agent)  # ``chat -q``: the agent is released right after the reply


class PreparedCompaction:
    """Prepared-summary state owned by one ``ContextCompressor``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: Optional[PreparedSummary] = None
        self._pending: Optional[_PendingPass] = None  # the one pass allowed to publish
        self._retry_after = 0.0  # monotonic; set by a failed pass
        # Set by the first compaction that needed memory-provider context; such compactions summarise fresh.
        self._needs_provider_context = False
        self._busy_window: Optional[tuple[str, int, int]] = None  # last window the slot cap refused (counted once)
        self._taken: Optional[PreparedSummary] = None  # handed to compress(), not yet spliced by adopt()

    def _count_unspliced(self) -> None:
        """A summary take() handed out that adopt() never spliced (the compaction aborted or no-opped after
        taking it) is counted once, at the next touch, so every summary ends as used or discarded."""
        with self._lock:
            taken, self._taken = self._taken, None
        if taken is not None:
            _record("discarded", reason="compaction_aborted", wait_seconds=taken.waited)

    def discard(self, reason: str, code: str = "session_boundary") -> None:
        """Drop the candidate and cancel a running pass (it can no longer publish); ``code`` is the closed
        reason opt-in shared metrics count."""
        self._count_unspliced()
        with self._lock:
            entry, pending, self._entry, self._pending = self._entry, self._pending, None, None
            if pending is not None:
                pending.cancel(code)  # under the lock: the pass reads its cancel reason after it
        if entry is not None or pending is not None:
            _record("discarded", reason=code)
        dropped = entry or (pending.window if pending is not None else None)
        if dropped is not None:
            logger.info(
                "prepared compaction: dropped session=%s window=%d-%d reason=%s%s", dropped.session_id,
                dropped.compress_start, dropped.compress_end, reason, "" if entry is not None else " (pass running)",
            )

    def stop_for_provider_context(self) -> None:
        """A memory provider supplied summary context: stop preparing summaries that would be discarded."""
        if not self._needs_provider_context:
            self._needs_provider_context = True
            logger.info("prepared compaction: off for this agent (memory provider supplies compaction context)")
            _record("auto_disabled", reason="provider_context")

    def take(
        self, messages: list[dict[str, Any]], session_id: str, start: int, end: int, *, eligible: bool,
    ) -> Optional[PreparedSummary]:
        """The completed candidate when it fits ``compress()``'s planned window, else what a running pass
        whose window fits publishes while this compaction waits for it, else None. Either way the state is
        cleared and a running pass is cancelled: this compaction consumes or supersedes it."""
        self._count_unspliced()
        with self._lock:
            entry, pending = self._entry, self._pending
            self._entry = None
        reason = None if eligible else _Miss(
            "provider_context" if self._needs_provider_context else "fresh_required", "fresh summary required",
        )
        if reason is None and entry is not None:
            reason = _mismatch(entry, messages, session_id, start, end)
        waited = None
        try:
            if eligible and (entry is None or reason is not None):
                if pending is not None and _mismatch(pending.window, messages, session_id, start, end) is None:
                    wait_started = time.monotonic()
                    entry, reason = self._await(pending)
                    waited = time.monotonic() - wait_started
                elif entry is None:
                    reason = _Miss(
                        "running_not_fit", "running pass does not fit",
                    ) if pending is not None else _Miss("none", "no candidate")
        finally:
            with self._lock:
                if self._pending is pending:
                    self._pending = None
                # A pass that published after the snapshot above planned on the pre-compaction transcript.
                self._entry = None
                if pending is not None:
                    # Done already, or superseded: stop paying for it.
                    code = getattr(reason, "code", None)
                    pending.cancel(code if code in _WAIT_CANCEL_REASONS else "superseded")
        if reason is not None or entry is None:
            if entry is not None or pending is not None:
                logger.info("prepared compaction: splice miss session=%s reason=%s", session_id, reason)
                if reason.code != "discarded_while_waiting":  # the discard() that woke the wait counted it
                    _record("discarded", reason=reason.code, wait_seconds=waited)
            return None
        logger.info(
            "prepared compaction: splice hit session=%s window=%d-%d (fresh cut %d)", session_id, entry.compress_start,
            entry.compress_end, end,
        )
        taken = replace(entry, waited=waited)  # counted as used once adopt() splices it
        with self._lock:
            self._taken = taken
        return taken

    def _await(self, pending: _PendingPass) -> tuple[Optional[PreparedSummary], Optional[_Miss]]:
        """Wait on the compacting thread for *pending*: what it published, or None and why (the host's
        deadline, or the compression aux timeout without one; a host cancel; a discard; a failed pass).
        Meanwhile the pass's streamed progress and local prompt reading feed this thread's hooks, so the
        host's idle watchdog and deadline extension see a live summary exactly as they would inline; the
        deadline is re-read on every poll because prompt reading pushes it back."""
        from agent.auxiliary_client import (
            _aux_progress, _capture_aux_cancel_check, _captured_aux_cancel_requested, _current_aux_stream_deadline,
            _effective_aux_timeout,
        )
        from agent.local_prompt_progress import current_prompt_reading_hook

        # Hooks first: the pass may stream while the rest of the setup runs (the timeout loads config).
        pending.on_progress = getattr(_aux_progress, "hook", None)
        pending.on_reading = current_prompt_reading_hook()
        cancel_check = _capture_aux_cancel_check()
        fallback = time.monotonic() + float(_effective_aux_timeout("compression", None))
        logger.info(
            "prepared compaction: waiting for the running pass session=%s window=%d-%d",
            pending.window.session_id, pending.window.compress_start, pending.window.compress_end,
        )
        try:
            while not pending.done.wait(
                max(0.0, min(_WAIT_POLL_SECONDS, (_current_aux_stream_deadline() or fallback) - time.monotonic())),
            ):
                if pending.cancelled.is_set():
                    return None, _Miss("discarded_while_waiting", "running pass discarded")
                if cancel_check is not None and _captured_aux_cancel_requested(cancel_check):
                    return None, _Miss("host_cancelled", "host cancelled the wait")
                if time.monotonic() >= (_current_aux_stream_deadline() or fallback):
                    return None, _Miss("wait_timeout", "running pass did not finish in time")
        finally:
            pending.on_progress = pending.on_reading = None
        with self._lock:  # a discard() between the pass publishing and this wake took the entry and cancelled
            entry, self._entry = self._entry, None
            discarded = pending.cancelled.is_set()
        if discarded:
            return None, _Miss("discarded_while_waiting", "running pass discarded")
        if entry is None or entry.identities != pending.window.identities:
            return None, _Miss("pass_failed_while_waiting", "running pass failed")
        return entry, None

    def _blocked(self, agent: Any, compressor: Any, origin: str) -> bool:
        if not getattr(agent, "compression_enabled", False) or getattr(agent, "_persist_disabled", False):
            return True  # compression off, or a background-review fork of the live session
        if (
            getattr(agent, "api_mode", None) == "codex_app_server"
            or getattr(agent, "codex_responses_native_compaction", False)
            or getattr(compressor, "_micro_compact_enabled", False)
        ):
            return True  # another path owns compaction, or history is rewritten every turn
        if self._needs_provider_context or _blocked_for_platform(agent, origin):
            return True
        if compressor.awaiting_real_usage_after_compression or compressor._fallback_compression_streak >= 2:
            return True  # no real usage yet, or the summary model is benched after repeated fallbacks
        tokens = compressor.last_prompt_tokens
        band_floor = int(compressor.threshold_tokens * (1 - PREPARE_BAND_RATIO))
        if tokens <= 0 or compressor.threshold_tokens <= 0 or tokens < band_floor:
            return True
        now = time.monotonic()
        return now < compressor._summary_failure_cooldown_until or now < self._retry_after

    def maybe_prepare(
        self, agent: Any, compressor: Any, messages: list[dict[str, Any]], origin: str,
    ) -> Optional[threading.Thread]:
        """Start the pass when the next compaction is near and none ran yet; returns the started thread.
        Never raises."""
        token: Optional[_PendingPass] = None
        try:
            self._count_unspliced()
            if self._blocked(agent, compressor, origin):
                return None
            with self._lock:
                if self._pending is not None or self._entry is not None:
                    return None  # one summary per compaction: later messages join the tail at the splice
            working, _, start, end = compressor._plan_compaction_window(messages)
            if start >= end:
                return None
            worker = _worker_copy(compressor)
            # Same handoff rehydration and window rows compress() uses (mutates only the worker).
            scan = worker._scan_window_handoffs(working, start, end, working[start:end])
            if scan.tail_start != end:
                return None  # a handoff inside the protected tail: leave it to compress()
            turns = scan.turns_to_summarize
            if not turns:
                return None
            focus = worker._derive_auto_focus_topic(working)
            window = _Window(compressor._session_id, start, end, prefix_identities(working, end))
            # Never queue threads that retain transcripts while every slot is busy.
            if not _slots.acquire(blocking=False):
                if self._busy_window != (window.session_id, start, end):  # count refused passes, not tool batches
                    self._busy_window = (window.session_id, start, end)
                    _record("skipped_busy", origin=origin)
                return None
            with self._lock:
                if self._pending is not None or self._entry is not None:
                    _slots.release()
                    return None
                token = self._pending = _PendingPass(window)
            from agent.memory_provider import spawn_context_thread

            # Profile and secret scope ride the contextvars into the pass.
            thread = spawn_context_thread(
                self._run_pass, name="compaction-prepare", args=(worker, copy.deepcopy(turns), focus, token, origin),
            )
            thread.start()
            _record("pass_started", origin=origin)
            return thread
        except Exception:
            if token is not None:
                with self._lock:
                    if self._pending is token:
                        self._pending = None
                token.done.set()
                _slots.release()
            logger.debug("prepared compaction: maybe_prepare failed", exc_info=True)
            _record("internal_error", origin=origin, failure_class="exception")
            return None

    def _run_pass(
        self, worker: Any, turns: list[dict[str, Any]], focus: Optional[str], token: _PendingPass, origin: str,
    ) -> None:
        from agent.auxiliary_client import AuxiliaryExplicitCancellation, aux_interrupt_protection, aux_progress_hook
        from agent.local_prompt_progress import prompt_reading_hook
        from agent.thread_scoped_output import thread_scoped_silence

        window = token.window
        entry: Optional[PreparedSummary] = None
        event, failure_class = "pass_cancelled", None  # a published summary records pass_completed instead
        started = time.monotonic()
        try:
            logger.info(
                "prepared compaction: pass started session=%s origin=%s window=%d-%d turns=%d",
                window.session_id, origin, window.compress_start, window.compress_end, len(turns),
            )
            with (
                thread_scoped_silence(), aux_interrupt_protection(cancel_event=token.cancelled),
                aux_progress_hook(token.touch), prompt_reading_hook(token.read),
            ):
                summary = worker._generate_summary(turns, focus_topic=focus)
            if token.cancelled.is_set():
                logger.info("prepared compaction: pass cancelled session=%s after %.1fs (%s)", window.session_id,
                            time.monotonic() - started, token.cancel_reason)
            elif summary:
                telemetry = worker._active_compression_telemetry or {}
                entry = PreparedSummary(
                    window.session_id, window.compress_start, window.compress_end, window.identities,
                    summary, worker._previous_summary,
                    call_state={name: getattr(worker, name, None) for name in _SUMMARY_CALL_STATE},
                    telemetry={key: telemetry.get(key) for key in _SUMMARY_CALL_TELEMETRY},
                )
                logger.info("prepared compaction: pass completed session=%s window=%d-%d in %.1fs",
                            window.session_id, window.compress_start, window.compress_end, time.monotonic() - started)
            else:
                logger.warning("prepared compaction: pass failed session=%s after %.1fs: %s", window.session_id,
                               time.monotonic() - started,
                               worker._last_summary_error or "summariser returned nothing")
                event, failure_class = "pass_failed", _pass_failure_class(worker)
        except AuxiliaryExplicitCancellation:
            # The closed stream raises a BaseException; it must not reach the thread's excepthook (stderr).
            logger.info("prepared compaction: pass cancelled session=%s after %.1fs (%s)", window.session_id,
                            time.monotonic() - started, token.cancel_reason)
        except Exception:
            logger.warning("prepared compaction: pass crashed session=%s", window.session_id, exc_info=True)
            event, failure_class = "internal_error", "exception"
        finally:
            with self._lock:
                if entry is not None and self._pending is token:
                    event = "pass_completed"
                if self._pending is token:
                    self._pending = None
                    if entry is not None:
                        self._entry = entry
                    else:
                        self._retry_after = max(
                            worker._summary_failure_cooldown_until, time.monotonic() + _FAILED_PASS_BACKOFF_SECONDS,
                        )
            token.done.set()
            _slots.release()
            _record(
                event, origin=origin, reason=token.cancel_reason if event == "pass_cancelled" else None,
                failure_class=failure_class,
            )


def _pass_failure_class(worker: Any) -> str:
    """The class an inline attempt logs for this failed summary call: a terminal failure first."""
    from agent.context_compressor import _TERMINAL_SUMMARY_FAILURES

    terminal = next((cls for flag, cls, _ in _TERMINAL_SUMMARY_FAILURES if getattr(worker, flag, False)), None)
    return terminal or (worker._active_compression_telemetry or {}).get("failure_class") or "summary_generation_failed"


def release_for_agent(agent: Any) -> None:
    """The agent is being released (cache eviction, rewind, close): no compaction of it can splice the summary,
    so stop paying for a running pass."""
    prepared = getattr(getattr(agent, "context_compressor", None), "prepared_compaction", None)
    if prepared is not None:
        prepared.discard("agent released", "agent_released")


def adopt(compressor: Any, prepared: PreparedSummary) -> None:
    """Leave the live compressor as the inline summary call would have: the summary is the iterative base,
    failure state cleared, the pass's main-model fallback (if any) carried over, its call in the telemetry."""
    state = getattr(compressor, "prepared_compaction", None)
    if state is not None:
        with state._lock:
            if state._taken is prepared:
                state._taken = None
    compressor._record_summary_success(prepared.body)
    for name, value in prepared.call_state.items():
        setattr(compressor, name, value)
    telemetry = getattr(compressor, "_active_compression_telemetry", None)
    if isinstance(telemetry, dict):
        telemetry.update({key: value for key, value in prepared.telemetry.items() if value is not None})
        telemetry["prepared_summary"] = True
    _record("used", reason="aux_fallback_main" if prepared.telemetry.get("fallback_used") else None,
            wait_seconds=prepared.waited)


def maybe_prepare(agent: Any, messages: list[dict[str, Any]], *, origin: str) -> Optional[threading.Thread]:
    """Turn-loop hook; a no-op unless ``compression.prepare_ahead`` gave the compressor state."""
    compressor = getattr(agent, "context_compressor", None)
    state = getattr(compressor, "prepared_compaction", None)
    return state.maybe_prepare(agent, compressor, messages, origin) if state is not None else None
