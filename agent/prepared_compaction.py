"""Opt-in prepared compaction (``compression.prepare_ahead``, default off).

When the provider-reported prompt size comes within ``PREPARE_BAND_RATIO`` of
the compaction trigger, the turn loop (before a tool batch runs, and at turn
end) starts a background pass. The pass plans the same window
``ContextCompressor.compress`` would plan and runs the existing summariser on a
copy of the compressor and of that window's messages. The live transcript and
compressor are untouched, so the prompt cache is too.

At the next automatic threshold compaction ``compress`` takes the completed
candidate if it still fits: same head boundary, a boundary no later than the
current tail cut, an unchanged transcript up to that boundary, and at most the
tail token budget between that boundary and the current cut. On a hit the
candidate's summary replaces ``messages[start:candidate_end]``; everything
after the candidate boundary stays outside the summary, as tail, including
messages a fresh plan would have summarised (existing tail handling, such as
lean-mode tool-result stubs, still applies). On a miss compaction runs inline
exactly as without the flag.

When no completed candidate fits but a running pass would, compaction waits for
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
fresh. A failed pass backs off further passes but never touches the live
compressor's cooldown or failure state.

The fingerprint ignores what the no-LLM prune rewrites (tool result bodies,
tool call arguments, image parts), so a prune between pass and splice does not
invalidate a summary written from a superset of the same information.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Start passes once the real prompt count is within this fraction of the
# compaction trigger (a band below ``threshold_tokens``).
PREPARE_BAND_RATIO = 0.12
# A completed candidate that still fits is only extended once this many rough
# tokens sit between its boundary and the current cut; until then it keeps
# splicing and the summariser is not called once per chatty turn.
MIN_EXTENSION_DELTA_TOKENS = 8_000
# Minimum pause after a failed pass (the summariser's own cooldown applies when longer).
_FAILED_PASS_BACKOFF_SECONDS = 60.0
# Background passes running across the process (a cancelled pass frees its slot as it unwinds; its
# provider stream closes on the aux cancel path, normally within one chunk).
_slots = threading.BoundedSemaphore(2)


# How often a compaction waiting on a running pass re-checks cancellation and the deadline.
_WAIT_POLL_SECONDS = 0.5


@dataclass(frozen=True)
class _Window:
    compress_start: int
    compress_end: int
    fingerprint: str


@dataclass(frozen=True)
class PreparedSummary(_Window):
    summary: str  # prefixed handoff text, exactly as ``_generate_summary`` returns it
    body: str  # bare body the next compaction extends iteratively
    prompt_tokens: int  # provider-reported prompt size when this candidate was last planned or checked


class _PendingPass:
    """The one running pass allowed to publish: its planned window, completion and cancel events, and
    the progress and prompt-reading hooks of a compaction waiting on it."""

    def __init__(self, window: _Window, prompt_tokens: int) -> None:
        self.window = window
        self.prompt_tokens = prompt_tokens
        self.done = threading.Event()
        # Set once nothing can use the pass any more; its summary stream is closed (the aux hard-cancel path).
        self.cancelled = threading.Event()
        self.on_progress: Optional[Callable[[], None]] = None
        self.on_reading: Optional[Callable[[float], None]] = None

    def touch(self) -> None:
        hook = self.on_progress
        if hook is not None:
            hook()

    def read(self, seconds: float) -> None:
        hook = self.on_reading
        if hook is not None:
            hook(seconds)


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else part["text"]
            for part in content
            if isinstance(part, str)
            or (isinstance(part, dict) and part.get("type") in (None, "text") and isinstance(part.get("text"), str))
        )
    return str(content)


def _message_identity(msg: Any) -> str:
    """Role, text, tool call ids/names and tool result ids; nothing the prune pass rewrites.
    Text is normalised the way a SessionDB round trip normalises it."""
    if not isinstance(msg, dict):
        return "?"
    from agent.memory_manager import sanitize_context

    role = str(msg.get("role") or "")
    if role == "tool":
        return f"tool\x1f{msg.get('tool_call_id') or ''}"
    parts = [role, sanitize_context(_text_of(msg.get("content"))).strip()]
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            parts.append(f"{tc.get('id', '')}:{fn.get('name', '') if isinstance(fn, dict) else ''}")
        else:
            parts.append(f"{getattr(tc, 'id', '')}:{getattr(getattr(tc, 'function', None), 'name', '')}")
    return "\x1f".join(parts)


def fingerprint(messages: list[dict[str, Any]], session_id: str, start: int, end: int) -> str:
    """Identity of ``messages[:end]`` for a window ``[start, end)`` of ``session_id``."""
    digest = hashlib.sha256(f"{session_id}\x1f{start}:{end}".encode("utf-8", errors="replace"))
    for msg in messages[:end]:
        digest.update(_message_identity(msg).encode("utf-8", errors="replace"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _mismatch(
    entry: _Window, messages: list[dict[str, Any]], session_id: str, start: int, end: int,
) -> Optional[str]:
    """Why ``entry`` does not fit the planned window ``[start, end)`` of ``messages``, or None."""
    if entry.compress_start != start:
        return f"head moved ({entry.compress_start} -> {start})"
    if entry.compress_end > end:
        return f"prepared boundary {entry.compress_end} is inside the protected tail (cut {end})"
    if entry.compress_end > len(messages) or fingerprint(messages, session_id, start, entry.compress_end) != entry.fingerprint:
        return "transcript changed"
    return None


def _worker_copy(compressor: Any) -> Any:
    """A compressor copy whose writes stay off the live one: no durable row, no attempt
    telemetry, no host cancel consult (those belong to a foreground attempt)."""
    worker = copy.copy(compressor)
    worker._session_db = None
    worker._active_compression_telemetry = worker._last_compression_telemetry = None
    worker._compression_cancelled_check = None
    return worker


class PreparedCompaction:
    """Prepared-summary state owned by one ``ContextCompressor``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: Optional[PreparedSummary] = None
        self._pending: Optional[_PendingPass] = None  # the one pass allowed to publish
        self._retry_after = 0.0  # monotonic; set by a failed pass

    def discard(self, reason: str) -> None:
        """Drop the candidate and cancel a running pass (it can no longer publish)."""
        with self._lock:
            entry, pending, self._entry, self._pending = self._entry, self._pending, None, None
        if pending is not None:
            pending.cancelled.set()
        if entry is not None:
            logger.info(
                "prepared compaction: dropped window=%d-%d reason=%s", entry.compress_start, entry.compress_end, reason,
            )

    def take(
        self, messages: list[dict[str, Any]], session_id: str, start: int, end: int, *,
        eligible: bool, max_gap_tokens: int,
    ) -> Optional[PreparedSummary]:
        """The completed candidate when it fits ``compress()``'s planned window, else what a running pass
        whose window fits publishes while this compaction waits for it, else None. A window whose boundary
        trails the cut by more than ``max_gap_tokens`` is stale. Either way the state is cleared and a
        running pass is cancelled: this compaction consumes or supersedes it."""
        with self._lock:
            entry, pending = self._entry, self._pending
            self._entry = None
        reason = None if eligible else "fresh summary required"
        if reason is None and entry is not None:
            reason = self._misfit(entry, messages, session_id, start, end, max_gap_tokens)
        try:
            if eligible and (entry is None or reason is not None):
                if pending is not None and self._misfit(pending.window, messages, session_id, start, end, max_gap_tokens) is None:
                    entry, reason = self._await(pending)
                elif entry is None:
                    reason = "running pass does not fit" if pending is not None else "no candidate"
        finally:
            with self._lock:
                if self._pending is pending:
                    self._pending = None
                # A pass that published after the snapshot above planned on the pre-compaction transcript.
                self._entry = None
            if pending is not None:
                pending.cancelled.set()  # done already, or superseded: stop paying for it
        if reason is not None or entry is None:
            if entry is not None or pending is not None:
                logger.info("prepared compaction: splice miss reason=%s", reason)
            return None
        logger.info(
            "prepared compaction: splice hit window=%d-%d (fresh cut %d)", entry.compress_start, entry.compress_end, end,
        )
        return entry

    @staticmethod
    def _misfit(
        window: _Window, messages: list[dict[str, Any]], session_id: str, start: int, end: int, max_gap_tokens: int,
    ) -> Optional[str]:
        """Why *window* cannot splice into the planned window ``[start, end)``, or None."""
        reason = _mismatch(window, messages, session_id, start, end)
        if reason is None:
            from agent.model_metadata import estimate_messages_tokens_rough

            gap = estimate_messages_tokens_rough(messages[window.compress_end:end])
            if gap > max_gap_tokens:
                reason = f"stale candidate (~{gap} tokens past its boundary, budget {max_gap_tokens})"
        return reason

    def _await(self, pending: _PendingPass) -> tuple[Optional[PreparedSummary], Optional[str]]:
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
            "prepared compaction: waiting for the running pass window=%d-%d",
            pending.window.compress_start, pending.window.compress_end,
        )
        try:
            while not pending.done.wait(
                max(0.0, min(_WAIT_POLL_SECONDS, (_current_aux_stream_deadline() or fallback) - time.monotonic())),
            ):
                if pending.cancelled.is_set():
                    return None, "running pass discarded"
                if cancel_check is not None and _captured_aux_cancel_requested(cancel_check):
                    return None, "host cancelled the wait"
                if time.monotonic() >= (_current_aux_stream_deadline() or fallback):
                    return None, "running pass did not finish in time"
        finally:
            pending.on_progress = pending.on_reading = None
        if pending.cancelled.is_set():
            return None, "running pass discarded"
        with self._lock:
            entry, self._entry = self._entry, None
        if entry is None or entry.fingerprint != pending.window.fingerprint:
            return None, "running pass failed"
        return entry, None

    def _blocked(self, agent: Any, compressor: Any) -> bool:
        if not getattr(agent, "compression_enabled", False) or getattr(agent, "_persist_disabled", False):
            return True  # compression off, or a background-review fork of the live session
        if (
            getattr(agent, "api_mode", None) == "codex_app_server"
            or getattr(agent, "codex_responses_native_compaction", False)
            or getattr(compressor, "_micro_compact_enabled", False)
        ):
            return True  # another path owns compaction, or history is rewritten every turn
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
        """Start a pass when the next compaction is near; returns the started thread. Never raises."""
        token: Optional[_PendingPass] = None
        try:
            if self._blocked(agent, compressor):
                return None
            with self._lock:
                if self._pending is not None:
                    return None
                completed = self._entry
            tokens = compressor.last_prompt_tokens
            if completed is not None and 0 <= tokens - completed.prompt_tokens < MIN_EXTENSION_DELTA_TOKENS:
                # The prompt has barely grown since the candidate's boundary was the cut: keep it without
                # re-planning and re-fingerprinting the transcript before every tool batch.
                return None
            working, _, start, end = compressor._plan_compaction_window(messages)
            if start >= end:
                return None
            session_id = compressor._session_id
            extend = None
            if completed is not None and _mismatch(completed, working, session_id, start, end) is None:
                from agent.model_metadata import estimate_messages_tokens_rough

                delta = estimate_messages_tokens_rough(working[completed.compress_end:end])
                if delta < MIN_EXTENSION_DELTA_TOKENS:
                    logger.debug("prepared compaction: keeping window=%d-%d, delta ~%d tokens",
                                 completed.compress_start, completed.compress_end, delta)
                    with self._lock:  # measure the next skip from this check, not from the plan
                        if self._entry is completed:
                            self._entry = dataclasses.replace(completed, prompt_tokens=tokens)
                    return None
                extend = completed
            worker = _worker_copy(compressor)
            # Same handoff rehydration and window rows compress() uses (mutates only the worker).
            scan = worker._scan_window_handoffs(working, start, end, working[start:end])
            if scan.tail_start != end:
                return None  # a handoff inside the protected tail: leave it to compress()
            turns = scan.turns_to_summarize
            if extend is not None:
                worker._previous_summary = extend.body
                turns = working[extend.compress_end:end]
            if not turns:
                return None
            focus = worker._derive_auto_focus_topic(working)
            window_fingerprint = fingerprint(working, session_id, start, end)
            # Never queue threads that retain transcripts while every slot is busy.
            if not _slots.acquire(blocking=False):
                return None
            with self._lock:
                if self._pending is not None:
                    _slots.release()
                    return None
                token = self._pending = _PendingPass(_Window(start, end, window_fingerprint), tokens)
            from agent.memory_provider import spawn_context_thread

            # Profile and secret scope ride the contextvars into the pass.
            thread = spawn_context_thread(
                self._run_pass, name="compaction-prepare",
                args=(worker, copy.deepcopy(turns), focus, token, origin, extend),
            )
            thread.start()
            return thread
        except Exception:
            if token is not None:
                with self._lock:
                    if self._pending is token:
                        self._pending = None
                token.done.set()
                _slots.release()
            logger.debug("prepared compaction: maybe_prepare failed", exc_info=True)
            return None

    def _run_pass(
        self, worker: Any, turns: list[dict[str, Any]], focus: Optional[str], token: _PendingPass, origin: str,
        extend: Optional[PreparedSummary],
    ) -> None:
        from agent.auxiliary_client import AuxiliaryExplicitCancellation, aux_interrupt_protection, aux_progress_hook
        from agent.local_prompt_progress import prompt_reading_hook
        from agent.thread_scoped_output import thread_scoped_silence

        start, end, window_fingerprint = token.window.compress_start, token.window.compress_end, token.window.fingerprint
        entry: Optional[PreparedSummary] = None
        started = time.monotonic()
        try:
            logger.info(
                "prepared compaction: pass started origin=%s window=%d-%d turns=%d extends=%s",
                origin, start, end, len(turns), extend is not None,
            )
            with (
                thread_scoped_silence(), aux_interrupt_protection(cancel_event=token.cancelled),
                aux_progress_hook(token.touch), prompt_reading_hook(token.read),
            ):
                summary = worker._generate_summary(turns, focus_topic=focus)
            if token.cancelled.is_set():
                logger.info("prepared compaction: pass cancelled after %.1fs (superseded)", time.monotonic() - started)
            elif summary:
                entry = PreparedSummary(
                    start, end, window_fingerprint, summary, worker._previous_summary, token.prompt_tokens,
                )
                logger.info("prepared compaction: pass completed window=%d-%d in %.1fs",
                            start, end, time.monotonic() - started)
            else:
                logger.warning("prepared compaction: pass failed after %.1fs: %s", time.monotonic() - started,
                               worker._last_summary_error or "summariser returned nothing")
        except AuxiliaryExplicitCancellation:
            # The closed stream raises a BaseException; it must not reach the thread's excepthook (stderr).
            logger.info("prepared compaction: pass cancelled after %.1fs (superseded)", time.monotonic() - started)
        except Exception:
            logger.warning("prepared compaction: pass crashed", exc_info=True)
        finally:
            with self._lock:
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


def maybe_prepare(agent: Any, messages: list[dict[str, Any]], *, origin: str) -> Optional[threading.Thread]:
    """Turn-loop hook; a no-op unless ``compression.prepare_ahead`` gave the compressor state."""
    compressor = getattr(agent, "context_compressor", None)
    state = getattr(compressor, "prepared_compaction", None)
    return state.maybe_prepare(agent, compressor, messages, origin) if state is not None else None
