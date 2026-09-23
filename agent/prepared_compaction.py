"""Opt-in prepared compaction (``compression.prepare_ahead``, default off).

When the provider-reported prompt size comes within ``PREPARE_BAND_RATIO`` of
the context window below the compaction trigger, the turn loop (before a tool
batch runs, and at turn end) starts a background pass. The pass plans the same
window ``ContextCompressor.compress`` would plan and runs the existing
summariser on a copy of the compressor and of that window's messages. The
live transcript and compressor are untouched, so the prompt cache is too.

At the next automatic threshold compaction ``compress`` takes the completed
candidate if it still fits: same head boundary, a boundary no later than the
current tail cut, and an unchanged transcript up to that boundary. On a hit the
candidate's summary replaces ``messages[start:candidate_end]``; everything
after the candidate boundary stays outside the summary, as tail, including
messages a fresh plan would have summarised (existing tail handling, such as
lean-mode tool-result stubs, still applies). On a miss, or when the pass has not finished,
compaction runs inline exactly as without the flag. A running pass is never
waited on (the wait would sit inside the host's compression idle timeout) and
loses the right to publish once compaction runs.

State is per compressor and in memory only; every session boundary discards
it. Manual, focus, overflow-recovery and memory-provider-context compactions
always summarise fresh. A failed pass backs off further passes but never
touches the live compressor's cooldown or failure state.

The fingerprint ignores what the no-LLM prune rewrites (tool result bodies,
tool call arguments, image parts), so a prune between pass and splice does not
invalidate a summary written from a superset of the same information.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Start passes once the real prompt count is within this fraction of the
# context window below the compaction trigger.
PREPARE_BAND_RATIO = 0.12
# A completed candidate that still fits is only extended once this many rough
# tokens sit between its boundary and the current cut; until then it keeps
# splicing and the summariser is not called once per chatty turn.
MIN_EXTENSION_DELTA_TOKENS = 8_000
# Minimum pause after a failed pass (the summariser's own cooldown applies when longer).
_FAILED_PASS_BACKOFF_SECONDS = 60.0
# Background summariser calls in flight across the process.
_slots = threading.BoundedSemaphore(2)


@dataclass(frozen=True)
class PreparedSummary:
    compress_start: int
    compress_end: int
    fingerprint: str
    summary: str  # prefixed handoff text, exactly as ``_generate_summary`` returns it
    body: str  # bare body the next compaction extends iteratively


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


def fingerprint(messages: List[Dict[str, Any]], session_id: str, start: int, end: int) -> str:
    """Identity of ``messages[:end]`` for a window ``[start, end)`` of ``session_id``."""
    digest = hashlib.sha256(f"{session_id}\x1f{start}:{end}".encode("utf-8", errors="replace"))
    for msg in messages[:end]:
        digest.update(_message_identity(msg).encode("utf-8", errors="replace"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def _mismatch(
    entry: PreparedSummary, messages: List[Dict[str, Any]], session_id: str, start: int, end: int,
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
        self._pending: Optional[object] = None  # token of the one pass allowed to publish
        self._retry_after = 0.0  # monotonic; set by a failed pass

    def discard(self, reason: str) -> None:
        """Drop the candidate; a running pass may finish but can no longer publish."""
        with self._lock:
            entry, self._entry, self._pending = self._entry, None, None
        if entry is not None:
            logger.info(
                "prepared compaction: dropped window=%d-%d reason=%s", entry.compress_start, entry.compress_end, reason,
            )

    def take(
        self, messages: List[Dict[str, Any]], session_id: str, start: int, end: int, *, eligible: bool,
    ) -> Optional[PreparedSummary]:
        """The completed candidate when it fits ``compress()``'s planned window, else None.
        Either way the state is cleared: this compaction consumes or supersedes it."""
        with self._lock:
            entry, running = self._entry, self._pending is not None
            self._entry = self._pending = None
        if not eligible:
            reason = "fresh summary required"
        elif entry is None:
            reason = "pass still running" if running else "no candidate"
        else:
            reason = _mismatch(entry, messages, session_id, start, end)
        if reason is not None:
            if entry is not None or running:
                logger.info("prepared compaction: splice miss reason=%s", reason)
            return None
        logger.info(
            "prepared compaction: splice hit window=%d-%d (fresh cut %d)", entry.compress_start, entry.compress_end, end,
        )
        return entry

    def _blocked(self, agent: Any, compressor: Any) -> bool:
        if not getattr(agent, "compression_enabled", False) or getattr(agent, "_persist_disabled", False):
            return True  # compression off, or a background-review fork of the live session
        if (
            getattr(agent, "api_mode", None) == "codex_app_server"
            or getattr(agent, "codex_responses_native_compaction", False)
            or getattr(compressor, "_micro_compact_enabled", False)
        ):
            return True  # another path owns compaction, or history is rewritten every turn
        if compressor.awaiting_real_usage_after_compression:
            return True
        tokens = compressor.last_prompt_tokens
        band_floor = compressor.threshold_tokens - int(compressor.context_length * PREPARE_BAND_RATIO)
        if tokens <= 0 or compressor.threshold_tokens <= 0 or tokens < band_floor:
            return True
        now = time.monotonic()
        return now < compressor._summary_failure_cooldown_until or now < self._retry_after

    def maybe_prepare(
        self, agent: Any, compressor: Any, messages: List[Dict[str, Any]], origin: str,
    ) -> Optional[threading.Thread]:
        """Start a pass when the next compaction is near; returns the started thread. Never raises."""
        token: Optional[object] = None
        try:
            if self._blocked(agent, compressor):
                return None
            with self._lock:
                if self._pending is not None:
                    return None
                completed = self._entry
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
                token = self._pending = object()
            from agent.memory_provider import spawn_context_thread

            # Profile and secret scope ride the contextvars into the pass.
            thread = spawn_context_thread(
                self._run_pass, name="compaction-prepare",
                args=(worker, copy.deepcopy(turns), focus, start, end, window_fingerprint, token, origin, extend),
            )
            thread.start()
            return thread
        except Exception:
            if token is not None:
                with self._lock:
                    if self._pending is token:
                        self._pending = None
                _slots.release()
            logger.debug("prepared compaction: maybe_prepare failed", exc_info=True)
            return None

    def _run_pass(
        self, worker: Any, turns: List[Dict[str, Any]], focus: Optional[str], start: int, end: int,
        window_fingerprint: str, token: object, origin: str, extend: Optional[PreparedSummary],
    ) -> None:
        from agent.thread_scoped_output import thread_scoped_silence

        entry: Optional[PreparedSummary] = None
        started = time.monotonic()
        try:
            logger.info(
                "prepared compaction: pass started origin=%s window=%d-%d turns=%d extends=%s",
                origin, start, end, len(turns), extend is not None,
            )
            with thread_scoped_silence():
                summary = worker._generate_summary(turns, focus_topic=focus)
            if summary:
                entry = PreparedSummary(start, end, window_fingerprint, summary, worker._previous_summary)
                logger.info("prepared compaction: pass completed window=%d-%d in %.1fs",
                            start, end, time.monotonic() - started)
            else:
                logger.warning("prepared compaction: pass failed after %.1fs: %s", time.monotonic() - started,
                               worker._last_summary_error or "summariser returned nothing")
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
            _slots.release()


def maybe_prepare(agent: Any, messages: List[Dict[str, Any]], *, origin: str) -> Optional[threading.Thread]:
    """Turn-loop hook; a no-op unless ``compression.prepare_ahead`` gave the compressor state."""
    compressor = getattr(agent, "context_compressor", None)
    state = getattr(compressor, "prepared_compaction", None)
    return state.maybe_prepare(agent, compressor, messages, origin) if state is not None else None
