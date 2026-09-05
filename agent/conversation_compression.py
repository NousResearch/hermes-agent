"""Context compression: feasibility probe, warning replay, compress, image fix.

Thread-safety contract for extension points: with ``compression.context_timeout_seconds > 0`` (default) the
whole pass, context engines and memory providers included, runs on a pooled daemon thread. Calls may arrive
on any pooled thread (never rely on thread-affinity/locals); the message list is a private deep snapshot
(in-place mutation is invisible to the live conversation unless the pass commits); state is published ONLY
on an admitted :class:`CompressionCommitFence` commit (work still running after a host timeout is discarded);
one pass per session at a time (durable lock) but sessions run concurrently, so shared engines must be thread-safe.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import dataclasses
import inspect
import json
import logging
import math
import os
import tempfile
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_engine import automatic_compaction_status_message, sanitize_memory_context
from agent.memory_provider import PRE_COMPRESS_CHECKPOINT_API_VERSION
from agent.model_metadata import estimate_messages_tokens_rough, estimate_request_tokens_rough
from agent.session_activity import ActivityProvenance, normalize_activity_provenance

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _swallow(message: str, *, exc_info: bool = False):
    """Run a best-effort block; on Exception log ``message`` at DEBUG and continue."""
    try:
        yield
    except Exception as exc:
        logger.debug(message, exc_info=True) if exc_info else logger.debug(message, exc)


# Terminal outcomes from host/hygiene timeout or cooldown writers. Detached heartbeat workers must not clobber these
# (timeout unobservable). Seeing one latches the heartbeat silent so a later UNKNOWN rewrite can't re-arm a zombie.
_TERMINAL_COMPRESSION_PROVENANCES = frozenset(
    {ActivityProvenance.AGENT_COMPRESSION_TIMEOUT, ActivityProvenance.AGENT_COMPRESSION_COOLDOWN}
)

# Split failures are usually transient lease/DB conditions, so use the FIRST
# timeout-ladder rung (60s), not the 600s summary-provider cooldown.
# Cooldown armed when a compression SPLIT fails (session_split_failed / rotation rollback, #97948 symptom
# B).
_SPLIT_FAILURE_COOLDOWN_SECONDS = 60

# Marker tui_gateway/server.py::_status_update matches to tag kind="compacting" for drivers' "Summarizing…" UI. Keep
# the phrase intact when rewording. Idle/preflight/retry lines lack it; is_compaction_progress_status covers those.
COMPACTION_STATUS_MARKER = "Compacting context"
COMPACTION_STATUS = f"🗜️ {COMPACTION_STATUS_MARKER} — summarizing earlier conversation so I can continue..."
# Periodic heartbeat re-emitted while a long compression is still running so remote transports with
# idle-turn watchdogs (#98371) see progress. Same marker as COMPACTION_STATUS so consumers classify it alike.
COMPACTION_HEARTBEAT_STATUS = f"🗜️ {COMPACTION_STATUS_MARKER} — still summarizing earlier conversation so I can continue..."

COMPACTION_DONE_STATUS = "✓ Context compaction complete — continuing turn..."

# Start and completion share a dedicated key so clients can replace one
# compaction bubble in place without unrelated lifecycle statuses overwriting
# it. End-of-turn progress cleanup still treats the bubble as ephemeral.
COMPACTION_STATUS_KEY = "compaction"


def _strip_marker_for_comparison(msgs: Any) -> Any:
    """Copy ``msgs`` with the ``_db_persisted`` marker removed for no-op comparison.
    Live dicts carry the marker while ``compress()`` output is swept, so a raw ``==`` would misclassify an
    identical no-op copy as progress. Non-list inputs and non-dict entries pass through unchanged."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    if not isinstance(msgs, list):
        return msgs
    return [{k: v for k, v in m.items() if k != _DB_PERSISTED_MARKER} if isinstance(m, dict) else m for m in msgs]


def _emit_compaction_done(agent: Any) -> None:
    """Emit the structured terminal edge for a started compaction."""
    status_callback = getattr(agent, "status_callback", None)
    if not status_callback:
        logger.info("compaction done: no status_callback; completion notice not emitted")
        return
    with _swallow('status_callback error in compaction completion', exc_info=True):
        status_callback(COMPACTION_STATUS_KEY, COMPACTION_DONE_STATUS)
        logger.info("compaction done: completion notice emitted")


# Every ROUTINE compression status line lives here: suppressed on chat platforms
# by _TELEGRAM_NOISY_STATUS_RE (gateway/run.py); update that regex + telegram
# noise test when rewording. Failure notices and /compress feedback: NOT here.
PRE_API_COMPRESSION_STATUS_TEMPLATE = (
    "📦 Pre-API compression: ~{tokens:,} tokens near the context/output limit. Compacting before the next model call."
)
PREFLIGHT_COMPRESSION_STATUS_TEMPLATE = (
    "📦 Preflight compression: ~{tokens:,} tokens >= {threshold:,} threshold. This may take a moment."
)
IDLE_COMPACTION_STATUS_TEMPLATE = (
    "💤 Resumed after {idle_seconds}s idle — compacting ~{tokens:,} tokens before continuing."
)
COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE = (
    "🗜️ Context too large (~{tokens:,} tokens) — compressing ({attempt}/{cap})..."
)
COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE = "🗜️ Compressed {before} → {after} messages, retrying..."
COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE = "🗜️ Compressed ~{before:,} → ~{after:,} tokens, retrying..."
COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE = (
    "🗜️ Context reduced to {new_ctx:,} tokens (was {old_ctx:,}), retrying..."
)

# FAILURE-class notice: compression blocked, so the session grows until the provider limit kills it. Must stay visible
# on gateways: never add it to ROUTINE_COMPRESSION_STATUS_SAMPLES or _TELEGRAM_NOISY_STATUS_RE.
# FAILURE-CLASS notice — a deliberate carve-out from routine-compression silence (#16775 class): the context
# is over the compression threshold but compression is blocked (summary-LLM cooldown / anti-thrash breaker),
# so the session will keep growing until the hard provider token limit kills it. Do NOT add it to
# ROUTINE_COMPRESSION_STATUS_SAMPLES or the gateway noise regex (_TELEGRAM_NOISY_STATUS_RE); it is pinned
# un-swallowed in tests/gateway/test_telegram_noise_filter.py::VISIBLE_COMPRESSION_MESSAGES.
CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE = (
    "⚠ Context is over the compression threshold (~{tokens:,} tokens >= {threshold:,}) "
    "but compression is currently blocked ({reason}). The model may stop responding. Run /new to start a fresh "
    "session or /compress to retry immediately."
)

# Formatted from the same constants the emission sites use, so noise-filter tests exercise the ACTUAL wording.
ROUTINE_COMPRESSION_STATUS_SAMPLES = (
    COMPACTION_STATUS, COMPACTION_HEARTBEAT_STATUS, COMPACTION_DONE_STATUS,
    PRE_API_COMPRESSION_STATUS_TEMPLATE.format(tokens=123456),
    PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(tokens=120000, threshold=100000),
    IDLE_COMPACTION_STATUS_TEMPLATE.format(idle_seconds=3600, tokens=120000),
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE.format(tokens=250000, attempt=1, cap=3),
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=30, after=12),
    COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(before=250000, after=120000),
    COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE.format(new_ctx=120000, old_ctx=250000),
)


def is_compaction_progress_status(text: str | None) -> bool:
    """True for in-progress auto-compaction lifecycle lines (not the done edge).
    The gateway re-tags matches as ``kind="compacting"`` for the whole pause; matching only the marker left
    idle/preflight/retry lines looking hung. ``COMPACTION_DONE_STATUS`` is the terminal edge (emitted under
    ``COMPACTION_STATUS_KEY``; the TUI maps it to ``compacted``) and must not match here."""
    body = text.strip() if isinstance(text, str) else ""
    if not body:
        return False
    if COMPACTION_STATUS_MARKER in body:
        return True
    if body == COMPACTION_DONE_STATUS:
        return False
    lowered = body.lower()
    # The failure-class overflow warning mentions compression but is a blocked notice, not progress.
    if "compaction complete" in lowered or "compression is currently blocked" in lowered:
        return False
    return "compact" in lowered or "compress" in lowered or "context reduced to" in lowered


def _refresh_agent_tool_definitions(agent) -> bool:
    """Rebuild agent.tools at the compaction commit boundary (the only moment config reaches a forever-session's
    frozen tool schemas; the prompt cache is already invalid). Returns True when tools were added."""
    from tools.mcp_tool_agent import refresh_agent_mcp_tools
    added = refresh_agent_mcp_tools(agent, content_aware=True)
    if added:
        logger.info("Compaction tool refresh added tools: %s", sorted(added))
    return bool(added)


_COMPRESSOR_ATTEMPT_STATE_FIELDS = (
    "_previous_summary", "_summary_has_user_turn", "compression_count", "_last_compression_savings_pct",
    "_ineffective_compression_count", "_anti_thrash_recovery_deadline", "_fallback_compression_streak",
    "_verify_compaction_cleared_threshold", "_last_compression_made_progress", "_summary_failure_cooldown_until",
    "_cooldown_persist_failed", "_last_summary_error", "_consecutive_timeout_failures", "_last_summary_dropped_count",
    "_last_summary_fallback_used", "_last_compress_aborted", "_last_summary_auth_failure",
    "_last_summary_network_failure", "_last_summary_empty_content_failure", "_last_summary_truncated_failure",
    "_last_aux_model_failure_error", "_last_aux_model_failure_model", "_summary_model_fallen_back", "summary_model",
    "_last_compression_telemetry", "_active_compression_telemetry", "_compression_telemetry_seed",
    "_proactive_prune_rearm_tokens",
)

_COMPRESSOR_COOLDOWN_STATE_FIELDS = ("_summary_failure_cooldown_until", "_last_summary_error", "_cooldown_persist_failed")


def _snapshot_compressor_attempt_state(compressor: Any) -> dict[str, Any]:
    """Copy only the mutable bookkeeping owned by one compression attempt.
    The allow-list avoids copying clients, DB handles, locks and plugin resources; missing fields are ignored
    so legacy/third-party compressors keep working."""
    try:
        values = vars(compressor)
    except TypeError:
        return {}
    selected = {name: values[name] for name in _COMPRESSOR_ATTEMPT_STATE_FIELDS if name in values}
    # Copy the collection as one object so aliases between fields (notably
    # _active_compression_telemetry and _last_compression_telemetry) survive.
    return copy.deepcopy(selected)


# Attempt ownership: stall-fallback detaches a timed-out worker and reuses the compressor, so its late unwind could
# restore a stale snapshot or clear the fallback's cancel check. Generation guards ATTRIBUTE writes; fence, COMMITs.

# --------------------------------------------------------------------------- Attempt ownership (#96634
# follow-up). The stall-fallback path deliberately DETACHES a timed-out primary worker (fence cancel wins;
# the future stays on the shared pool) and immediately starts a fallback attempt against the SAME
# ContextCompressor. Two races follow from that overlap: 1. The late primary's unwind still calls
# _restore_compressor_attempt_state with the PRIMARY's pre-attempt snapshot. Landing after the fallback's
# commit, it rolls _previous_summary / cooldown / provenance / telemetry back to pre-primary values —
# silently discarding fallback-owned state. 2. _compression_cancelled_check is one shared attribute: the
# late primary's ``finally`` clears the callback the fallback just installed, so the fallback's F4
# cancellation consult reads None. Both are fixed with a monotonic per-compressor attempt generation,
# claimed under one module lock. Restores and callback set/clear are keyed to the claiming generation and
# no-op when a newer attempt owns the compressor. The commit fence still owns COMMIT admission; the
# generation owns compressor-ATTRIBUTE writes — two different boundaries.
# ---------------------------------------------------------------------------
_COMPRESSOR_ATTEMPT_LOCK = threading.Lock()


def _claim_compressor_attempt(compressor: Any) -> int:
    """Claim the compressor for a new attempt; return its monotonic generation id.
    Restores or cancelled-check mutations stamped with an OLDER generation no-op, so a detached late attempt
    cannot clobber its successor's state."""
    with _COMPRESSOR_ATTEMPT_LOCK:
        generation = int(getattr(compressor, "_compression_attempt_generation", 0) or 0) + 1
        try:
            compressor._compression_attempt_generation = generation
        except Exception:
            return 0  # slotted/frozen compressor: gen 0 disables the guard (per-instance, so never mixed)
        return generation


def _compressor_attempt_is_current(compressor: Any, generation: int) -> bool:
    """True when *generation* still owns the compressor (or guard disabled)."""
    if not generation:
        return True
    with _COMPRESSOR_ATTEMPT_LOCK:
        return int(getattr(compressor, "_compression_attempt_generation", 0) or 0) == generation


def _install_compression_cancelled_check(compressor: Any, check: Any, generation: int) -> None:
    """Install the F4 cancellation consult, stamped with its owner attempt."""
    with _COMPRESSOR_ATTEMPT_LOCK:
        with contextlib.suppress(Exception):
            compressor._compression_cancelled_check = check
            compressor._compression_cancelled_check_owner = generation


def _clear_compression_cancelled_check_if_owner(compressor: Any, generation: int) -> bool:
    """Clear the cancellation consult only when *generation* installed it.
    Prevents a detached late primary from tearing down a newer fallback's callback. Returns True when cleared."""
    with _COMPRESSOR_ATTEMPT_LOCK:
        owner = getattr(compressor, "_compression_cancelled_check_owner", None)
        if owner is not None and generation and owner != generation:
            return False
        with contextlib.suppress(Exception):
            compressor._compression_cancelled_check = None
            compressor._compression_cancelled_check_owner = None
        return True


def _rollback_durable_cooldown(
    compressor: Any, snapshot: dict[str, Any], authoritative: Optional[bool], durable_state: Optional[dict[str, Any]]
) -> None:
    """Recreate/clear the durable cooldown row from the attempt snapshot.
    Authoritative captures use the exact raw-row restore API (verifies read-back, propagates failure); the
    legacy path re-derives deadline/error best-effort."""
    session_db = vars(compressor).get("_session_db")
    session_id = vars(compressor).get("_session_id")
    if session_db is None or not session_id:
        return
    if authoritative is True:
        restorer = getattr(type(session_db), "restore_compression_failure_cooldown_row", None)
        if not callable(restorer) or durable_state is None:
            raise RuntimeError("exact compression cooldown rollback API is unavailable")
        restorer(session_db, session_id, copy.deepcopy(durable_state))
        return
    with _swallow('compression cooldown persistence rollback failed', exc_info=True):
        deadline = float(snapshot["_summary_failure_cooldown_until"] or 0.0)
        remaining = max(0.0, deadline - time.monotonic())
        if remaining > 0:
            recorder = getattr(type(session_db), "record_compression_failure_cooldown", None)
            if callable(recorder):
                recorder(session_db, session_id, time.time() + remaining, snapshot.get("_last_summary_error"))
        else:
            clearer = getattr(type(session_db), "clear_compression_failure_cooldown", None)
            if callable(clearer):
                clearer(session_db, session_id)


def _restore_compressor_attempt_state(
    compressor: Any, snapshot: dict[str, Any], *, durable_cooldown_authoritative: Optional[bool] = None,
    durable_cooldown_state: Optional[dict[str, Any]] = None, attempt_generation: Optional[int] = None,
) -> None:
    """Restore the per-attempt snapshot after a pre-commit hard cancel.
    A restore stamped with a stale ``attempt_generation`` no-ops so a timed-out primary's late unwind cannot
    roll back state owned by the fallback attempt.

    ``attempt_generation`` (when provided) is the claim the calling attempt took via
    :func:`_claim_compressor_attempt`. See #96634.
    """
    if attempt_generation is not None and not _compressor_attempt_is_current(compressor, attempt_generation):
        logger.warning(
            "Skipping stale compressor attempt-state restore: attempt "
            "generation %s no longer owns the compressor (current: %s). A newer (stall-fallback) attempt's state is preserved.",
            attempt_generation, getattr(compressor, "_compression_attempt_generation", None),
        )
        return
    # Success clears the durable cooldown pre-commit; recreate/clear that row BEFORE
    # restoring in-memory values or the next refresh overwrites the rollback. Never
    # turn unknown durable state / unpersisted local cooldowns into DB writes.
    if (
        "_summary_failure_cooldown_until" in snapshot
        and durable_cooldown_authoritative is not False
        and (durable_cooldown_authoritative is True or not bool(snapshot.get("_cooldown_persist_failed", False)))
    ):
        _rollback_durable_cooldown(compressor, snapshot, durable_cooldown_authoritative, durable_cooldown_state)
    restored = copy.deepcopy(snapshot)
    # Re-validate under the claim lock: the slow durable rollback above leaves a
    # window where a fallback may have claimed; stale writes must not interleave.
    # The rollback itself is safe: landing after a fallback needs a prior claim.
    with _COMPRESSOR_ATTEMPT_LOCK:
        if attempt_generation and int(getattr(compressor, "_compression_attempt_generation", 0) or 0) != attempt_generation:
            logger.warning(
                "Skipping stale compressor attempt-state restore at write "
                "time: attempt generation %s lost the compressor mid-restore.", attempt_generation,
            )
            return
        for name, value in restored.items():
            setattr(compressor, name, value)


def _capture_authoritative_cooldown_under_lease(
    compressor: Any, attempt_snapshot: dict[str, Any]
) -> tuple[Optional[bool], Optional[dict[str, Any]]]:
    """Refresh and snapshot built-in durable cooldown state under the lease.
    Third-party compressors are not invoked: plugin code must not run under the lease. Returns ``False`` on
    durable read failure (rollback must not mistake unknown state for an empty row) and ``None`` when the
    legacy API is absent."""
    try:
        from agent.context_compressor import ContextCompressor
        if not isinstance(compressor, ContextCompressor):
            return None, None
        values = vars(compressor)
        session_db = values.get("_session_db")
        session_id = values.get("_session_id")
        if session_db is None or not session_id:
            # Unbound compressors have no durable row to mutate or restore.
            return None, None
        raw_reader = getattr(type(session_db), "get_compression_failure_cooldown_row", None)
        if not callable(raw_reader):
            return False, None
        # Read the raw persisted row: the active getter filters expired rows and is not a lossless rollback snapshot.
        durable_state = raw_reader(session_db, session_id)
        if not isinstance(durable_state, dict):
            raise TypeError("raw compression cooldown snapshot must be a mapping")
        ContextCompressor.get_active_compression_failure_cooldown(compressor, refresh=True)
    except Exception as exc:
        logger.debug("authoritative compression cooldown capture failed: %s", exc)
        return False, None
    authoritative = getattr(compressor, "_last_cooldown_refresh_was_authoritative", None)
    if authoritative is not True:
        return authoritative, None
    values = vars(compressor)
    for name in _COMPRESSOR_COOLDOWN_STATE_FIELDS:
        if name in values:
            attempt_snapshot[name] = copy.deepcopy(values[name])
    return True, copy.deepcopy(durable_state)


class CompressionCommitFence:
    """Fence timeout cancellation against post-summary session mutation.
    The sync worker thread cannot be killed; the fence makes the commit boundary deterministic: cancellation
    wins before mutation starts, or waits for an already-started commit to finish completely."""

    def __init__(self, total_ceiling_seconds: float | None = None) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._commit_started = False
        # Readable WITHOUT the lock (begin_commit holds it until finish_commit): hosts see a hung commit.
        # Lock-free commit-phase marker (#76354 review F1). ``begin_commit`` RETAINS ``self._lock`` until
        # ``finish_commit``, so any host-side observation that needs the lock (``try_cancel_before_commit``)
        # blocks/space-outs for the whole commit. This Event is set inside ``begin_commit`` while the lock
        # is held but is READABLE WITHOUT the lock, so a host can observe "a commit was admitted and may be
        # in flight" even while the commit itself is hung — which is exactly when the overrun warning must
        # be able to fire.
        self._commit_phase = threading.Event()
        # Set on ANY host unwind without the fence lock so FUTURE commits are blocked; bool store is atomic.
        # Lock-free admission revocation (#76354 review F2). Set by :meth:`revoke_commit_admission` on ANY
        # host unwind (KeyboardInterrupt, cancellation, unexpected exception) without touching the fence
        # lock, so a host that cannot afford to block behind an in-flight commit can still guarantee no
        # FUTURE commit is admitted.
        self._admission_revoked = False
        # Holder-scoped release published by the worker once it owns the durable lock (no ABA on a NEW holder).
        # Holder-qualified durable-lock release hook (#76354 review F4; transplanted from PR #71569 by
        # @ciabata-git). The worker publishes an idempotent, holder-scoped release callable once it owns the
        # durable compression lock; a timed-out host invokes it to free the lease without racing a NEW
        # holder (DB release is holder-qualified, so a stale release can never delete a replacement's row —
        # no ABA).
        self._lock_release_guard = threading.Lock()
        self._cancelled_lock_release: Optional[Callable[[], None]] = None
        self._cancelled_lock_release_requested = False
        # Touched per streamed token so waiters tell SLOW-but-alive from HUNG (no fixed wall-clock kill).
        self._last_progress = time.monotonic()
        self._progress_observed = False
        self._deadline: float | None = None
        self._retain_cancelled_lock_until_worker_done = False
        # Set once the active-row watermark is captured: later rows survive as tail, so hosts may keep admission.
        self._commit_watermark_fenced = False
        if total_ceiling_seconds is not None:
            self.set_total_ceiling_seconds(total_ceiling_seconds)

    def set_total_ceiling_seconds(self, seconds: float) -> None:
        """Arm the wall-clock deadline shared by the host and worker."""
        seconds = float(seconds)
        if seconds <= 0:
            raise ValueError("total compression ceiling must be positive")
        self._deadline = time.monotonic() + seconds

    def touch_progress(self) -> None:
        """Record forward progress (a streamed token); a bare float store is atomic, so no lock."""
        self._last_progress = time.monotonic()
        self._progress_observed = True

    @property
    def progress_observed(self) -> bool:
        """Whether semantic provider progress was reported for this attempt."""
        return self._progress_observed

    @property
    def deadline_exceeded(self) -> bool:
        deadline = self._deadline
        return deadline is not None and time.monotonic() >= deadline

    @property
    def deadline_monotonic(self) -> float | None:
        """Armed deadline (absolute monotonic); the worker's stream consumer stops when the host stops waiting.

        :meth:`set_total_ceiling_seconds` documents this deadline as "shared by the host and worker", but
        until #99692 only the host could read it — ``deadline_exceeded`` answers "is it past?" for a caller
        that is already polling, which is useless to a worker blocked inside a provider stream. Publishing
        the instant itself lets the worker's stream consumer stop at exactly the moment the host stops
        waiting (see ``auxiliary_client.aux_stream_deadline``).
        """
        return self._deadline

    def seconds_since_progress(self) -> float:
        """Seconds since the worker last reported forward progress."""
        return max(0.0, time.monotonic() - self._last_progress)

    def cancel_before_commit(self, cancel_event: Any = None) -> bool:
        """Cancel a pending commit (``True``), or block until an active commit finishes (``False``)."""
        with self._lock:
            if not self._commit_started:
                self._cancelled = True
            if cancel_event is not None:
                cancel_event.set()
            return not self._commit_started

    def try_cancel_before_commit(self) -> Optional[bool]:
        """Non-blocking :meth:`cancel_before_commit`; ``None`` while an active commit owns the fence."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if not self._commit_started:
                self._cancelled = True
            return not self._commit_started
        finally:
            self._lock.release()

    def begin_commit(self, cancel_event: Any = None) -> bool:
        """Atomically admit commit unless a hard cancellation already won."""
        self._lock.acquire()
        if self.is_cancelled or self._admission_revoked or (cancel_event is not None and bool(cancel_event.is_set())):
            self._cancelled = True
            self._lock.release()
            if self._admission_revoked:
                # A revoke that lost the fence-lock race deferred its lease release; commit refused: release now.
                self.release_cancelled_compression_lock()
            return False
        self._commit_started = True
        # Set under the fence lock so commit_in_flight is never True for a commit that lost to cancellation.
        self._commit_phase.set()
        return True

    def finish_commit(self) -> None:
        """Leave a commit boundary entered by :meth:`begin_commit`."""
        self._commit_phase.clear()
        self._lock.release()
        if self._admission_revoked:
            # A revoke during THIS commit deferred its lease release (never free mid-mutation); release now.
            self.release_cancelled_compression_lock()

    @property
    def commit_in_flight(self) -> bool:
        """Lock-free read: an admitted commit is in progress (hosts reach the overrun loop on a hung commit)."""
        return self._commit_phase.is_set()

    @property
    def is_cancelled(self) -> bool:
        """True after cancellation won before the commit boundary."""
        return self._cancelled or self._admission_revoked or self.deadline_exceeded

    def retain_compression_lock_until_worker_done(self) -> None:
        """Prevent a timed-out live worker from overlapping a retry."""
        self._retain_cancelled_lock_until_worker_done = True

    def mark_commit_watermark_fenced(self) -> None:
        """Record a watermark-bounded commit (later rows survive as tail); a detached worker may keep admission.

        Called by the compression worker right after it captures ``get_active_message_watermark()`` under
        the durable compression lock (#75316/#87484). A watermark-fenced commit archives ONLY rows at or
        below the watermark; rows appended later — e.g. the user turn the host released at the turn-hold
        boundary (#97963) — are cloned as live concurrent tail. That is exactly the property a host needs
        before letting a detached worker keep its commit admission.
        """
        self._commit_watermark_fenced = True

    @property
    def commit_watermark_fenced(self) -> bool:
        """Lock-free read: the worker's commit is watermark-bounded."""
        return self._commit_watermark_fenced

    def allow_cancelled_lock_release(self) -> None:
        """Undo :meth:`retain_compression_lock_until_worker_done` once a bounded join proved the worker exited."""
        self._retain_cancelled_lock_until_worker_done = False

    def revoke_commit_admission(self) -> None:
        """Revoke FUTURE commit admission without blocking on the fence lock.
        An in-flight commit is never abandoned (``begin_commit`` re-checks the flag under the lock). The lease
        release must not run mid-commit: released now if the lock is free, else deferred to
        ``finish_commit``/refusal (holder-qualified)."""
        self._admission_revoked = True
        if self._lock.acquire(blocking=False):
            try:
                self.release_cancelled_compression_lock()
            finally:
                self._lock.release()

    # ── Holder-qualified durable-lease cancellation: release is DELETE WHERE
    # holder = ?, so a stale release can never free a NEW holder's lease (no ABA).

    # ── Holder-qualified durable-lease cancellation (#76354 F4) ────────── Transplanted from PR #71569
    # (@ciabata-git): the worker publishes an idempotent, holder-scoped release hook once it owns the
    # durable compression lock, and the host invokes it after winning cancellation. ABA safety comes from
    # SessionDB.release_compression_lock being holder-qualified (DELETE ... WHERE holder = ?), so a stale
    # release can never free a NEW holder's lease.
    def begin_lock_setup(self) -> bool:
        """Hold the fence across lock acquisition + release-hook publication so a timeout cannot win between."""
        self._lock.acquire()
        if self.is_cancelled or self._admission_revoked:
            self._lock.release()
            return False
        return True

    def finish_lock_setup(self) -> None:
        """Leave a lock setup boundary entered by :meth:`begin_lock_setup`."""
        self._lock.release()

    def register_cancelled_lock_release(self, release: Callable[[], None]) -> bool:
        """Publish the worker's holder-qualified release; if cleanup was already requested, run it and return True."""
        with self._lock_release_guard:
            self._cancelled_lock_release = release
            requested = self._cancelled_lock_release_requested
        if requested:
            release()
        return requested

    def clear_cancelled_lock_release(self, release: Callable[[], None]) -> None:
        """Forget ``release`` after the worker's normal cleanup finishes."""
        with self._lock_release_guard:
            if self._cancelled_lock_release is release:
                self._cancelled_lock_release = None

    def release_cancelled_compression_lock(self) -> None:
        """After cancellation won: release the worker's lock (a request ahead of hook publication is retained)."""
        if self._retain_cancelled_lock_until_worker_done:
            return
        with self._lock_release_guard:
            self._cancelled_lock_release_requested = True
            release = self._cancelled_lock_release
        if release is not None:
            release()


# Defaults for the in-agent progress-aware wrap; mirror hermes_cli.config.DEFAULT_CONFIG["compression"] keys.
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS = 600.0

# Unlike explicit_interrupt, a /stop after the stall window arms the durable backoff (no automatic re-entry).
# Distinct from ``explicit_interrupt``: a /stop that arrived after the summary stream had already crossed
# the no-progress stall window (#96775). Ordinary early /stop stays cooldown-neutral; this class arms the
# durable backoff so the next automatic turn does not re-enter the same stalled strategy.
STALL_INTERRUPTED_FAILURE_CLASS = "stall_interrupted"

# Daemon pool so a fence-cancelled hung worker cannot block interpreter exit; never shut down per call.
_compress_timeout_executor = None
_compress_timeout_executor_lock = threading.Lock()

# Overrun waits proceed in bounded slices so each window logs (escalating); clamped to ceiling for tiny values.
_COMMIT_OVERRUN_WAIT_SLICE_SECONDS = 30.0

# A worker exiting within the grace proves no provider call is in flight, so its lease may be released even
# on the total-ceiling path; one that doesn't exit is orphaned behind the poison fence and keeps its lease.
# Bounded grace given to a fence-cancelled compression worker to actually exit before the host moves on
# (#97488).
_CANCELLED_WORKER_TEARDOWN_GRACE_SECONDS = 5.0


def _join_cancelled_worker(future: Any, grace_seconds: float) -> bool:
    """Best-effort bounded join of a fence-cancelled compression worker.
    Returns True when the future settled within ``grace_seconds`` (thread provably exited); False for a
    still-running worker, which the caller must treat as an orphan behind the poison fence."""
    grace = 0.0
    with contextlib.suppress(TypeError, ValueError):
        grace = max(float(grace_seconds), 0.0)
    try:
        future.result(timeout=grace)
        return True
    except concurrent.futures.TimeoutError:
        return False
    except concurrent.futures.CancelledError:
        # Never started; nothing can be in flight.
        return True
    except Exception:
        # The host already chose the fallback result; the fence keeps the failed attempt from touching state.
        logger.debug("cancelled compression worker exited with an exception", exc_info=True)
        return True


# The executor queue is unbounded and a queued job would run stale, so admission is capped at the worker
# count (fail fast, continue uncompressed). Slots free via done-callback; a never-returning worker loses one.
# Bounded admission for the shared compress-timeout pool (#76354 review F6). The stdlib executor queue is
# unbounded: with all four workers wedged in hung summaries, a fifth compression would queue silently, wait
# out its whole timeout without ever starting, and remain eligible to run as a stale job whenever a worker
# recovered. Admission is therefore capped at the worker count — when every worker slot is occupied (running
# OR admitted-not-started) submission FAILS FAST and the caller continues without compression. Recovery
# contract when all workers are wedged: new compressions fail fast (no queue growth, conversation continues
# uncompressed, a warning is logged each attempt); wedged workers are fence-cancelled so they cannot publish
# anything when they eventually return, and each recovery frees its admission slot via the future
# done-callback, restoring normal service. If a worker NEVER returns, its slot is lost for the process
# lifetime — bounded, observable degradation instead of an unbounded stale-job queue.
_COMPRESS_EXECUTOR_MAX_WORKERS = 4
_compress_admission_lock = threading.Lock()
_compress_admitted_count = 0


def _try_admit_compression_job() -> bool:
    """Reserve one bounded compression-pool admission slot (F6)."""
    global _compress_admitted_count
    with _compress_admission_lock:
        if _compress_admitted_count >= _COMPRESS_EXECUTOR_MAX_WORKERS:
            return False
        _compress_admitted_count += 1
        return True


def _release_compression_admission(_future=None) -> None:
    """Free an admission slot (future done-callback or failed submit)."""
    global _compress_admitted_count
    with _compress_admission_lock:
        if _compress_admitted_count > 0:
            _compress_admitted_count -= 1


def _get_compress_timeout_executor():
    """Return the process-wide compress-timeout DaemonThreadPoolExecutor."""
    global _compress_timeout_executor
    executor = _compress_timeout_executor
    if executor is not None:
        return executor
    from tools.daemon_pool import DaemonThreadPoolExecutor
    with _compress_timeout_executor_lock:
        if _compress_timeout_executor is None:
            # Small pool sized for live compress + cancelled workers winding down, not asyncio's min(32, cpu+4).
            _compress_timeout_executor = DaemonThreadPoolExecutor(
                max_workers=_COMPRESS_EXECUTOR_MAX_WORKERS, thread_name_prefix="compress-ctx-timeout"
            )
        return _compress_timeout_executor


def resolve_context_compression_timeouts(compression_cfg: Optional[dict] = None) -> Tuple[float, float]:
    """Return ``(idle_timeout_seconds, total_ceiling_seconds)``.
    ``idle_timeout_seconds <= 0`` disables the progress-aware wrapper. The ceiling is clamped to at least one
    idle window when the idle budget is positive."""
    idle = DEFAULT_CONTEXT_TIMEOUT_SECONDS
    ceiling = DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS
    cfg = compression_cfg
    if cfg is None:
        cfg = {}
        with contextlib.suppress(Exception):
            from hermes_cli.config import load_config
            raw = load_config()
            maybe = raw.get("compression", {}) if isinstance(raw, dict) else {}
            cfg = maybe if isinstance(maybe, dict) else {}
    if isinstance(cfg, dict):
        # Explicit 0/negative idle disables; a non-positive ceiling is ignored.
        with contextlib.suppress(TypeError, ValueError):
            if cfg.get("context_timeout_seconds") is not None:
                idle = float(cfg["context_timeout_seconds"])
        with contextlib.suppress(TypeError, ValueError):
            if cfg.get("context_total_ceiling_seconds") is not None and float(cfg["context_total_ceiling_seconds"]) > 0:
                ceiling = float(cfg["context_total_ceiling_seconds"])
    if idle > 0:
        ceiling = max(ceiling, idle)
    return idle, ceiling


def compression_attempt_stalled(
    *, commit_fence: Optional[CompressionCommitFence], started_at: float, idle_timeout_seconds: Optional[float] = None
) -> bool:
    """Return whether a pre-commit cancel landed after the stall window.
    An early ``/stop`` stays cooldown-neutral; an interrupt after the inactivity budget counts as a stall so
    the next automatic turn does not blindly retry.

    When the fence (or, without a fence, the attempt clock) has already sat idle for the configured
    compression inactivity budget, the interrupt is a stalled attempt — the same condition the host timeout
    uses — and the next automatic turn must not blindly retry that strategy (#96775).
    """
    idle = idle_timeout_seconds
    if idle is None:
        idle, _ceiling = resolve_context_compression_timeouts()
    try:
        idle = float(idle)
    except (TypeError, ValueError):
        return False
    if idle <= 0:
        return False
    if commit_fence is not None:
        try:
            return float(commit_fence.seconds_since_progress()) >= idle
        except Exception:
            return False
    try:
        return (time.monotonic() - float(started_at)) >= idle
    except (TypeError, ValueError):
        return False


def _stall_source_fingerprint(agent: Any, messages: Any, approx_tokens: Optional[int]) -> str:
    """Identity of the stalled source context + summary strategy."""
    compressor = getattr(agent, "context_compressor", None)
    model = getattr(compressor, "summary_model", None) or getattr(agent, "model", None) or ""
    n_messages = len(messages) if isinstance(messages, list) else 0
    tokens = 0
    with contextlib.suppress(TypeError, ValueError):
        tokens = int(approx_tokens or 0)
    return f"msgs={n_messages}:tokens={tokens}:model={model}"


def _record_stall_interrupted_backoff(
    agent: Any, *, commit_fence: Optional[CompressionCommitFence], started_at: float, messages: Any,
    approx_tokens: Optional[int],
) -> bool:
    """Persist a stall-interrupted cooldown after snapshot restore.
    Must run *after* ``_restore_compressor_attempt_state`` so rollback cannot wipe the new row. Returns True
    when the backoff was recorded."""
    if not compression_attempt_stalled(commit_fence=commit_fence, started_at=started_at):
        return False
    compressor = getattr(agent, "context_compressor", None)
    # Same timeout cooldown ladder as summary-LLM timeouts (#62452): avoid re-burning the full idle budget
    # every turn.
    record = getattr(compressor, "record_timeout_failure", None)
    if not callable(record):
        return False
    error = f"{STALL_INTERRUPTED_FAILURE_CLASS}:{_stall_source_fingerprint(agent, messages, approx_tokens)}"
    try:
        record(error, failure_kind="stall_interrupted")
    except Exception:
        logger.debug("stall-interrupted compression cooldown persist failed", exc_info=True)
        return False
    logger.info(
        "Recorded stall-interrupted compression backoff (session=%s, %s)", getattr(agent, "session_id", None) or "none",
        error,
    )
    return True


def resolve_compression_fallback_route() -> Optional[dict]:
    """Return the first usable ``auxiliary.compression.fallback_chain`` entry.
    The aux client applies the chain only from its exception handler, so a silent stall never reaches it; this
    pins the route onto one bounded retry instead. Only the first complete entry: if it errors, the aux
    client's own exception path walks the rest. ``None`` when none is usable (skip compression)."""
    try:
        from agent.auxiliary_client import _fallback_entry_api_key, _get_auxiliary_task_config
        chain = _get_auxiliary_task_config("compression").get("fallback_chain")
    except Exception:
        logger.debug("compression fallback_chain lookup failed", exc_info=True)
        return None
    if not isinstance(chain, list):
        return None
    for index, entry in enumerate(chain):
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        # Both are required to name a route. _resolve_fallback_entry applies
        # the same rule when the aux client walks this chain itself.
        if not provider or not model:
            continue
        try:
            api_key = _fallback_entry_api_key(entry)
        except Exception:
            logger.debug("compression fallback_chain[%d] api key resolution failed", index, exc_info=True)
            api_key = None
        from agent.auxiliary_client import _coerce_positive_timeout
        timeout = _coerce_positive_timeout(entry.get("timeout"))
        return {
            "label": f"fallback_chain[{index}]({provider})",
            "provider": provider,
            "model": model,
            "base_url": str(entry.get("base_url") or "").strip() or None,
            "api_key": api_key or None,
            "api_mode": str(entry.get("api_mode") or entry.get("transport") or "").strip() or None,
            "timeout": timeout,
        }
    return None


def _retry_compression_on_fallback_chain(
    *, worker: Callable[[CompressionCommitFence], Tuple[list, str]], messages: list,
    system_prompt_fallback: Any, idle_timeout_seconds: float, total_ceiling_seconds: float,
    on_commit_overrun: Optional[Callable[[float, float], None]] = None,
    on_timeout_cause: Optional[Callable[[bool, bool], None]] = None, telemetry_agent: Any = None,
    new_fence: Optional[Callable[[], CompressionCommitFence]] = None,
) -> Optional[Tuple[list, str]]:
    """Re-run an aborted compression once with the summary route pinned.
    Returns ``(messages, system_prompt)`` on real compression, else ``None`` and the caller degrades as
    before. The entry's ``timeout`` sets the idle window. Re-runs the whole worker, so pre-compression
    callbacks must be idempotent.

    The retry is bounded the same way the primary was: silence for one idle window ends it, while a fallback
    that is streaming keeps its ceiling. The entry's own ``timeout`` (when declared) sets that idle window,
    so a fallback tuned for a slower-but-healthy backend is not held to a deadline the stalled primary
    defined (#62452 semantics, applied to the stall path).
    Known limitation (accepted, #96634 review): the retry re-runs the COMPLETE worker, which repeats
    memory/plugin pre-compression callbacks. Built-in callbacks are idempotent (re-reads and overwrites of
    attempt-scoped state); third-party plugin callbacks are advised to be. Splitting the worker to resume
    mid-pipeline would couple this path to every host's callback ordering — deliberately out of scope.
    """
    # An explicit stop is not a stalled route. The retry worker would abort on
    # the same event anyway, but starting one at all makes /stop look ignored.
    hard_cancel = getattr(telemetry_agent, "_hard_interrupt_requested", None)
    if callable(getattr(hard_cancel, "is_set", None)) and hard_cancel.is_set():
        return None
    route = resolve_compression_fallback_route()
    if route is None:
        return None

    # The aborted fence refuses all commits; mint a fresh one via the host factory
    # so a /stop during the retry serializes against THIS attempt's commit boundary.
    retry_fence = None
    if new_fence is not None:
        try:
            retry_fence = new_fence()
        except Exception:
            logger.warning(
                "compression stall-fallback fence factory failed; the retry "
                "will run on an unpublished fence (a /stop mid-retry cannot serialize against its commit boundary)",
                exc_info=True,
            )
    if not isinstance(retry_fence, CompressionCommitFence):
        logger.warning(
            "compression stall-fallback retry running on an unpublished fence; "
            "hard-interrupt admission will read the aborted attempt's fence rather than the retry's commit boundary"
        )
        retry_fence = CompressionCommitFence()
    idle = float(route.get("timeout") or idle_timeout_seconds)
    ceiling = max(float(total_ceiling_seconds), idle)
    logger.warning(
        "Context compression stalled on the configured summary route — "
        "retrying once on %s (%s) before continuing without compression", route["label"], route["model"],
    )
    try:
        from agent.context_compressor import pin_summary_route
        with pin_summary_route(route):
            result_msgs, result_prompt = run_compress_context_with_progress_timeout(
                worker=worker, messages=messages, system_prompt_fallback=system_prompt_fallback,
                idle_timeout_seconds=idle, total_ceiling_seconds=ceiling, on_commit_overrun=on_commit_overrun,
                on_timeout_cause=on_timeout_cause, fence=retry_fence, telemetry_agent=telemetry_agent,
                stall_fallback=False,
            )
    except Exception:
        # The primary already failed; a failing fallback must degrade, never
        # turn "continue without compression" into a raised turn.
        logger.warning("Context compression fallback attempt on %s failed", route["label"], exc_info=True)
        return None
    if result_msgs is messages:
        # Aborted or no-op: the worker hands back the caller's own list.
        logger.warning(
            "Context compression fallback attempt on %s produced no compression; continuing without compression",
            route["label"],
        )
        return None
    logger.info("Context compression recovered on %s after the primary summary route stalled", route["label"])
    return result_msgs, result_prompt


def _await_worker_within_budget(
    future: Any, fence: CompressionCommitFence, *, idle: float, ceiling: float, wait_started: float
) -> Tuple[bool, Any]:
    """Poll ``future`` under the idle budget + ceiling; ``(True, result)`` when it settled."""
    while True:
        waited = time.monotonic() - wait_started
        remaining_ceiling = ceiling - waited
        if remaining_ceiling <= 0:
            return False, None
        # Charge idle budget from LAST PROGRESS, not slice start, or silence could approach 2x the budget.
        since_progress = fence.seconds_since_progress()
        wait_slice = min(max(idle - since_progress, 0.005), remaining_ceiling)
        try:
            return True, future.result(timeout=wait_slice)
        except concurrent.futures.TimeoutError:
            waited = time.monotonic() - wait_started
            since_progress = fence.seconds_since_progress()
            if not fence.deadline_exceeded and since_progress < idle and waited < ceiling:
                logger.info(
                    "Context compression still streaming after %.0fs (last progress %.1fs ago) — extending wait (ceiling %.0fs)",
                    waited, since_progress, ceiling,
                )
                continue
            return False, None


def _await_in_flight_commit(
    future: Any, *, ceiling: float, wait_started: float, on_commit_overrun: Optional[Callable[[float, float], None]]
) -> Any:
    """begin_commit won the race: the SessionDB mutation cannot be fence-cancelled, so wait
    in bounded slices, logging (escalating) + surfacing once via ``on_commit_overrun``
    WHILE the commit hangs. Never silently hung or abandoned.
    """
    overrun_surfaced = False
    overrun_reports = 0
    while True:
        waited = time.monotonic() - wait_started
        remaining = ceiling - waited
        if remaining <= 0:
            # Bounded increments so each overrun window is visible in logs rather than one silent unbounded block.
            remaining = min(_COMMIT_OVERRUN_WAIT_SLICE_SECONDS, max(ceiling, 0.05))
            overrun_reports += 1
            log = logger.warning if overrun_reports <= 2 else logger.error
            log(
                "Context compression SessionDB commit still running "
                "%.1fs past the total ceiling (waited %.1fs, ceiling %.1fs); commit cannot be abandoned mid-flight — "
                "continuing to wait (check SessionDB health if this persists)", waited - ceiling, waited,
                ceiling,
            )
            if not overrun_surfaced and on_commit_overrun is not None:
                overrun_surfaced = True
                with _swallow('compress_context commit-overrun callback failed', exc_info=True):
                    on_commit_overrun(waited, ceiling)
        try:
            return future.result(timeout=remaining)
        except concurrent.futures.TimeoutError:
            # Commit-phase progress is informative only — the commit must complete; loop
            # and re-report with the updated overrun window.
            continue


def _cancel_or_join_worker(fence: CompressionCommitFence) -> bool:
    """Cancel pre-commit; ``False`` when an admitted commit owns the fence (caller waits)."""
    while True:
        # begin_commit holds the fence lock until finish_commit, so try_cancel spins
        # forever on a hung commit; lock-free marker makes the overrun loop reachable.
        if fence.commit_in_flight:
            return False
        cancelled = fence.try_cancel_before_commit()
        if cancelled is not None:
            return cancelled
        # Fence is held only transiently here, but that window rides SessionDB write
        # patience (seconds). 25ms keeps sub-tick latency without a 1kHz spin.
        time.sleep(0.025)


def _release_cancelled_worker(
    future: Any, fence: CompressionCommitFence, *, total_exhausted: bool, ceiling: float
) -> None:
    """Idle-timeout unwind: free the worker's durable lease via the holder-qualified hook.
    Total-ceiling only: bounded grace for the worker to exit (it checks the fence between provider phases; an
    uninterruptible call is orphaned). Idle-stall skips the join: the worker is hung, the fallback needs a
    prompt return, the fence guards."""
    if total_exhausted:
        grace = min(_CANCELLED_WORKER_TEARDOWN_GRACE_SECONDS, ceiling)
        if _join_cancelled_worker(future, grace):
            # Worker provably exited: no provider call can outlive this attempt, so lease
            # retention is unneeded and a retry cannot overlap.
            fence.allow_cancelled_lock_release()
        else:
            logger.warning(
                "Cancelled compression worker did not exit within %.1fs "
                "grace — orphaning it behind the poison fence (late result will be discarded); retaining the session "
                "compression lease until it exits so no new attempt overlaps it", grace,
            )
    fence.release_cancelled_compression_lock()


def run_compress_context_with_progress_timeout(
    *, worker: Callable[[CompressionCommitFence], Tuple[list, str]], messages: list,
    system_prompt_fallback: Any, idle_timeout_seconds: float, total_ceiling_seconds: float,
    on_timeout: Optional[Callable[[float, float, float], None]] = None,
    on_timeout_cause: Optional[Callable[[bool, bool], None]] = None,
    on_commit_overrun: Optional[Callable[[float, float], None]] = None,
    fence: Optional[CompressionCommitFence] = None, telemetry_agent: Any = None, stall_fallback: bool = True,
    new_fence: Optional[Callable[[], CompressionCommitFence]] = None,
) -> Tuple[list, str]:
    """Run ``worker(fence)`` under a sync progress-aware (idle + ceiling) timeout.
    Budgets bound the PRE-commit phase only: an admitted commit always completes (overrun logged, surfaced
    once via ``on_commit_overrun``). A pre-commit cancel returns ``(messages, system_prompt_fallback)`` (lazy
    callable), detaching the worker; a stall first retries the chain once on ``new_fence``, then on_timeout"""
    if idle_timeout_seconds <= 0:
        raise ValueError(
            "run_compress_context_with_progress_timeout requires "
            "idle_timeout_seconds > 0; call compress_context directly to disable"
        )

    def _resolve_fallback_prompt() -> str:
        return system_prompt_fallback() if callable(system_prompt_fallback) else system_prompt_fallback

    ceiling = max(float(total_ceiling_seconds), float(idle_timeout_seconds))
    idle = float(idle_timeout_seconds)
    fence = fence if fence is not None else CompressionCommitFence()
    fence.set_total_ceiling_seconds(ceiling)
    # Sync mirror of gateway hygiene's run_in_executor + wait_for loop: offload,
    # poll idle budget + ceiling, fence-cancel on timeout so no late commit lands.
    from tools.thread_context import propagate_context_to_thread
    executor = _get_compress_timeout_executor()
    # Refuse rather than queue when the pool is full: a queued job would wait out
    # its budget unstarted and run stale later. Skip compression this cycle.
    # A queued job would silently wait out its whole budget without starting and stay eligible to run as a
    # stale cancelled job when a worker recovers. Fail fast: continue without compression this cycle. See
    # #76354.
    if not _try_admit_compression_job():
        logger.warning(
            "Context compression pool saturated (%d workers busy) — refusing new compression this cycle and continuing without "
            "compression. Wedged workers are fence-cancelled and free their "
            "slot when they return; if this persists, check the summary provider health.",
            _COMPRESS_EXECUTOR_MAX_WORKERS,
        )
        # Saturation refusals must hit the same telemetry stream as other failures, or
        # a wedged pool looks like compression simply stopped being attempted.
        if telemetry_agent is not None:
            _emit_compression_attempt_telemetry(
                telemetry_agent, started_at=time.monotonic(), commit_status="aborted", split_status="aborted",
                failure_class="pool_saturated",
            )
        return messages, _resolve_fallback_prompt()

    def _fence_gated_worker(worker_fence: CompressionCommitFence):
        # An admitted job may start after the host stopped waiting; check the fence
        # BEFORE summary work so a stale job never burns an LLM call.
        if worker_fence.deadline_exceeded:
            raise concurrent.futures.TimeoutError("compression deadline expired before worker start")
        if worker_fence.is_cancelled:
            logger.info("Skipping stale compression job: fence cancelled before start")
            return messages, ""
        return worker(worker_fence)

    # Bare pool workers start with an empty ContextVar map; propagate the
    # parent conversation/approval context into the worker.
    try:
        future = executor.submit(propagate_context_to_thread(_fence_gated_worker), fence)
    except BaseException:
        _release_compression_admission()
        raise
    future.add_done_callback(_release_compression_admission)
    wait_started = time.monotonic()
    # EVERY host unwind must revoke commit admission or a detached worker could
    # later mutate durable state; handled_exit marks paths that settle it themselves
    handled_exit = False
    try:
        settled, result = _await_worker_within_budget(
            future, fence, idle=idle, ceiling=ceiling, wait_started=wait_started
        )
        if settled:
            handled_exit = True
            return result

        # F6: a not-yet-started future must not linger as a stale queued job.
        # cancel() is a no-op for a running worker (fence handles that path).
        future.cancel()
        total_exhausted = time.monotonic() - wait_started >= ceiling or fence.deadline_exceeded
        # #97488 teardown (total-ceiling path only): give the cancelled worker a bounded grace to actually
        # exit before this host moves on. The worker checks the poison fence between provider phases, so a
        # cooperative worker exits quickly; an uninterruptible provider call is orphaned behind the fence
        # after the grace elapses (its late result is discarded and cannot touch session state). The
        # idle-stall path intentionally skips the join: its worker is by definition silent/hung, the
        # stall-fallback retry below needs a prompt host return (pinned by the #76354 S3 latency contract),
        # and the fence poison + attempt-generation supersession already protect state against its late
        # unwind.
        if total_exhausted:
            # A total-ceiling candidate may be unwinding a healthy provider call; keep its
            # lease until it exits so no other attempt overlaps the unchanged source.
            fence.retain_compression_lock_until_worker_done()
        if on_timeout_cause is not None:
            with _swallow('compress_context timeout-cause callback failed', exc_info=True):
                on_timeout_cause(total_exhausted, fence.progress_observed)
        if not _cancel_or_join_worker(fence):
            result = _await_in_flight_commit(
                future, ceiling=ceiling, wait_started=wait_started, on_commit_overrun=on_commit_overrun
            )
            handled_exit = True
            return result

        # Idle-timeout: cancel won pre-commit. Also free the worker's durable lease via
        # the holder-qualified hook so a NEW compressor can acquire at once (no ABA).
        handled_exit = True
        _release_cancelled_worker(future, fence, total_exhausted=total_exhausted, ceiling=ceiling)
        waited = time.monotonic() - wait_started
        # #76354 S3 analogue for this wait: charge the idle budget from the LAST PROGRESS event, not from
        # the start of this wait slice. Waiting a full ``idle`` after progress that landed early in the
        # previous slice would allow silence to approach 2x the budget.
        since_progress = fence.seconds_since_progress()
        # Lease is free, so run the fallback BEFORE on_timeout: that callback records
        # the summary-failure cooldown, which would no-op the retry's summary call.
        if stall_fallback:
            recovered = _retry_compression_on_fallback_chain(
                worker=worker, messages=messages, system_prompt_fallback=system_prompt_fallback,
                idle_timeout_seconds=idle, total_ceiling_seconds=ceiling, on_commit_overrun=on_commit_overrun,
                on_timeout_cause=on_timeout_cause, telemetry_agent=telemetry_agent, new_fence=new_fence,
            )
            if recovered is not None:
                return recovered
        if on_timeout is not None:
            with _swallow('compress_context timeout callback failed', exc_info=True):
                on_timeout(idle, waited, since_progress)
        else:
            logger.warning(
                "Context compression made no progress for %.1fs (total wait %.1fs, ceiling %.1fs); continuing without "
                "compression", since_progress, waited, ceiling,
            )
        # Leave the future on the shared pool: fence cancel won, so a late
        # commit cannot land (same detachment model as gateway hygiene).
        return messages, _resolve_fallback_prompt()
    finally:
        if not handled_exit:
            # Any unwind while waiting: revoke commit admission and release the worker's
            # lease before the host unwinds, so the detached worker can never publish.
            fence.revoke_commit_admission()


class CompressionCheckpointUnavailable(RuntimeError):
    """Raised when required durable pre-compress checkpointing is unavailable."""


def _checkpoint_blocked(reason: str) -> CompressionCheckpointUnavailable:
    return CompressionCheckpointUnavailable(
        f"BLOCKED_MISSING_PREREQUISITE: required pre-compress checkpoint unavailable: {reason}"
    )


def _lock_api_is_absent_on_session_db(lock_db: Any) -> bool:
    """Whether the live in-memory SessionDB class structurally predates locks.
    Only the exact old ``hermes_state.SessionDB`` class (hot-reload skew) may fail open; proxies, lookalikes,
    non-callables and descriptor failures fail closed."""
    try:
        from hermes_state import SessionDB
        missing = object()
        return (
            type(lock_db) is SessionDB
            and inspect.getattr_static(SessionDB, "try_acquire_compression_lock", missing) is missing
        )
    except Exception:
        return False


def _refresh_persisted_compression_guards(compressor: Any, *, include_cooldown: bool = True) -> None:
    """Refresh durable automatic-compression guards on a built-in compressor."""
    method_calls = [("_load_fallback_compression_streak", {}), ("_load_ineffective_compression_count", {})]
    if include_cooldown:
        method_calls.insert(0, ("get_active_compression_failure_cooldown", {"refresh": True}))
    for method_name, kwargs in method_calls:
        method = getattr(type(compressor), method_name, None)
        if not callable(method):
            continue
        try:
            method(compressor, **kwargs)
        except Exception as exc:
            logger.debug("compression guard refresh failed (%s): %s", method_name, exc)


def _session_was_rotated_by_compression(session_db: Any, session_id: str) -> bool:
    """Return whether another path already rotated this compression parent."""
    getter = getattr(type(session_db), "get_session", None)
    if not callable(getter):
        return False
    session = getter(session_db, session_id)
    return bool(session and session.get("ended_at") is not None and session.get("end_reason") == "compression")


def _emit_compression_attempt_telemetry(
    agent: Any, *, started_at: float, commit_status: str, split_status: str, failure_class: str | None = None,
    commit_started_at: float | None = None,
) -> None:
    """Emit one content-free JSON log line for a compression attempt."""
    with _swallow('failed to emit compression attempt telemetry: %s'):
        compressor = agent.context_compressor
        telemetry = getattr(compressor, "_last_compression_telemetry", None)
        if not isinstance(telemetry, dict):
            telemetry = {}
        payload = dict(telemetry)
        payload.setdefault("event", "compression_attempt")
        payload.setdefault("attempt_id", getattr(agent, "_compression_attempt_id", "") or uuid.uuid4().hex)
        payload.setdefault("session_id", getattr(agent, "session_id", "") or "")
        payload.update(
            total_duration_ms=int((time.monotonic() - started_at) * 1000), commit_status=commit_status,
            split_status=split_status,
        )
        if commit_started_at is not None:
            telemetry["commit_ms"] = payload["commit_ms"] = max(0, int((time.monotonic() - commit_started_at) * 1000))
        if failure_class:
            payload["failure_class"] = failure_class
        payload.setdefault("chunking", False)
        payload.setdefault("chunk_count", 0)
        payload["fallback_used"] = bool(
            payload.get("fallback_used")
            or getattr(compressor, "_last_summary_fallback_used", False)
            or getattr(compressor, "_last_aux_model_failure_model", None)
        )
        logger.info(
            "context compression attempt telemetry: %s", json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )


def _existing_system_prompt(agent: Any, system_message: str) -> str:
    """Cached system prompt, or a fresh build when nothing is cached (abort paths)."""
    return getattr(agent, "_cached_system_prompt", None) or agent._build_system_prompt(system_message)


def _emit_aborted_attempt_telemetry(agent: Any, started_at: float, failure_class: str | None) -> None:
    _emit_compression_attempt_telemetry(
        agent, started_at=started_at, commit_status="aborted", split_status="aborted", failure_class=failure_class
    )


def _restore_messages_snapshot(messages: list, snapshot: Optional[list]) -> None:
    """Put the pre-compression deep snapshot back into the live list if it drifted."""
    if snapshot is not None and messages != snapshot:
        messages[:] = copy.deepcopy(snapshot)


def _restore_prune_rearm_tokens(compressor: Any, snapshot: dict) -> None:
    """Restore ONLY the prune runway from the attempt snapshot.
    compress() zeroes it in memory while the durable copy only clears on a successful commit; a kept
    transcript keeps its cached prefix, and 0 would let the next prune break that cache."""
    if "_proactive_prune_rearm_tokens" in snapshot:
        compressor._proactive_prune_rearm_tokens = snapshot["_proactive_prune_rearm_tokens"]


def compression_skipped_due_to_lock(agent: Any) -> bool:
    """Type-pinned read of the per-session lock-skip signal.
    ``agent._compression_skipped_due_to_lock`` is a holder string or ``True`` when a pass no-oped because the
    lock was held, ``None`` otherwise. Pinning avoids MagicMock auto-attributes hijacking mocked agents into
    the lock-skip branch."""
    _sig = getattr(agent, "_compression_skipped_due_to_lock", None)
    return _sig is True or isinstance(_sig, str)


def _get_context_compression_timeout_state(
    agent: Any, *, create: bool
) -> Optional[Tuple[Any, Optional[threading.local]]]:
    """Return the stable lock and thread-local timeout state for an agent."""
    try:
        attributes = vars(agent)
    except TypeError:
        return None
    lock = attributes.setdefault("_context_compression_timeout_state_lock", threading.Lock())
    with lock:
        state = attributes.get("_context_compression_timeout_state")
        if create and not isinstance(state, threading.local):
            state = threading.local()
            attributes["_context_compression_timeout_state"] = state
        return lock, state if isinstance(state, threading.local) else None


def _set_context_compression_timeout_outcome(agent: Any, timed_out: bool) -> None:
    """Write this thread's owned-compression timeout outcome.
    The ``agent._last_compression_timed_out`` mirror stays authoritative for minimal agent doubles that do not
    support ``vars()``."""
    lock, state = _get_context_compression_timeout_state(agent, create=True) or (None, None)
    if state is None:
        agent._last_compression_timed_out = timed_out
        return
    with lock:
        state.timed_out = timed_out
        agent._last_compression_timed_out = timed_out


def reset_context_compression_timeout_outcome(agent: Any) -> None:
    """Clear the current thread's owned-compression timeout outcome."""
    _set_context_compression_timeout_outcome(agent, False)


def mark_context_compression_timed_out(agent: Any) -> None:
    """Mark the current owned compression as host-timed-out."""
    _set_context_compression_timeout_outcome(agent, True)


def context_compression_timed_out(agent: Any) -> bool:
    """Return whether this thread's owned compression hit its host timeout.
    Thread-local so overlapping automatic/manual entrypoints cannot hide each other's timeout; attribute
    fallback for minimal doubles; reads type-pinned."""
    locked_state = _get_context_compression_timeout_state(agent, create=False)
    if locked_state is not None:
        lock, state = locked_state
        with lock:
            if isinstance(state, threading.local):
                return getattr(state, "timed_out", None) is True
    return getattr(agent, "_last_compression_timed_out", None) is True


def _automatic_compression_gate_blocks(agent: Any, bypass_cooldown: bool, *, include_cooldown: bool = True) -> bool:
    """Refresh durable guards, then evaluate the compressor's automatic breaker gate.
    ``bypass_cooldown`` ignores the cooldown when the gate accepts ``ignore_cooldown`` (engines predating it get the
    legacy no-argument call). When blocked, the transient-block signal is published for automatic-path consumers.
    """
    compressor = agent.context_compressor
    _refresh_persisted_compression_guards(compressor, include_cooldown=include_cooldown)
    blocked = getattr(type(compressor), "_automatic_compression_blocked", None)
    if not callable(blocked):
        return False
    accepts = False
    if bypass_cooldown:
        with contextlib.suppress(TypeError, ValueError):
            accepts = "ignore_cooldown" in inspect.signature(blocked).parameters
    result = bool(blocked(compressor, ignore_cooldown=True) if accepts else blocked(compressor))
    if result:
        _mark_compression_blocked_transient(agent, compressor)
    return result


def compression_blocked_transiently(agent: Any) -> bool:
    """Type-pinned read of the transient-block signal.
    Set when an automatic pass no-ops on a TRANSIENT guard (summary-failure cooldown or structural backoff).
    Consumers must defer, not count it toward ``compression_exhausted``, or an overflow auto-reset wipes a
    session that was merely cooling down. The permanent ``ineffective`` breaker never sets it.

    See #97488.
    Consumers (the overflow-recovery loops in ``conversation_loop``) must treat such a no-op as a temporary
    defer, NOT as evidence the session is incompressible: counting it toward ``compression_exhausted`` lets
    a real upstream ``context_length_exceeded`` auto-reset (wipe) a session whose compression was merely
    cooling down (#97488). The permanent ``ineffective`` breaker intentionally does NOT set this signal — a
    genuinely incompressible session must still be able to exhaust.
    """
    _sig = getattr(agent, "_compression_blocked_transient", None)
    return isinstance(_sig, str) and bool(_sig)


def _mark_compression_blocked_transient(agent: Any, compressor: Any) -> None:
    """Publish the transient-block signal when the active guard is transient.
    Classification comes from ``_compression_block_reason``: ``cooldown:*`` and ``structural_backoff:*`` are
    transient; ``ineffective`` stays unmarked."""
    reason_fn = getattr(compressor, "_compression_block_reason", None)
    reason = None
    if callable(reason_fn):
        with _swallow('compression block-reason read failed', exc_info=True):
            reason = reason_fn()
    if isinstance(reason, str) and (reason.startswith("cooldown") or reason.startswith("structural_backoff")):
        logger.info(
            "Skipping automatic compression re-entry: transient guard "
            "active (%s, session=%s, last failure: %s) — will retry after "
            "the backoff lapses; /compress forces an immediate retry", reason,
            getattr(agent, "session_id", None) or "none",
            getattr(compressor, "_last_summary_error", None) or "unknown",
        )
        with contextlib.suppress(Exception):
            agent._compression_blocked_transient = reason


def _rebind_session_context(session_id: str) -> None:
    """Point the worker thread's session ContextVar and log context at ``session_id``."""
    try:
        from gateway.session_context import set_current_session_id
        set_current_session_id(session_id)
    except Exception:
        os.environ["HERMES_SESSION_ID"] = session_id
    with contextlib.suppress(Exception):
        from hermes_logging import set_session_context
        set_session_context(session_id)


def _adopt_live_compression_child(
    agent: Any, session_db: Any, parent_session_id: str
) -> Optional[List[Dict[str, Any]]]:
    """Move a stale compression contender onto the live continuation tip.
    Resolve and load first, then mutate the agent, so ambiguous lineage or an unreadable handoff fails closed.
    Uses the transitive ``get_compression_tip`` walk; a tip is adopted only while its row is still live.

    Resolution uses the canonical transitive walk ``get_compression_tip`` so a lineage with >=2 compression
    hops (root -> mid -> tip) recovers to the live tip — the depth-1 ``find_live_compression_child`` lookup
    this used to call finds no live *direct* child in that shape and skipped recovery (#82001). The tip walk
    returns the input id when no continuation exists, and a resolved tip is adopted only while its row is
    still live — both cases fail closed exactly as before.
    """
    resolver = getattr(type(session_db), "get_compression_tip", None)
    row_getter = getattr(type(session_db), "get_session", None)
    loader = getattr(type(session_db), "get_messages_as_conversation", None)
    if not callable(resolver) or not callable(row_getter) or not callable(loader):
        return None
    tip = resolver(session_db, parent_session_id)
    if not tip or str(tip) == str(parent_session_id):
        return None
    child_session_id = str(tip)
    child = row_getter(session_db, child_session_id)
    if not isinstance(child, dict) or child.get("ended_at") is not None:
        return None
    recovered = loader(session_db, child_session_id)
    if not (isinstance(recovered, list) and recovered):
        return None
    # Revalidate after loading: the tip may have rotated or a competing
    # continuation may have appeared between the two DB reads.
    confirmed = resolver(session_db, parent_session_id)
    if not confirmed or str(confirmed) != child_session_id:
        return None
    agent.session_id = child_session_id
    _rebind_session_context(child_session_id)
    agent._session_db_created = True
    if child.get("system_prompt"):
        agent._cached_system_prompt = child["system_prompt"]
    agent._last_flushed_db_idx = len(recovered)
    agent._flushed_db_message_session_id = child_session_id
    agent._flushed_db_message_ids = {id(message) for message in recovered if isinstance(message, dict)}
    on_session_start = getattr(agent.context_compressor, "on_session_start", None)
    if callable(on_session_start):
        with _swallow('context engine compression-child adoption failed: %s'):
            on_session_start(
                child_session_id, boundary_reason="compression", old_session_id=parent_session_id,
                session_db=session_db, platform=getattr(agent, "platform", None) or "cli",
                conversation_id=getattr(agent, "_gateway_session_key", None),
            )
    else:
        bind_state = getattr(agent.context_compressor, "bind_session_state", None)
        if callable(bind_state):
            with contextlib.suppress(Exception):
                bind_state(session_db=session_db, session_id=child_session_id)
    with _swallow('memory manager compression-child adoption failed: %s'):
        if agent._memory_manager:
            agent._memory_manager.on_session_switch(
                child_session_id, parent_session_id=parent_session_id, reset=False, reason="compression"
            )
    return recovered


def _reopen_orphaned_parent(session_db: Any, session_id: str) -> None:
    """Reopen a compression-ended parent that has no continuation and no lease holder."""
    orphan_reopener = getattr(type(session_db), "reopen_orphaned_compression_session", None)
    if not callable(orphan_reopener):
        return
    try:
        if orphan_reopener(session_db, session_id):
            logger.warning("compression recovery: reopened orphaned session=%s with no continuation", session_id)
    except Exception as exc:
        logger.warning("orphaned compression session reopen failed for %s: %s", session_id, exc)


def recover_rotated_compression_session(agent: Any) -> Optional[List[Dict[str, Any]]]:
    """Recover a stale live agent before a new turn writes to its old parent."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None) or ""
    if session_db is None or not session_id:
        return None
    try:
        if not _session_was_rotated_by_compression(session_db, session_id):
            return None
        # Rotation holds the parent lease until the child handoff is durable; wait
        # briefly rather than observe the parent-ended/child-empty intermediate state.
        holder_getter = getattr(session_db, "get_compression_lock_holder", None)
        for attempt in range(21):
            recovered = _adopt_live_compression_child(agent, session_db, session_id)
            if recovered is not None:
                return recovered
            holder = holder_getter(session_id) if callable(holder_getter) else None
            if not holder:
                _reopen_orphaned_parent(session_db, session_id)
                return None
            if attempt == 20:
                return None
            time.sleep(0.05)
        return None
    except Exception as exc:
        logger.warning(
            "compression session recovery failed for session=%s (%s: %s)", session_id, type(exc).__name__, exc
        )
        return None


def _compression_lock_holder(agent: Any) -> str:
    """Build a unique lock holder id: ``pid:tid:agent-instance:uuid``.
    pid+tid tell crashed holders apart in diagnostics; instance id and per-acquire uuid disambiguate
    co-resident agents on one thread or pooled compressions."""
    return f"pid={os.getpid()}:tid={threading.get_ident()}:agent={id(agent):x}:nonce={uuid.uuid4().hex[:8]}"


def _supported_compression_kwargs(
    compress_fn: Any, *, current_tokens: Optional[int], focus_topic: Optional[str], force: bool,
    memory_context: str, bypass_cooldown: bool = False,
) -> dict:
    """Return only compression kwargs accepted by an engine callable.
    Inspecting first keeps older plugin signatures compatible without catching ``TypeError`` and running a
    stateful compressor twice."""
    candidates = {"current_tokens": current_tokens, "focus_topic": focus_topic, "force": force}
    if bypass_cooldown:
        candidates["bypass_cooldown"] = True
    if memory_context:
        candidates["memory_context"] = memory_context
    try:
        parameters = inspect.signature(compress_fn).parameters
    except (TypeError, ValueError):
        # current_tokens has always been in the ContextEngine ABC; use the oldest call
        # shape when the callable has no inspectable signature.
        return {"current_tokens": current_tokens}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return candidates
    return {name: value for name, value in candidates.items() if name in parameters}


class _CompressionActivityHeartbeat:
    """Refresh the agent inactivity tracker while compression blocks in an aux call."""

    def __init__(
        self, agent: Any, interval_seconds: float | None = None, *, emit_client_status: bool = False,
        commit_fence: Optional[CompressionCommitFence] = None,
    ) -> None:
        self._agent = agent
        self._commit_fence = commit_fence
        # Latched once host cancel/timeout wins or a terminal stamp is observed,
        # so a later UNKNOWN rewrite cannot re-arm a detached zombie heartbeat.
        self._suppressed = False
        if interval_seconds is None:
            interval_seconds = getattr(agent, "_compression_activity_heartbeat_interval", 60.0)
        try:
            interval_seconds = float(interval_seconds or 60.0)
            if not math.isfinite(interval_seconds):
                interval_seconds = 60.0
        except (TypeError, ValueError):
            interval_seconds = 60.0
        self._interval_seconds = max(0.1, interval_seconds)
        # Only a compression that opened a VISIBLE compaction phase (the
        # routine start status was emitted) keeps it alive with heartbeats;
        # quiet context engines emit neither (#98371 follow-up).
        self._emit_client_status = emit_client_status
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="compression-activity-heartbeat", daemon=True)

    def start(self) -> "_CompressionActivityHeartbeat":
        # A new compression episode always republishes agent.compression even
        # if a prior timeout/cooldown stamp is still on the agent.
        self._suppressed = False
        self._touch("context compression started", allow_terminal_overwrite=True)
        self._thread.start()
        return self

    def stop(self, desc: str = "context compression completed") -> None:
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)
        # Host timeout already owns the terminal stamp; a detached worker's
        # late stop must not republish agent.compression / "completed".
        if self._should_suppress():
            return
        # Force persist: /compress never hits run_conversation's turn-end clear, so
        # durable labels would stay "in progress" for the 60s persist window.
        self._touch(desc, force_persist=True)

    def _should_suppress(self) -> bool:
        if not self._suppressed and self._commit_fence is not None and self._commit_fence.is_cancelled:
            self._suppressed = True
        return self._suppressed

    def _touch(self, desc: str, *, allow_terminal_overwrite: bool = False, force_persist: bool = False) -> None:
        with _swallow('compression activity heartbeat touch failed', exc_info=True):
            if not allow_terminal_overwrite:
                if self._should_suppress():
                    return
                current = normalize_activity_provenance(getattr(self._agent, "_last_activity_provenance", None))
                if current in _TERMINAL_COMPRESSION_PROVENANCES:
                    self._suppressed = True
                    return
            touch = getattr(self._agent, "_touch_activity", None)
            if callable(touch):
                # Re-check after reading provenance: host may cancel/stamp
                # TIMEOUT between the earlier guard and the write.
                if not allow_terminal_overwrite and self._should_suppress():
                    return
                touch(desc, provenance=ActivityProvenance.AGENT_COMPRESSION, force_persist=force_persist)

    def _emit_progress_status(self) -> None:
        """Re-publish the compacting status so remote transports see progress.

        Compression can stream for minutes with no deltas, tool events, or
        status lines reaching remote transports. Idle-progress watchdogs on
        those clients (e.g. the Android relay app's 180s turn watchdog)
        treat the silence as a dead turn and fire ``session.interrupt`` —
        killing a healthy compression mid-flight and rolling back its work,
        which retriggers on the next prompt and loops forever on sessions
        near the context ceiling (#98371).

        Routed through ``agent._emit_status`` like every other compaction
        status: same "lifecycle" key (the TUI gateway re-tags it to
        ``compacting``; Telegram edits one bubble per key), same chat-platform
        filter, same CLI print path.
        """
        if not self._emit_client_status:
            return
        emit = getattr(self._agent, "_emit_status", None)
        if not callable(emit):
            return
        try:
            emit(COMPACTION_HEARTBEAT_STATUS)
        except Exception:
            logger.debug(
                "status emit error in compression heartbeat", exc_info=True
            )

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            if self._should_suppress():
                return
            self._touch("context compression in progress")
            self._emit_progress_status()


def _direct_messages_for_pre_compress_memory(messages: Any) -> list[dict[str, Any]]:
    """Return direct user/assistant evidence safe for memory checkpointing.
    Summaries, tool rows and system messages are omitted; assistant prose is kept with ``tool_calls``
    stripped, and pure tool-call wrappers are dropped."""
    # Deferred import: context_compressor → turn_context → this module would form an import cycle.
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY
    direct_messages: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"} or message.get(COMPRESSED_SUMMARY_METADATA_KEY):
            continue
        if role == "assistant" and message.get("tool_calls"):
            content = message.get("content")
            has_prose = bool(content.strip() if isinstance(content, str) else content)
            if not has_prose:
                continue
            message = {k: v for k, v in message.items() if k != "tool_calls"}
        direct_messages.append(message)
    return direct_messages


class _CompressionLockLeaseRefresher:
    def __init__(
        self, db: Any, session_id: str, holder: str, ttl_seconds: float, refresh_interval_seconds: float | None = None
    ) -> None:
        self._db = db
        self._session_id = session_id
        self._holder = holder
        self._ttl_seconds = ttl_seconds
        if refresh_interval_seconds is None:
            refresh_interval_seconds = max(1.0, min(60.0, ttl_seconds / 2.0))
        self._refresh_interval_seconds = max(0.1, float(refresh_interval_seconds))
        # Tolerate transient refresh failures for at most one TTL so the lease cannot
        # outlive its TTL; floor 1 so interval >= ttl still tolerates one blip.
        self._max_consecutive_failures = max(1, int(self._ttl_seconds / self._refresh_interval_seconds))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="compression-lock-refresh", daemon=True)

    def start(self) -> "_CompressionLockLeaseRefresher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        # join() timing out mid-UPDATE is safe: daemon thread, and a late refresh on a
        # released lock is a rowcount-0 no-op. stop() does not guarantee quiescence.
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # A single falsy refresh (transient DB blip) must not kill the lease; only
        # ttl/interval consecutive failures do, so a stuck refresher never outlives TTL.
        consecutive_failures = 0
        # Refresh immediately: work between try_acquire() and start() is charged to the
        # first lease, so on a short TTL it could expire before tick #1.
        first = True
        while first or not self._stop.wait(self._refresh_interval_seconds):
            if first:
                first = False
                if self._stop.is_set():
                    break
            refreshed = False
            with _swallow("compression lock refresh raised: %s"):
                refreshed = self._db.refresh_compression_lock(self._session_id, self._holder, ttl_seconds=self._ttl_seconds)
            if refreshed:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures >= self._max_consecutive_failures:
                logger.debug(
                    "compression lock refresh failed %d times in a row; stopping lease refresher for session %s",
                    consecutive_failures, self._session_id,
                )
                break


def _lower_threshold_to_aux_context(
    agent: Any, *, aux_model: str, aux_context: int, aux_provider: str, aux_base_url: str
) -> None:
    """Lower the live threshold to the aux model's window and tell the user how to fix config.
    The summariser sends one user prompt (no system/tools), so threshold == aux_context is safe.
    tail_token_budget and threshold_percent are kept in lockstep (as update_model does) or the 1.5x tail
    ceiling exceeds the trigger and re-fires."""
    compressor = agent.context_compressor
    old_threshold = compressor.threshold_tokens
    new_threshold = compressor.threshold_tokens = aux_context
    summary_target_ratio = getattr(compressor, "summary_target_ratio", None)
    if isinstance(summary_target_ratio, (int, float)):
        compressor.tail_token_budget = int(new_threshold * summary_target_ratio)
    main_ctx = compressor.context_length
    if main_ctx:
        compressor.threshold_percent = new_threshold / main_ctx
    safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
    # Mirror the compressor's threshold math (percent floor, output reservation, 64K floor): a suggestion it
    # would override is silently ignored and this warning reappears every session. External engines: keep it plain.
    # The "lower the threshold" suggestion must survive the built-in trigger recomputation (#67422):
    # _effective_threshold_percent() raises sub-75% values back up for main windows under 512K, and
    # _compute_threshold_tokens() further applies the output-token reservation, the 64K floor, and the
    # degenerate-window guard. Recommending a value those would override is silently ignored and this
    # warning would reappear every session — so mirror the compressor's own math and only offer the option
    # when the recomputed trigger actually fits the auxiliary model's context.
    from agent.context_compressor import ContextCompressor as _CC
    recomputed_threshold = None
    if main_ctx and isinstance(compressor, _CC):
        recomputed_threshold = _CC._compute_threshold_tokens(
            main_ctx, _CC._effective_threshold_percent(main_ctx, safe_pct / 100),
            getattr(compressor, "max_tokens", None),
        )
    threshold_suggestion_viable = recomputed_threshold is None or recomputed_threshold <= aux_context
    # "model (provider)" labels for both sides; empty/"auto" provider falls back to the client's base_url hostname.
    _main_model = getattr(agent, "model", "") or "?"
    _main_provider = getattr(agent, "provider", "") or ""
    _aux_provider_label = aux_provider if aux_provider and aux_provider != "auto" else ""
    if not _aux_provider_label:
        try:
            from urllib.parse import urlparse
            _aux_provider_label = urlparse(aux_base_url).hostname or aux_base_url
        except Exception:
            _aux_provider_label = aux_base_url or "auto"
    _main_label = f"{_main_model} ({_main_provider})" if _main_provider else _main_model
    _aux_label = f"{aux_model} ({_aux_provider_label})"
    msg = (
        f"⚠ Compression model {_aux_label} context is {aux_context:,} tokens, but the main model "
        f"{_main_label}'s compression threshold was {old_threshold:,} tokens. "
        f"Auto-lowered this session's threshold to {new_threshold:,} tokens so compression can run.\n"
    )
    if threshold_suggestion_viable:
        msg += (
            f"  To make this permanent, edit config.yaml — either:\n  1. Use a larger compression model:\n"
            f"       auxiliary:\n         compression:\n           model: <model-with-{old_threshold:,}+-context>\n"
            f"  2. Lower the compression threshold:\n       compression:\n         threshold: 0.{safe_pct:02d}"
        )
    else:
        msg += (
            f"  To make this permanent, use a larger compression model in config.yaml:\n       auxiliary:\n"
            f"         compression:\n           model: <model-with-{old_threshold:,}+-context>\n"
            f"  (Lowering compression.threshold cannot help here — with {_main_label}'s {main_ctx:,}-token window, "
            f"Hermes's small-context floor and output reservation would recompute the trigger to "
            f"{recomputed_threshold:,} tokens, still above the compression model's {aux_context:,}.)"
        )
    agent._compression_warning = msg
    agent._emit_status(msg)
    logger.warning(
        "Auxiliary compression model %s has %d token context, below the main model's compression threshold of %d "
        "tokens — auto-lowered session threshold to %d to keep compression working.", aux_model, aux_context,
        old_threshold, new_threshold,
    )


def check_compression_model_feasibility(agent: Any) -> None:
    """Warn at session start if the aux compression context is below the threshold.
    Called from ``AIAgent.__init__`` (CLI sees it via ``_vprint``); the gateway wires ``status_callback``
    later, so ``replay_compression_warning`` resends it."""
    if not agent.compression_enabled:
        return
    try:
        from agent.auxiliary_client import (
            _resolve_task_provider_model, _try_configured_fallback_for_unavailable_client,
            get_text_auxiliary_client,
        )
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH, get_model_context_length
        # Provider may be "auto"; fall back to the client's base_url hostname so the
        # user can tell where the compression model is actually called.
        try:
            _aux_cfg_provider, _, _, _, _ = _resolve_task_provider_model("compression")
        except Exception:
            _aux_cfg_provider = ""
        client, aux_model = get_text_auxiliary_client("compression", main_runtime=agent._current_main_runtime())
        if client is None or not aux_model:
            fb_client, fb_model, fb_label = _try_configured_fallback_for_unavailable_client(
                "compression", _aux_cfg_provider
            )
            if fb_client is not None and fb_model:
                client, aux_model = fb_client, fb_model
                if "(" in fb_label and fb_label.endswith(")"):
                    _aux_cfg_provider = fb_label.rsplit("(", 1)[1][:-1]
        if client is None or not aux_model:
            if _aux_cfg_provider and _aux_cfg_provider != "auto":
                msg = (
                    "⚠ Configured auxiliary compression provider "
                    f"'{_aux_cfg_provider}' is unavailable — context "
                    "compression will drop middle turns without a summary. "
                    "Check auxiliary.compression in config.yaml and reauthenticate that provider."
                )
            else:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "Auxiliary compression client unavailable (%s) — summaries "
                "disabled; compression will drop middle turns without a "
                "summary. configured_provider=%r aux_model=%r session=%s",
                (
                    "configured provider failed to resolve"
                    if (_aux_cfg_provider and _aux_cfg_provider != "auto")
                    else "no provider configured"
                ),
                _aux_cfg_provider or "",
                aux_model or "",
                getattr(agent, "session_id", None) or "none",
            )
            return
        aux_base_url = str(getattr(client, "base_url", ""))
        # client.api_key may be a callable (Entra bearer); the resolver only needs a key
        # for live catalogue probes, so pass "" rather than mint a JWT for a lookup.
        _raw_aux_key = getattr(client, "api_key", "")
        aux_api_key = "" if (callable(_raw_aux_key) and not isinstance(_raw_aux_key, str)) else str(_raw_aux_key or "")
        # Resolve each model with its own provider so provider-specific paths (Bedrock table, OpenRouter API)
        # hit the correct client, not the main model's.
        _aux_provider = (
            _aux_cfg_provider if _aux_cfg_provider and _aux_cfg_provider != "auto" else getattr(agent, "provider", "")
        )
        aux_context = get_model_context_length(
            aux_model, base_url=aux_base_url, api_key=aux_api_key,
            config_context_length=getattr(agent, "_aux_compression_context_length_config", None),
            provider=_aux_provider, custom_providers=agent._custom_providers,
        )
        # Aux model must meet MINIMUM_CONTEXT_LENGTH like the main model, else it cannot summarise a full window.
        if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Auxiliary compression model {aux_model} has a context "
                f"window of {aux_context:,} tokens, which is below the "
                f"minimum {MINIMUM_CONTEXT_LENGTH:,} required by Hermes "
                f"Agent.  Choose a compression model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context (set "
                f"auxiliary.compression.model in config.yaml), or set "
                f"auxiliary.compression.context_length to override the "
                f"detected value if it is wrong."
            )
        if aux_context < agent.context_compressor.threshold_tokens:
            _lower_threshold_to_aux_context(
                agent, aux_model=aux_model, aux_context=aux_context, aux_provider=_aux_cfg_provider,
                aux_base_url=aux_base_url,
            )
    except ValueError:
        # Hard rejections (aux below minimum context) must propagate so the session refuses to start.
        raise
    except Exception as exc:
        logger.debug("Compression feasibility check failed (non-fatal): %s", exc)


def replay_compression_warning(agent: Any) -> None:
    """Re-send the stored compression warning through ``status_callback``.
    Called once at the start of the first ``run_conversation()``, when the gateway callback (absent during
    ``__init__``) is finally wired."""
    msg = getattr(agent, "_compression_warning", None)
    if msg and agent.status_callback:
        with contextlib.suppress(Exception):
            agent.status_callback("lifecycle", msg)


def conversation_history_after_compression(
    agent: Any, messages: list, previous_history: Optional[list] = None
) -> Optional[list]:
    """Return the correct flush baseline after a compression boundary.
    Session rotation returns ``None`` so the child gets the full compacted list. In-place compaction returns a
    shallow copy of the already-persisted rows (else the identity flush re-appends them). Aborted/no-op
    attempts keep the baseline: marking all persisted drops unflushed turns; clearing re-appends rows."""
    if bool(getattr(agent, "_last_compression_attempt_recorded", False)):
        attempt_in_place = getattr(agent, "_last_compression_attempt_in_place", None)
        if attempt_in_place is True:
            return list(messages)
        if attempt_in_place is False:
            return None
        return previous_history
    if bool(getattr(agent, "_last_compaction_in_place", False)):
        return list(messages)
    return None


_SYNTHETIC_USER_PREFIXES = (
    "[System: Your previous response was truncated", "[System: The previous response was cut off",
    "[System: Your previous tool call", "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
)


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("content") or "") for part in content if isinstance(part, dict)
        )
    return ""


_SYNTHETIC_USER_FLAGS = (
    "_todo_snapshot_synthetic", "_empty_recovery_synthetic", "_verification_stop_synthetic", "_pre_verify_synthetic",
    "_dropped_toolcall_nudge",
)


def _is_real_user_message(message: Any) -> bool:
    """Distinguish human intent from user-role runtime scaffolding.
    A compaction summary flipped to ``role="user"`` for alternation is scaffolding and must not short-circuit
    anchor restoration."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if any(message.get(flag) for flag in _SYNTHETIC_USER_FLAGS):
        return False
    text = _message_text(message).strip()
    if not text or text.startswith(_SYNTHETIC_USER_PREFIXES):
        return False
    from agent.context_compressor import ContextCompressor
    return not ContextCompressor._is_synthetic_compression_user_turn(message)


_STEER_FALLBACK_OPEN = "[OUT-OF-BAND USER MESSAGE"
_STEER_FALLBACK_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"


def _steer_markers() -> Tuple[str, str]:
    """``(open, close)`` steer markers from prompt_builder, or the stable fallback literals."""
    try:
        from agent.prompt_builder import STEER_MARKER_CLOSE, STEER_MARKER_OPEN
        return STEER_MARKER_OPEN, STEER_MARKER_CLOSE
    except Exception:
        return _STEER_FALLBACK_OPEN, _STEER_FALLBACK_CLOSE


def _message_contains_busy_steer(message: Any) -> bool:
    """Return whether *message* carries a busy-steer marker.
    Steer follow-ups live as markers inside ``role=tool`` results, so they carry user intent that
    ``_is_real_user_message`` alone would miss."""
    text = _message_text(message)
    if not text:
        return False
    open_marker, close_marker = _steer_markers()
    return open_marker in text and close_marker in text


def _extract_steer_text_from_message(message: Any) -> Optional[str]:
    """Extract the inner user text from a steer marker, or None."""
    text = _message_text(message)
    if not text:
        return None
    open_marker, close_marker = _steer_markers()
    start = text.find(open_marker)
    if start == -1:
        # Fallback: marker wording may evolve; look for the stable prefix, then skip to the end of the opening line.
        start = text.find(_STEER_FALLBACK_OPEN)
        if start == -1:
            return None
        nl = text.find("\n", start)
        start = nl + 1 if nl != -1 else start + len(_STEER_FALLBACK_OPEN)
    else:
        start += len(open_marker)
    end = text.find(close_marker, start)
    if end == -1:
        end = text.find(_STEER_FALLBACK_CLOSE, start)
        if end == -1:
            return None
    return text[start:end].strip() or None


def _compressed_has_busy_steer(messages: list) -> bool:
    """Whether *messages* already carries a steer marker in a ``role=tool`` row.
    Only tool rows count, so a summary merely quoting the marker text is not mistaken for live intent."""
    return any(
        isinstance(msg, dict) and msg.get("role") == "tool" and _message_contains_busy_steer(msg) for msg in messages
    )


def _strip_stale_todo_snapshot(content: Any) -> Any:
    """Remove a previously merged todo-snapshot block from message content.
    Snapshots are appended to the trailing user turn, so a surviving header is stale; stripping before
    re-injection prevents accumulation across boundaries."""
    from tools.todo_tool import TODO_INJECTION_HEADER
    if isinstance(content, str):
        idx = content.find(TODO_INJECTION_HEADER)
        if idx == -1:
            return content
        return content[:idx].rstrip()
    if isinstance(content, list):
        cleaned = []
        for part in content:
            text = str(part.get("text") or "") if isinstance(part, dict) and part.get("type") == "text" else ""
            idx = text.find(TODO_INJECTION_HEADER) if text else -1
            if idx == -1:
                cleaned.append(part)
            elif stripped := text[:idx].rstrip():
                cleaned.append({**part, "text": stripped})
        return cleaned
    return content


def _todo_snapshot_is_only_content(content: Any, stripped: Any) -> bool:
    """Return whether stripping the snapshot leaves no structured content.
    Text snapshots trail a string; structured ones occupy their own text part, so only an empty remainder
    proves the row was scaffolding alone. Text extraction is deliberately not used: image, audio and other
    non-text parts must survive."""
    if isinstance(content, str) and isinstance(stripped, str):
        return not stripped.strip()
    if isinstance(content, list) and isinstance(stripped, list):
        return not stripped
    return False


def _replace_message_content(message: dict, content: Any) -> None:
    """Rewrite message content without allowing an old API sidecar to replay."""
    from agent.turn_context import drop_stale_api_content
    message["content"] = content
    drop_stale_api_content(message)


# Compaction re-injects the todo list verbatim but prunes skills to markers, so
# couple them: tell the model to reload pruned skills BEFORE acting on tasks.
# Lives after TODO_INJECTION_HEADER so it strips with the snapshot next time.
_PRUNED_SKILL_RELOAD_NOTICE_HEADER = "[Skills pruned during compression — reload before acting on these tasks]"


def _pruned_skill_reload_notice(compressed: list) -> str:
    """Reload notice for skills whose bodies were pruned, or ``""``.
    Scans ``[SKILL_PRUNED: ...]`` markers in the post-compression transcript; first-seen order, deduplicated,
    capped at ``_MAX_PRUNED_SKILL_MARKERS``."""
    from agent.context_compressor import _MAX_PRUNED_SKILL_MARKERS, _extract_pruned_skill_names
    names: list = []
    for message in compressed:
        if not isinstance(message, dict):
            continue
        for name in _extract_pruned_skill_names(_message_text(message)):
            if name not in names:
                names.append(name)
    del names[_MAX_PRUNED_SKILL_MARKERS:]
    if not names:
        return ""
    calls = "; ".join(f"skill_view(name='{name}')" for name in names)
    return (
        f"{_PRUNED_SKILL_RELOAD_NOTICE_HEADER}\n"
        "The task list above crossed the compression boundary verbatim, but "
        "the skill instructions that governed it were pruned. Before "
        f"executing any preserved task that depends on these skills, reload "
        f"them first: {calls}. After reloading, re-check that each pending "
        "task is still justified — findings recorded before the boundary may have invalidated it."
    )


def _merge_anchor_into_user_message(target: dict, anchor: dict) -> None:
    """Fold the human anchor into an existing user-role scaffolding turn.
    Used only when any insertion would create consecutive user turns. Anchor text leads, scaffolding follows,
    and synthetic flags are cleared."""
    anchor_content = anchor.get("content")
    target_content = target.get("content")
    if isinstance(anchor_content, list) or isinstance(target_content, list):

        def _parts(content: Any) -> list:
            return list(content) if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]

        _replace_message_content(target, _parts(anchor_content) + _parts(target_content))
    else:
        merged = f"{anchor_content or ''}\n\n{target_content or ''}".strip()
        _replace_message_content(target, merged)
    for flag in _SYNTHETIC_USER_FLAGS:
        target.pop(flag, None)


CompressedUserTurnOutcome = Literal["inserted", "merged", "already_present", "placeholder_appended"]


def _insert_real_user_anchor(messages: list, anchor: dict) -> CompressedUserTurnOutcome:
    """Insert the latest human turn without breaking role alternation."""
    from agent.context_compressor import _DB_PERSISTED_MARKER, ContextCompressor

    def _role(msg: Any) -> Optional[str]:
        return msg.get("role") if isinstance(msg, dict) else None

    def _place(index: int) -> CompressedUserTurnOutcome:
        anchor[_DB_PERSISTED_MARKER] = True
        messages.insert(index, anchor)
        return "inserted"

    # Preferred anchor: the summary boundary — first assistant message not preceded
    # by a user turn. Left neighbour is then non-user, right is an assistant.
    for index, message in enumerate(messages):
        if _role(message) == "assistant" and (index == 0 or _role(messages[index - 1]) != "user"):
            return _place(index)
    # Every assistant is user-preceded (or there are none). Appending is safe whenever the transcript does not already
    # end with a user turn. Never merge into a summary either: its prefix must stay at message start for summary
    # detection; repair_message_sequence merges adjacent user turns summary-first.
    if (
        not messages
        or _role(messages[-1]) != "user"
        or ContextCompressor._is_context_summary_content(_message_text(messages[-1]))
    ):
        return _place(len(messages))
    # Trailing user-role scaffolding (e.g. the todo snapshot): merge instead
    # of inserting a consecutive same-role message (#55677 strict templates).
    _merge_anchor_into_user_message(messages[-1], anchor)
    messages[-1][_DB_PERSISTED_MARKER] = True
    return "merged"


def _ensure_compressed_has_user_turn(original_messages: list, compressed: list) -> CompressedUserTurnOutcome:
    """Preserve human intent, not merely a synthetic user-role placeholder."""
    if any(_is_real_user_message(message) for message in compressed) or _compressed_has_busy_steer(compressed):
        return "already_present"
    # Post-commit contract (#98450, mirrors _sync_micro_compact_to_db): archive_and_compact just durably
    # wrote every dict in `compressed` as the new active set, but compress() returned marker-swept COPIES
    # (_strip_persistence_markers, #57491). These exact dict instances become the live message list the
    # caller keeps, so without the stamp the next _persist_session → _flush_messages_to_session_db_unlocked
    # walk treats the whole compacted transcript as unpersisted and re-INSERTs it — the live set doubles on
    # every compaction (~58K → ~512K tokens in production).
    from agent.context_compressor import (
        _INFLIGHT_REPLAY_MERGED_KEY, COMPRESSION_CONTINUATION_USER_CONTENT, _fresh_compaction_message_copy,
    )
    if any(isinstance(message, dict) and message.get(_INFLIGHT_REPLAY_MERGED_KEY) for message in compressed):
        # The in-flight request was restated onto the summary carrier (#100818); an anchor would duplicate it.
        return "already_present"
    # One reversed scan over BOTH kinds: scanning steer then user would let an older
    # consumed steer outrank a newer real user request and replay it.
    # One reversed positional scan: the anchor is whichever intent-bearing row is LAST in the original
    # transcript — a real ``role=user`` turn or a steer marker riding inside a ``role=tool`` result. See
    # #100053.
    for message in reversed(original_messages):
        if _is_real_user_message(message):
            return _insert_real_user_anchor(compressed, _fresh_compaction_message_copy(message))
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        steer_text = _extract_steer_text_from_message(message)
        if steer_text:
            return _insert_real_user_anchor(compressed, {"role": "user", "content": steer_text})
    from agent.message_metadata import append_message
    append_message(compressed, {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT})
    return "placeholder_appended"


def _messages_match_scoped_identity(left: Any, right: Any) -> bool:
    """Compare the live turn identity we care about for rotation stamping."""
    if (
        not isinstance(left, dict)
        or not isinstance(right, dict)
        or left.get("role") != right.get("role")
        or left.get("content") != right.get("content")
    ):
        return False
    left_timestamp = left.get("timestamp")
    right_timestamp = right.get("timestamp")
    if left_timestamp is not None and right_timestamp is not None:
        return left_timestamp == right_timestamp
    return True


def _stamp_scoped_twins(targets: list, source: dict, *, exact_counts_stamped: bool = False) -> None:
    """Stamp ``_db_persisted`` on every unstamped scoped twin of ``source`` in ``targets``.
    Exact-timestamp twins are preferred: when the source carries a timestamp and any exact twin was stamped
    (or, with ``exact_counts_stamped``, merely exists), the broad scoped pass is skipped so a content-equal
    old duplicate is left alone."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    source_timestamp = source.get("timestamp")
    exact_hit = False
    if source_timestamp is not None:
        for target in targets:
            if (
                not isinstance(target, dict)
                or target.get("timestamp") != source_timestamp
                or not _messages_match_scoped_identity(target, source)
            ):
                continue
            if target.get(_DB_PERSISTED_MARKER):
                exact_hit = exact_hit or exact_counts_stamped
                continue
            target[_DB_PERSISTED_MARKER] = True
            exact_hit = True
        if exact_hit:
            return
    for target in targets:
        if (
            isinstance(target, dict)
            and not target.get(_DB_PERSISTED_MARKER)
            and _messages_match_scoped_identity(target, source)
        ):
            target[_DB_PERSISTED_MARKER] = True


_PENDING_CONTEXT_ENGINE_NOTIFICATION = "_pending_context_engine_compression_notification"


def _notify_context_engine_compression_complete(agent: Any, *, new_session_id: str, old_session_id: str) -> bool:
    """Notify the active context engine after a durable compression commit."""
    # Opt-in relay session-span segmentation. Observer semantics — failure must
    # never undo or delay the committed compression.
    with _swallow('relay segment rotation notification failed', exc_info=True):
        from agent import relay_runtime
        relay_runtime.SESSION_COORDINATOR.notify_session_compacted(
            profile_key=relay_runtime.current_profile_key(), session_id=new_session_id, old_session_id=old_session_id
        )
    callback = getattr(agent.context_compressor, "on_session_start", None)
    if not callable(callback):
        return False
    try:
        callback(
            new_session_id, boundary_reason="compression", old_session_id=old_session_id,
            platform=getattr(agent, "platform", None) or "cli",
            conversation_id=getattr(agent, "_gateway_session_key", None),
        )
        return True
    except Exception:
        # Context-engine hooks are observers. A callback failure must not undo
        # history that the core or an outer host transaction already committed.
        logger.debug("context engine on_session_start (compression) failed", exc_info=True)
        return False


def _queue_context_engine_compression_notification(agent: Any, *, new_session_id: str, old_session_id: str) -> None:
    """Stage exactly one existing hook call for an outer host transaction."""
    if callable(getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)):
        raise RuntimeError("a compression notification is already pending")

    def _notify() -> bool:
        return _notify_context_engine_compression_complete(
            agent, new_session_id=new_session_id, old_session_id=old_session_id
        )

    setattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, _notify)


def finalize_context_engine_compression_notification(agent: Any, *, committed: bool) -> bool:
    """Emit or discard a deferred notification; repeated calls are no-ops."""
    pending = getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)
    setattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)
    if not committed or not callable(pending):
        return False
    return bool(pending())


class _CompactionLifecycle:
    """Owns the one-shot terminal edge of the compaction status lifecycle.
    ``commit_status`` is rebound to "committed" only on success and read at ``complete()`` time, so abort
    paths keep the terminal edge suppressed."""

    def __init__(self, agent: Any, status_emitted: bool) -> None:
        self._agent = agent
        self.status_emitted = status_emitted
        self._done_emitted = False
        self.commit_status = "aborted"

    def complete(self, *, force_terminal: bool = False) -> None:
        if self._done_emitted:
            return
        self._done_emitted = True
        # Suppressed start → no terminal edge. Non-compacting aborts (lock contender,
        # cancelled fence) opt in via force_terminal so clients can retire their phase.
        # Failure warnings go through _emit_warning and are never suppressed here.
        if self.status_emitted and (self.commit_status == "committed" or force_terminal):
            _emit_compaction_done(self._agent)


class _CompressionLease:
    """The per-attempt durable compression lock plus its lifecycle plumbing.
    ``holder`` is None when no durable lock is owned (legacy DB, no session db); ``watermark`` is MAX(id) of
    active rows at lease start (None = archive everything, no concurrent-tail preservation this cycle)."""

    def __init__(
        self, agent: Any, *, db: Any, sid: str, ttl: float, refresh_interval: Any,
        commit_fence: Optional[CompressionCommitFence], lifecycle: _CompactionLifecycle,
    ) -> None:
        self._agent = agent
        self.db = db
        self.sid = sid
        self.ttl = ttl
        self._refresh_interval = refresh_interval
        self._commit_fence = commit_fence
        self._lifecycle = lifecycle
        self.holder: Optional[str] = None
        self.watermark: Optional[int] = None
        self._refresher: Optional[_CompressionLockLeaseRefresher] = None
        self._released = False
        self._release_guard = threading.Lock()
        # Fence lock acquisition + release-hook publication together so a host timeout
        # cannot win between acquiring the lock and having a way to release it.
        self._lock_setup_entered = False

    @property
    def status_emitted(self) -> bool:
        """True when the routine compaction start status was shown (heartbeats may follow it)."""
        return self._lifecycle.status_emitted

    def begin_lock_setup(self) -> bool:
        if self._commit_fence is None:
            return True
        self._lock_setup_entered = self._commit_fence.begin_lock_setup()
        return self._lock_setup_entered

    def finish_lock_setup(self) -> None:
        if not self._lock_setup_entered or self._commit_fence is None:
            return
        self._lock_setup_entered = False
        self._commit_fence.finish_lock_setup()

    def start_refresher(self) -> None:
        if self.holder is None:
            return
        candidate = _CompressionLockLeaseRefresher(self.db, self.sid, self.holder, self.ttl, self._refresh_interval)
        # Cancellation may release the holder between hook publication and this
        # start; serialize with the release path so no refresher starts on a freed lock.
        with self._release_guard:
            if not self._released:
                self._refresher = candidate.start()

    def release_holder_only(self) -> None:
        """Stop this holder's refresher and release only its durable lock.
        Holder-qualified and idempotent: safe for the host after a timeout because a newer holder's lease can
        never be deleted by this stale release."""
        with self._release_guard:
            if self._released:
                return
            self._released = True
            if getattr(self._agent, "_active_compression_lock_holder", None) == self.holder:
                self._agent._active_compression_lock_holder = None
            if self._refresher is not None:
                with _swallow('compression lock refresher stop failed: %s'):
                    self._refresher.stop()
            if self.db is not None and self.sid and self.holder:
                with _swallow('compression lock release failed: %s'):
                    self.db.release_compression_lock(self.sid, self.holder)

    def release(self) -> None:
        """Finish lifecycle cleanup and release the OLD session lock once."""
        try:
            self._lifecycle.complete()
        finally:
            try:
                self.release_holder_only()
            finally:
                try:
                    if self._commit_fence is not None:
                        self._commit_fence.clear_cancelled_lock_release(self.release_holder_only)
                finally:
                    self.finish_lock_setup()


def _resolve_lock_api(lock_db: Any) -> Tuple[Any, Optional[Exception]]:
    """Return ``(try_acquire_compression_lock, lookup_error)`` for ``lock_db``.
    ``(None, None)`` = no db or legacy SessionDB without the lock API (fail open); ``(None, exc)`` = lookup
    itself failed (caller fails closed)."""
    if lock_db is None:
        return None, None
    try:
        if _lock_api_is_absent_on_session_db(lock_db):
            return None, None
        try_acquire = lock_db.try_acquire_compression_lock
    except Exception as exc:
        return None, exc
    if not callable(try_acquire):
        return None, TypeError("compression lock API is present but not callable")
    return try_acquire, None


def _abort_lease(
    agent: Any, lifecycle: _CompactionLifecycle, system_message: str, attempt_started_at: float,
    failure_class: str, prompt: Optional[str] = None,
) -> Tuple[None, str]:
    """Sit-out return for lease acquisition: prompt, aborted telemetry, terminal status edge."""
    if prompt is None:
        prompt = _existing_system_prompt(agent, system_message)
    _emit_aborted_attempt_telemetry(agent, attempt_started_at, failure_class)
    lifecycle.complete(force_terminal=True)
    return None, prompt


def _try_acquire_durable_lock(lease: _CompressionLease, try_acquire: Any, commit_fence: Any) -> bool:
    """Acquire the durable lock for ``lease.holder`` and capture the start watermark.
    Watermark = MAX(id) of active rows at START: appends aren't blocked during summary; later rows are
    concurrent tail that archive_and_compact re-sequences. Capture is safety-additive (fallback archives
    everything), so its failure never aborts. An acquire that raises is not version skew: fail closed and
    release holder-qualified best-effort (safe if never acquired)."""
    try:
        acquired = try_acquire(lease.sid, lease.holder, ttl_seconds=lease.ttl)
        if acquired:
            try:
                lease.watermark = lease.db.get_active_message_watermark(lease.sid)
                # A captured watermark makes the commit safe against later rows on BOTH commit
                # paths; tell the fence so a host may keep this attempt's admission.
                if commit_fence is not None:
                    with contextlib.suppress(AttributeError):  # test doubles without the method
                        commit_fence.mark_commit_watermark_fenced()
            except Exception as _wm_err:
                logger.warning(
                    "compression watermark capture failed for session=%s (%s) — concurrent appends this cycle "
                    "will be archived with the snapshot", lease.sid, _wm_err,
                )
                lease.watermark = None
        return acquired
    except Exception as _lock_err:
        with _swallow('compression lock cleanup after failed acquire failed: %s'):
            lease.db.release_compression_lock(lease.sid, lease.holder)
        lease.holder = None
        logger.warning(
            "compression lock acquisition raised unexpectedly for session=%s (%s: %s) — skipping compression this cycle",
            lease.sid, type(_lock_err).__name__, _lock_err,
        )
        return False


def _sit_out_lock_contention(
    agent: Any, lease: _CompressionLease, lifecycle: _CompactionLifecycle, system_message: str,
    approx_tokens: Optional[int], attempt_started_at: float,
) -> Tuple[None, str]:
    """Another path holds the lock: publish the lock-skip signal, warn once, sit out."""
    existing = None
    with contextlib.suppress(Exception):
        existing = lease.db.get_compression_lock_holder(lease.sid)
    logger.warning(
        "compression skipped: another path is compressing session=%s "
        "(holder=%s) — returning messages unchanged to avoid session fork", lease.sid, existing,
    )
    lease.holder = None  # don't release a lock we don't own
    # Distinguish lock-contention no-op from "nothing to compress" so manual
    # /compress can show a clear status instead of "No changes".
    agent._compression_skipped_due_to_lock = existing or True
    # Surface to the user once — quiet for downstream auto-compress loops
    if getattr(agent, "_last_compression_lock_warning_sid", None) != lease.sid:
        agent._last_compression_lock_warning_sid = lease.sid
        with contextlib.suppress(Exception):
            agent._emit_warning(
                "⚠ Skipping concurrent compression — another path is already compressing this session. Will retry "
                "after it finishes."
            )
    _existing_sp = _existing_system_prompt(agent, system_message)
    with contextlib.suppress(Exception):
        if hasattr(agent.context_compressor, "_begin_compression_telemetry"):
            agent.context_compressor._begin_compression_telemetry(current_tokens=approx_tokens)
    return _abort_lease(agent, lifecycle, system_message, attempt_started_at, "lock_contended", _existing_sp)


def _acquire_compression_lease(
    agent: Any, *, commit_fence: Optional[CompressionCommitFence], lifecycle: _CompactionLifecycle,
    system_message: str, approx_tokens: Optional[int], attempt_started_at: float,
) -> Tuple[Optional[_CompressionLease], Optional[str]]:
    """Take the per-session compression lock; ``(None, prompt)`` means sit out.
    Two AIAgents sharing a session_id (e.g. background review fork) would both rotate and orphan a child.
    Keyed on the OLD id (what rivals read from SessionEntry). Loser sits out: messages unchanged, caller sees
    no-op. Only structural absence of the lock API (version skew) fails open; once resolved, any exception
    fails closed since unlocked runs can fork lineage."""
    _lock_db = getattr(agent, "_session_db", None)
    _lock_sid = agent.session_id or ""
    # Clear stale lock-skip so this call's outcome alone is visible; else a manual
    # /compress after an auto lock-skip falsely reports "already in progress".
    agent._compression_skipped_due_to_lock = None
    _try_acquire_lock, _lock_lookup_error = _resolve_lock_api(_lock_db)
    _lock_ttl = 300.0
    with contextlib.suppress(TypeError, ValueError):
        _lock_ttl = float(getattr(agent, "_compression_lock_ttl_seconds", 300.0) or 300.0)
    lease = _CompressionLease(
        agent, db=_lock_db, sid=_lock_sid, ttl=_lock_ttl,
        refresh_interval=getattr(agent, "_compression_lock_refresh_interval", None), commit_fence=commit_fence,
        lifecycle=lifecycle,
    )
    if _lock_db is not None and _lock_sid:
        lease.holder = _compression_lock_holder(agent)
        if _lock_lookup_error is not None:
            # Attribute lookup itself failed for a reason other than a missing
            # lock API. It is unsafe to proceed without a lock in that case.
            lease.holder = None
            logger.warning(
                "compression lock lookup raised unexpectedly for session=%s (%s: %s) — skipping compression this cycle",
                _lock_sid, type(_lock_lookup_error).__name__, _lock_lookup_error,
            )
            _lock_acquired = False
        elif _try_acquire_lock is None:
            # Lock API absent on this instance: log once, proceed unlocked so version skew
            # cannot stall the outer auto-compression loop forever.
            lease.holder = None
            if getattr(agent, "_last_compression_lock_error_sid", None) != _lock_sid:
                agent._last_compression_lock_error_sid = _lock_sid
                logger.warning(
                    "compression lock subsystem unavailable for session=%s — proceeding without lock. This usually means a stale "
                    "in-memory module after an update; restart the process (or `hermes update`) to resync.",
                    _lock_sid,
                )
            _lock_acquired = True  # acquired-but-unlocked compatibility path
        else:
            if not lease.begin_lock_setup():
                logger.info(
                    "Compression commit cancelled before lock acquisition (session=%s).", agent.session_id or "none"
                )
                agent._last_compaction_in_place = False
                return _abort_lease(agent, lifecycle, system_message, attempt_started_at, "commit_fence_cancelled")
            _lock_acquired = _try_acquire_durable_lock(lease, _try_acquire_lock, commit_fence)
        if not _lock_acquired:
            lease.finish_lock_setup()
            return _sit_out_lock_contention(
                agent, lease, lifecycle, system_message, approx_tokens, attempt_started_at
            )
    if lease.holder is not None:
        agent._active_compression_lock_holder = lease.holder
        if commit_fence is not None and commit_fence.register_cancelled_lock_release(lease.release_holder_only):
            # Cancellation won during lock setup (hook ran synchronously, lease gone): abort before any summary work.
            logger.info(
                "Compression commit cancelled before summary dispatch (session=%s).", agent.session_id or "none"
            )
            agent._last_compaction_in_place = False
            _existing_sp = _existing_system_prompt(agent, system_message)
            _emit_aborted_attempt_telemetry(agent, attempt_started_at, "commit_fence_cancelled")
            lease.release()
            return None, _existing_sp
    return lease, None


def _adopt_if_parent_rotated(
    agent: Any, lease: _CompressionLease, messages: list, system_message: str
) -> Optional[Tuple[list, str]]:
    """Sit out (or adopt the live child) when the parent was already rotated.
    A late contender can take the parent lock after the winner released it and rotated; holding the lock does
    not prove this agent still owns a live parent. Returns the ``compress_context`` result to hand back, or
    None to proceed."""
    if lease.db is None or not lease.sid:
        return None
    try:
        _parent_already_rotated = _session_was_rotated_by_compression(lease.db, lease.sid)
    except Exception as _session_err:
        logger.warning(
            "compression session ownership lookup failed for session=%s (%s: %s) - skipping compression this cycle",
            lease.sid, type(_session_err).__name__, _session_err,
        )
        lease.release()
        return messages, _existing_system_prompt(agent, system_message)
    if not _parent_already_rotated:
        return None
    recovered_messages = _adopt_live_compression_child(agent, lease.db, lease.sid)
    lease.release()
    _existing_sp = _existing_system_prompt(agent, system_message)
    if recovered_messages is not None:
        logger.warning("compression recovery: stale session=%s adopted live child=%s", lease.sid, agent.session_id)
        return recovered_messages, _existing_sp
    logger.warning(
        "compression skipped: session=%s was already rotated by "
        "another compression path, but no unique live child could be adopted", lease.sid,
    )
    return messages, _existing_sp


def _adopt_grown_durable_parent(agent: Any, lease: _CompressionLease, messages: list) -> Optional[list]:
    """Return the durable parent transcript when it outgrew the in-memory snapshot.
    Rotation only (in-place never loses rows). The snapshot predates the lease: if durable grew, a writer
    committed a turn — ADOPT it (aborting wedged busy sessions forever). Length check only: in-memory edits of
    past turns are legal."""
    if lease.db is None or not lease.sid:
        return None
    durable_loader = getattr(type(lease.db), "get_messages_as_conversation", None)
    if not callable(durable_loader):
        return None
    durable_parent = durable_loader(lease.db, lease.sid)
    if not (isinstance(durable_parent, list) and len(durable_parent) > len(messages)):
        return None
    # In-memory carries this turn's un-persisted user tail; flush it via the normal
    # rotation-boundary path before adopting, else skip adoption (would drop input).
    # The in-memory transcript carries the CURRENT turn's un-persisted user tail (anchored by
    # _persist_user_message_idx) that the durable snapshot read above does not contain yet. Flush that tail
    # through the normal rotation-boundary path (conversation_history = the already-durable prefix, #68196
    # boundary) BEFORE adopting, then re-read the durable parent so the adopted snapshot includes the live
    # input. If the flush fails (or the anchor is unknown), skip adoption entirely: replacing the in-memory
    # transcript with a snapshot that lacks the user's input would silently drop it from the summarized and
    # rotated history (#adopt-live-tail).
    _preflush_idx = getattr(agent, "_persist_user_message_idx", None)
    # No un-persisted tail means the transcript is fully durable: adopting the longer parent cannot drop input.
    _preflush_ok = True
    if isinstance(_preflush_idx, int) and 0 <= _preflush_idx < len(messages):
        _preflush_ok = False
        with contextlib.suppress(Exception):
            _preflush_ok = agent._flush_messages_to_session_db(messages, conversation_history=messages[:_preflush_idx])
    if not _preflush_ok:
        logger.warning(
            "compression: session=%s grew before lease (%d → %d msgs) but the pre-adoption flush of the "
            "live tail failed; skipping durable-snapshot adoption so un-persisted user input is kept",
            lease.sid, len(messages), len(durable_parent),
        )
        return None
    # Re-read after the flush so the adopted snapshot carries the just-persisted tail.
    durable_parent = durable_loader(lease.db, lease.sid)
    if not (isinstance(durable_parent, list) and len(durable_parent) > len(messages)):
        return None
    logger.info(
        "compression: session=%s grew before lease (%d → %d msgs); adopting durable snapshot", lease.sid, len(messages),
        len(durable_parent),
    )
    return durable_parent


def _pre_compress_memory_context(agent: Any, messages: list, checkpoint_required: bool) -> str:
    """Provider ``on_pre_compress()`` insights to surface in the summary ("" if none).
    Raw messages stay the API v1 provider contract; normalized evidence goes only to API v2+ checkpoint
    providers inside MemoryManager.on_pre_compress(). Raises :class:`CompressionCheckpointUnavailable` when a
    required checkpoint cannot be taken."""
    memory_context = ""
    memory_manager = getattr(agent, "_memory_manager", None)
    evidence_messages = _direct_messages_for_pre_compress_memory(messages)
    if checkpoint_required:
        supports_checkpoint = getattr(memory_manager, "supports_pre_compress_checkpoint", None)
        if memory_manager is None or not callable(supports_checkpoint):
            raise _checkpoint_blocked(
                f"no active provider implements checkpoint API v{PRE_COMPRESS_CHECKPOINT_API_VERSION}"
            )
        try:
            compatible = bool(supports_checkpoint(PRE_COMPRESS_CHECKPOINT_API_VERSION))
        except Exception as exc:
            raise _checkpoint_blocked("provider capability probe failed") from exc
        if not compatible:
            raise _checkpoint_blocked(
                f"active provider does not implement checkpoint API v{PRE_COMPRESS_CHECKPOINT_API_VERSION}"
            )
        try:
            _maybe_ctx = memory_manager.on_pre_compress(
                messages, evidence_messages=evidence_messages, require_checkpoint=True,
                checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
            )
        except Exception as exc:
            logger.warning("Required pre-compress checkpoint failed (%s)", type(exc).__name__)
            raise _checkpoint_blocked(f"provider checkpoint API v{PRE_COMPRESS_CHECKPOINT_API_VERSION} failed") from exc
        if isinstance(_maybe_ctx, str):
            memory_context = sanitize_memory_context(_maybe_ctx)
    elif memory_manager:
        with contextlib.suppress(Exception):
            _maybe_ctx = memory_manager.on_pre_compress(messages, evidence_messages=evidence_messages)
            if isinstance(_maybe_ctx, str):
                memory_context = sanitize_memory_context(_maybe_ctx)
    return memory_context


def _resolve_compress_call(
    agent: Any, *, approx_tokens: Optional[int], focus_topic: Optional[str], force: bool, memory_context: str,
    bypass_cooldown: bool,
) -> Tuple[Callable[..., Any], dict[str, Any]]:
    """Bind ``compress()`` and only the kwargs its signature accepts."""
    compress_fn = agent.context_compressor.compress
    compress_kwargs = _supported_compression_kwargs(
        compress_fn, current_tokens=approx_tokens, focus_topic=focus_topic, force=force, memory_context=memory_context,
        bypass_cooldown=bypass_cooldown,
    )
    if memory_context.strip() and "memory_context" not in compress_kwargs:
        engine_name = getattr(agent.context_compressor, "name", type(agent.context_compressor).__name__)
        if getattr(agent, "_last_memory_context_unsupported_engine", None) != engine_name:
            agent._last_memory_context_unsupported_engine = engine_name
            logger.warning(
                "context engine %s does not accept memory_context; continuing without provider-supplied summary context",
                engine_name,
            )
    return compress_fn, compress_kwargs


def _run_summary_dispatch(
    agent: Any, messages: list, compress_fn: Callable[..., Any], compress_kwargs: dict[str, Any], *,
    commit_fence: Optional[CompressionCommitFence], attempt_generation: Any, hard_cancel_event: Any,
) -> list:
    """Run the compressor under the fence's progress hook, deadline and interrupt guard."""
    # Publish progress to the commit fence so hosts extend deadlines while tokens
    # flow. Any active hook (even no-op) selects the streamed path: the timeout is
    # inactivity-based and a byte-trickling provider hits the stream total ceiling.
    from agent.auxiliary_client import aux_interrupt_protection, aux_progress_hook, aux_stream_deadline
    _progress_hook = commit_fence.touch_progress if commit_fence is not None else (lambda: None)
    # Return leg: cancel frees the owner but the provider daemon streams on to its
    # own larger ceiling; share the host deadline so orphan streams stop with it.
    _host_stream_deadline = commit_fence.deadline_monotonic if commit_fence is not None else None
    # A LATE successful summary must not undo the host's timeout cooldown: the
    # compressor checks cancellation before clearing; removed in finally (no leak).
    if commit_fence is not None:
        # Install a cancellation check the compressor consults BEFORE clearing the failure cooldown; removed
        # in the finally below so it cannot leak into later attempts (e.g. a manual /compress force-clear).
        # See #76354.
        _install_compression_cancelled_check(
            agent.context_compressor, lambda: commit_fence.is_cancelled, attempt_generation
        )

    def _compression_cancel_requested() -> bool:
        return bool(
            (hard_cancel_event is not None and hard_cancel_event.is_set())
            or (commit_fence is not None and commit_fence.is_cancelled)
        )

    try:
        # F6: never start expensive summary work for an already-cancelled
        # fence (a stale queued job admitted after host departure).
        if commit_fence is not None and commit_fence.is_cancelled:
            logger.info(
                "Compression cancelled before summary dispatch (session=%s) — skipping summary work.",
                agent.session_id or "none",
            )
            compressed = messages
        else:
            with (
                aux_progress_hook(_progress_hook), aux_stream_deadline(_host_stream_deadline),
                aux_interrupt_protection(cancel_check=_compression_cancel_requested),
            ):
                compressed = compress_fn(messages, **compress_kwargs)
                # Freeze a hard stop that arrived after the last provider attempt but before session state rotates.
                if hard_cancel_event is not None and hard_cancel_event.is_set():
                    raise AuxiliaryExplicitCancellation()
    finally:
        if commit_fence is not None:
            _clear_compression_cancelled_check_if_owner(agent.context_compressor, attempt_generation)
    return compressed


def _fold_todo_snapshot(agent: Any, compressed: list) -> None:
    """Strip stale todo snapshots from ``compressed`` and fold the live one in (in place)."""
    todo_snapshot = agent._todo_store.format_for_injection()
    # Non-empty store (even all done) is authoritative: drop the old snapshot. A
    # truly empty store may be un-rehydrated post-compaction: keep the snapshot.
    _todo_has_items = getattr(agent._todo_store, "has_items", None)
    # Store may implement only format_for_injection(); unknown authority must
    # preserve the pending snapshot rather than risk deleting it.
    _todo_store_is_authoritative = False
    with contextlib.suppress(Exception):
        _todo_store_is_authoritative = bool(_todo_has_items()) if callable(_todo_has_items) else False
    if _todo_store_is_authoritative:
        for _todo_idx in range(len(compressed) - 1, -1, -1):
            _todo_message = compressed[_todo_idx]
            if not isinstance(_todo_message, dict) or _todo_message.get("role") != "user":
                continue
            _todo_content = _todo_message.get("content")
            _todo_stripped = _strip_stale_todo_snapshot(_todo_content)
            if _todo_stripped == _todo_content:
                continue
            if _todo_message.get("_todo_snapshot_synthetic") and _todo_snapshot_is_only_content(
                _todo_content, _todo_stripped
            ):
                compressed.pop(_todo_idx)
                if _todo_idx < len(compressed):
                    # A standalone snapshot can drift from the tail; deleting it may expose two
                    # assistant rows, so use the normal replay repair to keep metadata consistent.
                    agent._repair_message_sequence(compressed)
            else:
                _replace_message_content(_todo_message, _todo_stripped)
                # No longer todo-only scaffolding; other synthetic flags stay authoritative and
                # _is_real_user_message() recomputes provenance from content + flags.
                _todo_message.pop("_todo_snapshot_synthetic", None)
            break
    if todo_snapshot:
        # If this boundary pruned skill bodies, the policy behind the todos is gone:
        # add a reload notice after TODO_INJECTION_HEADER so both strip together.
        # Retention parity (#84718): the snapshot below re-injects the imperative verbatim. If this same
        # boundary pruned skill bodies to [SKILL_PRUNED: ...] markers, the policy that governed those tasks
        # is gone — couple a reload instruction to the snapshot so the imperative never crosses the boundary
        # alone.
        _reload_notice = _pruned_skill_reload_notice(compressed)
        if _reload_notice:
            todo_snapshot = f"{todo_snapshot}\n\n{_reload_notice}"
        # Fold the snapshot into a trailing REAL user msg (no synthetic user/user pair);
        # strip old snapshots first. Scaffolding tails must not absorb it (provenance).
        # Any snapshot merged at an earlier boundary is stripped first so repeated compactions refresh
        # rather than accumulate todo state (#26981). Scaffolding tails (continuation marker, summary
        # handoff, a bare stale snapshot row) must never absorb the snapshot: merging would upgrade them to
        # "real user" evidence and break zero-user provenance (#69292), so those keep the flagged standalone
        # append and the real-user preservation pass continues to see todo scaffolding, not human intent.
        from agent.context_compressor import _append_text_to_content
        merged = False
        _tail = compressed[-1] if compressed and isinstance(compressed[-1], dict) else None
        if _tail is not None and _tail.get("role") == "user":
            _stripped = _strip_stale_todo_snapshot(_tail.get("content"))
            _probe = {key: value for key, value in _tail.items() if key != "content"}
            _probe["content"] = _stripped
            if _is_real_user_message(_probe):
                _snapshot_text = f"\n\n{todo_snapshot}" if isinstance(_stripped, str) and _stripped else todo_snapshot
                _replace_message_content(_tail, _append_text_to_content(_stripped, _snapshot_text))
                merged = True
            elif (
                _stripped != _tail.get("content") and not _message_text({"role": "user", "content": _stripped}).strip()
            ):
                # The tail was nothing but an earlier snapshot row —
                # refresh it in place instead of stacking a duplicate.
                _replace_message_content(_tail, todo_snapshot)
                _tail["_todo_snapshot_synthetic"] = True
                merged = True
        if not merged:
            compressed.append({"role": "user", "content": todo_snapshot, "_todo_snapshot_synthetic": True})


def _rebuild_system_prompt_at_boundary(agent: Any, system_message: str) -> str:
    """Refresh tool schemas and rebuild the system prompt at the commit boundary."""
    cached_system_prompt = agent._cached_system_prompt
    agent._invalidate_system_prompt()

    # Refresh dynamic tool schemas at the same admitted-commit boundary that rebuilds the system prompt
    # (maintainer-directed, #95681 arc): forever-sessions (Bot Mode chats, gateway channels) never
    # restart, so compaction is the ONLY point where a config change — image model swap, delegation
    # depth, code_execution mode — can reach agent.tools. The prompt cache is already broken here, so
    # the refresh is free; when nothing changed the snapshot is byte-equal and we keep the existing list
    # object (identity matters to provider-side tool-block caching on some backends).
    try:
        _refresh_agent_tool_definitions(agent)
    except Exception:  # noqa: BLE001
        logger.warning(
            "compaction tool-definition refresh failed; keeping the session's existing tool snapshot", exc_info=True
        )

    # ALWAYS rebuild the prompt here: keeping old bytes meant prompt-builder changes
    # never reached long sessions. Equal bytes keep KV; preserve object identity.
    # ALWAYS rebuild the prompt at the admitted-commit boundary (maintainer-directed, #95681 arc). The
    # previous "keep-prompt" containment branch put the OLD bytes back whenever the reloaded memory blocks
    # were already embedded — which meant prompt-builder changes (guidance diets, new blocks, renames) NEVER
    # reached a long-lived session. The cache argument for keeping bytes was hollow: when nothing changed,
    # the rebuild is byte-identical and local KV prefixes survive on equality; when something changed, the
    # cache was stale by definition and propagation is the point. Preserve OBJECT identity on byte-equality
    # for backends that key on it.
    rebuilt_system_prompt = agent._build_system_prompt(system_message)
    if cached_system_prompt is not None and rebuilt_system_prompt == cached_system_prompt:
        new_system_prompt = agent._cached_system_prompt = cached_system_prompt
        from agent.system_prompt import reconstruct_static_prefix
        reconstruct_static_prefix(agent, system_message=system_message, log_label="compression keep-prompt")
    else:
        new_system_prompt = agent._cached_system_prompt = rebuilt_system_prompt
        if cached_system_prompt is not None:
            logger.info(
                "Compaction rebuilt a drifted system prompt (session=%s, %d -> %d chars): builder output changed "
                "since the stored snapshot (update, config change, or memory/skills growth)",
                agent.session_id or "none", len(cached_system_prompt), len(new_system_prompt),
            )
    return new_system_prompt


def _salvage_or_refuse_grown_transcript(
    agent: Any, messages: list, compressed: list, *, system_message: str, attempt_started_at: float,
    attempt_snapshot: dict,
) -> Tuple[Optional[list], Optional[str]]:
    """Anti-growth guard at the COMMIT SITE (in-place commits before the gateway can inspect).
    Compares like-for-like rough estimates; on growth tries one mechanical salvage pass, else treats the
    attempt as a refused no-op. Returns ``(compressed, None)`` to proceed or ``(None, prompt)`` when refused
    (caller releases the lease)."""
    # Anti-growth guard at the COMMIT SITE: never persist a compression that makes the transcript larger
    # (observed: 379K -> 687K when the generated summary plus retained reasoning exceeded what it replaced).
    # Compare like-for-like (both rough estimates of the same message shape) so an "actual vs estimate"
    # measurement mismatch cannot produce a false verdict. The gateway has a rotation-path-only guard
    # (#83339), but in-place compaction commits inside this method via archive_and_compact — before the
    # gateway can inspect the result — so the guard must live here to protect both paths. On growth, treat
    # the attempt as a no-op: the original transcript stays untouched and durable.
    _rough_in = estimate_messages_tokens_rough(messages)
    _rough_out = estimate_messages_tokens_rough(compressed)
    if _rough_out > _rough_in:
        # Todo refresh and user-turn anchoring run after the compressor's own size check
        # and can tip a break-even candidate; give it one mechanical salvage pass.
        from agent.context_compressor import salvage_grown_transcript
        _salvaged = salvage_grown_transcript(messages, compressed, budget=_rough_in)
        if _salvaged is not None:
            _salv_est = estimate_messages_tokens_rough(_salvaged)
            if _salv_est < _rough_in:
                logger.info(
                    "Compression salvage recovered a shrinking transcript (session=%s, ~%s -> ~%s tokens)",
                    agent.session_id or "none", f"{_rough_in:,}", f"{_salv_est:,}",
                )
                compressed = _salvaged
                _rough_out = _salv_est
    if _rough_out > _rough_in:
        logger.warning(
            "Compression refused: compressed transcript would be larger than the original (session=%s, ~%s -> ~%s "
            "tokens); keeping the original transcript unchanged", agent.session_id or "none",
            f"{_rough_in:,}",
            f"{_rough_out:,}",
        )
        # Flag the refusal on compressor state so /compress feedback reports it instead
        # of comparing list lengths (adoption can change the count), claiming success.
        with contextlib.suppress(Exception):
            agent.context_compressor._last_compress_refused_would_grow = True
        with contextlib.suppress(Exception):
            agent._emit_warning(
                "⚠️ Compression refused: the generated summary would have GROWN the conversation instead of "
                "shrinking it. No messages were dropped — conversation continues unchanged."
            )
        _existing_sp = _existing_system_prompt(agent, system_message)
        _emit_aborted_attempt_telemetry(agent, attempt_started_at, "would_grow")
        # Count the refusal as an ineffective-compaction strike so the anti-thrash
        # breaker latches; otherwise auto-compress retries the same summary every turn.
        with _swallow('could not record rejected-compaction strike', exc_info=True):
            # Without this, the unchanged transcript stays over the compression threshold and automatic
            # compression retries the identical summary request on every turn (#88568). Manual /compress
            # keeps bypassing the latch (force=True skips the guards).
            agent.context_compressor.record_rejected_compaction()
        _restore_prune_rearm_tokens(agent.context_compressor, attempt_snapshot)
        return None, _existing_sp
    return compressed, None


def _parent_deliberately_ended(session_db: Any, session_id: str) -> bool:
    """True when the parent row was ended by a non-automatic reason. Fails OPEN: an
    unreadable row must not turn a cheap guard into a new way to lose compression."""
    reader = getattr(session_db, "get_session", None)
    if not callable(reader):
        return False
    try:
        from hermes_state_common import is_automatic_end_reason
        row = reader(session_id) or {}
        return row.get("ended_at") is not None and not is_automatic_end_reason(row.get("end_reason"))
    except Exception:
        return False


def _carry_session_state_to_child(agent: Any, old_session_id: str, old_title: Any) -> None:
    """Migrate /goal, /heartbeat, /loop state and the title from the parent to the child.
    Each lookup is a flat per-session read with no parent walk, so state would silently die at the boundary. The title
    is carried unchanged (renumbering per rotation made one session look like many); its provenance is read BEFORE the
    transfer clears the ancestor's row, then restored so an inherited auto-title stays upgradeable.
    """
    with _swallow('Could not migrate goal on compression: %s'):
        # Carry a persistent /goal onto the continuation session. Compression mints a fresh child id;
        # load_goal does a flat per-session lookup with no parent walk, so without this an active goal
        # silently dies at the boundary (#33618).
        from hermes_cli.goals import migrate_goal_to_session
        migrate_goal_to_session(old_session_id, agent.session_id, reason="compression")
    with _swallow('Could not migrate heartbeat on compression: %s'):
        from hermes_cli.heartbeat import migrate_heartbeat_to_session
        migrate_heartbeat_to_session(old_session_id, agent.session_id)
    with _swallow('Could not migrate loop on compression: %s'):
        from hermes_cli.loops import migrate_loop_to_session
        migrate_loop_to_session(old_session_id, agent.session_id, reason="compression")
    if not old_title:
        return
    _src = None
    with _swallow('Could not read title provenance: %s'):
        _src = agent._session_db.get_session_title_source(old_session_id)
    try:
        agent._session_db.set_session_title(agent.session_id, old_title)
    except Exception as e:
        logger.debug("Could not propagate title on compression: %s", e)
        return
    # set_session_title() records "user"; restore the original authority.
    if _src is not None:
        with _swallow('Could not propagate title provenance: %s'):
            agent._session_db.set_session_title_source(agent.session_id, _src)


def _publish_rotated_compaction(
    agent: Any, messages: list, compressed: list, *, new_system_prompt: str, lease: _CompressionLease,
    old_session_id: str, compressed_user_turn_outcome: str,
) -> None:
    """Rotate the session: flush the parent, publish the child, re-point the agent.
    Flushes current-turn msgs to the OLD session, passing the durable prefix (messages[:persist idx]) so
    preflight, which runs before rows are marker-stamped, can't re-append them."""
    current_idx = getattr(agent, "_persist_user_message_idx", None)
    persisted_history = (
        messages[:current_idx] if isinstance(current_idx, int) and 0 <= current_idx <= len(messages) else None
    )
    # The flush is durable and NOT rolled back on abort: a deliberately-ended parent
    # fails publish forever, so check that before writing. Automatic end stamps are
    # healed by publish (don't abort); the lease is re-acquirable (don't check it).
    if _parent_deliberately_ended(agent._session_db, old_session_id):
        raise RuntimeError(f"Compression parent already ended: {old_session_id}")
    # Foreign-tail ceiling: the flush below writes OUR rows (already in handoff);
    # rows above the start watermark up to this MAX(id) are foreign appends.
    # No trustworthy ceiling means the clone could duplicate the handoff: skip tail preservation this rotation.
    _foreign_tail_ceiling = None
    with contextlib.suppress(Exception):
        _foreign_tail_ceiling = agent._session_db.get_active_message_watermark(agent.session_id)
    with contextlib.suppress(Exception):  # best-effort — don't block compression on a flush error
        agent._flush_messages_to_session_db(messages, conversation_history=persisted_history)
    # Publish closure + child + handoff in one transaction so no reader sees an
    # empty child. Child stays on the parent's profile ("default" persists as NULL);
    # publish also COALESCEs from the parent row for threads lacking HERMES_HOME.
    _profile_for_child = None
    with contextlib.suppress(Exception):
        from hermes_cli.profiles import get_active_profile_name
        _profile_for_child = get_active_profile_name()
    if _profile_for_child == "default":
        _profile_for_child = None
    old_title = agent._session_db.get_session_title(agent.session_id)
    new_session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    from agent.context_compressor import _DB_PERSISTED_MARKER
    agent._session_db.publish_compression_child(
        parent_session_id=old_session_id, child_session_id=new_session_id,
        source=agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"), model=agent.model,
        model_config=agent._session_init_model_config, system_prompt=new_system_prompt, messages=compressed,
        cwd=getattr(agent, "working_directory", None), profile_name=_profile_for_child,
        compression_lock_holder=lease.holder, require_compression_lease=lease.holder is not None,
        require_lease_refresh=lease.holder is not None, lease_ttl_seconds=lease.ttl,
        watermark=(lease.watermark if _foreign_tail_ceiling is not None else None),
        watermark_ceiling=_foreign_tail_ceiling,
    )
    # `already_present` stamping is done by run_agent's _sync_persisted_markers;
    # this branch covers inserted/merged only; direct callers must use that wrapper.
    if compressed_user_turn_outcome in {"inserted", "merged"}:
        # Stamp the anchor source row itself, not the (drifted, possibly out-of-range)
        # persist index; don't match the HANDOFF row — for `merged` it is a superset.
        _compressed_anchor_source = next((m for m in reversed(messages) if _is_real_user_message(m)), None)
        if isinstance(_compressed_anchor_source, dict):
            _compressed_anchor_source[_DB_PERSISTED_MARKER] = True
            _session_messages = getattr(agent, "_session_messages", None)
            if isinstance(_session_messages, list) and _session_messages is not messages:
                # Adoption may leave _session_messages on the pre-adoption list with an out-of-range idx; stamp every
                # scoped twin against the ANCHOR SOURCE, as the wrapper. An already-stamped exact twin still
                # suppresses the broad pass here, or a content-equal old duplicate would get stamped.
                _stamp_scoped_twins(_session_messages, _compressed_anchor_source, exact_counts_stamped=True)
    for _handoff_message in compressed:
        if isinstance(_handoff_message, dict):
            _handoff_message[_DB_PERSISTED_MARKER] = True
    agent.session_id = new_session_id
    agent._db_flush_scan_prefix = None
    _rebind_session_context(agent.session_id)
    agent._session_db_created = True
    _carry_session_state_to_child(agent, old_session_id, old_title)


def _warn_summary_or_aux_fallback(agent: Any) -> None:
    """Surface a failed summary, or a recovered-but-broken aux compression model, once."""
    summary_error = getattr(agent.context_compressor, "_last_summary_error", None)
    if summary_error:
        if getattr(agent, "_last_compression_summary_warning", None) != summary_error:
            agent._last_compression_summary_warning = summary_error
            agent._emit_warning(f"⚠ Compression summary failed: {summary_error}. Inserted a fallback context marker.")
    else:
        # Aux model may have errored and been recovered on main; tell the user their
        # auxiliary.compression.model is broken even though compression succeeded.
        _aux_fail_model = getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
        _aux_fail_err = getattr(agent.context_compressor, "_last_aux_model_failure_error", None)
        # Dedup on (model, error) so we don't spam on every compaction
        _aux_key = (_aux_fail_model, _aux_fail_err)
        if _aux_fail_model and getattr(agent, "_last_aux_fallback_warning_key", None) != _aux_key:
            agent._last_aux_fallback_warning_key = _aux_key
            agent._emit_warning(
                f"ℹ Configured compression model '{_aux_fail_model}' failed "
                f"({_aux_fail_err or 'unknown error'}). Recovered using main model — "
                "check auxiliary.compression.model in config.yaml."
            )


def _reset_read_dedup_caches(task_id: str, *, skills: bool = True) -> None:
    """Advance the file-read (and skill_view) repeat-read dedup to a fresh generation after a boundary.
    The mtime map is kept: the first read of each unchanged key returns full content compaction may have
    omitted; later reads return stubs, and stub-hit counters restart at the same boundary (#84857).
    """
    with contextlib.suppress(Exception):
        from tools.file_tools_read_tracking import reset_file_dedup
        reset_file_dedup(task_id)
    if not skills:
        return
    with contextlib.suppress(Exception):
        from tools.skills_tool import reset_skill_view_dedup
        reset_skill_view_dedup(task_id)


def _finish_compaction_boundary(
    agent: Any, compressed: list, *, new_system_prompt: str, old_session_id: Optional[str], in_place: bool,
    compacted_in_place: bool, session_commit_succeeded: bool, defer_context_engine_notification: bool,
    compression_made_progress: bool, compression_used_fallback: bool, compression_feasibility_skip: bool,
    task_id: str,
) -> int:
    """Post-commit bookkeeping: notify engines/providers/hooks, re-arm usage tracking.
    Returns the rough post-compression token estimate (diagnostics only)."""
    # old_session_id is bound only on rotation; _boundary_parent is the id the
    # boundary notifications attribute prior state to (old id, or same id in-place).
    _old_sid = old_session_id
    _boundary_parent = _old_sid or agent.session_id or ""

    # The heartbeat's terminal stamp landed on the PARENT before the id re-pointed;
    # clear labels (keep last_activity_at) so the archived row isn't falsely fresh.
    if _old_sid and session_commit_succeeded:
        with _swallow("failed to clear archived compression parent's activity labels (ignored)", exc_info=True):
            _labels_db = getattr(agent, "_session_db", None)
            if callable(_clear_labels := getattr(type(_labels_db), "clear_session_activity_labels", None)):
                _clear_labels(_labels_db, _old_sid)

    # Plugin engines use boundary_reason="compression" to keep lineage/checkpoint
    # state. Fires in BOTH modes: in-place passes the same id, the boundary is real.
    if session_commit_succeeded and (bool(_old_sid) or compacted_in_place):
        notify = (
            _queue_context_engine_compression_notification
            if defer_context_engine_notification
            else _notify_context_engine_compression_complete
        )
        notify(agent, new_session_id=agent.session_id or "", old_session_id=_boundary_parent)

    # Providers refresh cached per-session state; reset=False, conversation goes on.
    # Fires in BOTH modes so buffers don't double-count dropped turns in-place.
    with _swallow('memory manager on_session_switch (compression): %s'):
        if (bool(_old_sid) or in_place) and agent._memory_manager:
            agent._memory_manager.on_session_switch(
                agent.session_id or "", parent_session_id=_boundary_parent, reset=False, reason="compression"
            )

    # Route via _emit_status so the warning reaches gateway platforms; store it on
    # _compression_warning so a late-bound status_callback can replay it.
    compressor = agent.context_compressor
    _cc = compressor.compression_count
    if _cc >= 2:
        _cc_msg = (
            f"{agent.log_prefix}⚠️  Session compressed {_cc} times — accuracy may degrade. Consider /new to start fresh."
        )
        agent._compression_warning = _cc_msg
        agent._emit_status(_cc_msg)

    # session:compress lets hooks ingest the old session before it's lost;
    # in_place=True tells them the same id was compacted rather than rotated.
    if getattr(agent, "event_callback", None):
        with _swallow('event_callback error on session:compress: %s'):
            agent.event_callback(
                "session:compress",
                {
                    "platform": agent.platform or "", "session_id": agent.session_id,
                    "old_session_id": _old_sid or "", "in_place": in_place,
                    "compression_count": compressor.compression_count,
                },
            )

    # Rotation-independent flag: the gateway uses it (not an id diff) to re-baseline
    # transcript handling (history_offset=0 + rewrite on the same id) in-place.
    agent._last_compression_attempt_in_place = compacted_in_place
    agent._last_compaction_in_place = compacted_in_place

    # Diagnostics only, not provider usage: schema-heavy rough estimates can stay
    # above threshold even after the next real request fits.
    _compressed_est = estimate_request_tokens_rough(
        compressed, system_prompt=new_system_prompt or "", tools=agent.tools or None
    )
    compressor.last_compression_rough_tokens = _compressed_est
    compressor.last_prompt_tokens = -1
    compressor.last_completion_tokens = 0
    compressor.awaiting_real_usage_after_compression = True
    # Transcript rewritten: invalidate the usage anchor's base snapshot explicitly
    # (its structural check would fail closed anyway); estimate until re-anchored.
    agent._usage_anchor = None
    agent._turn_base_usage_anchor = None
    # Arm the effectiveness verdict only after a completed rewrite crosses the
    # boundary so later usage isn't charged to an attempt that changed nothing.
    if compression_made_progress:
        record_boundary = getattr(type(compressor), "record_completed_compaction", None)
        if callable(record_boundary):
            record_boundary(
                compressor, used_fallback=compression_used_fallback, feasibility_skip=compression_feasibility_skip
            )
        else:
            compressor._verify_compaction_cleared_threshold = True
    _reset_read_dedup_caches(task_id)
    return _compressed_est


def _candidate_rejected(
    agent: Any, compressed: Any, messages: list, messages_before_compression: list, *,
    attempt_generation: Any, attempt_started_at: float,
) -> bool:
    """Reject an unusable compression candidate before any session mutation.
    Order matters: compressor-reported abort, no progress, empty transcript, superseded attempt. Each branch surfaces
    its own warning/telemetry; the caller releases the lease and returns the input unchanged when True.
    """
    # Aborted compression returns input unchanged: surface the error, skip rotation
    # (no session ended); auto-compress callers detect no-op via equal lengths.
    if getattr(agent.context_compressor, "_last_compress_aborted", False):
        _summary_error = getattr(agent.context_compressor, "_last_summary_error", None)
        _err = _summary_error or "unknown error"
        if getattr(agent, "_last_compression_summary_warning", None) != _err:
            agent._last_compression_summary_warning = _err
            agent._emit_warning(
                f"⚠ Compression aborted: {_err}. "
                "No messages were dropped — conversation continues unchanged. "
                "Run /compress to retry, or /new to start a fresh session."
            )
        _emit_aborted_attempt_telemetry(
            agent, attempt_started_at, _summary_error and "summary_generation_aborted"
        )
        return True

    # Compare semantic state, not identity: engines may return an equal copy or
    # mutate the live list. ``==`` first (subclass __eq__), then marker-insensitive.
    # Neither case may rotate or rewrite the session. The raw ``==`` leg runs FIRST so a list subclass
    # returned by an engine keeps its ``__eq__`` semantics (tests seam on this); the marker-insensitive leg
    # (#92231) then covers the cold-resume shape where the stamped snapshot differs from the marker-swept
    # compress() output only by ``_db_persisted``.
    if compressed == messages_before_compression or (
        _strip_marker_for_comparison(compressed) == _strip_marker_for_comparison(messages_before_compression)
    ):
        if messages != messages_before_compression:
            messages[:] = copy.deepcopy(messages_before_compression)
        logger.info(
            "Compression made no progress (session=%s) — skipping boundary rewrite.", agent.session_id or "none"
        )
        # Unchanged output would fail identically next turn; arm structural backoff so
        # auto-compress stops re-firing each turn (success lifts it, force overrides).
        with _swallow('no-progress backoff arm failed', exc_info=True):
            if callable(_recorder := getattr(agent.context_compressor, "_record_structural_no_op", None)):
                _recorder("compaction returned the transcript unchanged (no_progress)")
        _emit_aborted_attempt_telemetry(agent, attempt_started_at, "no_progress")
        return True
    if not compressed:
        logger.error(
            "context compression returned an empty transcript; refusing to rotate session=%s so the parent remains resumable",
            agent.session_id or "none",
        )
        with contextlib.suppress(Exception):
            agent._emit_warning(
                "⚠ Compression returned an empty transcript. No session split was performed; conversation continues unchanged."
            )
        return True

    # A newer attempt claiming this compressor supersedes us; discard the late
    # candidate. Fence poison alone misses a successor that minted its own fence.
    if not _compressor_attempt_is_current(agent.context_compressor, attempt_generation):
        logger.warning(
            "Discarding late compression candidate: attempt generation "
            "%s was superseded by a newer attempt (current: %s) (session=%s).", attempt_generation,
            getattr(agent.context_compressor, "_compression_attempt_generation", None),
            agent.session_id or "none",
        )
        _restore_messages_snapshot(messages, messages_before_compression)
        agent._last_compaction_in_place = False
        _emit_aborted_attempt_telemetry(agent, attempt_started_at, "attempt_superseded")
        return True
    return False


@dataclasses.dataclass
class _CommitOutcome:
    """Result of the SessionDB commit phase (in-place or rotation)."""

    compressed: list
    commit_started_at: float
    refused_prompt: Optional[str] = None
    old_session_id: Optional[str] = None
    split_status: str = "not_applicable"
    session_commit_succeeded: bool = False
    compacted_in_place: bool = False
    made_progress: bool = False


def _commit_compaction(
    agent: Any, messages: list, compressed: list, *, in_place: bool, lease: _CompressionLease,
    new_system_prompt: str, system_message: str, compressed_user_turn_outcome: str,
    messages_before_compression: Optional[list], made_progress: bool, attempt: _Attempt,
) -> _CommitOutcome:
    """Persist the compacted transcript: memory extraction, anti-growth guard, then the
    in-place archive or the parent->child rotation.

    Failures roll the live list back and arm the split-failure cooldown; a refused (would-grow) candidate returns
    ``refused_prompt`` so the caller hands back the input unchanged.
    """
    session_commit_succeeded = False
    compacted_in_place = False
    commit_started_at = time.monotonic()
    split_status = "not_applicable"
    old_session_id: Optional[str] = None  # bound only once rotation begins
    if agent._session_db:
        split_status = "pending"
        try:
            # Memory extraction runs in BOTH modes: pre-compaction turns are summarized
            # away whether or not the id rotates.
            agent.commit_memory_session(messages)

            # Pop _compaction_tail tags before the size estimate / rotation: they must not
            # inflate anti-growth or reach the provider. Track ids: salvage may subset list.
            _tail_tagged_ids = {id(m) for m in compressed if isinstance(m, dict) and m.pop("_compaction_tail", None)}
            compressed, _refused_sp = _salvage_or_refuse_grown_transcript(
                agent, messages, compressed, system_message=system_message, attempt_started_at=attempt.started_at,
                attempt_snapshot=attempt.snapshot,
            )
            if compressed is None:
                return _CommitOutcome(
                    compressed=messages, refused_prompt=_refused_sp, commit_started_at=commit_started_at
                )
            if in_place:
                # In-place compaction: same session_id; soft-archive old turns (active=0, still
                # searchable) + insert `compressed` atomically; no pre-flush (tail already in).
                from agent.context_compressor import PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY, stamp_db_persisted_markers
                # Tail rows tagged by compress() are archived as superseded duplicates, not
                # compacted=1. Count against the FINAL list — salvage may have dropped rows.
                agent._session_db.archive_and_compact(
                    agent.session_id, compressed, model_config_patch={PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY: None},
                    watermark=lease.watermark, lock_holder=lease.holder,
                    tail_count=sum(1 for m in compressed if id(m) in _tail_tagged_ids),
                )
                split_status = "in_place_committed"
                # compress() returned marker-swept copies; stamp them as persisted or the next
                # flush re-INSERTs the whole compacted transcript, doubling the live set. Reset
                # the flush identity set so next turn diffs against the COMPACTED transcript.
                stamp_db_persisted_markers(compressed)
                agent._flushed_db_message_ids = set()
                # Rotation-independent signal; the gateway reads this (not an id diff) to
                # re-baseline transcript handling.
                compacted_in_place = True
                # In-place still updates the current row's prompt; rotation published it atomically above.
                agent._session_db.update_system_prompt(agent.session_id, new_system_prompt)
                agent._last_flushed_db_idx = 0
            else:
                # Bind old_session_id first: it is the rollback key in the handler below.
                # ── Rotation (legacy): end this session, fork a continuation ─ Flush any un-persisted
                # current-turn messages to the OLD session before ending it, so they survive in the
                # preserved parent transcript (#47202). (In-place skips this — see above.) Pass the
                # already-durable prefix as conversation_history so the flush skips it by identity (#68196).
                # Preflight compression runs BEFORE the normal turn flush has stamped the cold-resumed
                # history dicts with _DB_PERSISTED_MARKER, so without a boundary
                # _flush_messages_to_session_db treats every restored row as new and re-appends the whole
                # transcript to the parent. turn_context anchors _persist_user_message_idx at the
                # current-turn user message before preflight runs, so messages[:idx] is exactly the
                # persisted prefix; only the current turn's new messages get written. Bound to
                # old_session_id, hoisted above the flush: the ``except`` handler below keys its in-memory
                # rollback off this name, so anything that fails from here on rolls the transcript back
                # instead of leaving the failed attempt's compacted snapshot in place.
                old_session_id = agent.session_id
                _publish_rotated_compaction(
                    agent, messages, compressed, new_system_prompt=new_system_prompt, lease=lease,
                    old_session_id=old_session_id, compressed_user_turn_outcome=compressed_user_turn_outcome,
                )
                split_status = "rotated_committed"
                agent._last_flushed_db_idx = len(compressed)
                agent._flushed_db_message_session_id = agent.session_id
            session_commit_succeeded = True
        except Exception as e:
            # Rotation: atomic publication failed (including lease loss) — keep the parent live and discard the stale
            # compacted snapshot. In-place: archive_and_compact is atomic so old rows stay active, but marker-swept
            # `compressed` would re-INSERT on top of them (doubling each try); gate on split_status (set right after
            # commit). Either way the deepcopy keeps markers/identity and only the prune runway rolls back (the full
            # snapshot restore is for pre-commit cancels; telemetry keeps the failed values). _db_flush_scan_prefix is
            # intentionally NOT cleared: the scan is identity-based and the deepcopy replaces every row.
            rotation_rollback = not in_place and old_session_id and agent.session_id == old_session_id
            if rotation_rollback or (
                in_place and split_status != "in_place_committed" and messages_before_compression is not None
            ):
                if rotation_rollback:
                    old_session_id = None
                # In-place sibling of the rotation rollback above (#99477). archive_and_compact() is atomic,
                # so a raise before it returned means EVERY pre-compaction row is still ``active = 1`` in
                # state.db — nothing was archived and the compacted set was never inserted. But
                # ``compressed`` is the marker-swept output of compress() (_strip_persistence_markers,
                # #57491) and the post-commit ``stamp_db_persisted_markers`` never ran, so handing it back
                # makes the next append-only flush treat the whole compacted transcript as new and INSERT it
                # ON TOP of the rows it was supposed to replace. The active set then holds the summary AND
                # the turns it summarized; the next resume reloads both, the token count goes UP, preflight
                # fires again, and each failed attempt appends another copy of the protected head + tail
                # (#99477: ~15 real turns stored as 3,814 rows, the first user message repeated 893 times).
                # Gate on ``split_status`` rather than ``compacted_in_place``: it is assigned on the
                # statement immediately after the atomic commit returns, so a committed compaction can never
                # be rolled back into a live/durable mismatch of the opposite sign. The deepcopy carries
                # each row's _DB_PERSISTED_MARKER from the pre-compression snapshot, so the restored
                # transcript is correctly skipped by the flush, and replacing every dict breaks
                # _db_flush_scan_prefix identity (same reasoning as the rotation branch — no explicit clear
                # needed).
                messages[:] = copy.deepcopy(messages_before_compression)
                compressed = messages
                made_progress = False
                _restore_prune_rearm_tokens(agent.context_compressor, attempt.snapshot)
            split_status = "aborted" if old_session_id is None and not in_place else "failed_not_indexed"
            # If rotation rolled back to the parent, agent.session_id is the indexed parent
            # and old_session_id was cleared: recovery, not an un-indexed orphan.
            if old_session_id is None and not in_place:
                logger.warning(
                    "Compression rotation aborted and rolled back to the parent session (%s): %s",
                    agent.session_id or "?", e,
                )
            else:
                logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)
            # Arm the failure cooldown so the next turn can't rerun the doomed compression;
            # try/except so a stub compressor can't mask the original error in this handler.
            with _swallow('could not record split-failure cooldown', exc_info=True):
                # See #97948.
                agent.context_compressor._record_compression_failure_cooldown(
                    _SPLIT_FAILURE_COOLDOWN_SECONDS, f"session_split_failed: {e}"
                )
    return _CommitOutcome(
        compressed=compressed, commit_started_at=commit_started_at, old_session_id=old_session_id,
        split_status=split_status, session_commit_succeeded=session_commit_succeeded,
        compacted_in_place=compacted_in_place, made_progress=made_progress,
    )


@dataclasses.dataclass
class _SummaryPhase:
    """Outcome of the summary phase; ``abort_prompt`` set means hand ``messages`` back."""

    messages: list
    compressed: Any = None
    messages_before_compression: Optional[list] = None
    approx_tokens: Optional[int] = None
    pre_msg_count: int = 0
    abort_prompt: Optional[str] = None


def _run_summary_phase(
    agent: Any, messages: list, *, lease: _CompressionLease, in_place: bool, checkpoint_required: bool,
    approx_tokens: Optional[int], focus_topic: Optional[str], force: bool, bypass_cooldown: bool,
    commit_fence: Optional[CompressionCommitFence], hard_cancel_event: Any, system_message: str,
    attempt: _Attempt,
) -> _SummaryPhase:
    """Adopt a grown durable parent, gather memory context and run the summarizer.
    A hard cancel restores the compressor snapshot + live list, records a stall backoff while the lease is
    still held, and aborts; any other failure releases the lease and re-raises."""
    pre_msg_count = len(messages)
    _activity_heartbeat: Optional[_CompressionActivityHeartbeat] = None
    messages_before_compression = None

    def _stop_heartbeat(desc: str) -> None:
        nonlocal _activity_heartbeat
        if _activity_heartbeat is not None:
            _activity_heartbeat.stop(desc)
            _activity_heartbeat = None

    try:
        lease.start_refresher()
        if not in_place:
            _adopted_parent = _adopt_grown_durable_parent(agent, lease, messages)
            if _adopted_parent is not None:
                messages = _adopted_parent
                pre_msg_count = len(messages)
                # Estimate was for the stale snapshot; force re-derivation from adopted rows.
                approx_tokens = 0
                # Adopted list is fully durable: re-anchor persist idx at the end so the post-
                # compression flush skips it; run_agent marker sync realigns _session_messages.
                agent._persist_user_message_idx = len(messages)
        memory_context = _pre_compress_memory_context(agent, messages, checkpoint_required)
        compress_fn, compress_kwargs = _resolve_compress_call(
            agent, approx_tokens=approx_tokens, focus_topic=focus_topic, force=force, memory_context=memory_context,
            bypass_cooldown=bypass_cooldown,
        )
        messages_before_compression = copy.deepcopy(messages)
        _activity_heartbeat = _CompressionActivityHeartbeat(
            agent, commit_fence=commit_fence, emit_client_status=lease.status_emitted,
        ).start()
        compressed = _run_summary_dispatch(
            agent, messages, compress_fn, compress_kwargs, commit_fence=commit_fence,
            attempt_generation=attempt.generation, hard_cancel_event=hard_cancel_event,
        )
    except AuxiliaryExplicitCancellation:
        try:
            attempt.restore_compressor(agent.context_compressor)
        except BaseException as _rollback_exc:
            # Compensation failure must surface, but it must not strand the
            # session lease or retain an in-memory transcript mutation.
            _restore_messages_snapshot(messages, messages_before_compression)
            _stop_heartbeat("context compression rollback failed")
            lease.release()
            _emit_aborted_attempt_telemetry(agent, attempt.started_at, f"rollback:{type(_rollback_exc).__name__}")
            raise
        _restore_messages_snapshot(messages, messages_before_compression)
        # Record after restore so rollback cannot wipe a stall backoff, and
        # while the lease is still held so the next turn cannot race it.
        _stall_backoff = _record_stall_interrupted_backoff(
            agent, commit_fence=commit_fence, started_at=attempt.started_at, messages=messages,
            approx_tokens=approx_tokens,
        )
        _stop_heartbeat("context compression cancelled")
        lease.release()
        _emit_aborted_attempt_telemetry(
            agent, attempt.started_at, (STALL_INTERRUPTED_FAILURE_CLASS if _stall_backoff else "explicit_interrupt")
        )
        return _SummaryPhase(messages=messages, abort_prompt=_existing_system_prompt(agent, system_message))
    except BaseException as _compress_exc:
        # Any failure after lock acquisition must release it or the session is permanently blocked from compression.
        _stop_heartbeat("context compression failed")
        lease.release()
        _emit_aborted_attempt_telemetry(agent, attempt.started_at, f"exception:{type(_compress_exc).__name__}")
        raise
    finally:
        _stop_heartbeat("context compression completed")
    return _SummaryPhase(
        messages=messages, compressed=compressed, messages_before_compression=messages_before_compression,
        approx_tokens=approx_tokens, pre_msg_count=pre_msg_count,
    )


@dataclasses.dataclass
class _Attempt:
    """Per-attempt ownership state threaded through the compress_context phases."""

    snapshot: dict
    generation: int
    started_at: float
    durable_cooldown_authoritative: Optional[bool] = None
    durable_cooldown_state: Optional[dict[str, Any]] = None

    def restore_compressor(self, compressor: Any) -> None:
        """Roll the compressor back to this attempt's snapshot (durable cooldown included)."""
        _restore_compressor_attempt_state(
            compressor, self.snapshot, durable_cooldown_authoritative=self.durable_cooldown_authoritative,
            durable_cooldown_state=self.durable_cooldown_state, attempt_generation=self.generation,
        )


def _begin_compression_attempt(agent: Any, *, force: bool, defer_notification: bool) -> _Attempt:
    """Snapshot + claim the compressor, reset per-attempt agent signals, seed telemetry.
    The claim stops a late-unwinding sibling (stall-fallback overlap) from restoring its snapshot over ours or
    clearing our cancellation consult. Signals are cleared at the VERY TOP, before codex/breaker
    early-returns, so a stale value cannot make a later no-op look like lock contention;
    ``_last_compression_attempt_in_place=None`` means aborted/no boundary for
    ``conversation_history_after_compression()``."""
    snapshot = _snapshot_compressor_attempt_state(agent.context_compressor)
    generation = _claim_compressor_attempt(agent.context_compressor)
    if defer_notification and callable(getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)):
        raise RuntimeError("a compression notification is already pending")
    agent._last_compression_attempt_recorded = True
    agent._last_compression_attempt_in_place = None
    agent._compression_skipped_due_to_lock = None
    # Clear the lock-skip signal at the VERY TOP, before the codex route and the breaker gates below can
    # early-return (per-attempt state rule, #58630/#69853). A stale ``True``/holder value from a prior
    # lock-skip must never make a later breaker/codex no-op look like lock contention to the automatic-path
    # consumers (compression_deferred, #49874) — the second clear before lock acquisition below stays for
    # the same reason it was added in #69870 and is simply idempotent now.
    # Transient-block signal (#97488): cleared with the same per-attempt rule; set by the breaker gates
    # below when a TRANSIENT guard (cooldown / structural backoff) no-ops this pass.
    agent._compression_blocked_transient = None
    started_at = time.monotonic()
    attempt_id = uuid.uuid4().hex
    with contextlib.suppress(Exception):
        agent._compression_attempt_id = attempt_id
        agent.context_compressor._compression_telemetry_seed = {
            "attempt_id": attempt_id, "session_id": agent.session_id or "",
            "trigger_source": "manual" if force else "auto",
        }
    return _Attempt(snapshot, generation, started_at)


def _route_codex_compaction(
    agent: Any, messages: list, system_message: str, *, commit_fence: Optional[CompressionCommitFence],
    attempt: _Attempt, approx_tokens: Optional[int], task_id: str, force: bool,
) -> Tuple[list, str]:
    """Codex owns the real thread: run its own compact under the commit fence bracket."""
    if commit_fence is not None and not commit_fence.begin_commit(getattr(agent, "_hard_interrupt_requested", None)):
        attempt.restore_compressor(agent.context_compressor)
        return messages, _existing_system_prompt(agent, system_message)
    try:
        return _compress_context_via_codex_app_server(
            agent, messages, system_message, approx_tokens=approx_tokens, task_id=task_id, force=force
        )
    finally:
        if commit_fence is not None:
            commit_fence.finish_commit()


def _announce_compression_start(
    agent: Any, *, message_count: int, approx_tokens: Optional[int], focus_topic: Optional[str], force: bool
) -> _CompactionLifecycle:
    """Log the attempt, emit the (engine-customisable) compacting status, return the lifecycle."""
    logger.info(
        "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r", agent.session_id or "none",
        message_count, f"{approx_tokens:,}" if approx_tokens else "unknown", agent.model, focus_topic,
    )
    status = COMPACTION_STATUS
    if not force:
        status = automatic_compaction_status_message(
            agent.context_compressor, phase="compress", default_message=status, approx_tokens=approx_tokens,
            message_count=message_count, model=agent.model, focus_topic=focus_topic,
        )
    if status:
        # Never-raise wrapper: a status sink failure must not abort compaction.
        with contextlib.suppress(Exception):
            agent._emit_status_kind(COMPACTION_STATUS_KEY, status, origin="_announce_compression_start")
    return _CompactionLifecycle(agent, bool(status))


def compress_context(
    agent: Any, messages: list, system_message: str, *, approx_tokens: Optional[int] = None,
    task_id: str = "default", focus_topic: Optional[str] = None, force: bool = False,
    bypass_cooldown: bool = False, defer_context_engine_notification: bool = False,
    commit_fence: Optional[CompressionCommitFence] = None,
) -> Tuple[list, str]:
    """Compress conversation context and split the session in SQLite.
    ``force`` (manual /compress) clears the summary-failure cooldown; ``bypass_cooldown`` (provider-proven
    overflow) skips it once, breakers still apply. ``commit_fence`` stops a timed-out worker mutating session
    state. Returns ``(messages, system_prompt)``; on abort input is unchanged, NOT split.

    Args: agent: The owning :class:`AIAgent`. messages: Current message history (will be summarised).
    system_message: Current system prompt; used when compression needs a rebuilt cached prompt.
    approx_tokens: Pre-compression token estimate, logged for ops. task_id: Tool task scope (used for
    clearing file-read dedup state). focus_topic: Optional focus string for guided compression — the
    summariser will prioritise preserving information related to this topic. Inspired by Claude Code's
    ``/compact <focus>``. force: If True, bypass any active summary-failure cooldown. Set by the manual
    ``/compress`` slash command so users can retry immediately after an auto-compress abort. Auto-compress
    callers use the default ``False``. bypass_cooldown: If True, the automatic breaker gates ignore ONLY the
    summary-failure cooldown for this attempt (#100661). Set by the provider-proven overflow recovery path:
    the provider already rejected the request, so deferring until the cooldown lapses wedges the session.
    Unlike ``force`` it does not clear the cooldown, and the ineffective/structural breakers still apply; a
    failed attempt records its cooldown normally. defer_context_engine_notification: Delay the existing
    context-engine hook until a manual host commits its outer history transaction. commit_fence: Optional
    cooperative fence for executor callers that may time out. It prevents a late worker from mutating
    session state after its caller has moved on.
    """
    attempt = _begin_compression_attempt(agent, force=force, defer_notification=defer_context_engine_notification)

    # The Claude Code CLI owns the authoritative conversation and compacts it
    # itself. Automatic Hermes compaction would rewrite only the local mirror,
    # spend an auxiliary call, and rebuild the system prompt (forcing an SDK
    # session respawn) without shrinking the provider context. Manual
    # ``/compress`` remains an explicit request and therefore still runs.
    if not force and getattr(agent, "api_mode", None) == "claude_agent_sdk":
        attempt.restore_compressor(agent.context_compressor)
        logger.info(
            "claude-agent-sdk lane: skipping Hermes auto-compaction "
            "(CLI owns context; use /compress to force). session=%s "
            "messages=%d est_tokens=~%s",
            getattr(agent, "session_id", None) or "none",
            len(messages),
            f"{approx_tokens:,}" if approx_tokens else "unknown",
        )
        return messages, _existing_system_prompt(agent, system_message)

    # Codex owns the real thread; route compaction to its own compact (config
    # compression.codex_app_server_auto). Memory handoff is Hermes-only: no native
    # summary prompt to inject into. `is True`: MagicMock attributes are truthy.
    checkpoint_required = getattr(agent, "compression_checkpoint_required", False) is True
    if getattr(agent, "api_mode", None) == "codex_app_server":
        if checkpoint_required:
            raise _checkpoint_blocked(
                "codex_app_server owns the authoritative thread and does not expose a truthful pre-compaction transcript boundary"
            )
        return _route_codex_compaction(
            agent, messages, system_message, commit_fence=commit_fence, attempt=attempt, approx_tokens=approx_tokens,
            task_id=task_id, force=force,
        )

    # All automatic entrypoints honor compressor cooldown/breaker state; hygiene's
    # fresh AIAgent loads the persisted streak via bind_session_state() first.
    if not force and _automatic_compression_gate_blocks(agent, bypass_cooldown):
        return messages, _existing_system_prompt(agent, system_message)

    # Lazy feasibility probe (~400ms cold) on first attempt, not __init__; it sets
    # _compression_warning so status replay still surfaces the warning. Marked checked
    # only after the probe completes (transient failures are swallowed inside).
    if not getattr(agent, "_compression_feasibility_checked", False):
        check_compression_model_feasibility(agent)
        agent._compression_feasibility_checked = True
    _pre_msg_count = len(messages)
    # In-place keeps the SAME session_id (no rotation/child/renumber/re-sync). A
    # missing attribute must default True, not rotation, which can wedge sessions.
    in_place = bool(getattr(agent, "compression_in_place", True))
    lifecycle = _announce_compression_start(
        agent, message_count=_pre_msg_count, approx_tokens=approx_tokens, focus_topic=focus_topic, force=force
    )
    lease, _abort_prompt = _acquire_compression_lease(
        agent, commit_fence=commit_fence, lifecycle=lifecycle, system_message=system_message,
        approx_tokens=approx_tokens, attempt_started_at=attempt.started_at,
    )
    if lease is None:
        return messages, _abort_prompt

    # Publish the holder-qualified release hook before a timeout can win the
    # fence. If no durable lock was acquired there is no hook to publish.
    lease.finish_lock_setup()
    _adopted = _adopt_if_parent_rotated(agent, lease, messages, system_message)
    if _adopted is not None:
        return _adopted

    # Snapshot durable cooldown only once we own the lease. Runs for force=True
    # too but skips the automatic breaker gate: manual compression retries now.
    attempt.durable_cooldown_authoritative, attempt.durable_cooldown_state = (
        _capture_authoritative_cooldown_under_lease(agent.context_compressor, attempt.snapshot)
    )
    if attempt.durable_cooldown_authoritative is False:
        # Durable cooldown read failed under a built-in compressor: force=True could
        # clear an unknown newer row before cancellation could restore it. Abort.
        lease.release()
        return messages, _existing_system_prompt(agent, system_message)

    # Another path may have compacted this session in place since construction;
    # re-read breaker state under the lock, not the bind_session_state() snapshot.
    if not force and _automatic_compression_gate_blocks(agent, bypass_cooldown, include_cooldown=False):
        lease.release()
        return messages, _existing_system_prompt(agent, system_message)

    # Interrupts/redirects must not tear a summary in half. Use the explicit stop
    # Event (message fields race) + fence timeout so pool slots free promptly.
    # Explicit stop surfaces set a separate Event atomically; never infer cause from the racy message
    # fields. A host timeout also cancels the attempt's commit fence. Feed BOTH into the protected
    # auxiliary-call seam so the compression owner unwinds promptly while an isolated provider stream
    # finishes or closes in its daemon worker. Otherwise four timed-out streams retain all four shared
    # compression-pool slots until the auxiliary stream's longer absolute ceiling expires. See #23975.
    _hard_cancel_event = getattr(agent, "_hard_interrupt_requested", None)
    phase = _run_summary_phase(
        agent, messages, lease=lease, in_place=in_place, checkpoint_required=checkpoint_required,
        approx_tokens=approx_tokens, focus_topic=focus_topic, force=force, bypass_cooldown=bypass_cooldown,
        commit_fence=commit_fence, hard_cancel_event=_hard_cancel_event, system_message=system_message, attempt=attempt,
    )
    if phase.abort_prompt is not None:
        return phase.messages, phase.abort_prompt
    messages, compressed = phase.messages, phase.compressed
    messages_before_compression = phase.messages_before_compression
    approx_tokens, _pre_msg_count = phase.approx_tokens, phase.pre_msg_count
    _commit_fence_entered = False
    try:
        # Capture the verdict before rotation callbacks: lifecycle hooks may reset
        # compressor fields on rebind; record only after the full boundary commits.
        _compression_made_progress, _compression_used_fallback, _compression_feasibility_skip = (
            bool(getattr(agent.context_compressor, name, False))
            for name in ("_last_compression_made_progress", "_last_summary_fallback_used", "_last_feasibility_skip")
        )
        if _candidate_rejected(
            agent, compressed, messages, messages_before_compression, attempt_generation=attempt.generation,
            attempt_started_at=attempt.started_at,
        ):
            return messages, _existing_system_prompt(agent, system_message)
        if commit_fence is not None:
            _commit_fence_entered = commit_fence.begin_commit(_hard_cancel_event)
            if not _commit_fence_entered:
                attempt.restore_compressor(agent.context_compressor)
                _restore_messages_snapshot(messages, messages_before_compression)
                logger.info(
                    "Compression commit cancelled before session mutation (session=%s).", agent.session_id or "none"
                )
                agent._last_compaction_in_place = False
                _stall_backoff = _record_stall_interrupted_backoff(
                    agent, commit_fence=commit_fence, started_at=attempt.started_at, messages=messages,
                    approx_tokens=approx_tokens,
                )
                _existing_sp = _existing_system_prompt(agent, system_message)
                _emit_aborted_attempt_telemetry(
                    agent, attempt.started_at,
                    STALL_INTERRUPTED_FAILURE_CLASS if _stall_backoff else "commit_fence_cancelled",
                )
                return messages, _existing_sp
        _warn_summary_or_aux_fallback(agent)
        _fold_todo_snapshot(agent, compressed)
        compressed_user_turn_outcome = _ensure_compressed_has_user_turn(messages, compressed)
        new_system_prompt = _rebuild_system_prompt_at_boundary(agent, system_message)
        commit = _commit_compaction(
            agent, messages, compressed, in_place=in_place, lease=lease, new_system_prompt=new_system_prompt,
            system_message=system_message, compressed_user_turn_outcome=compressed_user_turn_outcome,
            messages_before_compression=messages_before_compression, made_progress=_compression_made_progress,
            attempt=attempt,
        )
        if commit.refused_prompt is not None:
            return messages, commit.refused_prompt
        compressed = commit.compressed
        split_status = commit.split_status
        _compressed_est = _finish_compaction_boundary(
            agent, compressed, new_system_prompt=new_system_prompt, old_session_id=commit.old_session_id,
            in_place=in_place, compacted_in_place=commit.compacted_in_place,
            session_commit_succeeded=commit.session_commit_succeeded,
            defer_context_engine_notification=defer_context_engine_notification,
            compression_made_progress=commit.made_progress, compression_used_fallback=_compression_used_fallback,
            compression_feasibility_skip=_compression_feasibility_skip, task_id=task_id,
        )
        logger.info(
            "context compression done: session=%s messages=%d->%d rough_tokens=~%s awaiting_real_usage=true",
            agent.session_id or "none", _pre_msg_count, len(compressed), f"{_compressed_est:,}",
        )
        lifecycle.commit_status = (
            "committed" if split_status in {"not_applicable", "in_place_committed", "rotated_committed"} else "aborted"
        )
        _emit_compression_attempt_telemetry(
            agent, started_at=attempt.started_at, commit_status=lifecycle.commit_status, split_status=split_status,
            failure_class=("session_split_failed" if split_status in {"failed_not_indexed", "aborted"} else None),
            commit_started_at=commit.commit_started_at,
        )
        return compressed, new_system_prompt
    finally:
        # Release the OLD session's lock only after rotation and all post-rotation
        # bookkeeping; a waking contender then sees the NEW id and acquires on that.
        try:
            lease.release()
        finally:
            if _commit_fence_entered:
                commit_fence.finish_commit()


def _codex_compaction_cooldown_remaining(agent: Any) -> float:
    """Seconds left on this session's compaction-failure cooldown (0 = clear)."""
    compressor = getattr(agent, "context_compressor", None)
    getter = getattr(compressor, "get_active_compression_failure_cooldown", None)
    if not callable(getter):
        return 0.0
    try:
        state = getter(refresh=True)
    except Exception:
        logger.debug("codex compaction cooldown lookup failed", exc_info=True)
        return 0.0
    try:
        return max(0.0, float(state.get("remaining_seconds") or 0.0)) if state else 0.0
    except (TypeError, ValueError):
        return 0.0


def _record_codex_compaction_failure(agent: Any, error: str) -> None:
    """Arm the shared compression-failure cooldown after a failed codex compaction.
    The codex path returns the transcript unchanged, so without a cooldown the still-over-threshold session
    would retry every turn."""
    from agent.context_compressor import _SUMMARY_FAILURE_COOLDOWN_SECONDS
    compressor = getattr(agent, "context_compressor", None)
    recorder = getattr(compressor, "_record_compression_failure_cooldown", None)
    if not callable(recorder):
        return
    with _swallow('codex compaction cooldown persist failed', exc_info=True):
        recorder(_SUMMARY_FAILURE_COOLDOWN_SECONDS, error)


def _compress_context_via_codex_app_server(
    agent: Any, messages: list, system_message: Optional[str], *, approx_tokens: Optional[int] = None,
    task_id: str = "default", force: bool = False,
) -> Tuple[list, str]:
    """Route compaction to Codex app-server for Codex-owned threads.
    Rewriting the local transcript would not shrink the Codex thread, so Codex compacts its own thread and
    Hermes' transcript is left unchanged."""
    _sid = getattr(agent, "session_id", None) or "none"
    _tokens = f"{approx_tokens:,}" if approx_tokens else "unknown"
    auto_mode = str(getattr(agent, "codex_app_server_auto_compaction", "native") or "native").lower()
    if auto_mode not in {"native", "hermes", "off"}:
        auto_mode = "native"
    skip_reason = None
    if not force and auto_mode != "hermes":
        skip_reason = f"mode={auto_mode} force=false"
    elif not force:
        # Automatic entrypoints honor the compressor-owned cooldown: a recent compaction
        # failed, and retrying every turn is what thrashes.
        _cooldown_remaining = _codex_compaction_cooldown_remaining(agent)
        if _cooldown_remaining > 0:
            skip_reason = f"failure cooldown active for {_cooldown_remaining:.0f}s"
    codex_session = getattr(agent, "_codex_session", None)
    if skip_reason is None and codex_session is None:
        skip_reason = "no active codex thread"
    if skip_reason is not None:
        logger.info(
            "codex app-server compaction skipped: %s (session=%s messages=%d tokens=~%s)", skip_reason, _sid,
            len(messages), _tokens,
        )
        return messages, _existing_system_prompt(agent, system_message)
    logger.info("codex app-server compaction started: session=%s messages=%d tokens=~%s", _sid, len(messages), _tokens)
    with contextlib.suppress(Exception):
        agent._emit_status_kind(COMPACTION_STATUS_KEY, COMPACTION_STATUS, origin="codex_thread_compaction")
    _activity_heartbeat = _CompressionActivityHeartbeat(agent, emit_client_status=True).start()
    try:
        result = codex_session.compact_thread()
    except BaseException:
        _activity_heartbeat.stop("context compression failed")
        raise
    failed = bool(getattr(result, "interrupted", False) or getattr(result, "error", None))
    _activity_heartbeat.stop("context compression failed" if failed else "context compression completed")
    if getattr(result, "should_retire", False):
        with contextlib.suppress(Exception):
            codex_session.close()
        agent._codex_session = None
    if failed:
        with contextlib.suppress(Exception):
            agent._emit_warning(f"⚠ Codex app-server compaction failed: {result.error}")
        # The transcript is returned unchanged, so the session is still over
        # threshold. Without a brake the next turn retries immediately.
        _record_codex_compaction_failure(agent, str(getattr(result, "error", None) or "compaction interrupted"))
        return messages, _existing_system_prompt(agent, system_message)
    with _swallow('codex compaction bookkeeping failed', exc_info=True):
        from agent.codex_runtime import _record_codex_app_server_compaction, _record_codex_app_server_usage
        _record_codex_app_server_compaction(agent, result, approx_tokens=approx_tokens, force=True)
        # An empty usage report must consume the pending verdict, not leave deferral
        # armed until a later turn; minimal test engines may lack update_from_response.
        if hasattr(agent.context_compressor, "update_from_response"):
            _record_codex_app_server_usage(agent, result)
    _reset_read_dedup_caches(task_id, skills=False)
    logger.info(
        "codex app-server compaction done: session=%s thread=%s turn=%s", _sid,
        getattr(result, "thread_id", None) or "", getattr(result, "turn_id", None) or "",
    )
    existing_prompt = _existing_system_prompt(agent, system_message)
    # Terminal edge only on success — failure/interrupt paths above return
    # without it, matching the main compress_context() gating.
    _emit_compaction_done(agent)
    return messages, existing_prompt


# 4 MB leaves headroom under Anthropic's 5 MB; shrinking loses quality but only
# runs after a confirmed provider rejection, so the alternative is failure.
_IMAGE_SHRINK_TARGET_BYTES = 4 * 1024 * 1024
_IMAGE_SUFFIX_BY_MIME = {
    "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/bmp": ".bmp",
}


def _data_url_mime(header: str, default: str = "image/jpeg") -> str:
    """``image/*`` mime from a ``data:`` URL header, else ``default``."""
    if header.startswith("data:"):
        candidate = header[len("data:") :].split(";", 1)[0].strip()
        if candidate.startswith("image/"):
            return candidate
    return default


def _decode_pixels(data_url: str) -> Optional[tuple]:
    """``(width, height)`` of a base64 data URL; None when Pillow is missing or the payload is corrupt."""
    try:
        import base64, io
        _, _, data_d = data_url.partition(",")
        if not data_d or not data_url.startswith("data:"):
            return None
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(data_d))) as _img:
            return _img.size
    except Exception:
        return None


def _shrink_data_url(url: str, *, max_dimension: int, resize_fn: Any) -> tuple:
    """Return ``(resized_url, unshrinkable)`` for a data URL.
    ``resized_url`` is None when no rewrite applied. ``unshrinkable`` is True only when the image violated a
    constraint and resizing failed to satisfy that same constraint, so the caller knows a retry is pointless.
    The accept gate MUST use the axis that triggered the shrink: a pixel downscale can re-encode to MORE bytes
    (PNG non-monotonic); a byte-only reject wedges."""
    target_bytes = _IMAGE_SHRINK_TARGET_BYTES
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, False
    triggered_by = "bytes" if len(url) > target_bytes else None  # over byte budget
    if triggered_by is None:
        # Bytes fine; check pixels against the provider cap (tiny bytes, huge pixels).
        dims = _decode_pixels(url)
        if dims is None or max(dims) <= max_dimension:
            return None, False
        triggered_by = "dimension"
    try:
        header, _, data = url.partition(",")
        mime = _data_url_mime(header)
        import base64 as _b64
        raw = _b64.b64decode(data)
        tmp = tempfile.NamedTemporaryFile(
            prefix="hermes_shrink_", suffix=_IMAGE_SUFFIX_BY_MIME.get(mime, ".jpg"), delete=False
        )
        try:
            tmp.write(raw)
            tmp.close()
            resized = resize_fn(
                Path(tmp.name), mime_type=mime, max_base64_bytes=target_bytes, max_dimension=max_dimension
            )
        finally:
            with contextlib.suppress(Exception):
                Path(tmp.name).unlink(missing_ok=True)
        if not resized:
            return None, True  # Pillow couldn't help
        new_dims = _decode_pixels(resized)
        if triggered_by == "bytes":
            # Byte budget is binding — bytes must shrink; and the resizer may return an
            # over-cap blob (long side freezes at the 64px short-side floor) → still 400.
            ok = len(resized) < len(url) and (new_dims is None or max(new_dims) <= max_dimension)
        elif new_dims is not None:
            # Dimension cap is binding: accept a byte-larger re-encode if now within cap.
            ok = max(new_dims) <= max_dimension
        else:
            # Can't verify dimensions: fall back to the bytes-must-shrink gate so we never
            # accept an unverifiable byte-larger blob.
            ok = len(resized) < len(url)
        return (resized, False) if ok else (None, True)
    except Exception as exc:
        logger.warning("image-shrink recovery: re-encode failed — %s", exc)
        return None, triggered_by is not None


def _source_to_data_url(source: Any) -> Optional[str]:
    """Anthropic ``{"type": "base64", ...}`` image source → data URL, else None."""
    if not isinstance(source, dict) or source.get("type") != "base64":
        return None
    data = source.get("data")
    if not isinstance(data, str) or not data:
        return None
    media_type = str(source.get("media_type") or "image/jpeg").strip()
    return f"data:{media_type if media_type.startswith('image/') else 'image/jpeg'};base64,{data}"


def _write_data_url_to_source(source: dict, data_url: str) -> dict:
    """Return a NEW source dict carrying the re-encoded payload.
    Copy-on-write: parts may be shared with the persistent history, so mutating in place would store the
    degraded image; the caller replaces the part."""
    header, _, data = data_url.partition(",")
    return {**source, "type": "base64", "media_type": _data_url_mime(header), "data": data}


def try_shrink_image_parts_in_messages(api_messages: list, *, max_dimension: int = 8000) -> bool:
    """Re-encode oversized native image parts to recover from image-too-large errors.
    Mutates ``api_messages`` in place. Returns True if any part was replaced, False if nothing to shrink or
    Pillow could not help. Targets data-URL parts over 4 MB or ``max_dimension`` (Anthropic's per-side pixel
    cap, parsed from the rejection by the caller); http(s) image URLs are left untouched."""
    if not api_messages:
        return False
    try:
        from tools.vision_tools import _resize_image_for_vision
    except Exception as exc:
        logger.warning("image-shrink recovery: vision_tools unavailable — %s", exc)
        return False
    changed_count = 0
    # Track over-target parts that could not be shrunk: if any remain, a retry
    # re-sends the same payload and wastes the single retry budget.
    unshrinkable_oversized = 0

    def _shrink(url: Any) -> tuple:
        return _shrink_data_url(url, max_dimension=max_dimension, resize_fn=_resize_image_for_vision)

    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        # Copy-on-write: part/source dicts can alias stored history, so build a new
        # content list and reassign msg["content"] on the per-call copy.
        new_content: list | None = None
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            replacement = None
            if ptype == "image":
                source = part.get("source")
                resized, unshrinkable = _shrink(_source_to_data_url(source) or "")
                if resized and isinstance(source, dict):
                    replacement = {**part, "source": _write_data_url_to_source(source, resized)}
            elif ptype in {"image_url", "input_image"}:
                image_value = part.get("image_url")
                # OpenAI chat.completions: {"image_url": {"url": "data:..."}}
                # OpenAI Responses: {"image_url": "data:..."}
                if isinstance(image_value, dict):
                    resized, unshrinkable = _shrink(image_value.get("url", ""))
                    if resized:
                        replacement = {**part, "image_url": {**image_value, "url": resized}}
                elif isinstance(image_value, str):
                    resized, unshrinkable = _shrink(image_value)
                    if resized:
                        replacement = {**part, "image_url": resized}
                else:
                    continue
            else:
                continue
            if replacement is not None:
                if new_content is None:
                    new_content = list(content)
                new_content[part_idx] = replacement
                changed_count += 1
            elif unshrinkable:
                unshrinkable_oversized += 1
        if new_content is not None:
            msg["content"] = new_content
    target_mb = _IMAGE_SHRINK_TARGET_BYTES / (1024 * 1024)
    if changed_count:
        logger.info("image-shrink recovery: re-encoded %d image part(s) to fit under %.0f MB", changed_count, target_mb)
    if unshrinkable_oversized:
        # An unshrinkable oversized image makes retry pointless; signal no progress even
        # if others shrank so the caller surfaces the original error.
        logger.warning(
            "image-shrink recovery: %d oversized image part(s) could not be "
            "shrunk under %.0f MB — not retrying (would re-send rejected payload)", unshrinkable_oversized,
            target_mb,
        )
        return False
    return changed_count > 0


__all__ = [
    "COMPACTION_STATUS", "COMPACTION_STATUS_KEY", "COMPACTION_DONE_STATUS", "COMPACTION_HEARTBEAT_STATUS", "COMPACTION_STATUS_MARKER", "is_compaction_progress_status",
    "check_compression_model_feasibility", "replay_compression_warning", "compress_context",
    "try_shrink_image_parts_in_messages",
]


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

class CompressionExecutorSaturatedError(RuntimeError):
    """All compression pool slots are occupied; submission was refused."""
# ---- END PLUGIN-COMPAT ----
