"""Context compression — extract the AIAgent methods that drive summarisation.

Three concerns live here:

* :func:`check_compression_model_feasibility` — startup probe of the
  configured auxiliary compression model.  Warns when the aux context
  window can't fit the main model's compression threshold; auto-lowers
  the session threshold when possible; hard-rejects auxes below
  ``MINIMUM_CONTEXT_LENGTH``.

* :func:`replay_compression_warning` — re-emit a stored warning through
  the gateway ``status_callback`` once it's wired up (the callback is
  set after :class:`AIAgent` construction).

* :func:`compress_context` — the actual compression call.  Runs the
  configured compressor, splits the SQLite session, rotates the
  session_id, notifies plugin context engines / memory providers, and
  returns the compressed message list and active system prompt.

* :func:`try_shrink_image_parts_in_messages` — image-too-large recovery
  helper that re-encodes ``data:image/...;base64,...`` parts at a smaller
  size so retries can fit under provider ceilings (Anthropic's 5 MB).

``run_agent`` keeps thin wrappers for each so existing call sites
(``self._compress_context(...)``) keep working.  Tests that exercise
these paths see no behavioural change.

Thread-safety contract for extension points (#76354 review)
------------------------------------------------------------

When the host-level progress-aware timeout is enabled (the default:
``compression.context_timeout_seconds > 0``), the WHOLE compression pass —
including plugin/legacy **context engines** (``compress()`` /
``on_session_start`` / boundary callbacks) and **memory providers**
(``on_pre_compress`` / ``on_session_switch``) — runs on a pooled daemon
thread, not the conversation thread. Extension authors must assume:

* Calls may arrive on an arbitrary pooled thread; do not rely on
  thread-affinity or ``threading.local`` state shared with the caller.
* The input message list is a private deep snapshot owned by the worker;
  engines MAY mutate it in place (legacy contract preserved), and that
  mutation is invisible to the live conversation unless the pass commits.
* Publication to caller-visible / durable state happens ONLY on an admitted
  commit (:class:`CompressionCommitFence`); after a host timeout the still-
  running engine's work is discarded.
* Two compression passes never run concurrently for one session (durable
  per-session lock), but passes for DIFFERENT sessions may run concurrently
  on pool siblings — engine/provider instances shared across sessions must
  be thread-safe or internally locked.
"""

from __future__ import annotations

import concurrent.futures
import copy
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
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_engine import (
    automatic_compaction_status_message,
    sanitize_memory_context,
)
from agent.memory_provider import PRE_COMPRESS_CHECKPOINT_API_VERSION
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)
from agent.session_activity import ActivityProvenance, normalize_activity_provenance

logger = logging.getLogger(__name__)

# Terminal compression outcomes published by host/hygiene timeout or cooldown
# writers. Detached heartbeat workers must not clobber these back to
# agent.compression after cancel (otherwise timeout is unobservable). Observing
# a terminal stamp (or a cancelled commit fence) also latches the heartbeat
# silent so a later UNKNOWN rewrite cannot re-arm a zombie worker.
_TERMINAL_COMPRESSION_PROVENANCES = frozenset(
    {
        ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
    }
)

# Stable marker the gateway matches on to re-tag the auto-compaction lifecycle
# status as ``kind="compacting"`` (tui_gateway/server.py::_status_update), so
# drivers like the desktop app can show an explicit "Summarizing…" indicator
# instead of the transcript appearing to silently reset. Keep the marker phrase
# intact if you reword COMPACTION_STATUS.
COMPACTION_STATUS_MARKER = "Compacting context"
COMPACTION_STATUS = (
    f"🗜️ {COMPACTION_STATUS_MARKER} — summarizing earlier conversation so I can continue..."
)

COMPACTION_DONE_STATUS = "✓ Context compaction complete — continuing turn..."


def _strip_marker_for_comparison(msgs: Any) -> Any:
    """Copy ``msgs`` with the ``_db_persisted`` persistence marker removed.

    Used by the no-op progress check: live and loaded dicts carry the marker
    (loaded rows are stamped at materialization time, #92231) while
    ``compress()`` output is marker-swept, so a raw ``==`` would misclassify a
    semantically-identical no-op copy as progress. Non-list inputs and
    non-dict entries pass through unchanged.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER

    if not isinstance(msgs, list):
        return msgs
    return [
        {k: v for k, v in m.items() if k != _DB_PERSISTED_MARKER}
        if isinstance(m, dict)
        else m
        for m in msgs
    ]


def _emit_compaction_done(agent: Any) -> None:
    """Emit the structured terminal edge for a started compaction."""
    status_callback = getattr(agent, "status_callback", None)
    if not status_callback:
        return
    try:
        status_callback("compacted", COMPACTION_DONE_STATUS)
    except Exception:
        logger.debug("status_callback error in compaction completion", exc_info=True)


# ── Routine compression status templates ────────────────────────────────────
# Every ROUTINE (non-failure, non-manual-/compress) compression status line the
# agent emits lives here so the gateway noise filter and its tests can couple
# to the real emitted wording instead of hand-copied literals. These are
# suppressed on human-facing chat platforms by _TELEGRAM_NOISY_STATUS_RE
# (gateway/run.py) — when rewording ANY of them, update that regex and the
# pinned data in tests/gateway/test_telegram_noise_filter.py in the same PR.
# Failure notices (⚠ Compression aborted / empty transcript / codex compaction
# failed) and manual /compress feedback (manual_compression_feedback.py) are
# deliberate carve-outs from silence and must NOT be added here.
PRE_API_COMPRESSION_STATUS_TEMPLATE = (
    "📦 Pre-API compression: ~{tokens:,} tokens "
    "near the context/output limit. Compacting before the next model call."
)
PREFLIGHT_COMPRESSION_STATUS_TEMPLATE = (
    "📦 Preflight compression: ~{tokens:,} tokens "
    ">= {threshold:,} threshold. This may take a moment."
)
IDLE_COMPACTION_STATUS_TEMPLATE = (
    "💤 Resumed after {idle_seconds}s idle — compacting "
    "~{tokens:,} tokens before continuing."
)
COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE = (
    "🗜️ Context too large (~{tokens:,} tokens) — compressing ({attempt}/{cap})..."
)
COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE = (
    "🗜️ Compressed {before} → {after} messages, retrying..."
)
COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE = (
    "🗜️ Compressed ~{before:,} → ~{after:,} tokens, retrying..."
)
COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE = (
    "🗜️ Context reduced to {new_ctx:,} tokens (was {old_ctx:,}), retrying..."
)

# FAILURE-CLASS notice — a deliberate carve-out from routine-compression
# silence (#16775 class): the context is over the compression threshold but
# compression is blocked (summary-LLM cooldown / anti-thrash breaker), so the
# session will keep growing until the hard provider token limit kills it.
# This MUST stay visible on chat gateways. Do NOT add it to
# ROUTINE_COMPRESSION_STATUS_SAMPLES or the gateway noise regex
# (_TELEGRAM_NOISY_STATUS_RE); it is pinned un-swallowed in
# tests/gateway/test_telegram_noise_filter.py::VISIBLE_COMPRESSION_MESSAGES.
CONTEXT_OVERFLOW_BLOCKED_WARNING_TEMPLATE = (
    "⚠ Context is over the compression threshold "
    "(~{tokens:,} tokens >= {threshold:,}) "
    "but compression is currently blocked ({reason}). "
    "The model may stop responding. Run /new to start a fresh "
    "session or /compress to retry immediately."
)

# Sample-formatted instances of every routine compression status line, for
# behavioral tests that iterate the ACTUAL emitted wording (formatted from the
# same constants the emission sites use) through the gateway noise filter.
ROUTINE_COMPRESSION_STATUS_SAMPLES = (
    COMPACTION_STATUS,
    COMPACTION_DONE_STATUS,
    PRE_API_COMPRESSION_STATUS_TEMPLATE.format(tokens=123456),
    PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(tokens=120000, threshold=100000),
    IDLE_COMPACTION_STATUS_TEMPLATE.format(idle_seconds=3600, tokens=120000),
    COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE.format(tokens=250000, attempt=1, cap=3),
    COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE.format(before=30, after=12),
    COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE.format(before=250000, after=120000),
    COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE.format(
        new_ctx=120000, old_ctx=250000
    ),
)


def _builtin_memory_prompt_snapshot(agent: Any) -> Optional[Tuple[str, str]]:
    """Return the built-in memory text that can affect a system prompt.

    ``MemoryStore`` freezes this text until ``load_from_disk()``.  Rendering
    the frozen blocks after that reload lets compression retain the exact
    cached system prompt when it already embeds the current memory (see
    :func:`_cached_prompt_reflects_builtin_memory`).  An unreadable snapshot
    returns ``None`` so callers take the conservative rebuild path.
    """
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return "", ""
    try:
        memory = (
            store.format_for_system_prompt("memory") or ""
            if getattr(agent, "_memory_enabled", False)
            else ""
        )
        user = (
            store.format_for_system_prompt("user") or ""
            if getattr(agent, "_user_profile_enabled", False)
            else ""
        )
    except Exception:
        return None
    return memory, user


def _cached_prompt_reflects_builtin_memory(agent: Any, cached_prompt: str) -> bool:
    """Whether the cached system prompt already embeds current built-in memory.

    The retention fast path must NOT compare the memory snapshot before vs
    after the disk reload: on fresh-agent surfaces (gateway, TUI) the cached
    prompt is restored from the session DB and can predate mid-session memory
    writes that the fresh ``MemoryStore`` already picked up at init — the
    snapshot is then identical on both sides of the reload while the prompt
    itself is stale, and retaining it would latch old memory for the life of
    the session (and re-persist it via ``update_system_prompt``).

    Instead, verify the CURRENT (post-reload) rendered blocks appear verbatim
    in the cached prompt, and that no leftover block header remains for a
    target whose entries have since been emptied or disabled.
    """
    snapshot = _builtin_memory_prompt_snapshot(agent)
    if snapshot is None:
        return False
    try:
        from tools.memory_tool import MEMORY_BLOCK_HEADERS
    except Exception:
        return False
    for target, block in zip(("memory", "user"), snapshot):
        block = block.strip()
        if block:
            # build_system_prompt_parts embeds the stripped block verbatim;
            # the rendered text includes the usage header, so any entry
            # change (or char-count change) breaks containment → rebuild.
            if block not in cached_prompt:
                return False
        elif MEMORY_BLOCK_HEADERS[target] in cached_prompt:
            # The prompt still carries a block for a target that is now
            # empty/disabled — stale; rebuild.
            return False
    return True


_COMPRESSOR_ATTEMPT_STATE_FIELDS = (
    "_previous_summary",
    "_summary_has_user_turn",
    "compression_count",
    "_last_compression_savings_pct",
    "_ineffective_compression_count",
    "_anti_thrash_recovery_deadline",
    "_fallback_compression_streak",
    "_verify_compaction_cleared_threshold",
    "_last_compression_made_progress",
    "_summary_failure_cooldown_until",
    "_cooldown_persist_failed",
    "_last_summary_error",
    "_consecutive_timeout_failures",
    "_last_summary_dropped_count",
    "_last_summary_fallback_used",
    "_last_compress_aborted",
    "_last_summary_auth_failure",
    "_last_summary_network_failure",
    "_last_summary_empty_content_failure",
    "_last_aux_model_failure_error",
    "_last_aux_model_failure_model",
    "_summary_model_fallen_back",
    "summary_model",
    "_last_compression_telemetry",
    "_active_compression_telemetry",
    "_compression_telemetry_seed",
    "_proactive_prune_rearm_tokens",
)

_COMPRESSOR_COOLDOWN_STATE_FIELDS = (
    "_summary_failure_cooldown_until",
    "_last_summary_error",
    "_cooldown_persist_failed",
)


def _snapshot_compressor_attempt_state(compressor: Any) -> dict[str, Any]:
    """Copy only mutable bookkeeping owned by one compression attempt.

    The explicit allow-list avoids copying provider clients, SessionDB handles,
    locks, and plugin resources. Missing fields are intentionally ignored so
    legacy and third-party compressors keep their existing contract.
    """
    try:
        values = vars(compressor)
    except TypeError:
        return {}
    selected = {
        name: values[name]
        for name in _COMPRESSOR_ATTEMPT_STATE_FIELDS
        if name in values
    }
    # Copy the collection as one object so aliases between fields (notably
    # _active_compression_telemetry and _last_compression_telemetry) survive.
    return copy.deepcopy(selected)


def _restore_compressor_attempt_state(
    compressor: Any,
    snapshot: dict[str, Any],
    *,
    durable_cooldown_authoritative: Optional[bool] = None,
    durable_cooldown_state: Optional[dict[str, Any]] = None,
) -> None:
    """Restore the safe per-attempt snapshot after a pre-commit hard cancel."""
    # A successful summary clears the durable cooldown before the outer commit
    # boundary. Recreate (or clear) that row before restoring exact in-memory
    # values, otherwise the next refresh would overwrite this rollback. Unknown
    # durable state and intentionally unpersisted local cooldowns are never
    # converted into destructive DB writes during cancellation.
    if (
        "_summary_failure_cooldown_until" in snapshot
        and durable_cooldown_authoritative is not False
        and (
            durable_cooldown_authoritative is True
            or not bool(snapshot.get("_cooldown_persist_failed", False))
        )
    ):
        session_db = vars(compressor).get("_session_db")
        session_id = vars(compressor).get("_session_id")
        if session_db is not None and session_id:
            if durable_cooldown_authoritative is True:
                restorer = getattr(
                    type(session_db),
                    "restore_compression_failure_cooldown_row",
                    None,
                )
                if not callable(restorer) or durable_cooldown_state is None:
                    raise RuntimeError(
                        "exact compression cooldown rollback API is unavailable"
                    )
                # This API restores raw columns (including expired and null
                # combinations), verifies the read-back, and propagates failure.
                restorer(
                    session_db,
                    session_id,
                    copy.deepcopy(durable_cooldown_state),
                )
            else:
                try:
                    deadline = float(
                        snapshot["_summary_failure_cooldown_until"] or 0.0
                    )
                    remaining = max(0.0, deadline - time.monotonic())
                    durable_deadline = time.time() + remaining
                    durable_error = snapshot.get("_last_summary_error")
                    if remaining > 0:
                        recorder = getattr(
                            type(session_db),
                            "record_compression_failure_cooldown",
                            None,
                        )
                        if callable(recorder):
                            recorder(
                                session_db,
                                session_id,
                                durable_deadline,
                                durable_error,
                            )
                    else:
                        clearer = getattr(
                            type(session_db),
                            "clear_compression_failure_cooldown",
                            None,
                        )
                        if callable(clearer):
                            clearer(session_db, session_id)
                except Exception:
                    # Legacy/third-party compatibility path: its existing APIs
                    # do not provide a verifiable transaction contract.
                    logger.debug(
                        "compression cooldown persistence rollback failed",
                        exc_info=True,
                    )
    restored = copy.deepcopy(snapshot)
    for name, value in restored.items():
        setattr(compressor, name, value)


def _capture_authoritative_cooldown_under_lease(
    compressor: Any,
    attempt_snapshot: dict[str, Any],
) -> tuple[Optional[bool], Optional[dict[str, Any]]]:
    """Refresh and snapshot built-in durable cooldown state under the lease.

    Third-party compressors are deliberately not invoked here: arbitrary plugin
    callbacks must not run while the session lease is held. A durable read
    failure returns ``False`` so rollback cannot mistake unknown durable state
    for an authoritative empty row and clear it; an unavailable legacy API
    returns ``None`` and preserves the compatibility path.
    """
    try:
        from agent.context_compressor import ContextCompressor

        if not isinstance(compressor, ContextCompressor):
            return None, None
        values = vars(compressor)
        session_db = values.get("_session_db")
        session_id = values.get("_session_id")
        raw_reader = (
            getattr(
                type(session_db), "get_compression_failure_cooldown_row", None
            )
            if session_db is not None
            else None
        )
        if session_db is None or not session_id:
            # Unbound compressors have no durable row to mutate or restore.
            return None, None
        if not callable(raw_reader):
            return False, None
        # Capture the exact persisted representation first. The active getter
        # intentionally filters expired rows and therefore cannot serve as a
        # lossless rollback snapshot.
        durable_state = raw_reader(session_db, session_id)
        if not isinstance(durable_state, dict):
            raise TypeError("raw compression cooldown snapshot must be a mapping")
        ContextCompressor.get_active_compression_failure_cooldown(
            compressor,
            refresh=True,
        )
    except Exception as exc:
        logger.debug("authoritative compression cooldown capture failed: %s", exc)
        return False, None
    authoritative = getattr(
        compressor, "_last_cooldown_refresh_was_authoritative", None
    )
    if authoritative is not True:
        return authoritative, None

    values = vars(compressor)
    for name in _COMPRESSOR_COOLDOWN_STATE_FIELDS:
        if name in values:
            attempt_snapshot[name] = copy.deepcopy(values[name])
    return True, copy.deepcopy(durable_state)


class CompressionCommitFence:
    """Fence timeout cancellation against post-summary session mutation.

    Compression itself is synchronous and may be running in an executor thread.
    A caller can stop waiting for the summary, but it cannot kill that thread.
    This fence makes the commit boundary deterministic: cancellation either wins
    before session mutation starts, or waits until an already-started commit is
    fully complete before the caller proceeds.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._commit_started = False
        # Lock-free commit-phase marker (#76354 review F1). ``begin_commit``
        # RETAINS ``self._lock`` until ``finish_commit``, so any host-side
        # observation that needs the lock (``try_cancel_before_commit``)
        # blocks/space-outs for the whole commit. This Event is set inside
        # ``begin_commit`` while the lock is held but is READABLE WITHOUT the
        # lock, so a host can observe "a commit was admitted and may be in
        # flight" even while the commit itself is hung — which is exactly when
        # the overrun warning must be able to fire.
        self._commit_phase = threading.Event()
        # Lock-free admission revocation (#76354 review F2). Set by
        # :meth:`revoke_commit_admission` on ANY host unwind (KeyboardInterrupt,
        # cancellation, unexpected exception) without touching the fence lock,
        # so a host that cannot afford to block behind an in-flight commit can
        # still guarantee no FUTURE commit is admitted. Plain bool store —
        # atomic in CPython.
        self._admission_revoked = False
        # Holder-qualified durable-lock release hook (#76354 review F4;
        # transplanted from PR #71569 by @ciabata-git). The worker publishes an
        # idempotent, holder-scoped release callable once it owns the durable
        # compression lock; a timed-out host invokes it to free the lease
        # without racing a NEW holder (DB release is holder-qualified, so a
        # stale release can never delete a replacement's row — no ABA).
        self._lock_release_guard = threading.Lock()
        self._cancelled_lock_release: Optional[Callable[[], None]] = None
        self._cancelled_lock_release_requested = False
        # Forward-progress telemetry: the compression worker touches this
        # whenever the streamed summary call produces a token (see
        # ContextCompressor._call_summary_llm). Waiters use it to distinguish
        # a SLOW-but-alive summary model from a HUNG one, so slow models are
        # not killed by a fixed wall-clock deadline while tokens are moving.
        self._last_progress = time.monotonic()

    def touch_progress(self) -> None:
        """Record forward progress (e.g. a streamed summary token arriving).

        Called from the compression worker thread; read by async waiters via
        :meth:`seconds_since_progress`. A bare float store is atomic in
        CPython, so no lock is needed.
        """
        self._last_progress = time.monotonic()

    def seconds_since_progress(self) -> float:
        """Seconds since the worker last reported forward progress."""
        return max(0.0, time.monotonic() - self._last_progress)

    def cancel_before_commit(self, cancel_event: Any = None) -> bool:
        """Cancel a pending commit, or wait for an active commit to finish.

        Returns ``True`` when cancellation won before the commit boundary.
        Returns ``False`` when the worker had already entered the boundary; in
        that case acquiring this lock waits until all session mutation finishes.
        """
        with self._lock:
            if self._commit_started:
                if cancel_event is not None:
                    cancel_event.set()
                return False
            self._cancelled = True
            if cancel_event is not None:
                cancel_event.set()
            return True

    def try_cancel_before_commit(self) -> Optional[bool]:
        """Non-blocking form of :meth:`cancel_before_commit`.

        Returns ``None`` while an active commit owns the fence, allowing an
        async caller to yield instead of blocking its event loop.
        """
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._commit_started:
                return False
            self._cancelled = True
            return True
        finally:
            self._lock.release()

    def begin_commit(self, cancel_event: Any = None) -> bool:
        """Atomically admit commit unless a hard cancellation already won."""
        self._lock.acquire()
        if (
            self._cancelled
            or self._admission_revoked
            or (cancel_event is not None and bool(cancel_event.is_set()))
        ):
            self._cancelled = True
            self._lock.release()
            if self._admission_revoked:
                # Round-2 #1: a revoke that lost the fence-lock race to this
                # very begin_commit deferred its lease release; the commit was
                # refused, so the release is safe (and idempotent with the
                # worker's own holder-qualified cleanup) right now.
                self.release_cancelled_compression_lock()
            return False
        self._commit_started = True
        # Set while the fence lock is held so observers can never see
        # commit_in_flight=True for a commit that lost to cancellation.
        self._commit_phase.set()
        return True

    def finish_commit(self) -> None:
        """Leave a commit boundary entered by :meth:`begin_commit`."""
        self._commit_phase.clear()
        self._lock.release()
        if self._admission_revoked:
            # Round-2 #1: a revoke that arrived while THIS commit was in
            # flight deferred its durable-lease release rather than freeing
            # the lock out from under an active SessionDB mutation. The
            # commit is now fully complete, so perform the deferred release
            # here — promptly, without relying on the (possibly parked)
            # worker thread's outer cleanup. Idempotent with that cleanup:
            # the DB release is holder-qualified.
            self.release_cancelled_compression_lock()

    @property
    def commit_in_flight(self) -> bool:
        """Lock-free read: an admitted commit has begun and not yet finished.

        Safe to call from the host while the worker holds the fence lock for
        the whole commit (a hung SessionDB write). Hosts use this to reach
        their overrun-warning loop WHILE the commit is blocked instead of
        spinning on ``try_cancel_before_commit`` (which needs the lock the
        worker retains until ``finish_commit``).
        """
        return self._commit_phase.is_set()

    @property
    def is_cancelled(self) -> bool:
        """True after cancellation won before the commit boundary."""
        return self._cancelled or self._admission_revoked

    def revoke_commit_admission(self) -> None:
        """Revoke FUTURE commit admission without blocking on the fence lock.

        #76354 review F2: every host unwind path (KeyboardInterrupt, task
        cancellation, unexpected exception while waiting) must guarantee a
        detached worker cannot later enter the commit boundary and mutate
        durable/session state. The flag store is lock-free: a commit that is
        ALREADY in flight cannot be safely abandoned (the invariant "commit
        never abandoned mid-mutation" holds), but no NEW commit will be
        admitted after this call — ``begin_commit`` re-checks the flag under
        the fence lock.

        Round-2 #1 (durable-lease timing): the worker's holder-qualified
        lease release (F4) must NOT run while an admitted commit is still
        mutating SessionDB — a second compressor could otherwise acquire the
        durable lock mid-commit and interleave with the first commit's
        writes. The release decision is therefore made under the fence lock:

        - non-blocking acquire succeeds → no commit is in flight (an
          admitted commit RETAINS the lock until ``finish_commit``), so the
          lease is released immediately, while still holding the lock so a
          concurrent ``begin_commit`` cannot slip in between the check and
          the release (it would be refused anyway — the flag is already set).
        - acquire fails → the lock holder is either an in-flight commit or a
          transient boundary (lock-setup / cancel admission). Defer: the
          release then runs in ``finish_commit`` (after the mutation fully
          completes) or on the ``begin_commit``-refusal path, whichever the
          worker reaches first. Both are idempotent with the worker's own
          outer cleanup because the DB release is holder-qualified.
        """
        self._admission_revoked = True
        if self._lock.acquire(blocking=False):
            try:
                self.release_cancelled_compression_lock()
            finally:
                self._lock.release()
        # else: deferred — finish_commit()/begin_commit() re-check
        # _admission_revoked and perform the release once no commit can be
        # mid-mutation.

    # ── Holder-qualified durable-lease cancellation (#76354 F4) ──────────
    # Transplanted from PR #71569 (@ciabata-git): the worker publishes an
    # idempotent, holder-scoped release hook once it owns the durable
    # compression lock, and the host invokes it after winning cancellation.
    # ABA safety comes from SessionDB.release_compression_lock being
    # holder-qualified (DELETE ... WHERE holder = ?), so a stale release can
    # never free a NEW holder's lease.

    def begin_lock_setup(self) -> bool:
        """Fence durable-lock acquisition and release-hook publication.

        The caller keeps the fence until it has either published the exact
        holder-qualified release hook or established that no lock was
        acquired. A timeout cannot therefore win in the gap between acquiring
        the durable lock and making its cancellation cleanup callable.
        """
        self._lock.acquire()
        if self._cancelled or self._admission_revoked:
            self._lock.release()
            return False
        return True

    def finish_lock_setup(self) -> None:
        """Leave a lock setup boundary entered by :meth:`begin_lock_setup`."""
        self._lock.release()

    def register_cancelled_lock_release(
        self, release: Callable[[], None]
    ) -> bool:
        """Publish the timed-out worker's holder-qualified lock release.

        Returns whether cancellation cleanup was requested before publication.
        In that race, the release runs synchronously before this method returns.
        """
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
        """Release the cancelled worker's lock without finalizing its clients.

        Callers invoke this only after cancellation won (fence cancelled or
        admission revoked). A request that races ahead of lock-hook
        publication is retained and fulfilled synchronously when the worker
        publishes the hook.
        """
        with self._lock_release_guard:
            self._cancelled_lock_release_requested = True
            release = self._cancelled_lock_release
        if release is not None:
            release()


# Defaults for the in-agent (non-hygiene) progress-aware compress_context wrap.
# Mirror hermes_cli.config.DEFAULT_CONFIG["compression"] keys of the same name.
DEFAULT_CONTEXT_TIMEOUT_SECONDS = 120.0
DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS = 600.0

# Shared daemon pool for sync compress_context timeout wraps — analogous to
# asyncio's default executor used by gateway session hygiene's
# ``loop.run_in_executor(None, ...)``, but daemon so a fence-cancelled hung
# worker cannot block interpreter exit via concurrent.futures' atexit join.
# Created lazily; never shut down per call (a timed-out worker may still be
# winding down after fence cancel).
_compress_timeout_executor = None
_compress_timeout_executor_lock = threading.Lock()

# Commit-phase overrun wait slice: once an in-flight SessionDB commit runs
# past the total ceiling, keep waiting in bounded increments of this size so
# every overrun window produces a fresh (escalating) log line instead of one
# silent unbounded future.result(). Clamped down to the ceiling for tiny test
# ceilings so overrun reporting stays observable at test timescales.
_COMMIT_OVERRUN_WAIT_SLICE_SECONDS = 30.0

# Bounded admission for the shared compress-timeout pool (#76354 review F6).
# The stdlib executor queue is unbounded: with all four workers wedged in hung
# summaries, a fifth compression would queue silently, wait out its whole
# timeout without ever starting, and remain eligible to run as a stale job
# whenever a worker recovered. Admission is therefore capped at the worker
# count — when every worker slot is occupied (running OR admitted-not-started)
# submission FAILS FAST and the caller continues without compression.
#
# Recovery contract when all workers are wedged: new compressions fail fast
# (no queue growth, conversation continues uncompressed, a warning is logged
# each attempt); wedged workers are fence-cancelled so they cannot publish
# anything when they eventually return, and each recovery frees its admission
# slot via the future done-callback, restoring normal service. If a worker
# NEVER returns, its slot is lost for the process lifetime — bounded,
# observable degradation instead of an unbounded stale-job queue.
_COMPRESS_EXECUTOR_MAX_WORKERS = 4
_compress_admission_lock = threading.Lock()
_compress_admitted_count = 0


class CompressionExecutorSaturatedError(RuntimeError):
    """All compression pool slots are occupied; submission was refused."""


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
            # Small pool: compress is rare and heavy. Sized for a few
            # overlapping calls (live compress + fence-cancelled workers
            # still winding down), not asyncio's min(32, cpu+4) fan-out.
            _compress_timeout_executor = DaemonThreadPoolExecutor(
                max_workers=_COMPRESS_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="compress-ctx-timeout",
            )
        return _compress_timeout_executor


def resolve_context_compression_timeouts(
    compression_cfg: Optional[dict] = None,
) -> Tuple[float, float]:
    """Return ``(idle_timeout_seconds, total_ceiling_seconds)``.

    ``idle_timeout_seconds <= 0`` disables the owned progress-aware wrapper.
    The ceiling is clamped to at least one idle window when the idle budget
    is positive, matching gateway hygiene semantics.
    """
    idle = DEFAULT_CONTEXT_TIMEOUT_SECONDS
    ceiling = DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS
    cfg = compression_cfg
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            raw = load_config()
            maybe = raw.get("compression", {}) if isinstance(raw, dict) else {}
            cfg = maybe if isinstance(maybe, dict) else {}
        except Exception:
            cfg = {}
    if isinstance(cfg, dict):
        raw_idle = cfg.get("context_timeout_seconds")
        if raw_idle is not None:
            try:
                parsed = float(raw_idle)
                # Explicit 0/negative disables; positive values win.
                idle = parsed
            except (TypeError, ValueError):
                pass
        raw_ceiling = cfg.get("context_total_ceiling_seconds")
        if raw_ceiling is not None:
            try:
                parsed = float(raw_ceiling)
                if parsed > 0:
                    ceiling = parsed
            except (TypeError, ValueError):
                pass
    if idle > 0:
        ceiling = max(ceiling, idle)
    return idle, ceiling


def run_compress_context_with_progress_timeout(
    *,
    worker: Callable[[CompressionCommitFence], Tuple[list, str]],
    messages: list,
    system_prompt_fallback: Any,
    idle_timeout_seconds: float,
    total_ceiling_seconds: float,
    on_timeout: Optional[Callable[[float, float, float], None]] = None,
    on_commit_overrun: Optional[Callable[[float, float], None]] = None,
    fence: Optional[CompressionCommitFence] = None,
    telemetry_agent: Any = None,
) -> Tuple[list, str]:
    """Run ``worker(fence)`` under a sync progress-aware timeout.

    The idle budget is inactivity-based (same idea as gateway session hygiene):
    streamed summary progress via :meth:`CompressionCommitFence.touch_progress`
    extends the wait. A hard ceiling still bounds a degenerate trickle stream.

    When cancellation wins before the commit boundary, returns
    ``(messages, system_prompt_fallback)`` immediately and leaves the worker
    thread detached — the fence prevents a late commit from mutating session
    state. When the worker already entered the commit boundary, waits for that
    commit to finish and returns its result.

    Timeout budgets (``idle_timeout_seconds`` / ``total_ceiling_seconds``) cover
    the **pre-commit** wait only — the summary / stream phase before
    :meth:`CompressionCommitFence.begin_commit`. Once the worker holds the
    commit fence, SessionDB mutation is already in flight and cannot be safely
    abandoned without risking transcript divergence; the commit is therefore
    always allowed to complete. The commit-phase wait is still *bounded in
    increments* against the remaining total ceiling: if the commit runs past
    ``total_ceiling_seconds``, the overrun is logged loudly (escalating from
    WARNING to ERROR on repeat) and surfaced once via ``on_commit_overrun``,
    while the host keeps waiting in bounded slices until the commit finishes.
    The documented guarantee is: **summary phase bounded by the ceiling;
    commit phase logged + surfaced if it exceeds it** (never silently hung,
    never abandoned mid-commit).

    ``system_prompt_fallback`` may be a string or a zero-arg callable resolved
    only on the timeout path, so successful compression never pays for (or
    fails on) an eager prompt rebuild.
    """
    if idle_timeout_seconds <= 0:
        raise ValueError(
            "run_compress_context_with_progress_timeout requires "
            "idle_timeout_seconds > 0; call compress_context directly to disable"
        )

    def _resolve_fallback_prompt() -> str:
        if callable(system_prompt_fallback):
            return system_prompt_fallback()
        return system_prompt_fallback

    fence = fence if fence is not None else CompressionCommitFence()
    ceiling = max(float(total_ceiling_seconds), float(idle_timeout_seconds))
    idle = float(idle_timeout_seconds)
    # Sync mirror of gateway session-hygiene's run_in_executor(None, ...) +
    # wait_for loop (gateway/run.py): offload compress_context onto the shared
    # daemon pool, poll with an inactivity budget + total ceiling, then
    # fence-cancel on timeout so a late commit cannot land. Daemon workers
    # match tool_executor: a cancelled hung summary must not block process exit.
    from tools.thread_context import propagate_context_to_thread

    executor = _get_compress_timeout_executor()
    # Bounded admission (#76354 F6): refuse rather than queue when every pool
    # slot is occupied. A queued job would silently wait out its whole budget
    # without starting and stay eligible to run as a stale cancelled job when
    # a worker recovers. Fail fast: continue without compression this cycle.
    if not _try_admit_compression_job():
        logger.warning(
            "Context compression pool saturated (%d workers busy) — "
            "refusing new compression this cycle and continuing without "
            "compression. Wedged workers are fence-cancelled and free their "
            "slot when they return; if this persists, check the summary "
            "provider health.",
            _COMPRESS_EXECUTOR_MAX_WORKERS,
        )
        # Round-2 #6: saturation refusals must be visible in the same
        # telemetry stream as every other failed attempt, or a wedged pool
        # looks like compression simply stopped being attempted.
        if telemetry_agent is not None:
            _emit_compression_attempt_telemetry(
                telemetry_agent,
                started_at=time.monotonic(),
                commit_status="aborted",
                split_status="aborted",
                failure_class="pool_saturated",
            )
        return messages, _resolve_fallback_prompt()

    def _fence_gated_worker(worker_fence: CompressionCommitFence):
        # F6: an admitted job can still start after the host stopped waiting
        # (worker slot freed late). Check the fence BEFORE any expensive
        # summary work so a stale job never burns an LLM call; its return
        # value is discarded by the already-departed host.
        if worker_fence.is_cancelled:
            logger.info(
                "Skipping stale compression job: fence cancelled before start"
            )
            return messages, ""
        return worker(worker_fence)

    # Bare pool workers start with an empty ContextVar map; propagate the
    # parent conversation/approval context into the worker.
    try:
        future = executor.submit(
            propagate_context_to_thread(_fence_gated_worker), fence
        )
    except BaseException:
        _release_compression_admission()
        raise
    future.add_done_callback(_release_compression_admission)
    wait_started = time.monotonic()
    # F2: EVERY host unwind (KeyboardInterrupt, task cancellation, unexpected
    # exception while waiting) must revoke future commit admission before the
    # host resumes, or a detached worker could later commit and mutate durable
    # state behind the caller's back. ``handled_exit`` marks the paths that
    # settle admission themselves (worker result returned, or fence cancel
    # won); everything else revokes in the ``finally``.
    handled_exit = False
    try:
        while True:
            waited = time.monotonic() - wait_started
            remaining_ceiling = ceiling - waited
            if remaining_ceiling <= 0:
                break
            # #76354 S3 analogue for this wait: charge the idle budget from
            # the LAST PROGRESS event, not from the start of this wait slice.
            # Waiting a full ``idle`` after progress that landed early in the
            # previous slice would allow silence to approach 2x the budget.
            since_progress = fence.seconds_since_progress()
            wait_slice = min(
                max(idle - since_progress, 0.005), remaining_ceiling
            )
            try:
                result = future.result(timeout=wait_slice)
                handled_exit = True
                return result
            except concurrent.futures.TimeoutError:
                waited = time.monotonic() - wait_started
                since_progress = fence.seconds_since_progress()
                if since_progress < idle and waited < ceiling:
                    logger.info(
                        "Context compression still streaming after %.0fs "
                        "(last progress %.1fs ago) — extending wait "
                        "(ceiling %.0fs)",
                        waited,
                        since_progress,
                        ceiling,
                    )
                    continue
                break

        # F6: a not-yet-started future must not linger as a stale queued job.
        # cancel() is a no-op for a running worker (fence handles that path).
        future.cancel()

        cancelled: Optional[bool] = None
        while cancelled is None:
            # F1: ``begin_commit`` retains the fence lock until
            # ``finish_commit``, so a hung commit makes
            # ``try_cancel_before_commit`` return None forever. The lock-free
            # phase marker breaks the spin so the overrun-warning loop below
            # is reachable WHILE the commit is still blocked.
            if fence.commit_in_flight:
                cancelled = False
                break
            cancelled = fence.try_cancel_before_commit()
            if cancelled is None:
                # Round-2 #5: the fence is only held transiently here (lock
                # setup / cancel admission — an in-flight commit is caught by
                # the commit_in_flight check above), but that window rides
                # SessionDB write patience and can last seconds. 25ms keeps
                # sub-tick latency without a 1kHz spin.
                time.sleep(0.025)
        if not cancelled:
            # Pre-commit ceiling already elapsed, but begin_commit() won the
            # race. Waiting is intentional: SessionDB mutation cannot be
            # fence-cancelled. The wait is bounded in increments against the
            # remaining ceiling: a commit that overruns total_ceiling_seconds
            # is logged loudly and surfaced once (on_commit_overrun), then
            # waited on in bounded slices with escalating log level until it
            # completes. Guarantee: summary phase bounded by ceiling; commit
            # phase logged + surfaced if it exceeds it — never silently hung,
            # never abandoned mid-commit. F1: this loop is reachable WHILE
            # the commit is blocked (commit_in_flight is lock-free), so the
            # warning + on_commit_overrun fire during the hang, not after it.
            overrun_surfaced = False
            overrun_reports = 0
            while True:
                waited = time.monotonic() - wait_started
                remaining = ceiling - waited
                if remaining <= 0:
                    # Ceiling breached while the commit is in flight. Wait in
                    # bounded increments so each overrun window is visible in
                    # logs rather than one silent unbounded block.
                    remaining = min(
                        _COMMIT_OVERRUN_WAIT_SLICE_SECONDS,
                        max(ceiling, 0.05),
                    )
                    overrun_reports += 1
                    log = (
                        logger.warning if overrun_reports <= 2 else logger.error
                    )
                    log(
                        "Context compression SessionDB commit still running "
                        "%.1fs past the total ceiling (waited %.1fs, ceiling "
                        "%.1fs); commit cannot be abandoned mid-flight — "
                        "continuing to wait (check SessionDB health if this "
                        "persists)",
                        waited - ceiling,
                        waited,
                        ceiling,
                    )
                    if not overrun_surfaced and on_commit_overrun is not None:
                        overrun_surfaced = True
                        try:
                            on_commit_overrun(waited, ceiling)
                        except Exception:
                            logger.debug(
                                "compress_context commit-overrun callback "
                                "failed",
                                exc_info=True,
                            )
                try:
                    result = future.result(timeout=remaining)
                    handled_exit = True
                    return result
                except concurrent.futures.TimeoutError:
                    # Fence progress (commit-phase touch_progress) is
                    # informative only — the commit must complete regardless;
                    # loop and re-report with the updated overrun window.
                    continue

        # Idle-timeout path: cancellation won before the commit boundary.
        # The fence already blocks any future commit; F4 additionally frees
        # the timed-out worker's durable lease via the holder-qualified hook
        # so a NEW compressor can acquire the lock immediately (no ABA: the
        # DB release is holder-scoped).
        handled_exit = True
        fence.release_cancelled_compression_lock()
        waited = time.monotonic() - wait_started
        since_progress = fence.seconds_since_progress()
        if on_timeout is not None:
            try:
                on_timeout(idle, waited, since_progress)
            except Exception:
                logger.debug(
                    "compress_context timeout callback failed",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Context compression made no progress for %.1fs "
                "(total wait %.1fs, ceiling %.1fs); continuing without "
                "compression",
                since_progress,
                waited,
                ceiling,
            )
        # Leave the future on the shared pool: fence cancel won, so a late
        # commit cannot land (same detachment model as gateway hygiene).
        return messages, _resolve_fallback_prompt()
    finally:
        if not handled_exit:
            # F2: KeyboardInterrupt / cancellation / any unexpected exception
            # while waiting — revoke commit admission (and release the
            # worker's durable lease via the holder-qualified hook) before
            # the host unwinds, so the detached worker can never publish.
            fence.revoke_commit_admission()

class CompressionCheckpointUnavailable(RuntimeError):
    """Raised when required durable pre-compress checkpointing is unavailable."""


def _checkpoint_blocked(reason: str) -> CompressionCheckpointUnavailable:
    return CompressionCheckpointUnavailable(
        "BLOCKED_MISSING_PREREQUISITE: required pre-compress checkpoint "
        f"unavailable: {reason}"
    )


def _lock_api_is_absent_on_session_db(lock_db: Any) -> bool:
    """Whether the live in-memory SessionDB class structurally predates locks.

    In the supported hot-reload skew, this module is new while the already
    imported ``hermes_state.SessionDB`` class (and its live instances) is old.
    Only that exact class identity may fail open. Proxies, nominal lookalikes,
    non-callables, and descriptor failures must fail closed. Static lookup
    avoids invoking a present-but-broken descriptor.
    """
    try:
        from hermes_state import SessionDB

        missing = object()
        return (
            type(lock_db) is SessionDB
            and inspect.getattr_static(
                SessionDB, "try_acquire_compression_lock", missing
            ) is missing
        )
    except Exception:
        return False


def _refresh_persisted_compression_guards(
    compressor: Any,
    *,
    include_cooldown: bool = True,
) -> None:
    """Refresh durable automatic-compression guards on a built-in compressor."""
    method_calls = [
        ("_load_fallback_compression_streak", {}),
        ("_load_ineffective_compression_count", {}),
    ]
    if include_cooldown:
        method_calls.insert(
            0,
            ("get_active_compression_failure_cooldown", {"refresh": True}),
        )
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
    return bool(
        session
        and session.get("ended_at") is not None
        and session.get("end_reason") == "compression"
    )


def _emit_compression_attempt_telemetry(
    agent: Any,
    *,
    started_at: float,
    commit_status: str,
    split_status: str,
    failure_class: str | None = None,
) -> None:
    """Emit one content-free JSON log line for a compression attempt."""
    try:
        telemetry = getattr(agent.context_compressor, "_last_compression_telemetry", None)
        if not isinstance(telemetry, dict):
            telemetry = {}
        payload = dict(telemetry)
        payload.setdefault("event", "compression_attempt")
        payload.setdefault("attempt_id", getattr(agent, "_compression_attempt_id", "") or uuid.uuid4().hex)
        payload.setdefault("session_id", getattr(agent, "session_id", "") or "")
        payload["total_duration_ms"] = int((time.monotonic() - started_at) * 1000)
        payload["commit_status"] = commit_status
        payload["split_status"] = split_status
        if failure_class:
            payload["failure_class"] = failure_class
        payload.setdefault("chunking", False)
        payload.setdefault("chunk_count", 0)
        payload["fallback_used"] = bool(
            payload.get("fallback_used")
            or getattr(agent.context_compressor, "_last_summary_fallback_used", False)
            or getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
        )
        logger.info(
            "context compression attempt telemetry: %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
    except Exception as exc:
        logger.debug("failed to emit compression attempt telemetry: %s", exc)


def compression_skipped_due_to_lock(agent: Any) -> bool:
    """Type-pinned read of the #69870 lock-skip signal.

    ``agent._compression_skipped_due_to_lock`` is set by ``compress_context``
    when a compression pass no-ops because another path holds the per-session
    compression lock (holder string when the holder was confirmed, ``True``
    otherwise) and cleared to ``None`` at the entry of every call.

    The read MUST be type-pinned (``is True or isinstance(x, str)``), never
    bare truthiness: MagicMock test-double agents auto-create truthy
    attributes, and a bare ``if getattr(agent, ...)`` would hijack every
    mocked agent in sibling suites into the lock-skip branch (the
    #69870 × #69840 type-ahead incident).
    """
    _sig = getattr(agent, "_compression_skipped_due_to_lock", None)
    return _sig is True or isinstance(_sig, str)


def _adopt_live_compression_child(
    agent: Any,
    session_db: Any,
    parent_session_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Move a stale compression contender onto the live continuation tip.

    Resolve and load first, then mutate the live agent. This ordering keeps the
    stale contender fail-closed when lineage is ambiguous or the compacted
    handoff cannot be read.

    Resolution uses the canonical transitive walk ``get_compression_tip`` so a
    lineage with >=2 compression hops (root -> mid -> tip) recovers to the live
    tip — the depth-1 ``find_live_compression_child`` lookup this used to call
    finds no live *direct* child in that shape and skipped recovery (#82001).
    The tip walk returns the input id when no continuation exists, and a
    resolved tip is adopted only while its row is still live — both cases fail
    closed exactly as before.
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
    if not isinstance(recovered, list) or not recovered:
        return None
    # Revalidate after loading: the tip may have rotated or a competing
    # continuation may have appeared between the two DB reads.
    confirmed = resolver(session_db, parent_session_id)
    if not confirmed or str(confirmed) != child_session_id:
        return None

    agent.session_id = child_session_id
    try:
        from gateway.session_context import set_current_session_id

        set_current_session_id(child_session_id)
    except Exception:
        os.environ["HERMES_SESSION_ID"] = child_session_id
    try:
        from hermes_logging import set_session_context

        set_session_context(child_session_id)
    except Exception:
        pass

    agent._session_db_created = True
    if child.get("system_prompt"):
        agent._cached_system_prompt = child["system_prompt"]
    agent._last_flushed_db_idx = len(recovered)
    agent._flushed_db_message_session_id = child_session_id
    agent._flushed_db_message_ids = {
        id(message) for message in recovered if isinstance(message, dict)
    }

    on_session_start = getattr(agent.context_compressor, "on_session_start", None)
    if callable(on_session_start):
        try:
            on_session_start(
                child_session_id,
                boundary_reason="compression",
                old_session_id=parent_session_id,
                session_db=session_db,
                platform=getattr(agent, "platform", None) or "cli",
                conversation_id=getattr(agent, "_gateway_session_key", None),
            )
        except Exception as exc:
            logger.debug("context engine compression-child adoption failed: %s", exc)
    else:
        bind_state = getattr(agent.context_compressor, "bind_session_state", None)
        if callable(bind_state):
            try:
                bind_state(session_db=session_db, session_id=child_session_id)
            except Exception:
                pass
    try:
        if agent._memory_manager:
            agent._memory_manager.on_session_switch(
                child_session_id,
                parent_session_id=parent_session_id,
                reset=False,
                reason="compression",
            )
    except Exception as exc:
        logger.debug("memory manager compression-child adoption failed: %s", exc)

    return recovered


def recover_rotated_compression_session(
    agent: Any,
) -> Optional[List[Dict[str, Any]]]:
    """Recover a stale live agent before a new turn writes to its old parent."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None) or ""
    if session_db is None or not session_id:
        return None
    try:
        if not _session_was_rotated_by_compression(session_db, session_id):
            return None
        # Rotation publication holds the parent compression lease until the
        # child handoff is durable. A concurrent turn waits briefly rather than
        # observing the intentional parent-ended/child-empty intermediate state.
        holder_getter = getattr(session_db, "get_compression_lock_holder", None)
        for attempt in range(21):
            recovered = _adopt_live_compression_child(agent, session_db, session_id)
            if recovered is not None:
                return recovered
            holder = holder_getter(session_id) if callable(holder_getter) else None
            if not holder or attempt == 20:
                if not holder:
                    orphan_reopener = getattr(
                        type(session_db),
                        "reopen_orphaned_compression_session",
                        None,
                    )
                    if callable(orphan_reopener):
                        try:
                            if orphan_reopener(session_db, session_id):
                                logger.warning(
                                    "compression recovery: reopened orphaned "
                                    "session=%s with no continuation",
                                    session_id,
                                )
                        except Exception as exc:
                            logger.warning(
                                "orphaned compression session reopen failed "
                                "for %s: %s",
                                session_id,
                                exc,
                            )
                return None
            time.sleep(0.05)
        return None
    except Exception as exc:
        logger.warning(
            "compression session recovery failed for session=%s (%s: %s)",
            session_id,
            type(exc).__name__,
            exc,
        )
        return None


def _compression_lock_holder(agent: Any) -> str:
    """Build a unique holder id for the lock: pid:tid:agent-instance:uuid.

    The pid+tid prefix lets ops tell crashed/abandoned holders apart from
    live ones (expiry-based recovery uses the timestamp, but ``holder``
    is what shows up in diagnostics + log lines). The agent instance id
    and a per-acquire uuid disambiguate two co-resident agents on the
    same thread (background_review forks run on a worker thread, but
    on machines where compression itself dispatches to a thread pool
    we want each acquire to be unique).
    """
    import threading
    return (
        f"pid={os.getpid()}"
        f":tid={threading.get_ident()}"
        f":agent={id(agent):x}"
        f":nonce={uuid.uuid4().hex[:8]}"
    )


def _supported_compression_kwargs(
    compress_fn: Any,
    *,
    current_tokens: Optional[int],
    focus_topic: Optional[str],
    force: bool,
    memory_context: str,
) -> dict:
    """Return only compression kwargs accepted by an engine callable.

    Context-engine plugins can outlive additions to the optional host contract.
    Inspecting the callable before invoking it keeps those older signatures
    compatible without catching an internal ``TypeError`` and executing a
    stateful compressor twice.
    """
    candidates = {
        "current_tokens": current_tokens,
        "focus_topic": focus_topic,
        "force": force,
    }
    if memory_context:
        candidates["memory_context"] = memory_context
    try:
        parameters = inspect.signature(compress_fn).parameters
    except (TypeError, ValueError):
        # ``current_tokens`` has been part of the ContextEngine ABC since its
        # introduction. Keep the oldest documented call shape when a C-backed
        # or otherwise opaque callable has no inspectable signature.
        return {"current_tokens": current_tokens}

    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        return candidates
    return {name: value for name, value in candidates.items() if name in parameters}


class _CompressionActivityHeartbeat:
    """Refresh the agent inactivity tracker while compression blocks in an aux call."""

    def __init__(
        self,
        agent: Any,
        interval_seconds: float | None = None,
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
        except (TypeError, ValueError):
            interval_seconds = 60.0
        if not math.isfinite(interval_seconds):
            interval_seconds = 60.0
        self._interval_seconds = max(0.1, interval_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="compression-activity-heartbeat",
            daemon=True,
        )

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
        # Terminal completed/failed must reach SessionDB even inside the
        # ordinary 60s activity persist window — otherwise durable labels
        # stay on "context compression in progress" after /compress (which
        # never hits run_conversation's turn-end clear).
        self._touch(desc, force_persist=True)

    def _fence_cancelled(self) -> bool:
        fence = self._commit_fence
        return fence is not None and fence.is_cancelled

    def _should_suppress(self) -> bool:
        if self._suppressed:
            return True
        if self._fence_cancelled():
            self._suppressed = True
            return True
        return False

    def _touch(
        self,
        desc: str,
        *,
        allow_terminal_overwrite: bool = False,
        force_persist: bool = False,
    ) -> None:
        try:
            if not allow_terminal_overwrite:
                if self._should_suppress():
                    return
                current = normalize_activity_provenance(
                    getattr(self._agent, "_last_activity_provenance", None)
                )
                if current in _TERMINAL_COMPRESSION_PROVENANCES:
                    self._suppressed = True
                    return
            touch = getattr(self._agent, "_touch_activity", None)
            if callable(touch):
                # Re-check after reading provenance: host may cancel/stamp
                # TIMEOUT between the earlier guard and the write.
                if not allow_terminal_overwrite and self._should_suppress():
                    return
                touch(
                    desc,
                    provenance=ActivityProvenance.AGENT_COMPRESSION,
                    force_persist=force_persist,
                )
        except Exception:
            logger.debug("compression activity heartbeat touch failed", exc_info=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            if self._should_suppress():
                return
            self._touch("context compression in progress")

def _direct_messages_for_pre_compress_memory(messages: Any) -> list[dict[str, Any]]:
    """Return direct user/assistant evidence safe for memory checkpointing.

    Compression summaries are derivative context, not new source evidence.
    Tool rows and system messages are likewise omitted so memory providers
    receive one normalized host contract instead of having to infer Hermes
    transcript internals independently. Assistant messages that carry both
    prose and ``tool_calls`` keep their prose (the ``tool_calls`` payload is
    stripped); pure tool-call wrappers without prose are dropped.
    """
    # Deferred import: context_compressor imports turn_context, which imports
    # this module — a module-level import here would close that cycle
    # (upstream introduced the turn_context edge with the api_content work).
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

    direct_messages: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY):
            continue
        if role == "assistant" and message.get("tool_calls"):
            content = message.get("content")
            has_prose = bool(
                content.strip() if isinstance(content, str) else content
            )
            if not has_prose:
                continue
            message = {k: v for k, v in message.items() if k != "tool_calls"}
        direct_messages.append(message)
    return direct_messages


class _CompressionLockLeaseRefresher:
    def __init__(
        self,
        db: Any,
        session_id: str,
        holder: str,
        ttl_seconds: float,
        refresh_interval_seconds: float | None = None,
    ) -> None:
        self._db = db
        self._session_id = session_id
        self._holder = holder
        self._ttl_seconds = ttl_seconds
        if refresh_interval_seconds is None:
            refresh_interval_seconds = max(1.0, min(60.0, ttl_seconds / 2.0))
        self._refresh_interval_seconds = max(0.1, float(refresh_interval_seconds))
        # Tolerate transient refresh failures for at most one lease's worth of
        # time, so the give-up window is genuinely bounded by the TTL the
        # acquirer set (a single blip recovers on the next tick; a persistent
        # failure stops before the lease could outlive its TTL). Floor of 1 so a
        # degenerate interval >= ttl still tolerates one blip.
        self._max_consecutive_failures = max(
            1, int(self._ttl_seconds / self._refresh_interval_seconds)
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="compression-lock-refresh",
            daemon=True,
        )

    def start(self) -> "_CompressionLockLeaseRefresher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        # join() may time out while the refresher is mid-UPDATE; that's safe —
        # it's a daemon thread, and a late refresh on an already-released lock
        # matches rowcount 0 (a no-op). stop() returning does not guarantee the
        # thread has fully quiesced, only that we've signalled it and waited
        # briefly.
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # A single falsy refresh must NOT permanently kill the lease: a
        # transient DB blip (write contention escaping _execute_write's retry
        # budget, a momentary "database is locked") returns False just like a
        # genuine lost-ownership, but only the latter should stop the loop.
        # Tolerate consecutive failures for at most one lease's worth of time
        # (_max_consecutive_failures = ttl / interval), so a one-off blip
        # recovers on the next tick while the total give-up window stays bounded
        # by the TTL the acquirer set — the lock can never be held past its TTL
        # by a stuck refresher.
        consecutive_failures = 0
        # First refresh happens immediately, not one interval late. Everything
        # between try_acquire() and start() (the rotation-ownership lookup, the
        # durable-breaker re-read, thread startup) is charged against the very
        # first lease, so on a short TTL under load the lock could already be
        # expired — and reclaimable by a competing path — before tick #1.
        first = True
        while first or not self._stop.wait(self._refresh_interval_seconds):
            if first:
                first = False
                if self._stop.is_set():
                    break
            try:
                refreshed = self._db.refresh_compression_lock(
                    self._session_id,
                    self._holder,
                    ttl_seconds=self._ttl_seconds,
                )
            except Exception as exc:
                logger.debug("compression lock refresh raised: %s", exc)
                refreshed = False
            if refreshed:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures >= self._max_consecutive_failures:
                logger.debug(
                    "compression lock refresh failed %d times in a row; "
                    "stopping lease refresher for session %s",
                    consecutive_failures, self._session_id,
                )
                break


def check_compression_model_feasibility(agent: Any) -> None:
    """Warn at session start if the auxiliary compression model's context
    window is smaller than the main model's compression threshold.

    When the auxiliary model cannot fit the content that needs summarising,
    compression will either fail outright (the LLM call errors) or produce
    a severely truncated summary.

    Called during ``AIAgent.__init__`` so CLI users see the warning
    immediately (via ``_vprint``).  The gateway sets ``status_callback``
    *after* construction, so :func:`replay_compression_warning` re-sends
    the stored warning through the callback on the first
    ``run_conversation()`` call.
    """
    if not agent.compression_enabled:
        return
    try:
        from agent.auxiliary_client import (
            _resolve_task_provider_model,
            _try_configured_fallback_for_unavailable_client,
            get_text_auxiliary_client,
        )
        from agent.model_metadata import (
            MINIMUM_CONTEXT_LENGTH,
            get_model_context_length,
        )

        # Best-effort aux provider label for the warning message. The
        # configured provider may be "auto", in which case we fall back
        # to the client's base_url hostname so the user can still tell
        # where the compression model is actually being called.
        try:
            _aux_cfg_provider, _, _, _, _ = _resolve_task_provider_model("compression")
        except Exception:
            _aux_cfg_provider = ""
        client, aux_model = get_text_auxiliary_client(
            "compression",
            main_runtime=agent._current_main_runtime(),
        )
        if client is None or not aux_model:
            fb_client, fb_model, fb_label = _try_configured_fallback_for_unavailable_client(
                "compression",
                _aux_cfg_provider,
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
                    "Check auxiliary.compression in config.yaml and "
                    "reauthenticate that provider."
                )
            else:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "No auxiliary LLM provider for compression — "
                "summaries will be unavailable."
            )
            return

        aux_base_url = str(getattr(client, "base_url", ""))
        # ``client.api_key`` may be a callable (Azure Foundry Entra ID
        # bearer provider). The context-length resolver chain expects a
        # string, but it only needs a key for live catalogue probes
        # (provider model lists). For Entra clients the model-metadata
        # chain still resolves via models.dev + hardcoded family
        # fallbacks, which don't require auth — pass empty string rather
        # than minting a bearer JWT just to look up a context length.
        _raw_aux_key = getattr(client, "api_key", "")
        aux_api_key = "" if (callable(_raw_aux_key) and not isinstance(_raw_aux_key, str)) else str(_raw_aux_key or "")

        aux_context = get_model_context_length(
            aux_model,
            base_url=aux_base_url,
            api_key=aux_api_key,
            config_context_length=getattr(agent, "_aux_compression_context_length_config", None),
            # Each model must be resolved with its own provider so that
            # provider-specific paths (e.g. Bedrock static table, OpenRouter API)
            # are invoked for the correct client, not inherited from the main model.
            provider=(_aux_cfg_provider if _aux_cfg_provider and _aux_cfg_provider != "auto" else getattr(agent, "provider", "")),
            custom_providers=agent._custom_providers,
        )

        # Hard floor: the auxiliary compression model must have at least
        # MINIMUM_CONTEXT_LENGTH (64K) tokens of context.  The main model
        # is already required to meet this floor (checked earlier in
        # __init__), so the compression model must too — otherwise it
        # cannot summarise a full threshold-sized window of main-model
        # content.  Mirrors the main-model rejection pattern.
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

        threshold = agent.context_compressor.threshold_tokens
        if aux_context < threshold:
            # Auto-correct: lower the live session threshold so
            # compression actually works this session.  The hard floor
            # above guarantees aux_context >= MINIMUM_CONTEXT_LENGTH,
            # so the new threshold is always >= 64K.
            #
            # The compression summariser sends a single user-role
            # prompt (no system prompt, no tools) to the aux model, so
            # new_threshold == aux_context is safe: the request is
            # the raw messages plus a small summarisation instruction.
            old_threshold = threshold
            new_threshold = aux_context
            agent.context_compressor.threshold_tokens = new_threshold
            # ``tail_token_budget`` is derived from the trigger threshold, not
            # directly from the model window. Keep it in lockstep with this
            # just-in-time correction exactly as ContextCompressor.update_model()
            # does. Leaving the old budget behind can make the tail's 1.5x soft
            # ceiling wider than the lowered trigger, so compression preserves
            # nearly the entire request and repeatedly re-fires.
            summary_target_ratio = getattr(
                agent.context_compressor, "summary_target_ratio", None
            )
            if isinstance(summary_target_ratio, (int, float)):
                agent.context_compressor.tail_token_budget = int(
                    new_threshold * summary_target_ratio
                )
            # Keep threshold_percent in sync so future main-model
            # context_length changes (update_model) re-derive from a
            # sensible number rather than the original too-high value.
            main_ctx = agent.context_compressor.context_length
            if main_ctx:
                agent.context_compressor.threshold_percent = (
                    new_threshold / main_ctx
                )
            safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
            # The "lower the threshold" suggestion must survive the built-in
            # trigger recomputation (#67422): _effective_threshold_percent()
            # raises sub-75% values back up for main windows under 512K, and
            # _compute_threshold_tokens() further applies the output-token
            # reservation, the 64K floor, and the degenerate-window guard.
            # Recommending a value those would override is silently ignored
            # and this warning would reappear every session — so mirror the
            # compressor's own math and only offer the option when the
            # recomputed trigger actually fits the auxiliary model's context.
            # External engines own compaction policy (#44439); the built-in
            # floor doesn't apply to them, so keep the plain suggestion.
            from agent.context_compressor import ContextCompressor as _CC

            recomputed_threshold = None
            if main_ctx and isinstance(agent.context_compressor, _CC):
                recomputed_threshold = _CC._compute_threshold_tokens(
                    main_ctx,
                    _CC._effective_threshold_percent(main_ctx, safe_pct / 100),
                    getattr(agent.context_compressor, "max_tokens", None),
                )
            threshold_suggestion_viable = (
                recomputed_threshold is None or recomputed_threshold <= aux_context
            )
            # Build human-readable "model (provider)" labels for both
            # the main model and the compression model so users can
            # tell at a glance which provider each side is actually
            # using. When the configured provider is empty or "auto",
            # fall back to the client's base_url hostname.
            _main_model = getattr(agent, "model", "") or "?"
            _main_provider = getattr(agent, "provider", "") or ""
            _aux_provider_label = (
                _aux_cfg_provider
                if _aux_cfg_provider and _aux_cfg_provider != "auto"
                else ""
            )
            if not _aux_provider_label:
                try:
                    from urllib.parse import urlparse
                    _aux_provider_label = (
                        urlparse(aux_base_url).hostname or aux_base_url
                    )
                except Exception:
                    _aux_provider_label = aux_base_url or "auto"
            _main_label = (
                f"{_main_model} ({_main_provider})"
                if _main_provider
                else _main_model
            )
            _aux_label = f"{aux_model} ({_aux_provider_label})"
            msg = (
                f"⚠ Compression model {_aux_label} context is "
                f"{aux_context:,} tokens, but the main model "
                f"{_main_label}'s compression threshold was "
                f"{old_threshold:,} tokens. "
                f"Auto-lowered this session's threshold to "
                f"{new_threshold:,} tokens so compression can run.\n"
            )
            if threshold_suggestion_viable:
                msg += (
                    f"  To make this permanent, edit config.yaml — either:\n"
                    f"  1. Use a larger compression model:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{old_threshold:,}+-context>\n"
                    f"  2. Lower the compression threshold:\n"
                    f"       compression:\n"
                    f"         threshold: 0.{safe_pct:02d}"
                )
            else:
                msg += (
                    f"  To make this permanent, use a larger compression "
                    f"model in config.yaml:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{old_threshold:,}+-context>\n"
                    f"  (Lowering compression.threshold cannot help here — "
                    f"with {_main_label}'s {main_ctx:,}-token window, "
                    f"Hermes's small-context floor and output reservation "
                    f"would recompute the trigger to "
                    f"{recomputed_threshold:,} tokens, still above the "
                    f"compression model's {aux_context:,}.)"
                )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "Auxiliary compression model %s has %d token context, "
                "below the main model's compression threshold of %d "
                "tokens — auto-lowered session threshold to %d to "
                "keep compression working.",
                aux_model,
                aux_context,
                old_threshold,
                new_threshold,
            )
    except ValueError:
        # Hard rejections (aux below minimum context) must propagate
        # so the session refuses to start.
        raise
    except Exception as exc:
        logger.debug(
            "Compression feasibility check failed (non-fatal): %s", exc
        )


def replay_compression_warning(agent: Any) -> None:
    """Re-send the compression warning through ``status_callback``.

    During ``__init__`` the gateway's ``status_callback`` is not yet
    wired, so ``_emit_status`` only reaches ``_vprint`` (CLI).  This
    method is called once at the start of the first
    ``run_conversation()`` — by then the gateway has set the callback,
    so every platform (Telegram, Discord, Slack, etc.) receives the
    warning.
    """
    msg = getattr(agent, "_compression_warning", None)
    if msg and agent.status_callback:
        try:
            agent.status_callback("lifecycle", msg)
        except Exception:
            pass


def conversation_history_after_compression(
    agent: Any,
    messages: list,
    previous_history: Optional[list] = None,
) -> Optional[list]:
    """Return the correct flush baseline after a compression boundary.

    Legacy compression rotates to a fresh child session. That child has not
    seen the compacted transcript through the normal same-turn flush path yet,
    so callers must clear ``conversation_history`` to ``None`` and let the next
    persistence call write the whole compacted list.

    In-place compaction is different: ``archive_and_compact()`` has already
    soft-archived the previous active rows and inserted ``messages`` as the new
    active live transcript under the same session id. If the same agent turn
    continues with ``conversation_history=None``, the identity-based flush path
    treats those already-persisted compacted dicts as new and appends them a
    second time, doubling the active context and retriggering compression.

    A shallow copy is intentional: it captures the current compacted dict
    identities as history while allowing later same-turn appends to remain new.

    An aborted or no-op attempt after an earlier in-place compaction must retain
    the pre-attempt baseline.  Treating all current messages as persisted would
    drop any later, unflushed turns on restart; clearing the baseline would
    append the already-persisted compacted rows a second time.
    """
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
    "[System: Your previous response was truncated",
    "[System: The previous response was cut off",
    "[System: Your previous tool call",
    "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
)


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    return ""


_SYNTHETIC_USER_FLAGS = (
    "_todo_snapshot_synthetic",
    "_empty_recovery_synthetic",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_dropped_toolcall_nudge",
)


def _is_real_user_message(message: Any) -> bool:
    """Distinguish human intent from user-role runtime scaffolding.

    A compaction summary pinned to ``role="user"`` (the compressor flips the
    summary role to preserve alternation when the tail starts with an
    assistant message) is scaffolding too: treating it as human intent would
    short-circuit anchor restoration with a message the model is explicitly
    told NOT to act on.
    """
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if any(message.get(flag) for flag in _SYNTHETIC_USER_FLAGS):
        return False
    text = _message_text(message).strip()
    if not text:
        return False
    if text.startswith(_SYNTHETIC_USER_PREFIXES):
        return False
    from agent.context_compressor import ContextCompressor

    return not ContextCompressor._is_synthetic_compression_user_turn(message)


def _strip_stale_todo_snapshot(content: Any) -> Any:
    """Remove a previously merged todo-snapshot block from message content.

    Snapshot merges (see the injection site in ``compress_context``) always
    append the block at the end of the trailing user turn, so a surviving
    header marks stale todo state from an earlier compaction boundary.
    Stripping before re-injection keeps repeated boundaries from
    accumulating outdated snapshots (#26981).
    """
    from tools.todo_tool import TODO_INJECTION_HEADER

    if isinstance(content, str):
        idx = content.find(TODO_INJECTION_HEADER)
        if idx == -1:
            return content
        return content[:idx].rstrip()
    if isinstance(content, list):
        return [
            part
            for part in content
            if not (
                isinstance(part, dict)
                and part.get("type") == "text"
                and str(part.get("text") or "")
                .lstrip()
                .startswith(TODO_INJECTION_HEADER)
            )
        ]
    return content


# Retention-parity notice (#84718): compaction re-injects the todo list
# verbatim while skill instructions are pruned to [SKILL_PRUNED: ...] markers,
# so the imperative crosses the boundary without the policy that governed it.
# When BOTH happen at the same boundary, couple them: the re-injected snapshot
# carries an explicit instruction to reload the pruned skills BEFORE acting on
# any preserved task. Deterministic (derived only from the compressed
# transcript), bounded (marker cap shared with the summary re-injection), and
# stripped together with the snapshot at the next boundary because it lives
# after TODO_INJECTION_HEADER inside the same block.
_PRUNED_SKILL_RELOAD_NOTICE_HEADER = (
    "[Skills pruned during compression — reload before acting on these tasks]"
)


def _pruned_skill_reload_notice(compressed: list) -> str:
    """Reload instruction for skills whose bodies were pruned, or ``""``.

    Scans the post-compression transcript for the canonical
    ``[SKILL_PRUNED: ...]`` markers (summary ``## Pruned Skills`` section,
    pruned tool rows surviving in the protected tail) and renders one bounded
    notice naming each skill with its exact ``skill_view`` reload call.
    First-seen order, deduplicated, capped at ``_MAX_PRUNED_SKILL_MARKERS``.
    """
    from agent.context_compressor import (
        _MAX_PRUNED_SKILL_MARKERS,
        _extract_pruned_skill_names,
    )

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
        "task is still justified — findings recorded before the boundary may "
        "have invalidated it."
    )


def _merge_anchor_into_user_message(target: dict, anchor: dict) -> None:
    """Fold the human anchor into an existing user-role scaffolding turn.

    Used only when every insertion slot would create two consecutive
    user-role messages. The anchor text leads (it is the active task), the
    scaffolding content is preserved after it, and the synthetic flags are
    cleared because the merged turn now carries real human intent.
    """
    anchor_content = anchor.get("content")
    target_content = target.get("content")
    if isinstance(anchor_content, list) or isinstance(target_content, list):
        anchor_parts = (
            list(anchor_content)
            if isinstance(anchor_content, list)
            else [{"type": "text", "text": str(anchor_content or "")}]
        )
        target_parts = (
            list(target_content)
            if isinstance(target_content, list)
            else [{"type": "text", "text": str(target_content or "")}]
        )
        target["content"] = anchor_parts + target_parts
    else:
        merged = f"{anchor_content or ''}\n\n{target_content or ''}".strip()
        target["content"] = merged
    for flag in _SYNTHETIC_USER_FLAGS:
        target.pop(flag, None)


def _insert_real_user_anchor(messages: list, anchor: dict) -> None:
    """Insert the latest human turn without breaking role alternation."""

    def _role(msg: Any) -> Optional[str]:
        return msg.get("role") if isinstance(msg, dict) else None

    # Preferred: the summary boundary — before the first assistant message
    # not already preceded by a user turn. The left neighbour is then
    # non-user by construction and the right neighbour is an assistant.
    for index, message in enumerate(messages):
        if _role(message) != "assistant":
            continue
        previous_role = _role(messages[index - 1]) if index > 0 else None
        if previous_role != "user":
            messages.insert(index, anchor)
            return
    # Every assistant is user-preceded (or there are none). Appending is
    # safe whenever the transcript does not already end with a user turn.
    if not messages or _role(messages[-1]) != "user":
        messages.append(anchor)
        return
    # The transcript ends with a user-role message and no slot avoids
    # user/user adjacency.
    from agent.context_compressor import ContextCompressor

    if ContextCompressor._is_context_summary_content(
        _message_text(messages[-1])
    ):
        # Never merge into a compaction summary: the summary prefix must
        # stay at the start of its message for downstream summary detection.
        # Appending after it makes the anchor "the latest user message after
        # the summary" — exactly what the handoff prefix instructs — and the
        # adjacent user turns are merged summary-first by
        # repair_message_sequence before the next API call.
        messages.append(anchor)
        return
    # Trailing user-role scaffolding (e.g. the todo snapshot): merge instead
    # of inserting a consecutive same-role message (#55677 strict templates).
    _merge_anchor_into_user_message(messages[-1], anchor)


def _ensure_compressed_has_user_turn(original_messages: list, compressed: list) -> None:
    """Preserve human intent, not merely a synthetic user-role placeholder."""
    if any(_is_real_user_message(message) for message in compressed):
        return
    from agent.context_compressor import (
        COMPRESSION_CONTINUATION_USER_CONTENT,
        _fresh_compaction_message_copy,
    )

    for message in reversed(original_messages):
        if _is_real_user_message(message):
            _insert_real_user_anchor(
                compressed,
                _fresh_compaction_message_copy(message),
            )
            return
    from agent.message_metadata import append_message

    append_message(
        compressed,
        {
            "role": "user",
            "content": COMPRESSION_CONTINUATION_USER_CONTENT,
        },
    )


_PENDING_CONTEXT_ENGINE_NOTIFICATION = (
    "_pending_context_engine_compression_notification"
)


def _notify_context_engine_compression_complete(
    agent: Any,
    *,
    new_session_id: str,
    old_session_id: str,
) -> bool:
    """Notify the active context engine after a durable compression commit."""
    # Relay session-span segmentation (opt-in, gateway.telemetry.
    # session_segments.on_compaction): flag the session so its telemetry
    # scope rotates at the next turn boundary. Observer semantics — a
    # failure here must never undo or delay the committed compression.
    try:
        from agent import relay_runtime

        relay_runtime.SESSION_COORDINATOR.notify_session_compacted(
            profile_key=relay_runtime.current_profile_key(),
            session_id=new_session_id,
            old_session_id=old_session_id,
        )
    except Exception:
        logger.debug("relay segment rotation notification failed", exc_info=True)
    callback = getattr(agent.context_compressor, "on_session_start", None)
    if not callable(callback):
        return False
    try:
        callback(
            new_session_id,
            boundary_reason="compression",
            old_session_id=old_session_id,
            platform=getattr(agent, "platform", None) or "cli",
            conversation_id=getattr(agent, "_gateway_session_key", None),
        )
    except Exception:
        # Context-engine hooks are observers. A callback failure must not undo
        # history that the core or an outer host transaction already committed.
        logger.debug(
            "context engine on_session_start (compression) failed",
            exc_info=True,
        )
        return False
    return True


def _queue_context_engine_compression_notification(
    agent: Any,
    *,
    new_session_id: str,
    old_session_id: str,
) -> None:
    """Stage exactly one existing hook call for an outer host transaction."""
    if callable(getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)):
        raise RuntimeError("a compression notification is already pending")

    def _notify() -> bool:
        return _notify_context_engine_compression_complete(
            agent,
            new_session_id=new_session_id,
            old_session_id=old_session_id,
        )

    setattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, _notify)


def finalize_context_engine_compression_notification(
    agent: Any,
    *,
    committed: bool,
) -> bool:
    """Emit or discard a deferred notification; repeated calls are no-ops."""
    pending = getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)
    setattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None)
    if not committed or not callable(pending):
        return False
    return bool(pending())


def compress_context(
    agent: Any,
    messages: list,
    system_message: str,
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    focus_topic: Optional[str] = None,
    force: bool = False,
    defer_context_engine_notification: bool = False,
    commit_fence: Optional[CompressionCommitFence] = None,
) -> Tuple[list, str]:
    """Compress conversation context and split the session in SQLite.

    Args:
        agent: The owning :class:`AIAgent`.
        messages: Current message history (will be summarised).
        system_message: Current system prompt; used when compression needs a
            rebuilt cached prompt.
        approx_tokens: Pre-compression token estimate, logged for ops.
        task_id: Tool task scope (used for clearing file-read dedup state).
        focus_topic: Optional focus string for guided compression — the
            summariser will prioritise preserving information related to
            this topic.  Inspired by Claude Code's ``/compact <focus>``.
        force: If True, bypass any active summary-failure cooldown.  Set
            by the manual ``/compress`` slash command so users can retry
            immediately after an auto-compress abort.  Auto-compress
            callers use the default ``False``.
        defer_context_engine_notification: Delay the existing context-engine
            hook until a manual host commits its outer history transaction.
        commit_fence: Optional cooperative fence for executor callers that
            may time out. It prevents a late worker from mutating session state
            after its caller has moved on.

    Returns:
        ``(compressed_messages, new_system_prompt)`` tuple.  When
        compression aborts (aux LLM failed to produce a usable summary),
        returns the original messages unchanged and the existing system
        prompt — the session is NOT rotated.  Callers should detect the
        no-op via ``len(returned) == len(input)`` and stop the retry loop.
    """
    _compressor_attempt_snapshot = _snapshot_compressor_attempt_state(
        agent.context_compressor
    )
    _durable_cooldown_authoritative: Optional[bool] = None
    _durable_cooldown_state: Optional[dict[str, Any]] = None
    if (
        defer_context_engine_notification
        and callable(getattr(agent, _PENDING_CONTEXT_ENGINE_NOTIFICATION, None))
    ):
        raise RuntimeError("a compression notification is already pending")

    # ``conversation_history_after_compression()`` needs the latest attempt's
    # outcome, while ``_last_compaction_in_place`` remains the run-level signal
    # read by gateway callers. ``None`` means this attempt aborted or made no
    # boundary, so the previous flush baseline remains authoritative.
    agent._last_compression_attempt_recorded = True
    agent._last_compression_attempt_in_place = None
    # Clear the lock-skip signal at the VERY TOP, before the codex route and
    # the breaker gates below can early-return (per-attempt state rule,
    # #58630/#69853). A stale ``True``/holder value from a prior lock-skip
    # must never make a later breaker/codex no-op look like lock contention
    # to the automatic-path consumers (compression_deferred, #49874) — the
    # second clear before lock acquisition below stays for the same reason
    # it was added in #69870 and is simply idempotent now.
    agent._compression_skipped_due_to_lock = None

    _attempt_started_at = time.monotonic()
    _attempt_id = uuid.uuid4().hex
    _trigger_source = "manual" if force else "auto"
    try:
        agent._compression_attempt_id = _attempt_id
        setattr(agent.context_compressor, "_compression_telemetry_seed", {
            "attempt_id": _attempt_id,
            "session_id": agent.session_id or "",
            "trigger_source": _trigger_source,
        })
    except Exception:
        pass

    # Codex app-server sessions: the codex agent owns the real thread context;
    # Hermes' summarizer would only rewrite a local mirror without shrinking
    # the actual thread (#36801). Route compaction to the app server's own
    # thread/compact mechanism. Behavior is controlled by
    # ``compression.codex_app_server_auto`` (native|hermes|off).
    # The memory-provider context handoff below is intentionally Hermes-only:
    # the app server does not expose its native summary prompt, so there is no
    # truthful injection point for ``on_pre_compress()`` return text here.
    # `is True` (not bool()): unit tests drive this path with bare MagicMock
    # agents whose auto-created attributes are truthy; the gate must only arm
    # on the explicit boolean set by agent_init from config.
    checkpoint_required = (
        getattr(agent, "compression_checkpoint_required", False) is True
    )
    if getattr(agent, "api_mode", None) == "codex_app_server":
        if checkpoint_required:
            raise _checkpoint_blocked(
                "codex_app_server owns the authoritative thread and does not "
                "expose a truthful pre-compaction transcript boundary"
            )
        _codex_fence_entered = False
        if commit_fence is not None:
            _codex_fence_entered = commit_fence.begin_commit(
                getattr(agent, "_hard_interrupt_requested", None)
            )
            if not _codex_fence_entered:
                _restore_compressor_attempt_state(
                    agent.context_compressor, _compressor_attempt_snapshot
                )
                existing_prompt = getattr(agent, "_cached_system_prompt", None)
                if not existing_prompt:
                    existing_prompt = agent._build_system_prompt(system_message)
                return messages, existing_prompt
        try:
            return _compress_context_via_codex_app_server(
                agent,
                messages,
                system_message,
                approx_tokens=approx_tokens,
                task_id=task_id,
                force=force,
            )
        finally:
            if _codex_fence_entered:
                commit_fence.finish_commit()

    # Every automatic entrypoint must honor compressor-owned cooldown and
    # breaker state. Gateway hygiene constructs a fresh AIAgent, so the
    # persisted fallback streak is loaded by bind_session_state() before this.
    if not force:
        _refresh_persisted_compression_guards(agent.context_compressor)
        blocked = getattr(
            type(agent.context_compressor),
            "_automatic_compression_blocked",
            None,
        )
        if callable(blocked) and blocked(agent.context_compressor):
            existing_prompt = getattr(agent, "_cached_system_prompt", None)
            if not existing_prompt:
                existing_prompt = agent._build_system_prompt(system_message)
            return messages, existing_prompt

    # Lazy feasibility check — run the auxiliary-provider probe + context
    # length lookup just-in-time on the first compression attempt instead of
    # at AIAgent.__init__. Saves ~400ms cold off every short session that
    # never reaches the threshold (the vast majority of ``chat -q`` runs).
    # The check itself sets ``agent._compression_warning`` so the
    # status-callback replay machinery still emits the warning to the user
    # the first time it would matter.
    if not getattr(agent, "_compression_feasibility_checked", False):
        # Mark as checked only after the probe completes. If the check
        # raises (e.g. a fatal aux-context ValueError that aborts the
        # session), leaving the flag unset is harmless; a non-fatal
        # transient failure is swallowed inside the function so the flag
        # is set normally on the next successful pass.
        check_compression_model_feasibility(agent)
        agent._compression_feasibility_checked = True

    _pre_msg_count = len(messages)
    # In-place compaction (config: compression.in_place, see #38763). When True,
    # this compaction rewrites the message list and refreshes the system prompt
    # when necessary, but keeps the SAME session_id — no end_session, no
    # parent_session_id child, no
    # `name #N` renumber, no contextvar/env/logging re-sync, no memory/context-
    # engine session-switch. The conversation keeps one durable id for life,
    # eliminating the session-rotation bug cluster. Default True (2107b86024).
    # Default True matches DEFAULT_CONFIG / #38763. A missing attribute must
    # NOT fall back to rotation mode — that re-enables the pre-lease drift
    # path and can wedge busy sessions that never set the flag.
    in_place = bool(getattr(agent, "compression_in_place", True))
    # Set True once the in-place DB write actually completes (the DB block can
    # raise and skip it). Surfaced to the gateway via agent._last_compaction_in_place.
    compacted_in_place = False
    logger.info(
        "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
        agent.session_id or "none", _pre_msg_count,
        f"{approx_tokens:,}" if approx_tokens else "unknown", agent.model,
        focus_topic,
    )
    _compaction_status = COMPACTION_STATUS
    if not force:
        _compaction_status = automatic_compaction_status_message(
            agent.context_compressor,
            phase="compress",
            default_message=_compaction_status,
            approx_tokens=approx_tokens,
            message_count=_pre_msg_count,
            model=agent.model,
            focus_topic=focus_topic,
        )
    _compaction_status_emitted = bool(_compaction_status)
    if _compaction_status:
        agent._emit_status(_compaction_status)
    _compaction_done_emitted = False
    _compaction_succeeded = False

    def _complete_compaction_lifecycle() -> None:
        nonlocal _compaction_done_emitted
        if _compaction_done_emitted:
            return
        _compaction_done_emitted = True
        # A suppressed start (quiet context engine) opened no visible
        # compaction phase — emit no terminal edge either. Failure warnings
        # go through agent._emit_warning and are never suppressed here.
        if _compaction_status_emitted and _compaction_succeeded:
            _emit_compaction_done(agent)

    # ── Compression lock ────────────────────────────────────────────────
    # Atomic, state.db-backed lock per session_id.  Without this, two
    # AIAgent instances that share the same session_id (most commonly the
    # parent-turn agent and its background-review fork — see
    # ``agent/background_review.py``: ``review_agent.session_id =
    # agent.session_id``) can each call compress() on overlapping
    # snapshots of the same conversation.  Both succeed, both rotate
    # ``agent.session_id`` to a fresh id, both create child sessions in
    # state.db parented to the same old id.  The gateway's SessionEntry
    # only catches one rotation, so the other child becomes an orphan
    # that silently accumulates writes — Damien's repro shape.
    #
    # Acquire keyed on the OLD session_id (the rotation target's parent),
    # because that's the id that competing paths see and read from
    # SessionEntry at the start of their own compression attempt.
    #
    # If we can't acquire the lock, another path is mid-compression on
    # this session.  Aborting is correct: the messages are unchanged, the
    # other path's rotation will produce the canonical new session_id,
    # and our caller's auto-compress loop sees ``len(returned) == len(input)``
    # and stops retrying for this cycle. The session is NOT corrupted —
    # we just sit out this round and let the winner finish.
    _lock_db = getattr(agent, "_session_db", None)
    _lock_sid = agent.session_id or ""
    _lock_holder: Optional[str] = None
    # Watermark captured at compression start (#75316); None = fall back to
    # archive-everything (no concurrent-tail preservation this cycle).
    _commit_watermark: Optional[int] = None
    # Probe whether the lock subsystem is actually available on this
    # SessionDB instance. A process running mismatched module versions can have
    # this call site while its long-lived SessionDB instance predates the lock
    # API. Only that structural absence is safe to fail open for: compression
    # must make progress rather than spin forever after an update. Once the
    # method has been resolved, every exception from its implementation fails
    # closed because proceeding without a lock can fork the session lineage.
    _try_acquire_lock = None
    _lock_lookup_error: Optional[Exception] = None
    _legacy_session_db_without_lock_api = False
    # Clear any stale lock-skip signal from a prior call so this call's
    # outcome alone determines what callers see.  Without this an
    # auto-compress lock-skip followed by a successful manual /compress
    # would falsely report "Compression already in progress" and discard
    # the compression results.
    agent._compression_skipped_due_to_lock = None
    if _lock_db is not None:
        try:
            _legacy_session_db_without_lock_api = _lock_api_is_absent_on_session_db(
                _lock_db
            )
        except Exception as exc:
            _lock_lookup_error = exc
        if _lock_lookup_error is None and not _legacy_session_db_without_lock_api:
            try:
                _try_acquire_lock = _lock_db.try_acquire_compression_lock
                if not callable(_try_acquire_lock):
                    _lock_lookup_error = TypeError(
                        "compression lock API is present but not callable"
                    )
            except Exception as exc:
                _lock_lookup_error = exc
    try:
        _lock_ttl = float(getattr(agent, "_compression_lock_ttl_seconds", 300.0) or 300.0)
    except (TypeError, ValueError):
        _lock_ttl = 300.0
    _lock_refresh_interval = getattr(agent, "_compression_lock_refresh_interval", None)
    _lock_refresher: Optional[_CompressionLockLeaseRefresher] = None
    # F4 (#76354, transplanted from PR #71569 by @ciabata-git): fence the
    # durable-lock acquisition + release-hook publication so a host timeout
    # can never win in the gap between acquiring the durable lock and having
    # a holder-qualified way to release it.
    _lock_setup_entered = False

    def _finish_lock_setup() -> None:
        nonlocal _lock_setup_entered
        if not _lock_setup_entered or commit_fence is None:
            return
        _lock_setup_entered = False
        commit_fence.finish_lock_setup()

    if _lock_db is not None and _lock_sid:
        _lock_holder = _compression_lock_holder(agent)
        if _lock_lookup_error is not None:
            # Attribute lookup itself failed for a reason other than a missing
            # lock API. It is unsafe to proceed without a lock in that case.
            _lock_holder = None
            logger.warning(
                "compression lock lookup raised unexpectedly for session=%s "
                "(%s: %s) — skipping compression this cycle",
                _lock_sid, type(_lock_lookup_error).__name__, _lock_lookup_error,
            )
            _lock_acquired = False
        elif _try_acquire_lock is None:
            # The lock API itself is absent on this in-memory instance. Log once
            # and proceed unlocked so an update-version skew cannot leave the
            # outer auto-compression loop making no progress forever.
            _lock_holder = None
            if getattr(agent, "_last_compression_lock_error_sid", None) != _lock_sid:
                agent._last_compression_lock_error_sid = _lock_sid
                logger.warning(
                    "compression lock subsystem unavailable for session=%s "
                    "— proceeding without lock. This usually means a stale "
                    "in-memory module after an update; restart the process "
                    "(or `hermes update`) to resync.",
                    _lock_sid,
                )
            _lock_acquired = True  # acquired-but-unlocked compatibility path
        else:
            if commit_fence is not None:
                _lock_setup_entered = commit_fence.begin_lock_setup()
                if not _lock_setup_entered:
                    logger.info(
                        "Compression commit cancelled before lock acquisition "
                        "(session=%s).",
                        agent.session_id or "none",
                    )
                    agent._last_compaction_in_place = False
                    _existing_sp = getattr(agent, "_cached_system_prompt", None)
                    if not _existing_sp:
                        _existing_sp = agent._build_system_prompt(system_message)
                    _emit_compression_attempt_telemetry(
                        agent,
                        started_at=_attempt_started_at,
                        commit_status="aborted",
                        split_status="aborted",
                        failure_class="commit_fence_cancelled",
                    )
                    _complete_compaction_lifecycle()
                    return messages, _existing_sp
            try:
                _lock_acquired = _try_acquire_lock(
                    _lock_sid, _lock_holder, ttl_seconds=_lock_ttl
                )
                if _lock_acquired:
                    # Watermark (#75316): MAX(id) of active rows at compression
                    # START. Appends are NOT blocked while the slow provider
                    # summary runs — any row landing after this point is
                    # concurrent tail, and archive_and_compact() re-sequences
                    # it after the compacted set instead of archiving it.
                    try:
                        _commit_watermark = _lock_db.get_active_message_watermark(
                            _lock_sid
                        )
                    except Exception as _wm_err:
                        # Watermark capture is safety-additive: without it the
                        # commit falls back to archive-everything (historical
                        # behavior), so failure here must not abort compression.
                        logger.warning(
                            "compression watermark capture failed for "
                            "session=%s (%s) — concurrent appends this cycle "
                            "will be archived with the snapshot",
                            _lock_sid, _wm_err,
                        )
                        _commit_watermark = None
            except Exception as _lock_err:
                # The method exists and entered its implementation but failed.
                # Do not mistake an internal AttributeError or TypeError for
                # version skew: fail closed and preserve session lineage. A
                # failure after SQLite committed the acquire can leave our
                # holder row behind, so release it best-effort before returning
                # unchanged messages; release is holder-qualified and safe when
                # acquisition never succeeded.
                try:
                    _lock_db.release_compression_lock(_lock_sid, _lock_holder)
                except Exception as _release_err:
                    logger.debug(
                        "compression lock cleanup after failed acquire failed: %s",
                        _release_err,
                    )
                _lock_holder = None
                logger.warning(
                    "compression lock acquisition raised unexpectedly for "
                    "session=%s (%s: %s) — skipping compression this cycle",
                    _lock_sid, type(_lock_err).__name__, _lock_err,
                )
                _lock_acquired = False
        if not _lock_acquired:
            _finish_lock_setup()
            try:
                existing = _lock_db.get_compression_lock_holder(_lock_sid)
            except Exception:
                existing = None
            logger.warning(
                "compression skipped: another path is compressing session=%s "
                "(holder=%s) — returning messages unchanged to avoid session fork",
                _lock_sid, existing,
            )
            _lock_holder = None  # don't release a lock we don't own
            # Signal to callers that this no-op is due to a concurrent lock,
            # not a genuine "nothing to compress" or aux-model failure.
            # Manual /compress callers can surface a clear status message
            # instead of the misleading "No changes from compression" text.
            agent._compression_skipped_due_to_lock = existing or True
            # Surface to the user once — quiet for downstream auto-compress loops
            if getattr(agent, "_last_compression_lock_warning_sid", None) != _lock_sid:
                agent._last_compression_lock_warning_sid = _lock_sid
                try:
                    agent._emit_warning(
                        "⚠ Skipping concurrent compression — another path "
                        "is already compressing this session. Will retry "
                        "after it finishes."
                    )
                except Exception:
                    pass
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            try:
                if hasattr(agent.context_compressor, "_begin_compression_telemetry"):
                    agent.context_compressor._begin_compression_telemetry(current_tokens=approx_tokens)
            except Exception:
                pass
            _emit_compression_attempt_telemetry(
                agent,
                started_at=_attempt_started_at,
                commit_status="aborted",
                split_status="aborted",
                failure_class="lock_contended",
            )
            _complete_compaction_lifecycle()
            return messages, _existing_sp
    _lock_released = False
    _lock_release_guard = threading.Lock()

    def _release_lock_holder_only() -> None:
        """Stop this holder's refresher and release only its durable lock.

        Holder-qualified and idempotent (#76354 F4, from PR #71569): safe for
        the HOST to invoke after a timeout without an ABA race — the DB
        release is scoped to this worker's holder token, so a NEW holder's
        lease can never be deleted by this stale release.
        """
        nonlocal _lock_released
        with _lock_release_guard:
            if _lock_released:
                return
            _lock_released = True
            if getattr(agent, "_active_compression_lock_holder", None) == _lock_holder:
                agent._active_compression_lock_holder = None
            if _lock_refresher is not None:
                try:
                    _lock_refresher.stop()
                except Exception as _stop_err:
                    logger.debug("compression lock refresher stop failed: %s", _stop_err)
            if _lock_db is not None and _lock_sid and _lock_holder:
                try:
                    _lock_db.release_compression_lock(_lock_sid, _lock_holder)
                except Exception as _rel_err:
                    logger.debug("compression lock release failed: %s", _rel_err)

    def _release_lock() -> None:
        """Finish lifecycle cleanup and release the OLD session lock once."""
        try:
            _complete_compaction_lifecycle()
        finally:
            try:
                _release_lock_holder_only()
            finally:
                try:
                    if commit_fence is not None:
                        commit_fence.clear_cancelled_lock_release(
                            _release_lock_holder_only
                        )
                finally:
                    _finish_lock_setup()

    if _lock_holder is not None:
        agent._active_compression_lock_holder = _lock_holder
        if (
            commit_fence is not None
            and commit_fence.register_cancelled_lock_release(
                _release_lock_holder_only
            )
        ):
            # Cancellation already won while we were inside lock setup: the
            # hook just ran synchronously, our lease is gone — abort before
            # any summary work.
            logger.info(
                "Compression commit cancelled before summary dispatch "
                "(session=%s).",
                agent.session_id or "none",
            )
            agent._last_compaction_in_place = False
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            _emit_compression_attempt_telemetry(
                agent,
                started_at=_attempt_started_at,
                commit_status="aborted",
                split_status="aborted",
                failure_class="commit_fence_cancelled",
            )
            _release_lock()
            return messages, _existing_sp

    # Publish the holder-qualified release hook before a timeout can win the
    # fence. If no durable lock was acquired there is no hook to publish.
    _finish_lock_setup()

    # A delayed contender can acquire the parent lock after the winning path
    # has released it and completed rotation. The lock serializes work but does
    # not by itself prove that this stale agent still owns a live parent.
    if _lock_db is not None and _lock_sid:
        try:
            _parent_already_rotated = _session_was_rotated_by_compression(
                _lock_db, _lock_sid
            )
        except Exception as _session_err:
            logger.warning(
                "compression session ownership lookup failed for session=%s "
                "(%s: %s) - skipping compression this cycle",
                _lock_sid,
                type(_session_err).__name__,
                _session_err,
            )
            _release_lock()
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp
        if _parent_already_rotated:
            recovered_messages = _adopt_live_compression_child(
                agent, _lock_db, _lock_sid
            )
            _release_lock()
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            if recovered_messages is not None:
                logger.warning(
                    "compression recovery: stale session=%s adopted live child=%s",
                    _lock_sid,
                    agent.session_id,
                )
                return recovered_messages, _existing_sp
            logger.warning(
                "compression skipped: session=%s was already rotated by "
                "another compression path, but no unique live child could be adopted",
                _lock_sid,
            )
            return messages, _existing_sp

    # Snapshot the authoritative durable cooldown only after this attempt owns
    # the session lease. This runs for force=True too, but does not apply the
    # automatic breaker gate: manual compression still retries immediately.
    _durable_cooldown_authoritative, _durable_cooldown_state = (
        _capture_authoritative_cooldown_under_lease(
            agent.context_compressor,
            _compressor_attempt_snapshot,
        )
    )
    if _durable_cooldown_authoritative is False:
        # A bound built-in compressor reached its durable getter and the read
        # failed. Proceeding with force=True could clear an unknown newer row
        # before cancellation has enough information to restore it. This is a
        # persistence-safety abort, not automatic breaker gating.
        _release_lock()
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        return messages, existing_prompt

    # The agent may have been constructed before another path completed an
    # in-place compaction on the same session. Re-read durable breaker state
    # after acquiring the session lock so this final gate cannot act on the
    # stale snapshot loaded by bind_session_state().
    if not force:
        compressor = agent.context_compressor
        _refresh_persisted_compression_guards(
            compressor,
            include_cooldown=False,
        )
        blocked = getattr(
            type(compressor),
            "_automatic_compression_blocked",
            None,
        )
        if callable(blocked) and blocked(compressor):
            _release_lock()
            existing_prompt = getattr(agent, "_cached_system_prompt", None)
            if not existing_prompt:
                existing_prompt = agent._build_system_prompt(system_message)
            return messages, existing_prompt

    _activity_heartbeat: Optional[_CompressionActivityHeartbeat] = None
    messages_before_compression = None
    try:
        if _lock_holder is not None:
            _candidate_refresher = _CompressionLockLeaseRefresher(
                _lock_db,
                _lock_sid,
                _lock_holder,
                _lock_ttl,
                _lock_refresh_interval,
            )
            # Cancellation may release the holder after hook publication but
            # before this refresher starts. Serialize that check/start with
            # the idempotent release path so a refresher is never started for
            # an already-released lock (#76354 F4 / PR #71569).
            with _lock_release_guard:
                if not _lock_released:
                    _lock_refresher = _candidate_refresher
                    _lock_refresher.start()

        # The caller's history snapshot predates lease acquisition. Reload the
        # durable parent after the lease is live; MORE durable rows than the
        # snapshot carries means a frontend/background writer committed a turn
        # in that window, so publishing from this snapshot would omit it.
        # Deliberately a LENGTH check, not content equality: in-memory
        # mutation of past turns is legal (multimodal compression, retry
        # history replacement, think-tag stripping), and a content-equality
        # abort would permanently wedge compression on such sessions — the
        # #14694 failure shape.
        # Rotation-only: in-place compaction (archive_and_compact) is
        # non-destructive — pre-compaction rows are soft-archived (active=0,
        # compacted=1), stay searchable and recoverable, so snapshot/durable
        # drift cannot lose data there and must not abort compaction.
        #
        # When durable DID grow, ADOPT it and continue rather than aborting.
        # Aborting returned the stale snapshot unchanged, so busy sessions
        # (memory review / shared session_id writers) stayed permanently
        # behind the DB: every /compress and auto-compress saw
        # "changed before lease acquisition", surfaced as the misleading
        # "No changes from compression", and never reclaimed tokens.
        if not in_place and _lock_db is not None and _lock_sid:
            durable_loader = getattr(
                type(_lock_db), "get_messages_as_conversation", None
            )
            if callable(durable_loader):
                durable_parent = durable_loader(_lock_db, _lock_sid)
                if isinstance(durable_parent, list) and len(durable_parent) > len(messages):
                    # The in-memory transcript carries the CURRENT turn's
                    # un-persisted user tail (anchored by
                    # _persist_user_message_idx) that the durable snapshot read
                    # above does not contain yet. Flush that tail through the
                    # normal rotation-boundary path (conversation_history = the
                    # already-durable prefix, #68196 boundary) BEFORE adopting,
                    # then re-read the durable parent so the adopted snapshot
                    # includes the live input. If the flush fails (or the
                    # anchor is unknown), skip adoption entirely: replacing
                    # the in-memory transcript with a snapshot that lacks the
                    # user's input would silently drop it from the summarized
                    # and rotated history (#adopt-live-tail).
                    _preflush_idx = getattr(agent, "_persist_user_message_idx", None)
                    _preflush_ok = False
                    if (
                        isinstance(_preflush_idx, int)
                        and 0 <= _preflush_idx < len(messages)
                    ):
                        try:
                            _preflush_ok = agent._flush_messages_to_session_db(
                                messages,
                                conversation_history=messages[:_preflush_idx],
                            )
                        except Exception:
                            _preflush_ok = False
                    else:
                        # No known un-persisted tail (anchor unset or already
                        # at the end of the snapshot): the in-memory transcript
                        # is fully durable, so adopting the longer parent
                        # cannot drop live input — keep the legacy
                        # adopt-directly behavior for that shape
                        # (test_compression_concurrent_fork).
                        _preflush_ok = True
                    if not _preflush_ok:
                        logger.warning(
                            "compression: session=%s grew before lease "
                            "(%d → %d msgs) but the pre-adoption flush of the "
                            "live tail failed; skipping durable-snapshot "
                            "adoption so un-persisted user input is kept",
                            _lock_sid,
                            len(messages),
                            len(durable_parent),
                        )
                    else:
                        # Re-read after the flush so the adopted snapshot
                        # carries the just-persisted tail.
                        durable_parent = durable_loader(_lock_db, _lock_sid)
                    if (
                        _preflush_ok
                        and isinstance(durable_parent, list)
                        and len(durable_parent) > len(messages)
                    ):
                        logger.info(
                            "compression: session=%s grew before lease "
                            "(%d → %d msgs); adopting durable snapshot",
                            _lock_sid,
                            len(messages),
                            len(durable_parent),
                        )
                        messages = durable_parent
                        _pre_msg_count = len(messages)
                        # Token estimate was for the stale snapshot; clear it so
                        # the compressor re-derives from the adopted transcript
                        # instead of under-counting the newly visible rows.
                        approx_tokens = 0
                        # The whole adopted list is durable (DB re-read plus
                        # the just-flushed tail). Re-anchor the persist index
                        # at the end so the rotation-boundary flush that runs
                        # after compression skips the adopted rows by identity
                        # (conversation_history=messages[:idx]) instead of
                        # re-appending the concurrent rows and the live tail.
                        agent._persist_user_message_idx = len(messages)

        # Notify external memory provider before compression discards context.
        # The provider's on_pre_compress() may return a string of insights it
        # wants surfaced inside the compression summary; capture and forward it
        # instead of silently discarding the provider's return value.
        memory_context = ""
        memory_manager = getattr(agent, "_memory_manager", None)
        # Raw messages remain the historical (API v1) provider contract; the
        # normalized evidence list is handed only to API v2+ checkpoint
        # providers inside MemoryManager.on_pre_compress().
        evidence_messages = _direct_messages_for_pre_compress_memory(messages)
        if checkpoint_required:
            supports_checkpoint = getattr(
                memory_manager, "supports_pre_compress_checkpoint", None
            )
            if memory_manager is None or not callable(supports_checkpoint):
                raise _checkpoint_blocked(
                    f"no active provider implements checkpoint API "
                    f"v{PRE_COMPRESS_CHECKPOINT_API_VERSION}"
                )
            try:
                compatible = bool(
                    supports_checkpoint(PRE_COMPRESS_CHECKPOINT_API_VERSION)
                )
            except Exception as exc:
                raise _checkpoint_blocked("provider capability probe failed") from exc
            if not compatible:
                raise _checkpoint_blocked(
                    f"active provider does not implement checkpoint API "
                    f"v{PRE_COMPRESS_CHECKPOINT_API_VERSION}"
                )
            try:
                _maybe_ctx = memory_manager.on_pre_compress(
                    messages,
                    evidence_messages=evidence_messages,
                    require_checkpoint=True,
                    checkpoint_api_version=PRE_COMPRESS_CHECKPOINT_API_VERSION,
                )
            except Exception as exc:
                logger.warning(
                    "Required pre-compress checkpoint failed (%s)",
                    type(exc).__name__,
                )
                raise _checkpoint_blocked(
                    f"provider checkpoint API v{PRE_COMPRESS_CHECKPOINT_API_VERSION} failed"
                ) from exc
            if isinstance(_maybe_ctx, str):
                memory_context = sanitize_memory_context(_maybe_ctx)
        elif memory_manager:
            try:
                _maybe_ctx = memory_manager.on_pre_compress(
                    messages, evidence_messages=evidence_messages
                )
                if isinstance(_maybe_ctx, str):
                    memory_context = sanitize_memory_context(_maybe_ctx)
            except Exception:
                pass

        compress_fn = agent.context_compressor.compress
        compress_kwargs = _supported_compression_kwargs(
            compress_fn,
            current_tokens=approx_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )
        if memory_context.strip() and "memory_context" not in compress_kwargs:
            engine_name = getattr(
                agent.context_compressor,
                "name",
                type(agent.context_compressor).__name__,
            )
            if (
                getattr(agent, "_last_memory_context_unsupported_engine", None)
                != engine_name
            ):
                agent._last_memory_context_unsupported_engine = engine_name
                logger.warning(
                    "context engine %s does not accept memory_context; continuing "
                    "without provider-supplied summary context",
                    engine_name,
                )

        messages_before_compression = copy.deepcopy(messages)
        _activity_heartbeat = _CompressionActivityHeartbeat(
            agent, commit_fence=commit_fence
        ).start()
        # Publish forward progress to the commit fence while the summary LLM
        # call streams. Async hosts (gateway session hygiene) poll
        # ``commit_fence.seconds_since_progress()`` to extend their deadline
        # while tokens are moving — so a SLOW summary model is only killed
        # when it is actually silent, not merely thorough. The hook is
        # thread-local and the compress call is synchronous on this thread,
        # so it cannot leak into unrelated auxiliary calls.
        #
        # Callers that pass no commit_fence install a no-op progress hook
        # here.  AIAgent._compress_context injects an owned fence for
        # fenceless callers so the host-level progress-aware wait can
        # extend on streamed tokens; gateway hygiene already passes its
        # own fence.  An ACTIVE hook (even a no-op) is what switches the
        # summary call onto the streamed path — giving every compression
        # path the same two guarantees: the configured timeout acts on
        # inactivity (slow models finish), and a byte-trickling provider
        # that keeps the connection alive forever is cut off at the
        # streamed total ceiling (see _aux_stream_total_ceiling) instead of
        # outliving the SDK's inactivity timeout indefinitely.
        from agent.auxiliary_client import (
            aux_interrupt_protection,
            aux_progress_hook,
        )
        _progress_hook = (
            commit_fence.touch_progress if commit_fence is not None
            else (lambda: None)
        )
        # F4 state-ordering (#76354): a LATE successful summary must not undo
        # the timeout cooldown the host recorded. Install a cancellation
        # check the compressor consults BEFORE clearing the failure cooldown;
        # removed in the finally below so it cannot leak into later attempts
        # (e.g. a manual /compress force-clear).
        if commit_fence is not None:
            try:
                agent.context_compressor._compression_cancelled_check = (
                    lambda: commit_fence.is_cancelled
                )
            except Exception:
                pass
        # Incoming-message interrupts and active-turn redirects must not tear an
        # atomic summary in half (#23975). Explicit stop surfaces set a separate
        # Event atomically; never infer cause from the racy message fields.
        _hard_cancel_event = getattr(agent, "_hard_interrupt_requested", None)
        try:
            # F6: never start expensive summary work for an already-cancelled
            # fence (a stale queued job admitted after host departure).
            if commit_fence is not None and commit_fence.is_cancelled:
                logger.info(
                    "Compression cancelled before summary dispatch "
                    "(session=%s) — skipping summary work.",
                    agent.session_id or "none",
                )
                compressed = messages
            else:
                with aux_progress_hook(_progress_hook), aux_interrupt_protection(
                    cancel_event=_hard_cancel_event
                ):
                    compressed = compress_fn(messages, **compress_kwargs)
                    # Freeze a hard stop that arrived after the final provider
                    # attempt unwound but before this transaction can rotate
                    # session state.
                    if (
                        _hard_cancel_event is not None
                        and _hard_cancel_event.is_set()
                    ):
                        raise AuxiliaryExplicitCancellation()
        finally:
            if commit_fence is not None:
                try:
                    agent.context_compressor._compression_cancelled_check = None
                except Exception:
                    pass
    except AuxiliaryExplicitCancellation:
        try:
            _restore_compressor_attempt_state(
                agent.context_compressor,
                _compressor_attempt_snapshot,
                durable_cooldown_authoritative=_durable_cooldown_authoritative,
                durable_cooldown_state=_durable_cooldown_state,
            )
        except BaseException as _rollback_exc:
            # Compensation failure must surface, but it must not strand the
            # session lease or retain an in-memory transcript mutation.
            if (
                messages_before_compression is not None
                and messages != messages_before_compression
            ):
                messages[:] = copy.deepcopy(messages_before_compression)
            if _activity_heartbeat is not None:
                _activity_heartbeat.stop("context compression rollback failed")
                _activity_heartbeat = None
            _release_lock()
            _emit_compression_attempt_telemetry(
                agent,
                started_at=_attempt_started_at,
                commit_status="aborted",
                split_status="aborted",
                failure_class=f"rollback:{type(_rollback_exc).__name__}",
            )
            raise
        if (
            messages_before_compression is not None
            and messages != messages_before_compression
        ):
            messages[:] = copy.deepcopy(messages_before_compression)
        if _activity_heartbeat is not None:
            _activity_heartbeat.stop("context compression cancelled")
            _activity_heartbeat = None
        _release_lock()
        _emit_compression_attempt_telemetry(
            agent,
            started_at=_attempt_started_at,
            commit_status="aborted",
            split_status="aborted",
            failure_class="explicit_interrupt",
        )
        _existing_sp = getattr(agent, "_cached_system_prompt", None)
        if not _existing_sp:
            _existing_sp = agent._build_system_prompt(system_message)
        return messages, _existing_sp
    except BaseException as _compress_exc:
        # ANY exception after lock acquisition — memory hook, capability
        # inspection, engine lookup, or compress() — must release the lock so
        # the session isn't permanently blocked from future compression.
        if _activity_heartbeat is not None:
            _activity_heartbeat.stop("context compression failed")
            _activity_heartbeat = None
        _release_lock()
        _emit_compression_attempt_telemetry(
            agent,
            started_at=_attempt_started_at,
            commit_status="aborted",
            split_status="aborted",
            failure_class=f"exception:{type(_compress_exc).__name__}",
        )
        raise
    finally:
        if _activity_heartbeat is not None:
            _activity_heartbeat.stop("context compression completed")

    _commit_fence_entered = False
    try:
        # Capture boundary quality before session-rotation callbacks run. Built-in
        # and plugin lifecycle hooks may reset per-session compressor fields while
        # rebinding to the child id; the completed attempt's verdict must survive
        # that rebind and be recorded only after the full boundary commits.
        _compression_made_progress = bool(
            getattr(agent.context_compressor, "_last_compression_made_progress", False)
        )
        _compression_used_fallback = bool(
            getattr(agent.context_compressor, "_last_summary_fallback_used", False)
        )
        _compression_feasibility_skip = bool(
            getattr(agent.context_compressor, "_last_feasibility_skip", False)
        )

        # If compression aborted (aux LLM failed to produce a usable summary)
        # the compressor returns the input messages unchanged.  Surface the
        # error to the user, skip the session-rotation work entirely (no
        # session has logically ended), and let auto-compress callers detect
        # the no-op via len(returned) == len(input).
        if getattr(agent.context_compressor, "_last_compress_aborted", False):
            try:
                _err = getattr(agent.context_compressor, "_last_summary_error", None) or "unknown error"
                if getattr(agent, "_last_compression_summary_warning", None) != _err:
                    agent._last_compression_summary_warning = _err
                    agent._emit_warning(
                        f"⚠ Compression aborted: {_err}. "
                        "No messages were dropped — conversation continues unchanged. "
                        "Run /compress to retry, or /new to start a fresh session."
                    )
                _existing_sp = getattr(agent, "_cached_system_prompt", None)
                if not _existing_sp:
                    _existing_sp = agent._build_system_prompt(system_message)
                _emit_compression_attempt_telemetry(
                    agent,
                    started_at=_attempt_started_at,
                    commit_status="aborted",
                    split_status="aborted",
                    failure_class=(
                        getattr(agent.context_compressor, "_last_summary_error", None)
                        and "summary_generation_aborted"
                    ),
                )
                return messages, _existing_sp
            finally:
                _release_lock()

        # Compare against the pre-dispatch semantic state, not object identity:
        # legacy/plugin engines may return an equal copy for a no-op, or mutate
        # the live list while returning an unchanged snapshot. Neither case may
        # rotate or rewrite the session. The raw ``==`` leg runs FIRST so a
        # list subclass returned by an engine keeps its ``__eq__`` semantics
        # (tests seam on this); the marker-insensitive leg (#92231) then covers
        # the cold-resume shape where the stamped snapshot differs from the
        # marker-swept compress() output only by ``_db_persisted``.
        if compressed == messages_before_compression or (
            _strip_marker_for_comparison(compressed)
            == _strip_marker_for_comparison(messages_before_compression)
        ):
            if messages != messages_before_compression:
                messages[:] = copy.deepcopy(messages_before_compression)
            logger.info(
                "Compression made no progress (session=%s) — skipping boundary rewrite.",
                agent.session_id or "none",
            )
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            _emit_compression_attempt_telemetry(
                agent,
                started_at=_attempt_started_at,
                commit_status="aborted",
                split_status="aborted",
                failure_class="no_progress",
            )
            _release_lock()
            return messages, _existing_sp

        if not compressed:
            logger.error(
                "context compression returned an empty transcript; refusing to "
                "rotate session=%s so the parent remains resumable",
                agent.session_id or "none",
            )
            try:
                agent._emit_warning(
                    "⚠ Compression returned an empty transcript. "
                    "No session split was performed; conversation continues unchanged."
                )
            except Exception:
                pass
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            _release_lock()
            return messages, _existing_sp

        if commit_fence is not None:
            _commit_fence_entered = commit_fence.begin_commit(_hard_cancel_event)
            if not _commit_fence_entered:
                _restore_compressor_attempt_state(
                    agent.context_compressor,
                    _compressor_attempt_snapshot,
                    durable_cooldown_authoritative=_durable_cooldown_authoritative,
                    durable_cooldown_state=_durable_cooldown_state,
                )
                if (
                    messages_before_compression is not None
                    and messages != messages_before_compression
                ):
                    messages[:] = copy.deepcopy(messages_before_compression)
                logger.info(
                    "Compression commit cancelled before session mutation "
                    "(session=%s).",
                    agent.session_id or "none",
                )
                agent._last_compaction_in_place = False
                _existing_sp = getattr(agent, "_cached_system_prompt", None)
                if not _existing_sp:
                    _existing_sp = agent._build_system_prompt(system_message)
                _emit_compression_attempt_telemetry(
                    agent,
                    started_at=_attempt_started_at,
                    commit_status="aborted",
                    split_status="aborted",
                    failure_class="commit_fence_cancelled",
                )
                _release_lock()
                return messages, _existing_sp

        summary_error = getattr(agent.context_compressor, "_last_summary_error", None)
        if summary_error:
            if getattr(agent, "_last_compression_summary_warning", None) != summary_error:
                agent._last_compression_summary_warning = summary_error
                agent._emit_warning(
                    f"⚠ Compression summary failed: {summary_error}. "
                    "Inserted a fallback context marker."
                )
        else:
            # No hard failure — but did the configured aux model error out
            # and get recovered by retrying on main?  Surface that so users
            # know their auxiliary.compression.model setting is broken even
            # though compression succeeded.
            _aux_fail_model = getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
            _aux_fail_err = getattr(agent.context_compressor, "_last_aux_model_failure_error", None)
            if _aux_fail_model:
                # Dedup on (model, error) so we don't spam on every compaction
                _aux_key = (_aux_fail_model, _aux_fail_err)
                if getattr(agent, "_last_aux_fallback_warning_key", None) != _aux_key:
                    agent._last_aux_fallback_warning_key = _aux_key
                    agent._emit_warning(
                        f"ℹ Configured compression model '{_aux_fail_model}' failed "
                        f"({_aux_fail_err or 'unknown error'}). Recovered using main model — "
                        "check auxiliary.compression.model in config.yaml."
                    )

        todo_snapshot = agent._todo_store.format_for_injection()
        if todo_snapshot:
            # Retention parity (#84718): the snapshot below re-injects the
            # imperative verbatim. If this same boundary pruned skill bodies
            # to [SKILL_PRUNED: ...] markers, the policy that governed those
            # tasks is gone — couple a reload instruction to the snapshot so
            # the imperative never crosses the boundary alone. Appended after
            # TODO_INJECTION_HEADER, so the stale-snapshot strip removes both
            # together at the next boundary.
            _reload_notice = _pruned_skill_reload_notice(compressed)
            if _reload_notice:
                todo_snapshot = f"{todo_snapshot}\n\n{_reload_notice}"
            # Fold the snapshot into a trailing REAL user message so
            # compression never introduces a synthetic user/user pair. Any
            # snapshot merged at an earlier boundary is stripped first so
            # repeated compactions refresh rather than accumulate todo state
            # (#26981). Scaffolding tails (continuation marker, summary
            # handoff, a bare stale snapshot row) must never absorb the
            # snapshot: merging would upgrade them to "real user" evidence
            # and break zero-user provenance (#69292), so those keep the
            # flagged standalone append and the real-user preservation pass
            # continues to see todo scaffolding, not human intent.
            from agent.context_compressor import _append_text_to_content

            merged = False
            _tail = (
                compressed[-1]
                if compressed and isinstance(compressed[-1], dict)
                else None
            )
            if _tail is not None and _tail.get("role") == "user":
                _stripped = _strip_stale_todo_snapshot(_tail.get("content"))
                _probe = {
                    key: value for key, value in _tail.items() if key != "content"
                }
                _probe["content"] = _stripped
                if _is_real_user_message(_probe):
                    _snapshot_text = (
                        f"\n\n{todo_snapshot}"
                        if isinstance(_stripped, str) and _stripped
                        else todo_snapshot
                    )
                    _tail["content"] = _append_text_to_content(
                        _stripped, _snapshot_text
                    )
                    merged = True
                elif _stripped != _tail.get("content") and not _message_text(
                    {"role": "user", "content": _stripped}
                ).strip():
                    # The tail was nothing but an earlier snapshot row —
                    # refresh it in place instead of stacking a duplicate.
                    _tail["content"] = todo_snapshot
                    _tail["_todo_snapshot_synthetic"] = True
                    merged = True
            if not merged:
                compressed.append({
                    "role": "user",
                    "content": todo_snapshot,
                    "_todo_snapshot_synthetic": True,
                })
        _ensure_compressed_has_user_turn(messages, compressed)

        cached_system_prompt = agent._cached_system_prompt
        agent._invalidate_system_prompt()

        # Built-in memory is the only system-prompt input that a normal
        # compaction reloads. When the cached prompt already embeds the
        # freshly-reloaded memory blocks verbatim, keep the exact cached
        # prompt so local backends retain their KV-cache prefix. Containment
        # (not before/after snapshot equality) is required: fresh-agent
        # surfaces restore the cached prompt from the session DB, where it
        # can predate mid-session memory writes the in-memory snapshot has
        # already absorbed. External providers can change their own prompt
        # block during on_pre_compress(), so they retain the rebuild path.
        if (
            cached_system_prompt is not None
            and getattr(agent, "_memory_manager", None) is None
            and _cached_prompt_reflects_builtin_memory(agent, cached_system_prompt)
        ):
            new_system_prompt = cached_system_prompt
            agent._cached_system_prompt = cached_system_prompt
            # _invalidate_system_prompt() above also cleared the
            # cross-session-stable prefix marker boundary. The kept prompt
            # is byte-identical, so reconstruct the stable tier and reuse
            # it ONLY when the kept prompt still literally starts with it
            # (same startswith gate as the restore path); otherwise the
            # request layer falls back to the legacy single-breakpoint
            # layout with the prompt bytes untouched.
            from agent.system_prompt import reconstruct_static_prefix

            reconstruct_static_prefix(
                agent,
                system_message=system_message,
                log_label="compression keep-prompt",
            )
        else:
            new_system_prompt = agent._build_system_prompt(system_message)
            agent._cached_system_prompt = new_system_prompt

        _session_commit_succeeded = False
        split_status = "not_applicable"
        if agent._session_db:
            split_status = "pending"
            try:
                # Trigger memory extraction on the current session before the
                # transcript is rewritten (runs in BOTH modes — the logical
                # conversation's pre-compaction turns are about to be summarized
                # away regardless of whether the id rotates).
                agent.commit_memory_session(messages)

                # Anti-growth guard at the COMMIT SITE: never persist a
                # compression that makes the transcript larger (observed:
                # 379K -> 687K when the generated summary plus retained
                # reasoning exceeded what it replaced). Compare like-for-like
                # (both rough estimates of the same message shape) so an
                # "actual vs estimate" measurement mismatch cannot produce a
                # false verdict. The gateway has a rotation-path-only guard
                # (#83339), but in-place compaction commits inside this method
                # via archive_and_compact — before the gateway can inspect the
                # result — so the guard must live here to protect both paths.
                # On growth, treat the attempt as a no-op: the original
                # transcript stays untouched and durable.
                _rough_in = estimate_messages_tokens_rough(messages)
                _rough_out = estimate_messages_tokens_rough(compressed)
                if _rough_out > _rough_in:
                    # Todo refresh and user-turn anchoring happen after the
                    # compressor's own size check, so they can tip a break-even
                    # candidate over. Give it one mechanical salvage pass.
                    from agent.context_compressor import salvage_grown_transcript

                    _salvaged = salvage_grown_transcript(
                        messages, compressed, budget=_rough_in
                    )
                    if _salvaged is not None:
                        _salv_est = estimate_messages_tokens_rough(_salvaged)
                        if _salv_est < _rough_in:
                            logger.info(
                                "Compression salvage recovered a shrinking "
                                "transcript (session=%s, ~%s -> ~%s tokens)",
                                agent.session_id or "none",
                                f"{_rough_in:,}",
                                f"{_salv_est:,}",
                            )
                            compressed = _salvaged
                            _rough_out = _salv_est
                if _rough_out > _rough_in:
                    logger.warning(
                        "Compression refused: compressed transcript would be "
                        "larger than the original (session=%s, ~%s -> ~%s "
                        "tokens); keeping the original transcript unchanged",
                        agent.session_id or "none",
                        f"{_rough_in:,}",
                        f"{_rough_out:,}",
                    )
                    # Flag the refusal on the compressor state so manual
                    # /compress feedback can report it honestly. Without this,
                    # the CLI compared the returned list against its pre-call
                    # snapshot, saw a difference (durable-snapshot adoption can
                    # legitimately change the count), and printed
                    # "✅ Compressed: 8 → 14 messages" directly under the
                    # refusal warning (Aug 2026 full-surface CLI QA sweep).
                    try:
                        agent.context_compressor._last_compress_refused_would_grow = True
                    except Exception:
                        pass
                    try:
                        agent._emit_warning(
                            "⚠️ Compression refused: the generated summary "
                            "would have GROWN the conversation instead of "
                            "shrinking it. No messages were dropped — "
                            "conversation continues unchanged."
                        )
                    except Exception:
                        pass
                    _existing_sp = getattr(agent, "_cached_system_prompt", None)
                    if not _existing_sp:
                        _existing_sp = agent._build_system_prompt(system_message)
                    _emit_compression_attempt_telemetry(
                        agent,
                        started_at=_attempt_started_at,
                        commit_status="aborted",
                        split_status="aborted",
                        failure_class="would_grow",
                    )
                    # Record the rejected attempt as an ineffective
                    # compaction strike so the anti-thrash breaker latches
                    # after the normal threshold. Without this, the unchanged
                    # transcript stays over the compression threshold and
                    # automatic compression retries the identical summary
                    # request on every turn (#88568). Manual /compress keeps
                    # bypassing the latch (force=True skips the guards).
                    try:
                        agent.context_compressor.record_rejected_compaction()
                    except Exception:
                        logger.debug(
                            "could not record rejected-compaction strike",
                            exc_info=True,
                        )
                    # Restore ONLY the prune runway (same rationale as the
                    # rotation-failure rollback below): compress()'s successful
                    # tail already zeroed _proactive_prune_rearm_tokens in
                    # memory, but this refusal keeps the ORIGINAL transcript —
                    # whose cached prefix is intact. Leaving the runway at 0
                    # disarms the #79640 throttle, so the very next iteration's
                    # proactive prune rewrites history and breaks the prompt
                    # cache without the required regrowth interval (#91830).
                    # The durable copy was never cleared (that clear only rides
                    # the archive_and_compact / child-row commit that never
                    # ran), so restoring the snapshot re-aligns memory with
                    # disk.
                    if "_proactive_prune_rearm_tokens" in _compressor_attempt_snapshot:
                        agent.context_compressor._proactive_prune_rearm_tokens = (
                            _compressor_attempt_snapshot[
                                "_proactive_prune_rearm_tokens"
                            ]
                        )
                    _release_lock()
                    return messages, _existing_sp

                if in_place:
                    # ── In-place compaction: keep the same session_id ──────────
                    # No end_session, no new row, no parent_session_id, no title
                    # renumber, no contextvar/env/logging re-sync. The session's
                    # id, title, cwd, /goal, and gateway routing all stay put.
                    #
                    # Durable, NON-DESTRUCTIVE replace: soft-archive the
                    # pre-compaction turns (active=0, kept on disk + FTS-searchable +
                    # recoverable) and insert `compressed` as the new live (active=1)
                    # set, atomically. `compressed` already carries the surviving
                    # tail (current-turn messages the compressor kept via
                    # protect_last_n), so we DON'T pre-flush here — a flush would
                    # INSERT current-turn rows that archive_and_compact would then
                    # archive alongside the rest (harmless but wasted writes). The
                    # live-context load filters active=1, so a resume reloads ONLY
                    # the compacted set; the original turns remain under the SAME id
                    # for search/recovery (Teknium review — keep one durable id
                    # WITHOUT destroying history, unlike a hard replace_messages).
                    # See #38763.
                    from agent.context_compressor import (
                        PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY,
                    )

                    agent._session_db.archive_and_compact(
                        agent.session_id,
                        compressed,
                        model_config_patch={
                            PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY: None,
                        },
                        watermark=_commit_watermark,
                        lock_holder=_lock_holder,
                    )
                    split_status = "in_place_committed"
                    # Reset the flush identity set so the next turn's appends are
                    # diffed against the COMPACTED transcript: the compacted dicts
                    # are passed as conversation_history next turn and skipped by
                    # identity, so only genuinely new turn messages get appended
                    # (no dup of the summary, no resurrection of dropped turns).
                    agent._flushed_db_message_ids = set()
                    # Rotation-independent signal: the conversation was compacted in
                    # place (id unchanged). The gateway reads this (NOT an id-change
                    # diff) to re-baseline transcript handling.
                    compacted_in_place = True
                else:
                    # ── Rotation (legacy): end this session, fork a continuation ─
                    # Flush any un-persisted current-turn messages to the OLD
                    # session before ending it, so they survive in the preserved
                    # parent transcript (#47202). (In-place skips this — see above.)
                    #
                    # Pass the already-durable prefix as conversation_history so
                    # the flush skips it by identity (#68196). Preflight
                    # compression runs BEFORE the normal turn flush has stamped
                    # the cold-resumed history dicts with _DB_PERSISTED_MARKER, so
                    # without a boundary _flush_messages_to_session_db treats every
                    # restored row as new and re-appends the whole transcript to
                    # the parent. turn_context anchors _persist_user_message_idx at
                    # the current-turn user message before preflight runs, so
                    # messages[:idx] is exactly the persisted prefix; only the
                    # current turn's new messages get written.
                    #
                    # Bound to old_session_id, hoisted above the flush: the
                    # ``except`` handler below keys its in-memory rollback off
                    # this name, so anything that fails from here on rolls the
                    # transcript back instead of leaving the failed attempt's
                    # compacted snapshot in place.
                    old_session_id = agent.session_id
                    current_idx = getattr(agent, "_persist_user_message_idx", None)
                    persisted_history = (
                        messages[:current_idx]
                        if isinstance(current_idx, int)
                        and 0 <= current_idx <= len(messages)
                        else None
                    )
                    # The #47202 flush below is a DURABLE append to the parent
                    # and it is NOT undone when the rotation aborts: the
                    # ``except`` handler restores the in-memory transcript and
                    # keeps agent.session_id on the parent, but the rows it just
                    # wrote stay. Survivable for a one-off failure; pathological
                    # for a STICKY one. A parent row that already carries
                    # ``ended_at`` fails publish_compression_child on every
                    # attempt and nothing in this path clears it, so each
                    # auto-compaction appends another copy of the current turn to
                    # the transcript it was supposed to shrink — the session grows
                    # until the provider rejects the request outright (#88197:
                    # 303 unique messages stored as 2,611 rows after 7 aborted
                    # attempts, ~1.66M tokens, HTTP 400).
                    #
                    # So check that one precondition BEFORE writing. It is a plain
                    # read of the row the publish is about to read anyway, it
                    # raises the publish's own message so the log line, telemetry
                    # and rollback path are all unchanged, and it cannot mask a
                    # real rotation — a live parent reaches the flush exactly as
                    # before. Deliberately NOT extended to the compression lease:
                    # a lease is re-acquirable, so a transient miss here would
                    # abort a rotation that would otherwise have committed.
                    _parent_row_reader = getattr(agent._session_db, "get_session", None)
                    _parent_already_ended = False
                    if callable(_parent_row_reader):
                        try:
                            _parent_row = _parent_row_reader(old_session_id) or {}
                            _parent_already_ended = (
                                _parent_row.get("ended_at") is not None
                            )
                        except Exception:
                            # Fail OPEN: an unreadable row must not turn a cheap
                            # guard into a new way to lose compression.
                            _parent_already_ended = False
                    if _parent_already_ended:
                        raise RuntimeError(
                            f"Compression parent already ended: {old_session_id}"
                        )
                    # Foreign-tail ceiling (#75316): the flush below writes OUR
                    # OWN input transcript to the parent — those rows are
                    # already represented in the compacted handoff and must
                    # not be cloned into the child. Everything at or below
                    # this MAX(id) but above the start-watermark is a foreign
                    # concurrent append; everything above it is our flush.
                    try:
                        _foreign_tail_ceiling = (
                            agent._session_db.get_active_message_watermark(
                                agent.session_id
                            )
                        )
                    except Exception:
                        # Without a trustworthy ceiling the clone could
                        # duplicate the handoff — fall back to historical
                        # behavior (no tail preservation this rotation).
                        _foreign_tail_ceiling = None
                    try:
                        agent._flush_messages_to_session_db(
                            messages,
                            conversation_history=persisted_history,
                        )
                    except Exception:
                        pass  # best-effort — don't block compression on a flush error
                    # Publish parent closure + child row + compacted handoff in
                    # one transaction. No reader can observe a missing/empty child.
                    # The rotation child must stay on the parent's profile —
                    # mirror _ensure_db_session's stamp ("default" persists as
                    # NULL). publish_compression_child additionally COALESCEs
                    # from the parent row, covering app-global remote sessions
                    # whose thread lacks the HERMES_HOME context.
                    try:
                        from hermes_cli.profiles import get_active_profile_name

                        _profile_for_child = get_active_profile_name()
                        if _profile_for_child == "default":
                            _profile_for_child = None
                    except Exception:
                        _profile_for_child = None
                    old_title = agent._session_db.get_session_title(agent.session_id)
                    new_session_id = (
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                        f"{uuid.uuid4().hex[:6]}"
                    )
                    agent._session_db.publish_compression_child(
                        parent_session_id=old_session_id,
                        child_session_id=new_session_id,
                        source=agent.platform
                        or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                        model=agent.model,
                        model_config=agent._session_init_model_config,
                        system_prompt=new_system_prompt,
                        messages=compressed,
                        cwd=getattr(agent, "working_directory", None),
                        profile_name=_profile_for_child,
                        compression_lock_holder=_lock_holder,
                        require_compression_lease=_lock_holder is not None,
                        watermark=(
                            _commit_watermark
                            if _foreign_tail_ceiling is not None
                            else None
                        ),
                        watermark_ceiling=_foreign_tail_ceiling,
                    )
                    agent.session_id = new_session_id
                    try:
                        from gateway.session_context import set_current_session_id

                        set_current_session_id(agent.session_id)
                    except Exception:
                        os.environ["HERMES_SESSION_ID"] = agent.session_id
                    try:
                        from hermes_logging import set_session_context

                        set_session_context(agent.session_id)
                    except Exception:
                        pass
                    agent._session_db_created = True
                    split_status = "rotated_committed"
                    # Carry a persistent /goal onto the continuation session.
                    # Compression mints a fresh child id; load_goal does a flat
                    # per-session lookup with no parent walk, so without this an
                    # active goal silently dies at the boundary (#33618).
                    try:
                        from hermes_cli.goals import migrate_goal_to_session
                        migrate_goal_to_session(old_session_id, agent.session_id, reason="compression")
                    except Exception as _goal_err:
                        logger.debug("Could not migrate goal on compression: %s", _goal_err)
                    # Same boundary hazard for /heartbeat state — carry it too.
                    try:
                        from hermes_cli.heartbeat import migrate_heartbeat_to_session
                        migrate_heartbeat_to_session(old_session_id, agent.session_id)
                    except Exception as _hb_err:
                        logger.debug("Could not migrate heartbeat on compression: %s", _hb_err)
                    # Same boundary hazard for a persistent /loop — carry it
                    # onto the continuation session so the recurring wakeups
                    # survive compression.
                    try:
                        from hermes_cli.loops import migrate_loop_to_session
                        migrate_loop_to_session(old_session_id, agent.session_id, reason="compression")
                    except Exception as _loop_err:
                        logger.debug("Could not migrate loop on compression: %s", _loop_err)
                    # Carry the title across the compression boundary unchanged.
                    #
                    # This used to renumber ("Fix X" → "Fix X #2") on every
                    # rotation, which is why a long conversation ended up as
                    # "Smallville Map Architecture Plan #10" — ten forks of ONE
                    # session, each looking like a separate piece of work in the
                    # sidebar. Compression is an internal implementation detail;
                    # the user's conversation did not change topic, so its name
                    # must not change either. Uniqueness still holds because
                    # _set_session_title transfers the title off a hidden
                    # compression ancestor rather than raising on the conflict.
                    if old_title:
                        # Read provenance BEFORE the write: transferring the
                        # title off a hidden compression ancestor clears the
                        # ancestor's row, so reading afterwards always returns
                        # None and the child would be stamped "user" — freezing
                        # an auto-title that should still be upgradeable.
                        _src = None
                        try:
                            _src = agent._session_db.get_session_title_source(
                                old_session_id
                            )
                        except Exception as _src_err:
                            logger.debug(
                                "Could not read title provenance: %s", _src_err
                            )
                        try:
                            agent._session_db.set_session_title(
                                agent.session_id, old_title
                            )
                        except (ValueError, Exception) as e:
                            logger.debug("Could not propagate title on compression: %s", e)
                        else:
                            # set_session_title() records "user"; restore the
                            # original authority so an inherited auto-title
                            # stays upgradeable and a manual one stays pinned.
                            if _src is not None:
                                try:
                                    agent._session_db.set_session_title_source(
                                        agent.session_id, _src
                                    )
                                except Exception as _src_err:
                                    logger.debug(
                                        "Could not propagate title provenance: %s",
                                        _src_err,
                                    )

                # In-place mode still updates/replaces the current row here.
                # Rotation already published prompt + compacted handoff atomically.
                if in_place:
                    agent._session_db.update_system_prompt(
                        agent.session_id, new_system_prompt
                    )
                    agent._last_flushed_db_idx = 0
                else:
                    agent._last_flushed_db_idx = len(compressed)
                    agent._flushed_db_message_session_id = agent.session_id
                    agent._flushed_db_message_ids = {
                        id(message)
                        for message in compressed
                        if isinstance(message, dict)
                    }
                _session_commit_succeeded = True
            except Exception as e:
                if (
                    not in_place
                    and locals().get("old_session_id")
                    and agent.session_id == old_session_id
                ):
                    # Atomic publication failed (including lease loss): keep the
                    # parent live and discard the stale compacted snapshot.
                    old_session_id = None
                    messages[:] = copy.deepcopy(messages_before_compression)
                    compressed = messages
                    _compression_made_progress = False
                    # Restore ONLY the prune runway, not the full attempt
                    # snapshot: _restore_compressor_attempt_state is reserved
                    # for pre-commit cancels (fence deny / explicit cancel),
                    # while this branch is post-attempt — the other snapshot
                    # fields (telemetry, aborted flags) must keep the failed
                    # attempt's values. The runway is a property of transcript
                    # state, and the transcript was just rolled back to its
                    # pre-compression copy, so the runway rolls back with it.
                    # (compress() zeroed it in-memory on summary success; the
                    # durable copy was never cleared — that clear only rides
                    # the atomic archive_and_compact / child-row publication
                    # that just failed.)
                    if "_proactive_prune_rearm_tokens" in _compressor_attempt_snapshot:
                        agent.context_compressor._proactive_prune_rearm_tokens = (
                            _compressor_attempt_snapshot[
                                "_proactive_prune_rearm_tokens"
                            ]
                        )
                split_status = (
                    "aborted"
                    if locals().get("old_session_id") is None and not in_place
                    else "failed_not_indexed"
                )
                # If the rotation rolled back to the parent (orphan-avoidance
                # above), agent.session_id is the still-indexed parent and
                # old_session_id was cleared — so this is recovery, not an
                # un-indexed orphan. Otherwise an earlier step failed before the
                # child was created and the warning's original meaning holds.
                if locals().get("old_session_id") is None and not in_place:
                    logger.warning(
                        "Compression rotation aborted and rolled back to the "
                        "parent session (%s): %s", agent.session_id or "?", e,
                    )
                else:
                    logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # Compaction-boundary bookkeeping, computed once. `old_session_id` is only
        # bound in the rotation branch; in-place leaves it unset. `_boundary_parent`
        # is the id the boundary notifications attribute the prior state to: the old
        # id on rotation, the (unchanged) current id in-place.
        _old_sid = locals().get("old_session_id")
        _is_boundary = bool(_old_sid) or in_place
        _context_engine_boundary_committed = _session_commit_succeeded and (
            bool(_old_sid) or compacted_in_place
        )
        _boundary_parent = _old_sid or agent.session_id or ""

        # Round-2 #4: the activity heartbeat's terminal "context compression
        # completed" stamp landed on the PARENT row (force-persisted before
        # the rotation re-pointed agent.session_id at the child). Without a
        # cleanup, the archived parent advertises a fresh last_activity_at +
        # "context compression completed" forever — a permanent false-fresh
        # row for any activity consumer that scans ended sessions. Clear the
        # labels on the parent best-effort (keeps last_activity_at so idle
        # clocks stay continuous; the CHILD carries the live labels).
        if _old_sid and _session_commit_succeeded:
            try:
                _labels_db = getattr(agent, "_session_db", None)
                _clear_labels = getattr(
                    type(_labels_db) if _labels_db is not None else None,
                    "clear_session_activity_labels",
                    None,
                )
                if callable(_clear_labels):
                    _clear_labels(_labels_db, _old_sid)
            except Exception:
                logger.debug(
                    "failed to clear archived compression parent's activity "
                    "labels (ignored)",
                    exc_info=True,
                )

        # Notify the context engine that a compaction boundary occurred. Plugin
        # engines (e.g. hermes-lcm) use boundary_reason="compression" to preserve
        # DAG lineage / checkpoint per-session state across the boundary instead of
        # re-initializing fresh. See hermes-lcm#68. Built-in ContextCompressor
        # ignores kwargs. Fires in BOTH modes: rotation passes old→new ids; in-place
        # passes the SAME id (the boundary is real even though the id didn't move).
        if _context_engine_boundary_committed:
            if defer_context_engine_notification:
                _queue_context_engine_compression_notification(
                    agent,
                    new_session_id=agent.session_id or "",
                    old_session_id=_boundary_parent,
                )
            else:
                _notify_context_engine_compression_complete(
                    agent,
                    new_session_id=agent.session_id or "",
                    old_session_id=_boundary_parent,
                )

        # Notify memory providers of the compaction boundary so provider-cached
        # per-session state (Hindsight's _document_id, accumulated turn buffers,
        # counters) refreshes. reset=False because the logical conversation
        # continues. See #6672. Fires in BOTH modes: in-place uses the same id as
        # parent (the conversation didn't fork, but the buffer must still be told
        # the transcript was compacted so it doesn't double-count dropped turns).
        try:
            if _is_boundary and agent._memory_manager:
                agent._memory_manager.on_session_switch(
                    agent.session_id or "",
                    parent_session_id=_boundary_parent,
                    reset=False,
                    reason="compression",
                )
        except Exception as _me_err:
            logger.debug("memory manager on_session_switch (compression): %s", _me_err)

        # Warn on repeated compressions (quality degrades with each pass).
        # Route through _emit_status (like the other compression warnings above)
        # so the warning reaches the TUI / Telegram / Discord via status_callback,
        # not just CLI stdout. _emit_status still _vprints for the CLI, and
        # storing it on _compression_warning lets replay_compression_warning
        # re-deliver it once a late-bound gateway status_callback is wired (#36908).
        _cc = agent.context_compressor.compression_count
        if _cc >= 2:
            _cc_msg = (
                f"{agent.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh."
            )
            agent._compression_warning = _cc_msg
            agent._emit_status(_cc_msg)

        # Emit session:compress event so hooks (e.g. MemPalace sync) can ingest
        # the completed old session before its details are lost. In in-place mode
        # there is no old id (same session); ``in_place=True`` tells hooks the
        # transcript was compacted on the same id rather than rotated.
        if getattr(agent, "event_callback", None):
            try:
                agent.event_callback("session:compress", {
                    "platform": agent.platform or "",
                    "session_id": agent.session_id,
                    "old_session_id": _old_sid or "",
                    "in_place": in_place,
                    "compression_count": agent.context_compressor.compression_count,
                })
            except Exception as e:
                logger.debug("event_callback error on session:compress: %s", e)

        # Surface the compaction mode to the caller (run_conversation / gateway)
        # via a rotation-independent flag. The gateway uses this — NOT an
        # id-change diff — to re-baseline transcript handling (history_offset=0 +
        # rewrite on the same id) when compaction happened in place. See #38763.
        agent._last_compression_attempt_in_place = compacted_in_place
        agent._last_compaction_in_place = compacted_in_place

        # Keep the post-compression rough estimate for diagnostics, but do not
        # treat it as provider-reported prompt usage. Schema-heavy rough estimates
        # can remain above threshold even after the next real API request fits.
        _compressed_est = estimate_request_tokens_rough(
            compressed,
            system_prompt=new_system_prompt or "",
            tools=agent.tools or None,
        )
        agent.context_compressor.last_compression_rough_tokens = _compressed_est
        agent.context_compressor.last_prompt_tokens = -1
        agent.context_compressor.last_completion_tokens = 0
        agent.context_compressor.awaiting_real_usage_after_compression = True
        # Arm the effectiveness verdict only after a completed rewrite crosses
        # the full compaction boundary. Exceptions, aborts, and no-op attempts
        # leave this false, so unrelated later usage cannot be charged to an
        # attempt that never changed the transcript.
        if _compression_made_progress:
            record_boundary = getattr(
                type(agent.context_compressor),
                "record_completed_compaction",
                None,
            )
            if callable(record_boundary):
                record_boundary(
                    agent.context_compressor,
                    used_fallback=_compression_used_fallback,
                    feasibility_skip=_compression_feasibility_skip,
                )
            else:
                agent.context_compressor._verify_compaction_cleared_threshold = True

        # Clear the file-read dedup cache.  After compression the original
        # read content is summarised away — if the model re-reads the same
        # file it needs the full content, not a "file unchanged" stub.
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass
        # Same for the skill_view repeat-view dedup: a post-compression
        # re-view must return the full skill content again.
        try:
            from tools.skills_tool import reset_skill_view_dedup
            reset_skill_view_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d rough_tokens=~%s awaiting_real_usage=true",
            agent.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        _commit_status = "committed" if split_status in {"not_applicable", "in_place_committed", "rotated_committed"} else "aborted"
        _emit_compression_attempt_telemetry(
            agent,
            started_at=_attempt_started_at,
            commit_status=_commit_status,
            split_status=split_status,
            failure_class=(
                "session_split_failed"
                if split_status in {"failed_not_indexed", "aborted"}
                else None
            ),
        )
        _compaction_succeeded = True
        return compressed, new_system_prompt
    finally:
        # Release the lock on the OLD session_id only AFTER rotation completed
        # and all post-rotation bookkeeping (memory manager, context engine,
        # file dedup) ran. A concurrent path that wakes up the moment we
        # release will see the NEW session_id in state.db / SessionEntry and
        # acquire on that — no race against our just-finished work.
        try:
            _release_lock()
        finally:
            if _commit_fence_entered:
                commit_fence.finish_commit()


def _compress_context_via_codex_app_server(
    agent: Any,
    messages: list,
    system_message: Optional[str],
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    force: bool = False,
) -> Tuple[list, str]:
    """Route compaction to Codex app-server for Codex-owned threads.

    Hermes' normal compressor rewrites the local OpenAI-style transcript.
    That does not shrink the actual Codex app-server thread context. For this
    runtime, ask Codex to compact its own thread and keep Hermes' transcript
    unchanged.
    """
    auto_mode = str(
        getattr(agent, "codex_app_server_auto_compaction", "native") or "native"
    ).lower()
    if auto_mode not in {"native", "hermes", "off"}:
        auto_mode = "native"
    if not force and auto_mode != "hermes":
        logger.info(
            "codex app-server compaction skipped: mode=%s force=false "
            "(session=%s messages=%d tokens=~%s)",
            auto_mode,
            getattr(agent, "session_id", None) or "none",
            len(messages),
            f"{approx_tokens:,}" if approx_tokens else "unknown",
        )
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        return messages, existing_prompt

    codex_session = getattr(agent, "_codex_session", None)
    if codex_session is None:
        logger.info(
            "codex app-server compaction skipped: no active codex thread "
            "(session=%s messages=%d tokens=~%s)",
            getattr(agent, "session_id", None) or "none",
            len(messages),
            f"{approx_tokens:,}" if approx_tokens else "unknown",
        )
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        return messages, existing_prompt

    logger.info(
        "codex app-server compaction started: session=%s messages=%d tokens=~%s",
        getattr(agent, "session_id", None) or "none",
        len(messages),
        f"{approx_tokens:,}" if approx_tokens else "unknown",
    )
    try:
        agent._emit_status(COMPACTION_STATUS)
    except Exception:
        pass

    _compaction_done_emitted = False
    _compaction_succeeded = False

    def _complete_compaction_lifecycle() -> None:
        nonlocal _compaction_done_emitted
        if _compaction_done_emitted:
            return
        _compaction_done_emitted = True
        if _compaction_succeeded:
            _emit_compaction_done(agent)

    _activity_heartbeat: Optional[_CompressionActivityHeartbeat] = None
    try:
        _activity_heartbeat = _CompressionActivityHeartbeat(agent).start()
        result = codex_session.compact_thread()
    except BaseException:
        if _activity_heartbeat is not None:
            _activity_heartbeat.stop("context compression failed")
        _complete_compaction_lifecycle()
        raise

    if getattr(result, "interrupted", False) or getattr(result, "error", None):
        _activity_heartbeat.stop("context compression failed")
    else:
        _activity_heartbeat.stop("context compression completed")

    if getattr(result, "should_retire", False):
        try:
            codex_session.close()
        except Exception:
            pass
        agent._codex_session = None

    if getattr(result, "interrupted", False) or getattr(result, "error", None):
        try:
            agent._emit_warning(
                f"⚠ Codex app-server compaction failed: {result.error}"
            )
        except Exception:
            pass
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        _complete_compaction_lifecycle()
        return messages, existing_prompt

    try:
        from agent.codex_runtime import (
            _record_codex_app_server_compaction,
            _record_codex_app_server_usage,
        )

        _record_codex_app_server_compaction(
            agent,
            result,
            approx_tokens=approx_tokens,
            force=True,
        )
        # An empty usage report must consume the pending post-compaction verdict
        # rather than leaving preflight deferral armed until some unrelated later
        # Codex turn supplies usage. Minimal external test engines may not expose
        # the ContextEngine update hook; preserve their existing bookkeeping.
        if hasattr(agent.context_compressor, "update_from_response"):
            _record_codex_app_server_usage(agent, result)
    except Exception:
        logger.debug("codex compaction bookkeeping failed", exc_info=True)

    try:
        from tools.file_tools import reset_file_dedup

        reset_file_dedup(task_id)
    except Exception:
        pass

    logger.info(
        "codex app-server compaction done: session=%s thread=%s turn=%s",
        getattr(agent, "session_id", None) or "none",
        getattr(result, "thread_id", None) or "",
        getattr(result, "turn_id", None) or "",
    )
    existing_prompt = getattr(agent, "_cached_system_prompt", None)
    if not existing_prompt:
        existing_prompt = agent._build_system_prompt(system_message)
    _compaction_succeeded = True
    _complete_compaction_lifecycle()
    return messages, existing_prompt


def try_shrink_image_parts_in_messages(
    api_messages: list,
    *,
    max_dimension: int = 8000,
) -> bool:
    """Re-encode all native image parts at a smaller size to recover from
    image-too-large errors (Anthropic 5 MB, unknown other providers).

    Mutates ``api_messages`` in place. Returns True if any image part was
    actually replaced, False if there were no image parts to shrink or
    Pillow couldn't help (caller should surface the original error).

    Strategy: look for ``image_url`` / ``input_image`` parts carrying a
    ``data:image/...;base64,...`` payload, plus Anthropic-native
    ``{"type": "image", "source": {"type": "base64", ...}}`` blocks.
    For each one whose encoded size exceeds 4 MB (a safe target that slides
    under Anthropic's 5 MB ceiling with header overhead) or whose longest side
    exceeds ``max_dimension``, write the base64 to a tempfile, call
    ``vision_tools._resize_image_for_vision`` to produce a smaller data
    URL, and substitute it in place.

    Non-data-URL images (http/https URLs) are not touched — the provider
    fetches those itself and the size limit is different.
    """
    if not api_messages:
        return False

    try:
        from tools.vision_tools import _resize_image_for_vision
    except Exception as exc:
        logger.warning("image-shrink recovery: vision_tools unavailable — %s", exc)
        return False

    # 4 MB target leaves comfortable headroom under Anthropic's 5 MB.
    # Non-Anthropic providers we haven't observed rejecting are fine with
    # much larger; shrinking to 4 MB here loses quality but only fires
    # after a confirmed provider rejection, so the alternative is failure.
    target_bytes = 4 * 1024 * 1024
    # Anthropic enforces an 8000px per-side dimension cap independently of
    # the 5 MB byte cap.  In many-image requests, the provider can report a
    # lower cap (observed: 2000px).  The caller passes that parsed ceiling
    # when the rejection includes it.
    changed_count = 0
    # Track parts that are over the target but could NOT be shrunk under it.
    # If any survive, retrying is pointless — the same oversized payload will
    # be re-sent and rejected again, wasting the single retry budget.  We only
    # report success (caller retries) when every over-threshold image was
    # actually brought under the target.
    unshrinkable_oversized = 0

    def _decode_pixels(data_url: str) -> Optional[tuple]:
        """Return ``(width, height)`` of a base64 data URL, or None on failure.

        Soft-depends on Pillow; returns None (caller falls back to a
        bytes-only check) if Pillow is missing or the payload is corrupt.
        """
        try:
            import base64 as _b64_dim
            import io as _io_dim
            header_d, _, data_d = data_url.partition(",")
            if not data_d or not data_url.startswith("data:"):
                return None
            from PIL import Image as _PILImage
            with _PILImage.open(_io_dim.BytesIO(_b64_dim.b64decode(data_d))) as _img:
                return _img.size
        except Exception:
            return None

    def _shrink_data_url(url: str) -> tuple:
        """Return ``(resized_url, unshrinkable)`` for a data URL.

        ``resized_url`` is a smaller/dimension-correct data URL, or None when
        no rewrite was applied.  ``unshrinkable`` is True only when the image
        exceeded a constraint (byte-size or dimensions) and the resize failed
        to satisfy *that same* constraint — so the caller knows retrying is
        pointless even if a different image in the request shrank.
        """
        if not isinstance(url, str) or not url.startswith("data:"):
            return None, False

        # Determine which constraint is binding.  The accept/reject gate below
        # MUST be checked against the same axis that triggered the shrink: a
        # downscaled screenshot PNG routinely re-encodes to *more* bytes than
        # the original (PNG compression is non-monotonic in image size — a
        # smaller raster with LANCZOS resampling noise compresses worse than a
        # larger smooth one).  Rejecting a pixel-correct downscale purely
        # because its bytes grew permanently wedges sessions on the Anthropic
        # many-image 2000px path (#48013).
        needs_shrink = len(url) > target_bytes  # over byte budget
        triggered_by = "bytes" if needs_shrink else None
        if not needs_shrink:
            # Bytes are fine — check pixel dimensions against the provider's
            # reported per-side cap.  A screenshot can be tiny in bytes yet
            # too large in pixels.
            dims = _decode_pixels(url)
            if dims is None:
                # Pillow missing or corrupt data — fall back to byte-only.
                return None, False
            if max(dims) <= max_dimension:
                return None, False  # both bytes and pixels are within limits
            needs_shrink = True
            triggered_by = "dimension"

        try:
            header, _, data = url.partition(",")
            mime = "image/jpeg"
            if header.startswith("data:"):
                mime_part = header[len("data:"):].split(";", 1)[0].strip()
                if mime_part.startswith("image/"):
                    mime = mime_part
            import base64 as _b64
            raw = _b64.b64decode(data)
            suffix = {
                "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
                "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/bmp": ".bmp",
            }.get(mime, ".jpg")
            tmp = tempfile.NamedTemporaryFile(
                prefix="hermes_shrink_", suffix=suffix, delete=False,
            )
            try:
                tmp.write(raw)
                tmp.close()
                resized = _resize_image_for_vision(
                    Path(tmp.name),
                    mime_type=mime,
                    max_base64_bytes=target_bytes,
                    max_dimension=max_dimension,
                )
            finally:
                try:
                    Path(tmp.name).unlink(missing_ok=True)
                except Exception:
                    pass
            if not resized:
                # Resize returned nothing — Pillow couldn't help.
                return None, True
            if triggered_by == "bytes":
                # Byte budget is the binding constraint — bytes must shrink.
                if len(resized) >= len(url):
                    return None, True  # re-encode made it bigger
                # The per-side dimension cap is ALSO an active provider
                # constraint on this request (the caller passes the parsed cap
                # to both this helper and the resizer).  _resize_image_for_vision
                # returns a best-effort, possibly-over-cap blob when it
                # exhausts its halving budget — it freezes the long side once
                # the short side hits its 64px floor, so a very-high-aspect
                # image can stay over the cap even after bytes shrank.  If the
                # output is still over the cap, retrying would re-400 on
                # dimensions; treat it as unshrinkable.  (Skip when dims can't
                # be decoded — preserves historical byte-only behaviour.)
                new_dims = _decode_pixels(resized)
                if new_dims is not None and max(new_dims) > max_dimension:
                    return None, True
                return resized, False
            # triggered_by == "dimension": the per-side cap is binding.  The
            # re-encode may have grown in bytes; accept it as long as it is now
            # within the dimension cap.  Verify the new dimensions when we can.
            new_dims = _decode_pixels(resized)
            if new_dims is not None:
                if max(new_dims) <= max_dimension:
                    return resized, False
                # Still over the per-side cap — the resize didn't satisfy it.
                return None, True
            # Couldn't verify the re-encode's dimensions (corrupt output or
            # Pillow gone mid-call).  Fall back to the historical "bytes must
            # shrink" gate so we never accept an unverifiable, byte-larger blob.
            if len(resized) >= len(url):
                return None, True
            return resized, False
        except Exception as exc:
            logger.warning("image-shrink recovery: re-encode failed — %s", exc)
            return None, triggered_by is not None

    def _source_to_data_url(source: Any) -> Optional[str]:
        if not isinstance(source, dict) or source.get("type") != "base64":
            return None
        data = source.get("data")
        if not isinstance(data, str) or not data:
            return None
        media_type = str(source.get("media_type") or "image/jpeg").strip()
        if not media_type.startswith("image/"):
            media_type = "image/jpeg"
        return f"data:{media_type};base64,{data}"

    def _write_data_url_to_source(source: dict, data_url: str) -> dict:
        """Return a NEW source dict carrying the re-encoded payload.

        Copy-on-write: content parts on the per-call ``api_messages`` list may
        be shared references into the persistent conversation history (the
        per-message copy is shallow, and cache decoration only deep-copies the
        marked messages). Mutating the existing dict would rewrite the stored
        transcript with the degraded image — so the caller replaces the part,
        never edits it in place.
        """
        header, _, data = data_url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            candidate = header[len("data:"):].split(";", 1)[0].strip()
            if candidate.startswith("image/"):
                media_type = candidate
        return {
            **source,
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }

    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        # Copy-on-write per message: never mutate part/source dicts in place —
        # they can alias the stored conversation history (see
        # _write_data_url_to_source). Build a replacement content list on the
        # first shrunken part and reassign msg["content"] (a top-level write on
        # the per-call message copy, which never reaches history).
        new_content: list | None = None
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image":
                source = part.get("source")
                url = _source_to_data_url(source)
                resized, unshrinkable = _shrink_data_url(url or "")
                if resized and isinstance(source, dict):
                    if new_content is None:
                        new_content = list(content)
                    new_content[part_idx] = {
                        **part,
                        "source": _write_data_url_to_source(source, resized),
                    }
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
                continue
            if ptype not in {"image_url", "input_image"}:
                continue
            image_value = part.get("image_url")
            # OpenAI chat.completions: {"image_url": {"url": "data:..."}}
            # OpenAI Responses: {"image_url": "data:..."}
            if isinstance(image_value, dict):
                url = image_value.get("url", "")
                resized, unshrinkable = _shrink_data_url(url)
                if resized:
                    if new_content is None:
                        new_content = list(content)
                    new_content[part_idx] = {
                        **part,
                        "image_url": {**image_value, "url": resized},
                    }
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
            elif isinstance(image_value, str):
                resized, unshrinkable = _shrink_data_url(image_value)
                if resized:
                    if new_content is None:
                        new_content = list(content)
                    new_content[part_idx] = {**part, "image_url": resized}
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
        if new_content is not None:
            msg["content"] = new_content

    if changed_count:
        logger.info(
            "image-shrink recovery: re-encoded %d image part(s) to fit under %.0f MB",
            changed_count, target_bytes / (1024 * 1024),
        )
    if unshrinkable_oversized:
        # At least one oversized image could not be shrunk under the target.
        # Retrying would re-send it and fail identically, so signal "no
        # progress" even if other parts shrank — the caller will surface the
        # original error rather than burning its single retry on a no-op.
        logger.warning(
            "image-shrink recovery: %d oversized image part(s) could not be "
            "shrunk under %.0f MB — not retrying (would re-send rejected payload)",
            unshrinkable_oversized, target_bytes / (1024 * 1024),
        )
        return False
    return changed_count > 0


__all__ = [
    "COMPACTION_STATUS",
    "COMPACTION_DONE_STATUS",
    "COMPACTION_STATUS_MARKER",
    "check_compression_model_feasibility",
    "replay_compression_warning",
    "compress_context",
    "try_shrink_image_parts_in_messages",
]
