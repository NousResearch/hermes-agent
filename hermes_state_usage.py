"""Token/usage accounting mixin for SessionDB: the coalescing background token writer,
per-model usage rows, and billing-route columns. Writer thread state lives on the instance."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import logging
import hashlib
import os
import sqlite3
import stat
import threading
import uuid
import time
import weakref
from typing import Any, Dict, List, Optional, Tuple

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

_TOKEN_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")


def _token_update_sql(delta: bool) -> str:
    """``UPDATE sessions`` for one usage report: *delta* adds to the stored counters (CLI
    per-call path), otherwise sets them (gateway cumulative path). Cost/route columns
    COALESCE-fill either way (statement text is pinned by the SQL trace harness)."""
    def add(col: str) -> str:  # "col + ?" / "COALESCE(col, 0) + ?" in delta mode, bare "?" otherwise
        return f"{col} + ?" if delta else "?"
    def add0(col: str) -> str:
        return f"COALESCE({col}, 0) + ?" if delta else "?"
    counters = "".join(f"                   {c} = {add(c)},\n" for c in _TOKEN_COUNTERS)
    estimated = "COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0)" if delta else "COALESCE(?, 0)"
    return (
        "UPDATE sessions SET\n" + counters
        + f"""                   estimated_cost_usd = {estimated},
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE {add0("actual_cost_usd")}
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = {add0("api_call_count")}
                   WHERE id = ?"""
    )


_TOKEN_UPDATE_ABSOLUTE_SQL = _token_update_sql(delta=False)
_TOKEN_UPDATE_DELTA_SQL = _token_update_sql(delta=True)

_MODEL_USAGE_UPSERT_SQL = """INSERT INTO session_model_usage (
                   session_id, model, billing_provider, billing_base_url, billing_mode,
                   task, api_call_count, input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens, reasoning_tokens,
                   estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                   first_seen, last_seen
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, model, billing_provider, billing_base_url, billing_mode, task)
               DO UPDATE SET
                   api_call_count = api_call_count + excluded.api_call_count,
                   input_tokens = input_tokens + excluded.input_tokens,
                   output_tokens = output_tokens + excluded.output_tokens,
                   cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                   cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                   reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                   estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                   actual_cost_usd = actual_cost_usd + excluded.actual_cost_usd,
                   cost_status = COALESCE(excluded.cost_status, cost_status),
                   cost_source = COALESCE(excluded.cost_source, cost_source),
                   last_seen = excluded.last_seen"""


# Kwargs forwarded verbatim from update_token_counts / record_auxiliary_usage into
# _record_model_usage (the per-route attribution row).
_MODEL_USAGE_FIELDS = frozenset((
    "model", "billing_provider", "billing_base_url", "billing_mode", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd",
    "actual_cost_usd", "cost_status", "cost_source", "api_call_count"))


def _accounting_marker_identity(db):
    """Resolve a store's marker identity without creating a marker directory."""
    from pathlib import Path
    from hermes_state_lineage import _managed_root_and_current, SessionLineageInventoryError
    try:
        st = os.stat(db.db_path, follow_symlinks=False)
        if (not stat.S_ISREG(st.st_mode) or db._db_file_identity != (st.st_dev, st.st_ino)
                or db._db_file_was_replaced()):
            raise SessionLineageInventoryError("Accounting store identity is unavailable or changed")
        if st.st_nlink != 1:
            raise SessionLineageInventoryError("Hardlinked accounting stores cannot be inventoried safely")
        managed = _managed_root_and_current(db.db_path)
        root = managed[0] if managed else Path(db.db_path).resolve().parent
        physical = hashlib.sha256(f"{st.st_dev}:{st.st_ino}".encode()).hexdigest()
        location = hashlib.sha256(str(Path(db.db_path).resolve()).encode()).hexdigest()
        return root, bool(managed), f"usage-{location}-{physical}-"
    except OSError as exc:
        raise SessionLineageInventoryError("Cannot inventory accounting markers") from exc


def _legacy_accounting_marker_present(db, prefix):
    """Read legacy managed-home markers without creating, unlocking or deleting them."""
    from hermes_state_lineage import _managed_root_and_current, _validate_lineage_parent, SessionLineageInventoryError
    managed = _managed_root_and_current(db.db_path)
    if managed is None:
        return False
    legacy_root = managed[0] / ".session-lineage-locks"
    directory = legacy_root / "accounting"
    try:
        for path in (legacy_root, directory):
            try:
                st = path.lstat()
            except FileNotFoundError:
                return False
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise SessionLineageInventoryError("Unsafe legacy accounting marker directory")
            _validate_lineage_parent(path, st)
        _, location, physical, _ = prefix.split("-", 3)
        with os.scandir(directory) as scan:
            for count, entry in enumerate(scan, 1):
                if count > 8192:
                    raise SessionLineageInventoryError("Legacy accounting marker inventory exceeds bound")
                parts = entry.name.split("-", 3)
                if len(parts) == 4 and parts[0] == "usage" and (parts[1] == location or parts[2] == physical):
                    return True
        return False
    except OSError as exc:
        raise SessionLineageInventoryError("Cannot inspect legacy accounting marker inventory") from exc


def _require_legacy_accounting_clear(db):
    from hermes_state_lineage import _managed_root_and_current, SessionLineageInventoryError
    if _managed_root_and_current(db.db_path) is None:
        return
    _, _, prefix = _accounting_marker_identity(db)
    if _legacy_accounting_marker_present(db, prefix):
        raise SessionLineageInventoryError("Unresolved legacy accounting marker; inspect before accepting usage")


class _TokenBatchInterrupted(BaseException):
    """A batch stopped with a known unprocessed suffix and possibly unknown current item."""
    def __init__(self, cause, uncertain, pending):
        super().__init__(str(cause))
        self.cause, self.uncertain, self.pending = cause, uncertain, pending


class _TokenBatchWriteReceipt:
    """Per-delta transaction evidence; successful or unknown commits are sticky."""
    def __init__(self):
        self._committed = False
        self._unknown = False
        self._retryable_error = None

    def transaction_started(self):
        self._retryable_error = None

    def transaction_not_started(self, error, *, transaction_active):
        """Bind retry proof to this error only when BEGIN never opened a transaction."""
        self._retryable_error = None
        if (self._committed or self._unknown or transaction_active
                or not isinstance(error, sqlite3.Error)):
            self._unknown = True
            return
        self._retryable_error = error

    def transaction_committed(self):
        self._committed = True
        self._retryable_error = None

    def transaction_rolled_back(
        self, *, commit_attempted, transaction_active, rollback_succeeded, commit_rollback_proven,
        error=None,
    ):
        self._retryable_error = None
        if (not rollback_succeeded or not transaction_active
                or (commit_attempted and not commit_rollback_proven)):
            self._unknown = True
        elif not self._committed and not self._unknown:
            # Settlement comes from the active transaction's verified rollback,
            # not from the exception class raised by its callback.
            self._retryable_error = error

    def retryable_lock_timeout(self, error):
        if not self._committed and not self._unknown:
            self._retryable_error = error

    def has_uncertain_effect(self):
        return self._committed or self._unknown

    def proves_unapplied(self, error):
        return (not self._committed and not self._unknown
                and self._retryable_error is error)


class SessionUsageMixin:
    """Coalesced token writer, per-model usage rows, billing route."""

    def update_session_billing_route(
        self, session_id: str, *, provider: str, base_url: str, billing_mode: Optional[str] = None,
    ) -> None:
        """Unconditionally set the billing route (``update_token_counts`` only COALESCE-fills
        NULLs) so the dashboard reflects the latest /model switch.

        Route writers never touch the stored prompt; ``_stored_prompt_matches_runtime`` decides staleness.

        See #48173, #48248.
        """
        # Barrier against queued token deltas — see update_session_model.
        if not self.flush_token_counts():
            raise TimeoutError("timed out flushing accounting before billing route update")

        def _do(conn):
            conn.execute("""UPDATE sessions SET
                   billing_provider = ?,
                   billing_base_url = ?,
                   billing_mode = COALESCE(?, billing_mode)
                   WHERE id = ?""", (provider, base_url, billing_mode, session_id))
        from hermes_state_lineage import guarded_session_write
        guarded_session_write(self, session_id, _do)

    def _token_marker_prefix(self):
        """Locate this physical store's cross-process pending-accounting markers."""
        from hermes_state_lineage import _lineage_lock_directory, SessionLineageInventoryError
        try:
            root, managed, prefix = _accounting_marker_identity(self)
            if managed:
                lock_root = _lineage_lock_directory(root, create=True)
            else:
                lock_root = root / ".session-lineage-locks"
                lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            lst = lock_root.lstat()
            if not stat.S_ISDIR(lst.st_mode) or stat.S_ISLNK(lst.st_mode):
                raise SessionLineageInventoryError("Unsafe lineage lock directory")
            directory = lock_root / "accounting"
            directory.mkdir(mode=0o700, exist_ok=True)
            dst = directory.lstat()
            if not stat.S_ISDIR(dst.st_mode) or stat.S_ISLNK(dst.st_mode):
                raise SessionLineageInventoryError("Unsafe accounting marker directory")
            return directory, prefix
        except SessionLineageInventoryError:
            raise
        except OSError as exc:
            raise SessionLineageInventoryError("Cannot inventory accounting markers") from exc

    def _hold_token_marker(self):
        """Hold a raw advisory marker before accepting any volatile delta."""
        from hermes_state_lineage import _advisory_lock_file, SessionLineageInventoryError
        directory, prefix = self._token_marker_prefix()
        if _legacy_accounting_marker_present(self, prefix):
            raise SessionLineageInventoryError("Unresolved legacy accounting marker; inspect before accepting usage")
        if self._token_pending_marker is not None:
            if self._token_pending_marker_path is None:
                raise SessionLineageInventoryError("Accounting marker release is unresolved")
            if (self._token_pending_marker_path.parent != directory
                    or not self._token_pending_marker_path.name.startswith(prefix)):
                raise SessionLineageInventoryError("Accounting marker identity changed")
            try:
                held = self._token_pending_marker_path.lstat()
            except OSError as exc:
                raise SessionLineageInventoryError("Held accounting marker is unavailable") from exc
            if (not stat.S_ISREG(held.st_mode) or (held.st_dev, held.st_ino) != self._token_pending_marker_identity):
                raise SessionLineageInventoryError("Held accounting marker was replaced")
            return
        if self._token_pending_marker_path is None:
            self._token_pending_marker_path = directory / f"{prefix}{os.getpid()}-{uuid.uuid4().hex}.lock"
        elif (self._token_pending_marker_path.parent != directory
              or not self._token_pending_marker_path.name.startswith(prefix)):
            raise SessionLineageInventoryError("Accounting marker identity changed")
        directory_stat = directory.lstat()
        marker = _advisory_lock_file(self._token_pending_marker_path, 0)
        marker_fd = marker.__enter__()
        try:
            if (self._token_marker_prefix() != (directory, prefix)
                    or not os.path.samestat(directory_stat, directory.lstat())):
                raise SessionLineageInventoryError("Accounting store identity changed while accepting")
        except Exception:
            marker.__exit__(None, None, None)
            raise
        self._token_pending_marker = marker
        held = os.fstat(marker_fd)
        self._token_pending_marker_identity = (held.st_dev, held.st_ino)

    def _release_idle_token_marker(self):
        """Called under the queue condition only after a known-idle transition."""
        if (self._token_pending_marker is not None and not self._token_queue
                and not self._token_writer_busy and not self._token_writer_uncertain):
            if self._token_pending_marker_path is None:
                raise RuntimeError("Accounting marker release is unresolved")
            # Only the owning handle removes its own known-idle marker while still locked.
            try:
                held = self._token_pending_marker_path.lstat()
            except FileNotFoundError:
                held = None  # no pending work remains; releasing the unlinked lock is safe
            if held is not None:
                if (not stat.S_ISREG(held.st_mode)
                        or (held.st_dev, held.st_ino) != self._token_pending_marker_identity):
                    raise RuntimeError("Idle accounting marker was replaced")
                self._token_pending_marker_path.unlink()
            self._token_pending_marker_path = None
            self._token_pending_marker.__exit__(None, None, None)
            self._token_pending_marker = None
            self._token_pending_marker_identity = None

    def queue_token_counts(self, session_id: str, **kwargs) -> None:
        """Enqueue a token/cost delta for the background writer (same kwargs as
        :meth:`update_token_counts`). After close() stopped the writer, falls back to the
        synchronous path and may raise."""
        from hermes_state_lineage import session_lineage_guard
        with session_lineage_guard(self.db_path, session_id, timeout_s=self._TRANSCRIPT_WRITE_PATIENCE_S):
            with self._token_queue_cond:
                thread = self._token_writer_thread
                writer_alive = thread is not None and thread.is_alive()
                writer_stopped = self._token_writer_stop and not writer_alive
                if self._token_writer_uncertain:
                    # Do not let later work overtake an unknown-effect item.
                    self._hold_token_marker()
                    self._token_queue.append((session_id, kwargs))
                    writer_stopped = False
                elif writer_stopped and (self._token_queue or self._token_writer_busy):
                    # A failed close left older deltas pending; never overtake them.
                    self._hold_token_marker()
                    self._token_queue.append((session_id, kwargs))
                    writer_stopped = False
                elif not writer_stopped:
                    self._hold_token_marker()
                    self._token_queue.append((session_id, kwargs))
                    if not writer_alive:
                        # Daemon so exit never hangs on accounting; the atexit hook drains
                        # leftovers. ``not is_alive()`` respawns a writer that died unexpectedly.
                        thread = threading.Thread(
                            target=self._token_writer_loop, name="session-db-token-writer", daemon=True)
                        self._token_writer_thread = thread
                        thread.start()
                        if self._token_atexit_hook is None:
                            self_ref = weakref.ref(self)

                            def _drain_at_exit() -> None:
                                db = self_ref()
                                if db is not None:
                                    db._drain_token_queue_at_exit()

                            self._token_atexit_hook = _drain_at_exit
                            atexit.register(_drain_at_exit)
                    self._token_queue_cond.notify_all()
        if writer_stopped:
            # close() ran: enqueueing would drop the delta silently, so apply inline.
            self.update_token_counts(session_id, **kwargs)

    def has_pending_token_counts(self) -> bool:
        """Nonblocking queue/in-flight check for callers already holding a lineage guard."""
        with self._token_queue_cond:
            if (self._token_queue or self._token_writer_busy or self._token_writer_uncertain
                    or self._token_pending_marker is not None):
                return True
        from hermes_state_lineage import _advisory_lock_file
        try:
            directory, prefix = self._token_marker_prefix()
            if _legacy_accounting_marker_present(self, prefix):
                return True
            directory_stat = directory.lstat()
            _, location, physical, _ = prefix.split("-", 3)
            def inventory():
                names = []
                with os.scandir(directory) as scan:
                    for count, entry in enumerate(scan, 1):
                        if count > 8192:
                            raise RuntimeError("Accounting marker inventory exceeds bound")
                        parts = entry.name.split("-", 3)
                        if (len(parts) == 4 and parts[0] == "usage"
                                and (parts[1] == location or parts[2] == physical)):
                            names.append(entry.name)
                return sorted(names)
            names = inventory()
            for name in names:
                # Contention, unsafe files, and all IO failures fail closed.
                with _advisory_lock_file(directory / name, 0, create=False):
                    pass
            if (inventory() != names or self._token_marker_prefix() != (directory, prefix)
                    or not os.path.samestat(directory_stat, directory.lstat())):
                return True
            return False
        except Exception:
            logger.warning("Cannot verify pending accounting markers", exc_info=True)
            return True

    def _claim_token_batch_locked(self):
        """Claim the current queue while its condition is held."""
        batch = list(self._token_queue)
        self._token_writer_claim = batch
        self._token_writer_claim_owner = threading.get_ident()
        self._token_writer_busy = True
        self._token_queue.clear()
        return batch

    def _quarantine_abandoned_claim(self):
        """A signal at the claim boundary has no reliable per-item progress receipt."""
        with self._token_queue_cond:
            if self._token_writer_claim_owner != threading.get_ident():
                return
            batch = self._token_writer_claim or []
            if self._token_writer_busy:
                # If interrupted before clear(), remove the same objects from the live queue;
                # otherwise new arrivals remain behind the uncertain claim.
                if len(self._token_queue) >= len(batch) and all(
                    self._token_queue[i] is item for i, item in enumerate(batch)
                ):
                    for _ in batch:
                        self._token_queue.popleft()
                self._token_writer_uncertain.extend(batch)
                self._token_writer_busy = False
            self._token_writer_claim = self._token_writer_claim_owner = None
            self._token_queue_cond.notify_all()

    def _apply_claimed_batch(self, batch) -> bool:
        """Retain known suffixes and never automatically replay an unknown-effect item."""
        pending, uncertain, interrupt = [], [], None
        try:
            pending = self._apply_token_batch(batch) or []
        except _TokenBatchInterrupted as stopped:
            pending, uncertain = stopped.pending, stopped.uncertain
            if isinstance(stopped.cause, Exception):
                logger.warning("async token accounting: item outcome unknown; retaining work", exc_info=True)
            else:
                interrupt = stopped.cause
        except BaseException as exc:
            # The helper escaped without a progress receipt: no prefix is safe to replay.
            uncertain = batch
            if isinstance(exc, Exception):
                logger.warning("async token accounting: batch outcome unknown; retaining work", exc_info=True)
            else:
                interrupt = exc
        finally:
            with self._token_queue_cond:
                try:
                    self._token_writer_uncertain.extend(uncertain)
                    self._token_queue.extendleft(reversed(pending))
                finally:
                    self._token_writer_busy = False
                    self._token_writer_claim = self._token_writer_claim_owner = None
                    self._token_queue_cond.notify_all()
        if interrupt is not None:
            raise interrupt.with_traceback(interrupt.__traceback__)
        with self._token_queue_cond:
            try:
                self._release_idle_token_marker()
            except Exception:
                logger.warning("Cannot release idle accounting marker", exc_info=True)
                return False
        return not pending and not uncertain

    def flush_for_read(self) -> bool:
        """Read-side token barrier. On a thread running an asyncio loop it never waits: the writer can sit
        behind another connection's write lock, and a gateway handler reading a session there froze every
        socket for up to the 5 s flush budget (#141966). That read sees totals stale by the queued deltas,
        which the next read picks up; write barriers keep calling ``flush_token_counts`` and still wait."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self.flush_token_counts()
        return self.flush_token_counts(timeout=0.0)

    def flush_token_counts(self, timeout: float = 5.0) -> bool:
        """Block until every queued token delta has been applied. False on timeout (callers
        then read totals stale by the queued deltas). Process-control interrupts remain visible."""
        # Lock-free fast path: reads queue-then-busy (see ordering notes below).
        if (not self._token_queue and not self._token_writer_busy and not self._token_writer_uncertain
                and self._token_pending_marker is None):
            return True
        # Polling must not claim accepted work or enter a blocking write path.
        if timeout <= 0:
            return False
        deadline = time.monotonic() + timeout
        try:
            while True:
                if not self._token_queue_cond.acquire(timeout=max(0.0, deadline - time.monotonic())):
                    return False
                try:
                    batch = self._wait_or_claim_token_batch(deadline)
                finally:
                    self._token_queue_cond.release()
                if batch is None:
                    return False
                if not batch:
                    return True
                previous_deadline = getattr(self._token_batch_write_context, "deadline", None)
                self._token_batch_write_context.deadline = deadline
                try:
                    if not self._apply_claimed_batch(batch):
                        return False
                finally:
                    self._token_batch_write_context.deadline = previous_deadline
                # A respawned writer can claim work accepted during our drain. Rejoin
                # the same deadline-bound barrier rather than snapshotting its busy flag.

        finally:
            if self._token_writer_claim_owner == threading.get_ident():
                self._quarantine_abandoned_claim()

    def _wait_or_claim_token_batch(self, deadline):
        """Under the queue condition, return a claim, [] for settled, or None for failure."""
        while True:
            if self._token_writer_uncertain:
                return None
            if not self._token_queue and not self._token_writer_busy:
                try:
                    self._release_idle_token_marker()
                except Exception:
                    logger.warning("Cannot release idle accounting marker", exc_info=True)
                    return None
                return []
            # A live writer is authoritative even when stop-flagged: draining here would
            # race its in-flight batch and reorder deltas (breaking last-non-None-wins /
            # first-accounted-route / COALESCE-backfill fields). Only a dead writer lets
            # the caller take leftovers; a claimed busy means "wait".
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            thread = self._token_writer_thread
            if (thread is None or not thread.is_alive()) and not self._token_writer_busy:
                return self._claim_token_batch_locked()
            self._token_queue_cond.wait(remaining)

    def _token_writer_loop(self) -> None:
        while True:
            try:
                with self._token_queue_cond:
                    # A respawn may overlap a caller draining the previous writer's
                    # leftovers. Keep one claim authoritative until its receipt settles.
                    while self._token_writer_busy:
                        self._token_queue_cond.wait()
                    if self._token_writer_uncertain:
                        self._token_writer_thread = None
                        self._token_queue_cond.notify_all()
                        return
                    idle_deadline = time.monotonic() + self._TOKEN_WRITER_IDLE_SECONDS
                    while not self._token_queue and not self._token_writer_stop:
                        remaining = idle_deadline - time.monotonic()
                        if remaining <= 0:
                            # Retire under the lock queue_token_counts() spawns under, so no
                            # delta strands behind an exiting worker.
                            self._token_writer_thread = None
                            return
                        self._token_queue_cond.wait(remaining)
                    if not self._token_queue:
                        self._token_writer_thread = None
                        return  # stop requested and fully drained
                    # busy BEFORE clearing the queue: flush's lock-free fast path must never see
                    # "empty and idle" while a popped batch is unapplied.
                    batch = self._claim_token_batch_locked()
                if not self._apply_claimed_batch(batch):
                    # Leave rejected work queued without spinning; a later flush may retry.
                    with self._token_queue_cond:
                        self._token_writer_thread = None
                        self._token_queue_cond.notify_all()
                    return

            finally:
                self._quarantine_abandoned_claim()

    def _apply_token_batch(self, batch: list[tuple[str, dict[str, Any]]]) -> list:
        """Apply in order; retain the failed delta and every unprocessed successor."""
        try:
            coalesced = self._coalesce_token_deltas(batch)
        except Exception as exc:
            # Coalescing must never kill the writer; the merge is only an optimization.
            logger.warning("async token accounting: coalesce failed, applying raw batch: %s", exc)
            coalesced = batch
        except BaseException as exc:
            raise _TokenBatchInterrupted(exc, [], batch) from exc
        from hermes_state_lineage import SessionLineageError
        for index, (session_id, kwargs) in enumerate(coalesced):
            deadline = getattr(self._token_batch_write_context, "deadline", None)
            if deadline is not None and time.monotonic() >= deadline:
                return coalesced[index:]
            receipt = _TokenBatchWriteReceipt()
            previous = getattr(self._token_batch_write_context, "receipt", None)
            self._token_batch_write_context.receipt = receipt
            try:
                try:
                    self.update_token_counts(session_id, **kwargs)
                except SessionLineageError as exc:
                    if receipt.has_uncertain_effect():
                        raise _TokenBatchInterrupted(
                            exc, [coalesced[index]], coalesced[index + 1:]
                        ) from exc
                    logger.warning("async token accounting: apply refused; retaining queued deltas (session=%s): %s",
                                   session_id, exc)
                    return coalesced[index:]
                except BaseException as exc:
                    if receipt.proves_unapplied(exc):
                        if not isinstance(exc, Exception):
                            raise _TokenBatchInterrupted(exc, [], coalesced[index:]) from exc
                        logger.warning(
                            "async token accounting: write proven unapplied; "
                            "retaining the unapplied delta and suffix (session=%s)", session_id)
                        return coalesced[index:]
                    # A callback interruption or ambiguous commit may have persisted this item.
                    raise _TokenBatchInterrupted(
                        exc, [coalesced[index]], coalesced[index + 1:]
                    ) from exc
            finally:
                self._token_batch_write_context.receipt = previous
        return []

    def _coalesce_token_deltas(self, batch: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
        """Merge adjacent incremental deltas with an identical route, so ordering across
        sessions and /model switches is preserved exactly. absolute=True never merges."""
        groups: list[tuple[Optional[tuple], str, dict[str, Any]]] = []
        for session_id, kwargs in batch:
            key = None
            if not kwargs.get("absolute"):
                key = (session_id, *(kwargs.get(f) for f in self._TOKEN_DELTA_ROUTE_FIELDS))
            if groups and key is not None and groups[-1][0] == key:
                merged = groups[-1][2]
                for f in self._TOKEN_DELTA_SUM_FIELDS:
                    merged[f] = merged.get(f, 0) + kwargs.get(f, 0)
                for f in self._TOKEN_DELTA_COST_FIELDS:
                    value = kwargs.get(f)
                    if value is not None:
                        # All-None runs stay None so COALESCE keeps the stored value.
                        merged[f] = (merged.get(f) or 0.0) + value
            else:
                groups.append((key, session_id, dict(kwargs)))
        return [(sid, kw) for _, sid, kw in groups]

    def _stop_token_writer(self, join_timeout: float = 10.0) -> bool:
        """Stop the writer thread and drain remaining deltas. Never raises."""
        with self._token_queue_cond:
            self._token_writer_stop = True
            self._token_queue_cond.notify_all()
            thread = self._token_writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                # Writer stuck mid-apply: leave deltas unapplied rather than race it.
                logger.warning(
                    "async token accounting: writer did not stop within %.0fs; "
                    "%d queued delta(s) not persisted", join_timeout, len(self._token_queue))
                return False
        # Writer gone: apply leftovers synchronously under the same busy protocol. Wait out
        # a flush caller-drain that already claimed busy — close() nulls the connection
        # right after this returns and must not yank it mid-batch.
        try:
            with self._token_queue_cond:
                if self._token_writer_uncertain:
                    return False
                deadline = time.monotonic() + join_timeout
                while self._token_writer_busy:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        logger.warning(
                            "async token accounting: concurrent drain did not "
                            "finish within %.0fs; %d queued delta(s) not persisted",
                            join_timeout, len(self._token_queue))
                        return False
                    self._token_queue_cond.wait(remaining)
                    if self._token_writer_uncertain:
                        return False
                # busy BEFORE clearing the queue (same ordering as the writer loop).
                batch = self._claim_token_batch_locked() if self._token_queue else []
            if batch:
                if not self._apply_claimed_batch(batch):
                    return False
                with self._token_queue_cond:
                    return not (self._token_queue or self._token_writer_busy
                                or self._token_writer_uncertain or self._token_pending_marker is not None)
            with self._token_queue_cond:
                try:
                    self._release_idle_token_marker()
                except Exception:
                    logger.warning("Cannot release idle accounting marker", exc_info=True)
                    return False
            return True

        finally:
            self._quarantine_abandoned_claim()

    def _drain_token_queue_at_exit(self) -> None:
        with contextlib.suppress(Exception):  # never fatal at interpreter shutdown
            self._stop_token_writer()

    def update_token_counts(
        self, session_id: str, input_tokens: int=0, output_tokens: int=0, model: str | None=None, cache_read_tokens: int=0,
        cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None,
        actual_cost_usd: Optional[float]=None, cost_status: Optional[str]=None, cost_source: Optional[str]=None,
        pricing_version: Optional[str]=None, billing_provider: Optional[str]=None, billing_base_url: Optional[str]=None,
        billing_mode: Optional[str]=None, api_call_count: int=0, absolute: bool=False,
        source: Optional[str]=None, task: str = "",
    ) -> None:
        """Update totals and route attribution, ensuring the legacy missing-row backfill.
        *absolute*=False increments (per-API-call deltas, CLI path); *absolute*=True sets directly
        (gateway path, where the cached agent holds cumulative totals). ``source`` is the session's
        real surface for the row-existence guard; callers that don't know it leave the placeholder.
        ``task`` names a per-turn route (``voice_chat``): its calls count in the session totals and
        land in ``session_model_usage`` under that task, but never become the session's recorded route."""
        values = dict(locals())
        values.pop('self')
        values.pop('session_id')
        values.pop('source')
        # Ensure the row exists: under concurrent load create_session() may have failed on
        # locking, and the UPDATE would silently affect 0 rows. When this guard is the first
        # writer it must carry the agent's real source: the turn lease treats an existing row as
        # proof the create already happened, so the creator never returns to repair an anonymous
        # ``unknown`` placeholder and the session stays a phantom for life (#111999).
        from hermes_state_lineage import session_lineage_guard, SessionLineageInventoryError, _lineage_inventory_deadline
        from hermes_state_worker_lifecycle import insert_session_row_in_transaction
        deadline = getattr(self._token_batch_write_context, "deadline", None)
        def remaining():
            if deadline is None:
                return self._TRANSCRIPT_WRITE_PATIENCE_S
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise SessionLineageInventoryError("Accounting flush deadline expired before write")
            return min(self._TRANSCRIPT_WRITE_PATIENCE_S, budget)
        def write(conn):
            insert_session_row_in_transaction(self, conn, session_id, source or "unknown", model=model)
            self._update_token_counts_in_transaction(conn, session_id, **values)
        with _lineage_inventory_deadline(deadline):
            with session_lineage_guard(self.db_path, session_id, timeout_s=remaining()):
                # Claimed batches were already accepted behind their marker. Direct
                # synchronous writes are new admissions and must respect legacy work.
                if getattr(self._token_batch_write_context, "receipt", None) is None:
                    _require_legacy_accounting_clear(self)
                self._execute_write(write, patience_s=remaining())

    def _update_token_counts_in_transaction(
        self, conn, session_id: str, input_tokens: int=0, output_tokens: int=0, model: str | None=None, cache_read_tokens: int=0,
        cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None,
        actual_cost_usd: Optional[float]=None, cost_status: Optional[str]=None, cost_source: Optional[str]=None,
        pricing_version: Optional[str]=None, billing_provider: Optional[str]=None, billing_base_url: Optional[str]=None,
        billing_mode: Optional[str]=None, api_call_count: int=0, absolute: bool=False, task: str="",
    ) -> None:
        usage = {k: v for k, v in locals().items() if k in _MODEL_USAGE_FIELDS}
        sql = _TOKEN_UPDATE_ABSOLUTE_SQL if absolute else _TOKEN_UPDATE_DELTA_SQL
        has_usage = bool(input_tokens or output_tokens or cache_read_tokens or cache_write_tokens or reasoning_tokens
                         or api_call_count or estimated_cost_usd)
        has_accounted_usage = bool(has_usage or actual_cost_usd)
        params = (
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,
            estimated_cost_usd, actual_cost_usd, actual_cost_usd, cost_status, cost_source, pricing_version,
            billing_provider if has_accounted_usage else None,
            billing_base_url if has_accounted_usage else None,
            billing_mode if has_accounted_usage else None, model if has_accounted_usage else None,
            api_call_count, session_id)
        # Per-model attribution: the sessions row keeps one (model, provider) pair, so a
        # mid-session /model switch would attribute every token to the initial model. Only
        # the incremental path records here — absolute cumulative updates cannot be split
        # back into routes; Insights reconciles the residual instead.
        # ``update_token_counts`` is the single chokepoint every per-API-call delta flows through (CLI,
        # gateway, cron, delegated runs — see conversation_loop / codex_runtime), and each call carries the
        # model/provider *active at the time of that call*. Recording the per-call delta into
        # session_model_usage keyed by the live model preserves an accurate per-model breakdown regardless
        # of how many times the user switches. See #51607.
        record_model_usage = (not absolute) and has_usage

        row = conn.execute(
            "SELECT model, billing_provider, api_call_count FROM sessions WHERE id = ?", (session_id,),
        ).fetchone()
        existing = dict(row) if row is not None else {}
        # create_session records the requested route before any API call. If that fails
        # and fallback succeeds, the first accounted usage is the authoritative route;
        # after that keep the row as is (one row cannot represent mixed usage).
        first_accounted_route = (
            not task and int(existing.get("api_call_count") or 0) == 0 and has_accounted_usage and bool(model)
            and bool(billing_provider)
            and (existing.get("model") != model or existing.get("billing_provider") != billing_provider)
        )
        if first_accounted_route:
            conn.execute("""UPDATE sessions
                   SET model = ?, billing_provider = ?,
                   billing_base_url = ?, billing_mode = ?
                   WHERE id = ?""", (model, billing_provider, billing_base_url, billing_mode, session_id))
        conn.execute(sql, params)
        if record_model_usage:
            self._record_model_usage(conn, session_id, task=task, **usage)

    def _record_model_usage(
        self, conn, session_id: str, *, model: Optional[str]=None, billing_provider: Optional[str]=None,
        billing_base_url: Optional[str]=None, billing_mode: Optional[str]=None, input_tokens: int=0,
        output_tokens: int=0, cache_read_tokens: int=0, cache_write_tokens: int=0, reasoning_tokens: int=0,
        estimated_cost_usd: Optional[float]=None, actual_cost_usd: Optional[float]=None,
        cost_status: Optional[str]=None, cost_source: Optional[str]=None, api_call_count: int=0, task: str="",
    ) -> None:
        """Accumulate a per-API-call usage delta into session_model_usage, inside the caller's
        write txn after the ``sessions`` UPDATE. A missing model/provider falls back to
        the session row — except for aux rows (``task`` set), which must NOT inherit the
        main-loop route (vision on gemini while the main loop runs anthropic): missing
        info stays 'unknown'/empty.

        ``task`` distinguishes what kind of work consumed the tokens: ``''`` (empty) is the main agent loop;
        auxiliary calls record their task name (``vision``, ``compression``, ``title_generation``, ...) via
        :meth:`record_auxiliary_usage` (issue #23270).
        """
        row = conn.execute(
            "SELECT model, billing_provider, billing_base_url, billing_mode FROM sessions WHERE id = ?", (session_id,),
        ).fetchone()
        sess = dict(row) if (row is not None and not task) else {}
        counts = [v or 0 for v in (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)]
        now = time.time()
        conn.execute(_MODEL_USAGE_UPSERT_SQL, (
            session_id, model or sess.get("model") or "unknown",
            billing_provider or sess.get("billing_provider") or "",
            billing_base_url or sess.get("billing_base_url") or "",
            billing_mode or sess.get("billing_mode") or "", task or "", api_call_count or 0, *counts,
            float(estimated_cost_usd or 0.0), float(actual_cost_usd or 0.0), cost_status, cost_source, now, now))

    def record_auxiliary_usage(
        self, session_id: str, task: str, *, model: Optional[str]=None, billing_provider: Optional[str]=None,
        billing_base_url: Optional[str]=None, input_tokens: int=0, output_tokens: int=0, cache_read_tokens: int=0,
        cache_write_tokens: int=0, reasoning_tokens: int=0, estimated_cost_usd: Optional[float]=None,
        api_call_count: int=1,
    ) -> None:
        """Record an auxiliary LLM call's usage (vision, compression, title generation, ...)
        as a per-(model, provider, task) delta in ``session_model_usage`` WITHOUT touching
        the ``sessions`` summary row (the gateway overwrites those counters with absolute
        main-loop totals). ``api_call_count`` may aggregate N calls. Best-effort.

        See #23270.
        Background-review forks record an aggregate of N fork API calls in one write with
        ``task='background_review'`` (issue #87250).
        """
        usage = {k: v for k, v in locals().items() if k in _MODEL_USAGE_FIELDS}
        if not session_id or not task:
            return
        usage["api_call_count"] = 1 if api_call_count is None else int(api_call_count)
        # FK to sessions.id. A missing row is inserted as the ``unknown`` placeholder, which
        # the creator upsert later replaces. An existing row keeps source and created_source,
        # including a NULL surface update_token_counts can still fill; the UPDATE below only
        # fills a NULL profile_name, without writing ``unknown`` anywhere.
        from hermes_state_worker_lifecycle import insert_session_row_in_transaction

        def _write(conn):
            if conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
                insert_session_row_in_transaction(self, conn, session_id, "unknown")
            elif own_profile := self._own_profile_name():
                conn.execute(
                    "UPDATE sessions SET profile_name=? WHERE id=? AND profile_name IS NULL",
                    (own_profile, session_id))
            self._record_model_usage(conn, session_id, task=task, **usage)
        from hermes_state_lineage import session_lineage_guard
        with session_lineage_guard(self.db_path, session_id, timeout_s=self._TRANSCRIPT_WRITE_PATIENCE_S):
            _require_legacy_accounting_clear(self)
            self._execute_write(_write, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def auxiliary_usage_by_task(self, session_id: str) -> dict[str, dict[str, float]]:
        """Per-task auxiliary usage (``task != ''``: vision, compression, title_generation, ...) summed
        over the session's compression lineage. Aux calls bill to the id the turn STARTED with while
        compression mints child ids mid-turn, so a single-id read misses rows (#112848)."""
        if not session_id:
            return {}
        chain = self._session_lineage_root_to_tip(session_id)
        # An explicit ``/branch`` copy owns its spend: cut the walk at the nearest branch node so a
        # resumed branch never absorbs aux rows billed to its source (hermes_state_messages does the same).
        for i in range(len(chain) - 1, -1, -1):
            if self._is_explicit_branch_session(chain[i]):
                chain = chain[i:]
                break
        rows = self._read_all(
            f"""SELECT task,
                       COALESCE(SUM(api_call_count), 0) AS api_calls,
                       COALESCE(SUM(input_tokens), 0) AS input_tokens,
                       COALESCE(SUM(output_tokens), 0) AS output_tokens,
                       COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                       COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                       COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                       COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                  FROM session_model_usage
                 WHERE session_id IN ({','.join('?' * len(chain))}) AND task != ''
                 GROUP BY task""",
            chain,
        )
        return {row["task"]: {k: row[k] for k in row.keys() if k != "task"} for row in rows}

    def usage_totals(self, *, min_message_count: int = 1, include_archived: bool = False) -> dict[str, float]:
        """Tokens and spend across the whole store (one scan), so the sidebar total does not
        shrink with paging. Spend prefers the billed figure over the estimate."""
        where = ["parent_session_id IS NULL", "message_count >= ?"]
        params: list[Any] = [min_message_count]
        if not include_archived:
            where.append("COALESCE(archived, 0) = 0")
        row = self._read_one(f"""
            SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0),
                   COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd, 0)), 0)
              FROM sessions
             WHERE {' AND '.join(where)}
            """, params)
        return {"tokens": int(row[0] or 0), "cost_usd": float(row[1] or 0.0)}
