"""Per-attempt SQLite settlement evidence for queued accounting."""

import logging
import sqlite3

logger = logging.getLogger("hermes_state")


class AccountingWriteReceipt:
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


class WriteAttempt:
    """Track one write without interpreting an unobserved settlement as rollback."""

    def __init__(self, receipt):
        self.receipt = receipt
        self.fn_started = False
        self.commit_attempted = False
        self.commit_completed = False
        self.rollback_succeeded = False
        self.active_before_rollback = False

    def mark_mutex_refusal(self, error):
        """Only a refusal before the callback proves this attempt unapplied."""
        if self.receipt is not None and not self.fn_started:
            self.receipt.retryable_lock_timeout(error)

    def _commit_rollback_proven(self, error):
        code = getattr(error, "sqlite_errorcode", None)
        return (self.commit_attempted and self.active_before_rollback
                and self.rollback_succeeded and isinstance(code, int)
                and (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED))

    def retry_safe(self, error):
        # Only the accounting caller interprets settlement receipts. Ordinary storage
        # retains the existing retry policy until its whole call graph is converted.
        if self.receipt is None:
            return True
        if self.commit_completed:
            return False
        if not self.fn_started:
            return self.receipt.proves_unapplied(error)
        return (self.rollback_succeeded and self.active_before_rollback
                and (not self.commit_attempted or self._commit_rollback_proven(error)))

    def _rollback(self, conn, error):
        try:
            self.active_before_rollback = bool(conn.in_transaction)
        except Exception:
            logger.warning("Cannot inspect transaction before rollback; settlement is unknown", exc_info=True)
        try:
            conn.rollback()
            self.rollback_succeeded = not conn.in_transaction
        except Exception:
            logger.warning("Cannot verify transaction rollback; settlement is unknown", exc_info=True)
        if self.receipt is not None:
            self.receipt.transaction_rolled_back(
                error=error,
                commit_attempted=self.commit_attempted,
                transaction_active=self.active_before_rollback,
                rollback_succeeded=self.rollback_succeeded,
                commit_rollback_proven=self._commit_rollback_proven(error))

    def execute(self, conn, fn):
        try:
            conn.execute("BEGIN IMMEDIATE")
        except BaseException as error:
            if self.receipt is not None:
                try:
                    transaction_active = bool(conn.in_transaction)
                except Exception:
                    logger.warning("Cannot inspect transaction after failed BEGIN; settlement is unknown", exc_info=True)
                    transaction_active = True
                self.receipt.transaction_not_started(error, transaction_active=transaction_active)
            raise
        try:
            self.fn_started = True
            if self.receipt is not None:
                self.receipt.transaction_started()
            result = fn(conn)
            self.commit_attempted = True
            conn.commit()
            self.commit_completed = True
            if self.receipt is not None:
                self.receipt.transaction_committed()
            return result
        except BaseException as error:
            self._rollback(conn, error)
            raise
