"""Transaction attempt and settlement evidence for SessionDB writes."""

import logging
import sqlite3

logger = logging.getLogger("hermes_state")


class WriteAttempt:
    """Track one write without interpreting an unobserved settlement as rollback."""

    def __init__(self, receipt):
        self.receipt = receipt
        self.fn_started = False
        self.commit_attempted = False
        self.commit_completed = False
        self.rollback_succeeded = False
        self.active_before_rollback = False

    def _commit_rollback_proven(self, error):
        code = getattr(error, "sqlite_errorcode", None)
        return (self.commit_attempted and self.active_before_rollback
                and self.rollback_succeeded and isinstance(code, int)
                and (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED))

    def retry_safe(self, error):
        return (not self.commit_completed
                and (not self.fn_started or (
                    self.rollback_succeeded and self.active_before_rollback
                    and (not self.commit_attempted or self._commit_rollback_proven(error)))))

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
                commit_attempted=self.commit_attempted,
                transaction_active=self.active_before_rollback,
                rollback_succeeded=self.rollback_succeeded,
                commit_rollback_proven=self._commit_rollback_proven(error))

    def execute(self, conn, fn):
        conn.execute("BEGIN IMMEDIATE")
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


def retry_locked_write(db, error, retry_safe, deadline, patience_s, receipt, log_holders):
    """Retry only when settlement is known; retain the existing timeout diagnostic."""
    if not retry_safe:
        raise error
    if db._sleep_before_write_retry(deadline, patience_s):
        return True
    log_holders(db.db_path, patience_s)
    lock_error = sqlite3.OperationalError(
        f"database is locked (another Hermes process held the "
        f"state.db write lock for over {patience_s:.0f}s — "
        "likely a long maintenance operation such as VACUUM, "
        "a large WAL checkpoint, or an older pre-update "
        "process; the database itself is healthy)"
    )
    if receipt is not None:
        receipt.retryable_lock_timeout(lock_error)
    raise lock_error from error
