"""Managed operation leases and bounded failure isolation for optional forum metadata."""

from contextlib import contextmanager
import logging
import sqlite3
import threading

from hermes_state_errors import StateDbReplacedError, classify_persistence_error

logger = logging.getLogger(__name__)


class GroupTopicCacheStopped(RuntimeError):
    """Disconnect fenced new cache work; only explicit connect may resume it."""


def optional_cache_action(action, fallback=None):
    """Degrade storage metadata only; identity and programming guards remain fatal."""
    try:
        return action()
    except (FileNotFoundError, sqlite3.ProgrammingError, sqlite3.IntegrityError):
        # Preserve named-profile guards, invalid SQL/bindings and violated data invariants.
        raise
    except (OSError, sqlite3.DatabaseError, StateDbReplacedError, GroupTopicCacheStopped) as exc:
        logger.warning("Telegram forum topic cache unavailable (%s/%s)", type(exc).__name__, classify_persistence_error(exc))
        return fallback


class TelegramGroupTopicCacheMixin:
    def _init_group_topic_cache(self) -> None:
        self._group_topic_db_lock = threading.Lock()
        self._telegram_group_topic_cache_db = None
        self._group_topic_cache_epoch = 0
        self._group_topic_cache_stopped = False

    def _resume_group_topic_cache(self) -> None:
        with self._group_topic_db_lock:
            self._group_topic_cache_epoch += 1
            self._group_topic_cache_stopped = False

    def _group_topic_cache_generation(self):
        with self._group_topic_db_lock:
            if self._group_topic_cache_stopped:
                raise GroupTopicCacheStopped()
            return self._group_topic_cache_epoch

    def _get_telegram_group_topic_cache_db(self):
        """Adapter-owned reference. Production DB use must ALSO hold an operation lease."""
        from hermes_state_registry import acquire, release
        epoch = self._group_topic_cache_generation()
        with self._group_topic_db_lock:
            if self._telegram_group_topic_cache_db is not None:
                return self._telegram_group_topic_cache_db
        # Opening may block: never hold the lifecycle lock while waiting for SQLite/registry.
        db = acquire(self._update_receipt_dir / "state.db")
        with self._group_topic_db_lock:
            stale = self._group_topic_cache_stopped or epoch != self._group_topic_cache_epoch
            if not stale and self._telegram_group_topic_cache_db is None:
                self._telegram_group_topic_cache_db = db
                return db
            existing = self._telegram_group_topic_cache_db
        release(db)
        if stale:
            raise GroupTopicCacheStopped()
        return existing

    @contextmanager
    def _group_topic_cache_operation(self):
        from hermes_state_registry import acquire, release
        epoch = self._group_topic_cache_generation()
        self._get_telegram_group_topic_cache_db()
        db = acquire(self._update_receipt_dir / "state.db")
        try:
            with self._group_topic_db_lock:
                if self._group_topic_cache_stopped or epoch != self._group_topic_cache_epoch:
                    raise GroupTopicCacheStopped()
            yield db
        finally:
            release(db)

    def _release_telegram_group_topic_cache_db(self) -> None:
        from hermes_state_registry import release
        if not hasattr(self, "_group_topic_db_lock"):
            return
        with self._group_topic_db_lock:
            self._group_topic_cache_stopped = True
            self._group_topic_cache_epoch += 1
            db, self._telegram_group_topic_cache_db = self._telegram_group_topic_cache_db, None
        if db is not None:
            release(db)

    def _optional_group_topic_cache(self, operation, fallback=None):
        """Metadata failures never discard user input; identity/programming guards still raise."""
        def run():
            with self._group_topic_cache_operation() as db:
                return operation(db)
        return optional_cache_action(run, fallback)
