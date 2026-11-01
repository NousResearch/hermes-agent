"""SessionDB write phases: local mutex patience and post-commit maintenance."""
from contextlib import contextmanager
import sqlite3
import time


class WriterMutexTimeout(sqlite3.OperationalError):
    """The local writer is busy, not a damaged database or a provider failure."""


@contextmanager
def writer_mutex_until(lock, deadline):
    """Do not let an observation or lease renewal queue forever behind another writer."""
    if not lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
        raise WriterMutexTimeout(
            "database is locked (in-process state.db writer mutex exceeded write patience)"
        )
    try:
        yield
    finally:
        lock.release()


def maintain_after_commit(db):
    """Count committed writes and run due maintenance after releasing the writer mutex."""
    db._write_count += 1
    if db._write_count % db._CHECKPOINT_EVERY_N_WRITES == 0:
        db._try_wal_checkpoint()
    if db._write_count % db._FTS_MERGE_EVERY_N_WRITES == 0:
        db._try_incremental_merge_fts()
