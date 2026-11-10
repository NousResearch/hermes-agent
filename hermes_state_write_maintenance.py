"""Coalesced post-commit maintenance, owned and drained by one SessionDB."""
import logging
import threading

from agent.memory_provider import spawn_context_thread

logger = logging.getLogger('hermes_state')


class WriteMaintenance:
    """At most one active pass and one pending bit per kind, never an executor queue.

    The worker exits when idle. close() drains it before closing SQLite; neither
    maintenance operation reopens a closed connection. The spawning context binds
    the owning profile, just as the write that scheduled the pass did.
    """
    def __init__(self, db):
        self.db = db
        self.lock = threading.Lock()
        self.thread = None
        self.pending = set()
        self.stopping = False

    def schedule(self, *, checkpoint, merge):
        with self.lock:
            if self.stopping:
                return
            if checkpoint:
                self.pending.add('checkpoint')
            if merge:
                self.pending.add('merge')
            if self.pending and self.thread is None:
                try:
                    self.thread = spawn_context_thread(self._run, name='state-maintenance')
                    self.thread.start()
                except Exception:  # health: allow BLE001 -- retain pending work for the next write/close
                    self.thread = None
                    logger.warning('Post-commit maintenance scheduling failed')

    def _run(self):
        while True:
            with self.lock:
                if not self.pending:
                    self.thread = None
                    return
                pending, self.pending = self.pending, set()
            for kind in ('checkpoint', 'merge'):
                if kind not in pending:
                    continue
                try:
                    if kind == 'checkpoint':
                        self.db._try_wal_checkpoint()
                    else:
                        self.db._try_incremental_merge_fts()
                except Exception:  # health: allow BLE001 -- committed writes cannot fail due to maintenance
                    logger.warning('Post-commit maintenance failed kind=%s', kind)

    def drain(self):
        with self.lock:
            self.stopping = True
            thread = self.thread
        # Never join under the DB lock: the pass needs it to finish.
        if thread is not None:
            thread.join()
        else:
            self._run()  # a thread-start failure must not lose due maintenance
