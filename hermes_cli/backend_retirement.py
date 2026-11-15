"""Process-owned admission fence for cooperative Desktop backend retirement.

The gate belongs to the process, not a profile or transport. Diagnostic ledgers alone cannot
close the check/use race: every source of new work must reserve admission before dispatch.
"""

from contextlib import contextmanager
import logging
import secrets
import threading
import time


class RetirementFence:
    """Exclusive 30-second prepare permit; commit closes admission permanently.

    Tokens are random and recognized only by this process instance. A committed prepare
    returns the same token so a client can recover a lost commit response without reopening.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0
        self._thread_depth = {}
        self._draining = False
        self._drain_expires_at = 0.0
        self._preparing = False
        self._token = None
        self._committed = False
        self._expires_at = 0.0
        self._now = time.monotonic

    def _expire(self):
        now = self._now()
        if self._token is not None and not self._committed and now >= self._expires_at:
            self._token = None
            self._draining = False
            self._drain_expires_at = 0.0
        if self._draining and not self._committed and self._drain_expires_at and now >= self._drain_expires_at:
            self._draining = False
            self._drain_expires_at = 0.0

    def acquire(self, *, bind_thread: bool = True, force: bool = False):
        tid = threading.get_ident()
        with self._lock:
            self._expire()
            if self._committed:
                return False
            depth = self._thread_depth.get(tid, 0)
            if not force and (
                self._preparing
                or self._token is not None
                or (self._draining and self._active > 0 and depth == 0)
            ):
                return False
            self._active += 1
            if bind_thread:
                self._thread_depth[tid] = depth + 1
            return True

    def release(self, *, bind_thread: bool = True):
        tid = threading.get_ident()
        with self._lock:
            self._active = max(0, self._active - 1)
            if bind_thread:
                depth = self._thread_depth.get(tid, 1) - 1
                if depth <= 0:
                    self._thread_depth.pop(tid, None)
                else:
                    self._thread_depth[tid] = depth
            if self._active <= 0:
                self._draining = False
                self._drain_expires_at = 0.0

    @contextmanager
    def worker_thread_scope(self):
        """Bind re-entrant depth to a spawned worker thread whose reservation was acquired on its parent."""
        tid = threading.get_ident()
        with self._lock:
            self._thread_depth[tid] = self._thread_depth.get(tid, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                depth = self._thread_depth.get(tid, 1) - 1
                if depth <= 0:
                    self._thread_depth.pop(tid, None)
                else:
                    self._thread_depth[tid] = depth

    def active_count(self):
        with self._lock:
            return self._active

    @contextmanager
    def work(self):
        admitted = self.acquire()
        try:
            yield admitted
        finally:
            if admitted:
                self.release()

    def prepare(self):
        from hermes_cli.web_server_idle_proof import idle_proof

        with self._lock:
            self._expire()
            if self._committed:
                return {"ok": True, "idle": True, "token": self._token}
            if self._preparing or self._token is not None:
                return {"ok": False, "idle": False}
            if self._active:
                self._draining = True
                self._drain_expires_at = self._now() + 30.0
                return {"ok": False, "idle": False}
            self._preparing = True
        # No admissions can cross the freeze. Do not hold our lock while reading ledgers:
        # some callers already hold a session/cron lock when reserving their work.
        verdict = None
        try:
            verdict = idle_proof()["idle"]
        except Exception:
            logging.getLogger(__name__).warning("Retirement idle probe unavailable", exc_info=True)
        finally:
            with self._lock:
                self._preparing = False
                self._draining = False
                self._drain_expires_at = 0.0
                if verdict is True:
                    self._token = token = secrets.token_urlsafe(32)
                    self._expires_at = self._now() + 30.0
        if verdict is not True:
            return {"ok": False, "idle": verdict}
        return {"ok": True, "idle": True, "token": token}

    def commit(self, token):
        with self._lock:
            self._expire()
            if not token or token != self._token:
                return {"ok": False}
            self._committed = True
            return {"ok": True}

    def cancel(self, token):
        with self._lock:
            self._expire()
            if not token or token != self._token or self._committed:
                return {"ok": False}
            self._token = None
            self._draining = False
            self._drain_expires_at = 0.0
            return {"ok": True}


retirement = RetirementFence()
