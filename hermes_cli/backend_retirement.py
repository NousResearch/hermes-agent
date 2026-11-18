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
        self._preparing = False
        self._token = None
        self._committed = False
        self._expires_at = 0.0
        self._now = time.monotonic

    def _expire(self):
        if self._token is not None and not self._committed and self._now() >= self._expires_at:
            self._token = None

    def acquire(self):
        with self._lock:
            self._expire()
            if self._preparing or self._token is not None:
                return False
            self._active += 1
            return True

    def release(self):
        with self._lock:
            self._active -= 1

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
            if self._preparing or self._token is not None or self._active:
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
            return {"ok": True}


retirement = RetirementFence()
