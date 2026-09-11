"""Per-profile backend singleton (2026-09-01 state.db corruption, second cause).

Hermes Desktop's ``startHermes()`` re-spawn path once started a second
default-profile backend without stopping the first; both wrote the same
736MB WAL-mode ``state.db`` concurrently and corrupted the
``gateway_heartbeats`` index in about an hour. A second ``hermes serve`` /
``hermes dashboard`` for the same profile MUST refuse to start with a
machine-readable sentinel + exit 75 (BSD ``EX_TEMPFAIL`` convention used
elsewhere in this repo). Escape hatch: ``HERMES_SERVE_ALLOW_DUPLICATE=1``.

Linux ``flock`` is per-open-file-description, so same-process reentry
through ``acquire_backend_serve_lock`` is intentionally idempotent (the
caller already knows it owns the lock). Real conflict is across processes,
which is what this test exercises.
"""
import contextlib
import fcntl
import os
import pathlib
import subprocess
import sys
import unittest
from unittest.mock import patch


_REAL_ENV = os.environ.copy()


def _reset_env():
    for k in list(os.environ):
        os.environ.pop(k, None)
    for k, v in _REAL_ENV.items():
        os.environ[k] = v


@contextlib.contextmanager
def _patched_home(tmp_path, profile="default"):
    """Route ``get_hermes_home()`` at a tmp_path so the lock file is local
    and isolated from the host's actual ``~/.hermes``."""
    from hermes_constants import get_hermes_home
    fake = tmp_path / profile
    fake.mkdir(parents=True, exist_ok=True)
    with patch("hermes_constants.get_hermes_home", return_value=fake), \
         patch("gateway.status._get_process_hermes_home", return_value=fake):
        yield fake


class TestServeProfileSingleton(unittest.TestCase):

    def setUp(self):
        _reset_env()

    def test_lock_path_under_hermes_home(self):
        """Lock file lives next to ``gateway.lock``, scoped per-profile via ``get_hermes_home``."""
        from gateway.status import _get_serve_lock_path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fake = pathlib.Path(tmp) / "my-profile"
            fake.mkdir()
            with _patched_home(fake):
                self.assertEqual(_get_serve_lock_path(), fake / "default" / "serve.lock")

    def test_acquire_returns_true_idempotently_in_same_process(self):
        """Linux flock is per-open-file-description; same-process reentry
        through ``acquire_backend_serve_lock`` is intentionally idempotent so
        a caller that already owns the lock is not falsely refused."""
        from gateway.status import acquire_backend_serve_lock, release_backend_serve_lock
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with _patched_home(pathlib.Path(tmp)):
                self.assertTrue(acquire_backend_serve_lock())
                self.assertTrue(acquire_backend_serve_lock(),
                                "same-process reentry must be idempotent")
                release_backend_serve_lock()
                # After release, re-acquire still works.
                self.assertTrue(acquire_backend_serve_lock())
                release_backend_serve_lock()

    def test_release_then_reacquire_works(self):
        from gateway.status import acquire_backend_serve_lock, release_backend_serve_lock
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with _patched_home(pathlib.Path(tmp)):
                self.assertTrue(acquire_backend_serve_lock())
                release_backend_serve_lock()
                self.assertTrue(acquire_backend_serve_lock())
                release_backend_serve_lock()

    def test_external_subprocess_holding_lock_blocks_acquire(self):
        """A different process holding the lock must cause ``acquire_backend_serve_lock``
        to return False — the actual cross-process collision the fix exists to
        prevent."""
        from gateway.status import acquire_backend_serve_lock, release_backend_serve_lock, _get_serve_lock_path
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = pathlib.Path(tmp)
            with _patched_home(tmp_p):
                lock_path = str(_get_serve_lock_path())
                helper = pathlib.Path(__file__).parent / "_serve_lock_holder.py"
                # Spawn a child that holds an EX flock on the SAME path the
                # parent is about to try. Pass the path explicitly — the
                # child cannot see this process's monkeypatch of
                # get_hermes_home().
                child = subprocess.Popen(
                    [sys.executable, str(helper), lock_path])
                try:
                    # Wait for the child to actually take the flock by
                    # probing it from a fresh OFD in this process: a non-
                    # blocking EX probe fails iff another process holds it.
                    self.assertTrue(self._wait_for_peer_lock(lock_path, timeout=5.0),
                                    "child did not acquire the lock in time")
                    self.assertFalse(acquire_backend_serve_lock(),
                                     "acquire returned True while a peer process held the lock")
                finally:
                    child.terminate(); child.wait(timeout=5)
                # After the child released the lock, acquire succeeds again.
                self.assertTrue(acquire_backend_serve_lock())
                release_backend_serve_lock()

    @staticmethod
    def _wait_for_peer_lock(lock_path: str, *, timeout: float) -> bool:
        """True when a non-blocking flock on *lock_path* fails — meaning a
        different process holds it. Polls because the child races its own
        imports vs the Popen returning to the parent."""
        import time as _time
        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            try:
                h = open(lock_path, "a+b")
            except OSError:
                _time.sleep(0.05)
                continue
            try:
                fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                return True  # peer is holding it
            else:
                # We just acquired it — release and keep polling; the peer
                # hasn't taken it yet.
                fcntl.flock(h.fileno(), fcntl.LOCK_UN)
                h.close()
                _time.sleep(0.05)
        return False

    def test_escape_hatch_constant_and_exit_code(self):
        """The sentinel and exit code are part of the public contract that
        the Desktop spawn watcher greps — keep them stable."""
        from hermes_cli.web_server import (
            BACKEND_PROFILE_ALREADY_RUNNING_EXIT_CODE,
            _PROFILE_ALREADY_RUNNING_SENTINEL,
        )
        self.assertEqual(BACKEND_PROFILE_ALREADY_RUNNING_EXIT_CODE, 75)
        self.assertEqual(_PROFILE_ALREADY_RUNNING_SENTINEL, "BACKEND_PROFILE_ALREADY_RUNNING")
