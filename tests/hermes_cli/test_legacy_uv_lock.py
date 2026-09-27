"""Legacy uv cleanup takes PM's shared install lock, bounded and fail-closed.

Migrating/removing the pre-PM ``bin/uv{,x}`` is a shared mutation under the one install
lock, fail-closed on timeout.
Regression surface: an ``hermes update`` self-heal, a ``hermes doctor --fix`` and an
uninstall must all refuse to delete around a PM operation that is still running — and none of
them may hang behind one, nor resurrect a store an uninstall already removed.
"""

import os
import subprocess
import sys

import pytest

import hermes_cli.legacy_uv as legacy_uv
from hermes_cli.legacy_uv import remove_legacy_managed_uv

#: A separate process holding PM's lock. In-process cross-handle lock conflict differs per OS
#: (POSIX flock keys on the open file description, Windows byte locks do not), so the conflict
#: that proves "someone else holds it" is created the only way both agree on: another process.
_HOLD_LOCK = """\
import os, sys, time
from pm.filesystem import lock_fd

fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600)
if not lock_fd(fd, wait=True, timeout=10):
    raise SystemExit("child never took the lock")
print("held", flush=True)
time.sleep(60)
"""


def _hold_install_lock(lock_path):
    from pm.paths import repo_root

    child = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(lock_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=str(repo_root()),  # `python -c` puts cwd on sys.path, so `import pm` resolves
    )
    assert child.stdout.readline().strip() == "held", "lock holder never started"
    return child


def _stop(child):
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)
    if child.stdout is not None:
        child.stdout.close()


def _home_with_legacy_uv(tmp_path, monkeypatch):
    """A pre-PM ``bin/uv`` in the installer's real shape — an astral standalone
    BINARY (regular file, over the size floor) — plus a store root, both under
    this test's HERMES_HOME."""
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    legacy = home / "bin" / "uv"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pm.paths import writable_store_root

    store = writable_store_root()
    store.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(legacy_uv, "_LEGACY_UV_LOCK_TIMEOUT", 0.5)
    return home, legacy, store


@pytest.mark.platforms("posix")
def test_purge_fails_closed_while_the_install_lock_is_held(tmp_path, monkeypatch):
    home, legacy, store = _home_with_legacy_uv(tmp_path, monkeypatch)
    child = _hold_install_lock(store / ".install.lock")
    try:
        assert remove_legacy_managed_uv(home) == []
        assert legacy.exists()  # fail closed: leave it for the next run, never delete mid-flight
    finally:
        _stop(child)

    # Lock released: this call takes it and removes exactly the legacy binary.
    assert remove_legacy_managed_uv(home) == [legacy]
    assert not legacy.exists()


@pytest.mark.platforms("posix")
@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: 1)() == 0,
    reason="chmod-based read-only dirs are not enforceable on Windows or as root",
)
def test_purge_fails_closed_when_the_store_cannot_be_locked(tmp_path, monkeypatch):
    """A payload store PM cannot write (read-only on disk) must skip, not raise: doctor's
    check runs with ``on_error=None``, so an escaping OSError would abort the whole run."""
    home, legacy, store = _home_with_legacy_uv(tmp_path, monkeypatch)
    store.chmod(0o555)
    try:
        assert remove_legacy_managed_uv(home) == []
        assert legacy.exists()
    finally:
        store.chmod(0o755)


@pytest.mark.platforms("posix")
def test_purge_does_not_create_an_absent_store(tmp_path, monkeypatch):
    """An already-absent store is not recreated just to acquire the cleanup lock."""
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes"
    legacy = home / "bin" / "uv"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"\x7fELF" + b"\0" * (2 << 20))
    legacy.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from pm.paths import writable_store_root

    assert not writable_store_root().is_dir()
    assert remove_legacy_managed_uv(home) == [legacy]
    assert not writable_store_root().exists()
    assert not legacy.exists()


