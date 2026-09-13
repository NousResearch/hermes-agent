"""Named-profile binding must precede registry construction admission."""

from pathlib import Path
import threading
import shutil

import hermes_state_registry as registry
from hermes_cli import profile_lifecycle
from hermes_cli.profile_incarnation import ensure_profile_incarnation
from hermes_cli.web_server_sessions import _open_session_db_at_path
from tui_gateway import server


def test_rest_open_cannot_admit_while_gateway_holds_profile_lease(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = ensure_profile_incarnation(profile)
    path = (profile / "state.db").resolve()
    attempted_lease = threading.Event()
    real_lock = profile_lifecycle._PROFILE_LIFECYCLE_LOCK
    results = []
    errors = []

    def rest_open():
        try:
            results.append(_open_session_db_at_path(path, read_only=False))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=rest_open, daemon=True)

    class ObservedLock:
        def __enter__(self):
            if threading.current_thread() is worker:
                attempted_lease.set()
            return real_lock.__enter__()

        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    monkeypatch.setattr(profile_lifecycle, "_PROFILE_LIFECYCLE_LOCK", ObservedLock())
    gateway_db = None
    admitted = None
    try:
        with server._profile_home_lease(profile, token):
            worker.start()
            assert attempted_lease.wait(5), "REST did not reach the lifecycle lease"
            with registry._lock:
                admitted = path in registry._opening
            # Do not hang the unfixed runner: admission here proves the cycle.
            if not admitted:
                gateway_db = server._open_profile_session_db(
                    profile, expected_profile_incarnation=token,
                )
        worker.join(10)
        assert not worker.is_alive()
        assert errors == []
        assert not admitted, "REST admitted an opener that still needs the gateway's lease"
        assert results == [gateway_db]
    finally:
        for db in results:
            db.close()
        if gateway_db is not None:
            gateway_db.close()


def test_warm_acquire_rejects_generation_replaced_before_registry_lock(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = ensure_profile_incarnation(profile)
    path = (profile / "state.db").resolve()
    old_db = registry.acquire(path, expected_profile_incarnation=token)
    reached_lock = threading.Event()
    resume = threading.Event()
    results, errors = [], []
    real_lock = registry._lock

    def stale_acquire():
        try:
            results.append(registry.acquire(path, expected_profile_incarnation=token))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=stale_acquire, daemon=True)

    class PausedLock:
        def __enter__(self):
            if threading.current_thread() is worker and not reached_lock.is_set():
                reached_lock.set()
                assert resume.wait(10), "replacement did not finish"
            return real_lock.__enter__()

        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    monkeypatch.setattr(registry, "_lock", PausedLock())
    new_db = None
    try:
        worker.start()
        assert reached_lock.wait(5)
        old_db.close()
        old_db = None
        with profile_lifecycle.profile_lifecycle_lease():
            profile_lifecycle.mark_profile_deleting(profile, token)
            shutil.rmtree(profile)
            profile.mkdir()
            from hermes_cli.profile_incarnation import write_fresh_profile_incarnation

            new_token = write_fresh_profile_incarnation(profile)
            profile_lifecycle.publish_profile_generation(profile, new_token)
        new_db = registry.acquire(path, expected_profile_incarnation=new_token)
        resume.set()
        worker.join(10)
        assert not worker.is_alive()
        assert results == [], "stale acquire borrowed the replacement generation"
        assert len(errors) == 1 and isinstance(errors[0], FileNotFoundError)
        with registry._lock:
            assert registry._generations[path].refcount == 1
    finally:
        resume.set()
        worker.join(10)
        for db in results:
            db.close()
        if old_db is not None:
            old_db.close()
        if new_db is not None:
            new_db.close()
