"""Dispatch admission must survive target/mode changes and concurrent teardown."""

from __future__ import annotations

import contextvars
import json
import threading
from concurrent.futures import Future
from contextlib import contextmanager

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.computer_use import tool as cu
from tools.computer_use_tool import registry


@contextmanager
def _profile(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _spawn(fn, name):
    future = Future()
    context = contextvars.copy_context()

    def run():
        try:
            future.set_result(context.run(fn))
        except BaseException as exc:
            future.set_exception(exc)

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread, future


def _call():
    return json.loads(registry.dispatch("computer_use", {"action": "list_apps"}, session_id="shared"))


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("computer_use:\n  target: auto\n  permission_mode: standard\n")
    created = []

    class Backend(cu._NoopBackend):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode
            self.stopped = False
            self.count = 0
            self.before_action = lambda: None
            created.append(self)

        def stop(self):
            self.stopped = True

        def list_apps(self):
            assert not self.stopped, "dispatch used a retired backend"
            self.count += 1
            self.before_action()
            assert not self.stopped, "teardown ran during dispatch"
            return [{"name": str(created.index(self))}]

    cu.reset_backend_for_tests()
    monkeypatch.setattr(cu, "_new_backend", Backend)
    yield homes, created
    cu.reset_backend_for_tests()


@pytest.mark.parametrize("pause_at", ["after_lookup", "before_acquire"])
@pytest.mark.parametrize("change", ["target", "mode", "release", "queued_config"])
def test_dispatch_rechecks_admission_after_lookup_and_after_lock_wait(runtime, monkeypatch, pause_at, change):
    homes, created = runtime
    paused, resume = threading.Event(), threading.Event()
    real_get = cu._get_backend
    worker = None
    with _profile(homes[0]):
        old = real_get("shared")
        sid = cu._scoped_sid("shared")
        old_lock = cu._backend_call_locks[sid]

        def pause():
            if threading.current_thread().name == "waiting-call" and not paused.is_set():
                paused.set()
                assert resume.wait(10), "test did not release the waiting call"

        def get(session_id=""):
            backend = real_get(session_id)
            if pause_at == "after_lookup":
                pause()
            return backend

        class PausingLock:
            def __enter__(self):
                pause()
                old_lock.acquire()
                return self

            def __exit__(self, *_):
                old_lock.release()

        monkeypatch.setattr(cu, "_get_backend", get)
        if pause_at == "before_acquire":
            with cu._backend_lock:
                cu._backend_call_locks[sid] = PausingLock()
        try:
            worker, result = _spawn(_call, "waiting-call")
            assert paused.wait(10)
            if change == "release":
                assert cu.release_computer_use_session("shared")
            else:
                # Real profile config, not a mocked selection identity. The
                # queued_config case has no other lookup to retire the cache.
                mode = "bounded" if change == "mode" else "standard"
                target = "linux" if change != "mode" else "auto"
                (homes[0] / "config.yaml").write_text(
                    f"computer_use:\n  target: {target}\n  permission_mode: {mode}\n"
                )
                if change != "queued_config":
                    assert real_get("shared") is not old
            if change != "queued_config":
                assert old.stopped
            resume.set()
            value = result.result(timeout=10)
            assert "error" not in value, value
            assert old.count == 0
            assert sum(backend.count for backend in created) == 1
            assert value["apps"] == [{"name": str(created.index(real_get("shared")))}]
        finally:
            resume.set()
            if worker:
                worker.join(timeout=10)
                assert not worker.is_alive()


@pytest.mark.parametrize("fail_action", [False, True])
def test_release_waits_for_admitted_call_without_blocking_other_profile_or_retrying(runtime, monkeypatch, fail_action):
    homes, created = runtime
    action_entered, finish_action, stop_attempted = (threading.Event() for _ in range(3))
    threads = []
    with _profile(homes[0]):
        old = cu._get_backend("shared")

        def action():
            action_entered.set()
            assert finish_action.wait(10)
            if fail_action:
                raise RuntimeError("action outcome is unknown; do not replay")

        old.before_action = action
        real_stop = cu._stop_backend

        def stop(backend, lock, on_error):
            if backend is old:
                stop_attempted.set()
            return real_stop(backend, lock, on_error)

        monkeypatch.setattr(cu, "_stop_backend", stop)
        try:
            thread, result = _spawn(_call, "admitted-call")
            threads.append(thread)
            assert action_entered.wait(10)
            thread, released = _spawn(lambda: cu.release_computer_use_session("shared"), "release")
            threads.append(thread)
            assert stop_attempted.wait(10)
            assert not old.stopped
            assert not released.done()
            with _profile(homes[1]):
                thread, other = _spawn(_call, "other-profile")
                threads.append(thread)
                assert "error" not in other.result(timeout=10)
                assert not old.stopped
            finish_action.set()
            value = result.result(timeout=10)
            assert ("error" in value) is fail_action
            assert released.result(timeout=10) is True
            assert old.stopped and old.count == 1
            with _profile(homes[1]):
                assert not cu._get_backend("shared").stopped
        finally:
            finish_action.set()
            for thread in threads:
                thread.join(timeout=10)
                assert not thread.is_alive()
