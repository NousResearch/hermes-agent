"""Revoked backend lookups must never regain dispatch authority."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.computer_use import tool as cu
from tools.computer_use_tool import registry


@pytest.fixture
def backends(monkeypatch):
    class RecordingBackend(cu._NoopBackend):
        def __init__(self):
            super().__init__()
            self.stopped = False

        def stop(self):
            self.stopped = True

    created = []

    def create(sid, mode, provider):
        backend = RecordingBackend()
        created.append(backend)
        return backend

    cu.reset_backend_for_tests()
    monkeypatch.setattr(cu, "_new_backend", create)
    monkeypatch.setattr(cu, "_cua_permission_mode", lambda sid: "standard")
    monkeypatch.setattr(cu, "_approval_callback", lambda *args: "approve_once")
    yield created
    cu.reset_backend_for_tests()


def _click(session_id, element=1):
    return json.loads(registry.dispatch(
        "computer_use", {"action": "click", "element": element}, session_id=session_id,
    ))


@pytest.mark.parametrize("replace_before_resume", [False, True])
def test_released_lookup_cannot_dispatch(monkeypatch, backends, replace_before_resume):
    resolved, resume = threading.Event(), threading.Event()
    get_backend = cu._get_backend

    def paused_lookup(session_id=""):
        backend = get_backend(session_id)
        if backend is backends[0]:
            resolved.set()
            assert resume.wait(5)
        return backend

    monkeypatch.setattr(cu, "_get_backend", paused_lookup)
    owner = cu._backend_owner_key("released")
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(_click, "released")
        try:
            assert resolved.wait(5)
            assert pool.submit(cu.release_computer_use_session, "released").result(5)
            assert backends[0].stopped
            if replace_before_resume:
                assert _click("released")["ok"]
        finally:
            resume.set()
        result = pending.result(5)

    assert backends[0].calls == [], "a released lookup must not reach the backend"
    assert result.get("error"), result
    if not replace_before_resume:
        assert owner not in cu._backends
        assert owner not in cu._backend_call_locks, "stale dispatch must not recreate a lock"
    assert _click("released")["ok"], "an explicit new call may acquire fresh authority"
    assert not backends[-1].stopped
    assert backends[-1].calls


@pytest.mark.parametrize("queued_first", [False, True])
def test_release_drains_only_the_running_call(monkeypatch, backends, queued_first):
    running, finish = threading.Event(), threading.Event()
    queued, stopping = threading.Event(), threading.Event()
    resume_queue, resume_stop = threading.Event(), threading.Event()
    backend = cu._get_backend("released")
    click = backend.click

    def blocking_click(**kwargs):
        if kwargs["element"] == 1:
            running.set()
            assert finish.wait(5)
        return click(**kwargs)

    monkeypatch.setattr(backend, "click", blocking_click)
    owner = cu._backend_owner_key("released")

    class OrderedCallLock:
        """Use the real lock, choosing which waiter reaches it first."""
        entries = 0
        lock = cu._backend_call_locks[owner]

        def __enter__(self):
            self.entries += 1
            if self.entries == 2:
                queued.set()
                if not queued_first:
                    assert resume_queue.wait(5)
            elif self.entries == 3:
                stopping.set()
                if queued_first:
                    assert resume_stop.wait(5)
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    cu._backend_call_locks[owner] = OrderedCallLock()
    with ThreadPoolExecutor(max_workers=4) as pool:
        active = pool.submit(_click, "released", 1)
        try:
            assert running.wait(5)
            pending = pool.submit(_click, "released", 2)
            assert queued.wait(5)
            released = pool.submit(cu.release_computer_use_session, "released")
            assert stopping.wait(5)  # release has detached authority, but the action still owns its lock
            assert not backend.stopped
            assert not released.done()
            assert pool.submit(_click, "unrelated").result(5)["ok"]
            assert pool.submit(cu.release_computer_use_session, "unrelated").result(5)
            finish.set()
            assert active.result(5)["ok"]
            if queued_first:
                result = pending.result(5)
                assert not backend.stopped
                resume_stop.set()
                assert released.result(5)
            else:
                assert released.result(5)
                resume_queue.set()
                result = pending.result(5)
        finally:
            finish.set()
            resume_queue.set()
            resume_stop.set()

    assert [args["element"] for name, args in backend.calls] == [1], "queued authority was revoked"
    assert result.get("error"), result
    assert backend.stopped
    assert owner not in cu._backend_call_locks
    assert _click("released")["ok"]
