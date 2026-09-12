"""Shared lifecycle safety does not require a provider registry."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.computer_use import tool as cu
from tools.computer_use_tool import registry


@contextmanager
def _home(path):
    token = set_hermes_home_override(path)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


@pytest.fixture(autouse=True)
def reset_backend_state(monkeypatch):
    cu.reset_backend_for_tests()
    monkeypatch.setattr(cu, "_cua_permission_mode", lambda sid: "standard")
    yield
    cu.reset_backend_for_tests()


@pytest.mark.parametrize("stop_raises", [False, True])
def test_exit_stops_each_backend_in_its_owner_home(monkeypatch, tmp_path, stop_raises):
    stopped = []

    class Backend(cu._NoopBackend):
        def __init__(self):
            super().__init__()
            self.home = get_hermes_home()

        def stop(self):
            stopped.append((self.home, get_hermes_home()))
            if stop_raises:
                raise RuntimeError("inert teardown failure")

    monkeypatch.setattr(cu, "_new_backend", lambda mode: Backend())
    homes = [tmp_path / "owner-a", tmp_path / "owner-b"]
    for home in homes:
        with _home(home):
            cu._get_backend("")  # also populates the empty-session injection hook

    ambient = tmp_path / "exit-thread"
    with _home(ambient):
        cu._shutdown_backend_atexit()
        assert get_hermes_home() == ambient
        cu._shutdown_backend_atexit()
        assert get_hermes_home() == ambient

    assert stopped == [(home, home) for home in homes]
    assert not cu._backends and not cu._backend and not cu._backend_call_locks


@pytest.mark.parametrize("grant", ["approve_session", "always_approve"])
@pytest.mark.parametrize("replace_generation", [False, True])
def test_failed_generation_drops_only_its_grants(monkeypatch, tmp_path, grant, replace_generation):
    failing_home, sibling_home = tmp_path / "failing", tmp_path / "sibling"
    entered, resume = threading.Event(), threading.Event()
    created, prompts = [], []

    class Backend(cu._NoopBackend):
        def __init__(self):
            super().__init__()
            self.fail = get_hermes_home() == failing_home and not created
            self.stopped = False
            created.append(self)

        def start(self):
            if self.fail:
                entered.set()
                assert resume.wait(5)
                raise RuntimeError("inert startup failure")

        def stop(self):
            self.stopped = True

    def approve(*args):
        prompts.append(get_hermes_home())
        return grant

    def click(home):
        with _home(home):
            result = registry.dispatch("computer_use", {"action": "click", "element": 1}, session_id="same")
            return json.loads(result) if isinstance(result, str) else result

    monkeypatch.setattr(cu, "_new_backend", lambda mode: Backend())
    monkeypatch.setattr(cu, "_approval_callback", approve)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(click, failing_home)
        try:
            assert entered.wait(5)
            assert click(sibling_home)["ok"]
            if replace_generation:
                with _home(failing_home):
                    assert cu.release_computer_use_session("same")
                assert click(failing_home)["ok"]
        finally:
            resume.set()
        assert "error" in pending.result(5)

    assert created[0].stopped
    if not replace_generation:
        with _home(failing_home):
            owner = cu._backend_owner_key("same")
        assert owner not in cu._backend_start_locks
        assert owner not in cu._backends
        assert owner not in cu._session_auto_approve and owner not in cu._always_allow
    assert click(sibling_home)["ok"]
    assert click(failing_home)["ok"]
    assert prompts.count(sibling_home) == 1
    assert prompts.count(failing_home) == 2
    assert all(not backend.stopped for backend in created[1:])
