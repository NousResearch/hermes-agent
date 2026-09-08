"""Shared lifecycle safety does not require a provider registry."""

from contextlib import contextmanager

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.computer_use import tool as cu


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
