"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import os
import threading

import pytest


# ---------------------------------------------------------------------------
# os._exit tripwire (silent pytest-kill regression guard)
# ---------------------------------------------------------------------------
# In Aug 2026 the dashboard auth-gate tests silently killed the whole pytest
# process mid-file with exit code 0 (no summary, no junit XML): leaked
# HERMES_PARENT_* env from a Desktop-spawned developer shell armed
# web_server's production parent-death watchdog inside the test process, and
# its TZ-skewed start-marker mismatch fired ``os._exit(0)`` from a daemon
# thread. Any in-process ``os._exit`` during these tests is that bug class —
# the tests in this package that exercise real ``os._exit`` behavior do so in
# SUBPROCESSES (test_signal_handler_kanban_worker) or monkeypatch ``os._exit``
# themselves (test_update_handoff_exit, which simply replaces this wrapper for
# the test's duration). So: record the call, refuse to exit, and fail a test
# loudly instead of vanishing.
#
# The tripwire is installed for the WHOLE session (pytest_configure), not per
# test: the watchdog fires from a daemon thread tens of milliseconds after the
# test body finishes (a ``ps`` probe subprocess is in flight), which lands in
# the teardown/setup gap where a per-test patch has already been removed — the
# exact race that made a fixture-scoped version of this guard miss the kill.

_os_exit_tripped: list[tuple[int, str]] = []
_real_os_exit = os._exit


def _tripwire_os_exit(code: int) -> None:
    _os_exit_tripped.append((code, threading.current_thread().name))
    os.write(
        2,
        (
            "\n[tests/hermes_cli tripwire] os._exit(%r) called in-process "
            "(thread %r) — refusing to kill pytest; failing the active test "
            "instead. See tests/hermes_cli/conftest.py.\n"
            % (code, threading.current_thread().name)
        ).encode(),
    )
    raise RuntimeError(
        f"os._exit({code!r}) called inside the pytest process "
        f"(thread {threading.current_thread().name!r})"
    )


def pytest_configure(config):
    os._exit = _tripwire_os_exit


def pytest_unconfigure(config):
    os._exit = _real_os_exit


@pytest.fixture(autouse=True)
def _os_exit_tripwire():
    """Turn an in-process ``os._exit`` into a visible test failure.

    Without this, library code calling ``os._exit`` from any thread kills the
    interpreter with the given code — pytest prints no failure, no summary,
    writes no junit XML, and (exit code 0) a CI runner would report success
    while every test after the exit point silently never ran.

    The check runs per test so the failure is attributed close to the culprit;
    a trip that lands between tests is reported by the NEXT test's teardown,
    which is still a loud, non-zero-exit failure.
    """
    if _os_exit_tripped:
        # A trip from a previous test's background thread landed in the gap.
        calls = ", ".join(
            f"os._exit({code!r}) from thread {thread!r}"
            for code, thread in _os_exit_tripped
        )
        _os_exit_tripped.clear()
        pytest.fail(
            f"os._exit was called in-process before this test started: {calls}. "
            "A previous test armed a hard-exit path (production watchdog / "
            "signal handler) inside the pytest process."
        )
    try:
        yield
    finally:
        if _os_exit_tripped:
            calls = ", ".join(
                f"os._exit({code!r}) from thread {thread!r}"
                for code, thread in _os_exit_tripped
            )
            _os_exit_tripped.clear()
            pytest.fail(
                "os._exit was called inside the pytest process during this "
                f"test: {calls}. This would have silently killed the entire "
                "pytest run (exit 0, no summary). Find and isolate the "
                "hard-exit path (production watchdogs/signal handlers must "
                "not arm inside in-process tests)."
            )


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )
