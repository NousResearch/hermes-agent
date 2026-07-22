"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_worker_exit_registry():
    """Isolate process-level dispatcher exit caches between Kanban tests."""
    from hermes_cli import kanban_db as _kb

    saved_exits = dict(_kb._recent_worker_exits)
    saved_rl = getattr(_kb.detect_crashed_workers, "_last_rate_limited", None)
    saved_ab = getattr(_kb.detect_crashed_workers, "_last_auto_blocked", None)
    _kb._recent_worker_exits.clear()
    _kb.detect_crashed_workers._last_rate_limited = []  # type: ignore[attr-defined]
    _kb.detect_crashed_workers._last_auto_blocked = []  # type: ignore[attr-defined]
    try:
        yield
    finally:
        _kb._recent_worker_exits.clear()
        _kb._recent_worker_exits.update(saved_exits)
        _kb.detect_crashed_workers._last_rate_limited = (  # type: ignore[attr-defined]
            saved_rl if saved_rl is not None else []
        )
        _kb.detect_crashed_workers._last_auto_blocked = (  # type: ignore[attr-defined]
            saved_ab if saved_ab is not None else []
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
