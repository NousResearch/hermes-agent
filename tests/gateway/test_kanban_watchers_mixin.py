"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0."""
    import os

    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)


@pytest.mark.asyncio
async def test_auto_decompose_tick_installs_secret_scope_under_multiplex(
    monkeypatch, tmp_path,
):
    """Regression: the auto-decompose tick fires from asyncio.to_thread with
    no per-turn scope installed, same gap as the cron ticker (fdab380a1).
    Under profile isolation (multiplex active), decompose_task's auxiliary
    LLM call resolves credentials via resolve_runtime_provider -> get_secret,
    which fails closed with UnscopedSecretError outside an installed scope.

    Behavior contract: a profile secret scope is present while
    decompose_task runs and torn down again afterward — mirrors
    test_run_one_job_installs_secret_scope_under_multiplex (def6d6fe1).
    """
    import hermes_cli
    import gateway.kanban_watchers as kw
    from agent import secret_scope as ss

    (tmp_path / ".env").write_text("OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n")

    fake_cfg = {
        "kanban": {
            "dispatch_in_gateway": True,
            "auto_decompose": True,
            "auto_decompose_per_tick": 1,
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_cfg)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        kw, "_acquire_singleton_lock", lambda path: (object(), "held"),
    )
    monkeypatch.setattr(kw, "_release_singleton_lock", lambda handle: None)

    fake_kb = MagicMock()
    fake_kb.kanban_home.return_value = tmp_path
    fake_kb.DEFAULT_BOARD = "default"
    _board_list_calls = {"n": 0}

    def _list_boards(include_archived=False):
        # First call is _auto_decompose_tick's own board enumeration; every
        # later call (from _tick_once / _ready_nonempty in the same
        # iteration) gets an empty list so only the auto-decompose path
        # under test actually does anything this tick.
        _board_list_calls["n"] += 1
        if _board_list_calls["n"] == 1:
            return [{"slug": "default"}]
        return []

    fake_kb.list_boards.side_effect = _list_boards
    # "from hermes_cli import kanban_db as _kb" resolves the submodule via
    # the package attribute if already imported, so both the sys.modules
    # entry and the package attribute must point at the fake.
    monkeypatch.setattr(hermes_cli, "kanban_db", fake_kb, raising=False)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_db", fake_kb)

    self = SimpleNamespace(_running=True, _kanban_dispatcher_lock_handle=None)

    scope_during_call = {}

    def fake_decompose_task(task_id, author=None):
        # Flip the stop flag FIRST: on unmodified code (no scope installed)
        # ss.get_secret raises UnscopedSecretError under multiplex, which the
        # real _auto_decompose_tick swallows in a bare except — if _running
        # were set after the read, that exception would leave the watcher
        # loop spinning forever instead of failing the assertions below.
        self._running = False
        scope_during_call["scope"] = ss.current_secret_scope()
        scope_during_call["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    fake_decomp = SimpleNamespace(
        list_triage_ids=lambda: ["t1"],
        decompose_task=fake_decompose_task,
    )
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake_decomp, raising=False)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake_decomp)

    ss.set_multiplex_active(True)
    try:
        await GatewayKanbanWatchersMixin._kanban_dispatcher_watcher(self)
    finally:
        ss.set_multiplex_active(False)

    assert scope_during_call.get("scope") is not None, (
        "decompose_task ran with no secret scope installed"
    )
    assert scope_during_call.get("base_url") == "https://openrouter.ai/api/v1"
    # Torn down after the tick — no leak into subsequent work.
    assert ss.current_secret_scope() is None
