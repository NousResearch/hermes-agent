"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect

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


def test_gateway_dispatcher_stuck_warning_names_guard_reason(monkeypatch, caplog):
    """The embedded dispatcher's "stuck" warning names the respawn-guard reason
    holding the ready queue (#111910) instead of a bare zero-spawn count."""
    import asyncio
    import logging

    import gateway.kanban_watchers as kw
    from hermes_cli import kanban_db_dispatch as kbd

    held = kbd.DispatchResult(respawn_guarded=[("t_held", "active_pr")])
    runner = object.__new__(kw.GatewayKanbanWatchersMixin)
    runner._running = True
    monkeypatch.setattr(runner, "_kanban_dispatcher_boot", lambda: (lambda: {}, object(), {}))

    class _Dispatcher:
        def __init__(self, *a, **k):
            pass

        def tick_once(self):
            return [("board", held)]

        def ready_nonempty(self):
            return True

    ticks = {"n": 0}

    async def _direct(fn, *args):
        return fn(*args)

    async def _sleep(_delay):
        ticks["n"] += 1
        if ticks["n"] > kw._HEALTH_WINDOW:
            runner._running = False

    monkeypatch.setattr(kw, "_KanbanDispatcher", _Dispatcher)
    monkeypatch.setattr(kw, "_resolve_dispatcher_settings", lambda cfg, kb: type("S", (), {"interval": 1.0})())
    monkeypatch.setattr(kw, "_to_thread_process_service", _direct)
    monkeypatch.setattr(kw, "_kanban_dispatch_allowed", lambda: True)
    monkeypatch.setattr(kw, "_resolve_auto_decompose_settings", lambda load_config: (False, 0))
    monkeypatch.setattr(kbd, "reap_worker_zombies", lambda: [])
    monkeypatch.setattr(kw.asyncio, "sleep", _sleep)

    with caplog.at_level(logging.WARNING, logger=kw.logger.name):
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=5.0))

    stuck = [r.getMessage() for r in caplog.records if "dispatcher stuck" in r.getMessage()]
    assert stuck, [r.getMessage() for r in caplog.records]
    assert "Last tick held back: active_pr=1." in stuck[0]


def test_gateway_logs_mount_refusal_reason_and_task(caplog):
    import logging

    from gateway.kanban_watchers_dispatcher import _log_spawn_results
    from hermes_cli.kanban_db_dispatch import DispatchResult

    result = DispatchResult(workspace_refused=[
        ("t2", "workspaces_root_unmounted: /Volumes/ramscratch"),
        ("t1", "workspaces_root_unmounted: /Volumes/ramscratch"),
    ])
    with caplog.at_level(logging.ERROR):
        assert not _log_spawn_results([("default", result)])
        assert not _log_spawn_results([("default", result)])

    messages = [r.getMessage() for r in caplog.records]
    assert messages == [
        "kanban dispatcher tick [default]: workspace_refused=2 "
        "(workspaces_root_unmounted: t1, t2)"
    ]


def test_workspace_refusal_notifier_delivers_once_per_outage_and_rearms():
    from gateway.kanban_watchers_dispatcher import _WorkspaceRefusalOutageNotifier

    notifier = _WorkspaceRefusalOutageNotifier()
    deliveries = []

    def send(board, summary):
        deliveries.append((board, summary))
        return True

    refused = [("t_missing", "workspaces_root_unmounted: /Volumes/ramscratch")]
    assert notifier.observe("default", [], send) is False
    assert len(deliveries) == 0
    assert notifier.observe("default", refused, send) is True
    assert len(deliveries) == 1
    assert notifier.observe("default", refused, send) is False
    assert len(deliveries) == 1
    assert notifier.observe("default", [], send) is False
    assert notifier.observe("default", refused, send) is True
    assert len(deliveries) == 2


def test_workspace_refusal_notifier_retries_until_delivery_succeeds():
    from gateway.kanban_watchers_dispatcher import _WorkspaceRefusalOutageNotifier

    notifier = _WorkspaceRefusalOutageNotifier()
    outcomes = iter([False, True])
    attempts = []

    def send(board, summary):
        attempts.append((board, summary))
        return next(outcomes)

    refused = [("t_missing", "workspaces_root_unmounted: /Volumes/ramscratch")]
    assert notifier.observe("default", refused, send) is False
    assert notifier.observe("default", refused, send) is True
    assert len(attempts) == 2


def test_workspace_refusal_tick_observer_uses_delivery_latch(monkeypatch):
    import gateway.kanban_watchers_dispatcher as kwd
    from gateway.kanban_watchers_dispatcher import (
        _WorkspaceRefusalOutageNotifier,
        _observe_workspace_refusal_outages,
    )
    from hermes_cli.kanban_db_dispatch import DispatchResult

    deliveries = []

    def send(board, summary):
        deliveries.append((board, summary))
        return True

    monkeypatch.setattr(kwd, "_send_workspace_refusal_alert", send)
    notifier = _WorkspaceRefusalOutageNotifier()
    refused = DispatchResult(workspace_refused=[
        ("t_missing", "workspaces_root_unmounted: /Volumes/ramscratch"),
    ])
    healthy = DispatchResult()

    assert _observe_workspace_refusal_outages(notifier, [("default", refused)]) == 1
    assert _observe_workspace_refusal_outages(notifier, [("default", refused)]) == 0
    assert len(deliveries) == 1
    assert _observe_workspace_refusal_outages(notifier, [("default", healthy)]) == 0
    assert _observe_workspace_refusal_outages(notifier, [("default", refused)]) == 1
    assert len(deliveries) == 2


def test_workspace_refusal_sender_uses_default_profile_error_route(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace

    import gateway.kanban_watchers_dispatcher as kwd

    script = tmp_path / ".hermes" / "scripts" / "notify.py"
    script.parent.mkdir(parents=True)
    script.write_text("")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(kwd.subprocess, "run", run)
    assert kwd._send_workspace_refusal_alert("default", "workspace_refused=1")
    argv, kwargs = calls[0]
    assert argv[argv.index("--profile") + 1] == "default"
    assert argv[argv.index("--sev") + 1] == "error"
    assert kwargs["stdin"] is subprocess.DEVNULL
