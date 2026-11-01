"""Kanban dispatcher tick logging and alerting (PR-added tests; the pre-existing module was purged upstream in 61fa0f47ec)."""


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
