"""Regression tests for #127446: the dashboard's manual ``POST /dispatch`` nudge
must pass ``kanban.max_in_progress`` / ``max_in_progress_per_profile`` /
``default_assignee`` from config to ``dispatch_once`` with the same semantics as
the CLI dispatch and the gateway 60 s tick (#33488 fixed those two; the
dashboard endpoint was missed and dispatched uncapped)."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest


def _dashboard_plugin_api():
    mod_name = "hermes_dashboard_plugin_kanban_dispatch_passthrough_test"
    if mod_name not in sys.modules:
        plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
        spec = importlib.util.spec_from_file_location(mod_name, plugin_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[mod_name]


@contextmanager
def _fake_board_conn(board=None):
    yield board, object()


def _run_dispatch(monkeypatch, fake_config):
    from hermes_cli import kanban_db

    api = _dashboard_plugin_api()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)
    monkeypatch.setattr(api, "_board_conn", _fake_board_conn)

    captured = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult()

    with patch("hermes_cli.kanban_db_dispatch.dispatch_once", fake_dispatch_once):
        api.dispatch(dry_run=True, max_n=4, board=None)
    return captured


def test_dashboard_dispatch_passes_max_in_progress_from_config(monkeypatch):
    """#127446: the nudge endpoint must honour the configured global concurrency
    cap instead of spawning up to the ``max`` query param regardless of it."""
    captured = _run_dispatch(monkeypatch, {
        "kanban": {
            "max_in_progress": 1,
            "default_assignee": "default",
            "max_in_progress_per_profile": 2,
        }
    })

    assert captured.get("max_in_progress") == 1, (
        f"dashboard /dispatch must pass kanban.max_in_progress from config; "
        f"got {captured.get('max_in_progress')!r}"
    )
    assert captured.get("default_assignee") == "default"
    assert captured.get("max_in_progress_per_profile") == 2
    # The explicit UI knobs keep flowing through unchanged.
    assert captured.get("max_spawn") == 4
    assert captured.get("dry_run") is True


def test_dashboard_dispatch_keeps_query_and_board_defaults(monkeypatch):
    """Unset config leaves the per-profile cap and assignee None (same as the CLI
    path) while the ``max`` query param and board routing stay intact."""
    captured = _run_dispatch(monkeypatch, {"kanban": {}})

    assert "default_assignee" not in captured or captured.get("default_assignee") is None
    assert captured.get("max_in_progress_per_profile") is None
    assert captured.get("max_spawn") == 4
    assert captured.get("dry_run") is True


def test_dashboard_dispatch_survives_config_read_failure(monkeypatch):
    """A broken config must not take the endpoint down (the nudge is a manual
    recovery affordance) and must not uncap the board either: like the gateway
    60 s tick, the fallback resolves the memory-derived default instead of
    inheriting the CLI's uncapped fail-open (review feedback on #127463)."""

    def _boom():
        raise RuntimeError("config unreadable")

    captured = {}
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch as kbd

    api = _dashboard_plugin_api()
    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    monkeypatch.setattr(api, "_board_conn", _fake_board_conn)
    # 2 GiB total: (2 * 1024 // 512) = 4 derived workers — comfortably between
    # DERIVED_MAX_IN_PROGRESS_FLOOR (2) and _CEILING (8).
    monkeypatch.setattr(
        kbd, "_system_memory_sample", lambda: {"mem_total_kib": 2 * 1024 * 1024}
    )

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult()

    with patch("hermes_cli.kanban_db_dispatch.dispatch_once", fake_dispatch_once):
        api.dispatch(dry_run=False, max_n=8, board="proj")

    assert captured.get("max_in_progress") == 4, (
        f"config read failure must fail closed to the memory-derived cap, got "
        f"{captured.get('max_in_progress')!r}"
    )
    assert captured.get("default_assignee") is None
    assert captured.get("max_in_progress_per_profile") is None
    assert captured.get("board") == "proj"
