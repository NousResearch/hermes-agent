"""Regression test for #55446 — kanban.default_assignee hot-reload.

Verifies that the dispatcher re-reads ``kanban.default_assignee`` from
config on each dispatch tick instead of capturing it once at boot, and
falls back to the boot value when config loading fails.
"""

from __future__ import annotations

import types
from pathlib import Path

import gateway.kanban_watchers_dispatcher as kwd
from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher


class _StubKb:
    """Just enough ``kanban_db`` for fingerprinting: a per-board path."""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp = tmp_path

    def kanban_db_path(self, board=None):
        return self._tmp / f"kanban-{board}.db"


def _make_dispatcher(monkeypatch, tmp_path, *, boot_assignee=None, load_config):
    captured: list = []

    def _dispatch_once(conn, board=None, **kwargs):
        captured.append(kwargs.get("default_assignee"))
        return None

    monkeypatch.setattr(kwd, "_kbc", lambda: types.SimpleNamespace(connect=lambda board=None: None))
    monkeypatch.setattr(kwd, "_kbd", lambda: types.SimpleNamespace(dispatch_once=_dispatch_once))

    settings = _DispatcherSettings(
        interval=1.0,
        max_spawn=None,
        max_in_progress=None,
        failure_limit=3,
        stale_timeout_seconds=0,
        reconcile_orphans=False,
        default_assignee=boot_assignee,
        max_in_progress_per_profile=None,
    )
    dispatcher = _KanbanDispatcher(_StubKb(tmp_path), settings, load_config)
    return dispatcher, captured


def test_dispatcher_rereads_default_assignee_each_tick(monkeypatch, tmp_path):
    """Config changes to default_assignee take effect on the next tick."""
    cfg_state = {"kanban": {"default_assignee": "atlas"}}
    dispatcher, captured = _make_dispatcher(
        monkeypatch, tmp_path, boot_assignee="atlas", load_config=lambda: cfg_state
    )

    dispatcher.tick_once_for_board("default")
    assert captured == ["atlas"]

    # Operator edits config.yaml while the gateway keeps running.
    cfg_state["kanban"]["default_assignee"] = "indigo"
    dispatcher.tick_once_for_board("default")
    assert captured == ["atlas", "indigo"]

    # Clearing the key also takes effect (back to "no fallback").
    cfg_state["kanban"]["default_assignee"] = ""
    dispatcher.tick_once_for_board("default")
    assert captured == ["atlas", "indigo", None]


def test_dispatcher_assignee_fallback_on_config_failure(monkeypatch, tmp_path):
    """A broken config load falls back to the boot-time value, not a crash."""

    def _broken_load_config():
        raise RuntimeError("config.yaml is mid-rewrite")

    dispatcher, captured = _make_dispatcher(
        monkeypatch, tmp_path, boot_assignee="atlas", load_config=_broken_load_config
    )

    dispatcher.tick_once_for_board("default")
    assert captured == ["atlas"]


def test_dispatcher_without_load_config_uses_boot_value(monkeypatch, tmp_path):
    """No loader wired (default): the boot-time settings value is used as-is."""
    settings = _DispatcherSettings(
        interval=1.0,
        max_spawn=None,
        max_in_progress=None,
        failure_limit=3,
        stale_timeout_seconds=0,
        reconcile_orphans=False,
        default_assignee="atlas",
        max_in_progress_per_profile=None,
    )
    captured: list = []
    monkeypatch.setattr(kwd, "_kbc", lambda: types.SimpleNamespace(connect=lambda board=None: None))
    monkeypatch.setattr(
        kwd, "_kbd",
        lambda: types.SimpleNamespace(
            dispatch_once=lambda conn, board=None, **kw: captured.append(kw.get("default_assignee")) or None
        ),
    )
    dispatcher = _KanbanDispatcher(_StubKb(tmp_path), settings)  # no load_config
    assert dispatcher._load_config is None

    dispatcher.tick_once_for_board("default")
    assert captured == ["atlas"]
