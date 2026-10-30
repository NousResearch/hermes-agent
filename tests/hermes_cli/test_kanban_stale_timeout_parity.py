"""Every dispatch entry point must honour ``kanban.dispatch_stale_timeout_seconds``.

The gateway-embedded dispatcher resolves it (merged-config default 14400),
but the standalone daemon (``hermes kanban daemon --force``) and one-shot
``hermes kanban dispatch`` never passed it to ``dispatch_once`` — so on
those entry points ``detect_stale_running`` ran with its ``0`` default and
a worker wedged before its first heartbeat was never reclaimed, despite
both paths claiming gateway-identical semantics for the sibling
``max_in_progress`` setting.
"""

from __future__ import annotations

import argparse
import contextlib
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_ops


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_configured_stale_timeout_seconds_parsing(monkeypatch):
    import hermes_cli.config as cfgmod

    cases = [
        ({"kanban": {"dispatch_stale_timeout_seconds": 14400}}, 14400),
        ({"kanban": {"dispatch_stale_timeout_seconds": "3600"}}, 3600),
        ({"kanban": {"dispatch_stale_timeout_seconds": 0}}, 0),  # explicit off
        ({"kanban": {"dispatch_stale_timeout_seconds": "soon"}}, 0),
        ({"kanban": {"dispatch_stale_timeout_seconds": -5}}, 0),
        ({"kanban": {}}, 0),
        ({}, 0),
    ]
    for config, expected in cases:
        monkeypatch.setattr(
            cfgmod, "load_config_readonly", lambda c=config: c
        )
        assert kbd.configured_stale_timeout_seconds() == expected, config


def test_run_daemon_passes_stale_timeout_from_config(kanban_home, monkeypatch):
    """The standalone daemon tick must forward the resolved stale timeout,
    re-resolved per tick like ``max_in_progress``."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config_readonly",
        lambda: {"kanban": {"dispatch_stale_timeout_seconds": 14400}})

    captured: dict = {}
    stop = threading.Event()

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kb.DispatchResult()

    monkeypatch.setattr(kbd, "dispatch_once", fake_dispatch_once)
    monkeypatch.setattr(kbd, "configured_max_in_progress", lambda: None)
    monkeypatch.setattr(kbd, "derive_default_max_in_progress", lambda sample=None: 3)

    kbd.run_daemon(interval=0.01, stop_event=stop, on_tick=lambda res: stop.set())

    assert captured.get("stale_timeout_seconds") == 14400


def test_cli_dispatch_passes_stale_timeout_from_config(kanban_home, monkeypatch):
    """`hermes kanban dispatch` honours the same config key as the gateway."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"dispatch_stale_timeout_seconds": 7200}})

    captured: dict = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kb.DispatchResult()

    monkeypatch.setattr(kbd, "dispatch_once", fake_dispatch_once)
    monkeypatch.setattr(
        kanban_ops.kbc, "connect_closing",
        lambda *a, **k: contextlib.nullcontext(object()))

    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kanban_ops._cmd_dispatch(args)

    assert captured.get("stale_timeout_seconds") == 7200
