"""``kanban.default_assignee`` names a profile, and profile ids may be all digits
(``PROFILE_ID_RE``). YAML loads an unquoted ``default_assignee: 2024`` as an int,
so every dispatcher entry point must read it as the profile name it spells.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from gateway.run import GatewayRunner
from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db
from hermes_cli import kanban_db_dispatch as kbd


def _write_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-root"))
    home = os.environ["HERMES_HOME"]
    with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write("kanban:\n  default_assignee: 2024\n  max_in_progress: 3\n"
                 "  dispatch_interval_seconds: 1\n  auto_decompose: false\n")


def test_gateway_dispatcher_ticks_with_numeric_default_assignee(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch)
    runner = object.__new__(GatewayRunner)
    runner._running = True
    seen: dict = {}

    def _dispatch_once(conn, **kwargs):
        seen.update(kwargs)
        runner._running = False
        return kanban_db.DispatchResult()

    async def _no_sleep(_delay):
        return None

    monkeypatch.setattr(kbd, "dispatch_once", _dispatch_once)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=10.0))

    assert seen.get("default_assignee") == "2024"


def test_cli_dispatch_keeps_caps_with_numeric_default_assignee(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch)
    seen: dict = {}
    monkeypatch.setattr(
        kbd, "dispatch_once", lambda conn, **kw: (seen.update(kw), kanban_db.DispatchResult())[1],
    )

    kb_cli._cmd_dispatch(argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False))

    assert seen.get("default_assignee") == "2024"
    assert seen.get("max_in_progress") == 3
