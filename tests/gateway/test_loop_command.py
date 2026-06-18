"""Gateway routing test: `/loop` flows through the shared durable loop engine.

The gateway handler is a thin wrapper that reconstructs the command text from
the event and delegates to ``hermes_cli.loops.handle_loop_command`` — the same
narrow waist the CLI uses. We exercise the real handler with a minimal fake
event to prove parity without standing up a full gateway runner.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

from gateway.slash_commands import GatewaySlashCommandsMixin


class _FakeEvent:
    def __init__(self, args: str) -> None:
        self._args = args

    def get_command_args(self) -> str:
        return self._args


def _dispatch(args: str) -> str:
    handler = GatewaySlashCommandsMixin._handle_loop_command
    return asyncio.run(handler(SimpleNamespace(), _FakeEvent(args)))


def test_gateway_loop_routes_start_and_status(tmp_path, monkeypatch):
    # Force profile-global state into the tmp home and run outside any repo.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    started = _dispatch("start Gateway Outcome")
    assert "loop: gateway outcome" in started.lower()

    status = _dispatch("status")
    assert "Loop: Gateway Outcome" in status

    # bare /loop returns usage through the same path
    assert "Usage:" in _dispatch("")


def test_gateway_loop_plan_and_run_share_cli_logic(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    _dispatch("start Shared Logic")
    assert "Wrote PRD scaffold" in _dispatch("plan")
    run_msg = _dispatch("run next")
    assert "Story execution prompt — S1" in run_msg

    loop_dir = tmp_path / "home" / "loops" / "shared-logic"
    run = json.loads((loop_dir / "runs" / "s1" / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "running"
