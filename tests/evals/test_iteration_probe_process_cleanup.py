"""The status probe waits for revision lookup and captures its output."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock


def test_revision_lookup_is_reaped(monkeypatch, capsys):
    from evals.gateway_status_render import iteration_ceiling_ab as probe

    monkeypatch.setattr(sys, "argv", ["iteration_ceiling_ab.py"])
    monkeypatch.setattr(probe, "_agent", lambda _: SimpleNamespace(max_iterations=250))
    monkeypatch.setattr(probe, "_busy_ack", AsyncMock(return_value="busy"))
    monkeypatch.setattr(probe, "_heartbeat", AsyncMock(return_value="heartbeat"))
    monkeypatch.setattr(probe, "_timeout", lambda _: "timeout")
    # subprocess.run reaps the child and closes captured pipes before returning.
    run = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="abc123\n"))
    monkeypatch.setattr(subprocess, "run", run)

    probe.main()

    run.assert_called_once_with(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=False)
    assert json.loads(capsys.readouterr().out)["head"] == "abc123"
