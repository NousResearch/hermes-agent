"""Test the desktop dashboard ready-file emits project_root (SPEC 2026-07-01 AC-3 mechanism)."""
import json
import os
from pathlib import Path

import hermes_cli.web_server as ws


def test_ready_file_includes_project_root(tmp_path, monkeypatch):
    ready = tmp_path / "ready.json"
    monkeypatch.setenv("HERMES_DESKTOP_READY_FILE", str(ready))
    ws._write_dashboard_ready_file(4567)
    data = json.loads(ready.read_text())
    assert data["port"] == 4567
    # project_root is the tree the backend runs from — the AC-3 effect-gate signal.
    assert data["project_root"] == str(ws.PROJECT_ROOT)
    assert Path(data["project_root"]).name == "hermes-agent" or Path(data["project_root"]).exists()


def test_ready_file_noop_without_env(tmp_path, monkeypatch):
    # no HERMES_DESKTOP_READY_FILE -> writes nothing, does not raise
    monkeypatch.delenv("HERMES_DESKTOP_READY_FILE", raising=False)
    ws._write_dashboard_ready_file(1234)  # must not raise
