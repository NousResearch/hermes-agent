"""security.kanban_allow_delegated_writes — operator opt-in lifting the delegated-child fence.

Behavior contract: the fence stays fail-closed by default (existing refusal path is
unchanged) and the config knob is the only way to open it; when open, both the CLI
fast-fail guard and the durable kanban_db permission check allow the mutation.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run_hermes(home: Path, *args: str, marker: bool = False) -> "subprocess.CompletedProcess[str]":
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    for name in (
        "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_CLAIM_LOCK",
        "HERMES_KANBAN_BOARD", "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if marker:
        env["HERMES_DELEGATED_CHILD_CONTEXT"] = "1"
    else:
        env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=30,
    )


def _set_knob(home: Path, value: bool) -> None:
    import yaml

    cfg = home / "config.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    with cfg.open("a", encoding="utf-8") as fh:
        yaml.safe_dump({"security": {"kanban_allow_delegated_writes": value}}, fh)


def test_refusal_by_default(tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    created = _run_hermes(home, "kanban", "create", "fence probe", "--json")
    assert created.returncode == 0, created.stderr
    task_id = json.loads(created.stdout)["id"]
    refused = _run_hermes(home, "kanban", "comment", task_id, "must be refused", marker=True)
    assert refused.returncode == 1
    assert "cannot mutate Kanban tasks via the CLI" in refused.stderr


def test_knob_allows_delegated_child_when_true(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    _set_knob(home, True)
    created = _run_hermes(home, "kanban", "create", "opt-in probe", "--json", marker=True)
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["id"].startswith("t_")


def test_knob_default_false_in_defaults():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    security = DEFAULT_CONFIG["security"]
    assert security["kanban_allow_delegated_writes"] is False
