import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from hermes_cli import factory_db


def test_factory_db_creates_project_with_default_dual_lanes(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    result = factory_db.create_project(
        "Demo Payments",
        repo_path="/tmp/demo",
        base_branch="main",
        summary="Pilot project",
    )

    assert result["project_id"] == "demo-payments"
    assert {lane["methodology"] for lane in result["lanes"]} == {
        "zeus_native",
        "bmad_hybrid",
    }

    snapshot = factory_db.status("demo-payments")
    assert Path(snapshot["db_path"]) == hermes_home / "factory" / "factory.db"
    assert [project["project_id"] for project in snapshot["projects"]] == ["demo-payments"]
    assert {lane["lane_id"] for lane in snapshot["lanes"]} == {
        "demo-payments-zeus",
        "demo-payments-bmad",
    }


def test_factory_db_seeds_expected_agent_roster(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    agents = factory_db.list_agents()

    agent_ids = {agent["agent_id"] for agent in agents}
    assert len(agent_ids) == 12
    assert "factory-orchestrator" in agent_ids
    assert "claude-builder" in agent_ids
    assert "codex-builder" in agent_ids
    assert "openhands-lab" in agent_ids
    assert "quality-reviewer" in agent_ids


def test_factory_cli_project_create_outputs_dual_lanes(tmp_path):
    hermes_home = tmp_path / "hermes"
    env = {
        **os.environ,
        "HERMES_HOME": str(hermes_home),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "factory",
            "project",
            "create",
            "Demo Factory",
            "--project-id",
            "demo-factory",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["project_id"] == "demo-factory"
    assert {lane["lane_id"] for lane in payload["lanes"]} == {
        "demo-factory-zeus",
        "demo-factory-bmad",
    }

    conn = sqlite3.connect(hermes_home / "factory" / "factory.db")
    try:
        count = conn.execute("SELECT count(*) FROM factory_agents").fetchone()[0]
        assert count == 12
    finally:
        conn.close()


def test_factory_cli_status_json(tmp_path):
    hermes_home = tmp_path / "hermes"
    env = {
        **os.environ,
        "HERMES_HOME": str(hermes_home),
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }

    subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "factory", "init"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "factory", "status", "--json"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["projects"] == []
    assert payload["lanes"] == []
    assert payload["tasks"] == []
    assert payload["gates"] == []
