"""Live subprocesses, real ledger I/O, and isolated profile homes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_cli import process_identity
from hermes_constants import hermes_home_key


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return root


@contextmanager
def running_install(home: Path, install: Path):
    env = dict(os.environ, HERMES_HOME=str(home))
    for key in ("HERMES_SPAWN", "HERMES_PARENT_PID", "HERMES_PARENT_START_MARKER"):
        env.pop(key, None)
    script = """
import sys
from pathlib import Path
from hermes_cli.process_identity import register_self
assert register_self('serve', project_root=Path(sys.argv[1]))
print('ready', flush=True)
sys.stdin.readline()
"""
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(install)],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        yield child
    finally:
        child.communicate("exit\n", timeout=15)
        assert child.returncode == 0


def test_warning_tracks_live_other_install_in_same_home(homes, tmp_path):
    from hermes_cli.shared_profile_warning import shared_profile_warning

    current = tmp_path / "stable"
    other = tmp_path / "canary"
    assert shared_profile_warning(project_root=current) == ""
    with running_install(homes, other) as child:
        warning = shared_profile_warning(project_root=current)
        assert warning and "profile" in warning.lower()
        assert shared_profile_warning(project_root=other) == ""
        entry = next(e for e in process_identity.ledger_entries(project_root=other) if e["pid"] == child.pid)
        assert entry["hermes_home"] == hermes_home_key(homes)
        assert shared_profile_warning(home=homes / "profiles" / "work", project_root=current) == ""
    # A stale file is not proof of concurrent use.
    assert process_identity._ledger_path().exists()
    assert shared_profile_warning(project_root=current) == ""
    work = homes / "profiles" / "work"
    work.mkdir(parents=True)
    with running_install(work, other):
        assert shared_profile_warning(project_root=current) == ""
        assert shared_profile_warning(home=work, project_root=current)


@contextmanager
def running_cli(home: Path, install: Path):
    """Run the real CLI constructor and entrypoint; stop at terminal interaction."""
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home.parent), USERPROFILE=str(home.parent))
    for key in ("HERMES_SPAWN", "HERMES_PARENT_PID", "HERMES_PARENT_START_MARKER"):
        env.pop(key, None)
    script = """
import sys
from pathlib import Path
import hermes_constants
# Model two installed runtime roots without copying the source tree.
hermes_constants.PROJECT_ROOT = Path(sys.argv[1])
import cli

def terminal_boundary(self):
    print('cli-ready', flush=True)
    sys.stdin.readline()

cli.HermesCLI.run = terminal_boundary
cli.main(model='test-model', provider='openai', api_key='test-only',
         base_url='http://127.0.0.1:1/v1', toolsets='none', ignore_rules=True)
"""
    output = {}
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(install)],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stderr is not None
        for line in child.stdout:
            if line.strip() == "cli-ready":
                break
        else:
            raise AssertionError(child.stderr.read())
        yield child, output
    finally:
        _, output["stderr"] = child.communicate("exit\n", timeout=15)
        assert child.returncode == 0, output["stderr"]


@pytest.mark.parametrize("other_surface", ["serve", "cli"])
def test_cli_entrypoint_registers_and_warns_once_for_live_shared_home(homes, tmp_path, monkeypatch, other_surface):
    from hermes_cli.shared_profile_warning import shared_profile_warning

    stable = tmp_path / "stable"
    canary = tmp_path / "canary"
    start_other = running_install if other_surface == "serve" else running_cli
    with start_other(homes, stable):
        with running_cli(homes, canary) as (child, output):
            entries = process_identity.ledger_entries(project_root=canary, verified_only=True)
            own = [entry for entry in entries if entry["pid"] == child.pid]
            assert len(own) == 1
            assert own[0]["purpose"] == "cli"
            assert own[0]["hermes_home"] == hermes_home_key(homes)
            assert shared_profile_warning(project_root=stable)
            # A CLI record must not make a terminal process an update-owned backend.
            import hermes_constants
            from hermes_cli.update_cmd_windows import _ledger_manual_serve_holders, _ledger_reapable_backend_pids
            from hermes_cli.update_inventory import UpdatePlan, _collect_ledger_runtimes

            with monkeypatch.context() as patcher:
                patcher.setattr(hermes_constants, "PROJECT_ROOT", canary, raising=False)
                assert own == process_identity.ledger_entries(verified_only=True)
                matches = [(child.pid, "python", own[0]["argv"])]
                assert own[0]["purpose"] not in process_identity.REAPABLE_PURPOSES
                assert _ledger_manual_serve_holders(matches) == []
                assert _ledger_reapable_backend_pids(matches) == []
                plan = UpdatePlan()
                _collect_ledger_runtimes(plan, set())
                assert plan.runtimes == []
                assert child.poll() is None
        assert output["stderr"].count("Another Hermes installation") == 1
        assert shared_profile_warning(project_root=stable) == ""

    work = homes / "profiles" / "work"
    work.mkdir(parents=True)
    with running_install(homes, stable), running_cli(work, canary) as (_, output):
        assert shared_profile_warning(project_root=stable) == ""
    assert "Another Hermes installation" not in output["stderr"]


@pytest.mark.parametrize("corrupt", [b"{broken", b"\xff"])
def test_cli_startup_quarantines_corrupt_ledger(homes, tmp_path, corrupt):
    ledger = process_identity._ledger_path()
    ledger.write_bytes(corrupt)
    install = tmp_path / "cli-install"
    with running_cli(homes, install) as (child, _):
        assert ledger.with_suffix(".json.corrupt").read_bytes() == corrupt
        entries = process_identity.ledger_entries(project_root=install, verified_only=True)
        assert [entry["pid"] for entry in entries] == [child.pid]
        assert entries[0]["hermes_home"] == hermes_home_key(homes)


def test_status_surfaces_live_warning_without_host_details(homes, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from hermes_cli import web_server

    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    client = TestClient(web_server.app)
    assert client.get("/api/status").json()["shared_profile_warning"] is False
    with running_cli(homes, tmp_path / "canary"):
        response = client.get("/api/status")
        assert response.status_code == 200
        assert response.json()["shared_profile_warning"] is True
    assert client.get("/api/status").json()["shared_profile_warning"] is False
    work = homes / "profiles" / "work"
    work.mkdir(parents=True)
    (work / "config.yaml").write_text("{}")
    with running_install(work, tmp_path / "canary"):
        assert client.get("/api/status").json()["shared_profile_warning"] is False
        response = client.get("/api/status?profile=work")
        assert response.status_code == 200
        assert response.json()["shared_profile_warning"] is True


def test_warning_rejects_reused_or_unverifiable_process_identity(homes, tmp_path):
    from hermes_cli.shared_profile_warning import shared_profile_warning

    current = tmp_path / "stable"
    with running_install(homes, tmp_path / "canary"):
        ledger = process_identity._ledger_path()
        entries = json.loads(ledger.read_text())
        original = entries[0]["create_time"]
        assert shared_profile_warning(project_root=current)
        for bad_create in (original - 60, original - 0.5, None):
            entries[0]["create_time"] = bad_create
            ledger.write_text(json.dumps(entries))
            assert shared_profile_warning(project_root=current) == ""
        entries[0]["create_time"] = original
        entries[0].pop("hermes_home", None)
        ledger.write_text(json.dumps(entries))
        assert shared_profile_warning(project_root=current) == ""
        entries[0]["hermes_home"] = hermes_home_key(homes / "profiles" / "other")
        ledger.write_text(json.dumps(entries))
        assert shared_profile_warning(project_root=current) == ""
        entries[0]["hermes_home"] = hermes_home_key(homes)
        ledger.write_text(json.dumps(entries))
        assert shared_profile_warning(project_root=current)

        # Even a valid record is not live proof if the process probe is unavailable.
        from unittest.mock import patch
        with patch.object(process_identity, "_pid_alive_matches", return_value=None):
            assert shared_profile_warning(project_root=current) == ""
        assert shared_profile_warning(project_root=current)
        ledger.write_text("{broken")
        assert shared_profile_warning(project_root=current) == ""
        assert ledger.with_suffix(".json.corrupt").exists()
        ledger.write_text(json.dumps(entries))
        assert shared_profile_warning(project_root=current)
        ledger.unlink()
        assert shared_profile_warning(project_root=current) == ""
        assert not ledger.exists()
        assert not (homes / "config.yaml").exists()
        assert not (homes / "state.db").exists()
