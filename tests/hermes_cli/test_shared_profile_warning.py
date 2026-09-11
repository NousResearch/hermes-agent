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


def test_status_surfaces_live_warning_without_host_details(homes, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from hermes_cli import web_server

    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    client = TestClient(web_server.app)
    assert client.get("/api/status").json().get("shared_profile_warning", "") == ""
    with running_install(homes, tmp_path / "canary"):
        response = client.get("/api/status")
        assert response.status_code == 200
        warning = response.json().get("shared_profile_warning", "")
        assert warning
        assert str(homes) not in warning
        assert str(tmp_path / "canary") not in warning
    assert client.get("/api/status").json().get("shared_profile_warning", "") == ""
    work = homes / "profiles" / "work"
    work.mkdir(parents=True)
    (work / "config.yaml").write_text("{}")
    with running_install(work, tmp_path / "canary"):
        assert client.get("/api/status").json().get("shared_profile_warning", "") == ""
        response = client.get("/api/status?profile=work")
        assert response.status_code == 200
        assert response.json().get("shared_profile_warning")


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
