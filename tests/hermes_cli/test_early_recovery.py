"""Startup triggers PM recovery without importing the damaged dependencies.

Real generation rebuilds and pre-activation startup live in tests/pm; these
checks keep marker ownership, retry limits and single-flight behavior intact.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import _early_recovery as er
from pm import recovery

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("prefix", [[], ["-p", "default"], ["--profile=default"]])
def test_bootstrap_and_pm_cli_work_without_site_packages(tmp_path, prefix):
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "home"), "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-S", "-m", "hermes_cli.main", *prefix, "pm", "repair", "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "hermes pm repair" in result.stdout


@pytest.mark.parametrize("marker_name", [".update-incomplete", ".lazy-refresh-incomplete"])
def test_marker_requests_pm_repair_then_clears(tmp_path, monkeypatch, capsys, marker_name):
    root = _project(tmp_path)
    marker = root / marker_name
    marker.write_text("interrupted", encoding="utf-8")
    calls = []
    monkeypatch.setattr(recovery, "repair_dependencies", calls.append)
    assert er.recover_if_needed(root, argv=[]) is True
    assert calls == [root]
    assert not marker.exists()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("body", ["", "not json", '{"attempts": 1}', "started=1\npid=0\n"])
def test_failed_repair_keeps_marker_and_stops_at_retry_limit(tmp_path, monkeypatch, capsys, body):
    root = _project(tmp_path)
    marker = root / ".update-incomplete"
    marker.write_text(body, encoding="utf-8")
    attempts = er._read_marker_attempts(marker)
    calls = []
    def fail(project):
        calls.append(project)
        raise RuntimeError("dependency build failed")
    monkeypatch.setattr(recovery, "repair_dependencies", fail)
    for expected in range(attempts + 1, er._EARLY_CORE_INSTALL_MAX_ATTEMPTS + 1):
        assert er.recover_if_needed(root, argv=[]) is False
        assert er._read_marker_attempts(marker) == expected
        if "pid=" in body:
            assert marker.read_text(encoding="utf-8").startswith(body)
    before = len(calls)
    assert er.recover_if_needed(root, argv=[]) is False
    assert len(calls) == before
    output = capsys.readouterr()
    assert output.out == ""
    assert "hermes pm repair" in output.err


def test_recovery_obeys_live_owner_and_single_flight(tmp_path, monkeypatch):
    root = _project(tmp_path)
    marker = root / ".update-incomplete"
    marker.write_text(f"started=1\npid={os.getpid()}\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(recovery, "repair_dependencies", calls.append)
    assert er.recover_if_needed(root, argv=[]) is False
    assert calls == []
    assert marker.exists()
    marker.write_text("interrupted", encoding="utf-8")
    fd = er._claim_recovery_lock(root)
    assert fd is not None
    try:
        assert er.recover_if_needed(root, argv=[]) is False
        assert calls == []
        assert marker.exists()
    finally:
        os.close(fd)
    assert er.recover_if_needed(root, argv=["update"]) is True
    assert calls == [root]


def test_missing_environment_cannot_write_a_retry_marker_without_lock(tmp_path, monkeypatch):
    from hermes_cli.runtime_paths import install_state_dir, runtime_facts_path
    from pm.lock import Facts

    root = _project(tmp_path)
    state = install_state_dir(root)
    Facts(runtime_facts_path(root)).record_state("venv", "old", [], environment=state / "environments" / "old" / "venv")
    fd = er._claim_recovery_lock(root)
    assert fd is not None
    try:
        assert er.recover_if_needed(root, argv=[]) is False
        assert not (state / ".repair-incomplete").exists()
    finally:
        os.close(fd)


def test_pm_commands_and_healthy_startup_do_not_repair(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setattr(recovery, "repair_dependencies", lambda _: pytest.fail("unexpected install"))
    assert er.recover_if_needed(root, argv=[]) is False
    marker = root / ".update-incomplete"
    marker.write_text("interrupted", encoding="utf-8")
    assert er.recover_if_needed(root, argv=["pm", "repair"]) is False
    assert marker.exists()



def test_pid_liveness_recognizes_current_process():
    assert er._pid_is_running(os.getpid()) is True
    assert er._pid_is_running(0) is False

def test_marker_owner_liveness_uses_recorded_pid(tmp_path, monkeypatch):
    marker = tmp_path / ".update-incomplete"
    marker.write_text("started=1\npid=4321\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        er, "_pid_is_running", lambda pid: seen.append(pid) or True
    )

    assert er._marker_owner_is_live(marker) is True
    assert seen == [4321]

def _project(tmp_path: Path, *, pyproject: bool = True) -> Path:
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    if pyproject:
        (root / "pyproject.toml").write_text(
            '[project]\nname = "x"\ndependencies = [\n'
            '  "ruamel.yaml==0.18.17",\n'
            '  "python-dotenv==1.2.2",\n'
            '  "PyJWT[crypto]==2.13.0",\n'
            "]\n",
            encoding="utf-8",
        )
    return root










