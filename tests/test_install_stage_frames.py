"""The real Bash stage dispatcher emits one JSON result for every exit."""
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install.sh"


def _env(tmp_path):
    return dict(os.environ, HOME=tmp_path.as_posix(),
                HERMES_HOME=(tmp_path / "home").as_posix(),
                HERMES_INSTALL_DIR=(tmp_path / "install").as_posix())


def _frame(result):
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(lines) == 1, result.stdout + result.stderr
    return json.loads(lines[0])


@pytest.mark.parametrize("case", ["explicit-failure", "write-failure", "unknown", "skipped", "success"])
def test_stage_result_matches_the_actual_exit(tmp_path, case):
    env = _env(tmp_path)
    stage = {"explicit-failure": "repository", "write-failure": "complete",
             "unknown": 'unknown"\\\n\x1fstage', "skipped": "setup", "success": "node-deps"}[case]
    env["PROBE_STAGE"] = stage
    if case == "explicit-failure":
        repo = tmp_path / "install"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(tmp_path / "missing")],
                       check=True, capture_output=True)
    elif case == "write-failure":
        (tmp_path / "install").write_text("not a directory", encoding="utf-8")
    script = '''source "$1" --manifest
JSON=true
STAGE="$PROBE_STAGE"
NON_INTERACTIVE=true
INSTALL_COMMIT=fixture
run_stage "$STAGE"
'''
    result = subprocess.run(["bash", "-c", script, "stage-test", SCRIPT.as_posix()],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    frame = _frame(result)
    assert frame["stage"] == stage
    assert frame["ok"] is (result.returncode == 0)
    assert frame["skipped"] is (case == "skipped")
    if case in ("skipped", "success"):
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert frame["reason"]
    if case == "write-failure":
        assert "install complete" not in result.stdout


@pytest.mark.platforms("windows", "posix")
def test_real_single_stage_cli_reports_admission_or_execution_failure(tmp_path):
    stage = 'unknown"\\\n\x1fstage'
    env = dict(_env(tmp_path), PROBE_STAGE=stage)
    result = subprocess.run(["bash", "-c", 'exec bash "$1" --stage "$PROBE_STAGE" --json',
                             "stage-test", SCRIPT.as_posix()],
                            cwd=tmp_path, env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode != 0
    frame = _frame(result)
    assert frame["stage"] == stage and frame["ok"] is False
    assert frame["skipped"] is False and frame["reason"]


@pytest.mark.parametrize("flag", ["--hermes-home", "-HermesHome"])
def test_manifest_accepts_the_desktop_home_argument(tmp_path, flag):
    home = tmp_path / "custom home"
    result = subprocess.run(["bash", str(SCRIPT), "--manifest", flag, str(home)],
                            cwd=tmp_path, env=_env(tmp_path), capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert any(row["name"] == "path" for row in manifest["stages"])
    assert not home.exists()
    env = dict(_env(tmp_path), PROBE_FLAG=flag, PROBE_HOME=home.as_posix())
    env.pop("HERMES_INSTALL_DIR")
    script = '''source "$1" --manifest "$PROBE_FLAG" "$PROBE_HOME"
printf '%s\\n' "$INSTALL_DIR"
bash -c 'printf "%s\\n" "$HERMES_HOME"'
'''
    resolved = subprocess.run(["bash", "-c", script, "home-test", SCRIPT.as_posix()],
                              cwd=tmp_path, env=env, capture_output=True,
                              text=True, encoding="utf-8", timeout=30)
    assert resolved.returncode == 0, resolved.stderr
    assert resolved.stdout.splitlines() == [(home / "hermes-agent").as_posix(), home.as_posix()]
