"""Native Windows runner preserves hermetic execution without an MSYS fork."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _run_launcher(tmp_path: Path, exit_code: int = 0):
    pwsh = shutil.which("pwsh")
    if not pwsh:
        pytest.skip("PowerShell 7 is required")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(Path(__file__).resolve().parents[2] / "scripts" / "run_tests.ps1",
                 scripts / "run_tests.ps1")
    capture = tmp_path / "captured.json"
    (scripts / "run_tests_parallel.py").write_text(
        "import json,os,pathlib,sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
        "'argv':sys.argv[2:],'env':dict(os.environ),'cwd':os.getcwd()}))\n"
        "print('runner-output-receipt')\n"
        "print('runner-error-receipt',file=sys.stderr)\n"
        f"sys.exit({exit_code})\n", encoding="utf-8")
    env = dict(os.environ, HERMES_PYTHON=sys.executable,
               OPENAI_API_KEY="synthetic-not-a-credential", UNRELATED_VALUE="do-not-forward",
               HERMES_HOME=str(tmp_path / "must-not-use-live-home"),
               HERMES_TEST_WORKERS="2", HERMES_TEST_FILE_RETRIES="0")
    result = subprocess.run(
        [pwsh, "-NoProfile", "-File", str(scripts / "run_tests.ps1"), str(capture),
         "argument with spaces", "--file-timeout", "123"],
        env=env, capture_output=True, text=True, timeout=60)
    assert capture.exists(), result.stdout + result.stderr
    return result, json.loads(capture.read_text())


@pytest.mark.windows_only
def test_native_runner_isolates_environment_and_preserves_arguments(tmp_path):
    result, captured = _run_launcher(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'runner-output-receipt' in result.stdout
    assert 'runner-error-receipt' in result.stderr
    assert captured["argv"] == ["argument with spaces", "--file-timeout", "123"]
    assert Path(captured["cwd"]) == tmp_path
    env = captured["env"]
    assert not {"OPENAI_API_KEY", "UNRELATED_VALUE", "HERMES_HOME", "HERMES_PYTHON"} & env.keys()
    assert env["TZ"] == "UTC"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["HERMES_TEST_WORKERS"] == "2"
    assert env["HERMES_TEST_FILE_RETRIES"] == "0"
    assert env["SYSTEMROOT"] == os.environ["SYSTEMROOT"]


@pytest.mark.windows_only
def test_native_runner_preserves_child_failure_exit_code(tmp_path):
    result, _ = _run_launcher(tmp_path, exit_code=37)
    assert result.returncode == 37
