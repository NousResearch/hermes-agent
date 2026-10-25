"""The canonical shell must run tests and propagate a failing test's status."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest


@pytest.mark.platforms("posix")
def test_shell_runner_executes_tests_and_propagates_failure(tmp_path):
    case = tmp_path / "test_runner_canary.py"
    marker = tmp_path / "executed"
    case.write_text(
        "from pathlib import Path\nimport os\n"
        "def test_canary():\n"
        f"    Path({str(marker)!r}).write_text('executed')\n"
        "    assert os.environ['PATHEXT'] == '.COM;.EXE;.BAT;.CMD'\n"
        "    assert False, 'runner failure propagation canary'\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", str(root / "scripts/run_tests.sh"), "-j", "1", str(case)],
        cwd=tmp_path, capture_output=True, text=True, timeout=180,
        env={**os.environ, "HERMES_PYTHON": sys.executable, "HERMES_TEST_FILE_RETRIES": "0",
             "PATHEXT": ".COM;.EXE;.BAT;.CMD"},
             check=False,
    )
    assert marker.read_text(encoding="utf-8") == "executed", result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert "runner failure propagation canary" in result.stdout + result.stderr


@pytest.mark.platforms("windows")
def test_tests_never_get_the_windows_directory_as_their_temp_dir(tmp_path):
    """Git for Windows' ``/tmp`` is a ``usertemp`` mount that the first MSYS process of a
    session resolves once, with GetTempPathW. A session begun by a process without TMP,
    TEMP and USERPROFILE maps it to the Windows directory, which the runner handed to every
    test and every native child (GetTempPath2W, Rust's ``temp_dir()``): a user cannot write
    there, an administrator litters it. A shell whose TEMP/TMP name the Windows directory
    delivers that same value without having to start a fresh MSYS session."""
    # Imported here so the POSIX canary above, the runner check the Termux lane runs,
    # keeps importing nothing beyond the stdlib and pytest.
    from tests.pm.activation_support import bash, child_env, posix

    report = tmp_path / "seen.json"
    case = tmp_path / "test_temp_probe.py"
    case.write_text(
        "import ctypes, json, os, tempfile\n"
        "def test_probe():\n"
        "    native = ctypes.create_unicode_buffer(32768)\n"
        "    ctypes.windll.kernel32.GetTempPathW(len(native), native)\n"
        "    seen = {'TEMP': os.environ.get('TEMP', ''), 'TMP': os.environ.get('TMP', ''),\n"
        "            'GetTempPathW': native.value}\n"
        "    try:\n"
        "        tempfile.TemporaryFile(dir=native.value).close()\n"
        "        seen['writable'] = True\n"
        "    except OSError:\n"
        "        seen['writable'] = False\n"
        f"    with open({str(report)!r}, 'w', encoding='utf-8') as out:\n"
        "        json.dump(seen, out)\n",
        encoding="utf-8",
    )
    env = child_env()
    env.pop("__HERMES_ACTIVATED", None)
    windows_dir = env.get("SYSTEMROOT") or env["SystemRoot"]
    env.update({"HERMES_PYTHON": sys.executable, "HERMES_TEST_FILE_RETRIES": "0",
                "TEMP": windows_dir, "TMP": windows_dir})
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [bash(), posix(root / "scripts" / "run_tests.sh"), "-j", "1", posix(case)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300, env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    seen = json.loads(report.read_text(encoding="utf-8"))
    windows = os.path.normcase(windows_dir.rstrip("\\"))
    for name in ("TEMP", "TMP", "GetTempPathW"):
        assert os.path.normcase(seen[name].rstrip("\\")) != windows, seen
    assert seen["writable"], seen
