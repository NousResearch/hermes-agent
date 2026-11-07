"""Live Windows probe: cmd.exe decodes the batch launcher's own encoding, so ``gateway.cmd``
under a non-ASCII directory reaches its working directory and environment.

The ``.vbs`` arm of the same bug is covered by ``test_gateway_windows_wsh_live``; cmd.exe has no
BOM to lean on, so the batch file carries ``chcp 65001`` instead.

Runs only on real Windows. It renders the launcher through the production
``_build_gateway_cmd_script`` under a non-ASCII temp home and runs it with ``cmd.exe``; the argv
is swapped for a python one-liner, so no gateway starts and no task or Startup entry is touched.
"""

import subprocess
import sys

import pytest

from hermes_cli import gateway_windows


@pytest.mark.platforms("windows")
def test_gateway_cmd_reaches_a_non_ascii_home(tmp_path, monkeypatch):
    home = tmp_path / "方舟 Usuário" / "hermes"
    home.mkdir(parents=True)
    marker = tmp_path / "ran.txt"
    payload = (
        "import os, pathlib; "
        f"pathlib.Path({str(marker)!r}).write_text("
        "os.environ.get('HERMES_HOME', '') + '|' + os.getcwd(), encoding='utf-8')"
    )
    monkeypatch.setattr(
        gateway_windows,
        "_gateway_run_argv",
        lambda python_exe, profile_arg: [sys.executable, "-c", payload],
    )

    script = gateway_windows._build_gateway_cmd_script(sys.executable, str(home), str(home), "")
    launcher = home / "gateway.cmd"
    launcher.write_bytes(script.encode("utf-8"))
    run = subprocess.run(["cmd.exe", "/c", str(launcher)], capture_output=True, text=True, timeout=180, check=False)

    # cmd.exe fails `cd /d` on the mojibake path, so only a correctly decoded batch file reaches
    # the interpreter at all -- and then both the working directory and HERMES_HOME are the home.
    assert run.returncode == 0, run.stdout + run.stderr
    assert marker.read_text(encoding="utf-8") == f"{home}|{home}"
