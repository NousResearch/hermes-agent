"""Windows system tools Hermes spawns never run a same-named program planted in the cwd.

``CreateProcess`` looks for a bare ``taskkill``/``powershell`` in the parent's current directory
before System32, and Hermes's cwd is usually a repository someone else controls.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Appends its own name to ran.txt in the cwd, so a planted copy that ran is visible afterwards.
_IMPOSTOR_SOURCE = (
    "public static class P { public static int Main(string[] a) {"
    " System.IO.File.AppendAllText(\"ran.txt\", System.AppDomain.CurrentDomain.FriendlyName + \"\\n\");"
    " return 0; } }"
)


@pytest.mark.platforms("windows")
def test_planted_system_tools_in_cwd_never_run(tmp_path, monkeypatch):
    from gateway.status import _get_process_start_time, terminate_pid
    from hermes_cli.clipboard import write_clipboard_text

    repo = tmp_path / "repo"
    repo.mkdir()
    powershell = Path(os.environ["SystemRoot"]) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    impostor = repo / "impostor.exe"
    subprocess.run(
        [str(powershell), "-NoProfile", "-Command",
         f"Add-Type -TypeDefinition '{_IMPOSTOR_SOURCE}' -OutputAssembly '{impostor}' -OutputType ConsoleApplication"],
        check=True, stdin=subprocess.DEVNULL, timeout=120,
    )
    for name in ("powershell.exe", "taskkill.exe"):
        shutil.copy(impostor, repo / name)
    monkeypatch.chdir(repo)
    # Prove the absolute paths on their own: drop the process-wide switch hermes_bootstrap set.
    for key in [k for k in os.environ if k.upper() == "NODEFAULTCURRENTDIRECTORYINEXEPATH"]:
        monkeypatch.delenv(key)

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
    try:
        write_clipboard_text("hermes")
        terminate_pid(child.pid, force=True, expected_start_time=_get_process_start_time(child.pid))
        ran = repo / "ran.txt"
        assert not ran.exists(), f"planted copies ran: {ran.read_text(encoding='utf-8-sig').split()}"
        child.wait(timeout=30)  # the real taskkill ended it
    finally:
        child.kill()
