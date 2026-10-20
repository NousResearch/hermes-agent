"""Shared native fixtures for Windows marker ownership, parsing, and bridge tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from tests.installation_launcher_fixture import publish_fixture_launcher
from tests.scripts.desktop_update.windows_handoff_support import _creation_time

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = ROOT / 'scripts/desktop-update/windows.ps1'
MARKER = '.hermes-update-in-progress'
CLI = """
import sys
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    return 0
if __name__ == '__main__':
    sys.exit(main())
"""


@pytest.fixture
def sleeper():
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
    yield proc
    proc.kill()
    proc.wait()


def _protected_pid() -> int:
    """A live pid that grants only limited query rights (a SYSTEM/protected service): what
    Get-Process .StartTime needs is denied, what CIM/GetProcessTimes need is granted."""
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    names = ('csrss.exe', 'smss.exe', 'wininit.exe', 'services.exe', 'lsass.exe', 'MsMpEng.exe')
    rows = []
    for process in psutil.process_iter(['pid', 'name'], ad_value=None):
        pid, name = process.info['pid'], process.info['name']
        rows.append(f'{pid} {name}')
        if name not in names:
            continue
        full = k32.OpenProcess(0x0400, False, int(pid))         # PROCESS_QUERY_INFORMATION
        limited = k32.OpenProcess(0x1000, False, int(pid))      # ..._LIMITED_INFORMATION
        for handle in (full, limited):
            if handle:
                k32.CloseHandle(ctypes.c_void_p(handle))
        if limited and not full:
            return int(pid)
    pytest.fail('no process here grants only limited query rights: ' + ', '.join(rows[:40]))


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def _run(home: Path, *args: str, install: Path | None = None, timeout: int = 120):
    command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
               '-InstallRoot', str(install or home / 'hermes-agent'), '-NoUi', *args]
    env = {**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(home / 'empty-store')}
    proc = subprocess.Popen(command, cwd=home, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True, check=False)
        out, _ = proc.communicate()
        pytest.fail(f'hand-off did not finish within {timeout}s: {out}')
    return proc.pid, proc.returncode, out


def _result(home: Path) -> dict:
    return json.loads((home / '.hermes-update-result.json').read_text(encoding='utf-8-sig'))


