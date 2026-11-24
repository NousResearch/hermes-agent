"""Shared helpers for the real-process Windows hand-off protocol tests (not collected)."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = ROOT / 'scripts/desktop-update/windows.ps1'
MARKER_PS1 = ROOT / 'scripts/desktop-update/marker.ps1'
MARKER = '.hermes-update-in-progress'
POWERSHELL = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'),
                          'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
HOLD_CLI = """
import os, sys, time
from pathlib import Path
def main():
    if '--version' in sys.argv:
        print('Install directory: ' + str(Path(__file__).resolve().parents[1])); return 0
    if '--help' in sys.argv:
        print('--keep-stash'); return 0
    if sys.argv[1:2] == ['update']:   # the update's first act: say so, then wait to be released
        hold = Path(os.environ['HANDOFF_HOLD'])
        Path(str(hold) + '.pid').write_text(str(os.getpid()), encoding='utf-8')
        while not hold.exists():
            time.sleep(0.05)
    return 0
if __name__ == '__main__':
    sys.exit(main())
"""


def _ps(command: str, timeout: int = 60) -> str:
    return subprocess.run([POWERSHELL, '-NoProfile', '-Command', command],
                          capture_output=True, text=True, timeout=timeout, check=True).stdout.strip()


def _creation_time(pid: int) -> str:
    out = _ps(f"$c = (Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CreationDate; "
              "[DateTimeOffset]::new($c.ToUniversalTime()).ToUnixTimeMilliseconds().ToString()")
    return f'{int(out) / 1000:.3f}'


class _ProcessWatch:
    """A SYNCHRONIZE handle on a process, opened while it lives. The handle pins the pid (Windows
    cannot hand it to a new process while it is open) and is signalled exactly when the process
    has exited, so a death that lands asynchronously (a job's kill-on-close) is awaited, not
    guessed at with a sleep and a pid probe that a recycled pid or a lingering entry fools."""

    def __init__(self, pid: int) -> None:
        import ctypes
        from ctypes import wintypes
        self._k = ctypes.WinDLL('kernel32', use_last_error=True)
        self._k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._k.OpenProcess.restype = wintypes.HANDLE
        self._k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._k.WaitForSingleObject.restype = wintypes.DWORD
        self._k.CloseHandle.argtypes = [wintypes.HANDLE]
        self._handle = self._k.OpenProcess(0x00100000 | 0x1000, False, pid)   # SYNCHRONIZE | QUERY_LIMITED
        assert self._handle, f'could not open pid {pid} (winerror={ctypes.get_last_error()})'

    def exited_within(self, seconds: float) -> bool:
        return self._k.WaitForSingleObject(self._handle, int(seconds * 1000)) == 0   # WAIT_OBJECT_0

    def close(self) -> None:
        self._k.CloseHandle(self._handle)


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])
    proc.wait()
    return proc.pid


def _env(home: Path, **extra: str) -> dict[str, str]:
    return {**os.environ, 'HERMES_HOME': str(home), 'HERMES_RUNTIME_DIR': str(home / 'empty-store'), **extra}


def _script(home: Path, *args: str, install: Path | None = None, **env: str) -> subprocess.Popen:
    return subprocess.Popen(
        [POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
         '-InstallRoot', str(install or home / 'hermes-agent'), '-NoUi', *args],
        cwd=home, env=_env(home, **env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _finish(proc: subprocess.Popen, timeout: int = 120) -> tuple[int, str]:
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)], capture_output=True, check=False)
        out, _ = proc.communicate()
        pytest.fail(f'hand-off did not finish within {timeout}s: {out}')
    return proc.returncode, out


def _op(home: Path, *args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [POWERSHELL, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', str(SCRIPT),
         '-InstallRoot', str(home / 'hermes-agent'), *args],
        cwd=home, env=_env(home), capture_output=True, text=True, timeout=120, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _log(home: Path) -> str:
    """The hand-off log so far. windows.ps1 appends with Add-Content, which holds the file without
    read sharing for the length of each write: a read landing in that window is a sharing
    violation (PermissionError), not a missing line, so it is retried."""
    path = home / 'logs/desktop-update-handoff.log'
    for _ in range(50):
        try:
            return path.read_text(encoding='utf-8-sig') if path.exists() else ''
        except PermissionError:
            time.sleep(0.1)
    return path.read_text(encoding='utf-8-sig')


def _custodian(home: Path, handoff: int) -> str:
    """The pid windows.ps1 names on line 1 before any update work starts: its custodian, which
    outlives the hand-off so an old Desktop never reads a dead owner ('' until it is named)."""
    found = re.search(rf'names its custodian pid (\d+) \(hand-off pid {handoff}\)', _log(home))
    return found.group(1) if found else ''


class _HeldLock:
    """Hold the A7 sidecar lock from another process (this one): an open handle
    conflicts with the script's FileShare.None open."""

    def __init__(self, home: Path) -> None:
        deadline = time.monotonic() + 30
        while True:
            try:
                self._f = open(home / (MARKER + '.lock'), 'a+b')   # windows-footgun: ok — binary mode
                return
            except PermissionError:   # the script holds it right now
                assert time.monotonic() < deadline, 'never got the marker lock'
                time.sleep(0.02)

    def release(self) -> None:
        self._f.close()


def _wait_for_log(home: Path, needle: str, timeout: int = 90) -> str:
    deadline = time.monotonic() + timeout
    while needle not in _log(home):
        assert time.monotonic() < deadline, f'{needle!r} never logged: {_log(home)}'
        time.sleep(0.1)
    return _log(home)


def _hold_against_writers(path: Path) -> int:
    """Open ``path`` the way a scanner or a .NET reader does: read access, shared for reading only,
    so every append fails with a sharing violation until the handle closes."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    generic_read, file_share_read, open_existing = 0x80000000, 0x1, 3
    handle = kernel32.CreateFileW(str(path), generic_read, file_share_read, None, open_existing, 0, None)
    assert handle not in (None, wintypes.HANDLE(-1).value), ctypes.get_last_error()
    return handle


def _release(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    assert kernel32.CloseHandle(handle)
