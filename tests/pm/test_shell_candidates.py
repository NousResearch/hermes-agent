"""Windows bash resolution as pure data: Program Files Git beats PATH, and
the WSL / MSIX stubs never win (#116818). Host-independent — the candidate
ladder takes the ``which`` result and env as arguments."""
import pytest

from pm.shell import is_windows_app_alias, windows_bash_candidates

PF = r"D:\Progs"


@pytest.mark.parametrize(
    "stub",
    [r"C:\Windows\System32\bash.exe", r"C:\WINDOWS\system32\bash.exe",
     r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\bash.exe"],
)
def test_system_stub_bash_is_never_a_candidate(stub):
    assert stub not in windows_bash_candidates(stub, {"ProgramFiles": PF})


def test_program_files_git_precedes_the_path_bash():
    on_path = r"C:\msys64\usr\bin\bash.exe"
    candidates = windows_bash_candidates(on_path, {"ProgramFiles": PF})
    assert candidates[0] == PF + r"\Git\bin\bash.exe"
    assert candidates[-1] == on_path

def test_per_user_and_32bit_git_roots_are_candidates():
    candidates = windows_bash_candidates(None, {
        "ProgramFiles": PF, "ProgramFiles(x86)": r"D:\Progs32",
        "LOCALAPPDATA": r"C:\Users\u\AppData\Local",
    })
    assert r"D:\Progs32\Git\bin\bash.exe" in candidates
    assert r"C:\Users\u\AppData\Local\Programs\Git\bin\bash.exe" in candidates
    assert r"C:\Users\u\AppData\Local\hermes\git\usr\bin\bash.exe" in candidates

def test_nonstarting_bash_is_rejected(monkeypatch):
    import subprocess
    from pm import shell

    monkeypatch.setattr(shell.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1))
    assert shell._bash_starts("broken-bash.exe") is False


@pytest.mark.parametrize(
    "stub",
    [r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe",
     r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python3.exe",
     r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\Python3.EXE",
     "/c/Users/u/AppData/Local/Microsoft/WindowsApps/python3",
     "/c/Users/u/AppData/Local/Microsoft/WindowsApps/python3.exe"],
)
def test_windows_app_alias_detects_store_python_stubs(stub, monkeypatch):
    """#129102: python/python3 under WindowsApps are Store aliases, in native
    and MSYS spellings. 0-byte size probe mocked — the real check stats the
    filesystem so a non-zero WindowsApps shim (e.g. PIM) is not flagged."""
    import os

    monkeypatch.setattr(os.path, "getsize", lambda _p: 0)
    assert is_windows_app_alias(stub) is True


def test_windows_app_alias_ignores_nonzero_windowsapps_shim(monkeypatch):
    """#129121: a WindowsApps binary with non-zero size is a real
    interpreter/shim (e.g. Python Install Manager), not a Store stub."""
    import os

    monkeypatch.setattr(os.path, "getsize", lambda _p: 4096)
    assert is_windows_app_alias(
        r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe") is False


def test_windows_app_alias_dead_redirector_counts_as_stub(monkeypatch):
    """#129121: a dead redirector (stat fails, reparse point with 0 size)
    still counts as a stub."""
    import os

    class _Lstat:
        st_size = 0

    def _getsize(_p):
        raise OSError("dead redirector")

    monkeypatch.setattr(os.path, "getsize", _getsize)
    monkeypatch.setattr(os, "lstat", lambda _p: _Lstat())
    assert is_windows_app_alias(
        r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python3.exe") is True


def test_windows_app_alias_unstatable_path_is_not_flagged(monkeypatch):
    """Unstatable with no lstat either (e.g. MSYS spelling on a POSIX host
    with no file present) cannot be proven a stub — do not flag."""
    import os

    def _getsize(_p):
        raise OSError("missing")

    def _lstat(_p):
        raise OSError("missing")

    monkeypatch.setattr(os.path, "getsize", _getsize)
    monkeypatch.setattr(os, "lstat", _lstat)
    assert is_windows_app_alias(
        r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe") is False


@pytest.mark.parametrize(
    "real",
    [None, "",
     r"C:\Python314\python.exe",
     r"C:\Users\u\AppData\Local\hermes\tools\python-3.14.7-win32-x64\python.exe",
     r"C:\msys64\usr\bin\python3",
     "/usr/bin/python3"],
)
def test_windows_app_alias_rejects_real_interpreters(real):
    assert is_windows_app_alias(real) is False


def test_windows_app_alias_needs_a_path_component():
    """A substring is not enough: C:\\mywindowsapps\\python.exe is real."""
    assert is_windows_app_alias(r"C:\mywindowsapps\python.exe") is False
