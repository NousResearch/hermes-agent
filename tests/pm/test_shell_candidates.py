"""Windows bash resolution as pure data: Program Files Git beats PATH, and
the WSL / MSIX stubs never win (#116818). Host-independent — the candidate
ladder takes the ``which`` result and env as arguments."""
import pytest

from pm.shell import (
    is_windows_app_alias,
    is_windows_python_store_alias,
    windows_bash_candidates,
)

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
    "path",
    [r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe",
     r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python3.exe",
     "C:/Users/u/AppData/Local/Microsoft/WindowsApps/Python3.11.exe",
     r"C:\WINDOWSAPPS\python.exe"],
)
def test_windows_app_alias_matches_windowsapps_paths(path):
    assert is_windows_app_alias(path) is True


@pytest.mark.parametrize(
    "path",
    [r"C:\Python311\python.exe", r"C:\msys64\usr\bin\bash.exe",
     r"C:\Windows\System32\bash.exe", ""],
)
def test_windows_app_alias_rejects_non_windowsapps_paths(path):
    assert is_windows_app_alias(path) is False


@pytest.mark.parametrize(
    "path",
    [r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe",
     r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python3.exe",
     "C:/Users/u/AppData/Local/Microsoft/WindowsApps/python3.11.exe"],
)
def test_python_store_alias_matches_python_stubs(path):
    assert is_windows_python_store_alias(path) is True


@pytest.mark.parametrize(
    "path",
    [r"C:\Python311\python.exe", r"C:\Python311\python3.exe",
     r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\bash.exe",
     r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\pip.exe"],
)
def test_python_store_alias_rejects_real_interpreters_and_other_aliases(path):
    assert is_windows_python_store_alias(path) is False
