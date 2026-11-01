"""Windows bash resolution as pure data: Program Files Git beats PATH, and
the WSL / MSIX stubs never win (#116818). Host-independent — the candidate
ladder takes the ``which`` result and env as arguments."""
import pytest

from pm.shell import windows_bash_candidates

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


@pytest.mark.platforms("posix")
def test_fork_broken_bash_is_rejected_and_override_beats_staged(monkeypatch, tmp_path):
    """Regression for #127561: a bash that runs builtins but cannot fork must
    fail the probe, and an explicit HERMES_GIT_BASH_PATH wins over staged."""
    from pm import shell

    def fake(name, forks):
        path = tmp_path / name
        body = 'exit 0' if forks else 'case "$2" in *\\$\\(*) exit 254;; esac; exit 0'
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
        return str(path)

    broken, good = fake("broken", False), fake("good", True)
    assert shell._bash_starts(broken) is False
    assert shell._bash_starts(good) is True

    monkeypatch.setattr(shell, "_staged_bash", lambda: good)
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", broken)
    assert shell.bash() == good  # broken override is skipped
    override = fake("override", True)
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", override)
    assert shell.bash() == override
