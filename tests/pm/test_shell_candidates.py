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


def test_broken_bash_is_rejected_without_retry(monkeypatch):
    import subprocess
    from pm import shell

    calls = []

    def broken_run(*a, **kw):
        calls.append(1)
        return subprocess.CompletedProcess(a, 1)

    monkeypatch.setattr(shell.subprocess, "run", broken_run)
    assert shell._bash_starts("broken-bash.exe") is False
    assert len(calls) == 1


def test_transient_probe_timeout_is_retried(monkeypatch):
    import subprocess
    from pm import shell

    calls = []

    def flaky_run(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(a[0], kw.get("timeout") or 0)
        return subprocess.CompletedProcess(a, 0)

    monkeypatch.setattr(shell.subprocess, "run", flaky_run)
    assert shell._bash_starts("slow-bash.exe") is True
    assert len(calls) == 2


def test_persistent_probe_timeout_is_rejected_after_one_retry(monkeypatch):
    import subprocess
    from pm import shell

    calls = []

    def hung_run(*a, **kw):
        calls.append(1)
        raise subprocess.TimeoutExpired(a[0], kw.get("timeout") or 0)

    monkeypatch.setattr(shell.subprocess, "run", hung_run)
    assert shell._bash_starts("hung-bash.exe") is False
    assert len(calls) == 2


def test_probe_budget_is_env_tunable(monkeypatch):
    import subprocess
    from pm import shell

    seen = {}

    def run(*a, **kw):
        seen["timeout"] = kw.get("timeout")
        return subprocess.CompletedProcess(a, 0)

    monkeypatch.setattr(shell.subprocess, "run", run)
    monkeypatch.setenv("HERMES_BASH_PROBE_TIMEOUT", "45")
    assert shell._bash_starts("bash.exe") is True
    assert seen["timeout"] == 45.0

    monkeypatch.setenv("HERMES_BASH_PROBE_TIMEOUT", "not-a-number")
    assert shell._bash_starts("bash.exe") is True
    assert seen["timeout"] == 15.0
