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


# ---------------------------------------------------------------------------
# extra_roots — registry InstallPath coverage for custom-drive Git installs
# ---------------------------------------------------------------------------

def test_registry_root_covers_custom_drive_installs():
    # A Git for Windows install on a custom drive (registry InstallPath hit,
    # e.g. D:\git) must yield both layout bash.exe paths with no ProgramFiles.
    candidates = windows_bash_candidates(None, {}, extra_roots=[r"D:\git"])
    assert r"D:\git\bin\bash.exe" in candidates
    assert r"D:\git\usr\bin\bash.exe" in candidates

def test_registry_roots_rank_after_fixed_locations_before_path():
    on_path = r"C:\msys64\usr\bin\bash.exe"
    candidates = windows_bash_candidates(
        on_path,
        {"ProgramFiles": PF, "LOCALAPPDATA": r"C:\Users\u\AppData\Local"},
        extra_roots=[r"D:\git"],
    )
    assert candidates.index(r"C:\Users\u\AppData\Local\Programs\Git\bin\bash.exe") \
        < candidates.index(r"D:\git\bin\bash.exe")
    assert candidates.index(r"D:\git\bin\bash.exe") < candidates.index(on_path)

def test_registry_roots_are_deduplicated():
    candidates = windows_bash_candidates(None, {}, extra_roots=[r"D:\git", r"D:\git"])
    assert candidates.count(r"D:\git\bin\bash.exe") == 1
    assert candidates.count(r"D:\git\usr\bin\bash.exe") == 1

def test_registry_root_trailing_separator_is_normalized():
    candidates = windows_bash_candidates(None, {}, extra_roots=[r"D:\git\\"])
    assert r"D:\git\bin\bash.exe" in candidates
    assert not any("bin\\bin" in c or "\\\\" in c for c in candidates)

def test_override_still_beats_registry_roots():
    # HERMES_GIT_BASH_PATH stays the explicit first choice even when the
    # registry offers another install.
    candidates = windows_bash_candidates(
        None, {"HERMES_GIT_BASH_PATH": r"E:\custom\Git\bin\bash.exe"},
        extra_roots=[r"D:\git"],
    )
    assert candidates[0] == r"E:\custom\Git\bin\bash.exe"
