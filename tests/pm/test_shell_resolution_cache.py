"""pm.shell.bash() proves a bash starts once per process, not once per terminal
command, and re-resolves when its inputs change. Host-independent: the staged
bash and the start probe are substituted."""
import pytest

from pm import shell


@pytest.fixture
def staged(tmp_path, monkeypatch):
    bash_exe = tmp_path / "bash.exe"
    bash_exe.write_bytes(b"")
    probes = []
    monkeypatch.setattr(shell, "_resolved", None)
    monkeypatch.setattr(shell, "_staged_bash", lambda: str(bash_exe))
    monkeypatch.setattr(shell, "_bash_starts", lambda c: probes.append(c) or True)
    return bash_exe, probes


def test_repeated_resolution_probes_bash_once(staged):
    bash_exe, probes = staged
    assert [shell.bash() for _ in range(5)] == [str(bash_exe)] * 5
    assert probes == [str(bash_exe)]


def test_changed_environment_or_vanished_bash_re_resolves(staged, monkeypatch):
    bash_exe, probes = staged
    shell.bash()
    monkeypatch.setenv("HERMES_GIT_BASH_PATH", str(bash_exe.parent / "other.exe"))
    shell.bash()
    assert len(probes) == 2

    restaged = bash_exe.parent / "git-new" / "bash.exe"
    restaged.parent.mkdir()
    restaged.write_bytes(b"")
    monkeypatch.setattr(shell, "_staged_bash", lambda: str(restaged))
    bash_exe.unlink()
    assert shell.bash() == str(restaged)
