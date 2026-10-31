"""The build swap must also stop a foreign process whose CWD is inside ``release``.

Windows refuses to rename a directory that is any live process's current working
directory. A third-party resident helper (Sogou IME's ``SogouCloud.exe``, an AV or
assistant component) whose exe lives elsewhere can still hold
``release/win-unpacked`` as its CWD — it inherited it at spawn time while a build
was running — and then every stage-and-swap dies with WinError 32 even though the
build itself succeeded and the app's own processes are gone (#101789).

The stop list used to be "processes whose exe lives under release", which cannot
see that helper. These tests pin the CWD arm of the check.
"""

import sys

import pytest

pytestmark = pytest.mark.platforms("windows")


class _FakeProc:
    """Minimal psutil.Process stand-in: info dict, cwd(), terminate()."""

    def __init__(self, pid, exe, cwd, raises_cwd=False):
        self.pid = pid
        self.info = {"pid": pid, "exe": str(exe)}
        self._cwd = str(cwd)
        self._raises_cwd = raises_cwd
        self.terminated = False

    def cwd(self):
        if self._raises_cwd:
            raise OSError("access denied")
        return self._cwd

    def terminate(self):
        self.terminated = True

    def kill(self):  # pragma: no cover - only hit when terminate is ignored
        self.terminated = True


class _FakeSelf:
    def parents(self):
        return []


@pytest.fixture
def release_tree(tmp_path):
    release = tmp_path / "release"
    (release / "win-unpacked").mkdir(parents=True)
    return tmp_path


def _run(monkeypatch, procs):
    import psutil

    from hermes_cli import main_desktop as md

    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: iter(procs))
    monkeypatch.setattr(psutil, "Process", lambda pid: _FakeSelf())
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: ([], []))
    return md._stop_desktop_processes_locking_build


def test_stops_foreign_process_whose_cwd_is_inside_release(monkeypatch, release_tree):
    """SogouCloud.exe lives outside the tree but sits in win-unpacked -> must be stopped."""
    helper = _FakeProc(
        4242,
        exe="C:/Program Files (x86)/SogouInput/16.1/SogouCloud.exe",
        cwd=release_tree / "release" / "win-unpacked",
    )
    stop = _run(monkeypatch, [helper])

    stopped = stop(release_tree)

    assert helper.terminated, "a helper holding release/ as CWD must be stopped"
    assert stopped == [4242]


def test_stops_foreign_process_whose_cwd_is_the_release_dir_itself(monkeypatch, release_tree):
    helper = _FakeProc(4343, exe="C:/Program Files/AV/scanner.exe", cwd=release_tree / "release")
    stop = _run(monkeypatch, [helper])

    assert stop(release_tree) == [4343]
    assert helper.terminated


def test_leaves_foreign_process_alone_when_its_cwd_is_elsewhere(monkeypatch, release_tree):
    """The negative case: an unrelated resident process must not be killed."""
    bystander = _FakeProc(
        5151,
        exe="C:/Program Files (x86)/SogouInput/16.1/SogouCloud.exe",
        cwd="C:/Windows/System32",
    )
    stop = _run(monkeypatch, [bystander])

    assert stop(release_tree) == []
    assert not bystander.terminated


def test_survives_a_process_that_refuses_cwd(monkeypatch, release_tree):
    """Access-denied on cwd() must not abort the scan (fail-open, as the whole helper is)."""
    locked = _FakeProc(
        6161,
        exe="C:/Program Files/x/guard.exe",
        cwd="C:/x",
        raises_cwd=True,
    )
    inside = _FakeProc(6262, exe="C:/Program Files/x/guard2.exe", cwd=release_tree / "release" / "win-unpacked")
    stop = _run(monkeypatch, [locked, inside])

    assert stop(release_tree) == [6262]
    assert not locked.terminated
