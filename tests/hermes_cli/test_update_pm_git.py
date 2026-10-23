"""Windows installs whose git is PM's: install.ps1 stages it, PM's facts must record it."""
import os

import pm
import pm.paths
import pytest

from hermes_cli import _subprocess_compat as compat
from pm.package import Runner


@pytest.fixture
def windows(monkeypatch, tmp_path):
    store = tmp_path / "tools"
    staged_git = store / "git-2.53.0+3-win32-x64" / "cmd" / "git.exe"
    staged_git.parent.mkdir(parents=True)
    staged_git.touch()
    (tmp_path / "checkout" / ".git").mkdir(parents=True)
    monkeypatch.setattr(compat.sys, "platform", "win32")
    monkeypatch.setattr(pm.paths, "store_root", lambda: store)
    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    return tmp_path, staged_git


@pytest.mark.parametrize("git_on_path", ["none", "installer-staged"])
def test_pm_git_is_acquired_and_recorded_when_windows_has_no_git_of_its_own(windows, monkeypatch, git_on_path):
    root, staged_git = windows
    calls = []
    store_path = r"C:\store\git-2.53.0+3-win32-x64\cmd;C:\store\git-2.53.0+3-win32-x64\usr\bin;C:\Windows\System32"

    def ensure(name, **kwargs):
        calls.append((name, kwargs))
        return Runner(name, {"Path": store_path})

    found = str(staged_git) if git_on_path == "installer-staged" else None
    monkeypatch.setattr(compat.shutil, "which", lambda name, *a, **k: found)
    monkeypatch.setattr(pm, "ensure", ensure)

    compat.expose_pm_git(root / "checkout")

    assert calls == [("git", {"explicit": True})]
    assert os.environ["PATH"] == store_path


@pytest.mark.parametrize("install", ["own-git", "git-less-zip"])
def test_own_git_and_git_less_zip_installs_are_left_alone(windows, monkeypatch, install):
    root, _ = windows
    own_git = str(root / "Git" / "cmd" / "git.exe") if install == "own-git" else None
    if install == "git-less-zip":
        (root / "checkout" / ".git").rmdir()
    monkeypatch.setattr(compat.shutil, "which", lambda name, *a, **k: own_git)
    monkeypatch.setattr(pm, "ensure", lambda *a, **k: pytest.fail("acquired PM git"))

    compat.expose_pm_git(root / "checkout")

    assert os.environ["PATH"] == r"C:\Windows\System32"
