"""Windows Git provisioning tests against the canonical runtime owner and resolver."""
from __future__ import annotations

import os

import pm
import pm.paths
import pytest

from hermes_platform import resolver
from hermes_platform.resolver import Candidate, Resolution
from pm.package import Runner
from runtime import git_subprocess


pytestmark = pytest.mark.platforms("windows")


@pytest.fixture
def windows(monkeypatch, tmp_path):
    store = tmp_path / "tools"
    staged_git = store / "git-2.53.0+3-win32-x64" / "cmd" / "git.exe"
    staged_git.parent.mkdir(parents=True)
    staged_git.touch()
    (tmp_path / "checkout" / ".git").mkdir(parents=True)
    monkeypatch.setattr(pm.paths, "store_root", lambda: store)
    monkeypatch.setenv("PATH", r"C:\Windows\System32")
    return tmp_path, staged_git


def _git_resolution(path):
    """Use the real resolver's .command shape, not the retired shutil.which API."""
    if path is None:
        return Resolution("missing")
    return Resolution("path_executable", (Candidate(str(path), "PATH", True),))


@pytest.mark.parametrize("git_on_path", ["none", "installer-staged"])
def test_pm_git_is_acquired_and_recorded_when_windows_has_no_git_of_its_own(
    windows, monkeypatch, git_on_path,
):
    root, staged_git = windows
    calls = []
    store_path = (
        r"C:\store\git-2.53.0+3-win32-x64\cmd;"
        r"C:\store\git-2.53.0+3-win32-x64\usr\bin;"
        r"C:\Windows\System32"
    )

    def ensure(name, **kwargs):
        calls.append((name, kwargs))
        return Runner(name, {"Path": store_path})

    found = staged_git if git_on_path == "installer-staged" else None

    def locate_command(name):
        assert name == "git"
        return _git_resolution(found)

    monkeypatch.setattr(resolver, "locate_command", locate_command)
    monkeypatch.setattr(pm, "ensure", ensure)

    git_subprocess.expose_pm_git(root / "checkout")

    assert calls == [("git", {"explicit": True})]
    assert os.environ["PATH"] == store_path


@pytest.mark.parametrize("install", ["own-git", "git-less-zip"])
def test_own_git_and_git_less_zip_installs_are_left_alone(windows, monkeypatch, install):
    root, _ = windows
    own_git = root / "Git" / "cmd" / "git.exe" if install == "own-git" else None
    if own_git is not None:
        own_git.parent.mkdir(parents=True)
        own_git.touch()
    else:
        (root / "checkout" / ".git").rmdir()

    def locate_command(name):
        if install == "git-less-zip":
            pytest.fail("resolved Git for an install without a Git checkout")
        assert name == "git"
        return _git_resolution(own_git)

    monkeypatch.setattr(resolver, "locate_command", locate_command)
    monkeypatch.setattr(pm, "ensure", lambda *a, **k: pytest.fail("acquired PM git"))

    git_subprocess.expose_pm_git(root / "checkout")

    assert os.environ["PATH"] == r"C:\Windows\System32"
