"""Windows machines whose only git is the one PM provides (install.ps1 fresh installs)."""
import os

import pm
import pytest

from hermes_cli import _subprocess_compat as compat
from pm.package import Runner


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(compat.sys, "platform", "win32")
    monkeypatch.setenv("PATH", r"C:\Windows\System32")


def test_pm_git_goes_on_path_when_windows_has_none(windows, monkeypatch):
    calls = []
    store_path = r"C:\store\git-2.53.0+3-win32-x64\cmd;C:\store\git-2.53.0+3-win32-x64\usr\bin;C:\Windows\System32"

    def ensure(name, **kwargs):
        calls.append((name, kwargs))
        return Runner(name, {"Path": store_path})

    monkeypatch.setattr(compat.shutil, "which", lambda name, *a, **k: None)
    monkeypatch.setattr(pm, "ensure", ensure)

    compat.expose_pm_git()

    assert calls == [("git", {"explicit": True})]
    assert os.environ["PATH"] == store_path


def test_a_working_windows_git_is_left_alone(windows, monkeypatch):
    monkeypatch.setattr(compat.shutil, "which", lambda name, *a, **k: r"C:\Program Files\Git\cmd\git.exe")
    monkeypatch.setattr(pm, "ensure", lambda *a, **k: pytest.fail("acquired PM git over a working one"))

    compat.expose_pm_git()

    assert os.environ["PATH"] == r"C:\Windows\System32"
