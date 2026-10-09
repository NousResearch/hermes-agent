"""``is_junction`` must treat paths it cannot stat as not-a-junction, never raise.

A fresh catalog install probes ``HERMES_HOME/plugins/<name>`` before that slot
exists (#135788); ``Path.is_symlink`` answers False for the missing path, so the
junction check may not be the half of the expression that raises. The module's
``os`` reference is swapped (rather than ``os.name`` globally) so a failure here
still reports cleanly on POSIX runners.
"""
import errno
from pathlib import Path
from types import SimpleNamespace

import pytest

import pm.filesystem
from pm.filesystem import is_junction


@pytest.fixture
def windows_os(monkeypatch):
    monkeypatch.setattr(pm.filesystem, "os", SimpleNamespace(name="nt"))


def test_missing_path_is_not_a_junction(windows_os, tmp_path):
    assert is_junction(tmp_path / "agent-log") is False


def test_path_under_a_file_is_not_a_junction(windows_os, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert is_junction(blocker / "child") is False


def test_other_lstat_errors_still_raise(windows_os, monkeypatch, tmp_path):
    def _refuse(self):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(Path, "lstat", _refuse)
    with pytest.raises(OSError):
        is_junction(tmp_path / "anything")


def test_non_windows_never_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(pm.filesystem, "os", SimpleNamespace(name="posix"))

    def _refuse(self):
        raise AssertionError("non-Windows must not stat at all")

    monkeypatch.setattr(Path, "lstat", _refuse)
    assert is_junction(tmp_path / "anything") is False
