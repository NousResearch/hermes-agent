"""The resident-runtime stamp read tells unreadable from corrupt (#133351)."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

import pm.paths
from pm.package import InstallError
from pm.runtime import _resident_runtime


@pytest.fixture
def stamp(tmp_path, monkeypatch) -> Path:
    project = tmp_path / "install" / "repo"
    project.mkdir(parents=True)
    monkeypatch.setattr(pm.paths, "repo_root", lambda: project)
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    return pm.paths.install_stamp_path(project)


def test_unreadable_stamp_names_the_permission_not_a_reinstall(stamp, monkeypatch):
    stamp.write_text("{}", encoding="utf-8")
    real = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == stamp:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    with pytest.raises(InstallError) as caught:
        _resident_runtime()

    message = str(caught.value)
    assert str(stamp) in message
    assert "not readable" in message
    assert "reinstall" not in message


def test_corrupt_stamp_still_asks_for_a_reinstall(stamp):
    stamp.write_text("{not json", encoding="utf-8")

    with pytest.raises(InstallError, match="invalid package install stamp.*reinstall"):
        _resident_runtime()


def test_missing_stamp_is_not_a_packaged_install(stamp):
    assert not stamp.exists()

    assert _resident_runtime() is None
