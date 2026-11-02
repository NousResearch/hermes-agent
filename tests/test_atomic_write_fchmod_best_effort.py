"""The atomic writers must still write when the filesystem refuses fchmod.

Object-storage NFS exports and squashed/NAS mounts pin every file to a fixed owner and mode and
answer chmod/fchmod with EPERM even on a file the caller has just created. ``_atomic_write``
fchmod'd the temp fd unconditionally before the replace, so a plain ``config.yaml`` write raised
PermissionError on such a home while the post-replace chmod was already best-effort.
"""
import json
import os
import sys

import pytest

import utils


@pytest.fixture
def fchmod_refused(monkeypatch):
    calls = []

    def refuse(fd, mode):
        calls.append(mode)
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "fchmod", refuse, raising=False)
    return calls


def test_yaml_write_on_an_existing_file_survives_eperm(tmp_path, fchmod_refused):
    path = tmp_path / "config.yaml"
    path.write_text("model: {default: a}\n", encoding="utf-8")  # existing file -> mode preservation calls fchmod
    utils.atomic_yaml_write(path, {"model": {"default": "b"}})
    assert fchmod_refused, "an existing file still triggers the fchmod attempt"
    assert "default: b" in path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".config_*.tmp")), "no temp file is left behind"


def test_json_write_with_explicit_mode_survives_eperm(tmp_path, fchmod_refused):
    path = tmp_path / "state.json"
    utils.atomic_json_write(path, {"k": 1}, mode=0o600)
    assert fchmod_refused
    assert json.loads(path.read_text(encoding="utf-8")) == {"k": 1}


def test_text_write_preserving_mode_survives_eperm(tmp_path, fchmod_refused):
    path = tmp_path / "notes.md"
    path.write_text("x", encoding="utf-8")
    utils.atomic_write_text(path, "y", preserve_mode=True)
    assert fchmod_refused
    assert path.read_text(encoding="utf-8") == "y"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_mode_is_still_preserved_where_the_filesystem_allows_it(tmp_path):
    """Best-effort must not degrade into "never": on an ordinary filesystem the mode is kept."""
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    path.chmod(0o640)
    utils.atomic_yaml_write(path, {"a": 2})
    assert (path.stat().st_mode & 0o777) == 0o640
