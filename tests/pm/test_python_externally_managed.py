"""PEP 668 marker in staged store Pythons (#129097).

``Python.stage`` writes an ``EXTERNALLY-MANAGED`` file beside every staged
stdlib (found via ``os.py``, so POSIX/Windows/Termux layouts share one code
path) before publication, so the recorded digest covers it and pip refuses to
install into Hermes's interpreter. Trees without a stdlib are left alone.
"""

import pytest

from pm.packages import (
    _EXTERNALLY_MANAGED_MESSAGE,
    _write_externally_managed_markers,
    Python,
)


def _posix_tree(root):
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "python3").write_bytes(b"#!/bin/sh\n")
    stdlib = root / "lib" / "python3.14"
    stdlib.mkdir(parents=True)
    (stdlib / "os.py").write_text("# fixture stdlib\n", encoding="utf-8")
    return stdlib


def _windows_tree(root):
    (root / "python.exe").write_bytes(b"MZ")
    stdlib = root / "Lib"
    stdlib.mkdir(parents=True)
    (stdlib / "os.py").write_text("# fixture stdlib\n", encoding="utf-8")
    return stdlib


class TestMarkerWriter:
    def test_posix_layout(self, tmp_path):
        stdlib = _posix_tree(tmp_path)
        _write_externally_managed_markers(tmp_path)
        marker = stdlib / "EXTERNALLY-MANAGED"
        assert marker.is_file()
        text = marker.read_text(encoding="utf-8")
        assert text.startswith("[externally-managed]")
        assert "pip" in text and "Hermes" in text

    def test_windows_layout(self, tmp_path):
        stdlib = _windows_tree(tmp_path)
        _write_externally_managed_markers(tmp_path)
        assert (stdlib / "EXTERNALLY-MANAGED").is_file()

    def test_bionic_layout(self, tmp_path):
        prefix = tmp_path / "data" / "data" / "com.termux" / "files" / "usr"
        stdlib = prefix / "lib" / "python3.14"
        stdlib.mkdir(parents=True)
        (stdlib / "os.py").write_text("# fixture stdlib\n", encoding="utf-8")
        _write_externally_managed_markers(tmp_path)
        assert (stdlib / "EXTERNALLY-MANAGED").is_file()

    def test_tree_without_stdlib_untouched(self, tmp_path):
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "tool").write_bytes(b"x")
        _write_externally_managed_markers(tmp_path)
        assert list(tmp_path.rglob("EXTERNALLY-MANAGED")) == []

    def test_idempotent(self, tmp_path):
        stdlib = _posix_tree(tmp_path)
        _write_externally_managed_markers(tmp_path)
        first = (stdlib / "EXTERNALLY-MANAGED").read_text(encoding="utf-8")
        _write_externally_managed_markers(tmp_path)
        assert (stdlib / "EXTERNALLY-MANAGED").read_text(encoding="utf-8") == first

    def test_message_constant_shape(self):
        assert _EXTERNALLY_MANAGED_MESSAGE.startswith("[externally-managed]\n")


class TestPythonStage:
    @pytest.mark.parametrize("target,build", [
        ("linux-x64", _posix_tree),
        ("win32-x64", _windows_tree),
    ])
    def test_stage_writes_marker(self, tmp_path, monkeypatch, target, build):
        import pm.packages as packages_mod

        monkeypatch.setattr(
            packages_mod.subprocess, "run",
            lambda *a, **k: pytest.fail("stage must not execute anything"))
        staged = tmp_path / "staged"
        staged.mkdir()
        stdlib = build(staged)
        Python().stage(
            None, staged, "3.14.7+20260901", target)
        marker = stdlib / "EXTERNALLY-MANAGED"
        assert marker.is_file()
        assert marker.read_text(encoding="utf-8").startswith(
            "[externally-managed]")

    def test_stage_without_stdlib_writes_nothing(self, tmp_path):
        staged = tmp_path / "staged"
        (staged / "bin").mkdir(parents=True)
        (staged / "bin" / "python3").write_bytes(b"#!/bin/sh\n")
        Python().stage(None, staged, "3.14.7+20260901", "linux-x64")
        assert list(staged.rglob("EXTERNALLY-MANAGED")) == []
