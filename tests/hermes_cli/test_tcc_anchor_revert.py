"""Tests for the TCC-anchor revert heal (#95425 / #95541).

The interpreter anchor replaced venv/bin/python with a real-file copy that
could not load libpython on real Macs, bricking the CLI. The anchor is
reverted; doctor's check_macos_tcc_anchor_removed() restores anchored venvs
to symlinks using the marker the anchor left behind.
"""

import contextlib
import io
import os
from pathlib import Path

import hermes_cli.doctor as doctor_mod


def _capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _build_anchored_checkout(tmp_path):
    """A checkout whose venv the anchor converted: real-file python + marker."""
    root = tmp_path / "checkout"
    store_bin = tmp_path / "store" / "cpython-3.12.1-macos" / "bin"
    store_bin.mkdir(parents=True)
    source = store_bin / "python3.12"
    source.write_bytes(b"#!store interpreter")
    source.chmod(0o755)
    venv_bin = root / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_py = venv_bin / "python"
    venv_py.write_bytes(b"#!anchored copy (broken on real macs)")
    venv_py.chmod(0o755)
    (venv_bin / ".tcc-anchor-source").write_text(str(source), encoding="utf-8")
    os.symlink(venv_py, venv_bin / "python3")
    return root, source, venv_py


def test_silent_on_non_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "linux")
    assert _capture(doctor_mod.check_macos_tcc_anchor_removed) == ""


def test_silent_when_never_anchored(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root = tmp_path / "checkout"
    (root / "venv" / "bin").mkdir(parents=True)
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert out == ""


def test_heals_anchored_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)

    # Point the check's root resolution at the fixture checkout.
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "TCC anchor removed" in out
    assert venv_py.is_symlink()
    assert Path(os.readlink(venv_py)) == source
    assert not (venv_py.parent / ".tcc-anchor-source").exists()
    # Aliases restored to point at bin/python.
    alias = venv_py.parent / "python3"
    assert alias.is_symlink()
    assert os.readlink(alias) == "python"
