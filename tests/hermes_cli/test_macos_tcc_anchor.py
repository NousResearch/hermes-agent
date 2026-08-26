"""Tests for the macOS TCC anchor (issue #85345).

The anchor makes the TCC client path stable by replacing the venv's
``bin/python`` symlink (which resolves into uv's versioned store) with a
real-file copy of the interpreter.  All tests run on Linux against fake
checkout/uv-store layouts; ``platform.system`` is monkeypatched to simulate
macOS.
"""

import os
from pathlib import Path

import pytest

import hermes_cli.doctor as doctor
import hermes_cli.macos_tcc_anchor as tcc
from hermes_constants import venv_python_path

_STORE_ROOT = "cpython-3.11.15-macos-aarch64-none"


def _darwin(monkeypatch):
    monkeypatch.setattr(tcc.platform, "system", lambda: "Darwin")


def _linux(monkeypatch):
    monkeypatch.setattr(tcc.platform, "system", lambda: "Linux")


def _build_store(tmp_path, version: str = "3.11.15") -> Path:
    store = (
        tmp_path
        / "uv-store"
        / "uv"
        / "python"
        / f"cpython-{version}-macos-aarch64-none"
    )
    store_bin = store / "bin"
    store_bin.mkdir(parents=True)
    store_py = store_bin / "python3.11"
    store_py.write_bytes(f"#!fake interpreter {version}".encode())
    store_py.chmod(0o755)
    return store_bin


def _build_checkout(
    tmp_path,
    *,
    store_bin: Path | None = None,
    version: str = "3.11.15",
    anchored: bool = False,
    homebrew: bool = False,
) -> Path:
    root = tmp_path / "checkout"
    venv = root / ".venv"
    venv_bin = venv / "bin"
    venv_bin.mkdir(parents=True)
    if homebrew:
        brew = tmp_path / "opt" / "homebrew" / "bin"
        brew.mkdir(parents=True)
        brew_py = brew / "python3.14"
        brew_py.write_bytes(b"#!homebrew")
        brew_py.chmod(0o755)
        (venv / "pyvenv.cfg").write_text(f"home = {brew}\n")
        os.symlink(brew_py, venv_bin / "python")
        os.symlink(brew_py, venv_bin / "python3")
        return root
    if store_bin is None:
        store_bin = _build_store(tmp_path, version)
    (venv / "pyvenv.cfg").write_text(f"home = {store_bin}\n")
    store_py = store_bin / "python3.11"
    if anchored:
        venv_py = venv_bin / "python"
        venv_py.write_bytes(store_py.read_bytes())
        venv_py.chmod(0o755)
        (venv_bin / ".tcc-anchor-source").write_text(str(store_py), encoding="utf-8")
        os.symlink(venv_py, venv_bin / "python3")
    else:
        os.symlink(store_py, venv_bin / "python")
        os.symlink(store_py, venv_bin / "python3")
    return root


class TestUvStoreDetection:
    def test_matches_uv_macos_store_path(self):
        path = (
            "/Users/u/.local/share/uv/python/"
            "cpython-3.11.15-macos-aarch64-none/bin/python3.11"
        )
        assert tcc._is_uv_macos_store(path)

    def test_matches_hermes_runtime_repair_generation(self):
        # repair_vulnerable_runtime() rebuilds the venv against a generation
        # store under .hermes-runtime/python/ — no /uv/python/ segment. The
        # anchor must recognize it or every SQLite CVE repair silently
        # un-anchors the interpreter (issue #82427 integration).
        path = (
            "/Users/u/hermes-agent/.hermes-runtime/python/"
            "generation-a1b2c3/cpython-3.11.15-macos-aarch64-none/bin/python3.11"
        )
        assert tcc._is_uv_macos_store(path)

    def test_rejects_homebrew_interpreter(self):
        path = (
            "/opt/homebrew/Cellar/python@3.14/3.14.6/Frameworks/"
            "Python.framework/Versions/3.14/bin/python3.14"
        )
        assert not tcc._is_uv_macos_store(path)

    def test_rejects_linux_interpreter(self):
        assert not tcc._is_uv_macos_store("/usr/bin/python3")

    def test_rejects_uv_store_on_linux(self):
        path = (
            "/home/u/.local/share/uv/python/"
            "cpython-3.11.15-x86_64-unknown-linux-gnu/bin/python3.11"
        )
        assert not tcc._is_uv_macos_store(path)


class TestEnsureTccAnchor:
    def test_noop_on_non_macos(self, tmp_path, monkeypatch):
        _linux(monkeypatch)
        root = _build_checkout(tmp_path, store_bin=_build_store(tmp_path))
        venv_py = venv_python_path(root / ".venv")

        assert tcc.ensure_tcc_anchor(root) is None
        assert venv_py.is_symlink()  # untouched

    def test_install_signs_the_anchor_copy(self, tmp_path, monkeypatch):
        """The anchor copy gets identifier-DR signing before going live —
        without it every refresh changes the stored csreq (cdhash-based for
        ad-hoc source builds) and orphans the grant despite the stable path."""
        _darwin(monkeypatch)
        signed = []
        import hermes_cli.managed_uv as managed_uv

        monkeypatch.setattr(
            managed_uv, "_macos_sign_managed_python", lambda p: signed.append(Path(p)) or True
        )
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored is not None
        # Signing ran on the temp copy inside the venv bin dir (pre-rename).
        assert len(signed) == 1
        assert signed[0].parent == anchored.parent

    def test_anchors_repair_generation_interpreter(self, tmp_path, monkeypatch):
        """A venv re-created against a .hermes-runtime CVE-repair generation
        store must anchor too (issue #82427 integration)."""
        _darwin(monkeypatch)
        store = (
            tmp_path
            / "checkout"
            / ".hermes-runtime"
            / "python"
            / "generation-a1b2c3"
            / "cpython-3.11.15-macos-aarch64-none"
        )
        store_bin = store / "bin"
        store_bin.mkdir(parents=True)
        store_py = store_bin / "python3.11"
        store_py.write_bytes(b"#!fake generation interpreter")
        store_py.chmod(0o755)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        venv_py = venv_python_path(root / ".venv")
        assert venv_py.is_symlink()

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored == venv_py
        assert not venv_py.is_symlink()
        assert venv_py.read_bytes() == store_py.read_bytes()

    def test_anchors_uv_managed_interpreter(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        venv_py = venv_python_path(root / ".venv")
        assert venv_py.is_symlink()  # preconditions: uv layout

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored == venv_py
        # The venv interpreter is now a real file, not a symlink into the
        # versioned store — the TCC client path is stable.
        assert venv_py.is_file() and not venv_py.is_symlink()
        assert venv_py.read_bytes() == (store_bin / "python3.11").read_bytes()
        assert os.access(venv_py, os.X_OK)
        # Marker records the store binary the copy came from.
        marker = venv_py.parent / ".tcc-anchor-source"
        assert marker.read_text(encoding="utf-8").strip() == str(
            store_bin / "python3.11"
        )
        # Alias symlinks no longer resolve into the versioned store.
        alias = venv_py.parent / "python3"
        assert not tcc._is_uv_macos_store(str(alias.resolve(strict=False)))

    def test_idempotent(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin, anchored=True)
        venv_py = venv_python_path(root / ".venv")
        marker = venv_py.parent / ".tcc-anchor-source"
        before = marker.read_text(encoding="utf-8")

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored == venv_py
        assert venv_py.is_file() and not venv_py.is_symlink()
        assert marker.read_text(encoding="utf-8") == before

    def test_reanchors_after_patch_bump(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        old_bin = _build_store(tmp_path, version="3.11.15")
        root = _build_checkout(tmp_path, store_bin=old_bin, anchored=True)
        venv_py = venv_python_path(root / ".venv")

        # Simulate `uv sync` bumping 3.11.15 -> 3.11.16: uv re-links the venv
        # interpreter to the new store and rewrites pyvenv.cfg home.
        new_bin = _build_store(tmp_path, version="3.11.16")
        new_py = new_bin / "python3.11"
        venv_py.unlink()
        os.symlink(new_py, venv_py)
        (root / ".venv" / "pyvenv.cfg").write_text(f"home = {new_bin}\n")

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored == venv_py
        assert not venv_py.is_symlink()
        assert venv_py.read_bytes() == new_py.read_bytes()
        marker = venv_py.parent / ".tcc-anchor-source"
        assert marker.read_text(encoding="utf-8").strip() == str(new_py)

    def test_skips_homebrew_interpreter(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        root = _build_checkout(tmp_path, homebrew=True)
        venv_py = venv_python_path(root / ".venv")

        assert tcc.ensure_tcc_anchor(root) is None
        assert venv_py.is_symlink()  # untouched: stable identity already

    def test_no_venv_returns_none(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        assert tcc.ensure_tcc_anchor(tmp_path / "missing") is None

    def test_preserves_stdlib_source_home(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        cfg = root / ".venv" / "pyvenv.cfg"

        tcc.ensure_tcc_anchor(root)

        # pyvenv.cfg still points stdlib at the uv store — the anchor only
        # changes the executable identity, not where the stdlib loads from.
        assert f"home = {store_bin}" in cfg.read_text(encoding="utf-8")


class TestTccAnchorState:
    def test_state_missing_then_active(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)

        status, detail = tcc.tcc_anchor_state(root)
        assert status == "missing"
        assert str(venv_python_path(root / ".venv")) in detail

        tcc.ensure_tcc_anchor(root)

        status, detail = tcc.tcc_anchor_state(root)
        assert status == "active"

    def test_state_skip_on_linux(self, tmp_path, monkeypatch):
        _linux(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        status, detail = tcc.tcc_anchor_state(root)
        assert status == "skip"
        assert detail == "not macOS"

    def test_state_skip_for_homebrew(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        root = _build_checkout(tmp_path, homebrew=True)
        status, detail = tcc.tcc_anchor_state(root)
        assert status == "skip"
        assert "not uv-managed" in detail

    def test_state_stale_after_patch_bump(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        old_bin = _build_store(tmp_path, version="3.11.15")
        root = _build_checkout(tmp_path, store_bin=old_bin, anchored=True)
        # Simulate a patch bump where pyvenv.cfg now points at a new store
        # while the venv still holds the previous anchor copy.
        new_bin = _build_store(tmp_path, version="3.11.16")
        (root / ".venv" / "pyvenv.cfg").write_text(f"home = {new_bin}\n")
        status, _ = tcc.tcc_anchor_state(root)
        assert status == "stale"
        # ensure_tcc_anchor() refreshes the copy from the new interpreter.
        anchored = tcc.ensure_tcc_anchor(root)
        assert anchored == venv_python_path(root / ".venv")
        assert (root / ".venv" / "bin" / "python").read_bytes() == (
            new_bin / "python3.11"
        ).read_bytes()
        status, _ = tcc.tcc_anchor_state(root)
        assert status == "active"


class TestDoctorCheck:
    def test_missing_warns_without_fix(self, monkeypatch, capsys):
        monkeypatch.setattr(
            tcc, "tcc_anchor_state", lambda *a, **k: ("missing", "/x/.venv/bin/python")
        )
        doctor.check_macos_tcc_anchor(should_fix=False)
        out = capsys.readouterr().out
        assert "macOS TCC anchor missing" in out

    def test_fix_installs_anchor(self, monkeypatch, capsys):
        monkeypatch.setattr(
            tcc, "tcc_anchor_state", lambda *a, **k: ("missing", "/x/.venv/bin/python")
        )
        monkeypatch.setattr(
            tcc, "ensure_tcc_anchor", lambda *a, **k: Path("/x/.venv/bin/python")
        )
        doctor.check_macos_tcc_anchor(should_fix=True)
        out = capsys.readouterr().out
        assert "macOS TCC anchor installed" in out

    def test_active_reports_ok(self, monkeypatch, capsys):
        monkeypatch.setattr(
            tcc, "tcc_anchor_state", lambda *a, **k: ("active", "/x/.venv/bin/python")
        )
        doctor.check_macos_tcc_anchor(should_fix=False)
        out = capsys.readouterr().out
        assert "macOS TCC anchor active" in out

    def test_skip_is_silent_on_non_macos(self, monkeypatch, capsys):
        monkeypatch.setattr(
            tcc, "tcc_anchor_state", lambda *a, **k: ("skip", "not macOS")
        )
        doctor.check_macos_tcc_anchor(should_fix=False)
        assert capsys.readouterr().out == ""

    def test_never_crashes_on_exception(self, monkeypatch, capsys):
        def boom(*a, **k):
            raise RuntimeError("tccd down")

        monkeypatch.setattr(tcc, "tcc_anchor_state", boom)
        doctor.check_macos_tcc_anchor(should_fix=False)  # must not raise
        out = capsys.readouterr().out
        assert "macOS TCC anchor check failed" in out
