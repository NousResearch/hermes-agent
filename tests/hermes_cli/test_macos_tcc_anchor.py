"""Tests for the dylib-complete macOS TCC anchor (issue #95596).

Re-land of the interpreter anchor reverted in #95563.  The first landing
bricked real Macs two ways: dynamically-linked builds died in dyld because
``@executable_path/../lib/libpython`` resolved into ``venv/lib/`` (#95425),
and alias symlinks to the copied interpreter lost the venv prefix (#95541).

Linux tests use fake checkout/uv-store layouts with ``platform.system``
monkeypatched.  The real-interpreter E2E is ``macos_only`` so it runs on
the existing macOS CI job, not against one-byte fixtures.
"""

from __future__ import annotations

import errno
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_cli.doctor as doctor
import hermes_cli.macos_tcc_anchor as tcc
from hermes_constants import venv_python_path


def _darwin(monkeypatch):
    monkeypatch.setattr(tcc.platform, "system", lambda: "Darwin")


def _linux(monkeypatch):
    monkeypatch.setattr(tcc.platform, "system", lambda: "Linux")


def _build_store(tmp_path, version: str = "3.11.15", *, with_libpython: bool = False) -> Path:
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
    if with_libpython:
        lib = store / "lib"
        lib.mkdir(parents=True)
        (lib / "libpython3.11.dylib").write_bytes(b"fake dylib")
    return store_bin


def _build_checkout(
    tmp_path,
    *,
    store_bin: Path | None = None,
    version: str = "3.11.15",
    anchored: bool = False,
    homebrew: bool = False,
    with_libpython: bool = False,
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
        store_bin = _build_store(tmp_path, version, with_libpython=with_libpython)
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
        assert venv_py.is_symlink()

    def test_install_signs_the_anchor_copy(self, tmp_path, monkeypatch):
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
        assert len(signed) == 1
        assert signed[0].parent == anchored.parent

    def test_anchors_repair_generation_interpreter(self, tmp_path, monkeypatch):
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
        assert venv_py.is_symlink()

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored == venv_py
        assert venv_py.is_file() and not venv_py.is_symlink()
        assert venv_py.read_bytes() == (store_bin / "python3.11").read_bytes()
        assert os.access(venv_py, os.X_OK)
        marker = venv_py.parent / ".tcc-anchor-source"
        assert marker.read_text(encoding="utf-8").strip() == str(
            store_bin / "python3.11"
        )
        alias = venv_py.parent / "python3"
        assert alias.is_file() and not alias.is_symlink()
        assert alias.read_bytes() == venv_py.read_bytes()

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

    def test_repairs_alias_symlinks_left_by_predecessor(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin, anchored=True)
        venv_bin = root / ".venv" / "bin"
        venv_py = venv_bin / "python"
        assert (venv_bin / "python3").is_symlink()

        anchored = tcc.ensure_tcc_anchor(root)

        assert anchored == venv_py
        alias = venv_bin / "python3"
        assert alias.is_file() and not alias.is_symlink()
        assert alias.read_bytes() == venv_py.read_bytes()

    def test_reanchors_after_patch_bump(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        old_bin = _build_store(tmp_path, version="3.11.15")
        root = _build_checkout(tmp_path, store_bin=old_bin, anchored=True)
        venv_py = venv_python_path(root / ".venv")

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
        alias = venv_py.parent / "python3"
        assert alias.is_file() and not alias.is_symlink()
        assert alias.read_bytes() == new_py.read_bytes()

    def test_skips_homebrew_interpreter(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        root = _build_checkout(tmp_path, homebrew=True)
        venv_py = venv_python_path(root / ".venv")

        assert tcc.ensure_tcc_anchor(root) is None
        assert venv_py.is_symlink()

    def test_no_venv_returns_none(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        assert tcc.ensure_tcc_anchor(tmp_path / "missing") is None

    def test_preserves_stdlib_source_home(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        cfg = root / ".venv" / "pyvenv.cfg"

        tcc.ensure_tcc_anchor(root)

        assert f"home = {store_bin}" in cfg.read_text(encoding="utf-8")

    def test_provisions_libpython_as_hardlink_when_present(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path, with_libpython=True)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        src_dylib = store_bin.parent / "lib" / "libpython3.11.dylib"

        tcc.ensure_tcc_anchor(root)

        dst = root / ".venv" / "lib" / "libpython3.11.dylib"
        assert dst.is_file()
        assert dst.read_bytes() == src_dylib.read_bytes()
        assert dst.stat().st_ino == src_dylib.stat().st_ino

    def test_boot_gate_refusal_leaves_venv_untouched(self, tmp_path, monkeypatch):
        _darwin(monkeypatch)
        store_bin = _build_store(tmp_path)
        root = _build_checkout(tmp_path, store_bin=store_bin)
        venv_py = venv_python_path(root / ".venv")
        monkeypatch.setattr(tcc, "_passes_boot_gate", lambda *a, **k: False)

        assert tcc.ensure_tcc_anchor(root) is None
        assert venv_py.is_symlink()
        assert not (venv_py.parent / ".tcc-anchor-source").exists()


class TestBootGate:
    """Direct branch coverage for _passes_boot_gate.

    The gate is the never-brick guarantee, so each refusal direction gets its
    own test rather than only being exercised through the install rollback.
    """

    @staticmethod
    def _proc(argv, returncode, stdout="", stderr=""):
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    def test_refuses_nonzero_exit(self, tmp_path, monkeypatch):
        # dyld / encodings crash in the staged copy → refuse install.
        monkeypatch.setattr(
            tcc.subprocess, "run",
            lambda *a, **k: self._proc(a[0], 1, "", "ModuleNotFoundError: encodings"),
        )
        assert not tcc._passes_boot_gate(tmp_path / "staged", tmp_path / "venv")

    def test_refuses_wrong_prefix(self, tmp_path, monkeypatch):
        # Boots but resolves the build-time prefix (the /install signature).
        monkeypatch.setattr(
            tcc.subprocess, "run",
            lambda *a, **k: self._proc(a[0], 0, "/install\n", ""),
        )
        assert not tcc._passes_boot_gate(tmp_path / "staged", tmp_path / "venv")

    def test_refuses_timeout(self, tmp_path, monkeypatch):
        def hang(*a, **k):
            raise subprocess.TimeoutExpired(cmd=a[0], timeout=30)

        monkeypatch.setattr(tcc.subprocess, "run", hang)
        assert not tcc._passes_boot_gate(tmp_path / "staged", tmp_path / "venv")

    def test_skips_unexecutable_staging(self, tmp_path, monkeypatch):
        # ENOENT/ENOEXEC (fixture bad-shebang binaries, wrong-arch images)
        # mean the binary cannot run here at all — the symlinked venv was
        # equally dead, so installing cannot make things worse.
        def noexec(*a, **k):
            raise OSError(errno.ENOENT, "No such file or directory")

        monkeypatch.setattr(tcc.subprocess, "run", noexec)
        assert tcc._passes_boot_gate(tmp_path / "staged", tmp_path / "venv")

    def test_refuses_eacces_after_chmod(self, tmp_path, monkeypatch):
        # EACCES after copy2+chmod is a broken install about to go live —
        # refuse, never skip (review: kokhlo on #95605).
        def denied(*a, **k):
            raise PermissionError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(tcc.subprocess, "run", denied)
        assert not tcc._passes_boot_gate(tmp_path / "staged", tmp_path / "venv")

    def test_scrubs_pythonhome_pythonpath(self, tmp_path, monkeypatch):
        # A staged copy that dies with `No module named 'encodings'` boots
        # fine under PYTHONHOME=<venv> — the inherited var papers over the
        # exact failure the gate exists to catch (review: kokhlo on #95605,
        # verified live). The probe must run with a scrubbed environment.
        captured = {}

        def spy(*a, **k):
            captured.update(k)
            return self._proc(a[0], 0, f"{tmp_path / 'venv'}\n", "")

        monkeypatch.setattr(tcc.subprocess, "run", spy)
        monkeypatch.setenv("PYTHONHOME", "/poison")
        monkeypatch.setenv("PYTHONPATH", "/poison")
        venv = tmp_path / "venv"
        venv.mkdir(exist_ok=True)

        assert tcc._passes_boot_gate(tmp_path / "staged", venv)
        env = captured.get("env")
        assert env is not None, "gate must pass an explicit scrubbed env"
        assert "PYTHONHOME" not in env
        assert "PYTHONPATH" not in env

    def test_accepts_matching_prefix(self, tmp_path, monkeypatch):
        venv = tmp_path / "venv"
        venv.mkdir()
        monkeypatch.setattr(
            tcc.subprocess, "run",
            lambda *a, **k: self._proc(a[0], 0, f"{venv}\n", ""),
        )
        assert tcc._passes_boot_gate(tmp_path / "staged", venv)


class TestTccAnchorState:
    def test_state_active_through_unpatched_home_symlink(self, tmp_path, monkeypatch):
        # The managed-runtime layout symlinks cpython-3.11-macos-* →
        # cpython-3.11.15-macos-*, so pyvenv.cfg home and the marker record
        # different spellings of the same binary. State must resolve both
        # sides before comparing, or a fresh install reports stale
        # (review: kokhlo on #95605, hit on a live venv).
        _darwin(monkeypatch)
        patched = _build_store(tmp_path, version="3.11.15")
        versionless = patched.parent.parent / "cpython-3.11-macos-aarch64-none"
        os.symlink(patched.parent, versionless)
        home = versionless / "bin"

        root = tmp_path / "checkout"
        venv_bin = root / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (root / ".venv" / "pyvenv.cfg").write_text(f"home = {home}\n")
        os.symlink(home / "python3.11", venv_bin / "python")

        tcc.ensure_tcc_anchor(root)

        status, _ = tcc.tcc_anchor_state(root)
        assert status == "active"

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
        new_bin = _build_store(tmp_path, version="3.11.16")
        (root / ".venv" / "pyvenv.cfg").write_text(f"home = {new_bin}\n")
        status, _ = tcc.tcc_anchor_state(root)
        assert status == "stale"
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
        doctor.check_macos_tcc_anchor(should_fix=False)
        out = capsys.readouterr().out
        assert "macOS TCC anchor check failed" in out


@pytest.mark.macos_only
class TestAnchoredAliasesBootE2E:
    """Real-interpreter proof that the re-land stays bootable (#95596).

    Copies the running interpreter's real base binary into a fake uv-store
    layout (stdlib via a ``lib`` symlink) and actually executes every
    entry point after anchoring.  ``macos_only`` so Linux CI cannot
    greenwash this with a fixture.
    """

    def test_python_entry_points_boot_after_anchor(self, tmp_path):
        minor = f"python3.{sys.version_info.minor}"
        base = Path(sys.base_prefix)
        real_py = base / "bin" / minor
        if not real_py.is_file() or real_py.is_symlink():
            resolved = real_py.resolve() if real_py.exists() else None
            if resolved is None or not resolved.is_file():
                pytest.skip(f"no real base interpreter binary at {real_py}")
            real_py = resolved
        if not (base / "lib" / minor / "os.py").is_file():
            pytest.skip("base stdlib not in the expected lib layout")

        store = (
            tmp_path
            / "uv"
            / "python"
            / f"cpython-{platform.python_version()}-macos-aarch64-none"
        )
        store_bin = store / "bin"
        store_bin.mkdir(parents=True)
        shutil.copy2(real_py, store_bin / minor)
        os.symlink(base / "lib", store / "lib")

        root = tmp_path / "checkout"
        venv = root / ".venv"
        venv_bin = venv / "bin"
        venv_bin.mkdir(parents=True)
        (venv / "lib" / minor / "site-packages").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text(
            f"home = {store_bin}\nversion = {platform.python_version()}\n",
            encoding="utf-8",
        )
        os.symlink(store_bin / minor, venv_bin / "python")
        os.symlink("python", venv_bin / "python3")
        os.symlink("python", venv_bin / minor)

        anchored = tcc.ensure_tcc_anchor(root)
        assert anchored is not None

        for name in ("python", "python3", minor):
            probe = subprocess.run(
                [str(venv_bin / name), "-c",
                 "import encodings, sys; print(sys.prefix)"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert probe.returncode == 0, (
                f"{name} failed to boot after anchoring:\n{probe.stderr}"
            )
            assert str(venv) in probe.stdout
