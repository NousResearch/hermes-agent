"""Desktop hand-off self-heal for venvs bricked by the reverted macOS TCC
anchor (#95759), exercised against the REAL ``posix.sh`` functions.

The reverted anchor (#95425/#95541) left installs with a real-file
``venv/bin/python`` copy, a ``.tcc-anchor-source`` marker, and ``python3*``
aliases that die at interpreter init (``No module named 'encodings'``).
``venv/bin/hermes`` execs ``venv/bin/python3``, so the desktop update
hand-off — and every other CLI entrypoint, doctor included — dies before any
Python-side heal can run.  The heal therefore lives in the hand-off shell.

These tests drive ``posix.sh --self-test-tcc-heal`` (the script's real
``tcc_anchor_heal`` / ``tcc_pick_update_invoke`` functions, no re-implementation)
against synthetic venv trees on Linux.  macOS itself is NOT live-tested here —
there is no mac runner — but the heal logic is platform-independent shell and
the self-test path runs it without the ``uname`` gate.

Interpreter stubs: the probe is ``<interpreter> -c 'import encodings'``; a
stub that exits 0 stands in for a bootable interpreter and one that prints
the real crash text and exits 1 stands in for a bricked one.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
POSIX_SH = REPO_ROOT / "scripts" / "desktop-update" / "posix.sh"

GOOD_STUB = "#!/bin/bash\n# bootable interpreter stub\nexit 0\n"
BAD_STUB = (
    "#!/bin/bash\n"
    "echo \"Fatal Python error: init_fs_encoding: failed to get the Python "
    "codec of the filesystem encoding\" >&2\n"
    "echo \"ModuleNotFoundError: No module named 'encodings'\" >&2\n"
    "exit 1\n"
)
# Boots only when invoked via a path whose basename is exactly `python`:
# used to force the post-heal verification probe to fail (rollback path).
PICKY_STUB = (
    "#!/bin/bash\n"
    "[ \"$(basename \"$0\")\" = \"python\" ] && exit 0\n"
    "exit 1\n"
)

requires_bash = pytest.mark.skipif(
    not os.path.exists("/bin/bash"), reason="posix.sh needs /bin/bash"
)


def _write_exe(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def make_venv(tmp_path: Path, *, python: str | None, python3: str | None,
              marker: str | None, store_python: str | None = GOOD_STUB,
              alias_symlinks: bool = False) -> Path:
    """Build a synthetic install root with a venv/bin tree.

    ``python``/``python3`` are stub bodies (None = absent). ``marker`` is the
    ``.tcc-anchor-source`` content (None = absent, "STORE" = point it at the
    created store interpreter). ``store_python`` is the store interpreter stub
    body (None = do not create — the vanished-uv-store class).
    """
    root = tmp_path / "install"
    bin_dir = root / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    store = tmp_path / "uv-store" / "bin" / "python3.11"
    if store_python is not None:
        store.parent.mkdir(parents=True)
        _write_exe(store, store_python)
    if python is not None:
        _write_exe(bin_dir / "python", python)
    for name in ("python3", "python3.11"):
        if python3 is None:
            continue
        if alias_symlinks:
            (bin_dir / name).symlink_to("python")
        else:
            _write_exe(bin_dir / name, python3)
    if marker is not None:
        (bin_dir / ".tcc-anchor-source").write_text(
            str(store) if marker == "STORE" else marker, encoding="utf-8"
        )
    return root


def run_selftest(root: Path) -> str:
    proc = subprocess.run(
        ["/bin/bash", str(POSIX_SH), "--self-test-tcc-heal", "--no-ui",
         "--install-root", str(root)],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip().splitlines()[-1]


@requires_bash
class TestAnchorHeal:
    def test_healthy_venv_untouched(self, tmp_path):
        """A bootable python3 short-circuits the heal — nothing is examined."""
        root = make_venv(tmp_path, python=GOOD_STUB, python3=GOOD_STUB,
                         marker="STORE")
        before = (root / "venv/bin/python3").read_text(encoding="utf-8")
        out = run_selftest(root)
        assert "state=healthy" in out
        assert (root / "venv/bin/python3").read_text(encoding="utf-8") == before
        assert (root / "venv/bin/.tcc-anchor-source").exists()

    def test_alias_brick_healed_to_real_files(self, tmp_path):
        """#95541 class: anchored copy boots, alias symlinks are dead.

        The heal must replace the aliases with REAL FILES of the anchor (an
        alias symlink onto the copy is the crash shape) and KEEP the marker so
        ensure_tcc_anchor recognises the layout as active (no ping-pong).
        """
        root = make_venv(tmp_path, python=GOOD_STUB, python3=BAD_STUB,
                         marker="STORE")
        out = run_selftest(root)
        assert "state=healed-aliases" in out
        bin_dir = root / "venv/bin"
        for name in ("python3", "python3.11"):
            alias = bin_dir / name
            assert not alias.is_symlink()
            assert alias.read_text(encoding="utf-8") == GOOD_STUB
        assert (bin_dir / ".tcc-anchor-source").exists()
        assert not list(bin_dir.glob("*tcc-heal*"))
        assert "invoke=" in out and out.endswith("/venv/bin/hermes")

    def test_full_brick_restored_to_symlinks(self, tmp_path):
        """Anchored copy AND aliases dead, marker source alive: restore the
        pre-anchor symlink layout and drop the marker."""
        root = make_venv(tmp_path, python=BAD_STUB, python3=BAD_STUB,
                         marker="STORE")
        out = run_selftest(root)
        assert "state=healed-symlinks" in out
        bin_dir = root / "venv/bin"
        py = bin_dir / "python"
        assert py.is_symlink()
        assert py.resolve() == (tmp_path / "uv-store/bin/python3.11")
        for name in ("python3", "python3.11"):
            alias = bin_dir / name
            assert alias.is_symlink()
            assert os.readlink(alias) == "python"
        assert not (bin_dir / ".tcc-anchor-source").exists()
        assert not list(bin_dir.glob("*tcc-heal*"))

    def test_missing_source_fails_closed(self, tmp_path):
        """Vanished uv store (the #95759 comment class): touch nothing."""
        root = make_venv(tmp_path, python=BAD_STUB, python3=BAD_STUB,
                         marker="STORE", store_python=None)
        out = run_selftest(root)
        assert "state=source-missing" in out
        bin_dir = root / "venv/bin"
        assert (bin_dir / "python").read_text(encoding="utf-8") == BAD_STUB
        assert (bin_dir / ".tcc-anchor-source").exists()

    def test_no_marker_is_not_ours_to_heal(self, tmp_path):
        root = make_venv(tmp_path, python=BAD_STUB, python3=BAD_STUB,
                         marker=None)
        out = run_selftest(root)
        assert "state=no-marker" in out
        assert (root / "venv/bin/python").read_text(encoding="utf-8") == BAD_STUB

    def test_marker_inside_venv_is_unsafe(self, tmp_path):
        root = make_venv(tmp_path, python=BAD_STUB, python3=BAD_STUB,
                         marker="PLACEHOLDER")
        bin_dir = root / "venv/bin"
        inside = bin_dir / "recorded-python"
        _write_exe(inside, GOOD_STUB)
        (bin_dir / ".tcc-anchor-source").write_text(str(inside), encoding="utf-8")
        out = run_selftest(root)
        assert "state=unsafe-source" in out
        assert (bin_dir / "python").read_text(encoding="utf-8") == BAD_STUB

    def test_relative_marker_is_invalid(self, tmp_path):
        root = make_venv(tmp_path, python=BAD_STUB, python3=BAD_STUB,
                         marker="../python")
        out = run_selftest(root)
        assert "state=invalid-marker" in out

    def test_failed_verification_rolls_back(self, tmp_path):
        """If python3 still refuses to boot after the repair, every replaced
        file is restored and no staging/backup litter remains."""
        root = make_venv(tmp_path, python=PICKY_STUB, python3=BAD_STUB,
                         marker="STORE")
        out = run_selftest(root)
        assert "state=failed" in out
        bin_dir = root / "venv/bin"
        assert (bin_dir / "python3").read_text(encoding="utf-8") == BAD_STUB
        assert (bin_dir / "python3.11").read_text(encoding="utf-8") == BAD_STUB
        assert (bin_dir / ".tcc-anchor-source").exists()
        assert not list(bin_dir.glob("*tcc-heal*"))


@requires_bash
class TestUpdateInvokeFallback:
    def test_dead_alias_no_marker_falls_back_to_module_invocation(self, tmp_path):
        """No marker to heal from, but the venv python itself boots (the
        launchd-gateway shape): drive the update as python -m hermes_cli.main."""
        root = make_venv(tmp_path, python=GOOD_STUB, python3=BAD_STUB,
                         marker=None)
        out = run_selftest(root)
        assert "state=no-marker" in out
        assert out.endswith("/venv/bin/python -m hermes_cli.main")

    def test_healthy_venv_keeps_hermes_entrypoint(self, tmp_path):
        root = make_venv(tmp_path, python=GOOD_STUB, python3=GOOD_STUB,
                         marker=None)
        out = run_selftest(root)
        assert out.endswith("/venv/bin/hermes")


@requires_bash
class TestHandoffSurvivesBrickAB:
    """A/B analog of the reported loop: `venv/bin/hermes` execs python3."""

    def _make_hermes(self, root: Path) -> Path:
        hermes = root / "venv/bin/hermes"
        _write_exe(
            hermes,
            "#!/bin/bash\n"
            'exec "$(cd "$(dirname "$0")" && pwd)/python3" -c "import encodings"\n',
        )
        return hermes

    def test_bricked_entrypoint_fails_before_and_boots_after_heal(self, tmp_path):
        root = make_venv(tmp_path, python=GOOD_STUB, python3=BAD_STUB,
                         marker="STORE")
        hermes = self._make_hermes(root)
        # BEFORE the heal: the entrypoint dies exactly like the field logs.
        before = subprocess.run([str(hermes)], capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        assert before.returncode == 1
        assert "No module named 'encodings'" in before.stderr
        # Heal (real posix.sh function), then the same entrypoint boots.
        assert "state=healed-aliases" in run_selftest(root)
        after = subprocess.run([str(hermes)], capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
        assert after.returncode == 0
