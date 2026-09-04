"""venv_sync must work on trees where the venv does not exist yet.

It is the stdlib-only-at-import pre-venv entry point: the installers call
it on a fresh clone before any dependency is importable, and post_update
calls it after a tree swap when the venv is not trustworthy. Its
behaviour is driven with a FAKE uv — the module's job is deciding
whether/how to call uv and what to record, not resolving packages, and a
fake makes every decision observable (argv, env, cwd) without network.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_cli import venv_sync

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_bare(snippet: str) -> subprocess.CompletedProcess:
    """Run a snippet in an isolated interpreter (-I: no env, no user site)
    from the repo root, so a bare import graph is what gets exercised."""
    return subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(snippet)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )


# ── the stdlib-only contract, RUN not parsed ─────────────────────────────


class TestStdlibOnly:
    def test_imports_and_answers_bare(self, tmp_path):
        """The whole CLI surface must survive a stripped interpreter."""
        result = _run_bare(
            f"""
            import sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from hermes_cli import venv_sync
            root = {str(tmp_path)!r}
            out = venv_sync.sync(root, check=True)
            assert out["state"] == "failed", out  # empty dir: no pyproject
            print("ok")
            """
        )
        assert result.returncode == 0, result.stderr

    def test_no_third_party_module_ends_up_loaded(self):
        """Importing it must not drag in ANY non-stdlib module.

        Check sys.modules after import, because a parser cannot see lazy
        or conditional imports fire. First-party pm is allowed at SYNC
        time but must not load at IMPORT time either — the bare import
        happens on trees where even reading pm's ledger is premature.

        Measure the *delta*: snapshot the foreign modules already loaded
        before the import and only flag what the import adds. A hardcoded
        allowlist of ".pth"-injected bootstrap names is not portable — the
        interpreter loads ``.pth`` hooks at startup (setuptools'
        ``_distutils_hack``, pywin32's ``pywin32_bootstrap``, the
        editable-install ``__editable___…_finder``, ``_virtualenv``…) before
        the snippet runs, and that set varies per platform and install
        shape. A before/after diff is sensitive to the import and immune to
        that harness noise.
        """
        result = _run_bare(
            f"""
            import sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            stdlib = set(sys.stdlib_module_names)

            def _foreign():
                return {{
                    name.split(".")[0]
                    for name, mod in sys.modules.items()
                    if mod is not None and getattr(mod, "__file__", None)
                }} - stdlib

            before = _foreign()
            import hermes_cli.venv_sync
            dragged_in = {{n for n in _foreign() - before if n != "hermes_cli"}}
            assert not dragged_in, f"venv_sync loaded non-stdlib: {{dragged_in}}"
            print("ok")
            """
        )
        assert result.returncode == 0, result.stderr


# ── behaviour, driven through a fake uv ──────────────────────────────────


def _fake_uv(tmp_path: Path, exit_code: int = 0) -> tuple:
    """A uv that records its invocation and exits as told.

    Returns ``(binary_path, record_path)``; the binary shape is
    platform-appropriate (a .bat trampoline on Windows, a shebang script
    on POSIX) so subprocess can execute it directly.
    """
    record = tmp_path / "uv-calls.jsonl"
    recorder = tmp_path / "uv_recorder.py"
    recorder.write_text(
        "import json, os, sys\n"
        f"with open({str(record)!r}, 'a') as f:\n"
        "    f.write(json.dumps({\n"
        "        'argv': sys.argv[1:],\n"
        "        'cwd': os.getcwd(),\n"
        "        'venv_env': os.environ.get('UV_PROJECT_ENVIRONMENT'),\n"
        "        'virtual_env': os.environ.get('VIRTUAL_ENV'),\n"
        "    }) + '\\n')\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    if os.name == "nt":
        binary = tmp_path / "uv.bat"
        binary.write_text(
            f'@echo off\r\n"{sys.executable}" "{recorder}" %*\r\n', encoding="utf-8"
        )
    else:
        binary = tmp_path / "uv"
        binary.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{recorder}" "$@"\n', encoding="utf-8"
        )
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary, record


def _checkout(tmp_path: Path, name: str = "co") -> Path:
    """A minimal tree venv_sync classifies as a checkout."""
    root = tmp_path / name
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "uv.lock").write_text("lock-v1\n")
    return root


def _wire_uv(monkeypatch, tmp_path: Path, exit_code: int = 0) -> Path:
    """Point venv_sync's managed-uv resolution at a fake binary.

    The resolution itself is pm's (pm.ensure.uv) and pm's ledger wiring
    has its own tests; here the decision layer is under test, so the
    seam is venv_sync._managed_uv.
    """
    binary, record = _fake_uv(tmp_path, exit_code)
    monkeypatch.setattr(
        venv_sync, "_managed_uv", lambda: (str(binary), dict(os.environ))
    )
    return record


class TestCheckoutSync:
    def test_a_stale_checkout_syncs_frozen_against_its_own_venv(
        self, tmp_path, monkeypatch
    ):
        root = _checkout(tmp_path)
        record = _wire_uv(monkeypatch, tmp_path)

        out = venv_sync.sync(root)

        assert out == {"state": "synced", "ok": True}
        call = json.loads(record.read_text().splitlines()[0])
        assert call["argv"] == ["sync", "--frozen"]
        assert call["venv_env"] == str(root / "venv")
        assert call["virtual_env"] is None  # caller's venv must not leak in
        assert Path(call["cwd"]) == root

    def test_a_current_checkout_never_invokes_uv(self, tmp_path, monkeypatch):
        """The stamp is the fast path: currency costs a file read."""
        root = _checkout(tmp_path)
        record = _wire_uv(monkeypatch, tmp_path)

        assert venv_sync.sync(root)["state"] == "synced"
        assert venv_sync.sync(root)["state"] == "current"
        assert len(record.read_text().splitlines()) == 1  # one call total

    def test_an_edited_pyproject_invalidates_the_stamp(
        self, tmp_path, monkeypatch
    ):
        """Extras edits without a lock bump still re-sync."""
        root = _checkout(tmp_path)
        _wire_uv(monkeypatch, tmp_path)
        assert venv_sync.sync(root)["state"] == "synced"

        (root / "pyproject.toml").write_text("[project]\nname='y'\n")

        assert venv_sync.sync(root)["state"] == "synced"

    def test_a_lockless_tree_drops_the_frozen_flag(self, tmp_path, monkeypatch):
        """A clone without uv.lock still syncs — unfrozen."""
        root = _checkout(tmp_path)
        (root / "uv.lock").unlink()
        record = _wire_uv(monkeypatch, tmp_path)

        assert venv_sync.sync(root)["state"] == "synced"
        call = json.loads(record.read_text().splitlines()[0])
        assert call["argv"] == ["sync"]

    def test_a_failed_sync_writes_no_stamp(self, tmp_path, monkeypatch):
        """A failure must leave the next run trying again, not skipping."""
        root = _checkout(tmp_path)
        _wire_uv(monkeypatch, tmp_path, exit_code=3)

        out = venv_sync.sync(root)

        assert out["state"] == "failed" and "3" in out["detail"]
        assert venv_sync.read_stamp(root) == {}

    def test_check_mode_reports_and_changes_nothing(self, tmp_path, monkeypatch):
        root = _checkout(tmp_path)
        record = _wire_uv(monkeypatch, tmp_path)

        out = venv_sync.sync(root, check=True)

        assert out["state"] == "would-sync"
        assert not record.exists()  # no uv call
        assert venv_sync.read_stamp(root) == {}

    def test_no_managed_uv_is_a_failure_that_names_the_fix(
        self, tmp_path, monkeypatch
    ):
        root = _checkout(tmp_path)
        # pm has nothing installed here and may not lazy-install.
        monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
        monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")

        out = venv_sync.sync(root)

        assert out["state"] == "failed"
        assert "pm install" in out["detail"]

    def test_own_tree_delegates_to_pms_venv_ledger(self, tmp_path, monkeypatch):
        """When the root IS the running tree, pm.sync_venv owns the work
        (one extras ledger, one authority) — no direct uv drive."""
        root = _checkout(tmp_path)
        from pm import paths as pm_paths

        monkeypatch.setattr(pm_paths, "repo_root", lambda: root)
        calls = []

        import pm

        monkeypatch.setattr(pm, "sync_venv", lambda explicit=False: calls.append(explicit))

        out = venv_sync.sync(root)

        assert out == {"state": "synced", "ok": True}
        assert calls == [True]


class TestSealedTrees:
    def test_a_sealed_tree_is_a_clean_noop(self, tmp_path, monkeypatch):
        """The desktop payload and nix bundle must not fail, must not sync."""
        root = tmp_path / "sealed"
        root.mkdir()
        (root / "install-stamp.json").write_text(
            json.dumps({"commit": "abc123", "payload": "full", "updateMechanism": "electron-updater"})
        )
        record = _wire_uv(monkeypatch, tmp_path)

        out = venv_sync.sync(root)

        assert out == {"state": "sealed", "ok": True}
        assert not record.exists()

    def test_a_dev_tree_with_both_stamp_and_git_is_a_checkout(
        self, tmp_path, monkeypatch
    ):
        root = _checkout(tmp_path)
        (root / "install-stamp.json").write_text(
            json.dumps({"commit": "abc", "updateMechanism": "electron-updater"})
        )
        _wire_uv(monkeypatch, tmp_path)

        assert venv_sync.sync(root)["state"] == "synced"

    def test_a_stamp_without_update_mechanism_is_a_build_lane_bug(self, tmp_path):
        root = tmp_path / "sealed"
        root.mkdir()
        (root / "install-stamp.json").write_text(json.dumps({"commit": "abc"}))
        with pytest.raises(RuntimeError, match="updateMechanism"):
            venv_sync.sync(root)


class TestCliContract:
    def test_json_output_and_exit_codes(self, tmp_path):
        """post_update and the installers read exactly this."""
        root = tmp_path / "sealed"
        root.mkdir()
        (root / "install-stamp.json").write_text(
            json.dumps({"commit": "x", "updateMechanism": "electron-updater"})
        )

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hermes_cli.venv_sync",
                "--project-root",
                str(root),
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout) == {"state": "sealed", "ok": True}

    def test_failure_exits_nonzero(self, tmp_path):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hermes_cli.venv_sync",
                "--project-root",
                str(tmp_path),  # empty dir: no pyproject, no stamp
                "--json",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

        assert proc.returncode == 1
        assert json.loads(proc.stdout)["state"] == "failed"
