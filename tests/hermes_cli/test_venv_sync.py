"""venv_sync must work on trees where the venv does not exist yet.

It is the stdlib-only-at-import pre-venv entry point: the installers call
it on a fresh clone before any dependency is importable, and post_update
calls it after a tree swap when the venv is not trustworthy. Its
behaviour is driven through PM's public client; package resolution and
publication belong to PM, not this CLI entry point.
"""

from __future__ import annotations

import json
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


def _checkout(tmp_path: Path, name: str = "co") -> Path:
    root = tmp_path / name
    (root / ".git").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    (root / "uv.lock").write_text("lock-v1\n")
    return root


def _wire_pm(monkeypatch, *, current=False, error=None):
    import pm

    calls = []
    monkeypatch.setattr(pm, "venv_is_current", lambda *, project_root: current)

    def sync(*, explicit, project_root):
        calls.append((project_root, explicit))
        if error:
            raise pm.InstallError("venv", error)

    monkeypatch.setattr(pm, "sync_venv", sync)
    return calls


class TestCheckoutSync:
    @pytest.mark.parametrize("foreign", [False, True])
    def test_each_root_uses_the_public_pm_transaction(self, tmp_path, monkeypatch, foreign):
        root = _checkout(tmp_path)
        monkeypatch.setattr(venv_sync, "_project_root", lambda: root)
        calls = _wire_pm(monkeypatch)
        assert venv_sync.sync(root if foreign else None) == {"state": "synced", "ok": True}
        assert calls == [(root, True)]

    def test_current_selection_needs_no_transaction(self, tmp_path, monkeypatch):
        root = _checkout(tmp_path)
        calls = _wire_pm(monkeypatch, current=True)
        assert venv_sync.sync(root) == {"state": "current", "ok": True}
        assert calls == []

    def test_check_is_passive(self, tmp_path, monkeypatch):
        root = _checkout(tmp_path)
        calls = _wire_pm(monkeypatch)
        before = set(tmp_path.rglob("*"))
        assert venv_sync.sync(root, check=True) == {"state": "would-sync", "ok": True}
        assert calls == []
        assert set(tmp_path.rglob("*")) == before

    def test_a_failed_sync_is_reported_and_retried(self, tmp_path, monkeypatch):
        root = _checkout(tmp_path)
        calls = _wire_pm(monkeypatch, error="resolution failed")
        for _ in range(2):
            out = venv_sync.sync(root)
            assert out["state"] == "failed" and not out["ok"]
            assert "resolution failed" in out["detail"]
        assert calls == [(root, True), (root, True)]


class TestSealedTrees:
    def test_a_sealed_tree_is_a_clean_noop(self, tmp_path, monkeypatch):
        """The desktop payload and nix bundle must not fail, must not sync."""
        root = tmp_path / "sealed"
        root.mkdir()
        (root / "install-stamp.json").write_text(
            json.dumps({"commit": "abc123", "payload": "full", "updateMechanism": "electron-updater"})
        )
        calls = _wire_pm(monkeypatch)

        out = venv_sync.sync(root)

        assert out == {"state": "sealed", "ok": True}
        assert calls == []

    def test_a_dev_tree_with_both_stamp_and_git_is_a_checkout(
        self, tmp_path, monkeypatch
    ):
        root = _checkout(tmp_path)
        (root / "install-stamp.json").write_text(
            json.dumps({"commit": "abc", "updateMechanism": "electron-updater"})
        )
        _wire_pm(monkeypatch)

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
