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
    program = f"import sys\nsys.path.insert(0, {str(REPO_ROOT)!r})\n" + textwrap.dedent(snippet)
    return subprocess.run([sys.executable, '-I', '-S', '-c', program],
                          capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)


def test_bare_import_and_passive_paths(tmp_path):
    result = _run_bare(f"""
        from pathlib import Path
        from hermes_cli import venv_sync
        assert 'pm' not in sys.modules
        root = Path({str(tmp_path)!r})
        assert venv_sync.sync(root, check=True)['state'] == 'failed'
        (root / 'install-stamp.json').write_text('{{"updateMechanism":"external"}}')
        assert venv_sync.sync(root) == {{'state': 'sealed', 'ok': True}}
        assert venv_sync.prepare_launch(root, []) is None
        assert 'pm.install' not in sys.modules
    """)
    assert result.returncode == 0, result.stderr


def test_bare_harness_rejects_third_party_import():
    result = _run_bare('import requests')
    assert result.returncode != 0 and 'ModuleNotFoundError' in result.stderr


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

    def sync(*, explicit, project_root, evict_incompatible_plugins):
        assert evict_incompatible_plugins, "an update sync must disable misfit plugins, not fail"
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


class TestSupervisedLaunchDoesNotRunTheCompletionTail:
    """Regression for #123340.

    ``prepare_launch`` execs ``source_completion --finish-update``, which builds a fresh
    environment under ``installs/<hash>/environments/``. A tail that cannot finish leaves its
    marker, so on a unit with ``Restart=always`` every restart built another one until the
    volume was full (415 directories, ~74G, after which ENOSPC took down gateway stop and
    ``hermes update`` too). A supervised launch must never become the installer.
    """

    def test_supervised_launch_skips_prepare_launch_entirely(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
        assert venv_sync._is_supervised_launch() is True

    def test_interactive_shell_is_not_treated_as_supervised(self, monkeypatch):
        monkeypatch.delenv("HERMES_SUPERVISED_CHILD", raising=False)
        monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
        monkeypatch.delenv("HERMES_GATEWAY_EXTERNAL_SUPERVISOR", raising=False)
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        assert venv_sync._is_supervised_launch() is False

    def test_existing_passive_gate_still_works(self, tmp_path):
        """The new branch is additive — the other early returns are untouched."""
        (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        assert venv_sync.prepare_launch(tmp_path, ["-h"]) is None


class TestEnvironmentsGrowthBudget:
    """The disk-fill half of #123340: stop allocating once the tree is already huge."""

    def _stub_install_state(self, monkeypatch, env_dir: Path):
        import pm.environments as pm_env

        monkeypatch.setattr(
            pm_env, "install_state_dir", lambda _root: env_dir.parent, raising=False,
        )

    def test_healthy_tree_proceeds(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "state" / "environments"
        (env_dir / "gen-1").mkdir(parents=True)
        (env_dir / "gen-1" / "pyvenv.cfg").write_text("x", encoding="utf-8")
        self._stub_install_state(monkeypatch, env_dir)
        assert venv_sync._environments_budget_exceeded(tmp_path) is None

    def test_many_small_environments_proceed(self, tmp_path, monkeypatch):
        """Count alone must not trip it — a legitimate install keeps a few generations."""
        env_dir = tmp_path / "state" / "environments"
        env_dir.mkdir(parents=True)
        for i in range(20):
            (env_dir / f"gen-{i}").mkdir()
        self._stub_install_state(monkeypatch, env_dir)
        assert venv_sync._environments_budget_exceeded(tmp_path) is None

    def test_large_tree_is_refused_with_actionable_advice(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "state" / "environments"
        env_dir.mkdir(parents=True)
        # Sparse files: st_size reports the full length, so the gate sees a multi-gigabyte
        # tree while the test costs a few KB of disk. Writing real bytes here would make the
        # suite fail on a nearly-full volume rather than on a real regression.
        count = venv_sync._ENVIRONMENTS_GROWTH_LIMIT + 2
        per_dir = venv_sync._ENVIRONMENTS_GROWTH_BYTES // (count - 2)
        for i in range(count):
            entry = env_dir / f"gen-{i}"
            entry.mkdir()
            with open(entry / "payload.bin", "wb") as fh:
                fh.truncate(per_dir)
        self._stub_install_state(monkeypatch, env_dir)

        reason = venv_sync._environments_budget_exceeded(tmp_path)
        assert reason is not None
        # The message must say what to do, not just that something is wrong.
        assert "hermes update" in reason
        assert "hermes pm gc" in reason
