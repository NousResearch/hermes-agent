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


class TestTheLauncherRuntimeIsNotReEntered:
    """A launcher's runtime must be complete, because its children run no launcher body.

    A supervisor re-spawns the commands an install publishes as
    ``sys.executable -m hermes_cli.main`` -- kanban workers and external cron workers --
    and such a child enters neither the launcher nor ``activate_dependencies``. The
    interpreter the published launcher embeds therefore has to import the application and
    its dependencies on its own (2026-09-25: a publish embedded the dependency-less store
    Python and every child died importing ``hermes_cli`` for twelve minutes). Re-entering a
    process that already runs the embedded runtime would swap it for the store Python and
    hand that same interpreter to every child spawned afterwards.
    """

    @staticmethod
    def _install(tmp_path: Path, generation: tuple[int, int], *, monkeypatch=None):
        """A self-managed source install whose committed generation records *generation*.

        Only the layout is fabricated: ``pm.environments`` reads it for real.
        """
        import pm
        from pm.environments import install_state_dir, runtime_facts_path

        home, store = tmp_path / "home", tmp_path / "store"
        if monkeypatch is not None:
            monkeypatch.setattr(Path, "home", lambda: home)
            monkeypatch.setenv("HOME", str(home))
            monkeypatch.setenv("HERMES_HOME", str(home))
            monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
            monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
            # Currency is a neighbour of this subject, not part of it: never sync here.
            monkeypatch.setattr(pm, "venv_is_current", lambda **kwargs: True)

        root = tmp_path / "hermes-agent"
        (root / ".git").mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        (root / "install-stamp.json").write_text(
            json.dumps({"updateMechanism": "self"}), encoding="utf-8",
        )

        environment = install_state_dir(root) / "environments" / "generation" / "venv"
        (environment / "lib" / f"python{generation[0]}.{generation[1]}" / "site-packages").mkdir(parents=True)
        (environment / "pyvenv.cfg").write_text(
            f"home = /managed\nversion_info = {generation[0]}.{generation[1]}.0\n", encoding="utf-8",
        )
        runtime_facts_path(root).write_text(
            json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8",
        )

        # The managed interpreter: a distinct path, as _launchers resolves it from the store.
        store_python = store / "python-managed" / "bin" / "python3"
        store_python.parent.mkdir(parents=True)
        store_python.symlink_to(sys._base_executable)
        (store / "facts.json").write_text(
            json.dumps({"packages": {"python": {"entry": "python-managed"}}}), encoding="utf-8",
        )
        return root, environment, store_python

    @staticmethod
    def _caller() -> tuple[int, int]:
        return (sys.version_info[0], sys.version_info[1])

    @staticmethod
    def _generation_interpreter(environment: Path) -> Path:
        """Give the committed generation a real interpreter, as PM builds one."""
        binary = environment / "bin" / "python"
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.symlink_to(sys._base_executable)
        return binary

    @pytest.mark.platforms("posix")
    def test_the_launcher_runtime_is_not_re_entered(self, tmp_path, monkeypatch):
        """Re-entry here would hand every future child the dependency-less store Python."""
        root, environment, _ = self._install(tmp_path, self._caller(), monkeypatch=monkeypatch)
        self._generation_interpreter(environment)
        monkeypatch.setattr(sys, "prefix", str(environment))

        assert venv_sync.prepare_launch(root, []) is None

    @pytest.mark.platforms("posix")
    def test_re_entry_lands_on_the_runtime_the_launcher_embeds(self, tmp_path, monkeypatch):
        """When re-entry is owed, it must not land on the store interpreter."""
        root, environment, store_python = self._install(tmp_path, self._caller(), monkeypatch=monkeypatch)
        interpreter = self._generation_interpreter(environment)

        assert venv_sync.prepare_launch(root, []) == interpreter
        assert venv_sync.prepare_launch(root, []) != store_python
