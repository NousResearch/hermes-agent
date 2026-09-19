"""Out-of-tree installs: the update paths never fabricate a nonexistent VIRTUAL_ENV.

On installs whose interpreter lives outside the checkout (``$HERMES_HOME\\venvs\\hermes`` — the
layout the shipped Windows gateway launchers pin themselves), ``PROJECT_ROOT/venv`` does not
exist. Pointing ``uv`` there aborts every command with ``Failed to inspect Python interpreter``
before any work: the ``hermes tools`` import probe resolves no target, tool-dependency restores
and lazy refreshes fail, and the update still reports success. See #116148.
"""

import sys
from pathlib import Path

import pytest

import hermes_cli.update_cmd as update_cmd
from hermes_constants import running_venv_root


@pytest.fixture
def fake_venv(tmp_path: Path) -> Path:
    """A venv root with the layout ``running_venv_root`` detects (bin/ + pyvenv.cfg)."""
    root = tmp_path / "venvs" / "hermes"
    bin_dir = root / ("Scripts" if sys.platform == "win32" else "bin")
    bin_dir.mkdir(parents=True)
    (bin_dir / ("python.exe" if sys.platform == "win32" else "python")).touch()
    (root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    return root


def _venv_python(root: Path) -> Path:
    return root / ("Scripts" if sys.platform == "win32" else "bin") / (
        "python.exe" if sys.platform == "win32" else "python")


class TestRunningVenvRoot:
    def test_returns_root_for_a_real_venv_interpreter(self, monkeypatch, fake_venv: Path):
        monkeypatch.setattr(sys, "executable", str(_venv_python(fake_venv)))
        assert running_venv_root() == fake_venv

    def test_none_for_a_system_interpreter_shape(self, monkeypatch, tmp_path: Path):
        # /usr/bin/python has the same parent-dir shape as a venv bin dir; without
        # pyvenv.cfg it must NOT read as a venv root.
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        exe = bin_dir / "python"
        exe.touch()
        monkeypatch.setattr(sys, "executable", str(exe))
        assert running_venv_root() is None


class TestResolvedInstallVenvDir:
    def test_out_of_tree_install_falls_back_to_running_interpreters_venv(self, monkeypatch, fake_venv: Path):
        monkeypatch.setattr(update_cmd, "project_venv_dir", lambda _root: None)
        monkeypatch.setattr(update_cmd, "running_venv_root", lambda: fake_venv)
        assert update_cmd._resolved_install_venv_dir() == fake_venv

    def test_no_resolvable_venv_returns_none_not_a_fabricated_path(self, monkeypatch):
        monkeypatch.setattr(update_cmd, "project_venv_dir", lambda _root: None)
        monkeypatch.setattr(update_cmd, "running_venv_root", lambda: None)
        assert update_cmd._resolved_install_venv_dir() is None

    def test_in_tree_venv_wins_over_the_running_interpreters_venv(self, monkeypatch, tmp_path: Path):
        in_tree = tmp_path / "venv"
        in_tree.mkdir()
        monkeypatch.setattr(update_cmd, "project_venv_dir", lambda _root: in_tree)
        monkeypatch.setattr(update_cmd, "running_venv_root", lambda: tmp_path / "elsewhere")
        assert update_cmd._resolved_install_venv_dir() == in_tree


class TestPipInstallPrefix:
    def test_env_pins_the_resolved_venv(self, monkeypatch, fake_venv: Path):
        monkeypatch.setattr(update_cmd, "project_venv_dir", lambda _root: None)
        monkeypatch.setattr(update_cmd, "running_venv_root", lambda: fake_venv)
        prefix, env = update_cmd._pip_install_prefix("/fake/bin/uv")
        assert prefix == ["/fake/bin/uv", "pip"]
        assert env is not None and env["VIRTUAL_ENV"] == str(fake_venv)

    def test_env_leaves_virtual_env_unset_when_nothing_resolves(self, monkeypatch):
        monkeypatch.setattr(update_cmd, "project_venv_dir", lambda _root: None)
        monkeypatch.setattr(update_cmd, "running_venv_root", lambda: None)
        _prefix, env = update_cmd._pip_install_prefix("/fake/bin/uv")
        assert env is not None and "VIRTUAL_ENV" not in env
