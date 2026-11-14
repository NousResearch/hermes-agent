"""A tree guard fails when one of its scan roots is gone, instead of passing with nothing checked.

Each guard below walks a fixed list of top-level directories. When a directory moves (into
``src/hermes_agent/`` or a sibling package) a guard that skips missing roots keeps reporting
success while it checks none of the moved files.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from hermes_agent.compat.paths import current_path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["check_config_yaml_writers", "check_config_yaml_readers"])
def test_config_guard_fails_when_a_default_tree_is_missing(name, tmp_path, monkeypatch, capsys):
    guard = _load(name)
    for tree in map(current_path, guard.DEFAULT_TREES[1:]):
        source = REPO_ROOT / tree
        (tmp_path / tree).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / tree).mkdir() if source.is_dir() else shutil.copy(source, tmp_path / tree)
    monkeypatch.setattr(guard, "ROOT", tmp_path)

    assert guard.main([]) == 2
    assert guard.DEFAULT_TREES[0] in capsys.readouterr().err


def test_subprocess_stdin_guard_fails_when_a_context_dir_is_missing(tmp_path, monkeypatch, capsys):
    guard = _load("check_subprocess_stdin")
    for d in guard.TUI_CONTEXT_DIRS[1:]:
        (tmp_path / d).mkdir()
    monkeypatch.setattr(guard, "__file__", str(tmp_path / "scripts" / "check_subprocess_stdin.py"))
    monkeypatch.setattr(sys, "argv", ["check_subprocess_stdin.py"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)

    assert guard.main() == 2
    assert guard.TUI_CONTEXT_DIRS[0] in capsys.readouterr().err


def test_boundary_lint_fails_when_plugins_or_every_core_package_is_missing(tmp_path, capsys):
    lint = _load("boundary_lint")
    with pytest.raises(SystemExit) as missing_plugins:
        lint.module_dirs(tmp_path)
    assert missing_plugins.value.code == 2 and "plugins does not exist" in capsys.readouterr().err
    with pytest.raises(SystemExit) as no_core:
        list(lint._core_files(tmp_path, []))
    assert no_core.value.code == 2 and "no core file" in capsys.readouterr().err
