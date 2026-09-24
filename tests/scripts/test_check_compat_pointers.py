"""The compatibility-pointer scanner only traverses first-party Python trees."""

import importlib.util
import os
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_compat_pointers.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_compat_pointers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_py_files_prunes_dependency_trees_before_descent(tmp_path, monkeypatch):
    mod = _load()
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "package" / "apps").mkdir()
    (tmp_path / "package" / "apps" / "nested.py").write_text("VALUE = 3\n", encoding="utf-8")
    excluded = [
        tmp_path / ".venv",
        tmp_path / "venv",
        tmp_path / "node_modules",
        tmp_path / "package" / "__pycache__",
        tmp_path / "package" / "node_modules",
    ]
    for directory in excluded:
        directory.mkdir(parents=True)
        (directory / "dependency.py").write_text("VALUE = 2\n", encoding="utf-8")

    visited = []
    real_walk = os.walk

    def tracking_walk(root):
        for dirpath, dirnames, filenames in real_walk(root):
            visited.append(Path(dirpath))
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod.os, "walk", tracking_walk)

    assert set(mod._py_files()) == {
        tmp_path / "package" / "source.py",
        tmp_path / "package" / "apps" / "nested.py",
    }
    assert tmp_path in visited
    assert not set(excluded) & set(visited)