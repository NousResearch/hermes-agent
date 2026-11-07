"""The compat finder serves an old module name as the one real module, through chained renames."""
from __future__ import annotations

import importlib
import importlib.util
import runpy
import sys
from unittest import mock

import pytest

from hermes_agent.compat.finder import Finder

# Phase 1 moved compat_probe_old under compat_probe_tree; phase 2 then renamed it to cli. The
# real package compat_probe_tree is not a key, only its child compat_probe_tree.old is.
MOVES = {
    "compat_probe_old": "compat_probe_tree.cli",
    "compat_probe_tree.old": "compat_probe_tree.cli",
}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    cli = tmp_path / "compat_probe_tree" / "cli"
    cli.mkdir(parents=True)
    (cli.parent / "__init__.py").write_text("", encoding="utf-8")
    (cli / "__init__.py").write_text("", encoding="utf-8")
    (cli / "main.py").write_text(
        "STATE = {'n': 0}\n\ndef run():\n    STATE['n'] += 1\n    return STATE['n']\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield cli / "main.py"
    for name in [n for n in sys.modules if n.partition(".")[0] in {"compat_probe_old", "compat_probe_tree"}]:
        del sys.modules[name]


def test_every_old_name_is_the_real_module(tree, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", [Finder(MOVES), *sys.meta_path])
    import compat_probe_old.main as original
    import compat_probe_tree.old.main as interim
    from compat_probe_tree.cli import main as current

    assert original is interim is current
    original.run()
    interim.run()
    assert current.STATE["n"] == 2
    with mock.patch("compat_probe_tree.old.main.run", return_value=99):
        assert current.run() == 99
    assert current.run() == 3

    tree.write_text("STATE = {'n': 'reloaded'}\n", encoding="utf-8")
    assert importlib.reload(original) is current
    assert current.STATE["n"] == "reloaded"
    assert current.__spec__.name == "compat_probe_tree.cli.main"


def test_strict_mode_rejects_an_old_name_and_names_the_new_one(tree, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", [Finder(MOVES, strict=True), *sys.meta_path])
    with pytest.raises(ImportError, match=r"compat_probe_tree\.old has moved: import compat_probe_tree\.cli$") as raised:
        import compat_probe_tree.old.main
    assert type(raised.value) is ImportError
    import compat_probe_tree.cli.main


def test_concurrent_alias_imports_keep_the_real_spec(tree, monkeypatch):
    finder = Finder(MOVES)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    import compat_probe_tree.cli.main as current

    # Two threads import two old names: each import creates the module, then executes it.
    first = finder.find_spec("compat_probe_old.main")
    second = finder.find_spec("compat_probe_tree.old.main")
    module = importlib.util.module_from_spec(first)
    importlib.util.module_from_spec(second)
    first.loader.exec_module(module)
    second.loader.exec_module(module)

    assert module is current
    assert current.__spec__.name == "compat_probe_tree.cli.main"
    assert importlib.reload(current) is current


def test_a_child_moved_apart_from_its_package_resolves_in_a_from_import(tree, monkeypatch):
    (tree.parent / "renamed.py").write_text("NAME = 'renamed'\n", encoding="utf-8")
    moves = {**MOVES, "compat_probe_old.child": "compat_probe_tree.cli.renamed"}
    monkeypatch.setattr(sys, "meta_path", [Finder(moves), *sys.meta_path])

    from compat_probe_old import child
    from compat_probe_tree.cli import renamed

    assert child is renamed


@pytest.mark.parametrize("entrypoint", ["main", "__main__"])
def test_old_name_launches_execute_the_current_code(tree, monkeypatch, entrypoint):
    monkeypatch.setattr(sys, "meta_path", [Finder(MOVES), *sys.meta_path])
    source = tree.parent / f"{entrypoint}.py"
    source.write_text("from .helper import VALUE\n", encoding="utf-8")
    (tree.parent / "helper.py").write_text("VALUE = 'launched'\n", encoding="utf-8")
    old_name = "compat_probe_old.main" if entrypoint == "main" else "compat_probe_old"

    result = runpy.run_module(old_name, run_name="__main__")

    assert result["VALUE"] == "launched"
    assert result["__file__"] == str(source)
    assert f"compat_probe_tree.cli.{entrypoint}" not in sys.modules


def test_missing_current_module_keeps_its_import_error(tree, monkeypatch):
    moves = {"compat_probe_old.missing": "compat_probe_tree.cli.missing"}
    monkeypatch.setattr(sys, "meta_path", [Finder({**MOVES, **moves}), *sys.meta_path])

    with pytest.raises(ModuleNotFoundError) as raised:
        importlib.import_module("compat_probe_old.missing")

    assert raised.value.name == "compat_probe_tree.cli.missing"
