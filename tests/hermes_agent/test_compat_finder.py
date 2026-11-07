"""The compat finder serves an old module name as the one real module, through chained renames."""
from __future__ import annotations

import importlib
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
