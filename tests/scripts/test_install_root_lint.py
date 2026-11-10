"""scripts/install_root_lint.py fails on a new ``__file__`` climb to the install root or a new
module-level ``sys.path`` edit, and names the owner to call instead."""

import importlib.util
from pathlib import Path

from hermes_agent.compat.paths import PathMoves

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install_root_lint.py"


def _load():
    spec = importlib.util.spec_from_file_location("install_root_lint", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_time_edits_in_classes_inverted_guards_and_match_cases_are_counted(tmp_path):
    """Class bodies, ``match`` cases and the import branch of an inverted ``__main__`` guard run at
    import, and removals change the search path too; the script-only branch and function bodies do not."""
    lint = _load()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\ninclude = ["pkg", "pkg.*"]\n', encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "guarded.py").write_text(
        "import sys\n"
        "class Bootstrap:\n"
        "    sys.path.insert(0, '/a')\n"
        "if __name__ != '__main__':\n"
        "    sys.path.append('/b')\n"
        "else:\n"
        "    sys.path.append('/script-only')\n"
        "match sys.platform:\n"
        "    case 'linux':\n"
        "        sys.path.append('/c')\n"
        "if __name__ == '__main__':\n"
        "    sys.path.append('/script-only')\n"
        "def run():\n"
        "    sys.path.append('/call-time')\n"
        "sys.path.remove('/a')\n"
        "del sys.path[0]\n"
        "sys.path.clear()\n",
        encoding="utf-8")

    assert lint.collect(tmp_path)["module_sys_path"] == [
        "pkg/guarded.py:3", "pkg/guarded.py:5", "pkg/guarded.py:10",
        "pkg/guarded.py:15", "pkg/guarded.py:16", "pkg/guarded.py:17"]


def test_new_root_climb_and_module_sys_path_edit_fail_and_name_the_owner(tmp_path):
    lint = _load()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\ninclude = ["pkg", "pkg.*"]\n', encoding="utf-8")
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "sub" / "mod.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parents[2]\n"
        "sys.path.insert(0, str(ROOT))\n"
        "OWN_DATA = Path(__file__).parent / 'data.json'\n"
        "def f():\n"
        "    sys.path.insert(0, '/x')\n"
        "    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n",
        encoding="utf-8")
    (tmp_path / "top.py").write_text(
        "import os\nHERE = os.path.dirname(os.path.abspath(__file__))\n", encoding="utf-8")

    assert lint.collect(tmp_path) == {
        "root_from_file": ["pkg/sub/mod.py:3", "top.py:2"],
        "module_sys_path": ["pkg/sub/mod.py:4"],
        "moved_climb": [],
    }
    rises = lint.regressions(lint.tally(lint.collect(tmp_path)), {"root_from_file": {"top.py": 1}})
    assert rises == ["root_from_file: pkg/sub/mod.py (1, baseline 0)", "module_sys_path: pkg/sub/mod.py (1, baseline 0)"]
    assert "hermes_cli._startup_fast.project_root_str()" in lint.FIX_HINT


def test_a_moved_file_is_judged_where_it_is_and_a_climb_to_its_old_root_blocks(tmp_path, monkeypatch):
    """After ``pkg`` moves into ``hermes_agent/``: a climb that reaches the root counts under the old path,
    a baselined climb that reached the old root blocks, and a new file climbing to the package passes."""
    lint = _load()
    moves = PathMoves({"pkg": "hermes_agent.pkg"})
    monkeypatch.setattr(lint, "original_path", moves.original)
    monkeypatch.setattr(lint, "current_path", moves.current)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\ninclude = ["hermes_agent", "hermes_agent.*"]\n', encoding="utf-8")
    (tmp_path / "hermes_agent" / "pkg").mkdir(parents=True)
    climb = "from pathlib import Path\nROOT = Path(__file__).resolve().parents[{}]\n"
    for name, levels in (("fixed", 2), ("stale", 1), ("new", 1)):
        (tmp_path / "hermes_agent" / "pkg" / f"{name}.py").write_text(climb.format(levels), encoding="utf-8")

    found = lint.collect(tmp_path)
    assert found["root_from_file"] == ["hermes_agent/pkg/fixed.py:2"]
    assert lint.regressions(lint.tally(found), {}) == ["root_from_file: hermes_agent/pkg/fixed.py (1, baseline 0)"]
    baseline = {"root_from_file": {"pkg/fixed.py": 1, "pkg/stale.py": 1}}
    assert lint.regressions(lint.tally(found), baseline) == []
    assert lint.moved_climbs(found, baseline) == ["hermes_agent/pkg/stale.py:2"]
