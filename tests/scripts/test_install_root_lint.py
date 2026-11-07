"""scripts/install_root_lint.py fails on a new ``__file__`` climb to the install root or a new
module-level ``sys.path`` edit, and names the owner to call instead."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install_root_lint.py"


def _load():
    spec = importlib.util.spec_from_file_location("install_root_lint", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    }
    rises = lint.regressions(lint.tally(lint.collect(tmp_path)), {"root_from_file": {"top.py": 1}})
    assert rises == ["root_from_file: pkg/sub/mod.py (1, baseline 0)", "module_sys_path: pkg/sub/mod.py (1, baseline 0)"]
    assert "hermes_cli._startup_fast.project_root_str()" in lint.FIX_HINT
