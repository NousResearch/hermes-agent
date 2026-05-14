from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_windows_footguns_module():
    module_path = REPO_ROOT / "scripts" / "check-windows-footguns.py"
    spec = importlib.util.spec_from_file_location("check_windows_footguns", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_windows_footgun_checker_exits_nonzero_for_bare_open(tmp_path):
    module = _load_windows_footguns_module()
    target = tmp_path / "bad.py"
    target.write_text("with open('notes.txt') as f:\n    print(f.read())\n", encoding="utf-8")

    assert module.main([str(target)]) == 1


def test_windows_footgun_checker_allows_explicit_encoding(tmp_path):
    module = _load_windows_footguns_module()
    target = tmp_path / "good.py"
    target.write_text(
        "with open('notes.txt', encoding='utf-8') as f:\n    print(f.read())\n",
        encoding="utf-8",
    )

    assert module.main([str(target)]) == 0
