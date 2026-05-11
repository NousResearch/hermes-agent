from __future__ import annotations

from pathlib import Path

import tomllib


def test_new_top_level_modules_are_packaged():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    modules = set(pyproject["tool"]["setuptools"]["py-modules"])

    assert "hermes_logger" in modules
    assert "benchmark_record" in modules
    assert "loop_detector" in modules
