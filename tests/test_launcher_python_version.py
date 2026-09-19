"""Regression tests for the source-tree Hermes launcher."""

from __future__ import annotations

import contextlib
import io
import runpy
import sys
import types
from collections import namedtuple
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).parents[1] / "hermes"


def test_launcher_rejects_unsupported_python_before_cli_import(monkeypatch):
    """Old interpreters get an actionable error instead of an annotation traceback."""
    fake_cli = types.ModuleType("hermes_cli.main")
    fake_cli.main = lambda: None
    fake_package = types.ModuleType("hermes_cli")
    fake_package.__path__ = []
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_package)
    monkeypatch.setitem(sys.modules, "hermes_cli.main", fake_cli)
    version_info = namedtuple("version_info", "major minor micro")
    monkeypatch.setattr(sys, "version_info", version_info(3, 9, 21))

    stderr = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, contextlib.redirect_stderr(stderr):
        runpy.run_path(str(LAUNCHER), run_name="__main__")

    assert exc_info.value.code == 1
    assert "requires Python 3.11" in stderr.getvalue()
    assert "detected Python 3.9.21" in stderr.getvalue()
