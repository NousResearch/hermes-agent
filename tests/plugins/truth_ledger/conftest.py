from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _load_truth_module(module_name: str):
    repo_root = Path(__file__).resolve().parents[3]
    plugin_dir = repo_root / "plugins" / "truth-ledger"
    mod_path = plugin_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"hermes_plugins.truth_ledger.{module_name}",
        mod_path,
        submodule_search_locations=[str(plugin_dir)],
    )
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    if "hermes_plugins.truth_ledger" not in sys.modules:
        pkg = types.ModuleType("hermes_plugins.truth_ledger")
        pkg.__path__ = [str(plugin_dir)]
        sys.modules["hermes_plugins.truth_ledger"] = pkg
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def spool_mod():
    return _load_truth_module("spool")


@pytest.fixture
def ledger_mod():
    return _load_truth_module("ledger")


@pytest.fixture
def projection_mod():
    return _load_truth_module("projection")
