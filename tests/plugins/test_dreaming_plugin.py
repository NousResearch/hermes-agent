"""Tests for the dreaming plugin (config.yaml re-scope)."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield hermes_home


def _ensure_dreaming_package(plugin_dir: Path) -> None:
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    if "hermes_plugins.dreaming" not in sys.modules:
        dream_pkg = types.ModuleType("hermes_plugins.dreaming")
        dream_pkg.__path__ = [str(plugin_dir)]
        sys.modules["hermes_plugins.dreaming"] = dream_pkg


def _load_module(name: str, plugin_dir: Path):
    _ensure_dreaming_package(plugin_dir)
    spec = importlib.util.spec_from_file_location(
        f"hermes_plugins.dreaming.{name}",
        plugin_dir / f"{name}.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.dreaming"
    qual = f"hermes_plugins.dreaming.{name}"
    sys.modules[qual] = mod
    setattr(sys.modules["hermes_plugins.dreaming"], name, mod)
    spec.loader.exec_module(mod)
    return mod


def _load_config_mod():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "dreaming"
    return _load_module("_config", plugin_dir)


def _load_schedule_mod():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "dreaming"
    _load_module("_config", plugin_dir)
    _load_module("_score", plugin_dir)
    _load_module("_diary", plugin_dir)
    return _load_module("_schedule", plugin_dir)


class TestDreamingConfig:
    def test_ensure_user_config_seeds_defaults(self, _isolate_env):
        cfg_mod = _load_config_mod()
        path = cfg_mod.ensure_user_config(_isolate_env)
        assert path.exists()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data["enabled"] is False
        assert data["schedule"]["min_hours"] == 24

    def test_main_config_yaml_override(self, _isolate_env):
        cfg_mod = _load_config_mod()
        cfg_mod.ensure_user_config(_isolate_env)
        main = _isolate_env / "config.yaml"
        main.write_text(
            yaml.safe_dump({"dreaming": {"enabled": True, "schedule": {"min_hours": 6}}}),
            encoding="utf-8",
        )
        loaded = cfg_mod.load_config(_isolate_env)
        assert loaded["enabled"] is True
        assert loaded["schedule"]["min_hours"] == 6

    def test_user_config_overrides_bundled_defaults(self, _isolate_env):
        cfg_mod = _load_config_mod()
        user_path = cfg_mod.ensure_user_config(_isolate_env)
        user_path.write_text(
            yaml.safe_dump({"enabled": True, "rem": {"model": "llama3.2:3b"}}),
            encoding="utf-8",
        )
        rem = cfg_mod.rem(cfg_mod.load_config(_isolate_env))
        assert rem["model"] == "llama3.2:3b"


class TestDreamingSchedule:
    def test_dream_run_uses_config_threshold(self, _isolate_env, monkeypatch):
        sched = _load_schedule_mod()
        cfg_mod = _load_config_mod()

        dreams = _isolate_env / "dreams"
        dreams.mkdir(parents=True)
        staging = dreams / "staging.jsonl"
        candidate = {
            "text": "The deployment pipeline uses GitHub Actions for CI.",
            "hash": "abc",
            "role": "user",
            "created_at": 1.0,
            "frequency": 3,
            "query_count": 2,
            "word_count": 8,
        }
        staging.write_text(json.dumps(candidate) + "\n", encoding="utf-8")

        cfg = cfg_mod.load_config(_isolate_env)
        cfg["scoring"] = {"promote_threshold": 0.99}

        monkeypatch.setattr(sched, "_rem_narrative", lambda candidates, cfg=None: "themes")

        result = sched.dream_run(force=True, hermes_home=str(_isolate_env), cfg=cfg)
        assert result["promoted"] == 0

        memory_path = _isolate_env / "MEMORY.md"
        memory = memory_path.read_text(encoding="utf-8") if memory_path.exists() else ""
        assert "deployment pipeline" not in memory
