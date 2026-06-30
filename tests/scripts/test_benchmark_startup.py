"""Tests for scripts/benchmark_startup.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_benchmark_startup_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "benchmark_startup.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so @dataclass can resolve
    # the module for string annotation lookup.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_child_env_sets_isolated_home_and_scrubs_credentials(tmp_path):
    module = _load_module()
    env = module.build_child_env(
        tmp_path,
        base_env={"OPENROUTER_API_KEY": "secret", "SOME_TOKEN": "secret", "NORMAL_SETTING": "kept", "PYTHONPATH": "existing-path"},
    )
    assert env["HERMES_HOME"] == str(tmp_path)
    assert env["NORMAL_SETTING"] == "kept"
    assert "OPENROUTER_API_KEY" not in env
    assert "SOME_TOKEN" not in env
    assert str(module.REPO_ROOT) in env["PYTHONPATH"].split(module.os.pathsep)
    assert "existing-path" in env["PYTHONPATH"].split(module.os.pathsep)


def test_benchmark_targets_are_named_and_use_requested_python():
    module = _load_module()
    targets = module.benchmark_targets(python="python-test")
    names = {target.name for target in targets}
    assert {"import_run_agent", "import_model_tools", "import_hermes_cli_main", "cli_help"} <= names
    assert all(target.command[0] == "python-test" for target in targets)


def test_summarize_runs_reports_median_and_failure_tail():
    module = _load_module()
    summary = module.summarize_runs(
        [{"elapsed_ms": 30.0, "returncode": 0, "stderr_tail": ""},
         {"elapsed_ms": 10.0, "returncode": 1, "stderr_tail": "boom"},
         {"elapsed_ms": 20.0, "returncode": 0, "stderr_tail": ""}]
    )
    assert summary["runs"] == 3
    assert summary["ok"] is False
    assert summary["min_ms"] == 10.0
    assert summary["median_ms"] == 20.0
    assert summary["max_ms"] == 30.0
    assert summary["failure_stderr_tail"] == "boom"


def test_parse_args_rejects_invalid_repeats():
    module = _load_module()
    with pytest.raises(SystemExit):
        module.parse_args(["--repeats", "0"])
