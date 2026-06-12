"""Reporting and hook-smoke tests for completion-auditor."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_COMPLETION_AUDITOR_LOG_DIR", raising=False)
    yield hermes_home


def _load_plugin_package():
    plugin_dir = _repo_root() / "plugins" / "completion-auditor"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    for key in list(sys.modules):
        if key.startswith("hermes_plugins.completion_auditor_reporting_under_test"):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.completion_auditor_reporting_under_test",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.completion_auditor_reporting_under_test"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.completion_auditor_reporting_under_test"] = mod
    spec.loader.exec_module(mod)
    mod._reset_for_tests()
    return mod


def _records(hermes_home: Path) -> list[dict]:
    found = []
    for path in sorted((hermes_home / "logs" / "completion-auditor").glob("*.jsonl*")):
        for line in path.read_text(encoding="utf-8").splitlines():
            found.append(json.loads(line))
    return found


def test_log_retention_prunes_old_completion_audit_files(_isolate_env):
    _load_plugin_package()
    report_mod = sys.modules["hermes_plugins.completion_auditor_reporting_under_test.report"]
    config_mod = sys.modules["hermes_plugins.completion_auditor_reporting_under_test.config"]
    log_dir = _isolate_env / "logs" / "completion-auditor"
    log_dir.mkdir(parents=True)
    old = log_dir / "completion-audit-2000-01-01.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    old_time = (datetime.now(timezone.utc) - timedelta(days=30)).timestamp()
    os.utime(old, (old_time, old_time))

    settings = config_mod.AuditorConfig(log_dir=log_dir, log_retention_days=7)
    report_mod.write_record(settings, {"schema": "hermes-completion-audit-v1"})

    assert not old.exists()
    assert list(log_dir.glob("completion-audit-*.jsonl"))


def test_log_rotation_uses_numbered_jsonl_when_daily_file_is_full(_isolate_env):
    _load_plugin_package()
    report_mod = sys.modules["hermes_plugins.completion_auditor_reporting_under_test.report"]
    config_mod = sys.modules["hermes_plugins.completion_auditor_reporting_under_test.config"]
    log_dir = _isolate_env / "logs" / "completion-auditor"
    log_dir.mkdir(parents=True)
    settings = config_mod.AuditorConfig(log_dir=log_dir, max_log_size_mb=1)
    base = report_mod._daily_log_path(log_dir)
    base.write_text("x" * (1024 * 1024), encoding="utf-8")

    written = report_mod.write_record(settings, {"schema": "hermes-completion-audit-v1"})

    assert written != base
    assert written.name.endswith(".1.jsonl")
    assert written.exists()


def test_enabled_plugin_hook_smoke_writes_audit_without_mutating_response(_isolate_env):
    import yaml

    (_isolate_env / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["completion-auditor"]}}),
        encoding="utf-8",
    )
    for key in list(sys.modules):
        if key.startswith("hermes_plugins"):
            del sys.modules[key]

    import hermes_cli.plugins as plugins_mod

    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    mgr = plugins_mod._ensure_plugins_discovered(force=True)
    assert mgr.has_hook("post_tool_call")
    assert mgr.has_hook("post_llm_call")

    assert mgr.invoke_hook(
        "post_tool_call",
        session_id="s1",
        turn_id="t1",
        task_id="task-1",
        tool_name="terminal",
        status="success",
        args={"command": "python -m pytest tests/plugins -q"},
        result={"output": "54 passed", "exit_code": 0},
    ) == []
    response = "Tests passed: 54 passed."
    assert mgr.invoke_hook(
        "post_llm_call",
        session_id="s1",
        turn_id="t1",
        task_id="task-1",
        assistant_response=response,
    ) == []

    records = _records(_isolate_env)
    assert len(records) == 1
    record = records[0]
    assert record["schema"] == "hermes-completion-audit-v1"
    assert record["claim_text"] == response
    assert record["verdict"] == "supported"
    assert record["final_response_mutated"] is False
    assert record["tool_result_excerpt_included"] is False
