"""Tests for the loop-diagnostics plugin hook wiring.

Verifies the zero-cost-when-disabled contract:
  * disabled by config  -> register() registers zero hooks
  * enabled but not a kanban worker -> recorder stays inert (no files)
  * enabled in a kanban worker env -> hooks fire, trace written to disk
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLUGIN = PROJECT_ROOT / "plugins" / "observability" / "loop_diagnostics"


def _load_plugin(monkeypatch, *, enabled: bool, kanban_env: bool, tmp_path: Path):
    """Load the plugin module fresh with a stubbed config + env."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    # stub load_recorder_config so the plugin sees the desired state
    import hermes_cli.observability.loop_diagnostics_recorder as recorder_mod

    def fake_load_config(config=None):
        return {
            "enabled": enabled,
            "max_events_per_run": 100,
            "retain_runs": 2,
        }

    monkeypatch.setattr(recorder_mod, "load_recorder_config", fake_load_config)

    # stub the base_dir so tests stay hermetic
    monkeypatch.setattr(recorder_mod, "loop_traces_dir", lambda board=None: tmp_path)

    if kanban_env:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
        monkeypatch.setenv("HERMES_PROFILE", "default")

    import plugins.observability.loop_diagnostics as plugin_mod

    # reset module-level recorder state between loads
    plugin_mod._recorder = None
    return plugin_mod


class FakeCtx:
    def __init__(self):
        self.hooks = []

    def register_hook(self, name, fn):
        self.hooks.append((name, fn))


def test_disabled_registers_zero_hooks(monkeypatch, tmp_path):
    mod = _load_plugin(monkeypatch, enabled=False, kanban_env=True, tmp_path=tmp_path)
    ctx = FakeCtx()
    mod.register(ctx)
    assert ctx.hooks == []


def test_enabled_but_not_kanban_worker_writes_nothing(monkeypatch, tmp_path):
    mod = _load_plugin(monkeypatch, enabled=True, kanban_env=False, tmp_path=tmp_path)
    ctx = FakeCtx()
    mod.register(ctx)
    assert len(ctx.hooks) == 6

    # fire a tool call through the registered hooks
    for name, fn in ctx.hooks:
        if name == "pre_tool_call":
            fn(tool_name="terminal", args={"command": "echo hi"}, turn_id="t1")
        elif name == "post_tool_call":
            fn(tool_name="terminal", result="hi", turn_id="t1", status="ok")
        elif name == "on_session_end":
            fn()
    # no kanban identity -> recorder stays disabled -> nothing on disk
    assert not list(tmp_path.rglob("*.jsonl"))


def test_enabled_kanban_worker_writes_trace(monkeypatch, tmp_path):
    mod = _load_plugin(monkeypatch, enabled=True, kanban_env=True, tmp_path=tmp_path)
    ctx = FakeCtx()
    mod.register(ctx)
    assert len(ctx.hooks) == 6

    hook_map = {name: fn for name, fn in ctx.hooks}
    hook_map["pre_tool_call"](
        tool_name="terminal", args={"command": "echo hi"}, turn_id="t1",
        tool_call_id="call_1",
    )
    hook_map["post_tool_call"](
        tool_name="terminal", result={"output": "hi"}, turn_id="t1",
        tool_call_id="call_1", status="ok", duration_ms=5,
    )
    hook_map["on_session_end"]()

    trace_file = tmp_path / "t_test" / "1.jsonl"
    assert trace_file.exists()
    records = [json.loads(line) for line in trace_file.read_text().splitlines()]
    kinds = [r["kind"] for r in records]
    assert "run_header" in kinds
    assert "action_start" in kinds
    assert "action_end" in kinds
    assert "run_footer" in kinds
    footer = [r for r in records if r["kind"] == "run_footer"][0]
    assert footer["outcome"] == "completed"
    # redaction: raw command never hits disk
    blob = trace_file.read_text()
    assert "echo hi" not in blob


def test_plugin_manifest_declares_hooks():
    import yaml

    manifest = yaml.safe_load((PLUGIN / "plugin.yaml").read_text())
    assert manifest["name"] == "loop-diagnostics"
    assert set(manifest["hooks"]) == {
        "pre_tool_call",
        "post_tool_call",
        "subagent_start",
        "subagent_stop",
        "on_session_end",
        "on_session_finalize",
    }
