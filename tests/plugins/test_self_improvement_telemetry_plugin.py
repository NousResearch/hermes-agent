from __future__ import annotations

import json
from pathlib import Path

import plugins.self_improvement_telemetry as plugin


class FakeContext:
    def __init__(self) -> None:
        self.hooks: dict[str, object] = {}

    def register_hook(self, name: str, fn: object) -> None:
        self.hooks[name] = fn


def test_registers_observer_hooks():
    ctx = FakeContext()

    plugin.register(ctx)

    assert {"pre_tool_call", "post_tool_call", "on_session_end", "on_session_reset"}.issubset(ctx.hooks)


def test_post_tool_call_writes_sanitized_metric(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SELF_IMPROVEMENT_TELEMETRY_DIR", str(tmp_path))

    plugin.on_post_tool_call(
        session_id="s1",
        tool_name="skill_view",
        args={"name": "hermes-agent", "secret": "should-not-log-value"},
        result="x" * 12_000,
        duration_ms=42,
    )

    rows = [json.loads(line) for line in (tmp_path / "context_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "tool_call_metric"
    assert row["session_id"] == "s1"
    assert row["tool_name"] == "skill_view"
    assert row["args_keys"] == ["name", "secret"]
    assert row["result_chars"] == 12000
    assert "large_tool_output" in row["risk_flags"]
    assert "should-not-log-value" not in json.dumps(row)


def test_duplicate_skill_view_is_flagged_within_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SELF_IMPROVEMENT_TELEMETRY_DIR", str(tmp_path))
    plugin.reset_runtime_state()

    plugin.on_post_tool_call(session_id="s1", tool_name="skill_view", args={"name": "hermes-agent"}, result="ok")
    plugin.on_post_tool_call(session_id="s1", tool_name="skill_view", args={"name": "hermes-agent"}, result="ok")

    rows = [json.loads(line) for line in (tmp_path / "context_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "duplicate_skill_view" in rows[-1]["risk_flags"]


def test_repeated_cronjob_list_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_SELF_IMPROVEMENT_TELEMETRY_DIR", str(tmp_path))
    plugin.reset_runtime_state()

    plugin.on_post_tool_call(session_id="s1", tool_name="cronjob", args={"action": "list"}, result="ok")
    plugin.on_post_tool_call(session_id="s1", tool_name="cronjob", args={"action": "list"}, result="ok")

    rows = [json.loads(line) for line in (tmp_path / "context_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "repeated_cronjob_list" in rows[-1]["risk_flags"]
