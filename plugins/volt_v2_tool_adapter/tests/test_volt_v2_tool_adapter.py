"""Tests for the Volt V2 tool adapter proof plugin."""

from __future__ import annotations

import json
from pathlib import Path

from plugins.volt_v2_tool_adapter import register
from plugins.volt_v2_tool_adapter.adapter import (
    MUTATION_TOOLS,
    evaluate_call,
    handle_adapter_exception,
    on_post_tool_call,
    transform_tool_result,
)
from plugins.volt_v2_tool_adapter.audit import redact_args, read_jsonl_events
from plugins.volt_v2_tool_adapter.config import VoltV2ToolAdapterConfig, load_adapter_config


def _cfg(tmp_path: Path, **overrides) -> VoltV2ToolAdapterConfig:
    vault = tmp_path / "volt-v2"
    vault.mkdir(parents=True, exist_ok=True)
    audit_path = tmp_path / "audit" / "volt-v2-tool-adapter.jsonl"
    data = {
        "enabled": True,
        "mode": "observe",
        "fail_policy": "open",
        "artifact_root": str(vault),
        "audit_path": str(audit_path),
        "allowlist": {
            "tools": ["read_file", "search_files", "write_file"],
            "toolsets": [],
            "paths": [str(vault)],
        },
        "denylist": {
            "tools": ["terminal", "send_message", "cronjob", "memory"],
            "path_prefixes": [str(tmp_path / ".ssh"), str(vault / ".env")],
        },
        "verification": {
            "emit_events": True,
            "write_audit_jsonl": True,
            "require_result_marker": True,
        },
    }
    data.update(overrides)
    return VoltV2ToolAdapterConfig.from_mapping(data)


def test_default_config_is_disabled(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    cfg = load_adapter_config()

    assert cfg.enabled is False
    assert cfg.mode == "observe"
    assert cfg.fail_policy == "open"


def test_disabled_adapter_does_not_write_audit(tmp_path):
    cfg = _cfg(tmp_path, enabled=False)
    path = Path(cfg.artifact_root) / "note.md"
    path.write_text("ok")

    on_post_tool_call(
        tool_name="read_file",
        args={"path": str(path)},
        result='{"content":"ok"}',
        task_id="t1",
        session_id="s1",
        tool_call_id="tc1",
        duration_ms=3,
        config=cfg,
    )

    assert not Path(cfg.audit_path).exists()


def test_allowlisted_read_only_tool_writes_audit_without_result_mutation(tmp_path):
    cfg = _cfg(tmp_path)
    path = Path(cfg.artifact_root) / "note.md"
    path.write_text("ok")
    original_result = '{"content":"ok"}'

    on_post_tool_call(
        tool_name="read_file",
        args={"path": str(path)},
        result=original_result,
        task_id="t1",
        session_id="s1",
        tool_call_id="tc1",
        duration_ms=12,
        config=cfg,
    )

    events = read_jsonl_events(Path(cfg.audit_path))
    assert len(events) == 1
    event = events[0]
    assert event["tool_name"] == "read_file"
    assert event["mode"] == "observe"
    assert event["decision"] == "observed"
    assert event["allowlisted"] is True
    assert event["result_chars"] == len(original_result)
    assert event["args_shape"]["path"].startswith("sha256:")
    assert str(path) not in json.dumps(event)


def test_denylisted_path_does_not_emit_audit_or_leak_path(tmp_path):
    cfg = _cfg(tmp_path)
    denied_path = Path(cfg.artifact_root) / ".env"
    denied_path.write_text("SECRET=value")

    decision = evaluate_call("read_file", {"path": str(denied_path)}, cfg)
    on_post_tool_call(
        tool_name="read_file",
        args={"path": str(denied_path), "token": "sk-secret-value"},
        result="{}",
        config=cfg,
    )

    assert decision.allowlisted is False
    assert "denylisted_path" in decision.reasons
    assert not Path(cfg.audit_path).exists()
    redacted = redact_args({"path": str(denied_path), "token": "sk-secret-value"})
    rendered = json.dumps(redacted)
    assert str(denied_path) not in rendered
    assert "sk-secret-value" not in rendered
    assert redacted["token"] == "<redacted>"


def test_transform_mode_marks_only_allowlisted_string_results(tmp_path):
    cfg = _cfg(tmp_path, mode="transform")
    path = Path(cfg.artifact_root) / "note.md"
    path.write_text("ok")

    transformed = transform_tool_result(
        tool_name="read_file",
        args={"path": str(path)},
        result='{"content":"ok"}',
        config=cfg,
    )
    denied = transform_tool_result(
        tool_name="terminal",
        args={"command": "pwd"},
        result='{"output":"/tmp"}',
        config=cfg,
    )

    assert isinstance(transformed, str)
    parsed = json.loads(transformed)
    assert parsed["content"] == "ok"
    assert parsed["_volt_v2_adapter"]["mode"] == "transform"
    assert parsed["_volt_v2_adapter"]["decision"] == "transformed"
    assert denied is None


def test_adapter_hook_failures_fail_open_for_observe_and_transform(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    path = Path(cfg.artifact_root) / "note.md"
    path.write_text("ok")

    import plugins.volt_v2_tool_adapter.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "write_audit_event", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    on_post_tool_call(tool_name="read_file", args={"path": str(path)}, result="{}", config=cfg)
    assert transform_tool_result(tool_name="read_file", args={"path": str(path)}, result="{}", config=cfg) is None


def test_route_override_exception_fallback_policy_for_mutation_and_read_only(tmp_path):
    route_cfg = _cfg(tmp_path, mode="route")
    read_result = handle_adapter_exception("read_file", RuntimeError("boom"), route_cfg)
    write_result = handle_adapter_exception("write_file", RuntimeError("boom"), route_cfg)

    assert read_result is None
    assert json.loads(write_result)["error"].startswith("Volt V2 adapter failed closed")
    assert "write_file" in MUTATION_TOOLS


def test_plugin_registers_post_and_transform_hooks():
    hooks = []

    class Ctx:
        def register_hook(self, name, callback):
            hooks.append((name, callback.__name__))

    register(Ctx())

    assert ("post_tool_call", "on_post_tool_call") in hooks
    assert ("transform_tool_result", "transform_tool_result") in hooks
