"""Tests for the Volt V2 tool adapter proof plugin."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from plugins.volt_v2_tool_adapter import register
from plugins.volt_v2_tool_adapter.adapter import (
    MUTATION_TOOLS,
    evaluate_call,
    handle_adapter_exception,
    on_post_tool_call,
    transform_tool_result,
)
from plugins.volt_v2_tool_adapter.audit import (
    AUDIT_SCHEMA_VERSION,
    build_audit_event,
    read_jsonl_events,
    redact_args,
    validate_audit_event,
)
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
        args={"path": str(denied_path), "token": "***"},
        result="{}",
        config=cfg,
    )

    assert decision.allowlisted is False
    assert "denylisted_path" in decision.reasons
    assert not Path(cfg.audit_path).exists()
    redacted = redact_args({"path": str(denied_path), "token": "***"})
    rendered = json.dumps(redacted)
    assert str(denied_path) not in rendered
    assert "***" not in rendered
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




def test_malformed_config_falls_back_to_safe_defaults(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    cfg = VoltV2ToolAdapterConfig.from_mapping(
        {
            "enabled": "yes",
            "mode": "nonsense",
            "fail_policy": "unsafe",
            "audit_path": audit_path,
            "allowlist": "bad-shape",
            "denylist": {"tools": "terminal", "path_prefixes": 123},
            "verification": {"write_audit_jsonl": "off", "require_result_marker": "yes"},
        }
    )

    assert cfg.enabled is True
    assert cfg.mode == "observe"
    assert cfg.fail_policy == "open"
    assert cfg.allowlist_tools == ()
    assert cfg.denylist_tools == ("terminal",)
    assert cfg.denylist_path_prefixes == ()
    assert cfg.verification.write_audit_jsonl is False
    assert cfg.verification.require_result_marker is True


def test_audit_event_schema_v1_parser_and_redaction(tmp_path):
    raw_path = tmp_path / "vault" / "note.md"
    raw_path.parent.mkdir()
    raw_path.write_text("secret-ish")
    event = build_audit_event(
        tool_name="read_file",
        args={
            "path": str(raw_path),
            "token": "abc123",
            "nested": {"password": "pw", "safe": "ok"},
            "files": [str(raw_path), "plain"],
        },
        result={"content": "ok"},
        mode="observe",
        decision="observed",
        allowlisted=True,
        duration_ms=7,
    )

    validate_audit_event(event)
    assert event["schema_version"] == AUDIT_SCHEMA_VERSION
    rendered = json.dumps(event, ensure_ascii=False)
    assert str(raw_path) not in rendered
    assert "abc123" not in rendered
    assert "pw" not in rendered
    assert event["args_shape"]["token"] == "<redacted>"
    assert event["args_shape"]["nested"]["password"] == "<redacted>"
    assert event["args_shape"]["path"].startswith("sha256:")

    audit_file = tmp_path / "audit" / "events.jsonl"
    from plugins.volt_v2_tool_adapter.audit import write_audit_event

    write_audit_event(audit_file, event)
    assert read_jsonl_events(audit_file) == [event]


def test_audit_event_validation_rejects_wrong_schema():
    bad = {
        "schema_version": 999,
        "ts": "now",
        "session_id": "",
        "task_id": "",
        "tool_call_id": "",
        "tool_name": "read_file",
        "mode": "observe",
        "decision": "observed",
        "allowlisted": True,
        "duration_ms": 0,
        "args_shape": {},
        "result_chars": 0,
        "reason": "ok",
    }

    try:
        validate_audit_event(bad)
    except ValueError as exc:
        assert "schema_version" in str(exc)
    else:
        raise AssertionError("validate_audit_event should reject unsupported schema_version")


def test_plugin_manager_integration_loads_opt_in_bundled_plugin_and_hooks(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    vault = tmp_path / "vault"
    vault.mkdir(parents=True)
    note = vault / "note.md"
    note.write_text("integration-ok")
    audit_path = tmp_path / "audit" / "events.jsonl"
    (hermes_home / "config.yaml").parent.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {"enabled": ["volt_v2_tool_adapter"]},
                "volt_v2": {
                    "tool_adapter": {
                        "enabled": True,
                        "mode": "transform",
                        "audit_path": str(audit_path),
                        "allowlist": {"tools": ["read_file"], "paths": [str(vault)]},
                        "denylist": {"tools": ["terminal"], "path_prefixes": [str(vault / ".env")]},
                    }
                },
            }
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import hermes_cli.plugins as plugin_mod
    from hermes_cli.plugins import PluginManager
    from model_tools import handle_function_call

    mgr = PluginManager()
    monkeypatch.setattr(plugin_mod, "_plugin_manager", mgr, raising=False)
    mgr.discover_and_load(force=True)

    loaded = mgr._plugins.get("volt_v2_tool_adapter")
    assert loaded is not None
    assert loaded.enabled is True
    assert "post_tool_call" in mgr._hooks
    assert "transform_tool_result" in mgr._hooks

    result = handle_function_call(
        "read_file",
        {"path": str(note)},
        task_id="plugin-test",
        session_id="session-test",
        tool_call_id="call-test",
    )
    parsed = json.loads(result)
    assert "integration-ok" in parsed["content"]
    assert parsed["_volt_v2_adapter"]["decision"] == "transformed"
    events = read_jsonl_events(audit_path)
    assert [event["decision"] for event in events] == ["transform", "transformed"]
    assert str(note) not in audit_path.read_text()


def test_security_privacy_gate_blocks_sensitive_args_and_dangerous_paths(tmp_path):
    cfg = _cfg(tmp_path)
    vault = Path(cfg.artifact_root)

    sensitive = evaluate_call("read_file", {"path": str(vault / "note.md"), "api_key": "abc"}, cfg)
    denied_path = evaluate_call("read_file", {"path": str(vault / ".env")}, cfg)
    denied_tool = evaluate_call("terminal", {"command": "pwd"}, cfg)

    assert sensitive.allowlisted is False
    assert "sensitive_args" in sensitive.reasons
    assert denied_path.allowlisted is False
    assert "denylisted_path" in denied_path.reasons
    assert denied_tool.allowlisted is False
    assert "denylisted_tool" in denied_tool.reasons


def test_transform_value_gate_does_not_change_non_string_or_denied_results(tmp_path):
    cfg = _cfg(tmp_path, mode="transform")
    vault = Path(cfg.artifact_root)
    (vault / "note.md").write_text("ok")

    assert transform_tool_result("read_file", {"path": str(vault / "note.md")}, {"content": "ok"}, cfg) is None
    assert transform_tool_result("terminal", {"command": "pwd"}, '{"output":"ok"}', cfg) is None
