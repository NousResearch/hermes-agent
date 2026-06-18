"""Tests for the bundled observability/execution_receipts plugin."""

from __future__ import annotations

import importlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "execution_receipts"


class FakePluginContext:
    def __init__(self):
        self.hooks = {}
        self.commands = {}

    def register_hook(self, name, callback):
        self.hooks[name] = callback

    def register_command(self, name, *, handler, description=""):
        self.commands[name] = SimpleNamespace(handler=handler, description=description)


def _load_plugin():
    importlib.invalidate_caches()
    return importlib.import_module("plugins.observability.execution_receipts")


def _sample_receipt(sequence_number=1, *, tool_name="terminal", status="ok"):
    return {
        "schema_version": "hermes.execution_receipt.v0",
        "receipt_id": f"receipt-{sequence_number}",
        "receipt_type": "tool_complete",
        "trace_id": "trace-1",
        "span_id": f"span-{sequence_number}",
        "parent_span_id": None,
        "sequence_number": sequence_number,
        "timestamp": "2026-06-18T00:00:00Z",
        "session_id": "session-1",
        "task_id": "task-1",
        "turn_id": "turn-1",
        "api_request_id": "api-1",
        "tool_call_id": f"call-{sequence_number}",
        "tool_name": tool_name,
        "status": status,
        "duration_ms": 12,
        "args": {"redacted": True, "field_names": ["command"]},
        "result": {"redacted": True, "size_bytes": 200},
        "links": [],
        "evidence_gaps": [],
        "redaction_policy_version": "execution_receipts.v0",
        "redaction_status": "ok",
    }


def test_plugin_manifest_declares_execution_receipt_hook():
    data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())

    assert data["name"] == "execution_receipts"
    assert data["hooks"] == ["execution_receipt"]


def test_register_exposes_hook_and_receipts_command(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    mod = _load_plugin()
    ctx = FakePluginContext()

    mod.register(ctx)

    assert set(ctx.hooks) == {"execution_receipt"}
    assert "receipts" in ctx.commands
    assert "execution receipt" in ctx.commands["receipts"].description.lower()


def test_bundled_plugin_is_discovered_but_disabled_by_default(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from hermes_cli import plugins as pmod

    mgr = pmod.PluginManager()
    mgr.discover_and_load()

    loaded = mgr._plugins["observability/execution_receipts"]
    assert loaded.manifest.source == "bundled"
    assert not loaded.enabled
    assert loaded.error and "not enabled" in loaded.error


def test_bundled_plugin_loads_when_enabled(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["observability/execution_receipts"]}}),
        encoding="utf-8",
    )
    from hermes_cli import plugins as pmod

    mgr = pmod.PluginManager()
    mgr.discover_and_load()

    loaded = mgr._plugins["observability/execution_receipts"]
    assert loaded.enabled
    assert "execution_receipt" in loaded.hooks_registered
    assert "receipts" in loaded.commands_registered


def test_execution_receipt_hook_writes_owner_only_jsonl(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    path = hermes_home / "execution-receipts" / "receipts.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mod = _load_plugin()

    mod._on_execution_receipt(receipt=_sample_receipt())

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["tool_name"] == "terminal"
    assert stat.S_IMODE(path.parent.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


def test_receipts_status_tail_and_gaps_are_safe(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    path = hermes_home / "execution-receipts" / "receipts.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mod = _load_plugin()
    first = _sample_receipt(1, tool_name="terminal")
    second = _sample_receipt(3, tool_name="delegate_task", status="error")
    second["evidence_gaps"] = ["child_trace_unavailable"]
    # Simulate a buggy caller/plugin attempting to smuggle raw data; the
    # writer/commands must not surface it in summaries.
    second["args"] = {"raw": "do not show this secret"}

    mod._on_execution_receipt(receipt=first)
    mod._on_execution_receipt(receipt=second)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")

    status = mod._handle_slash("status")
    tail = mod._handle_slash("tail 5")
    gaps = mod._handle_slash("gaps")

    assert "total=2" in status
    assert "corrupt=1" in status
    assert "gaps=1" in status
    assert "terminal" in tail
    assert "delegate_task" in tail
    assert "do not show this secret" not in tail
    assert "missing sequence 2" in gaps
    assert "child_trace_unavailable" in gaps


def test_receipt_writer_strips_smuggled_payload_fields(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    path = hermes_home / "execution-receipts" / "receipts.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mod = _load_plugin()
    receipt = _sample_receipt()
    receipt["args"] = {
        "redacted": True,
        "kind": "dict",
        "field_names": ["command"],
        "raw": "secret command must not survive",
    }
    receipt["result"] = {
        "redacted": True,
        "size_bytes": 200,
        "raw_output": "secret output must not survive",
    }

    mod._on_execution_receipt(receipt=receipt)

    written = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    serialized = json.dumps(written)
    assert written["args"] == {"redacted": True, "kind": "dict", "field_names": ["command"]}
    assert written["result"] == {"redacted": True, "size_bytes": 200}
    assert "secret command" not in serialized
    assert "secret output" not in serialized


def test_receipt_writer_fail_open_on_malformed_receipt(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    path = hermes_home / "execution-receipts" / "receipts.jsonl"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    mod = _load_plugin()

    mod._on_execution_receipt(receipt="raw secret that must not be written")

    assert not path.exists()
    status = mod._handle_slash("status")
    assert "writer_errors=1" in status
    assert "raw secret" not in status
