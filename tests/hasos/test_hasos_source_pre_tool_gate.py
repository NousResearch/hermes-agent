from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import model_tools


POLICY_ENGINE_PATH = Path.home() / ".hermes" / "scripts" / "hasos_policy_engine.py"
pytestmark = pytest.mark.skipif(
    not POLICY_ENGINE_PATH.exists(),
    reason="local HASOS source-gate integration test requires ~/.hermes/scripts/hasos_policy_engine.py",
)


def load_policy_engine():
    spec = importlib.util.spec_from_file_location("hasos_policy_engine_source_gate_test", POLICY_ENGINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_policy_engine(monkeypatch):
    monkeypatch.setattr(model_tools, "_load_hasos_policy_engine", load_policy_engine)


def test_hasos_source_pre_tool_gate_blocks_ungated_level4_release(monkeypatch):
    install_policy_engine(monkeypatch)
    result = model_tools._hasos_source_pre_tool_gate(
        "terminal",
        {"command": "fastlane deliver submit for review"},
        task_id="test-task",
        tool_call_id="call-1",
        session_id="session-1",
    )
    assert result is not None
    data = json.loads(result)
    assert data["blocked"] is True
    assert data["blocked_by"] == "hasos_source_pre_tool_gate"
    assert data["level"] == 4
    assert data["action"] == "block"
    assert "release" in data["message"].lower()
    assert data["os_sandbox"] is False


def test_hasos_source_pre_tool_gate_blocks_level5_before_dispatch(monkeypatch):
    install_policy_engine(monkeypatch)
    result = model_tools._hasos_source_pre_tool_gate(
        "terminal",
        {"command": "rm -rf /"},
        task_id="test-task",
        tool_call_id="call-2",
        session_id="session-1",
    )
    assert result is not None
    data = json.loads(result)
    assert data["blocked"] is True
    assert data["level"] == 5
    assert data["action"] == "block"
    assert "Level 5" in data["message"]


def test_hasos_source_pre_tool_gate_allows_low_risk_reads(monkeypatch):
    install_policy_engine(monkeypatch)
    result = model_tools._hasos_source_pre_tool_gate(
        "read_file",
        {"path": "/tmp/example"},
        task_id="test-task",
        tool_call_id="call-3",
        session_id="session-1",
    )
    assert result is None


def test_hasos_source_pre_tool_gate_sanitizes_block_message(monkeypatch):
    class FakePolicyEngine:
        @staticmethod
        def evaluate_payload(_payload):
            return {
                "schema_version": "hasos.policy_decision.v1",
                "level": 4,
                "decision": "block",
                "action": "block",
                "message": "blocked token=TEST_FAKE_VALUE_abcdefghijkl",
                "reason_codes": ["test"],
                "required_gates": ["gate"],
                "missing_gates": ["gate"],
            }

        @staticmethod
        def sanitize_text(value):
            return str(value).replace("TEST_FAKE_VALUE_abcdefghijkl", "[REDACTED]")

    monkeypatch.setattr(model_tools, "_load_hasos_policy_engine", lambda: FakePolicyEngine)
    result = model_tools._hasos_source_pre_tool_gate(
        "terminal",
        {"command": "fastlane deliver submit for review"},
    )
    assert result is not None
    serialized = json.dumps(json.loads(result), ensure_ascii=False)
    assert "TEST_FAKE_VALUE_abcdefghijkl" not in serialized
    assert "[REDACTED]" in serialized
