from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

from agent.context_telemetry import (
    build_context_telemetry_payload,
    emit_context_telemetry,
    write_context_telemetry,
)


def _agent(last_prompt_tokens=12345, context_length=272000):
    return SimpleNamespace(
        model="gpt-5.5",
        provider="openai-codex",
        session_id="session-123",
        session_input_tokens=10,
        session_output_tokens=20,
        session_reasoning_tokens=3,
        session_prompt_tokens=11,
        session_completion_tokens=22,
        session_total_tokens=33,
        session_api_calls=2,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=last_prompt_tokens,
            context_length=context_length,
            compression_count=1,
        ),
    )


def test_build_payload_uses_current_context_not_cumulative_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "diggr-main"))
    payload = build_context_telemetry_payload(
        _agent(last_prompt_tokens=12000, context_length=24000),
        usage={"total": 999999, "context_used": 999999, "context_max": 24000},
        config={"context": {"telemetry": {"smart_zone_tokens": 120000, "smart_zone_context_pct": 45}}},
        now=datetime(2026, 7, 22, tzinfo=timezone.utc),
    )

    assert payload["source"] == "hermes_runtime"
    assert payload["source_kind"] == "native_context_telemetry"
    assert payload["profile"] == "diggr-main"
    assert payload["context"] == {
        "used_tokens": 12000,
        "max_tokens": 24000,
        "percent": 50,
        "token_source": "actual",
    }
    assert payload["session_usage"]["total_tokens"] == 999999
    assert "messages" not in json.dumps(payload).lower()
    assert "api_key" not in json.dumps(payload).lower()


def test_unknown_context_does_not_fake_percent(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    payload = build_context_telemetry_payload(
        _agent(last_prompt_tokens=0, context_length=128000),
        usage={"total": 999999},
        config={},
    )

    assert payload["context"]["used_tokens"] is None
    assert payload["context"]["max_tokens"] == 128000
    assert payload["context"]["percent"] is None
    assert payload["context"]["token_source"] == "unknown"


def test_emit_context_telemetry_writes_atomic_json(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "diggr-main"))
    path = tmp_path / "state" / "hermes-context-telemetry.json"
    written = emit_context_telemetry(
        _agent(),
        config={
            "context": {
                "telemetry": {
                    "enabled": True,
                    "path": str(path),
                    "min_write_interval_seconds": 0,
                }
            }
        },
        force=True,
    )

    assert written is True
    data = json.loads(path.read_text())
    assert data["schema_version"] == 1
    assert data["context"]["used_tokens"] == 12345
    assert not list(path.parent.glob("*.tmp"))


def test_write_context_telemetry_throttles_unchanged_payload(tmp_path):
    path = tmp_path / "telemetry.json"
    payload = {"schema_version": 1, "source": "hermes_runtime"}

    assert write_context_telemetry(payload, path, min_write_interval_seconds=60) is True
    first_mtime = os.stat(path).st_mtime_ns
    assert write_context_telemetry(payload, path, min_write_interval_seconds=60) is False
    assert os.stat(path).st_mtime_ns == first_mtime
