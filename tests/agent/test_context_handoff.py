"""Tests for durable context handoff snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.context_handoff import write_context_handoff


def _agent(session_id: str = "sess-123") -> SimpleNamespace:
    """Build a minimal agent-like object for handoff tests."""
    return SimpleNamespace(
        session_id=session_id,
        model="test/model",
        provider="test-provider",
        platform="qqbot",
        _gateway_session_key="qq:chat",
        _current_turn_id="turn-abc",
        _todo_store=SimpleNamespace(format_for_injection=lambda: "TODO: keep going"),
    )


def test_write_context_handoff_creates_json_and_markdown(tmp_path: Path, monkeypatch):
    """Handoff writer persists a recoverable JSON state and human-readable note."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    messages = [
        {"role": "user", "content": "实现自动交接"},
        {"role": "assistant", "content": "我会读取文件并测试"},
        {"role": "tool", "content": "SECRET_TOKEN=should-not-leak\nall good"},
    ]

    result = write_context_handoff(
        _agent(),
        messages,
        reason="pre_compression",
        approx_tokens=123456,
        focus_topic="auto handoff",
    )

    assert result is not None
    assert result.json_path.exists()
    assert result.markdown_path.exists()
    assert result.json_path.parent == tmp_path / "handoffs"

    payload = json.loads(result.json_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["reason"] == "pre_compression"
    assert payload["session_id"] == "sess-123"
    assert payload["approx_tokens"] == 123456
    assert payload["focus_topic"] == "auto handoff"
    assert (
        payload["messages_tail"][-1]["content"] == "SECRET_TOKEN=[REDACTED]\nall good"
    )
    assert payload["todo_snapshot"] == "TODO: keep going"

    markdown = result.markdown_path.read_text()
    assert "Hermes Context Handoff" in markdown
    assert "pre_compression" in markdown
    assert "SECRET_TOKEN=should-not-leak" not in markdown
    assert "SECRET_TOKEN=[REDACTED]" in markdown


def test_write_context_handoff_uses_stable_latest_files(tmp_path: Path, monkeypatch):
    """Latest handoff paths are overwritten atomically for easy resume discovery."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    first = write_context_handoff(
        _agent(),
        [{"role": "user", "content": "first"}],
        reason="pre_compression",
    )
    second = write_context_handoff(
        _agent(),
        [{"role": "user", "content": "second"}],
        reason="compression_aborted",
        error="boom",
    )

    assert first is not None and second is not None
    latest_json = tmp_path / "handoffs" / "latest.json"
    latest_md = tmp_path / "handoffs" / "latest.md"
    assert second.json_path == latest_json
    assert second.markdown_path == latest_md
    payload = json.loads(latest_json.read_text())
    assert payload["reason"] == "compression_aborted"
    assert payload["error"] == "boom"
    assert payload["messages_tail"][0]["content"] == "second"
    assert "compression_aborted" in latest_md.read_text()
