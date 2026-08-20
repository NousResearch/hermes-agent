"""Regression coverage for the ``auto_capture: false`` lifecycle contract.

Automatic session-end, session-switch, and shutdown ingestion must remain off
when capture is disabled. Explicit user-intent saves remain available.
"""

import json
from typing import Any

import pytest

from plugins.memory.supermemory import SupermemoryMemoryProvider

TRANSCRIPT = [
    {"role": "user", "content": "Explain the compaction boundary behavior in detail."},
    {"role": "assistant", "content": "Compaction rewrites the transcript in place."},
]
BUFFERED_TURNS = [
    {"user": "A buffered user request worth capturing.", "assistant": "A buffered assistant response worth capturing."},
]


class FakeClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.ingest_calls: list[dict] = []
        self.add_calls: list[dict] = []

    def ingest_conversation(self, session_id: str, messages: list[dict], metadata=None):
        self.ingest_calls.append({"session_id": session_id, "messages": messages, "metadata": metadata})

    def add_memory(self, content: str, *args: Any, **kwargs: Any) -> dict:
        self.add_calls.append({"content": content, "args": args, "kwargs": kwargs})
        return {"id": "mem_123"}


def _make_provider(monkeypatch, tmp_path, *, auto_capture: bool):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")
    monkeypatch.setattr("plugins.memory.supermemory._SupermemoryClient", FakeClient)
    (tmp_path / "supermemory.json").write_text(
        json.dumps({"container_tag": "hermes_solar", "auto_capture": auto_capture}), encoding="utf-8"
    )
    provider = SupermemoryMemoryProvider()
    provider.initialize("session-solar", hermes_home=str(tmp_path), platform="cli")
    return provider


@pytest.fixture
def disabled(monkeypatch, tmp_path):
    return _make_provider(monkeypatch, tmp_path, auto_capture=False)


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    return _make_provider(monkeypatch, tmp_path, auto_capture=True)


def test_config_flag_is_loaded(disabled, enabled):
    assert disabled._auto_capture is False
    assert enabled._auto_capture is True


def test_session_end_does_not_ingest_when_auto_capture_disabled(disabled):
    disabled.on_session_end(TRANSCRIPT)
    assert disabled._client.ingest_calls == []


def test_compression_style_repeated_session_end_does_not_ingest_when_disabled(disabled):
    for messages in (TRANSCRIPT, TRANSCRIPT[:1], TRANSCRIPT):
        disabled.on_session_end(messages)
    assert disabled._client.ingest_calls == []


def test_session_switch_does_not_ingest_when_auto_capture_disabled(disabled):
    disabled.on_turn_start(9, "outgoing session turn")
    disabled._session_turns[:] = BUFFERED_TURNS
    disabled.on_session_switch("session-solar-2")
    assert disabled._client.ingest_calls == []
    assert disabled._session_id == "session-solar-2"
    assert disabled._session_turns == []
    assert disabled._turn_count == 0


def test_shutdown_does_not_ingest_when_auto_capture_disabled(disabled):
    disabled._session_turns[:] = BUFFERED_TURNS
    disabled.shutdown()
    assert disabled._client.ingest_calls == []


def test_enabled_preserves_all_automatic_lifecycle_ingests(enabled):
    enabled.on_session_end(TRANSCRIPT)
    enabled._session_turns[:] = BUFFERED_TURNS
    enabled.on_session_switch("session-solar-2")
    enabled._session_turns[:] = BUFFERED_TURNS
    enabled.shutdown()
    calls = enabled._client.ingest_calls
    assert [call["session_id"] for call in calls] == ["session-solar", "session-solar", "session-solar-2"]
    assert calls[0]["metadata"]["type"] == "full_session"
    assert calls[1]["metadata"]["partial"] is True
    assert calls[2]["metadata"]["partial"] is True


def test_explicit_store_tool_writes_when_auto_capture_disabled(disabled):
    result = json.loads(disabled._tool_store({"content": "Kylan prefers Canada-first market framing."}))
    assert result["saved"] is True
    assert len(disabled._client.add_calls) == 1


def test_on_memory_write_writes_when_auto_capture_disabled(disabled):
    disabled.on_memory_write("add", "MEMORY.md", "Kylan prefers Canada-first market framing.")
    if disabled._write_thread is not None:
        disabled._write_thread.join(timeout=5)
    assert len(disabled._client.add_calls) == 1
