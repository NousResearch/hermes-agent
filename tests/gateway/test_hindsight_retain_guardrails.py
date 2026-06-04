from __future__ import annotations

import json

import pytest


def _provider(monkeypatch):
    from plugins.memory.hindsight import HindsightMemoryProvider

    p = HindsightMemoryProvider()
    p._auto_retain = True
    p._session_id = "s1"
    p._document_id = "doc1"
    p._bank_id = "hermes"
    p._retain_max_content_chars = 50
    p._retain_max_queue_size = 1
    monkeypatch.setattr(p, "_resolve_retain_target", lambda doc: (doc, None))
    return p


def test_auto_retain_drops_oversized_session_batch(monkeypatch):
    p = _provider(monkeypatch)
    p.sync_turn("u" * 100, "a")

    assert p._retain_queue.qsize() == 0
    assert p._session_turns == []


def test_auto_retain_respects_queue_cap(monkeypatch):
    p = _provider(monkeypatch)
    p._retain_max_content_chars = 10_000
    p._retain_queue.put(lambda: None)
    p.sync_turn("u", "a")

    assert p._retain_queue.qsize() == 1


def test_tool_retain_rejects_oversized_payload(monkeypatch):
    p = _provider(monkeypatch)
    result = json.loads(p.handle_tool_call("hindsight_retain", {"content": "x" * 100}))

    assert "error" in result
    assert "too large" in result["error"]
