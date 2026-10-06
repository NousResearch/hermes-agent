"""Regression coverage for completed-turn snapshots with event timestamps."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_sync_snapshot import snapshot_completed_turn


def _providers(calls):
    def make(version):
        return SimpleNamespace(
            name="snapshot" if version else "builtin",
            sync_turn_snapshot_version=version,
            get_tool_schemas=lambda: [],
            sync_turn=lambda *args, **kwargs: calls.append(kwargs["messages"]),
        )

    return make(1), make(0)


def test_snapshot_keeps_valid_timestamps_and_drops_private_fields():
    messages = [
        {
            "role": "user",
            "content": "q",
            "timestamp": "2025-01-02T03:04:05+00:00",
            "private_field": "must not cross the boundary",
        },
        {"role": "assistant", "content": "a", "timestamp": "2025-01-02T03:04:07+00:00"},
    ]
    snapshot = snapshot_completed_turn(messages, session_id="s", user_content="q", assistant_content="a")
    assert snapshot is not None
    assert snapshot.messages() == [
        {"role": "user", "content": "q", "timestamp": "2025-01-02T03:04:05+00:00"},
        {"role": "assistant", "content": "a", "timestamp": "2025-01-02T03:04:07+00:00"},
    ]


def test_snapshot_omits_missing_timestamps_and_rejects_invalid_timestamp():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "timestamp": {"epoch": 123}},
    ]
    snapshot = snapshot_completed_turn(messages, session_id="s", user_content="q", assistant_content="a")
    assert snapshot is not None
    assert snapshot.messages() == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_snapshot_detaches_nested_tool_calls_and_enforces_bounds(monkeypatch):
    calls, queued = [], []
    snapshot_provider, legacy_provider = _providers(calls)
    manager = MemoryManager()
    manager.add_provider(snapshot_provider)
    manager.add_provider(legacy_provider)
    messages = [
        {"role": "system", "content": "history"},
        {"role": "user", "content": "question", "timestamp": 1700000000},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "lookup", "arguments": '{"q": "x"}'}}],
        },
        {"role": "tool", "content": "result", "tool_call_id": "c1"},
        {
            "role": "assistant",
            "content": "answer",
            "timestamp": 1700000005,
            "_row_id": 42,
            "_db_persisted": True,
        },
    ]
    before = deepcopy(messages)
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: queued.append(fn))
    manager.sync_all("question", "answer", session_id="session", messages=messages)
    assert messages == before
    messages[2]["tool_calls"][0]["function"]["arguments"] = "mutated"
    queued[0]()
    # Legacy provider keeps the live history reference.
    assert calls[1] is messages
    # Snapshot provider gets the detached enqueue-time turn with timestamps.
    assert calls[0] is not None
    assert calls[0].session_id == "session"
    assert calls[0].messages() == before[1:]
    decoded = calls[0].messages()
    decoded[1]["tool_calls"].append({"changed": True})
    assert calls[0].messages() == before[1:]


@pytest.mark.parametrize("unsupported", ["partial", "multimodal", "oversize", "missing_user"])
def test_snapshot_returns_none_for_unsupported_context(monkeypatch, unsupported):
    calls, queued = [], []
    snapshot_provider, legacy_provider = _providers(calls)
    manager = MemoryManager()
    manager.add_provider(snapshot_provider)
    manager.add_provider(legacy_provider)
    messages = [
        {"role": "system", "content": "history"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "_row_id": 7, "_db_persisted": True},
    ]
    if unsupported == "partial":
        messages.pop()
    elif unsupported == "multimodal":
        messages[1]["content"] = [{"type": "text", "text": "question"}]
    elif unsupported == "oversize":
        messages[-1]["content"] = "a" * 64001
    elif unsupported == "missing_user":
        messages.pop(1)
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: queued.append(fn))
    manager.sync_all("question", "answer", session_id="session", messages=messages)
    queued[0]()
    assert calls[0] is None
    assert calls[1] is messages


def test_snapshot_skips_interrupted_turn():
    from run_agent import AIAgent

    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    snapshot = snapshot_completed_turn(messages, session_id="s", user_content="q", assistant_content="a")
    assert snapshot.messages() == messages
    agent = SimpleNamespace(
        _memory_manager=SimpleNamespace(sync_all=lambda *a, **kw: pytest.fail("interrupted sync"))
    )
    AIAgent._sync_external_memory_for_turn(
        agent,
        original_user_message="q",
        final_response="a",
        interrupted=True,
        messages=messages,
    )
