from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_sync_snapshot import snapshot_completed_turn


@pytest.mark.parametrize("unsupported", [None, "partial", "multimodal", "oversize", "missing_user"])
def test_enqueue_snapshot_is_detached_and_legacy_provider_keeps_history(monkeypatch, unsupported):
    calls, queued = [], []
    def provider(version):
        return SimpleNamespace(name="snapshot" if version else "builtin", sync_turn_snapshot_version=version, get_tool_schemas=lambda: [],
                               sync_turn=lambda *args, **kwargs: calls.append(kwargs["messages"]))
    manager = MemoryManager()
    manager.add_provider(provider(1))
    manager.add_provider(provider(0))
    messages = [dict(role="system", content="history"), dict(role="user", content="question"),
                dict(role="assistant", content="answer", _row_id=42, _db_persisted=True,
                     tool_calls=[])]
    mutations = {
        "partial": lambda: messages.pop(),
        "multimodal": lambda: messages[1].update(content=[{"type": "text", "text": "question"}]),
        "oversize": lambda: messages[-1].update(content="a" * 64001),
        "missing_user": lambda: messages.pop(1),
    }
    if unsupported:
        mutations[unsupported]()
    before = deepcopy(messages)
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: queued.append(fn))
    manager.sync_all("question", "answer", session_id="session", messages=messages)
    assert messages == before
    messages[-1]["tool_calls"] = [{"function": {"arguments": "mutated"}}]
    queued[0]()
    assert calls[1] is messages
    if unsupported:
        assert calls[0] is None
    else:
        assert calls[0].messages() == before[1:]
        assert calls[0].session_id == "session"
        decoded = calls[0].messages()
        decoded[-1]["tool_calls"].append({"changed": True})
        assert calls[0].messages() == before[1:]


def test_snapshot_does_not_invent_persistence_or_enqueue_interrupted_turn():
    messages = [dict(role="user", content="q"), dict(role="assistant", content="a")]
    snapshot = snapshot_completed_turn(messages, session_id="s", user_content="q", assistant_content="a")
    assert snapshot.messages() == messages
    from run_agent import AIAgent
    agent = SimpleNamespace(_memory_manager=SimpleNamespace(sync_all=lambda *a, **kw: pytest.fail("interrupted sync")))
    AIAgent._sync_external_memory_for_turn(agent, original_user_message="q", final_response="a",
                                         interrupted=True, messages=messages)


def test_snapshot_preserves_event_timestamps_without_copying_untrusted_fields():
    messages = [
        {"role": "user", "content": "q", "timestamp": "2025-01-02T03:04:05+00:00",
         "private_field": "must not cross the boundary"},
        {"role": "assistant", "content": "a", "timestamp": "2025-01-02T03:04:07+00:00"},
    ]
    snapshot = snapshot_completed_turn(messages, session_id="s", user_content="q", assistant_content="a")

    assert snapshot.messages() == [
        {"role": "user", "content": "q", "timestamp": "2025-01-02T03:04:05+00:00"},
        {"role": "assistant", "content": "a", "timestamp": "2025-01-02T03:04:07+00:00"},
    ]


@pytest.mark.parametrize("version", [0, 1])
def test_queued_author_is_detached_from_caller_and_other_providers(monkeypatch, version):
    calls, queued = [], []
    manager = MemoryManager()
    def capture(*args, **kwargs):
        calls.append(deepcopy(kwargs["turn_author"]))
        kwargs["turn_author"]["name"] = "provider mutation"
        kwargs["turn_author"]["context"]["labels"].append("provider mutation")
    for name in ("builtin", "external"):
        manager.add_provider(SimpleNamespace(
            name=name, sync_turn_snapshot_version=version,
            get_tool_schemas=lambda: [], sync_turn=capture))
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    author = {"id": "alice", "name": "Alice", "is_bot": False,
              "context": {"labels": ["original"]}}
    before = deepcopy(author)
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: queued.append(fn))
    manager.sync_all("q", "a", session_id="s", messages=messages, turn_author=author)
    author["id"] = "bob"
    author["context"]["labels"].append("caller mutation")
    queued[0]()
    assert calls == [before, before]
    assert author["name"] == "Alice"
    assert author["context"]["labels"] == ["original", "caller mutation"]


def test_snapshot_preserves_clocks_without_bypassing_provider_redaction(monkeypatch):
    secret = "sk-proj-abc123def456ghi789jkl012"
    calls, queued = [], []
    manager = MemoryManager()
    manager.add_provider(SimpleNamespace(
        name="snapshot", sync_turn_snapshot_version=1,
        get_tool_schemas=lambda: [],
        sync_turn=lambda *args, **kwargs: calls.append((args, kwargs["messages"]))))
    user, assistant = f"OPENAI_API_KEY={secret}", f"OPENAI_API_KEY={secret} noted"
    messages = [
        {"role": "user", "content": user, "timestamp": "2025-01-02T03:04:05+00:00"},
        {"role": "assistant", "content": assistant, "timestamp": "2025-01-02T03:04:07+00:00"},
    ]
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: queued.append(fn))
    manager.sync_all(user, assistant, session_id="s", messages=messages)
    queued[0]()
    args, snapshot = calls[0]
    assert snapshot is not None
    rows = snapshot.messages()
    assert secret not in repr(args) and secret not in repr(rows)
    assert [row["timestamp"] for row in rows] == [row["timestamp"] for row in messages]
    assert secret in messages[0]["content"]
