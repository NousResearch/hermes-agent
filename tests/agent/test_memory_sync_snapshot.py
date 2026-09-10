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
    if unsupported == "partial": messages.pop()
    elif unsupported == "multimodal": messages[1]["content"] = [{"type": "text", "text": "question"}]
    elif unsupported == "oversize": messages[-1]["content"] = "a" * 64001
    elif unsupported == "missing_user": messages.pop(1)
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
