"""Tests for agent.sleep_engine."""

from pathlib import Path

from agent.sleep_engine import SessionStats, SleepEngine


class _FakeMemoryStore:
    def __init__(self, entries):
        self.memory_entries = list(entries)
        self.saved_targets = []

    def save_to_disk(self, target: str):
        self.saved_targets.append(target)


def _session(session_id: str = "s1", importance_score: float = 0.0):
    return SessionStats(
        session_id=session_id,
        started_at=0.0,
        ended_at=3600.0,
        message_count=20,
        tool_call_count=3,
        title="test",
        source="cli",
        duration_hours=1.0,
        importance_score=importance_score,
    )


def test_sleep_rejects_unknown_mode():
    engine = SleepEngine(_FakeMemoryStore([]), db_path=Path("/tmp/missing.db"))

    result = engine.sleep("nap")

    assert result["success"] is False
    assert "Unsupported sleep mode" in result["error"]


def test_sleep_succeeds_when_no_memories_exist(monkeypatch):
    engine = SleepEngine(_FakeMemoryStore([]), db_path=Path("/tmp/missing.db"))
    sessions = [_session(importance_score=0.8), _session("s2", importance_score=0.2)]

    monkeypatch.setattr(engine, "_get_all_sessions", lambda: sessions)
    monkeypatch.setattr(
        engine,
        "learn_vocabulary",
        lambda _sessions: ({"project": 1.4}, {"weather": -0.8}),
    )

    result = engine.sleep("quick")

    assert result["success"] is True
    assert result["applied"] is False
    assert result["stats"]["sessions_analyzed"] == 2
    assert result["stats"]["memories_before"] == 0
    assert result["stats"]["memories_deleted"] == 0
    assert result["vocabulary"]["important_words_count"] == 1


def test_sleep_preview_does_not_persist_changes(monkeypatch):
    store = _FakeMemoryStore(["important project note", "weather lookup"])
    engine = SleepEngine(store, db_path=Path("/tmp/missing.db"))
    sessions = [_session(importance_score=0.9), _session("s2", importance_score=0.1)]

    monkeypatch.setattr(engine, "_get_all_sessions", lambda: sessions)
    monkeypatch.setattr(engine, "learn_vocabulary", lambda _sessions: ({}, {}))
    monkeypatch.setattr(
        engine,
        "_score_memory",
        lambda text: (0.9, {}) if "important" in text else (0.1, {}),
    )

    result = engine.sleep("quick")

    assert result["success"] is True
    assert result["applied"] is False
    assert store.memory_entries == ["important project note", "weather lookup"]
    assert store.saved_targets == []
    assert result["stats"]["memories_before"] == 2
    assert result["stats"]["memories_deleted"] == 1


def test_sleep_apply_persists_deletions(monkeypatch):
    store = _FakeMemoryStore(["important project note", "weather lookup"])
    engine = SleepEngine(store, db_path=Path("/tmp/missing.db"))
    sessions = [_session(importance_score=0.9), _session("s2", importance_score=0.1)]

    monkeypatch.setattr(engine, "_get_all_sessions", lambda: sessions)
    monkeypatch.setattr(engine, "learn_vocabulary", lambda _sessions: ({}, {}))
    monkeypatch.setattr(
        engine,
        "_score_memory",
        lambda text: (0.9, {}) if "important" in text else (0.1, {}),
    )

    result = engine.sleep("quick", apply_changes=True)

    assert result["success"] is True
    assert result["applied"] is True
    assert store.memory_entries == ["important project note"]
    assert store.saved_targets == ["memory"]
    assert result["stats"]["memories_before"] == 2
    assert result["stats"]["memories_deleted"] == 1


def test_sleep_feature_is_exposed_through_tooling():
    from model_tools import get_all_tool_names
    from toolsets import resolve_toolset

    all_tools = get_all_tool_names()

    assert "sleep_memory" in all_tools
    assert "dream_memory" not in all_tools
    assert "sleep_memory" in resolve_toolset("memory")
