"""build_provider_session_import — journey "recreate this conversation".

Covers the corpus→import_sessions transform: role mapping (explicit provider
roles, first-peer heuristic, alternation fallback), same-role merging, the
stable-id/idempotency contract, timestamp provenance, and the end-to-end
write through SessionDB.import_sessions.
"""

from types import SimpleNamespace

import pytest

import agent.learning_mutations as lm


class _FakeProvider:
    def __init__(self, messages):
        self._messages = messages
        self.calls = []

    def journey_session_messages(self, session_id, limit=500):
        self.calls.append((session_id, limit))
        return self._messages


def _wire(monkeypatch, provider, name="honcho"):
    import plugins.memory as pm

    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: name)
    monkeypatch.setattr(pm, "load_memory_provider", lambda n: provider)


def _msg(content, peer="toby", role=None, timestamp=None):
    out = {"content": content, "peer": peer, "timestamp": timestamp}
    if role is not None:
        out["role"] = role
    return out


# ── Transform ────────────────────────────────────────────────────────────────


def test_explicit_roles_win(monkeypatch):
    _wire(monkeypatch, _FakeProvider([
        _msg("q1", peer="toby", role="user", timestamp=1700000000),
        _msg("a1", peer="chatgpt", role="assistant", timestamp=1700000060),
    ]))

    res = lm.build_provider_session_import("chatgpt-import-abc")

    assert res["ok"]
    s = res["session"]
    assert s["id"] == "chatgpt-import-abc"
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]
    assert s["source"] == "journey:honcho"


def test_first_peer_heuristic_when_roles_absent(monkeypatch):
    """Providers without role attribution: whoever speaks first is the user."""
    _wire(monkeypatch, _FakeProvider([
        _msg("q", peer="someone"),
        _msg("a", peer="bot"),
        _msg("q2", peer="someone"),
    ]))

    res = lm.build_provider_session_import("sess")

    assert [m["role"] for m in res["session"]["messages"]] == [
        "user", "assistant", "user",
    ]


def test_consecutive_same_role_messages_merge(monkeypatch):
    """Two user turns in a row must merge — a recreated session is continued
    live, and the conversation loop enforces strict role alternation."""
    _wire(monkeypatch, _FakeProvider([
        _msg("part one", role="user", timestamp=100),
        _msg("part two", role="user", timestamp=200),
        _msg("answer", role="assistant", timestamp=300),
        _msg("answer cont.", role="assistant"),
    ]))

    msgs = lm.build_provider_session_import("sess")["session"]["messages"]

    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "part one\n\npart two"
    assert msgs[1]["content"] == "answer\n\nanswer cont."
    # Merged turn keeps the earliest timestamp it saw.
    assert msgs[0]["timestamp"] == 100


def test_timestamps_and_title(monkeypatch):
    _wire(monkeypatch, _FakeProvider([
        _msg("How do I plant tomatoes in clay soil?", role="user", timestamp=1690000500),
        _msg("Start with compost.", role="assistant", timestamp=1690000560),
    ]))

    res = lm.build_provider_session_import("sess")
    s = res["session"]

    assert s["started_at"] == 1690000500
    assert s["title"] == "How do I plant tomatoes in clay soil?"
    assert s["messages"][0]["timestamp"] == 1690000500


def test_empty_corpus_fails_cleanly(monkeypatch):
    _wire(monkeypatch, _FakeProvider([]))
    res = lm.build_provider_session_import("gone")
    assert not res["ok"]
    assert "no source data" in res["message"]


def test_no_active_provider(monkeypatch):
    import plugins.memory as pm

    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: None)
    res = lm.build_provider_session_import("sess")
    assert not res["ok"]


def test_blank_session_id():
    assert not lm.build_provider_session_import("  ")["ok"]


# ── End-to-end through SessionDB.import_sessions ────────────────────────────


@pytest.fixture()
def session_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def test_materialized_payload_imports_and_is_idempotent(monkeypatch, session_db):
    """The built payload must be accepted verbatim by import_sessions, and a
    second materialization of the same provider session must skip (stable id
    == never-clobber): re-recreating opens the existing session instead of
    duplicating it."""
    _wire(monkeypatch, _FakeProvider([
        _msg("original question", role="user", timestamp=1690000000),
        _msg("original answer", role="assistant", timestamp=1690000060),
    ]))

    built = lm.build_provider_session_import("chatgpt-import-e2e")
    assert built["ok"]

    first = session_db.import_sessions([built["session"]])
    assert first["imported"] == 1 and not first["errors"]

    # Same conversation again → skipped, not duplicated.
    again = lm.build_provider_session_import("chatgpt-import-e2e")
    second = session_db.import_sessions([again["session"]])
    assert second["imported"] == 0
    assert second["skipped"] == 1

    msgs = session_db.get_messages("chatgpt-import-e2e")
    roles = [(m["role"], m["content"]) for m in msgs]
    assert roles == [
        ("user", "original question"),
        ("assistant", "original answer"),
    ]
    # Original provenance timestamp survived the import.
    assert abs(msgs[0]["timestamp"] - 1690000000) < 1
