"""Tests for /recall command — gateway/recall.py."""

import pytest
from gateway.recall import (
    RecallSpec,
    RecallResult,
    SourceRef,
    parse_recall_args,
    _ago,
    _format_transcript,
    _format_header,
)


# ---------------------------------------------------------------------------
# parse_recall_args
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("",           RecallSpec(mode="thread", count=1)),
    ("  ",         RecallSpec(mode="thread", count=1)),
    ("3",          RecallSpec(mode="thread", count=3)),
    ("10",         RecallSpec(mode="thread", count=10)),
    ("99",         RecallSpec(mode="thread", count=10)),   # capped at 10
    ("last 5",     RecallSpec(mode="thread", count=5)),
    ("Last 2",     RecallSpec(mode="thread", count=2)),
    ("last 99",    RecallSpec(mode="thread", count=10)),   # capped
    ("24h",        RecallSpec(mode="window", window_hours=24)),
    ("7d",         RecallSpec(mode="window", window_hours=168)),
    ("2w",         RecallSpec(mode="window", window_hours=336)),
    ("3m",         RecallSpec(mode="window", window_hours=2160)),
    ("hex migration",        RecallSpec(mode="topic", query="hex migration")),
    ("bosun spec",           RecallSpec(mode="topic", query="bosun spec")),
    ("hex migration 7d",     RecallSpec(mode="topic", query="hex migration", window_hours=168)),
    ("spool stuck 24h",      RecallSpec(mode="topic", query="spool stuck", window_hours=24)),
])
def test_parse_recall_args(raw, expected):
    assert parse_recall_args(raw) == expected


# ---------------------------------------------------------------------------
# _ago
# ---------------------------------------------------------------------------

def test_ago_seconds():
    import time
    assert "just now" in _ago(time.time() - 30)

def test_ago_minutes():
    import time
    s = _ago(time.time() - 600)
    assert "m ago" in s

def test_ago_hours():
    import time
    s = _ago(time.time() - 7200)
    assert "h ago" in s

def test_ago_days():
    import time
    s = _ago(time.time() - 86400 * 3)
    assert "d ago" in s


# ---------------------------------------------------------------------------
# _format_transcript
# ---------------------------------------------------------------------------

def test_format_transcript_basic():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "tool", "content": "ignored"},
    ]
    out = _format_transcript(msgs, char_cap=5000)
    assert "USER: hello" in out
    assert "ASSISTANT: world" in out
    assert "tool" not in out.lower() or "ignored" not in out

def test_format_transcript_truncates():
    msgs = [{"role": "user", "content": "x" * 3000}]
    out = _format_transcript(msgs, char_cap=5000)
    assert "[...]" in out  # long individual msg gets trimmed at 2000 chars

def test_format_transcript_cap():
    msgs = [{"role": "user", "content": "abc"} for _ in range(1000)]
    out = _format_transcript(msgs, char_cap=500)
    assert "truncated" in out


# ---------------------------------------------------------------------------
# _format_header
# ---------------------------------------------------------------------------

def test_format_header_thread_single():
    import time
    refs = [SourceRef("sid1", "Test", 10, time.time() - 3600)]
    spec = RecallSpec(mode="thread", count=1)
    h = _format_header(spec, refs)
    assert "prior session in this thread" in h
    assert "10 msgs" in h

def test_format_header_thread_multi():
    import time
    refs = [SourceRef(f"s{i}", "T", 5, time.time() - 3600) for i in range(3)]
    spec = RecallSpec(mode="thread", count=3)
    h = _format_header(spec, refs)
    assert "last 3" in h

def test_format_header_window():
    import time
    refs = [SourceRef("s1", "T", 5, time.time() - 3600)]
    spec = RecallSpec(mode="window", window_hours=24)
    h = _format_header(spec, refs)
    assert "24h" in h

def test_format_header_topic():
    import time
    refs = [SourceRef("s1", "T", 5, time.time() - 3600)]
    spec = RecallSpec(mode="topic", query="hex migration")
    h = _format_header(spec, refs)
    assert "hex migration" in h


# ---------------------------------------------------------------------------
# resolve_sources — integration-style with fake stores
# ---------------------------------------------------------------------------

class FakeSessionDB:
    """Minimal fake for hermes_state.SessionDB."""

    def __init__(self, sessions: list[dict], messages: dict | None = None):
        self._sessions = sessions
        self._messages: dict[str, list[dict]] = messages or {}

    def get_session(self, session_id: str):
        return next((dict(r) for r in self._sessions if r.get("id") == session_id), None)

    def search_sessions(self, source: str = None, limit: int = 20, offset: int = 0):
        rows = self._sessions
        if source:
            rows = [r for r in rows if r.get("source") == source]
        return rows[:limit]

    def message_count(self, session_id: str = None) -> int:
        if session_id:
            return len(self._messages.get(session_id, []))
        return sum(len(v) for v in self._messages.values())

    def search_messages(self, query: str, limit: int = 20, **kwargs):
        # Very simple: return rows where query appears in content
        results = []
        for sid, msgs in self._messages.items():
            for msg in msgs:
                if query.lower() in (msg.get("content") or "").lower():
                    # find session row
                    sess = next((s for s in self._sessions if s["id"] == sid), {})
                    results.append({
                        "session_id": sid,
                        "content": msg.get("content", ""),
                        "session_title": sess.get("title", f"Session {sid[:8]}"),
                        "session_started_at": sess.get("started_at", 0),
                        "timestamp": 0,
                    })
        return results[:limit]


class FakeSessionStore:
    def __init__(self, entries: dict):
        self._entries = entries

    def load_transcript(self, session_id: str) -> list[dict]:
        return []


def _make_ts(hours_ago: float) -> float:
    import time
    return time.time() - hours_ago * 3600


def test_resolve_sources_thread_default():
    from gateway.recall import resolve_sources
    # Sessions linked by parent_session_id chain (real structure after /new).
    # current → sid-a → sid-b  (sid-c is a different thread, unlinked)
    sessions = [
        {"id": "sid-current", "source": "telegram", "title": "Current",
         "parent_session_id": "sid-a", "started_at": _make_ts(0), "ended_at": None},
        {"id": "sid-a", "source": "telegram", "title": "Alpha",
         "parent_session_id": "sid-b", "started_at": _make_ts(2), "ended_at": _make_ts(1)},
        {"id": "sid-b", "source": "telegram", "title": "Beta",
         "parent_session_id": None, "started_at": _make_ts(5), "ended_at": _make_ts(3)},
        {"id": "sid-c", "source": "telegram", "title": "Other thread",
         "parent_session_id": None, "started_at": _make_ts(1), "ended_at": _make_ts(0.5)},
    ]
    db = FakeSessionDB(sessions)
    store = FakeSessionStore({"sid-current": {}})
    store._entries = {"tg:123:456": type("E", (), {"session_id": "sid-current"})()}

    spec = RecallSpec(mode="thread", count=1)
    refs = resolve_sources(spec, "tg:123:456", store, db)
    assert len(refs) == 1
    assert refs[0].session_id == "sid-a"


def test_resolve_sources_thread_count():
    from gateway.recall import resolve_sources
    # Build a chain of 5 sessions: current → s4 → s3 → s2 → s1 → s0
    sessions = [{"id": "sid-current", "source": "telegram", "title": "Current",
                 "parent_session_id": "sid-4", "started_at": _make_ts(0), "ended_at": None}]
    for i in range(4, -1, -1):
        sessions.append({
            "id": f"sid-{i}", "source": "telegram", "title": f"S{i}",
            "parent_session_id": f"sid-{i-1}" if i > 0 else None,
            "started_at": _make_ts(i + 1), "ended_at": _make_ts(i + 0.5),
        })
    db = FakeSessionDB(sessions)
    store = FakeSessionStore({})
    store._entries = {"tg:1:1": type("E", (), {"session_id": "sid-current"})()}

    spec = RecallSpec(mode="thread", count=3)
    refs = resolve_sources(spec, "tg:1:1", store, db)
    assert len(refs) == 3
    assert refs[0].session_id == "sid-4"
    assert refs[1].session_id == "sid-3"
    assert refs[2].session_id == "sid-2"


def test_resolve_sources_window():
    from gateway.recall import resolve_sources
    import time
    now = time.time()
    # current → new (1h ago) → old (48h ago)
    sessions = [
        {"id": "sid-current", "source": "telegram", "title": "Current",
         "parent_session_id": "new", "started_at": now - 600, "ended_at": None},
        {"id": "new", "source": "telegram", "title": "New",
         "parent_session_id": "old", "started_at": now - 3600, "ended_at": now - 700},
        {"id": "old", "source": "telegram", "title": "Old",
         "parent_session_id": None, "started_at": now - 48*3600, "ended_at": now - 47*3600},
    ]
    db = FakeSessionDB(sessions)
    store = FakeSessionStore({})
    store._entries = {"tg:1:1": type("E", (), {"session_id": "sid-current"})()}

    spec = RecallSpec(mode="window", window_hours=24)
    refs = resolve_sources(spec, "tg:1:1", store, db, now=now)
    assert len(refs) == 1
    assert refs[0].session_id == "new"


def test_resolve_sources_topic():
    from gateway.recall import resolve_sources
    sessions = [
        {"id": "s1", "source": "tg:1:1", "title": "Spool incident", "started_at": _make_ts(1)},
        {"id": "s2", "source": "tg:2:2", "title": "Unrelated", "started_at": _make_ts(2)},
    ]
    messages = {
        "s1": [{"role": "user", "content": "the spool got stuck again"}],
        "s2": [{"role": "user", "content": "weather is nice"}],
    }
    db = FakeSessionDB(sessions, messages)
    store = FakeSessionStore({})
    store._entries = {}

    spec = RecallSpec(mode="topic", query="spool")
    refs = resolve_sources(spec, "tg:1:1", store, db)
    assert any(r.session_id == "s1" for r in refs)
    assert all(r.session_id != "s2" for r in refs)


def test_resolve_sources_no_db():
    from gateway.recall import resolve_sources
    spec = RecallSpec(mode="thread", count=1)
    refs = resolve_sources(spec, "tg:1:1", None, None)
    assert refs == []
