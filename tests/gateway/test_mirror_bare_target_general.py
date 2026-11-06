"""A bare (thread-less) group target mirrors into the General room, not the newest room.

Reported on a live install, 2026-10-17 (reproduced 2×): ``hermes send -t telegram:<group> ...``
(a bare target = the chat's General room, thread 1) delivered fine, but its delivery-mirror
landed in the NEWEST-STARTED live thread session of the chat — thread 95 — instead of the
General session (thread 1). The General agent then never saw the outgoing ask in context, so
later text replies could be mis-mapped to the wrong room's question.

Root cause: ``_find_session_id`` passed ``thread_id=None`` straight to
``find_session_by_origin``, whose thread-less scan has no thread filter and returns
``rows[0]`` ordered ``started_at DESC``. Its multi-user refusal only fires when several
distinct users exist, so a single-user threaded chat resolved to its newest room.

The fix: a bare target probes thread "1" (General) first, then the flat session "";
neither → None (drop the mirror rather than guess). Both lookup paths — state.db and the
sessions.json pre-migration fallback — take the same probes, so the fallback cannot
regress on its own.
"""

import json
from unittest.mock import patch

import pytest

import gateway.mirror as mirror_mod
from gateway.mirror import _find_session_id, mirror_to_session
from hermes_state import SessionDB

CHAT = "-1001"  # the reported shape: a Telegram forum group


def _thread_session(db, session_id, thread_id, *, user_id="u1"):
    db.create_session(
        session_id, "telegram", user_id=user_id,
        session_key=f"agent:main:telegram:forum:{CHAT}:{user_id}:{thread_id}",
        chat_id=CHAT, chat_type="forum", thread_id=thread_id,
    )


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


@pytest.fixture()
def wired(db, tmp_path):
    """Point mirror's lookups at the tmp SessionDB; the sessions.json fallback stays absent."""
    with patch("hermes_state_registry.acquire", return_value=db), \
         patch("hermes_state_registry.release_or_close"), \
         patch.object(mirror_mod, "_SESSIONS_INDEX", tmp_path / "sessions" / "sessions.json"):
        yield db


def test_bare_target_resolves_to_general_not_newest_room(wired):
    """The reported bug: General (thread 1) is live, thread 95 started later."""
    _thread_session(wired, "sess-general", "1")
    _thread_session(wired, "sess-newer", "95")

    assert _find_session_id("telegram", CHAT) == "sess-general"


def test_bare_target_mirror_lands_in_general(wired):
    """End to end: the delivery-mirror row lands in the General session's transcript."""
    _thread_session(wired, "sess-general", "1")
    _thread_session(wired, "sess-newer", "95")

    assert mirror_to_session("telegram", CHAT, "🟡 the ask", source_label="cli", user_id="u1") is True
    general = wired.get_messages("sess-general")
    assert [m["role"] for m in general] == ["assistant"]
    assert "the ask" in general[0]["content"]
    assert wired.get_messages("sess-newer") == []


def test_threaded_chat_without_general_refuses_instead_of_guessing(wired):
    """No live General (or flat) session: drop the mirror rather than contaminate a room."""
    _thread_session(wired, "sess-newer", "95")
    _thread_session(wired, "sess-other", "59")

    assert _find_session_id("telegram", CHAT) is None
    assert mirror_to_session("telegram", CHAT, "🟡 the ask", source_label="cli", user_id="u1") is False
    assert wired.get_messages("sess-newer") == []
    assert wired.get_messages("sess-other") == []


def test_flat_chat_still_resolves(wired):
    """Control: DMs / non-threaded chats (no thread_id at all) keep resolving."""
    wired.create_session("sess-dm", "telegram", user_id="u1",
                         session_key="agent:main:telegram:dm:u1", chat_id="123", chat_type="dm")

    assert _find_session_id("telegram", "123") == "sess-dm"


def test_explicit_thread_suffix_still_wins(wired):
    """Control: a thread-suffixed target keeps resolving to exactly that room."""
    _thread_session(wired, "sess-general", "1")
    _thread_session(wired, "sess-newer", "95")

    assert _find_session_id("telegram", CHAT, thread_id="95") == "sess-newer"


def test_sessions_json_fallback_prefers_thread1(tmp_path):
    """Pre-migration fallback path parity: the same bare-target rule applies there."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    index = sessions_dir / "sessions.json"
    index.write_text(json.dumps({
        "general": {"session_id": "sess-general",
                    "origin": {"platform": "telegram", "chat_id": CHAT, "thread_id": "1"},
                    "updated_at": "2026-01-01T00:00:00"},
        "newer": {"session_id": "sess-newer",
                  "origin": {"platform": "telegram", "chat_id": CHAT, "thread_id": "95"},
                  "updated_at": "2026-02-01T00:00:00"},
    }), encoding="utf-8")
    no_rows = type("NoRows", (), {"find_session_by_origin": staticmethod(lambda **kw: None)})()

    with patch("hermes_state_registry.acquire", return_value=no_rows), \
         patch("hermes_state_registry.release_or_close"), \
         patch.object(mirror_mod, "_SESSIONS_INDEX", index):
        assert _find_session_id("telegram", CHAT) == "sess-general"
