"""The canonical Bot Chat's title is its identity — renames must be refused.

Bot Mode resolves a bot's forever-chat by exact-title lookup on
(profile, "Bot Chat") every time it opens; there is no session-id pointer.
A user rename therefore orphans the whole conversation: resolution misses,
the next click mints an empty replacement, and UNIQUE(title) then blocks
renaming the original back (#92473).

The guard lives in SessionDB._set_session_title — the single write path
every rename surface funnels through (gateway session.title RPC, /title,
CLI rename, REST) — and keys on hidden + exact canonical title so ordinary
sessions a user happens to call "Bot Chat" stay freely renameable.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _make_canonical(db, session_id="forever"):
    db.create_session(session_id, source="desktop")
    assert db.set_session_title(session_id, SessionDB.CANONICAL_BOT_CHAT_TITLE)
    assert db.set_session_hidden(session_id, True)
    return session_id


def test_user_rename_of_canonical_bot_chat_is_refused(db):
    sid = _make_canonical(db)
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        db.set_session_title(sid, "My cool chat")
    # Identity intact: exact-title lookup still finds the forever chat.
    row = db.get_session_by_title(SessionDB.CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid


def test_clearing_the_canonical_title_is_refused(db):
    sid = _make_canonical(db)
    with pytest.raises(ValueError, match="canonical Bot Chat"):
        db.set_session_title(sid, "")
    row = db.get_session_by_title(SessionDB.CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid


def test_rewriting_the_same_canonical_title_is_a_noop_not_an_error(db):
    # The plugin's eager session.title write re-asserts the canonical title
    # on creation paths; that must never start failing.
    sid = _make_canonical(db)
    assert db.set_session_title(sid, SessionDB.CANONICAL_BOT_CHAT_TITLE)


def test_visible_session_titled_bot_chat_stays_renameable(db):
    # hidden discriminates the registry row: a normal visible session the
    # user happened to name "Bot Chat" is not canonical and renames freely.
    db.create_session("ordinary", source="cli")
    assert db.set_session_title("ordinary", SessionDB.CANONICAL_BOT_CHAT_TITLE)
    assert db.set_session_title("ordinary", "renamed away")
    assert db.get_session("ordinary")["title"] == "renamed away"


def test_auto_titler_still_cannot_touch_the_canonical_row(db):
    # Pre-existing provenance contract, re-pinned here: user-authority title
    # outranks derived/llm, so the turn-start auto-titler can never displace
    # the registry name.
    sid = _make_canonical(db)
    assert not db.set_auto_title(sid, "Chat about groceries", source=SessionDB.TITLE_SOURCE_LLM)
    row = db.get_session_by_title(SessionDB.CANONICAL_BOT_CHAT_TITLE)
    assert row and row["id"] == sid


def test_auto_titler_cannot_rename_derived_canonical_bot_chat(db):
    # #99517: the guard must be provenance-blind. A derived (rank 0) canonical
    # title loses to an llm (rank 1) auto-title on precedence alone, so the
    # identity check — not precedence — has to stop the write.
    db.create_session("derived", source="desktop")
    assert db._set_session_title(
        "derived",
        SessionDB.CANONICAL_BOT_CHAT_TITLE,
        source=SessionDB.TITLE_SOURCE_DERIVED,
    )
    assert db.set_session_hidden("derived", True)

    assert not db.set_auto_title(
        "derived",
        "Renamed by titler",
        source=SessionDB.TITLE_SOURCE_LLM,
    )
    row = db.get_session("derived")
    assert row["title"] == SessionDB.CANONICAL_BOT_CHAT_TITLE
    assert row["title_source"] == SessionDB.TITLE_SOURCE_DERIVED


def test_auto_titler_can_rename_visible_derived_bot_chat(db):
    # Control: hidden is still the discriminator — a visible session that
    # merely carries the text "Bot Chat" upgrades derived -> llm as usual.
    db.create_session("visible", source="desktop")
    assert db._set_session_title(
        "visible",
        SessionDB.CANONICAL_BOT_CHAT_TITLE,
        source=SessionDB.TITLE_SOURCE_DERIVED,
    )

    assert db.set_auto_title(
        "visible",
        "Renamed by titler",
        source=SessionDB.TITLE_SOURCE_LLM,
    )
    assert db.get_session("visible")["title"] == "Renamed by titler"
