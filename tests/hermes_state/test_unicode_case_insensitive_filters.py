"""Case-insensitive session filters must fold case in every script, not just ASCII.

SQLite's built-in ``LOWER()`` and ``LIKE`` fold ASCII letters only, so comparing a
Python-lowercased needle with ``LOWER(title)`` never matched an uppercase Cyrillic,
Greek or accented letter: a sentence-case title like "Обновление Гермеса" could not
be found by its own first word in ``/sessions search``, ``GET /api/sessions?title=``
or ``hermes sessions prune/archive --title``, and the LIKE message scan
(``role_filter`` with ``tool``, FTS fail-open, mid-rebuild gap) missed non-ASCII
words that differed in case. Each filter must agree with Python's casefolded
substring test.
"""

import pytest

from hermes_state import SessionDB

TEXTS = (
    "Обновление Гермеса",
    "Ενημέρωση Συστήματος",
    "Résumé du Café",
    "Größe der Übersicht",
    "Update Hermes Config",
)
NEEDLES = sorted({
    variant
    for text in TEXTS
    for word in text.split()
    for variant in (word, word.lower(), word.upper(), word.title())
})


def _expected(needle, rows):
    return {key for key, text in rows.items() if needle.casefold() in text.casefold()}


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def test_session_filters_match_non_ascii_titles_in_any_case(db):
    titles = {}
    for index, text in enumerate(TEXTS):
        sid = f"s{index}"
        db.create_session(session_id=sid, source="cli")
        db.set_session_title(sid, text)
        db._conn.execute("UPDATE sessions SET model = ?, git_branch = ? WHERE id = ?", (text, text, sid))
        db.append_message(sid, role="user", content="x")
        db.end_session(sid, "user_exit")
        titles[sid] = text

    misses = []
    for needle in NEEDLES:
        expected = _expected(needle, titles)
        searched = db.list_sessions_rich(limit=50, search_query=needle, order_by_last_active=True)
        if {row["id"] for row in searched} != expected:
            misses.append(("search_query", needle))
        for prune_filter in ("title_like", "model_like", "branch_like"):
            if {row["id"] for row in db.list_prune_candidates(**{prune_filter: needle})} != expected:
                misses.append((prune_filter, needle))
    assert misses == []


def test_like_message_scan_matches_non_ascii_words_in_any_case(db):
    db.create_session(session_id="s1", source="cli")
    tool_rows = {}
    for index, text in enumerate((*TEXTS, "x")):
        call_id = f"call{index}"
        db.append_message("s1", role="assistant", content="", tool_calls=[
            {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
        tool_rows[db.append_message("s1", role="tool", content=text, tool_call_id=call_id,
                                    tool_name="terminal")] = text
    # A row stored as invalid UTF-8 (corruption, a foreign writer) must not fail the whole scan.
    broken = max(tool_rows)
    db._conn.execute("UPDATE messages SET content = CAST(X'FF80FE' AS TEXT) WHERE id = ?", (broken,))

    misses = [
        needle for needle in NEEDLES
        if {hit["id"] for hit in db.search_messages(needle, role_filter=["tool"], limit=50)}
        != _expected(needle, tool_rows)
    ]
    assert misses == []


@pytest.mark.parametrize("needle", ["ss", "SS", "Ss", "ß", "fi", "FI"])
def test_ascii_needle_finds_casefold_equal_spelling_in_every_consumer(db, needle):
    """An ASCII needle ("ss") must find its casefold-equal non-ASCII spelling ("ß") in the LIKE
    message scan exactly as the title/model/branch filters do: gating the fold on the needle's
    script made the sidebar search miss a session that ``--title ss`` listed."""
    text = "das ß ist hier, ﬁle"  # only spellings: sharp s and the U+FB01 ligature, no ASCII "ss"/"fi"
    db.create_session(session_id="s1", source="cli")
    db.set_session_title("s1", text)
    db._conn.execute("UPDATE sessions SET model = ?, git_branch = ? WHERE id = 's1'", (text, text))
    db.append_message("s1", role="assistant", content="", tool_calls=[
        {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
    tool_row = db.append_message("s1", role="tool", content=text, tool_call_id="c1", tool_name="terminal")
    db.end_session("s1", "user_exit")
    assert needle.casefold() in text.casefold()

    assert [hit["id"] for hit in db.search_messages(needle, role_filter=["tool"], limit=10)] == [tool_row]
    for prune_filter in ("title_like", "model_like", "branch_like"):
        assert [row["id"] for row in db.list_prune_candidates(**{prune_filter: needle})] == ["s1"]
    assert [row["id"] for row in db.list_sessions_rich(limit=10, search_query=needle)] == ["s1"]
