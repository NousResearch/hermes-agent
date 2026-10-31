import string

from hermes_state import SessionDB


def test_punctuation_never_changes_search_results(tmp_path):
    """A trailing full stop (or any stray punctuation) must not turn a hit into an
    FTS5 syntax error that search_messages swallows as zero results (#146747)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="the deploy target is my-app.config")
        for c in string.punctuation:
            for q in (f"deploy target{c}", f"deploy {c} target", f"{c}deploy target"):
                assert db.search_messages(q), q
        assert db.search_messages("my-app.config.")
    finally:
        db.close()
