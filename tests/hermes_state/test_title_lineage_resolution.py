"""Title resolution regressions for canonical session listings."""

import pytest


@pytest.mark.parametrize("numbered", [False, True])
@pytest.mark.parametrize("lookup", ["list_session_title_candidates", "resolve_session_by_title"])
def test_title_resolution_uses_only_numeric_continuation_suffixes(db, numbered, lookup):
    titles = [("exact", "Foo")]
    if numbered:
        titles.append(("numbered", "Foo #2"))
    titles.extend([("archive", "Foo #archive"), ("nested", "Foo #archive #3")])
    for started_at, (sid, title) in enumerate(titles):
        db.create_session(sid, "cli")
        db.set_session_title(sid, title)
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (started_at, sid))
    db._conn.commit()

    result = getattr(db, lookup)("Foo")
    if lookup == "list_session_title_candidates":
        assert [row["id"] for row in result] == (["numbered", "exact"] if numbered else ["exact"])
    else:
        assert result == ("numbered" if numbered else "exact")
