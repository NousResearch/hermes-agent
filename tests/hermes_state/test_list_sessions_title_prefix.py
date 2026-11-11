"""``list_sessions_rich(title_prefix=...)`` filters before LIMIT, across compression lineages.

A client that lists by recency and filters titles itself loses every match older than the page
(Sparky's task list saw 1 of 7 ``Task:`` rows once cron runs filled the page), so the prefix
must constrain the query, not the returned rows.
"""

import time

import pytest

from hermes_state import SessionDB

PREFIX = "Task: "


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        yield database
    finally:
        database.close()


def _session(db, sid, started, title=None, **kwargs):
    db.create_session(sid, source="tui", **kwargs)
    db.append_message(session_id=sid, role="user", content=f"hello from {sid}")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (started, sid))
    db._conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (started, sid))
    if title:
        db.set_session_title(sid, title)


def _list(db, **kwargs):
    return db.list_sessions_rich(order_by_last_active=True, compact_rows=True, **kwargs)


def test_prefix_reaches_matches_older_than_the_recency_page(db):
    base = time.time() - 10_000
    _session(db, "task_old", base, title="Task: oldest")
    _session(db, "task_mid", base + 10, title="Task: middle")
    # Near misses: other case, no separator, the prefix mid-title, LIKE wildcards in the title.
    _session(db, "lower", base + 20, title="task: lower case")
    _session(db, "joined", base + 30, title="Taskforce review")
    _session(db, "inside", base + 40, title="Re: Task: inside")
    _session(db, "wild", base + 50, title="Task_ wildcard")
    for n in range(6):
        _session(db, f"noise{n}", base + 100 + n)
    db._conn.commit()

    page = {row["id"] for row in _list(db, limit=4)}
    assert not page & {"task_old", "task_mid"}  # both sit outside a plain recency page

    rows = _list(db, limit=4, title_prefix=PREFIX)

    assert [row["id"] for row in rows] == ["task_mid", "task_old"]
    assert all(row["title"].startswith(PREFIX) for row in rows)


@pytest.mark.parametrize("title_on", ["tip", "root"])
def test_prefix_lists_a_compressed_lineage_as_its_titled_tip(db, title_on):
    """A rotation moves the title onto the tip and clears the root; a rotation cut off before
    that leaves it on the root. Either way the conversation lists once, as its tip, under its name."""
    base = time.time() - 10_000
    _session(db, "root", base, title="Task: long running")
    db.end_session("root", "compression")
    _session(db, "tip", base + 10, parent_session_id="root")
    if title_on == "tip":
        db.set_session_title("tip", "Task: long running")
        assert db.get_session("root")["title"] is None
    for n in range(4):
        _session(db, f"noise{n}", base + 100 + n)
    db._conn.commit()

    rows = _list(db, limit=2, title_prefix=PREFIX)

    assert [(row["id"], row["_lineage_root_id"]) for row in rows] == [("tip", "root")]
    assert rows[0]["title"] == "Task: long running"
