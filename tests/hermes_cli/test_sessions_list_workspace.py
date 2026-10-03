"""``hermes sessions list --workspace X`` applies ``--limit`` to the matching sessions.

The workspace match used to run on the newest ``--limit`` sessions overall, so a workspace
whose sessions were all older than that window listed as "No sessions found.", and the
"more not shown" footer described the unfiltered page instead of the matches.
"""

from argparse import Namespace

import pytest

from hermes_cli import sessions_cmd


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def _seed(db, sessions):
    """``sessions``: (id, cwd) oldest first; ``started_at`` is pinned so the order is fixed."""
    for sid, cwd in sessions:
        db.create_session(sid, "cli", cwd=cwd)
    with db._lock:
        for i, (sid, _cwd) in enumerate(sessions):
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (1_000_000.0 + i, sid))


def _listed(db, limit, workspace, capsys):
    sessions_cmd._cmd_list(db, Namespace(limit=limit, source=None, workspace=workspace))
    out = capsys.readouterr().out
    ids = [line.split()[-1] for line in out.splitlines() if line.split() and line.split()[-1].startswith("s_")]
    return ids, out


@pytest.mark.parametrize("matches, limit", [(3, 4), (6, 4)])
def test_limit_counts_matching_sessions_not_the_newest_overall(db, capsys, matches, limit):
    # Every projb session is older than the newest `limit + 5` sessions (all in proja).
    projb = [(f"s_b{i}", "/w/projb") for i in range(matches)]
    proja = [(f"s_a{i}", "/w/proja") for i in range(limit + 5)]
    _seed(db, projb + proja)

    ids, out = _listed(db, limit, "projb", capsys)

    newest_matches = [sid for sid, _ in reversed(projb)][:limit]
    assert ids == newest_matches
    assert ("more not shown" in out) == (matches > limit)


def test_no_footer_when_every_match_is_listed(db, capsys):
    # Two projb sessions are the newest; many older sessions live elsewhere.
    _seed(db, [(f"s_a{i}", "/w/proja") for i in range(30)] + [("s_b0", "/w/projb"), ("s_b1", "/w/projb")])

    ids, out = _listed(db, 20, "projb", capsys)

    assert ids == ["s_b1", "s_b0"]
    assert "more not shown" not in out


def test_a_session_archived_mid_listing_does_not_hide_a_match(db, capsys, monkeypatch):
    # Finding old matches may take more than one read; another process archiving (or deleting)
    # a newer session in between must not shift a match out of view.
    projb = [(f"s_b{i}", "/w/projb") for i in range(3)]
    proja = [(f"s_a{i}", "/w/proja") for i in range(200)]
    _seed(db, projb + proja)
    read = db.list_sessions_rich
    reads = []

    def read_then_archive_newest(**kwargs):
        rows = read(**kwargs)
        if not reads:
            db.set_session_archived("s_a199", True)
        reads.append(kwargs)
        return rows

    monkeypatch.setattr(db, "list_sessions_rich", read_then_archive_newest)

    ids, _out = _listed(db, 20, "projb", capsys)

    assert ids == ["s_b2", "s_b1", "s_b0"]


def test_large_workspace_outside_the_recent_window_needs_at_most_two_reads(db, capsys, monkeypatch):
    # >200 matching sessions, none of them among the newest 200: the first (windowed) read finds
    # no match at all, and the single fallback read over every session must still list them.
    projb = [(f"s_b{i}", "/w/projb") for i in range(250)]
    proja = [(f"s_a{i}", "/w/proja") for i in range(200)]
    _seed(db, projb + proja)
    read = db.list_sessions_rich
    windows = []

    def counting_read(**kwargs):
        windows.append(kwargs["limit"])
        return read(**kwargs)

    monkeypatch.setattr(db, "list_sessions_rich", counting_read)

    ids, out = _listed(db, 3, "projb", capsys)

    assert ids == ["s_b249", "s_b248", "s_b247"]
    assert "more not shown" in out
    assert windows == [200, -1]
