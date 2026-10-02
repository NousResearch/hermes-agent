"""Review-lane quota-residue escape in check_respawn_guard.

Regression test for the bug where a card handed to review after a rate-limited
run kept its stamped ``last_failure_error`` ("exited rate-limited (quota wall)")
and the respawn guard re-classified that stale text as ``blocker_auth``,
parking the card in the review lane forever (the rate-limit escape only fires
while ``rate_limited`` is the LATEST outcome; ``review_requested`` supersedes
it). See the handoff-supersession block in check_respawn_guard.
"""
import sqlite3
import time

import pytest

from hermes_cli import kanban_db_dispatch as dd

QUOTA_TXT = "pid 397884 exited rate-limited (quota wall) — requeued without counting a failure"
AUTH_TXT = "401 unauthorized: invalid_api_key"


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, last_failure_error TEXT)")
    c.execute(
        "CREATE TABLE task_runs (id INTEGER PRIMARY KEY, task_id TEXT, outcome TEXT, ended_at INTEGER, metadata TEXT)"
    )
    yield c
    c.close()


def _seed(c, outcome, ended_at, err):
    c.execute("INSERT INTO tasks VALUES ('t1', ?)", (err,))
    c.execute(
        "INSERT INTO task_runs (task_id, outcome, ended_at) VALUES ('t1', ?, ?)",
        (outcome, ended_at),
    )


def test_review_requested_supersedes_quota_residue(conn):
    _seed(conn, "review_requested", int(time.time()) - 3600, QUOTA_TXT)
    assert dd.check_respawn_guard(conn, "t1", lane="review") is None


def test_rate_limit_cooldown_still_applies(conn):
    _seed(conn, "rate_limited", int(time.time()) - 10, QUOTA_TXT)
    assert dd.check_respawn_guard(conn, "t1", lane="review") == "rate_limit_cooldown"


def test_auth_residue_still_blocks(conn):
    _seed(conn, "review_requested", int(time.time()) - 3600, AUTH_TXT)
    assert dd.check_respawn_guard(conn, "t1", lane="review") == "blocker_auth"


def test_clean_card_passes(conn):
    _seed(conn, "review_requested", int(time.time()) - 3600, None)
    assert dd.check_respawn_guard(conn, "t1", lane="review") is None


def test_residue_regex_shape():
    assert dd._RATE_LIMIT_RESIDUE_RE.search(QUOTA_TXT)
    assert not dd._RATE_LIMIT_RESIDUE_RE.search(AUTH_TXT)
    assert not dd._RATE_LIMIT_RESIDUE_RE.search("ordinary crash output")
