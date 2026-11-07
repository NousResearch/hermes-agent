"""Compression chain stamping on the PostgreSQL backend.

The SQLite plan-pinning test (``test_compression_created_source.py``) cannot run against
PostgreSQL: it inspects a SQLite query plan. These tests instead exercise the same
stamped-usage-write flow through the shared store API, so the ``_UNSET_ANCESTOR_SQL``
probe (dialect-translated by ``hermes_state_postgres._convert_statement``) runs on real
PostgreSQL and the stamp walk lands on the PG tables.
"""
import pytest


def _compress(db, parent_created, child):
    parent = "parent-" + child
    db.create_session(parent, source="local")
    _set_created(db, parent, parent_created)
    db.append_message(parent, role="user", content="hi")
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="local",
        messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
    return child


def _set_created(db, sid, created):
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET created_source=? WHERE id=?", (created, sid)))


def _created(db, sid):
    return db.get_session(sid)["created_source"]


@pytest.mark.parametrize("mid_created, expected_root", [("cli", "cli"), ("tui", None)])
def test_stamp_walks_through_a_same_surface_mid_segment(postgres_db, mid_created, expected_root):
    """root(NULL) -> mid -> tip(NULL): a usage write on the tip stamps the root through a mid
    segment already on the tip's surface; a mid on another surface stops the walk."""
    db = postgres_db
    mid = _compress(db, None, "mid")
    db.publish_compression_child(
        parent_session_id=mid, child_session_id="tip", source="local",
        messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
    _set_created(db, mid, mid_created)
    _set_created(db, "tip", None)
    db.update_token_counts("tip", input_tokens=1, model="m", api_call_count=1, source="cli")
    assert [_created(db, sid) for sid in ("parent-mid", mid, "tip")] == [
        expected_root, mid_created, "cli"]


@pytest.mark.parametrize("child_created", [None, "unknown"])
def test_worker_usage_heals_an_unset_root(postgres_db, child_created):
    """root(NULL) -> child(NULL/unknown): the tip's usage write stamps the unset root, as on
    SQLite."""
    db = postgres_db
    child = _compress(db, None, "child")
    _set_created(db, child, child_created)
    db.update_token_counts("child", input_tokens=1, model="m", api_call_count=1, source="cli")
    assert _created(db, "parent-child") == "cli"
    assert _created(db, "child") == "cli"


def test_fully_stamped_chain_skips_the_walk(postgres_db):
    """Every ancestor already carries a real surface: the one-probe fast path answers with no
    rows (the dialect-translated ``CROSS JOIN ... ON`` runs and finds nothing)."""
    db = postgres_db
    mid = _compress(db, "cli", "mid")
    db.publish_compression_child(
        parent_session_id=mid, child_session_id="tip", source="local",
        messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
    _set_created(db, "tip", None)
    db.update_token_counts("tip", input_tokens=1, model="m", api_call_count=1, source="gui")
    # The chain is fully stamped (root and mid on ``cli``); the tip simply records its own
    # surface, and nothing upstream changes.
    assert _created(db, "tip") == "gui"
    assert _created(db, mid) == "cli"
    assert _created(db, "parent-mid") == "cli"
