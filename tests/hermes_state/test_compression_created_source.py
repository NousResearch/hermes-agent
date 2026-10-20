"""A managed worker's usage write heals a NULL compression child to the agent surface.

Automatic compression does not create the child again, so the worker's ``usage.main``
is the write that fills the column, and it carries the surface to the NULL root.
"""
import pytest

from hermes_state import SessionDB


def _compress(db, parent_created, child):
    parent = "parent-" + child
    db.create_session(parent, source="local")
    with db._lock:
        db._conn.execute("UPDATE sessions SET created_source=? WHERE id=?", (parent_created, parent))
        db._conn.commit()
    db.append_message(parent, role="user", content="hi")
    db.publish_compression_child(
        parent_session_id=parent, child_session_id=child, source="local",
        messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
    return child


def _created(db, sid):
    return db.get_session(sid)["created_source"]


@pytest.mark.parametrize("child_created", [None, "unknown"])
def test_desktop_worker_heals_gui_to_desktop(tmp_path, child_created):
    """A managed Desktop worker's platform is the policy label ``gui``. The healed
    column is ``desktop``, the same surface the in-process agent and the insert stamp.
    An auxiliary-usage ``unknown`` placeholder is unset too, as in the creator upsert."""
    import hermes_state_runtime as rt

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        child = _compress(db, None, "desktop-child")
        with db._lock:
            db._conn.execute("UPDATE sessions SET created_source=? WHERE id=?", (child_created, child))
            db._conn.commit()
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        scope = dict(epoch=epoch, execution_id="w", session_id=child, generation=0)
        rt.register_worker_execution(db, **scope, kind="compute", adoption_secret="private")
        rt.mutate_worker_execution(db, **scope, sequence=1, operation="usage.main", payload={
            "input_tokens": 4, "output_tokens": 1, "api_call_count": 1, "model": "m",
            "billing_provider": "p", "source": "gui"})
        assert _created(db, child) == "desktop"
        assert _created(db, "parent-" + child) == "desktop"
    finally:
        db.close()


@pytest.mark.parametrize("mid_created, expected_root", [("cli", "cli"), ("tui", None)])
def test_stamp_walks_through_a_same_surface_mid_segment(tmp_path, mid_created, expected_root):
    """root(NULL) -> mid -> tip(NULL): a usage write on the tip stamps the root through a mid
    segment already on the tip's surface; a mid on another surface stops the walk."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        mid = _compress(db, None, "mid")
        db.publish_compression_child(
            parent_session_id=mid, child_session_id="tip", source="local",
            messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
        with db._lock:
            db._conn.execute("UPDATE sessions SET created_source=? WHERE id=?", (mid_created, mid))
            db._conn.execute("UPDATE sessions SET created_source=NULL WHERE id='tip'")
            db._conn.commit()
        db.update_token_counts("tip", input_tokens=1, model="m", api_call_count=1, source="cli")
        assert [_created(db, sid) for sid in ("parent-mid", mid, "tip")] == [
            expected_root, mid_created, "cli"]
    finally:
        db.close()


@pytest.mark.parametrize("root_created", [None, "unknown", "local"])
def test_compressing_a_stamped_tip_stamps_its_legacy_root(tmp_path, root_created):
    """A chain compressed before write-time stamping has an unset root and a stamped tip. The
    next compression publish copies the tip's surface to the child and stamps the root."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        tip = _compress(db, root_created, "legacy-tip")
        with db._lock:
            db._conn.execute("UPDATE sessions SET created_source='cli' WHERE id=?", (tip,))
            db._conn.commit()
        db.publish_compression_child(
            parent_session_id=tip, child_session_id="new-tip", source="local",
            messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
        assert [_created(db, sid) for sid in ("parent-" + tip, tip, "new-tip")] == ["cli"] * 3
    finally:
        db.close()


def test_unset_parent_behind_a_reset_edge_takes_the_fast_path(tmp_path):
    """A stamped chain whose root was reset from a legacy NULL row has nothing to stamp: the
    ancestor stamp returns after its two lookups instead of walking the chain on every write."""
    from hermes_state_compression import stamp_compression_ancestors

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("legacy", source="local")
        db.create_session("reset-root", source="cli", parent_session_id="legacy",
                          model_config={"_reset_from": "legacy"})
        with db._lock:
            db._conn.execute("UPDATE sessions SET created_source=NULL, ended_at=1, "
                             "end_reason='session_reset' WHERE id='legacy'")
            db._conn.commit()
        db.append_message("reset-root", role="user", content="hi")
        db.publish_compression_child(
            parent_session_id="reset-root", child_session_id="tip", source="cli",
            messages=[{"role": "user", "content": "hi"}], require_compression_lease=False)
        statements = []
        with db._lock:
            db._conn.set_trace_callback(statements.append)
            try:
                stamp_compression_ancestors(db._conn, "tip")
            finally:
                db._conn.set_trace_callback(None)
        assert len(statements) == 2, statements
        assert _created(db, "legacy") is None
    finally:
        db.close()


def test_unset_ancestor_probe_reads_only_the_walked_rows(tmp_path):
    """The fast path runs on every usage write of a stamped compression child: its final lookup
    must join the walked ids to sessions by key, never scan the table (ms per write at 30k rows)."""
    from hermes_state_compression import _CHAIN_CAP, _UNSET_ANCESTOR_SQL

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        plan = [row[3] for row in db._conn.execute(
            "EXPLAIN QUERY PLAN " + _UNSET_ANCESTOR_SQL, ("tip", "cli", _CHAIN_CAP))]
        assert not [step for step in plan if step.startswith("SCAN") and "up" not in step.split()], plan
    finally:
        db.close()
