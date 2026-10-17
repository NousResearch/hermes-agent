"""A managed worker's usage write heals a NULL compression child to the agent surface.

Automatic compression does not create the child again, so the worker's ``usage.main``
is the write that fills the column, and it carries the surface to the NULL root.
"""
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


def test_desktop_worker_heals_gui_to_desktop(tmp_path):
    """A managed Desktop worker's platform is the policy label ``gui``. The healed
    column is ``desktop``, the same surface the in-process agent and the insert stamp."""
    import hermes_state_runtime as rt

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        child = _compress(db, None, "desktop-child")
        assert _created(db, child) is None
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
