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
