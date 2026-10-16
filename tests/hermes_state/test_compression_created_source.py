"""A compression child must not freeze ``local`` or ``unknown`` as created_source.

Automatic compression does not create the child again. The turn's
``update_token_counts(source=...)`` is the write that fills a NULL column,
including when auxiliary usage ran first.
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


def test_compression_child_heals_through_token_counts_after_aux_usage(tmp_path):
    """NULL, unknown and local parents leave the child unset.

    ``record_auxiliary_usage`` runs first and must not store ``unknown``.
    ``update_token_counts(source='cli')`` then sets ``cli``. A parent that
    already has ``cli`` is copied, and neither write replaces it.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        for parent_created, child in ((None, "heal-null"), ("unknown", "heal-unknown"), ("local", "heal-local")):
            _compress(db, parent_created, child)
            assert _created(db, child) is None
            db.record_auxiliary_usage(child, "compression", model="m", input_tokens=2, output_tokens=1)
            assert _created(db, child) is None
            db.update_token_counts(child, input_tokens=3, output_tokens=1, model="m",
                                   api_call_count=1, source="cli")
            assert _created(db, child) == "cli"
        copied = _compress(db, "cli", "copied")
        assert _created(db, copied) == "cli"
        db.record_auxiliary_usage(copied, "compression", model="m", input_tokens=1)
        db.update_token_counts(copied, input_tokens=1, output_tokens=1, model="m",
                               api_call_count=1, source="desktop")
        assert _created(db, copied) == "cli"
    finally:
        db.close()


def test_worker_usage_fills_a_null_compression_child(tmp_path):
    """A managed worker's usage.main carries the surface and drops it on the floor
    unless the column is still NULL. Auxiliary usage on that row does not freeze it."""
    import hermes_state_runtime as rt

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        child = _compress(db, None, "worker-child")
        assert _created(db, child) is None
        db.record_auxiliary_usage(child, "compression", model="aux", input_tokens=1)
        assert _created(db, child) is None
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        scope = dict(epoch=epoch, execution_id="w", session_id=child, generation=0)
        rt.register_worker_execution(db, **scope, kind="compute", adoption_secret="private")
        rt.mutate_worker_execution(db, **scope, sequence=1, operation="usage.main", payload={
            "input_tokens": 5, "output_tokens": 1, "api_call_count": 1, "model": "m",
            "billing_provider": "p", "source": "cli"})
        assert _created(db, child) == "cli"

        kept = _compress(db, "tui", "worker-kept")
        assert _created(db, kept) == "tui"
        scope = dict(epoch=epoch, execution_id="w2", session_id=kept, generation=0)
        rt.register_worker_execution(db, **scope, kind="compute", adoption_secret="private")
        rt.mutate_worker_execution(db, **scope, sequence=1, operation="usage.main", payload={
            "input_tokens": 1, "api_call_count": 1, "model": "m", "source": "cli"})
        assert _created(db, kept) == "tui"
    finally:
        db.close()
