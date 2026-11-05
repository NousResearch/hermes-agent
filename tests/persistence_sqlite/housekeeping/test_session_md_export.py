import time

from persistence.housekeeping.maintenance import PruneCriteria
from persistence.sqlite.conversation._store.compression import get_compression_lineage
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session


def test_export_candidates_via_prune_filters_ended_old_sessions(backend, scope_a, monkeypatch):
    store = backend.open(scope_a)
    db = store.holder()
    monkeypatch.setattr(time, "time", lambda: 2_000_000.0)
    create_session(db, "old_cli", source="cli")
    end_session(db, "old_cli", "done")
    db._write_sql("UPDATE sessions SET started_at=?, ended_at=? WHERE id=?", (1_000_000.0, 1_000_010.0, "old_cli"))

    create_session(db, "new_cli", source="cli")
    end_session(db, "new_cli", "done")
    db._write_sql("UPDATE sessions SET started_at=?, ended_at=? WHERE id=?", (1_990_000.0, 1_990_010.0, "new_cli"))

    create_session(db, "old_active", source="cli")
    db._write_sql("UPDATE sessions SET started_at=? WHERE id=?", (1_000_000.0, "old_active"))

    # Export uses the shared prune/archive candidate selection.
    candidates = store.maintenance.list_prune_candidates(
        criteria=PruneCriteria(started_before=2_000_000.0 - 5 * 86400, archived=None)
    )
    assert [c.session_id for c in candidates] == ["old_cli"]


def test_get_compression_lineage_returns_only_compression_chain(backend, scope_a):
    db = backend.open(scope_a).holder()
    create_session(db, "root", source="cli")
    end_session(db, "root", "compression")
    create_session(db, "child", source="cli", parent_session_id="root")
    end_session(db, "child", "compression")
    create_session(db, "tip", source="cli", parent_session_id="child")
    create_session(db, "branch", source="cli", parent_session_id="root", model_config={"_branched_from": "root"})
    create_session(db, "delegate", source="delegate", parent_session_id="child", model_config={"_delegate_from": "child"})
    create_session(db, "tool", source="tool", parent_session_id="child")

    assert get_compression_lineage(db, "tip") == ["root", "child", "tip"]
    assert get_compression_lineage(db, "branch") == ["branch"]
    assert get_compression_lineage(db, "delegate") == ["delegate"]
    assert get_compression_lineage(db, "tool") == ["tool"]


def test_fork_children_created_before_continuation_do_not_hijack_lineage(backend, scope_a):
    # Regression: the forward walk used to accept any non-branch child as the
    # compression continuation. A delegate/tool child spawned BEFORE the real
    # continuation row (the common runtime ordering: the subagent exists
    # before compression rotates the session) was picked as the successor,
    # so lineage and session .md export followed the subagent's transcript.
    store = backend.open(scope_a)
    db = store.holder()
    create_session(db, "root", source="cli")
    append_message(db, "root", role="user", content="root msg")
    create_session(
        db,
        "delegate",
        source="delegate",
        parent_session_id="root",
        model_config={"_delegate_from": "root"},
    )
    append_message(db, "delegate", role="user", content="delegate private msg")
    end_session(db, "root", "compression")
    create_session(db, "continuation", source="cli", parent_session_id="root")
    append_message(db, "continuation", role="user", content="continuation msg")

    create_session(db, "root2", source="cli")
    create_session(db, "toolchild", source="tool", parent_session_id="root2")
    end_session(db, "root2", "compression")
    create_session(db, "cont2", source="cli", parent_session_id="root2")

    assert get_compression_lineage(db, "root") == ["root", "continuation"]
    assert get_compression_lineage(db, "continuation") == ["root", "continuation"]
    assert get_compression_lineage(db, "root2") == ["root2", "cont2"]

    exported = store.maintenance.export_session_lineage("root")
    assert exported is not None
    assert list(exported.lineage_session_ids) == ["root", "continuation"]
    contents = [
        m.content
        for seg in exported.segments
        for m in seg.messages
    ]
    assert contents == ["root msg", "continuation msg"]
