"""``on_session_delete`` (#124511): every committed delete or deliberate archive notifies plugins once.

Real ``SessionDB`` + the real plugin hook registry. The callback reads the row back, so each event
also proves dispatch ran AFTER the commit, never before.
"""

import pytest

import hermes_state_runtime as rt
from hermes_cli.plugins import get_plugin_manager
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


@pytest.fixture()
def hook(db):
    """Register callbacks on the registry ``invoke_hook`` reads; restore it afterwards."""
    manager = get_plugin_manager()
    saved = {name: list(cbs) for name, cbs in manager._hooks.items()}
    events: list[dict] = []

    def _record(**kw):
        row = db.get_session(kw["session_id"])
        events.append({**kw, "row_after": None if row is None else bool(row["archived"])})

    manager._hooks.setdefault("on_session_delete", []).append(_record)
    try:
        yield manager, events
    finally:
        manager._hooks = saved


def _canonical(db, sid, operation, payload):
    epoch = rt.begin_runtime_epoch(db, instance_id="test")
    row = db.get_session(sid)
    return rt.mutate_runtime_session(
        db, epoch=epoch, principal_id="human", session_id=sid, request_id=f"{operation}-{sid}",
        expected_revision=row["runtime_revision"], expected_generation=row["runtime_generation"],
        operation=operation, payload=payload)


def test_every_delete_and_archive_path_fires_once_after_commit(db, hook):
    _, events = hook
    for sid in ("cli", "bulk1", "bulk2", "empty", "old", "desk", "arch", "desk-arch"):
        db.create_session(sid, source="cli")
        if sid != "empty":
            db.append_message(sid, role="user", content="hi")
    db.end_session("empty", "done")
    db.end_session("old", "done")
    for tip in ("arch", "desk-arch"):  # compression chains: archiving the tip hides the root too
        db.create_session(f"{tip}-root", source="cli")
        db.append_message(f"{tip}-root", role="user", content="hi")
        db.end_session(f"{tip}-root", "compression")
        db._execute_write(lambda c, t=tip: c.execute(
            "UPDATE sessions SET parent_session_id = ? WHERE id = ?", (f"{t}-root", t)))

    assert db.delete_session("cli")                          # hermes sessions delete, /exit --delete
    assert db.delete_session("missing") is False             # unknown id: silent
    assert db.delete_sessions(["bulk1", "bulk2", "nope"]) == 2   # Desktop/dashboard bulk delete
    assert db.delete_empty_sessions() == 1                   # "Delete empty (N)"
    assert db.prune_sessions(older_than_days=0) == 1         # retention prune (only ended row)
    _canonical(db, "desk", "delete", {})                     # Desktop DELETE /api/sessions/{id}
    assert db.set_session_archived("arch", True)             # TUI session.archive, CLI archive
    assert db.set_session_archived("arch", True)             # repeat archive: silent
    assert db.set_session_archived("arch", False)            # un-archive: silent
    _canonical(db, "desk-arch", "sidebar", {"archived": True})   # Desktop archive (PATCH)
    db.archive_stale_sessions(0)                             # idle auto-archive sweep: silent

    got = [(e["session_id"], e["reason"], e.get("deleted_ids"), e["row_after"]) for e in events]
    assert got == [
        ("cli", "delete", None, None),
        ("bulk1", "delete", ["bulk1", "bulk2"], None),
        ("empty", "empty_sweep", None, None),
        ("old", "prune", None, None),
        ("desk", "delete", None, None),
        ("arch", "archive", ["arch", "arch-root"], True),
        ("desk-arch", "archive", ["desk-arch", "desk-arch-root"], True),
    ]
    assert all(e["profile"] == events[0]["profile"] for e in events)


def test_raising_callback_never_breaks_deletion(db, hook):
    manager, events = hook

    def _boom(**_):
        raise RuntimeError("plugin down")

    manager._hooks["on_session_delete"].insert(0, _boom)
    db.create_session("s1", source="cli")
    db.create_session("s2", source="cli")

    assert db.delete_session("s1") is True
    assert db.set_session_archived("s2", True) is True
    assert db.get_session("s1") is None and db.get_session("s2")["archived"]
    assert [e["session_id"] for e in events] == ["s1", "s2"]  # later observers still ran
