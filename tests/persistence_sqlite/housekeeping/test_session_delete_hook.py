"""``on_session_delete`` (#124511): every committed delete or deliberate archive notifies plugins once.

The SQLite store's delete, prune and archive verbs and the runtime mutation, each followed by the ``session_retention``
dispatch its caller runs, with the real plugin hook registry. The callback reads the row back, so each event also
proves dispatch ran AFTER the commit, never before.
"""

import os

import pytest

import session_retention
from hermes_cli.plugins import get_plugin_manager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from persistence import SessionId
from persistence.authority.runtime import Delete, Sidebar
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import create_session, end_session, get_session
from tests.contracts._scopes import MEDIA_ROOT


@pytest.fixture()
def store(backend, scope_a):
    return backend.open(scope_a)


@pytest.fixture()
def hook(store):
    """Register callbacks on the registry ``invoke_hook`` reads (the store's profile home's); restore it afterwards."""
    token = set_hermes_home_override(os.path.abspath(store.scope.profile_dir))
    try:
        manager = get_plugin_manager()
    finally:
        reset_hermes_home_override(token)
    saved = {name: list(cbs) for name, cbs in manager._hooks.items()}
    events: list[dict] = []
    holder = store.holder()

    def _record(**kw):
        row = get_session(holder, kw["session_id"])
        events.append({**kw, "row_after": None if row is None else bool(row["archived"])})

    manager._hooks.setdefault("on_session_delete", []).append(_record)
    try:
        yield manager, events
    finally:
        manager._hooks = saved


def _deleted(store, removed, reason, primary=None):
    session_retention.after_sessions_removed(store, store.scope.profile_dir, removed, reason=reason, primary=primary)


def _canonical(store, sid, command):
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id="test")
    row = get_session(holder, sid)
    outcome = store.runtime.mutate(
        epoch, principal_id="human", session_id=sid, request_id=f"{command.operation}-{sid}",
        expected_revision=row["runtime_revision"], command=command, expected_generation=row["runtime_generation"],
        media_root=MEDIA_ROOT, owner=None)
    if outcome.gone is not None:
        session_retention.emit_session_deleted(store.scope.profile_dir, outcome.gone.session_ids,
                                               reason=outcome.gone.reason, primary=sid)


def _archived(store, sid, archived):
    change = store.sessions.set_archived(SessionId(sid), archived)
    session_retention.emit_session_deleted(store.scope.profile_dir, change.archived_ids, reason="archive", primary=sid)
    return change.changed


def test_every_delete_and_archive_path_fires_once_after_commit(store, hook):
    _, events = hook
    holder = store.holder()
    for sid in ("cli", "bulk1", "bulk2", "empty", "old", "desk", "arch", "desk-arch"):
        create_session(holder, sid, source="cli")
        if sid != "empty":
            append_message(holder, sid, role="user", content="hi")
    end_session(holder, "empty", "done")
    end_session(holder, "old", "done")
    for tip in ("arch", "desk-arch"):  # compression chains: archiving the tip hides the root too
        create_session(holder, f"{tip}-root", source="cli")
        append_message(holder, f"{tip}-root", role="user", content="hi")
        end_session(holder, f"{tip}-root", "compression")
        holder._execute_write(lambda c, t=tip: c.execute(
            "UPDATE sessions SET parent_session_id = ? WHERE id = ?", (f"{t}-root", t)))

    removed = store.sessions.delete(SessionId("cli"), media_root=MEDIA_ROOT)  # hermes sessions delete, /exit --delete
    assert removed
    _deleted(store, removed, "delete", primary="cli")
    removed = store.sessions.delete(SessionId("missing"), media_root=MEDIA_ROOT)  # unknown id: silent
    assert removed == ()
    _deleted(store, removed, "delete", primary="missing")
    bulk = store.sessions.delete_many([SessionId("bulk1"), SessionId("bulk2"), SessionId("nope")], skip_guarded=True,
                                      include_compression_chain=True, media_root=MEDIA_ROOT)  # Desktop/dashboard bulk delete
    assert bulk.deleted == 2
    _deleted(store, bulk.removed_ids, "delete")
    removed = store.sessions.delete_empty(media_root=MEDIA_ROOT)  # "Delete empty (N)"
    assert len(removed) == 1
    _deleted(store, removed, "empty_sweep")
    pruned = store.maintenance.prune_sessions(older_than_days=0, media_root=MEDIA_ROOT)  # retention prune (only ended row)
    assert pruned.count == 1
    _deleted(store, pruned.deleted_ids, "prune")
    _canonical(store, "desk", Delete())                       # Desktop DELETE /api/sessions/{id}
    assert _archived(store, "arch", True)                     # TUI session.archive, CLI archive
    assert _archived(store, "arch", True)                     # repeat archive: silent
    assert _archived(store, "arch", False)                    # un-archive: silent
    _canonical(store, "desk-arch", Sidebar(archived=True))    # Desktop archive (PATCH)
    session_retention.run_auto_archive(store, idle_days=0, min_interval_hours=0)  # idle auto-archive sweep: silent

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


def test_raising_callback_never_breaks_deletion(store, hook):
    manager, events = hook

    def _boom(**_):
        raise RuntimeError("plugin down")

    manager._hooks["on_session_delete"].insert(0, _boom)
    holder = store.holder()
    create_session(holder, "s1", source="cli")
    create_session(holder, "s2", source="cli")

    removed = store.sessions.delete(SessionId("s1"), media_root=MEDIA_ROOT)
    assert removed == ("s1",)
    _deleted(store, removed, "delete", primary="s1")
    assert _archived(store, "s2", True) is True
    assert get_session(holder, "s1") is None and get_session(holder, "s2")["archived"]
    assert [e["session_id"] for e in events] == ["s1", "s2"]  # later observers still ran
