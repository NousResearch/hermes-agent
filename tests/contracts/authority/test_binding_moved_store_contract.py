"""Contract cases for history bindings in a store that moved with its home.

A history binding stores ``[profile_id, subject]``, the profile half being the home path that wrote it. A store copied to another machine or a moved home keeps its bindings: every reader compares the owner only. Bindings are written through the public import verb under an actor of the older home.
"""

from __future__ import annotations

import pytest

from persistence import ProofRejected, SessionId
from persistence.authority import runtime as rt

DEPARTED, HOME, OWNER = "/mac/home", "/cloud", "owner"


def _import(store, epoch, home: str, sid: str, request_id: str, parent: str | None = None, subject: str = OWNER):
    session = {"id": sid, "source": "cli", "messages": []}
    if parent is not None:
        session["parent_session_id"] = parent
    return store.runtime.mutate(epoch, principal_id=subject, session_id=sid, request_id=request_id, expected_revision=0,
                                command=rt.Import((session,)), history=rt.ImportControl(rt.HistoryActor(home, subject, None, True), (sid,)),
                                owner=None)


def test_a_binding_written_under_an_older_home_still_authorizes_its_owner(backend, scope_a):
    # CONTRACT: ``authorize_history``: a history-control record names its owner whatever home path it was written under; another subject is still refused.
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-moved")
    _import(store, epoch, DEPARTED, "moved", "imp")
    assert store.bindings.authorize_history(session_id=SessionId("moved"), profile_id=HOME, subject=OWNER, may_create=False, owner=None) is True
    with pytest.raises(ProofRejected):
        store.bindings.authorize_history(session_id=SessionId("moved"), profile_id=HOME, subject="someone-else", may_create=True, owner=None)


def test_an_import_links_a_parent_its_owner_imported_before_the_home_moved(backend, scope_a):
    # CONTRACT: ``Import``: a parent outside the payload is linked when its binding names the child's owner, whatever home path wrote it; a parent another owner imported is detached.
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-moved")
    _import(store, epoch, DEPARTED, "parent", "imp-parent")
    _import(store, epoch, DEPARTED, "foreign", "imp-foreign", subject="someone-else")
    linked = _import(store, epoch, HOME, "child", "imp-child", parent="parent")
    assert linked.result["detached"] == 0
    assert store.sessions.get(SessionId("child")).parent_session_id == "parent"
    detached = _import(store, epoch, HOME, "orphan", "imp-orphan", parent="foreign")
    assert detached.result["detached"] == 1
    assert store.sessions.get(SessionId("orphan")).parent_session_id is None
