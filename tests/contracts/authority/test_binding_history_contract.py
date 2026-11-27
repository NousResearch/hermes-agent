"""Contract cases for ``SessionBindingRepository.authorize_history``.

History-control records are seeded through the reference meta space, so those cases gate on failpoints.
"""

from __future__ import annotations

import json

import pytest

from persistence import ProofRejected, SessionId
from persistence.conversation.sessions import NewSession
from tests.contracts._reference._authority import authority_state
from tests.contracts.conftest import require_failpoints

PROFILE, SUBJECT = "default", "auth:v1:" + json.dumps(["github", "", "alice"])


def _authorize(store, sid: str, *, subject: str = SUBJECT, may_create: bool = True, native_transport=None) -> bool:
    return store.bindings.authorize_history(session_id=SessionId(sid), profile_id=PROFILE, subject=subject, may_create=may_create, native_transport=native_transport, owner=None)


def _record(backend, scope, sid: str, owner: list[str]) -> None:
    with authority_state(backend, scope).transaction() as staged:
        staged.authority["meta"]["history_control:" + sid] = owner


def test_missing_and_routed_rows_are_not_history(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=SessionId("routed"), source="cli", session_key="agent:main:local:dm:x"))
    store_a.sessions.create(NewSession(session_id=SessionId("chat"), source="cli", chat_id="x"))
    assert [_authorize(store_a, sid, native_transport=False) for sid in ("missing", "routed", "chat")] == [False, False, False]


def test_unrouted_local_history_needs_create_capability_and_owner(backend, scope_a):
    # CONTRACT: sources are the legacy local sources plus import; an owner other than the subject or its login label refuses; no session:create refuses.
    store_a = backend.open(scope_a)
    for sid, source, user_id in (("h-cli", "cli", None), ("h-desk", "desktop", "github:alice"), ("h-nous", "desktop", "nous:bob"),
                                 ("h-bob", "tui", "bob"), ("h-api", "api_server", None), ("h-imp", "import", None), ("h-oneshot", "oneshot", None)):
        store_a.sessions.create(NewSession(session_id=SessionId(sid), source=source, user_id=user_id))
    assert _authorize(store_a, "h-cli") is True
    assert _authorize(store_a, "h-desk") is True
    assert _authorize(store_a, "h-nous") is True
    assert _authorize(store_a, "h-imp") is True
    assert _authorize(store_a, "h-oneshot") is True
    assert _authorize(store_a, "h-api") is False
    with pytest.raises(ProofRejected):
        _authorize(store_a, "h-bob")
    with pytest.raises(ProofRejected):
        _authorize(store_a, "h-cli", may_create=False)


def test_a_history_control_record_decides_alone(backend, scope_a):
    store_a = backend.open(scope_a)
    require_failpoints(backend)
    store_a.sessions.create(NewSession(session_id=SessionId("h-rec"), source="api_server"))
    _record(backend, scope_a, "h-rec", [PROFILE, SUBJECT])
    assert _authorize(store_a, "h-rec", may_create=False, native_transport=False) is True
    with pytest.raises(ProofRejected):
        _authorize(store_a, "h-rec", subject="someone")


def test_native_transport_claim_check_runs_first(backend, scope_a):
    # CONTRACT: with ``native_transport`` a boolean, an unrecorded row the subject does not own on this profile needs the registry's answer; None skips the check (the listing).
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=SessionId("h-plain"), source="cli"))
    store_a.sessions.create(NewSession(session_id=SessionId("h-own"), source="desktop", user_id="github:alice"))
    store_a.sessions.create(NewSession(session_id=SessionId("h-foreign"), source="cli", profile_name="other"))
    with pytest.raises(ProofRejected):
        _authorize(store_a, "h-plain", native_transport=False)
    assert _authorize(store_a, "h-plain", native_transport=True) is True
    assert _authorize(store_a, "h-plain") is True
    assert _authorize(store_a, "h-own", native_transport=False) is True
    with pytest.raises(ProofRejected):
        _authorize(store_a, "h-foreign", native_transport=False)
