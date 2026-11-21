"""Pinned (hosted) scopes: a backend declaring ``scope_pinning`` serves each ``(tenant, agent)`` its own rows only; any other backend refuses a pin at ``open``. Every case runs on every registered backend: the pinning cases where the backend declares it, the refusal case where it does not, never a skip.

Each pinned scope here is built in the folder of a served local profile (``scope_a``), so a backend that chose a pinned store by its folder instead of its pin would hand it that profile's rows and fail these cases.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from persistence import IsolationUnsupported, ProfileScope, ScopePinRefused, SessionId, SessionRowMissing
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionQuery, SessionWindow
from tests.contracts._scopes import pinned_scope


def _pinned(scope_a: ProfileScope, tenant: str, agent: str) -> ProfileScope:
    return pinned_scope(tenant, agent, profile_dir=scope_a.profile_dir)


def _open_pinned_or_refused(backend: Any, *scopes: ProfileScope) -> Optional[list[Any]]:
    """The pinned handles when the backend declares ``scope_pinning``; otherwise assert each open refuses and answer None."""
    if backend.capabilities.scope_pinning:
        return [backend.open(scope) for scope in scopes]
    for scope in scopes:
        with pytest.raises(ScopePinRefused):
            backend.open(scope)
    return None


def _seed(store: Any, session_id: str, cwd: str, text: str) -> None:
    store.sessions.create(NewSession(session_id=SessionId(session_id), source="cli", cwd=cwd))
    store.sessions.set_title(SessionId(session_id), f"title {text}")
    store.messages.append(SessionId(session_id), NewMessage(role="user", content=text))


def _listed(store: Any) -> set[str]:
    return {str(entry.session.session_id) for entry in store.sessions.list_sessions(SessionQuery(), SessionWindow(limit=100))}


def _view(store: Any, session_id: str) -> tuple[Any, ...]:
    """Everything a reader learns about ``session_id``: its row, its transcript, its watermark and whether it lists."""
    sid = SessionId(session_id)
    return (store.sessions.get(sid), store.messages.read_transcript(sid), store.messages.active_watermark(sid),
            session_id in _listed(store))


def _assert_foreign_reads_as_missing(store: Any, foreign_id: str) -> None:
    missing = _view(store, "never-created")
    assert missing[0] is None and missing[1] == () and missing[3] is False
    assert _view(store, foreign_id) == missing


def _texts(store: Any, session_id: str) -> list[str]:
    return [str(row.content) for row in store.messages.read_transcript(SessionId(session_id))]


def test_backend_without_pinning_refuses_a_pinned_scope(backend_name: str, backend, scope_a) -> None:
    """A backend not declaring scope_pinning refuses a pinned ProfileScope at open with ScopePinRefused, even one built in a folder it serves, and the local store stays as it was; a declaring backend binds the handle to the pin."""
    local = backend.open(scope_a)
    _seed(local, "local-s", "/local", "local-text")
    pinned = _pinned(scope_a, "tenant-a", "agent-1")
    handles = _open_pinned_or_refused(backend, pinned)
    if handles is not None:
        (store,) = handles
        assert store.scope == pinned and store.scope.pin == pinned.pin
        assert store.scope != scope_a
    assert _texts(local, "local-s") == ["local-text"]
    assert local.sessions.get(SessionId("local-s")).cwd == "/local"


def test_pinned_handles_see_only_their_own_tenant(backend_name: str, backend, scope_a) -> None:
    """Tenant B's pinned handle never sees tenant A's sessions or messages, and a foreign id reads as missing; neither pin sees the local profile its scope was built in."""
    _seed(backend.open(scope_a), "local-s", "/local", "local-text")
    handles = _open_pinned_or_refused(backend, _pinned(scope_a, "tenant-a", "agent-1"), _pinned(scope_a, "tenant-b", "agent-1"))
    if handles is None:
        return
    store_a, store_b = handles
    _seed(store_a, "a-only", "/a", "a-secret")
    _seed(store_a, "collide", "/a", "a-collide")
    _seed(store_b, "collide", "/b", "b-collide")
    _assert_foreign_reads_as_missing(store_b, "a-only")
    _assert_foreign_reads_as_missing(store_a, "local-s")
    _assert_foreign_reads_as_missing(store_b, "local-s")
    assert (_listed(store_a), _listed(store_b)) == ({"a-only", "collide"}, {"collide"})
    assert (_texts(store_a, "collide"), _texts(store_b, "collide")) == (["a-collide"], ["b-collide"])
    assert (store_a.sessions.get(SessionId("collide")).cwd, store_b.sessions.get(SessionId("collide")).cwd) == ("/a", "/b")
    assert _listed(backend.open(scope_a)) == {"local-s"}


def test_pinning_an_agent_never_widens_the_tenant(backend_name: str, backend, scope_a) -> None:
    """Same tenant, two agents: each agent's handle sees only its own rows, and the same agent id under another tenant is another store; pins whose ids differ only in where a separator falls stay apart."""
    scopes = (_pinned(scope_a, "tenant-t", "agent-x"), _pinned(scope_a, "tenant-t", "agent-y"),
              _pinned(scope_a, "tenant-u", "agent-x"), _pinned(scope_a, "t/x", "y"), _pinned(scope_a, "t", "x/y"))
    handles = _open_pinned_or_refused(backend, *scopes)
    if handles is None:
        return
    for index, store in enumerate(handles):
        _seed(store, f"own-{index}", f"/w{index}", f"text-{index}")
    for index, store in enumerate(handles):
        assert _listed(store) == {f"own-{index}"}
        assert _texts(store, f"own-{index}") == [f"text-{index}"]
        for other in range(len(handles)):
            if other != index:
                _assert_foreign_reads_as_missing(store, f"own-{other}")


def test_cross_tenant_write_is_refused_or_invisible(backend_name: str, backend, scope_a) -> None:
    """A write through B's handle naming A's session id answers exactly as for a missing id (an append is refused as missing) and leaves A's rows unchanged."""
    handles = _open_pinned_or_refused(backend, _pinned(scope_a, "tenant-a", "agent-1"), _pinned(scope_a, "tenant-b", "agent-1"))
    if handles is None:
        return
    store_a, store_b = handles
    _seed(store_a, "a-owned", "/a", "a-secret")
    before = store_a.sessions.get(SessionId("a-owned"))

    def writes(session_id: str) -> tuple[Any, ...]:
        sid = SessionId(session_id)
        cwd = store_b.sessions.update_cwd(sid, "/intruder")
        title = store_b.sessions.set_title(sid, "intruder title")
        store_b.sessions.end(sid, "cli_close")
        store_b.sessions.patch_model_config(sid, {"intruder": True})
        return cwd, title

    assert writes("a-owned") == writes("never-created-in-b")
    for session_id in ("a-owned", "never-created-in-b"):
        with pytest.raises(SessionRowMissing):
            store_b.messages.append(SessionId(session_id), NewMessage(role="user", content="b-text"))
    after = store_a.sessions.get(SessionId("a-owned"))
    assert (after.cwd, after.title, after.ended_at, after.end_reason) == (before.cwd, before.title, None, None)
    assert store_a.sessions.get_model_config_value(SessionId("a-owned"), "intruder") is None
    assert _texts(store_a, "a-owned") == ["a-secret"]
    assert _view(store_b, "a-owned") == _view(store_b, "never-created")


def test_assert_isolating_matches_declared_pinning(backend_name: str, backend) -> None:
    """assert_isolating passes on a backend declaring scope_pinning and raises IsolationUnsupported on any other."""
    if backend.capabilities.scope_pinning:
        assert backend.assert_isolating() is None
    else:
        with pytest.raises(IsolationUnsupported):
            backend.assert_isolating()
