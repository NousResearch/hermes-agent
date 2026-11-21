"""Contract cases for ``SessionRepository.backfill_acp_session_cwds`` (the ``persistence/conversation/sessions.py`` docstring).

Every case runs both registered backends, seeds through public protocol operations only and observes through ``sessions.get``.
"""

from __future__ import annotations


from persistence import SessionId
from persistence.conversation.sessions import NewSession



def _seed(backend, scope, sid: str, source: str, *, cwd=None, config=None) -> None:
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=SessionId(sid), source=source, cwd=cwd, model_config=config))


def _cwd(backend, scope, sid: str):
    store = backend.open(scope)
    return (store.sessions.get(SessionId(sid))).cwd


def test_backfill_promotes_only_acp_rows_without_a_cwd(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(backend, scope_a, "acp-legacy", "acp", config={"cwd": "/work/app"})
    _seed(backend, scope_a, "acp-own", "acp", cwd="/own", config={"cwd": "/work/other"})
    _seed(backend, scope_a, "acp-bare", "acp", config={"model": "m"})
    _seed(backend, scope_a, "acp-blank", "acp", config={"cwd": ""})
    _seed(backend, scope_a, "cli-legacy", "cli", config={"cwd": "/somewhere"})

    assert store_a.sessions.backfill_acp_session_cwds() == 1
    assert _cwd(backend, scope_a, "acp-legacy") == "/work/app"
    # An explicit column value wins; nothing recorded or an empty record promotes nothing; other sources are untouched.
    assert _cwd(backend, scope_a, "acp-own") == "/own"
    assert _cwd(backend, scope_a, "acp-bare") in (None, "")
    assert _cwd(backend, scope_a, "acp-blank") in (None, "")
    assert _cwd(backend, scope_a, "cli-legacy") in (None, "")
    # Idempotent.
    assert store_a.sessions.backfill_acp_session_cwds() == 0


def test_backfill_is_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    _seed(backend, scope_a, "acp-iso", "acp", config={"cwd": "/work/a"})
    _seed(backend, scope_b, "acp-iso", "acp", config={"cwd": "/work/b"})
    assert store_a.sessions.backfill_acp_session_cwds() == 1
    assert _cwd(backend, scope_b, "acp-iso") in (None, "")
    assert store_b.sessions.backfill_acp_session_cwds() == 1
    assert (_cwd(backend, scope_a, "acp-iso"), _cwd(backend, scope_b, "acp-iso")) == ("/work/a", "/work/b")
