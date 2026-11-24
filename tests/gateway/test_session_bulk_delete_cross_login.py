"""Cross-login scoping for the forwarded dashboard bulk delete (#138915, #143182 review F1).

A forwarded dashboard login (an ``auth:v1:`` subject) must only bulk-delete / empty-sweep rows its
single-row delete could remove. Rows another login imported are bound to that login in
``state_meta``; the same ``authorize_history`` rule a single delete runs decides each row. A native
local-operator ticket keeps store-wide reach. These tests drive the production
``AuthorityConnection`` dispatch against a real store with real history bindings.
"""
import json

import pytest

from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_identity import authenticated_subject
from persistence.sqlite.authority._binding_history import BINDING_PREFIX
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session

pytestmark = pytest.mark.platforms("posix")

_CAPS = [
    "session:create",
    "session:read",
    "session:submit",
    "session:control",
    "session:operator",
]


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _login_connection(authority, login):
    """A native-bootstrap operator connection whose actor subject is a dashboard login."""
    subject = authenticated_subject(
        {"user_id": login[2], "provider": login[0], "issuer": login[1]})
    identity = {
        "user_id": subject,
        "provider": "local",
        "profile_id": authority.profile_id,
        "instance_id": authority.instance_id,
        "native_bootstrap": True,
        "capabilities": list(_CAPS),
    }
    return AuthorityConnection(authority, object(), identity, operator=True)


def _bind_import(handle, profile_id, login, *session_ids):
    """Write the same history-control binding an import stamps for a dashboard login."""
    subject = authenticated_subject(
        {"user_id": login[2], "provider": login[0], "issuer": login[1]})
    binding = json.dumps([profile_id, subject], separators=(",", ":"))

    def write(conn):
        for sid in session_ids:
            conn.execute(
                "INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)",
                (BINDING_PREFIX + sid, binding),
            )

    raw_holder(handle)._execute_write(write)


def _end_empty(handle, sid):
    raw_holder(handle)._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET ended_at = started_at WHERE id = ?", (sid,)))


@pytest.fixture
def two_login_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handle = open_persistence(profile_scope(home=tmp_path))
    epoch = handle.runtime.begin_epoch(instance_id="owner")
    runner = _NS(_draining=False,
                 session_store=_NS(retire_runtime_sessions=lambda ids: None))
    authority = SessionAuthority(
        runner, profile_id=str(tmp_path), instance_id="owner", epoch=epoch, persistence=handle)

    login_a = ("oidc", "https://issuer-a.example", "user-a")
    login_b = ("oidc", "https://issuer-b.example", "user-b")

    # Rows with messages, each bound to its importing login.
    for sid, login in (("a-picked", login_a), ("b-own", login_b)):
        seed_session(handle, sid, "import")
        seed_messages(handle, sid, {"role": "user", "content": sid})
        _bind_import(handle, str(tmp_path), login, sid)
    # An empty, ended import row owned by A — the empty-sweep cross-login case.
    seed_session(handle, "a-empty", "import")
    _end_empty(handle, "a-empty")
    _bind_import(handle, str(tmp_path), login_a, "a-empty")

    with handle:
        yield _NS(handle=handle, authority=authority, login_a=login_a, login_b=login_b,
                  path=tmp_path / "state.db")


@pytest.mark.asyncio
async def test_forwarded_bulk_delete_keeps_rows_another_login_owns(two_login_store):
    store = two_login_store
    connection = _login_connection(store.authority, store.login_b)
    try:
        frame = await connection.dispatch({
            "id": 1,
            "method": "session.bulk_delete",
            "params": {"ids": ["a-picked", "b-own"]},
        })
    finally:
        await connection.close()

    assert "error" not in frame, frame
    result = frame["result"]
    # B deletes only its own row; A's row is reported back as skipped and survives.
    assert result["deleted"] == 1
    assert "a-picked" in result["skipped_active"]
    assert session_row(store.handle, "a-picked") is not None
    assert session_row(store.handle, "b-own") is None


@pytest.mark.asyncio
async def test_forwarded_empty_sweep_keeps_another_logins_empty_rows(two_login_store):
    store = two_login_store
    connection = _login_connection(store.authority, store.login_b)
    try:
        frame = await connection.dispatch({
            "id": 1,
            "method": "session.bulk_delete",
            "params": {"empty_sweep": True},
        })
    finally:
        await connection.close()

    assert "error" not in frame, frame
    # A's empty import is not B's to sweep; nothing is removed and it stays listed.
    assert frame["result"]["deleted"] == 0
    assert session_row(store.handle, "a-empty") is not None


@pytest.mark.asyncio
async def test_native_operator_still_deletes_across_logins(two_login_store):
    """The CLI/housekeeping native ticket is not a login: store-wide reach is preserved."""
    store = two_login_store
    identity = {
        "user_id": "local-operator",
        "provider": "local",
        "profile_id": store.authority.profile_id,
        "instance_id": store.authority.instance_id,
        "native_bootstrap": True,
        "capabilities": list(_CAPS),
    }
    connection = AuthorityConnection(store.authority, object(), identity, operator=True)
    try:
        frame = await connection.dispatch({
            "id": 1,
            "method": "session.bulk_delete",
            "params": {"ids": ["a-picked", "b-own"]},
        })
    finally:
        await connection.close()

    assert "error" not in frame, frame
    assert frame["result"]["deleted"] == 2
    assert session_row(store.handle, "a-picked") is None
    assert session_row(store.handle, "b-own") is None
