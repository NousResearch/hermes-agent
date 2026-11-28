"""Cross-login scoping for the forwarded dashboard prune (#138915, #159461 review F1).

A forwarded dashboard login (an ``auth:v1:`` subject) must only prune rows its single-row delete
could remove. Rows another login imported are bound to that login in ``state_meta``; the same
``authorize_history`` rule a single delete runs decides each row. A native local-operator ticket
keeps store-wide reach. These tests drive the production ``AuthorityConnection`` dispatch against
a real store with real history bindings, through the ``session.prune`` owner verb.
"""

import json
import time

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
        {"user_id": login[2], "provider": login[0], "issuer": login[1]}
    )
    identity = {
        "user_id": subject,
        "provider": "local",
        "profile_id": authority.profile_id,
        "instance_id": authority.instance_id,
        "native_bootstrap": True,
        "capabilities": list(_CAPS),
    }
    return AuthorityConnection(authority, object(), identity, operator=True)


def _native_connection(authority):
    identity = {
        "user_id": "local-operator",
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
        {"user_id": login[2], "provider": login[0], "issuer": login[1]}
    )
    binding = json.dumps([profile_id, subject], separators=(",", ":"))

    def write(conn):
        for sid in session_ids:
            conn.execute(
                "INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)",
                (BINDING_PREFIX + sid, binding),
            )

    raw_holder(handle)._execute_write(write)


def _age(handle, sid, days):
    at = time.time() - days * 86400

    def write(conn):
        conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=?, ended_at=? WHERE id=?",
            (at, at, at, sid),
        )
        conn.execute("UPDATE messages SET timestamp=? WHERE session_id=?", (at, sid))

    raw_holder(handle)._execute_write(write)


@pytest.fixture
def two_login_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handle = open_persistence(profile_scope(home=tmp_path))
    epoch = handle.runtime.begin_epoch(instance_id="owner")
    runner = _NS(
        _draining=False, session_store=_NS(retire_runtime_sessions=lambda ids: None)
    )
    authority = SessionAuthority(
        runner,
        profile_id=str(tmp_path),
        instance_id="owner",
        epoch=epoch,
        persistence=handle,
    )

    login_a = ("oidc", "https://issuer-a.example", "user-a")
    login_b = ("oidc", "https://issuer-b.example", "user-b")

    # Ended, aged rows with a message each (the prune filters path selects ended rows older than
    # the cutoff), each bound to its importing login.
    for sid, login in (("a-old", login_a), ("b-old", login_b)):
        seed_session(handle, sid, "import")
        seed_messages(handle, sid, {"role": "user", "content": sid})
        _bind_import(handle, str(tmp_path), login, sid)
        _age(handle, sid, 200)
    # A fresh row B owns: inside the 90-day window, the prune must leave it regardless of auth.
    seed_session(handle, "b-fresh", "import")
    seed_messages(handle, "b-fresh", {"role": "user", "content": "b-fresh"})
    _bind_import(handle, str(tmp_path), login_b, "b-fresh")
    _age(handle, "b-fresh", 1)

    with handle:
        yield _NS(
            handle=handle,
            authority=authority,
            login_a=login_a,
            login_b=login_b,
            path=tmp_path / "state.db",
        )


@pytest.mark.asyncio
async def test_forwarded_prune_keeps_rows_another_login_owns(two_login_store):
    """Review F1: a dashboard login cannot prune another login's sessions."""
    store = two_login_store
    connection = _login_connection(store.authority, store.login_b)
    try:
        frame = await connection.dispatch(
            {
                "id": 1,
                "method": "session.prune",
                "params": {"filters": {"older_than_days": 90}},
            }
        )
    finally:
        await connection.close()

    assert "error" not in frame, frame
    # B removes only its own aged row; A's aged row survives the cross-login prune.
    assert frame["result"]["deleted"] == 1
    assert session_row(store.handle, "a-old") is not None
    assert session_row(store.handle, "b-old") is None
    # The fresh row is outside the time window and stays either way.
    assert session_row(store.handle, "b-fresh") is not None


@pytest.mark.asyncio
async def test_native_operator_prune_still_reaches_across_logins(two_login_store):
    """The CLI/housekeeping native ticket is not a login: store-wide reach is preserved."""
    store = two_login_store
    connection = _native_connection(store.authority)
    try:
        frame = await connection.dispatch(
            {
                "id": 1,
                "method": "session.prune",
                "params": {"filters": {"older_than_days": 90}},
            }
        )
    finally:
        await connection.close()

    assert "error" not in frame, frame
    # Both logins' aged rows are pruned; only the in-window row survives.
    assert frame["result"]["deleted"] == 2
    assert session_row(store.handle, "a-old") is None
    assert session_row(store.handle, "b-old") is None
    assert session_row(store.handle, "b-fresh") is not None


def _insert_keyed_never_active(handle, sid, *, age_days):
    """Insert a keyed, open, message-less row the never-active selector picks."""
    at = time.time() - age_days * 86400

    def write(conn):
        conn.execute(
            """INSERT INTO sessions
               (id, source, user_id, session_key, chat_id, chat_type, started_at,
                message_count, tool_call_count, api_call_count, input_tokens, output_tokens,
                archived, pinned)
               VALUES (?, 'telegram', 'user-1', ?, 'chat-1', 'dm', ?, 0, 0, 0, 0, 0, 0, 0)""",
            (sid, f"agent:main:telegram:dm:{sid}", at),
        )

    raw_holder(handle)._execute_write(write)


@pytest.mark.asyncio
async def test_forwarded_never_active_prune_is_refused(two_login_store):
    """A dashboard login can never drive the store-wide never-active (routed-row) sweep.

    The never-active selector only matches keyed gateway rows whose ownership is the route/lane
    principal, not an import history binding, and the dashboard prune route only forwards the
    filters path. A forwarded login reaching this verb must be refused fail-closed with every
    row retained (#159461 review F1).
    """
    store = two_login_store
    for sid, login in (("a-junk", store.login_a), ("b-junk", store.login_b)):
        _insert_keyed_never_active(store.handle, sid, age_days=45)
        _bind_import(store.handle, str(store.authority.profile_id), login, sid)

    connection = _login_connection(store.authority, store.login_b)
    try:
        frame = await connection.dispatch(
            {
                "id": 1,
                "method": "session.prune",
                "params": {"never_active_days": 30},
            }
        )
    finally:
        await connection.close()

    assert "error" in frame, frame
    assert frame["error"]["message"] == "permission_denied", frame
    # Fail-closed: nothing is removed.
    assert session_row(store.handle, "a-junk") is not None
    assert session_row(store.handle, "b-junk") is not None


@pytest.mark.asyncio
async def test_native_never_active_prune_reaches_across_logins(two_login_store):
    """The native CLI ticket (no auth:v1: subject) keeps the whole-store never-active sweep."""
    store = two_login_store
    for sid in ("a-junk", "b-junk"):
        _insert_keyed_never_active(store.handle, sid, age_days=45)

    connection = _native_connection(store.authority)
    try:
        frame = await connection.dispatch(
            {
                "id": 1,
                "method": "session.prune",
                "params": {"never_active_days": 30},
            }
        )
    finally:
        await connection.close()

    assert "error" not in frame, frame
    assert frame["result"]["deleted"] == 2
    assert session_row(store.handle, "a-junk") is None
    assert session_row(store.handle, "b-junk") is None
