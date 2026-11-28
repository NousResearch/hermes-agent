"""``session.bulk_delete`` while the always-on gateway owns state.db.

The owner runs the dashboard's bulk delete and "Delete empty" sweeps over the control socket,
through the same store methods the offline route uses. These tests drive the production
``AuthorityConnection`` dispatch against a real store: a native local operator ticket is required
(a remote/browser grant is refused), a hand-picked selection is deleted with guarded rows skipped,
and the empty sweep only removes ended, message-less rows.
"""

import pytest

from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session

pytestmark = pytest.mark.platforms("posix")


def _connection(authority, *, operator, native=True):
    identity = {
        "user_id": "local-operator",
        "provider": "local",
        "profile_id": authority.profile_id,
        "instance_id": authority.instance_id,
        "native_bootstrap": native,
        "capabilities": [
            "session:create",
            "session:read",
            "session:submit",
            "session:control",
            "session:operator",
        ],
    }
    return AuthorityConnection(authority, object(), identity, operator=operator)


@pytest.fixture
def owned_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    handle = open_persistence(profile_scope(home=tmp_path))
    epoch = handle.runtime.begin_epoch(instance_id="owner")
    runner = SimpleNamespaceFixture(
        _draining=False,
        session_store=SimpleNamespaceFixture(retire_runtime_sessions=lambda ids: None),
    )
    authority = SessionAuthority(
        runner,
        profile_id=str(tmp_path),
        instance_id="owner",
        epoch=epoch,
        persistence=handle,
    )
    # Two deletable hand-picked rows with a message each, plus a fresh one the selection omits.
    for sid in ("pick-1", "pick-2", "keep-1"):
        seed_session(handle, sid, "cli")
        seed_messages(handle, sid, {"role": "user", "content": sid})
    # An ended, message-less row the empty sweep should remove, and an active empty one it must keep.
    seed_session(handle, "empty-ended", "cli")
    raw_holder(handle)._execute_write(
        lambda conn: conn.execute(
            "UPDATE sessions SET ended_at = started_at WHERE id = 'empty-ended'"
        )
    )
    seed_session(handle, "empty-active", "cli")  # ended_at stays NULL
    with handle:
        yield SimpleNamespaceFixture(
            handle=handle, authority=authority, path=tmp_path / "state.db"
        )


class SimpleNamespaceFixture:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.mark.asyncio
async def test_bulk_delete_is_local_operator_only(owned_store):
    """A non-native (remote/browser) ticket cannot bulk-delete across the store."""
    connection = _connection(owned_store.authority, operator=True, native=False)
    try:
        refused = await connection.dispatch(
            {"id": 1, "method": "session.bulk_delete", "params": {"ids": ["pick-1"]}}
        )
    finally:
        await connection.close()
    assert refused["error"]["message"] == "permission_denied", refused
    assert session_row(owned_store.handle, "pick-1") is not None


@pytest.mark.asyncio
async def test_native_bulk_delete_removes_the_selection(owned_store):
    connection = _connection(owned_store.authority, operator=True, native=True)
    try:
        frame = await connection.dispatch(
            {
                "id": 1,
                "method": "session.bulk_delete",
                "params": {"ids": ["pick-1", "pick-2", "ghost-id"]},
            }
        )
    finally:
        await connection.close()
    assert "error" not in frame, frame
    result = frame["result"]
    # Two existing requested rows deleted; the unknown id is skipped, the unselected row survives.
    assert result["deleted"] == 2
    assert session_row(owned_store.handle, "pick-1") is None
    assert session_row(owned_store.handle, "pick-2") is None
    assert session_row(owned_store.handle, "ghost-id") is None
    assert session_row(owned_store.handle, "keep-1") is not None


@pytest.mark.asyncio
async def test_empty_sweep_only_removes_ended_empty_rows(owned_store):
    connection = _connection(owned_store.authority, operator=True, native=True)
    try:
        frame = await connection.dispatch(
            {"id": 1, "method": "session.bulk_delete", "params": {"empty_sweep": True}}
        )
    finally:
        await connection.close()
    assert "error" not in frame, frame
    assert frame["result"]["deleted"] == 1
    assert session_row(owned_store.handle, "empty-ended") is None
    # The active empty row and every row with messages survive.
    assert session_row(owned_store.handle, "empty-active") is not None
    assert session_row(owned_store.handle, "keep-1") is not None


@pytest.mark.asyncio
async def test_bulk_delete_rejects_semantically_invalid_params(owned_store):
    """The handler's own gates: the two modes are mutually exclusive and an empty selection is a no-op
    success, not an error. Wire-shape type errors are already refused by the canonical pydantic contract."""
    connection = _connection(owned_store.authority, operator=True, native=True)
    try:
        # Both modes at once with a real selection: ambiguous, refused as invalid_params.
        frame = await connection.dispatch(
            {
                "id": 1,
                "method": "session.bulk_delete",
                "params": {"ids": ["x"], "empty_sweep": True},
            }
        )
        assert frame["error"]["message"] == "invalid_params", frame
        # An empty (but well-formed) selection deletes nothing and succeeds.
        ok = await connection.dispatch(
            {"id": 1, "method": "session.bulk_delete", "params": {"ids": []}}
        )
        assert ok["result"] == {"deleted": 0, "skipped_active": []}, ok
    finally:
        await connection.close()
