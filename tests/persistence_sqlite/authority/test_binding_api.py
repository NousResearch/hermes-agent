"""``SessionBindingRepository.bind_api`` on SQLite: the API route binding, built from the gateway's own records, is one
write transaction that a failure right after its first mutating statement leaves unchanged.
"""

from __future__ import annotations

from typing import Any

from persistence.authority.bindings import NewApiBinding
from persistence.core.identity import SessionId
from tests.persistence_sqlite.authority._admission_support import begin_epoch
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_write, build_seed

OWNER = "owner-a"
API_SID = "api-conv-1"


def seed(store: Any) -> None:
    build_seed(store, lambda db: begin_epoch(db, OWNER))


def test_bind_api_is_one_write(store):
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now

    seed(store)
    # The binding for ``API_SID`` with a declared key, from the gateway's own records.
    route, now = f"agent:main:api_server:dm:{API_SID}", _now()
    source = SessionSource(platform=Platform.API_SERVER, chat_id=API_SID, user_id="api", chat_type="dm")
    entry = SessionEntry(route, API_SID, now, now, origin=source, platform=Platform.API_SERVER).to_dict()
    binding = NewApiBinding(
        session_id=SessionId(API_SID), route=route, storage_source="api_server", title=None, declared_key="declared-1",
        caller_key=None, origin=source.to_dict(), entry=entry,
        receipt={"profile_id": "default", "session_id": API_SID, "route": route, "entry": entry, "declared_key": "declared-1"},
        started_at=now.timestamp())
    assert_one_write(store, lambda: store.bindings.bind_api(epoch=1, binding=binding))
