"""Profile identity purge for `hermes profile delete`: the delete-only path, and what an unserve
must leave alone.

`_unserve_profile()` runs for every name that leaves the served set — a rename's old name leaves it
exactly like a deleted one (its directory is gone either way) and the rename's rekey still needs that
identity — so the purge lives behind the delete-only ``purge-profile-identity`` control verb and
never in the unserve path (#111926, delete side).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_store(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    store = SessionStore(
        sessions_dir,
        GatewayConfig(sessions_dir=sessions_dir, write_sessions_json=False,
                      multiplex_profiles=True),
    )
    store._ensure_loaded()
    return store


def _entry(session_key, chat_id, profile):
    from gateway.session import SessionEntry, SessionSource, Platform
    from gateway.session_lifecycle import _now
    now = _now()
    return SessionEntry(
        session_key=session_key, session_id=f"sid-{chat_id}",
        platform=Platform.FEISHU, chat_type="dm", created_at=now, updated_at=now,
        origin=SessionSource(platform=Platform.FEISHU, chat_id=chat_id, profile=profile),
    )


def _state_db():
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    return SessionDB(Path(get_hermes_home()) / "state.db")


def _runner_stub(store):
    return SimpleNamespace(
        session_store=store,
        _profile_failed_platforms={},
        _profile_adapters={},
        pairing_stores={},
        _busy_text_modes_by_profile={},
        _busy_input_modes_by_profile={},
        _served_profile_homes={},
        _served_profile_signatures={},
        _agent_cache={},
        _evict_cached_agent=lambda key: None,
        # unserve shrinks the boot reservation; an unreserved stub has nothing to release
        config=SimpleNamespace(_runtime_profile_homes=()),
    )


@pytest.mark.asyncio
async def test_unserve_profile_keeps_identity_for_a_rename_to_rekey(tmp_path):
    """Unserving is not deleting: the purge must not run here.

    A rename's old name leaves the served set exactly like a deleted one, and
    ``migrate-profile-identity`` still has to find that identity to rekey it. A purge inside
    ``_unserve_profile()`` would delete it first and break the rename it runs beside.
    """
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin
    store = _make_store(tmp_path)
    with store._lock:
        store._entries["agent:oldname:feishu:dm:chatA"] = _entry(
            "agent:oldname:feishu:dm:chatA", "chatA", "oldname")
    db = _state_db()
    db.register_backend_heartbeat(
        backend_id="be1", pid=1, started_at=time.time(), profile="oldname", host="h")
    db.close()

    home = tmp_path / "home"
    home.mkdir()
    await GatewayProfileReconcileMixin._unserve_profile(_runner_stub(store), "oldname", home)

    assert "agent:oldname:feishu:dm:chatA" in store._entries
    db = _state_db()
    try:
        assert db._read_one(
            "SELECT COUNT(*) AS n FROM gateway_heartbeats WHERE profile = ?",
            ("oldname",))["n"] == 1
    finally:
        db.close()


def test_purge_verb_drops_routing_identity_and_reports_ok(tmp_path):
    """The delete-only verb settles identity in the process that owns the routing index."""
    from gateway.run_profile_reconcile import purge_profile_identity_verb
    store = _make_store(tmp_path)
    with store._lock:
        store._entries["agent:gone:feishu:dm:chatA"] = _entry(
            "agent:gone:feishu:dm:chatA", "chatA", "gone")
        store._entries["agent:keepme:feishu:dm:chatB"] = _entry(
            "agent:keepme:feishu:dm:chatB", "chatB", "keepme")
    db = _state_db()
    db.register_backend_heartbeat(
        backend_id="be1", pid=1, started_at=time.time(), profile="gone", host="h")
    db.close()

    answer = purge_profile_identity_verb(_runner_stub(store))({"name": "gone"})

    assert answer["ok"] is True
    assert answer["dropped"] == 1
    assert "agent:gone:feishu:dm:chatA" not in store._entries
    assert "agent:keepme:feishu:dm:chatB" in store._entries
    db = _state_db()
    try:
        assert db._read_one(
            "SELECT COUNT(*) AS n FROM gateway_heartbeats WHERE profile = ?", ("gone",))["n"] == 0
    finally:
        db.close()
