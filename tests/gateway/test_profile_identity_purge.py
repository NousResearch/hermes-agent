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


@pytest.mark.asyncio
async def test_unserve_finalizes_only_the_removed_profiles_plugin_manager(tmp_path):
    """Profile finalization runs in the owner's scope and leaves a sibling manager live."""
    from hermes_constants import (
        get_hermes_home,
        hermes_home_key,
        reset_hermes_home_override,
        set_hermes_home_override,
    )
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManifest
    from gateway.run_profile_reconcile import GatewayProfileReconcileMixin

    def in_home(home, callback):
        token = set_hermes_home_override(home)
        try:
            return callback()
        finally:
            reset_hermes_home_override(token)

    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    manager_a = in_home(home_a, plugins_mod.get_plugin_manager)
    manager_b = in_home(home_b, plugins_mod.get_plugin_manager)
    manager_a._discovered = manager_b._discovered = True
    ctx_a = PluginContext(PluginManifest(name="final-a", key="final-a"), manager_a)
    ctx_b = PluginContext(PluginManifest(name="final-b", key="final-b"), manager_b)
    finalized: list[str] = []
    delivered: list[str] = []
    ctx_a.register_hook("on_session_end", lambda **_kw: delivered.append("a-hook"))
    ctx_a.register_platform_handler("telegram", lambda _native, _adapter: delivered.append("a-handler"))
    ctx_a.subscribe("final-a:tick", lambda **_kw: delivered.append("a-event"))
    ctx_a.on_unload(lambda: finalized.append(hermes_home_key(get_hermes_home())))
    ctx_b.register_hook("on_session_end", lambda **_kw: delivered.append("b-hook"))
    ctx_b.register_platform_handler("telegram", lambda _native, _adapter: delivered.append("b-handler"))
    ctx_b.subscribe("final-b:tick", lambda **_kw: delivered.append("b-event"))

    assert in_home(home_a, lambda: ctx_a.emit("tick")) == 1
    assert manager_a._wait_for_event_dispatch(timeout=1)
    assert in_home(home_b, lambda: ctx_b.emit("tick")) == 1
    assert manager_b._wait_for_event_dispatch(timeout=1)
    worker_a, worker_b = manager_a._event_worker, manager_b._event_worker
    assert worker_a is not None and worker_a.is_alive()
    assert worker_b is not None and worker_b.is_alive()

    runner = _runner_stub(None)
    runner._served_profile_homes = {"profile-a": home_a, "profile-b": home_b}
    await GatewayProfileReconcileMixin._unserve_profile(  # type: ignore[arg-type]
        runner, "profile-a", home_a)

    assert finalized == [hermes_home_key(home_a)]
    assert not worker_a.is_alive()
    assert worker_b.is_alive()
    assert manager_a.get_platform_handler_factories("telegram") == []
    assert in_home(home_a, lambda: ctx_a.emit("tick")) == 0
    assert manager_b.get_platform_handler_factories("telegram")
    assert in_home(home_b, lambda: ctx_b.emit("tick")) == 1
    assert manager_b._wait_for_event_dispatch(timeout=1)
    replacement_a = in_home(home_a, plugins_mod.get_plugin_manager)
    assert replacement_a is not manager_a
    assert replacement_a.iter_hook_callbacks("on_session_end") == ()
    assert in_home(home_b, plugins_mod.get_plugin_manager) is manager_b
