"""A routing row whose platform's plugin is not registered yet (a platform that left core, installed only in a
secondary profile or after boot) survives a full routing save instead of being deleted."""

import json

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


def test_a_route_for_an_unregistered_platform_is_written_back_on_save(tmp_path):
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        live = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
        scope = store._routing_scope()
        saved = json.loads(json.dumps(live.to_dict()))
        saved.update(session_key="agent:main:qqbot:c2c:U1", platform="qqbot", session_id="sid-qq",
                     origin={"platform": "qqbot", "chat_id": "U1", "chat_type": "c2c"})
        db.routing.upsert_routing_entry("agent:main:qqbot:c2c:U1", json.dumps(saved), routing_scope=scope)

        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        restarted.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))  # full save
        with restarted._lock:
            restarted._save()
        rows = db.routing.load_routing_entries(routing_scope=scope)
        assert "agent:main:qqbot:c2c:U1" in rows
        assert json.loads(rows["agent:main:qqbot:c2c:U1"])["session_id"] == "sid-qq"


def test_an_older_sessions_json_copy_does_not_replace_the_held_db_route(tmp_path):
    """state.db is primary: the legacy sessions.json mirror never overwrites a held unparsed row."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        live = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
        scope, key = store._routing_scope(), "agent:main:qqbot:c2c:U1"
        newer = {**live.to_dict(), "session_key": key, "platform": "qqbot", "session_id": "sid-new",
                 "origin": {"platform": "qqbot", "chat_id": "U1", "chat_type": "c2c"}}
        db.routing.upsert_routing_entry(key, json.dumps(newer), routing_scope=scope)
        mirror = store.sessions_dir / "sessions.json"
        data = json.loads(mirror.read_text(encoding="utf-8")) if mirror.exists() else {}
        data[key] = {**newer, "session_id": "sid-old"}
        mirror.write_text(json.dumps(data), encoding="utf-8")

        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        restarted.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))
        with restarted._lock:
            restarted._save()
        assert json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])["session_id"] == "sid-new"


def _unregister(value):
    member = Platform._value2member_map_.pop(value, None)
    if member is not None:
        Platform._member_map_.pop(member._name_, None)


def _qq_source():
    return SessionSource(platform=Platform("qqbot"), chat_id="U1", chat_type="dm")


def test_a_held_route_s_stop_still_applies_once_its_platform_registers(tmp_path):
    """A multiplexed profile's plugin registers after the shared index loaded: the held row joins the live
    index then, so /stop's suspension still starts a new session after the restart (#7536), as core's QQ
    route did, instead of the stopped session being recovered from state.db."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()):
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        Platform._add_pseudo_member("qqbot")  # the previous run had the plugin
        try:
            stuck = store.get_or_create_session(_qq_source())
            assert store.suspend_session(stuck.session_key)
            _unregister("qqbot")
            restarted = SessionStore(store.sessions_dir, GatewayConfig())
            with restarted._lock:
                restarted._ensure_loaded_locked()  # boot: not registered yet, the row is held
            Platform._add_pseudo_member("qqbot")  # the profile's plugin registers
            assert restarted.get_or_create_session(_qq_source()).session_id != stuck.session_id
        finally:
            _unregister("qqbot")


def test_a_held_route_s_crash_left_turn_resumes_once_its_platform_registers(tmp_path):
    """The boot's turn-marker recovery runs before a multiplexed profile's plugin registers: the held row gets
    that verdict when it joins, so the interrupted turn is ``resume_pending`` as core's QQ route was."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()):
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        Platform._add_pseudo_member("qqbot")
        try:
            running = store.get_or_create_session(_qq_source())
            assert store.mark_turn_active(running.session_key)  # the previous run died mid-turn
            _unregister("qqbot")
            restarted = SessionStore(store.sessions_dir, GatewayConfig())
            restarted.recover_interrupted_turns()  # boot: the row is held, the verdict is not lost
            Platform._add_pseudo_member("qqbot")
            entry = restarted.get_or_create_session(_qq_source())
            assert (entry.session_id, entry.resume_pending, entry.active_turn_token) == (
                running.session_id, True, None)
        finally:
            _unregister("qqbot")


def _held_profile_route(tmp_path, db, key):
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    live = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
    row = {**live.to_dict(), "session_key": key, "platform": "qqbot", "session_id": "sid-qq",
           "origin": {"platform": "qqbot", "chat_id": "U1", "chat_type": "dm", "profile": "b"}}
    db.routing.upsert_routing_entry(key, json.dumps(row), routing_scope=store._routing_scope())
    restarted = SessionStore(store.sessions_dir, GatewayConfig())
    with restarted._lock:
        restarted._ensure_loaded_locked()
    return restarted


def test_deleting_a_profile_drops_its_held_routes_too(tmp_path):
    """The live gateway's profile purge (#111926) must not be undone by the next save writing a held row back."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = _held_profile_route(tmp_path, db, "agent:b:qqbot:dm:U1")
        store._routing_persistence.profile_identity.purge_profile("b")
        store.purge_profile_routing("b")
        store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))  # full save
        assert "agent:b:qqbot:dm:U1" not in db.routing.load_routing_entries(routing_scope=store._routing_scope())


def test_renaming_a_profile_moves_its_held_routes_too(tmp_path):
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = _held_profile_route(tmp_path, db, "agent:b:qqbot:dm:U1")
        store._routing_persistence.profile_identity.rekey_profile("b", "c")
        store.rekey_profile_routing("b", "c")
        store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t2"))  # full save
        rows = db.routing.load_routing_entries(routing_scope=store._routing_scope())
        assert "agent:b:qqbot:dm:U1" not in rows
        assert json.loads(rows["agent:c:qqbot:dm:U1"])["origin"]["profile"] == "c"


def test_a_route_broken_another_way_is_not_held_and_the_mirror_copy_recovers_it(tmp_path):
    """Only an unregistered platform makes a row held: a DB row with a corrupt ``updated_at`` stays skipped, so
    the valid sessions.json copy loads and the next save writes the valid one."""
    from persistence_assembly import open_persistence, profile_scope
    with open_persistence(profile_scope()) as db:
        store = SessionStore(tmp_path / "sessions", GatewayConfig())
        live = store.get_or_create_session(SessionSource(platform=Platform.TELEGRAM, chat_id="t1"))
        scope, key = store._routing_scope(), "agent:main:telegram:dm:t9"
        valid = {**live.to_dict(), "session_key": key, "session_id": "sid-ok",
                 "origin": {"platform": "telegram", "chat_id": "t9", "chat_type": "dm"}}
        db.routing.upsert_routing_entry(key, json.dumps({**valid, "updated_at": "not-a-date"}), routing_scope=scope)
        mirror = store.sessions_dir / "sessions.json"
        data = json.loads(mirror.read_text(encoding="utf-8")) if mirror.exists() else {}
        data[key] = valid
        mirror.write_text(json.dumps(data), encoding="utf-8")

        restarted = SessionStore(store.sessions_dir, GatewayConfig())
        with restarted._lock:
            restarted._ensure_loaded_locked()
            assert restarted._entries[key].session_id == "sid-ok"
            restarted._save()
        assert json.loads(db.routing.load_routing_entries(routing_scope=scope)[key])["updated_at"] != "not-a-date"
