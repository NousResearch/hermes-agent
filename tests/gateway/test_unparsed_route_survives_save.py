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
