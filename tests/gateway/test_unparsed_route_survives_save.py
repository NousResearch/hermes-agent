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
