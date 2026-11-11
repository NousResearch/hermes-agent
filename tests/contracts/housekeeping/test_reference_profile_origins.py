"""Profile renames rewrite matching session origins independently of route keys."""

import json

from persistence.conversation.sessions import NewSession


def test_profile_rekey_rewrites_session_origins(backend, scope_a):
    store_a = backend.open(scope_a)
    for sid, key, profile in [("a", "agent:old:a", "old"), ("b", "other:b", "old"), ("c", "agent:old:c", "other")]:
        store_a.sessions.create(NewSession(sid, "telegram", session_key=key))
        store_a.routing.record_session_peer(sid, source="telegram", session_key=key, origin_json=json.dumps({"profile": profile, "nested": [7]}))
    result = store_a.profile_identity.rekey_profile("old", "new")
    assert result.sessions_session_key == 2
    assert result.sessions_origin_json == 2
    assert store_a.sessions.get("a").origin == {"profile": "new", "nested": [7]}
    assert store_a.sessions.get("b").origin == {"profile": "new", "nested": [7]}
    assert store_a.sessions.get("c").origin == {"profile": "other", "nested": [7]}
