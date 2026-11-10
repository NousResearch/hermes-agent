"""Peer publication is visible through canonical session readers."""

import json

from persistence.conversation.sessions import NewSession


def test_peer_publication_reaches_session_readers(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession("s", "telegram", session_key="key"))
    store_a.routing.record_session_peer("s", source="telegram", session_key="key", display_name="Label", origin_json=json.dumps({"profile": "old"}))
    entry = next(e.session for e in store_a.sessions.list_gateway_sessions() if e.session.session_id == "s")
    assert entry.display_name == "Label"
    assert entry.origin == {"profile": "old"}
    recovered = store_a.routing.latest_session_for_peer(source="telegram", session_key="key")
    assert recovered.session.display_name == "Label"
    assert recovered.session.origin == {"profile": "old"}
