"""An entry whose token Desktop could not decrypt on import stays editable from the Webapp.

Contract: such a token is kept as its original ``safeStorage`` envelope. A Webapp save that does not
touch the token (a rename) succeeds and leaves that envelope byte-for-byte as stored; a new entry
with no token is still refused, and a new token replaces the envelope.
"""
import json

import pytest

from hermes_agent.hermes_cli import gateway_connections as store

LOCKED = {"encoding": "safeStorage", "value": "djEwAAAAlocked"}


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store.store_path().write_text(json.dumps({
        "version": 2, "revision": 1, "primary": "local", "launchMode": "primary", "lastUsed": "local",
        "connections": [{"id": "local", "kind": "local", "label": "This device"},
                        {"id": "stuck", "kind": "remote", "label": "Stuck", "url": "http://stuck.lan:9119",
                         "authMode": "token", "token": LOCKED}],
        "importNotices": [{"connectionId": "stuck", "field": "token", "reason": "undecryptable"}]}))
    return tmp_path


def _stored(connection_id: str) -> dict:
    doc = json.loads(store.store_path().read_text())
    return next(c for c in doc["connections"] if c["id"] == connection_id)


def test_a_rename_keeps_the_undecryptable_token_envelope_as_stored(home):
    connection, _registry = store.save_connection({"id": "stuck", "kind": "remote", "label": "Renamed",
                                                   "url": "http://stuck.lan:9119", "authMode": "token"})
    assert connection["label"] == "Renamed"
    assert _stored("stuck")["token"] == LOCKED

    with pytest.raises(store.ConnectionStoreError):
        store.save_connection({"kind": "remote", "label": "New", "url": "http://new.lan:1", "authMode": "token"})
    store.save_connection({"id": "stuck", "kind": "remote", "label": "Renamed", "url": "http://stuck.lan:9119",
                           "authMode": "token", "token": "fresh-token"})
    assert _stored("stuck")["token"] == {"encoding": "plain", "value": "fresh-token"}
