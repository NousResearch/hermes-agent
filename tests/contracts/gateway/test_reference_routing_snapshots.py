"""Routing reads hand out copies: the stored entry text never changes through a returned value."""

import json

import pytest


@pytest.mark.parametrize("replace_all", [False, True])
def test_routing_payloads_are_detached(backend, scope_a, replace_all):
    store_a = backend.open(scope_a)
    payload = {"session_id": "s", "origin": {"nested": {"v": 7}}, "state": {"nested": [7]}}
    text = json.dumps(payload)
    if replace_all:
        store_a.routing.replace_routing_entries({"key": text}, routing_scope="n")
    else:
        store_a.routing.upsert_routing_entry("key", text, routing_scope="n")
    payload["origin"]["nested"]["v"] = 99
    assert store_a.routing.load_routing_entries(routing_scope="n") == {"key": text}
    found = store_a.routing.routing_entry_for_session("s")
    assert found["origin"]["nested"]["v"] == 7
    found["origin"]["nested"]["v"] = 123
    found["state"]["nested"][0] = 123
    again = store_a.routing.routing_entry_for_session("s")
    assert again == {"session_id": "s", "origin": {"nested": {"v": 7}}, "state": {"nested": [7]}}
    assert store_a.routing.load_routing_entries(routing_scope="n") == {"key": text}
