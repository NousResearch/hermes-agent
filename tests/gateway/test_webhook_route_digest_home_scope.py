"""The webhook route digest resolves against the GATEWAY's home, never the routed profile's.

A ``/p/<profile>/webhooks/<route>`` POST is resolved in the gateway's own home (the route lives in
the process ``webhook_subscriptions.json``), but its admission — and any later destination readback
— runs inside the routed profile's runtime scope, whose home holds no subscriptions file. Reloading
the dynamic routes there used to clear them and refuse the delivery with ``not_found``.
"""
import json

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _WebhookDeliveryIdentity
from gateway.platforms.webhook_delivery import route_digest
from hermes_state_runtime import RuntimeStoreError


def _homes(tmp_path, monkeypatch):
    gateway_home = tmp_path / "gateway"
    profile_home = tmp_path / "profiles" / "coordinator"
    gateway_home.mkdir()
    profile_home.mkdir(parents=True)
    (gateway_home / "webhook_subscriptions.json").write_text(json.dumps({
        "jira": {"prompt": "{issue.key}", "profile": "coordinator", "secret": "route-secret"}}))
    monkeypatch.setenv("HERMES_HOME", str(gateway_home))
    return gateway_home, profile_home


def test_digest_inside_the_routed_profile_scope_matches_the_edge(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _gateway_home, profile_home = _homes(tmp_path, monkeypatch)
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": "global-secret"}))
    chat_id = _WebhookDeliveryIdentity.from_parts(
        "coordinator", "jira", "delivery-1").session_chat_id

    edge = route_digest(adapter, chat_id)  # the edge: gateway home, no profile scope

    token = set_hermes_home_override(str(profile_home))
    try:
        assert route_digest(adapter, chat_id) == edge  # admission: inside the routed profile
        assert set(adapter._routes) == {"jira"}  # the table survived the in-scope reload
    finally:
        reset_hermes_home_override(token)


def test_unknown_route_still_refuses_inside_the_routed_profile_scope(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    _gateway_home, profile_home = _homes(tmp_path, monkeypatch)
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": "global-secret"}))
    chat_id = _WebhookDeliveryIdentity.from_parts(
        "coordinator", "not-a-route", "delivery-2").session_chat_id

    token = set_hermes_home_override(str(profile_home))
    try:
        with pytest.raises(RuntimeStoreError):
            route_digest(adapter, chat_id)
    finally:
        reset_hermes_home_override(token)
