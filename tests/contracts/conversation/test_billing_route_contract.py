"""Contract cases for a usage delta whose route names a billing provider and no model or an empty one: the session row keeps the delta's billing fields and its own model, an empty model fills an unset one with ``''``.

Every case seeds through ``SessionRepository.create`` and ``UsageRepository.queue_usage``/``flush_usage`` and observes through ``SessionRepository.get`` and ``UsageRepository.recent_route``.
"""

from __future__ import annotations

from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import BillingRoute, RouteUsage, TokenCounts, UsageDelta
from persistence.core.identity import SessionId

PROVIDER_ONLY = BillingRoute(model=None, billing_provider="prov", billing_base_url="https://api.example.test", billing_mode="chat")
CALL = UsageDelta(tokens=TokenCounts(input_tokens=10, output_tokens=2), api_call_count=1, route=PROVIDER_ONLY, source="cli")


def _billing(store, sid: str) -> tuple:
    record = store.sessions.get(SessionId(sid))
    return (record.model, record.billing_provider, record.billing_base_url, record.billing_mode, record.api_call_count, record.input_tokens)


def _apply(store, sid: str, delta: UsageDelta = CALL) -> None:
    store.usage.queue_usage(SessionId(sid), delta)
    assert store.usage.flush_usage() is True


def test_a_route_without_a_model_is_representable():
    """CONTRACT: a route may name no model or an empty one."""
    assert PROVIDER_ONLY.model is None
    assert BillingRoute(model="", billing_provider="prov").model == ""


def test_a_provider_without_a_model_fills_the_billing_fields_and_leaves_the_model_unset(backend, scope_a):
    """CONTRACT: on a session with no model and no route the delta's provider, base URL and mode are stored and the model stays unset; the attribution row is ``unknown``, so there is no recent route."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("s1"), source="cli"))
    _apply(store, "s1")
    assert _billing(store, "s1") == (None, "prov", "https://api.example.test", "chat", 1, 10)
    assert store.usage.recent_route(SessionId("s1")) is None


def test_a_provider_without_a_model_keeps_the_session_model_for_the_row_and_its_attribution(backend, scope_a):
    """CONTRACT: on a session with a model the model is kept, the billing fields fill, and the attribution row carries the session's model with the delta's route."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("s2"), source="cli", model="m0"))
    _apply(store, "s2")
    assert _billing(store, "s2") == ("m0", "prov", "https://api.example.test", "chat", 1, 10)
    assert store.usage.recent_route(SessionId("s2")) == RouteUsage(route=BillingRoute("m0", "prov", "https://api.example.test", "chat"), api_call_count=1)


def test_a_provider_without_a_model_never_replaces_a_stored_route(backend, scope_a):
    """CONTRACT: with no model the first accounted call does not replace the stored route: set columns keep their value and unset ones fill; the attribution row still names the delta's route."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("s3"), source="cli", model="m0"))
    store.usage.set_billing_route(SessionId("s3"), provider="stored", base_url="https://stored.test")
    _apply(store, "s3")
    assert _billing(store, "s3") == ("m0", "stored", "https://stored.test", "chat", 1, 10)
    assert store.usage.recent_route(SessionId("s3")) == RouteUsage(route=BillingRoute("m0", "prov", "https://api.example.test", "chat"), api_call_count=1)


def test_a_provider_without_a_model_and_no_usage_writes_no_route(backend, scope_a):
    """CONTRACT: a delta that accounts nothing leaves the billing fields unset even when its route names a provider."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("s4"), source="cli"))
    _apply(store, "s4", UsageDelta(route=PROVIDER_ONLY, source="cli"))
    assert _billing(store, "s4") == (None, None, None, None, 0, 0)
    assert store.usage.recent_route(SessionId("s4")) is None


EMPTY_MODEL = BillingRoute(model="", billing_provider="prov", billing_base_url="https://api.example.test", billing_mode="chat")


def test_an_empty_model_fills_an_unset_session_model_with_empty_text(backend, scope_a):
    """CONTRACT: on a session with no model an empty route model is written as ``''`` beside the billing fields; the attribution row is ``unknown``, so there is no recent route."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("e1"), source="cli"))
    _apply(store, "e1", UsageDelta(tokens=TokenCounts(input_tokens=10, output_tokens=2), api_call_count=1, route=EMPTY_MODEL, source="cli"))
    assert _billing(store, "e1") == ("", "prov", "https://api.example.test", "chat", 1, 10)
    assert store.usage.recent_route(SessionId("e1")) is None


def test_an_empty_model_keeps_the_session_model(backend, scope_a):
    """CONTRACT: on a session with a model an empty route model changes nothing about the model; the attribution row carries the session's model with the delta's route."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("e2"), source="cli", model="m0"))
    _apply(store, "e2", UsageDelta(tokens=TokenCounts(input_tokens=10, output_tokens=2), api_call_count=1, route=EMPTY_MODEL, source="cli"))
    assert _billing(store, "e2") == ("m0", "prov", "https://api.example.test", "chat", 1, 10)
    assert store.usage.recent_route(SessionId("e2")) == RouteUsage(route=BillingRoute("m0", "prov", "https://api.example.test", "chat"), api_call_count=1)


def test_an_empty_model_without_usage_still_fills_the_model(backend, scope_a):
    """CONTRACT: a delta that accounts nothing leaves the billing fields unset but still fills an unset model with ``''``."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("e3"), source="cli"))
    _apply(store, "e3", UsageDelta(route=EMPTY_MODEL, source="cli"))
    assert _billing(store, "e3") == ("", None, None, None, 0, 0)
    assert store.usage.recent_route(SessionId("e3")) is None


def test_an_empty_model_alone_fills_the_model_and_no_billing_field(backend, scope_a):
    """CONTRACT: a route naming only an empty model accounts the call, fills an unset model with ``''`` and no billing field."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SessionId("e4"), source="cli"))
    _apply(store, "e4", UsageDelta(tokens=TokenCounts(input_tokens=1), api_call_count=1, route=BillingRoute(model=""), source="cli"))
    assert _billing(store, "e4") == ("", None, None, None, 1, 1)
    assert store.usage.recent_route(SessionId("e4")) is None
