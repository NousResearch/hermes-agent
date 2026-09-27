"""Real PostgreSQL contract for the first gateway route authority slice."""
from __future__ import annotations

from gateway.session_route_store import PostgreSQLSessionRouteStore
from state_store import PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore

_SETTINGS = PostgreSQLStateStoreConfig(
    dsn_env="HERMES_STATE_STORE_TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)


def _store(target):
    return PostgreSQLStateStore(_SETTINGS, target.dsn, schema=target.schema)


def _metadata(key, session_id, source="telegram"):
    return {
        "session_key": key, "session_id": session_id, "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00", "source": source,
    }


def test_gateway_postgresql_route_create_lookup_and_tenant_isolation(postgresql_test_target):
    store = _store(postgresql_test_target)
    try:
        first = PostgreSQLSessionRouteStore(store, tenant_namespace="profile-a")
        second = PostgreSQLSessionRouteStore(store, tenant_namespace="profile-b")
        route = first.get_or_create_route(session_key="agent:default:telegram:chat", session_id="route-a", metadata=_metadata("agent:default:telegram:chat", "route-a"))
        assert route.generation == 1
        assert first.lookup_by_session_id("route-a").session_key == route.session_key
        assert second.lookup_by_key(route.session_key) is None
    finally:
        store.close()


def test_gateway_postgresql_route_cas_reset_and_ambiguity_refusal(postgresql_test_target):
    store = _store(postgresql_test_target)
    try:
        routes = PostgreSQLSessionRouteStore(store, tenant_namespace="profile-a")
        key = "agent:default:telegram:chat"
        route = routes.get_or_create_route(session_key=key, session_id="route-a", metadata=_metadata(key, "route-a"))
        assert routes.switch_route(session_key=key, session_id="route-b", expected_session_id="route-a", expected_generation=route.generation, metadata=_metadata(key, "route-b"), flags={"new": True}) is not None
        stale = routes.switch_route(session_key=key, session_id="route-c", expected_session_id="route-a", expected_generation=route.generation, metadata=_metadata(key, "route-c"), flags={})
        assert stale is None
        current = routes.lookup_by_key(key)
        assert current.session_id == "route-b" and current.flags["new"] is True
        assert routes.delete_or_repair_route(session_key=key, expected_session_id="route-a", expected_generation=route.generation) is False
        assert routes.delete_or_repair_route(session_key=key, expected_session_id=current.session_id, expected_generation=current.generation) is True
    finally:
        store.close()
