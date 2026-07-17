"""Strict topic route registry: exact match, fail-closed everything."""

import pytest

from gateway.topic_routing import (
    GENERAL_TOPIC_THREAD_ID,
    REASON_FOREIGN_CHAT,
    REASON_INVALID_THREAD,
    REASON_MISSING_THREAD,
    REASON_UNKNOWN_THREAD,
    RouteDenied,
    TopicRoute,
    TopicRouteRegistry,
)

OWNER = "208214988"


def make_registry():
    return TopicRouteRegistry.from_config(
        [
            {"chat_id": OWNER, "thread_id": 1, "profile": "sol"},
            {"chat_id": OWNER, "thread_id": 77, "profile": "atlas"},
            {"chat_id": OWNER, "thread_id": 78, "profile": "metis"},
        ]
    )


class TestConstruction:
    def test_general_topic_id_is_one(self):
        assert GENERAL_TOPIC_THREAD_ID == 1

    def test_requires_at_least_one_route(self):
        with pytest.raises(ValueError):
            TopicRouteRegistry([])

    def test_duplicate_key_rejected(self):
        with pytest.raises(ValueError):
            TopicRouteRegistry(
                [
                    TopicRoute(OWNER, 1, "sol"),
                    TopicRoute(OWNER, 1, "other"),
                ]
            )

    def test_duplicate_profile_rejected(self):
        with pytest.raises(ValueError):
            TopicRouteRegistry(
                [
                    TopicRoute(OWNER, 1, "sol"),
                    TopicRoute(OWNER, 2, "sol"),
                ]
            )

    @pytest.mark.parametrize("thread_id", [0, -1, True, "1", None])
    def test_invalid_thread_ids_rejected_at_build(self, thread_id):
        with pytest.raises(ValueError):
            TopicRouteRegistry([TopicRoute(OWNER, thread_id, "sol")])

    def test_from_config_rejects_unknown_keys(self):
        with pytest.raises(ValueError):
            TopicRouteRegistry.from_config(
                [{"chat_id": OWNER, "thread_id": 1, "profile": "sol", "fallback": True}]
            )

    def test_from_config_rejects_missing_fields(self):
        with pytest.raises(ValueError):
            TopicRouteRegistry.from_config([{"chat_id": OWNER, "profile": "sol"}])

    def test_from_config_rejects_string_thread(self):
        with pytest.raises(ValueError):
            TopicRouteRegistry.from_config(
                [{"chat_id": OWNER, "thread_id": "1", "profile": "sol"}]
            )


class TestResolve:
    def test_general_routes_to_sol(self):
        route = make_registry().resolve(OWNER, 1)
        assert route.profile == "sol"
        assert route.thread_id == GENERAL_TOPIC_THREAD_ID

    def test_exact_topic_routes(self):
        registry = make_registry()
        assert registry.resolve(OWNER, 77).profile == "atlas"
        assert registry.resolve(OWNER, 78).profile == "metis"

    def test_string_thread_id_accepted_for_lookup(self):
        # Platform layers may hand over decimal strings.
        assert make_registry().resolve(OWNER, "77").profile == "atlas"

    def test_missing_thread_fails_closed(self):
        with pytest.raises(RouteDenied) as excinfo:
            make_registry().resolve(OWNER, None)
        assert excinfo.value.reason_code == REASON_MISSING_THREAD

    @pytest.mark.parametrize("bad", [0, -5, "0", "-1", "abc", 1.5, True, object()])
    def test_invalid_thread_fails_closed(self, bad):
        with pytest.raises(RouteDenied) as excinfo:
            make_registry().resolve(OWNER, bad)
        assert excinfo.value.reason_code == REASON_INVALID_THREAD

    def test_unknown_thread_fails_closed(self):
        with pytest.raises(RouteDenied) as excinfo:
            make_registry().resolve(OWNER, 2)
        assert excinfo.value.reason_code == REASON_UNKNOWN_THREAD

    def test_deleted_topic_id_fails_closed(self):
        # A formerly-live topic id that is not in the registry behaves
        # identically to never-registered: exact match only.
        with pytest.raises(RouteDenied) as excinfo:
            make_registry().resolve(OWNER, 9999)
        assert excinfo.value.reason_code == REASON_UNKNOWN_THREAD

    @pytest.mark.parametrize("chat", ["999", "", None, "-100555"])
    def test_foreign_chat_fails_closed(self, chat):
        with pytest.raises(RouteDenied) as excinfo:
            make_registry().resolve(chat, 1)
        assert excinfo.value.reason_code == REASON_FOREIGN_CHAT

    def test_foreign_chat_checked_before_thread(self):
        # A foreign chat with a *valid-looking* thread must deny as
        # foreign (route validation precedes everything).
        with pytest.raises(RouteDenied) as excinfo:
            make_registry().resolve("31337", None)
        assert excinfo.value.reason_code == REASON_FOREIGN_CHAT

    def test_no_default_or_last_active_recovery_surface(self):
        registry = make_registry()
        # The registry exposes no API that could return a route without
        # an exact key: resolve() is the only lookup by chat/thread.
        public = {name for name in dir(registry) if not name.startswith("_")}
        assert public == {"resolve", "route_for_profile", "profiles", "from_config"}

    def test_profiles_stable(self):
        assert make_registry().profiles() == ("atlas", "metis", "sol")
