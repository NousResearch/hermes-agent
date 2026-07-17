"""Telegram forum topic -> distinct persona profile routing contract.

Hermes has TWO disjoint "profile" concepts and this file pins the SECOND one:

  1. GATE layer  — ``gateway/topic_routing.py`` (``TopicRouteRegistry``). A
     fail-closed admission gate + hook binding. It decides *whether* an inbound
     topic is allowed and which topic hook fires. Covered by
     ``tests/gateway/test_topic_routing.py`` (atlas thread 77 / metis thread 78
     at the gate). NOT re-covered here.

  2. PERSONA layer — ``gateway/profile_routing.py``
     (``ProfileRoute`` + ``match_profile_route`` + ``parse_profile_routes``).
     This is the layer that actually swaps persona / toolset / HERMES_HOME:
     it matches on (platform, guild_id, chat_id, thread_id, parent_chat_id) and
     its result is stamped onto ``source.profile`` by ``build_source``
     (``gateway/platforms/base.py``) and resolved to a profile HOME by
     ``GatewayRunner._resolve_profile_home_for_source`` (``gateway/run.py``),
     gated behind ``gateway.multiplex_profiles``.

This file locks layer (2): routing an ATLAS / METIS Telegram forum topic to a
distinct persona is pure config — no core change. ATLAS/METIS topic personas
depend on this contract, so it must not silently regress.

NOTE ON ID TYPES: ``profile_routes`` compares discriminators by exact equality
and stores them as *strings* (see ``ProfileRoute`` / ``match_profile_route``),
whereas the ``topic_routing`` gate uses *integer* thread ids. The constants
below are therefore strings for the persona layer; the one composition test
that also exercises the gate feeds the same numeric values as ints there.

All ids are ID-free placeholders — never real owner chat ids.
"""

from gateway.profile_routing import (
    ProfileRoute,
    match_profile_route,
    parse_profile_routes,
)

PLATFORM = "telegram"

# Placeholder owner supergroup chat id (Telegram forum). NOT a real id.
OWNER_CHAT = "-1001000000000"
# A foreign chat that must never inherit the owner's personas.
FOREIGN_CHAT = "-1009999999999"

# Forum topic (message thread) ids. Strings, because profile_routes matches by
# exact string equality (unlike the int-keyed topic_routing gate).
SOL_GENERAL = "1"  # the "General" topic — no persona route registered
ATLAS_THREAD = "77"
METIS_THREAD = "78"
UNREGISTERED_THREAD = "999"  # a topic with no persona route


def _persona_routes():
    """Build the persona routes the real way — through ``parse_profile_routes``
    (the config.yaml -> ProfileRoute path), so the test exercises validation,
    normalization, and most-specific-first sorting exactly as the gateway does.
    """
    return parse_profile_routes(
        [
            {
                "name": "atlas-topic",
                "platform": PLATFORM,
                "chat_id": OWNER_CHAT,
                "thread_id": ATLAS_THREAD,
                "profile": "atlas",
            },
            {
                "name": "metis-topic",
                "platform": PLATFORM,
                "chat_id": OWNER_CHAT,
                "thread_id": METIS_THREAD,
                "profile": "metis",
            },
        ]
    )


def _match(routes, chat_id=OWNER_CHAT, thread_id=None, parent_chat_id=None):
    """Thin wrapper over the REAL matcher with the exact signature the gateway
    calls in ``GatewayRunner._profile_name_for_source`` (gateway/run.py)."""
    return match_profile_route(
        routes,
        platform=PLATFORM,
        guild_id=None,  # Telegram has no guild_id
        chat_id=chat_id,
        thread_id=thread_id,
        parent_chat_id=parent_chat_id,
    )


class TestTopicResolvesToPersona:
    """(1) A registered forum topic resolves to its distinct persona."""

    def test_atlas_topic_routes_to_atlas(self):
        matched = _match(_persona_routes(), thread_id=ATLAS_THREAD)
        assert matched is not None
        assert matched.profile == "atlas"

    def test_metis_topic_routes_to_metis(self):
        matched = _match(_persona_routes(), thread_id=METIS_THREAD)
        assert matched is not None
        assert matched.profile == "metis"

    def test_two_topics_do_not_cross_route(self):
        routes = _persona_routes()
        # Same chat, different topic -> different persona. No bleed-through.
        assert _match(routes, thread_id=ATLAS_THREAD).profile == "atlas"
        assert _match(routes, thread_id=METIS_THREAD).profile == "metis"


class TestFailClosedAtPersonaLayer:
    """(2) Every non-registered topic keeps default behavior — no persona is
    applied (matcher returns None), and it is never misrouted to atlas/metis.
    """

    def test_unregistered_thread_same_chat_returns_none(self):
        # A topic on the owner chat that has no persona route.
        assert _match(_persona_routes(), thread_id=UNREGISTERED_THREAD) is None

    def test_general_topic_returns_none(self):
        # The "General" topic (thread 1) has no persona route in this set,
        # so it stays on the default profile.
        assert _match(_persona_routes(), thread_id=SOL_GENERAL) is None

    def test_foreign_chat_returns_none_even_for_registered_thread_id(self):
        # A foreign chat reusing a registered thread id must NOT inherit atlas.
        assert (
            _match(_persona_routes(), chat_id=FOREIGN_CHAT, thread_id=ATLAS_THREAD)
            is None
        )

    def test_missing_thread_returns_none(self):
        # No thread id at all (a plain chat message) -> no thread-scoped persona.
        assert _match(_persona_routes(), thread_id=None) is None

    def test_no_misroute_to_atlas_or_metis(self):
        # Consolidated guard: none of the "other topic" shapes leak a persona.
        routes = _persona_routes()
        for chat_id, thread_id in (
            (OWNER_CHAT, UNREGISTERED_THREAD),
            (OWNER_CHAT, SOL_GENERAL),
            (OWNER_CHAT, None),
            (FOREIGN_CHAT, ATLAS_THREAD),
            (FOREIGN_CHAT, METIS_THREAD),
        ):
            matched = _match(routes, chat_id=chat_id, thread_id=thread_id)
            assert matched is None, (chat_id, thread_id, matched)


class TestSpecificityThreadPrecedence:
    """(3) profile_routing defines specificity (thread_id=8, chat_id=4,
    guild_id=2), and ``parse_profile_routes`` sorts most-specific-first. A
    thread-specific route must win over a same-chat chat-only catch-all.
    """

    @staticmethod
    def _routes_with_catchall():
        # A chat-wide default persona PLUS a thread-specific override.
        return parse_profile_routes(
            [
                {
                    "name": "owner-chat-default",
                    "platform": PLATFORM,
                    "chat_id": OWNER_CHAT,
                    "profile": "sol",  # catch-all for the whole forum
                },
                {
                    "name": "atlas-topic",
                    "platform": PLATFORM,
                    "chat_id": OWNER_CHAT,
                    "thread_id": ATLAS_THREAD,
                    "profile": "atlas",  # more specific override
                },
            ]
        )

    def test_parse_sorts_most_specific_first(self):
        routes = self._routes_with_catchall()
        # The thread-specific route (specificity 12) sorts ahead of the
        # chat-only route (specificity 4).
        assert [r.name for r in routes] == ["atlas-topic", "owner-chat-default"]
        assert routes[0].specificity > routes[1].specificity

    def test_thread_specific_wins_over_chat_catchall(self):
        routes = self._routes_with_catchall()
        # On the atlas thread, the specific persona wins.
        assert _match(routes, thread_id=ATLAS_THREAD).profile == "atlas"

    def test_other_threads_fall_through_to_chat_catchall(self):
        routes = self._routes_with_catchall()
        # Any other topic on the same chat gets the chat-wide default persona,
        # because the chat-only route has no thread constraint.
        assert _match(routes, thread_id=SOL_GENERAL).profile == "sol"
        assert _match(routes, thread_id=UNREGISTERED_THREAD).profile == "sol"

    def test_chat_catchall_does_not_leak_to_foreign_chat(self):
        routes = self._routes_with_catchall()
        assert _match(routes, chat_id=FOREIGN_CHAT, thread_id=SOL_GENERAL) is None


class TestGateAndPersonaCompose:
    """(4) Composition note (cleanly unit-testable, no full gateway):

    A registered atlas topic both (a) PASSES the strict ``topic_routing`` gate
    AND (b) resolves to persona "atlas" at the ``profile_routes`` layer. This
    documents that the two layers agree on the same topic while staying
    independent modules. The gate uses INT thread ids; the persona layer uses
    STRING thread ids — same numeric topic, different type contracts.
    """

    def test_atlas_topic_passes_gate_and_selects_persona(self):
        from gateway.topic_routing import TopicRouteRegistry

        atlas_int = int(ATLAS_THREAD)

        # (a) Gate layer: strict registry admits the atlas topic and binds it
        # to the atlas topic hook/profile.
        registry = TopicRouteRegistry.from_config(
            [
                {"chat_id": OWNER_CHAT, "thread_id": int(SOL_GENERAL), "profile": "sol"},
                {"chat_id": OWNER_CHAT, "thread_id": atlas_int, "profile": "atlas"},
                {"chat_id": OWNER_CHAT, "thread_id": int(METIS_THREAD), "profile": "metis"},
            ]
        )
        assert registry.resolve(OWNER_CHAT, atlas_int).profile == "atlas"

        # (b) Persona layer: profile_routes selects the atlas persona for the
        # same topic, which is what build_source stamps onto source.profile.
        matched = _match(_persona_routes(), thread_id=ATLAS_THREAD)
        assert matched is not None and matched.profile == "atlas"
