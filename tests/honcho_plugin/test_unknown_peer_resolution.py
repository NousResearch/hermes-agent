"""Regression tests for issue #42980 — Honcho must not mint a new peer for
every free-form name the model uses to address the user.

``_resolve_peer_id`` previously returned a sanitized version of ANY caller-
supplied peer string, and ``_get_or_create_peer`` then lazily created a brand-
new Honcho peer for it. So when the model addressed the user by a display
name (a Discord handle, a name pulled from chat history, ...), each variant
spawned a duplicate peer; Honcho's consolidation ("dream") pass later
discarded them, losing the user model. The resolver now collapses
unrecognized names to the stable user peer and only honors *known* peers
(the session's own peers + operator-configured aliases).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager


def _session() -> HonchoSession:
    return HonchoSession(
        key="discord:42",
        user_peer_id="user-discord-42",
        assistant_peer_id="hermes-assistant",
        honcho_session_id="discord-42",
    )


class TestUnknownPeerResolution:
    def test_user_alias_and_empty_resolve_to_user_peer(self):
        mgr = HonchoSessionManager()
        s = _session()
        assert mgr._resolve_peer_id(s, "user") == s.user_peer_id
        assert mgr._resolve_peer_id(s, None) == s.user_peer_id
        assert mgr._resolve_peer_id(s, "") == s.user_peer_id
        assert mgr._resolve_peer_id(s, "  ") == s.user_peer_id

    def test_ai_alias_resolves_to_assistant_peer(self):
        mgr = HonchoSessionManager()
        s = _session()
        assert mgr._resolve_peer_id(s, "ai") == s.assistant_peer_id

    def test_ai_aliases_are_case_insensitive(self):
        """Plausible model guesses ("AI", "Assistant", " ai ") must route to the
        assistant, not silently collapse into the user peer (#42980 review)."""
        mgr = HonchoSessionManager()
        s = _session()
        for alias in ("AI", "Ai", " ai ", "assistant", "Assistant", "ASSISTANT"):
            assert mgr._resolve_peer_id(s, alias) == s.assistant_peer_id, alias
        # "user" is likewise case-insensitive
        assert mgr._resolve_peer_id(s, "USER") == s.user_peer_id

    def test_unknown_name_collapses_to_user_peer(self):
        """The core #42980 fix: an invented display name must NOT become its
        own peer id."""
        mgr = HonchoSessionManager()
        s = _session()
        for name in ("JackOnDiscord", "Jack", "qa-bot", "Mr. User"):
            assert mgr._resolve_peer_id(s, name) == s.user_peer_id, name
            # and crucially it is NOT the sanitized free-form name
            assert mgr._resolve_peer_id(s, name) != mgr._sanitize_id(name), name

    def test_sessions_own_peer_ids_pass_through(self):
        """If the model echoes back a real resolved peer id, honor it."""
        mgr = HonchoSessionManager()
        s = _session()
        assert mgr._resolve_peer_id(s, s.user_peer_id) == s.user_peer_id
        assert mgr._resolve_peer_id(s, s.assistant_peer_id) == s.assistant_peer_id

    def test_configured_alias_is_honored(self):
        """Operator-declared peers (peer_name / user_peer_aliases) remain valid
        explicit targets — preserves the deliberate multi-peer escape hatch."""
        mgr = HonchoSessionManager()
        mgr._config = SimpleNamespace(
            peer_name="captain",
            user_peer_aliases={"discord-99": "first-mate"},
        )
        s = _session()
        assert mgr._resolve_peer_id(s, "captain") == mgr._sanitize_id("captain")
        assert mgr._resolve_peer_id(s, "first-mate") == mgr._sanitize_id("first-mate")
        # Case-insensitive: a configured alias referenced in a different case
        # still resolves to its canonical id (review fix — the alias matching
        # must be as case-insensitive as the built-in 'user'/'ai' aliases).
        assert mgr._resolve_peer_id(s, "First-Mate") == mgr._sanitize_id("first-mate")
        assert mgr._resolve_peer_id(s, "CAPTAIN") == mgr._sanitize_id("captain")
        # an unconfigured name still collapses to the user peer
        assert mgr._resolve_peer_id(s, "stowaway") == s.user_peer_id

    def test_real_peer_named_like_reserved_alias_is_not_hijacked(self):
        """A real peer whose id matches a reserved alias must win over the alias
        fallback — known peers are checked first (review regression fix)."""
        mgr = HonchoSessionManager()
        s = HonchoSession(
            key="x:1",
            user_peer_id="AI",  # pathological but legal: user peer literally "AI"
            assistant_peer_id="hermes-assistant",
            honcho_session_id="x-1",
        )
        # Without known-peers-first, "AI" would be hijacked to the assistant.
        assert mgr._resolve_peer_id(s, "AI") == "AI"
        assert mgr._resolve_peer_id(s, "ai") == "AI"

    def test_case_colliding_configured_ids_resolve_deterministically(self):
        """Regression (#43086 review): config parsing permits peer IDs that
        differ only by case. A case-insensitive query must resolve the SAME
        winner on every call — lexicographically-first among configured IDs —
        never whichever came out of set iteration."""
        mgr = HonchoSessionManager()
        mgr._config = SimpleNamespace(
            peer_name=None,
            user_peer_aliases={"a": "First-Mate", "b": "first-mate"},
        )
        s = _session()
        # Exact matches always return themselves.
        assert mgr._resolve_peer_id(s, "First-Mate") == "First-Mate"
        assert mgr._resolve_peer_id(s, "first-mate") == "first-mate"
        # A third casing hits the case-insensitive map: sorted() puts
        # "First-Mate" first ("F" < "f"), and that winner is stable.
        for _ in range(20):
            assert mgr._resolve_peer_id(s, "FIRST-MATE") == "First-Mate"

    def test_session_peer_outranks_case_colliding_configured_id(self):
        """On a case-insensitive collision between a session peer and a
        configured ID, the session's own peer wins deterministically."""
        mgr = HonchoSessionManager()
        mgr._config = SimpleNamespace(
            peer_name="USER-DISCORD-42",  # collides with the session user peer
            user_peer_aliases={},
        )
        s = _session()  # user_peer_id="user-discord-42"
        assert mgr._resolve_peer_id(s, "User-Discord-42") == s.user_peer_id

    def test_resolved_peer_label_for_user_peer_named_ai(self):
        """The friendly label follows the resolver's known-peers-first rule: a
        session user peer literally named "AI" is labeled "user", so the
        empty-card hint reports the settings that actually applied."""
        mgr = HonchoSessionManager()
        s = HonchoSession(
            key="x:1",
            user_peer_id="AI",
            assistant_peer_id="hermes-assistant",
            honcho_session_id="x-1",
        )
        mgr._cache[s.key] = s
        assert mgr.resolved_peer_label(s.key, "AI") == "user"
        assert mgr.resolved_peer_label(s.key, "ai") == "user"

    def test_create_conclusion_for_unknown_peer_writes_to_user_scope(self):
        """End-to-end through the WRITE path: a conclusion for an invented name
        must land in the user's conclusion scope, and ``_get_or_create_peer``
        must never be called with the sanitized free-form name (which is what
        lazily mints a duplicate Honcho peer). This exercises the real
        ``create_conclusion`` flow, not just the resolver in isolation."""
        mgr = HonchoSessionManager()
        s = _session()
        mgr._cache[s.key] = s
        mgr._ai_observe_others = True
        assistant = MagicMock()
        mgr._get_or_create_peer = MagicMock(return_value=assistant)

        assert mgr.create_conclusion(s.key, "likes dark mode", peer="JackOnDiscord") is True

        # The conclusion is scoped to the USER peer, not a new "jackondiscord".
        assistant.conclusions_of.assert_called_once_with(s.user_peer_id)
        called_with = [c.args[0] for c in mgr._get_or_create_peer.call_args_list]
        assert mgr._sanitize_id("JackOnDiscord") not in called_with

    def test_create_conclusion_unknown_peer_without_ai_observe_others(self):
        """The ai_observe_others=False branch is the actual mint site a
        non-collapsing resolver would hit: it calls _get_or_create_peer on the
        TARGET peer. An invented name must still target the user peer, so no new
        peer is minted there either (review coverage gap)."""
        mgr = HonchoSessionManager()
        s = _session()
        mgr._cache[s.key] = s
        mgr._ai_observe_others = False
        target = MagicMock()
        mgr._get_or_create_peer = MagicMock(return_value=target)

        assert mgr.create_conclusion(s.key, "fact", peer="MadeUpName") is True

        mgr._get_or_create_peer.assert_called_once_with(s.user_peer_id)
        target.conclusions_of.assert_called_once_with(s.user_peer_id)

    def test_resolved_peer_label_uses_friendly_aliases(self):
        """The conclude response reports the RESOLVED peer, but as the friendly
        'user'/'ai' label for the session peers rather than leaking the raw
        internal peer id into the model's context (#42980 review)."""
        mgr = HonchoSessionManager()
        s = _session()
        mgr._cache[s.key] = s
        # Invented name collapses to the user -> reported as "user", not the raw id.
        assert mgr.resolved_peer_label(s.key, "JackOnDiscord") == "user"
        assert mgr.resolved_peer_label(s.key, "user") == "user"
        assert mgr.resolved_peer_label(s.key, "ai") == "ai"
        # No session cached -> falls back to the sanitized request, never raises.
        assert mgr.resolved_peer_label("missing:key", "Whoever") == mgr._sanitize_id("Whoever")
