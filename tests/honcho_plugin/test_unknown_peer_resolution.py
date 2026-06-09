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
        # an unconfigured name still collapses to the user peer
        assert mgr._resolve_peer_id(s, "stowaway") == s.user_peer_id

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

    def test_resolved_peer_label_reports_collapsed_target(self):
        """The conclude tool response must report the RESOLVED peer, so it never
        confirms a write against a peer name the model invented (#42980 review)."""
        mgr = HonchoSessionManager()
        s = _session()
        mgr._cache[s.key] = s
        assert mgr.resolved_peer_label(s.key, "JackOnDiscord") == s.user_peer_id
        assert mgr.resolved_peer_label(s.key, "ai") == s.assistant_peer_id
        # No session cached -> falls back to the sanitized request, never raises.
        assert mgr.resolved_peer_label("missing:key", "Whoever") == mgr._sanitize_id("Whoever")
