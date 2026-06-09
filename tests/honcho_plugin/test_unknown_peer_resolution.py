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

    def test_unknown_peer_does_not_target_a_new_peer(self):
        """Resolving an unknown name yields the existing user peer, so any
        downstream ``_get_or_create_peer`` targets the existing peer rather
        than minting one named after the free-form string."""
        mgr = HonchoSessionManager()
        mgr._get_or_create_peer = MagicMock()
        s = _session()
        resolved = mgr._resolve_peer_id(s, "RandomName")
        assert resolved == s.user_peer_id
        assert resolved != mgr._sanitize_id("RandomName")
