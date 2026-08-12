"""Unit tests for relay wire-field hygiene (Phase 1 parity).

Covers _event_from_wire's consumption of the contract §3 user-identity
enrichment fields the connector has always sent but the gateway used to drop:

  - ``user_display_name`` — the human-facing display name (native parity: the
    native Discord adapter surfaces ``message.author.display_name`` as
    ``user_name``).
  - ``user_handle`` — the raw platform handle, used only as a last resort.

Precedence: user_display_name > user_name > user_handle. Session keys derive
from user_id (never user_name), so this mapping is presentation-only and must
remain key-stable — asserted here via build_session_key.

Pure unit tests: no socket, no websockets dependency.
"""

from __future__ import annotations

from gateway.relay.ws_transport import _event_from_wire
from gateway.session import build_session_key


def _wire_event(**src_overrides):
    src = {
        "platform": "discord",
        "chat_id": "chan-1",
        "chat_type": "group",
        "user_id": "u-1",
        "user_name": "rawusername",
        "scope_id": "guild-1",
    }
    src.update(src_overrides)
    return {"text": "hello", "message_type": "text", "source": src}


class TestUserIdentityEnrichment:


    def test_handle_is_last_resort(self):
        ev = _event_from_wire(
            _wire_event(user_name=None, user_handle="ben#1234")
        )
        assert ev.source.user_name == "ben#1234"


    def test_session_key_is_stable_across_name_shapes(self):
        """user_name is presentation-only: the same user_id keys the same
        session whether or not the connector sent the enrichment fields."""
        plain = _event_from_wire(_wire_event())
        enriched = _event_from_wire(
            _wire_event(user_display_name="Ben Display", user_handle="ben#1234")
        )
        assert build_session_key(plain.source) == build_session_key(enriched.source)
