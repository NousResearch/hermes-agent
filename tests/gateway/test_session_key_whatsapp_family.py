"""Session-key standardization for the WhatsApp family.

One human must map to ONE session key regardless of which transport delivered the
message (Baileys ``whatsapp``, WAHA plugin ``waha``, Meta Cloud ``whatsapp_cloud``) or
which wire dialect it used (``@s.whatsapp.net`` / ``@c.us`` / bare digits) — and
pre-canonical sessions must be ADOPTED onto the canonical key instead of orphaned.
"""
from __future__ import annotations

import threading
from datetime import datetime

import pytest

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.session_recovery import SessionRecoveryMixin

DIALECTS = ("6285157813352@c.us", "6285157813352@s.whatsapp.net", "6285157813352")


@pytest.mark.parametrize("platform_value", ("whatsapp", "waha", "whatsapp_cloud"))
@pytest.mark.parametrize("dialect", DIALECTS)
def test_one_human_one_key_across_dialects_and_transports(platform_value, dialect):
    source = SessionSource(platform=Platform(platform_value), chat_id=dialect,
                           chat_type="dm", user_id=dialect)
    assert build_session_key(source) == f"agent:main:{platform_value}:dm:6285157813352"


def test_group_participant_canonicalized_for_the_family():
    # Group keys isolate per participant; a dialect flip must not split one member.
    for dialect in DIALECTS:
        source = SessionSource(platform=Platform("waha"), chat_id="1203634@g.us",
                               chat_type="group", user_id=dialect)
        key = build_session_key(source, group_sessions_per_user=True)
        assert key.endswith(":6285157813352"), key


def test_non_whatsapp_platforms_keep_their_native_ids():
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123456789",
                           chat_type="dm", user_id="123456789")
    assert build_session_key(source) == "agent:main:telegram:dm:123456789"


class _Store(SessionRecoveryMixin):
    """Minimal harness: just the routing index + the collaborators adoption touches."""

    def __init__(self, entries):
        self._lock = threading.RLock()
        self._entries = entries
        self.saved = 0
        self.peers = []
        self._ensure_loaded_locked = lambda: None
        self._save_entries = lambda: setattr(self, "saved", self.saved + 1)
        self._record_gateway_session_peer = (
            lambda session_id, session_key, source, display_name=None: self.peers.append(session_key))


def _entry(key, session_id, updated_at):
    return SessionEntry(session_key=key, session_id=session_id,
                        created_at=datetime(2026, 9, 7), updated_at=updated_at)


def test_adoption_folds_legacy_dialect_keys_onto_the_canonical_key():
    store = _Store({
        "agent:main:waha:dm:6285157813352@c.us":
            _entry("agent:main:waha:dm:6285157813352@c.us", "sess-cus", datetime(2026, 9, 8, 10)),
        "agent:main:waha:dm:6285157813352@s.whatsapp.net":
            _entry("agent:main:waha:dm:6285157813352@s.whatsapp.net", "sess-swn", datetime(2026, 9, 8, 9)),
    })
    source = SessionSource(platform=Platform("waha"), chat_id="6285157813352@s.whatsapp.net",
                           chat_type="dm", user_id="6285157813352@s.whatsapp.net")
    canonical = build_session_key(source)

    store._adopt_legacy_whatsapp_entry(source, canonical)

    assert canonical in store._entries
    # Most recently updated variant wins; the moved entry keeps its transcript id.
    assert store._entries[canonical].session_id == "sess-cus"
    assert "agent:main:waha:dm:6285157813352@c.us" not in store._entries
    assert store.saved == 1 and store.peers == [canonical]


def test_adoption_is_a_noop_when_the_canonical_key_already_exists():
    store = _Store({
        "agent:main:waha:dm:6285157813352":
            _entry("agent:main:waha:dm:6285157813352", "sess-current", datetime(2026, 9, 8, 11)),
        "agent:main:waha:dm:6285157813352@c.us":
            _entry("agent:main:waha:dm:6285157813352@c.us", "sess-old", datetime(2026, 9, 8, 10)),
    })
    source = SessionSource(platform=Platform("waha"), chat_id="6285157813352@s.whatsapp.net",
                           chat_type="dm", user_id="6285157813352@s.whatsapp.net")

    store._adopt_legacy_whatsapp_entry(source, "agent:main:waha:dm:6285157813352")

    # The live canonical session is untouched; the legacy entry is left alone.
    assert store._entries["agent:main:waha:dm:6285157813352"].session_id == "sess-current"
    assert "agent:main:waha:dm:6285157813352@c.us" in store._entries
    assert store.saved == 0


def test_adoption_ignores_non_whatsapp_and_group_sources():
    store = _Store({})
    tg = SessionSource(platform=Platform.TELEGRAM, chat_id="12345@c.us", chat_type="dm", user_id="12345")
    grp = SessionSource(platform=Platform("waha"), chat_id="1203634@g.us", chat_type="group", user_id="1")
    store._adopt_legacy_whatsapp_entry(tg, "agent:main:telegram:dm:12345@c.us")
    store._adopt_legacy_whatsapp_entry(grp, "agent:main:waha:group:1203634@g.us")
    assert store._entries == {} and store.saved == 0
