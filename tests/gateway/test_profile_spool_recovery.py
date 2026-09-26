"""Boot recovery must replay a routed profile's spooled transcript backlog.

``_get_flush_dir`` follows the active HERMES_HOME, and a routed turn on a multiplexed gateway runs
inside its profile's scope, so a transcript backlog spooled while that profile's store is unwritable
lands in ``profiles/<name>/pending_messages/``. Boot recovery only scanned the launch home, and the
runtime drain keys on an in-memory set that a restart empties: after a restart nothing read those
files again, and the messages never reached state.db.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.base import Platform, SessionSource
from gateway.session import SessionEntry, SessionStore
from gateway.shutdown_flush import spool_dropped_transcript_message
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def multiplex_homes(tmp_path, monkeypatch):
    """A launch home plus a named ``work`` profile, as in test_multiplex_session_db_profile_scope."""
    import hermes_state

    root = tmp_path / "hermes"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")  # identity marker
    monkeypatch.setenv("HERMES_HOME", str(root))
    # Resolve state.db through get_hermes_home(), as production does (see the sibling suite).
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    return root, profile


def test_boot_recovery_replays_a_routed_profiles_spooled_backlog(multiplex_homes):
    from gateway.run import _recover_pending_flushes

    root, profile = multiplex_homes
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=root / "sessions", config=GatewayConfig(multiplex_profiles=True))
    store._loaded = True

    key, sid = "agent:work:telegram:dm:42", "20260926_000000_abcd1234"
    now = datetime.now()
    store._entries[key] = SessionEntry(
        session_key=key, session_id=sid, created_at=now, updated_at=now,
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42"),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    db = store._db_for_key(key)
    db.create_session(sid, source="telegram", session_key=key)

    # The routed turn spools under its own profile home while the store is unwritable.
    token = set_hermes_home_override(str(profile))
    try:
        assert spool_dropped_transcript_message(sid, {"role": "user", "content": "sent while locked"})
    finally:
        reset_hermes_home_override(token)
    assert list((profile / "pending_messages").glob("*.json"))

    recovered = _recover_pending_flushes(SimpleNamespace(config=store.config, session_store=store))

    assert recovered == 1
    assert [m["content"] for m in db.get_messages(sid)] == ["sent while locked"]
    assert not list((profile / "pending_messages").glob("*.json"))
