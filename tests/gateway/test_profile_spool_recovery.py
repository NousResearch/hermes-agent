"""Boot recovery must replay a routed profile's spooled transcript backlog (#123584).

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

import agent.secret_scope as ss
from gateway.config import GatewayConfig
from gateway.platforms.base import Platform, SessionSource
from gateway.session import SessionEntry, SessionStore
from gateway.shutdown_flush import recover_gateway_pending, spool_dropped_transcript_message
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


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
    ss.set_multiplex_active(True)
    yield root, profile
    ss.set_multiplex_active(False)


def _session(store: SessionStore, key: str, sid: str):
    now = datetime.now()
    store._entries[key] = SessionEntry(
        session_key=key, session_id=sid, created_at=now, updated_at=now,
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42"),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    db = store._db_for_key(key)
    db.create_session(sid, source="telegram", session_key=key)
    return db


def _spool_under(home, sid: str, text: str) -> None:
    token = set_hermes_home_override(str(home))
    try:
        assert spool_dropped_transcript_message(sid, {"role": "user", "content": text})
    finally:
        reset_hermes_home_override(token)
    assert list((home / "pending_messages").glob("*.json"))


def test_boot_recovery_replays_a_routed_profiles_spooled_backlog(multiplex_homes):
    root, profile = multiplex_homes
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=root / "sessions", config=GatewayConfig(multiplex_profiles=True))
    store._loaded = True

    default_sid, work_sid = "20260926_000000_default0", "20260926_000000_work0000"
    default_db = _session(store, "agent:main:telegram:dm:42", default_sid)
    work_db = _session(store, "agent:work:telegram:dm:42", work_sid)
    # Each routed turn spools under its own profile home while that store is unwritable.
    _spool_under(root, default_sid, "default while locked")
    _spool_under(profile, work_sid, "work while locked")

    recovered = recover_gateway_pending(SimpleNamespace(config=store.config, session_store=store))

    assert recovered == 2
    assert [m["content"] for m in default_db.get_messages(default_sid)] == ["default while locked"]
    assert [m["content"] for m in work_db.get_messages(work_sid)] == ["work while locked"]
    assert not list(root.glob("**/pending_messages/*.json"))
    assert get_hermes_home() == root  # A→B→A: the launch scope is restored after the routed pass
