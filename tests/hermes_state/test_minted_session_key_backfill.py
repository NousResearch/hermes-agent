"""The startup backfill gives minted-id rows the session_key their minted shape asserts.

A gateway restart mints row ids in the server key shape (``20261106_120000_ab12cd``,
``_MINTED_SESSION_KEY_RE``) but some rows never got their ``session_key`` column written, and
the canonical authority resolves sessions by that column: the lookup answers ``not_found`` and
the mount ladder has no route to rebuild a lane from, so webapp/TUI resume of the row fails
forever (#149905). The startup backfill stamps ``session_key = id`` — a record being restored,
not a guess — and the row becomes mountable like any keyed messaging row.

Regression for #149905 (v2: startup backfill beside the existing owner/created_source family,
after v1's legacy-resume heal landed on a path canonical deployments never execute).
"""
import json

import pytest

from hermes_state import SessionDB

# The minted shape: YYYYMMDD_HHMMSS_<6 lowercase hex> (tui_gateway/session_registry.py).
MINTED = "20261106_120000_ab12cd"
MINTED_CLI = "20261106_120001_ab12ce"
MINTED_KANBAN = "20261106_120002_ab12cf"
MINTED_CRON = "20261106_120003_ab12d0"
MINTED_KEYED = "20261106_120004_ab12d1"
MINTED_BLANK = "20261106_120005_ab12d2"

TELEGRAM_ORIGIN = json.dumps({"platform": "telegram", "chat_id": "424242", "chat_name": "DM",
                              "chat_type": "dm", "user_id": "1001", "user_name": "tester"})


def _session_key(db, sid):
    return db.get_session(sid)["session_key"]


def _set_key(db, sid, value):
    with db._lock:
        db._conn.execute("UPDATE sessions SET session_key=? WHERE id=?", (value, sid))
        db._conn.commit()


def test_startup_backfill_stamps_a_minted_row_and_mounts_it(tmp_path):
    """A telegram row with the minted id shape and a NULL session_key is unresumable by the
    canonical authority: the authoritative lookup keys on session_key and ``mountable`` has no
    route to rebuild the lane from. The startup backfill stamps ``session_key = id`` and the row
    mounts like any keyed messaging row — before any lane inbound has to run first."""
    from types import SimpleNamespace

    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority, backfill_null_minted_session_keys
    from gateway.session_route_recovery import mountable
    import hermes_state_runtime as rt

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = store._db
    db.create_session(MINTED, source="telegram")
    with db._lock:
        db._conn.execute("UPDATE sessions SET origin_json=? WHERE id=?", (TELEGRAM_ORIGIN, MINTED))
        db._conn.commit()

    async def handle(event):
        return "ok"

    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _handle_message=handle,
                             _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ("frozen", {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id="owned", instance_id="owner", db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id="owner"))

    assert mountable(authority, MINTED) is False  # the bug: a NULL key gives the mount nothing
    assert backfill_null_minted_session_keys(db) == 1
    assert _session_key(db, MINTED) == MINTED
    assert mountable(authority, MINTED) is True


def test_startup_backfill_keeps_the_adoption_and_keyless_contracts(tmp_path):
    """The stamp is a repair for minted rows only, and the exclusion contracts hold: legacy local
    rows keep adoption-by-resume (a session_key would make the legacy import refuse them), the
    automation labels keep the keyless convention, a row that already holds a key is never
    touched, a blank-string key is still a never-persisted one, and a second start finds nothing
    left to stamp."""
    from gateway.session_authority import backfill_null_minted_session_keys

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(MINTED_CLI, source="cli")
        db.create_session(MINTED_KANBAN, source="kanban")
        db.create_session(MINTED_CRON, source="cron")
        db.create_session(MINTED_KEYED, source="telegram")
        db.create_session(MINTED_BLANK, source="telegram")
        _set_key(db, MINTED_KEYED, "agent:main:xyz")
        _set_key(db, MINTED_BLANK, "")

        assert backfill_null_minted_session_keys(db) == 1
        assert _session_key(db, MINTED_CLI) is None  # adoption-by-resume contract preserved
        assert _session_key(db, MINTED_KANBAN) is None  # keyless convention preserved
        assert _session_key(db, MINTED_CRON) is None
        assert _session_key(db, MINTED_KEYED) == "agent:main:xyz"  # an explicit key always wins
        assert _session_key(db, MINTED_BLANK) == MINTED_BLANK  # the blank arm is part of the bug
        assert backfill_null_minted_session_keys(db) == 0  # idempotent: a second start is a no-op
    finally:
        db.close()
