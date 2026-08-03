"""A multiplexed gateway must not revive a sibling profile's session (#74285).

``SessionDB.find_latest_gateway_session_for_peer`` falls back to matching only
the peer tuple ``(source, user_id, chat_id, chat_type, thread_id)`` when the
exact ``session_key`` is missing. For a Telegram DM that tuple is
byte-identical across profiles — ``chat_id == user_id`` and ``thread_id IS
NULL`` for every bot — so the fallback can hand back a row owned by a sibling
profile.

``SessionStore._recovered_row_allowed_for_active_profile`` exists to stop
exactly that, but it returned early whenever ``multiplex_profiles`` was
enabled: precisely the configuration in which several profiles each own a bot
token and can serve the same allowlisted user. The sibling profile was then
actually executed — its persona, tools, credentials and filesystem scope — so
this is a privilege boundary, not a session-list display detail.

The requested key already carries the routed profile, so the multiplexed case
has the information it needs to compare profiles rather than wave the row
through. Two consequences the guard alone does not cover:

* the fallback query has to be namespaced too. It orders every profile's rows
  together under one ``LIMIT 1``, so a sibling that spoke more recently is the
  only candidate the guard ever sees — our own older row is never offered, and
  recovery ends in a needless fresh session.
* a row whose key names no profile cannot be adopted while multiplexing. The
  peer tuple has no profile discriminator, so nothing distinguishes it from a
  sibling's row. A single-profile gateway still adopts it — there is only one
  claimant, and pre-namespace rows have to stay recoverable.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionSource, SessionStore


def _db() -> MagicMock:
    """SessionDB mock: no routing state, recovery finds nothing by default."""
    db = MagicMock()
    db.get_session.return_value = None
    db.find_latest_gateway_session_for_peer.return_value = None
    db.reopen_session.return_value = None
    db.create_session.return_value = None
    # Mirror the real get_compression_tip identity for uncompressed sessions;
    # a bare Mock would be assigned as the session_id by the routing heal.
    db.get_compression_tip.side_effect = lambda sid: sid
    return db


def _store(tmp_path, db_mock: MagicMock, *, multiplex: bool) -> SessionStore:
    """Build a SessionStore with a mock SessionDB, bypassing disk load."""
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="none"),
        multiplex_profiles=multiplex,
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=config)
    store._db = db_mock
    store._loaded = True
    return store


def _dm_source(profile: Optional[str]) -> SessionSource:
    """A Telegram DM from one user — same peer tuple whichever bot received it."""
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8494508720",
        chat_type="dm",
        user_id="8494508720",
        profile=profile,
    )


def _row_owned_by(store: SessionStore, profile: Optional[str], session_id: str) -> dict:
    """A durable row the peer fallback would return, owned by ``profile``."""
    return {
        "id": session_id,
        "session_key": store._generate_session_key(_dm_source(profile)),
        "started_at": (datetime.now() - timedelta(hours=2)).timestamp(),
    }


class TestMultiplexedPeerFallbackProfileBoundary:
    """Multiplexing on: the routed profile in the requested key is authoritative."""

    def test_sibling_profile_row_is_rejected(self, tmp_path):
        """A row owned by ``admin`` must not be adopted for ``restricted``."""
        store = _store(tmp_path, _db(), multiplex=True)
        allowed = store._recovered_row_allowed_for_active_profile(
            requested_session_key=store._generate_session_key(_dm_source("restricted")),
            recovered=_row_owned_by(store, "admin", "sid_admin"),
        )
        assert allowed is False

    def test_same_profile_row_is_allowed(self, tmp_path):
        """Recovery within one profile keeps working."""
        store = _store(tmp_path, _db(), multiplex=True)
        allowed = store._recovered_row_allowed_for_active_profile(
            requested_session_key=store._generate_session_key(_dm_source("restricted")),
            recovered=_row_owned_by(store, "restricted", "sid_restricted"),
        )
        assert allowed is True

    def test_row_without_profile_namespace_is_rejected(self, tmp_path):
        """An unnamespaced row proves nothing, and here that has to be fatal.

        The peer tuple has no profile discriminator, so a row whose key names
        no profile is exactly as likely to belong to a sibling as to us.
        """
        store = _store(tmp_path, _db(), multiplex=True)
        allowed = store._recovered_row_allowed_for_active_profile(
            requested_session_key=store._generate_session_key(_dm_source("restricted")),
            recovered={"id": "sid_legacy", "session_key": "legacy-unnamespaced-key"},
        )
        assert allowed is False

    def test_row_without_any_session_key_is_rejected(self, tmp_path):
        """Same reasoning for a row that carries no key at all."""
        store = _store(tmp_path, _db(), multiplex=True)
        allowed = store._recovered_row_allowed_for_active_profile(
            requested_session_key=store._generate_session_key(_dm_source("restricted")),
            recovered={"id": "sid_keyless"},
        )
        assert allowed is False

    def test_recovery_does_not_reuse_sibling_profile_session_id(self, tmp_path):
        """End to end: a DM routed to one profile never lands on another's session."""
        db = _db()
        store = _store(tmp_path, db, multiplex=True)
        db.find_latest_gateway_session_for_peer.return_value = _row_owned_by(
            store, "admin", "sid_admin"
        )

        entry = store.get_or_create_session(_dm_source("restricted"))

        assert entry.session_id != "sid_admin"
        assert store._generate_session_key(
            _dm_source("restricted")
        ) != store._generate_session_key(_dm_source("admin"))


class TestPeerFallbackIsScopedToTheProfile:
    """The query must not hand back a sibling's row for the guard to reject.

    Rejecting is only half the fix: one ``LIMIT 1`` orders every profile's rows
    together, so a sibling that spoke more recently hides our own recoverable
    row and recovery ends in a needless fresh session.
    """

    def test_lookup_is_namespaced_when_multiplexing(self, tmp_path):
        db = _db()
        store = _store(tmp_path, db, multiplex=True)

        store.get_or_create_session(_dm_source("restricted"))

        kwargs = db.find_latest_gateway_session_for_peer.call_args.kwargs
        assert kwargs["session_key_prefix"] == "agent:restricted:"

    def test_lookup_is_unconstrained_without_multiplexing(self, tmp_path):
        """A single-profile gateway owns every row the peer tuple reaches."""
        db = _db()
        store = _store(tmp_path, db, multiplex=False)

        store.get_or_create_session(_dm_source(None))

        kwargs = db.find_latest_gateway_session_for_peer.call_args.kwargs
        assert "session_key_prefix" not in kwargs

    def test_older_same_profile_row_survives_a_newer_sibling(self, tmp_path):
        """Real SessionDB: the newest row is a sibling's, ours is older."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            peer = dict(
                source="telegram",
                user_id="8494508720",
                chat_id="8494508720",
                chat_type="dm",
                thread_id=None,
            )
            db.create_session(
                "sid_restricted",
                session_key="agent:restricted:telegram:dm:8494508720",
                **peer,
            )
            db.append_message("sid_restricted", "user", "mine")
            db.create_session(
                "sid_admin",
                session_key="agent:admin:telegram:dm:8494508720",
                **peer,
            )
            db.append_message("sid_admin", "user", "the sibling's")
            # The sibling spoke last — without the namespace filter it is the
            # only candidate the fallback ever considers.
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (1_700_000_000.0, "sid_restricted"),
            )
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (1_700_000_900.0, "sid_admin"),
            )
            db._conn.commit()

            unscoped = db.find_latest_gateway_session_for_peer(
                session_key="agent:restricted:telegram:dm:missing", **peer
            )
            scoped = db.find_latest_gateway_session_for_peer(
                session_key="agent:restricted:telegram:dm:missing",
                session_key_prefix="agent:restricted:",
                **peer,
            )

            assert unscoped["id"] == "sid_admin"
            assert scoped["id"] == "sid_restricted"
        finally:
            db.close()

    def test_prefix_matching_is_exact(self, tmp_path):
        """``restricted-2`` is a different profile, not a longer ``restricted``."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            peer = dict(
                source="telegram",
                user_id="8494508720",
                chat_id="8494508720",
                chat_type="dm",
                thread_id=None,
            )
            db.create_session(
                "sid_other",
                session_key="agent:restricted-2:telegram:dm:8494508720",
                **peer,
            )
            db.append_message("sid_other", "user", "hi")

            found = db.find_latest_gateway_session_for_peer(
                session_key="agent:restricted:telegram:dm:missing",
                session_key_prefix="agent:restricted:",
                **peer,
            )

            assert found is None
        finally:
            db.close()


class TestNonMultiplexedGuardUnchanged:
    """Multiplexing off: keep comparing against the process-wide active profile."""

    def test_unnamespaced_row_stays_recoverable(self, tmp_path):
        """Pre-namespace rows must not become unrecoverable for the one profile."""
        store = _store(tmp_path, _db(), multiplex=False)
        with patch.object(SessionStore, "_active_profile_name", staticmethod(lambda: "default")):
            allowed = store._recovered_row_allowed_for_active_profile(
                requested_session_key="agent:main:telegram:dm:8494508720",
                recovered={"id": "sid_legacy", "session_key": "legacy-unnamespaced-key"},
            )
        assert allowed is True

    def test_other_profile_row_still_rejected(self, tmp_path):
        store = _store(tmp_path, _db(), multiplex=False)
        with patch.object(SessionStore, "_active_profile_name", staticmethod(lambda: "default")):
            allowed = store._recovered_row_allowed_for_active_profile(
                requested_session_key="agent:main:telegram:dm:8494508720",
                recovered={
                    "id": "sid_other",
                    "session_key": "agent:coder:telegram:dm:8494508720",
                },
            )
        assert allowed is False

    def test_active_profile_row_allowed(self, tmp_path):
        store = _store(tmp_path, _db(), multiplex=False)
        with patch.object(SessionStore, "_active_profile_name", staticmethod(lambda: "coder")):
            allowed = store._recovered_row_allowed_for_active_profile(
                requested_session_key="agent:main:telegram:dm:8494508720",
                recovered={
                    "id": "sid_coder",
                    "session_key": "agent:coder:telegram:dm:8494508720",
                },
            )
        assert allowed is True
