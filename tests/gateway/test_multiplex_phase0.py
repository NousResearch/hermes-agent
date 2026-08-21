"""Phase 0 foundations for multi-profile gateway multiplexing.

Covers the three Phase 0 deliverables:
  1. ``gateway.multiplex_profiles`` config flag (default False, round-trips).
  2. ``hermes_cli.profiles.profiles_to_serve`` enumeration.
  3. Profile-stamped ``build_session_key`` that is BYTE-IDENTICAL when the
     flag is off (the orphan-every-session guard) and namespace-segmented when
     on, without disturbing the positional key layout downstream parsers rely
     on.
"""
import pytest
from datetime import datetime
from unittest.mock import patch
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_state import SessionDB


def _src(**kw) -> SessionSource:
    kw.setdefault("platform", Platform.TELEGRAM)
    kw.setdefault("chat_id", "99")
    kw.setdefault("chat_type", "dm")
    return SessionSource(**kw)


class TestSessionKeyByteIdenticalWhenOff:
    """The non-negotiable guard: with no profile (or 'default'), every key is
    byte-for-byte what it was before Phase 0. A diff here orphans every
    existing session on upgrade."""

    @pytest.mark.parametrize("profile", [None, "default"])
    def test_dm_with_chat_id(self, profile):
        s = _src(chat_id="99", chat_type="dm")
        assert build_session_key(s, profile=profile) == "agent:main:telegram:dm:99"


    @pytest.mark.parametrize("profile", [None, "default"])
    def test_group_per_user(self, profile):
        s = _src(platform=Platform.DISCORD, chat_id="g1", chat_type="group", user_id="alice")
        assert (
            build_session_key(s, profile=profile)
            == "agent:main:discord:group:g1:alice"
        )


class TestSessionKeyNamespacedWhenOn:
    """A named profile occupies the namespace slot, isolating its sessions."""


    def test_named_profile_group_per_user(self):
        s = _src(platform=Platform.DISCORD, chat_id="g1", chat_type="group", user_id="alice")
        assert (
            build_session_key(s, profile="coder")
            == "agent:coder:discord:group:g1:alice"
        )

    def test_two_profiles_same_chat_do_not_collide(self):
        s = _src(chat_id="99", chat_type="dm")
        a = build_session_key(s, profile="default")
        b = build_session_key(s, profile="coder")
        c = build_session_key(s, profile="writer")
        assert a != b != c and a != c


class TestMultiplexConfigFlag:
    """gateway.multiplex_profiles defaults off and round-trips."""

    def test_default_is_false(self):
        assert GatewayConfig().multiplex_profiles is False


    def test_from_dict_top_level(self):
        cfg = GatewayConfig.from_dict({"multiplex_profiles": True})
        assert cfg.multiplex_profiles is True

    def test_profile_allowlist_defaults_to_serve_all(self):
        assert GatewayConfig().multiplex_profile_allowlist is None

    def test_profile_allowlist_normalizes_and_round_trips(self):
        cfg = GatewayConfig.from_dict(
            {
                "gateway": {
                    "multiplex_profiles": True,
                    "multiplex_profile_allowlist": [
                        " Worker ",
                        "worker",
                        "Guest",
                        "default",
                        "bad/name",
                        7,
                    ],
                }
            }
        )

        assert cfg.multiplex_profile_allowlist == ["worker", "guest"]
        restored = GatewayConfig.from_dict(cfg.to_dict())
        assert restored.multiplex_profile_allowlist == ["worker", "guest"]

    def test_invalid_profile_allowlist_fails_safe_to_default_only(self, caplog):
        with caplog.at_level("WARNING", logger="gateway.config"):
            cfg = GatewayConfig.from_dict(
                {"gateway": {"multiplex_profile_allowlist": "worker"}}
            )

        assert cfg.multiplex_profile_allowlist == []
        assert "serving only the default profile" in caplog.text


class TestSessionStoreProfileResolution:
    """SessionStore._generate_session_key honors the flag: legacy namespace
    when off, active-profile namespace when on."""

    def _store(self, tmp_path, **cfg_kw):
        config = GatewayConfig(**cfg_kw)
        with patch("gateway.session.SessionStore._ensure_loaded"):
            s = SessionStore(sessions_dir=tmp_path, config=config)
        s._db = None
        s._loaded = True
        return s

    def test_flag_off_uses_legacy_namespace(self, tmp_path):
        store = self._store(tmp_path)  # multiplex_profiles defaults False
        s = _src(chat_id="99", chat_type="dm")
        assert store._generate_session_key(s) == "agent:main:telegram:dm:99"
        assert store._generate_session_key(s) == build_session_key(s)


class _RecoveringDB:
    def __init__(self, row):
        self.row = row
        self.reopened = []

    def find_latest_gateway_session_for_peer(self, **_kwargs):
        return self.row

    def reopen_session(self, session_id):
        self.reopened.append(session_id)


class TestSessionStoreUnmultiplexedRecovery:
    """Turning multiplexing off must not recover another profile's session."""

    def _store_with_row(self, tmp_path, row, **cfg_kw):
        config = GatewayConfig(**cfg_kw)
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = _RecoveringDB(row)
        store._loaded = True
        return store


    def test_flag_off_allows_active_profile_peer_fallback(self, tmp_path):
        row = {
            "id": "sess-coder",
            "started_at": 1700000000,
            "session_key": "agent:coder:telegram:dm:99",
        }
        store = self._store_with_row(tmp_path, row)
        source = _src(chat_id="99", chat_type="dm")

        with patch("hermes_cli.profiles.get_active_profile_name", return_value="coder"):
            recovered = store._recover_session_from_db(
                session_key="agent:main:telegram:dm:99",
                source=source,
                now=datetime.fromtimestamp(1700000001),
            )

        assert recovered is not None
        assert recovered.session_id == "sess-coder"
        assert recovered.session_key == "agent:main:telegram:dm:99"
        assert store._db.reopened == ["sess-coder"]


# ── Cross-profile recovery inside one state.db ───────────────────────────────
# Peer tuple shared by both namespaces below: one Telegram DM, one owner.
_PEER = dict(source="telegram", user_id="owner-1", chat_id="99", chat_type="dm")
_MAIN_KEY = "agent:main:telegram:dm:99"
_JARVIS_KEY = "agent:jarvis:telegram:dm:99"
_STARTED_AT = 1700000000.0
_RESET_AT = 1700000060.0
_ACTIVE_AT = 1700000120.0


def _seed_gateway_session(
    db, session_id, session_key, *, last_activity_at, reset=False
):
    """Write one keyed, message-bearing gateway session row."""
    db.create_session(
        session_id,
        _PEER["source"],
        user_id=_PEER["user_id"],
        session_key=session_key,
        chat_id=_PEER["chat_id"],
        chat_type=_PEER["chat_type"],
    )
    db.append_message(session_id, "user", "hello")
    db.append_message(session_id, "assistant", "hi")
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET started_at=?, last_activity_at=?, "
            "ended_at=?, end_reason=? WHERE id=?",
            (
                _STARTED_AT,
                last_activity_at,
                _RESET_AT if reset else None,
                "session_reset" if reset else None,
                session_id,
            ),
        )
        db._conn.commit()


def _shared_profile_db(db_path, *, reset_key=None):
    """One state.db holding BOTH profile namespaces for the same peer.

    ``reset_key`` ends that namespace's row on a reset boundary, which is what
    a plain ``/new`` leaves behind: the profile's own key stops resolving while
    the *other* profile's live row keeps sitting in the same file.
    """
    db = SessionDB(db_path=db_path)
    for key, session_id in (
        (_MAIN_KEY, "sess-main"),
        (_JARVIS_KEY, "sess-jarvis"),
    ):
        reset = key == reset_key
        _seed_gateway_session(
            db,
            session_id,
            key,
            last_activity_at=_RESET_AT if reset else _ACTIVE_AT,
            reset=reset,
        )
    return db


class TestMultiplexedRecoveryStaysInsideProfile:
    """Recovery must not cross profiles inside one shared state.db.

    #88734 gave each profile its own state.db *file* and #89860 minted session
    keys per bot, but neither reaches the recovery *query*: whenever two
    namespaces still resolve one store — an un-migrated root store, or a
    default-profile handler pinned to the root scope while profile-stamped
    keys land in the same file — ``find_latest_gateway_session_for_peer``'s
    fallback matches on (source, user_id, chat_id, chat_type, thread_id) with
    no profile predicate and hands one profile the other's live session. The
    downstream profile guard is inert here: it returns early once
    ``multiplex_profiles`` is on.
    """

    def _store(self, tmp_path, db):
        config = GatewayConfig(multiplex_profiles=True)
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
        # Pin the one handle both namespaces resolve to (the precondition).
        store._db = db
        store._loaded = True
        return store

    def test_one_state_db_holds_both_profile_namespaces(self, tmp_path):
        """Precondition, asserted rather than assumed."""
        db = _shared_profile_db(tmp_path / "state.db")
        try:
            with db._lock:
                rows = db._conn.execute(
                    "SELECT session_key, source, user_id, chat_id, chat_type, "
                    "thread_id FROM sessions ORDER BY session_key"
                ).fetchall()
        finally:
            db.close()

        keys = [r["session_key"] for r in rows]
        assert keys == [_JARVIS_KEY, _MAIN_KEY]
        peers = {
            (r["source"], r["user_id"], r["chat_id"], r["chat_type"], r["thread_id"])
            for r in rows
        }
        assert peers == {("telegram", "owner-1", "99", "dm", None)}

    @pytest.mark.parametrize(
        "recover",
        [
            lambda store, key, source, now: store._recover_session_from_db(
                session_key=key, source=source, now=now
            ),
            lambda store, key, source, now: store._query_recoverable_session(
                session_key=key, source=source, now=now
            ),
        ],
        ids=["recover_session_from_db", "query_recoverable_session"],
    )
    @pytest.mark.parametrize(
        "requested_key,profile",
        [
            (_JARVIS_KEY, "jarvis"),
            (_MAIN_KEY, "default"),
        ],
        ids=["jarvis_requests", "default_requests"],
    )
    def test_reset_profile_never_adopts_the_other_profiles_session(
        self, tmp_path, recover, requested_key, profile
    ):
        db = _shared_profile_db(tmp_path / "state.db", reset_key=requested_key)
        try:
            store = self._store(tmp_path, db)
            source = _src(user_id=_PEER["user_id"], profile=profile)

            recovered = recover(
                store,
                requested_key,
                source,
                datetime.fromtimestamp(_ACTIVE_AT + 1),
            )

            assert recovered is None, (
                f"{requested_key} recovered {getattr(recovered, 'session_id', None)}"
            )
        finally:
            db.close()

    @pytest.mark.parametrize(
        "requested_key,profile,own_id,foreign_id",
        [
            (_JARVIS_KEY, "jarvis", "sess-jarvis", "sess-main"),
            (_MAIN_KEY, "default", "sess-main", "sess-jarvis"),
        ],
        ids=["jarvis_requests", "default_requests"],
    )
    def test_exact_profile_key_still_recovers_its_own_session(
        self, tmp_path, requested_key, profile, own_id, foreign_id
    ):
        """Guardrail: scoping the fallback must not disable exact recovery."""
        db = _shared_profile_db(tmp_path / "state.db")
        try:
            store = self._store(tmp_path, db)
            source = _src(user_id=_PEER["user_id"], profile=profile)

            recovered = store._recover_session_from_db(
                session_key=requested_key,
                source=source,
                now=datetime.fromtimestamp(_ACTIVE_AT + 1),
            )

            assert recovered is not None
            assert recovered.session_id == own_id
            assert recovered.session_id != foreign_id
            assert recovered.session_key == requested_key
        finally:
            db.close()
