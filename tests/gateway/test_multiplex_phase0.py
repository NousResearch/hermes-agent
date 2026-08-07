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
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from gateway.config import GatewayConfig, Platform
from gateway.session import (
    MultiplexSessionCollisionError,
    SessionEntry,
    SessionSource,
    SessionStore,
    build_session_key,
)


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


class _ExactOrPeerRecoveringDB:
    """Model SessionDB's exact-key lookup followed by peer fallback."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []
        self.reopened = []

    def find_latest_gateway_session_for_peer(self, **kwargs):
        self.calls.append(kwargs)
        exact = self.rows.get(kwargs["session_key"])
        if exact is not None:
            return exact
        if kwargs.get("chat_id") is not None:
            return next(iter(self.rows.values()), None)
        return None

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


class TestSessionStoreMultiplexRecovery:
    def _store_with_rows(self, tmp_path, rows):
        config = GatewayConfig(multiplex_profiles=True)
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = _ExactOrPeerRecoveringDB(rows)
        store._loaded = True
        return store

    @pytest.mark.parametrize(
        "recover",
        [
            lambda store, key, source, now: store._recover_session_from_db(
                session_key=key,
                source=source,
                now=now,
            ),
            lambda store, key, source, now: store._query_recoverable_session(
                session_key=key,
                source=source,
                now=now,
            ),
        ],
    )
    def test_missing_jarvis_key_never_adopts_default_peer_session(
        self, tmp_path, recover
    ):
        rows = {
            "agent:main:telegram:dm:99": {
                "id": "sess-black-panther",
                "started_at": 1700000000,
                "session_key": "agent:main:telegram:dm:99",
            }
        }
        store = self._store_with_rows(tmp_path, rows)
        source = _src(
            chat_id="99",
            user_id="same-owner",
            profile="jarvis",
        )

        recovered = recover(
            store,
            "agent:jarvis:telegram:dm:99",
            source,
            datetime.fromtimestamp(1700000001),
        )

        assert recovered is None
        assert store._db.calls[-1]["chat_id"] is None
        assert store._db.reopened == []

    @pytest.mark.parametrize(
        "recover",
        [
            lambda store, key, source, now: store._recover_session_from_db(
                session_key=key,
                source=source,
                now=now,
            ),
            lambda store, key, source, now: store._query_recoverable_session(
                session_key=key,
                source=source,
                now=now,
            ),
        ],
    )
    def test_exact_profile_key_recovery_still_works(self, tmp_path, recover):
        key = "agent:jarvis:telegram:dm:99"
        rows = {
            key: {
                "id": "sess-jarvis",
                "started_at": 1700000000,
                "session_key": key,
            }
        }
        store = self._store_with_rows(tmp_path, rows)
        source = _src(chat_id="99", user_id="same-owner", profile="jarvis")

        recovered = recover(
            store,
            key,
            source,
            datetime.fromtimestamp(1700000001),
        )

        assert recovered is not None
        assert recovered.session_id == "sess-jarvis"
        assert recovered.session_key == key
        assert store._db.calls[-1]["chat_id"] is None
        assert store._db.reopened == ["sess-jarvis"]


class TestMultiplexSessionCollisionInvariant:
    def _store(self, tmp_path, entries):
        config = GatewayConfig(multiplex_profiles=True)
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        store._loaded = True
        store._entries = entries
        return store

    @staticmethod
    def _entry(key, session_id):
        now = datetime.fromtimestamp(1700000000)
        return SessionEntry(
            session_key=key,
            session_id=session_id,
            created_at=now,
            updated_at=now,
        )

    def test_cross_profile_session_id_blocks_startup(self, tmp_path):
        main_key = "agent:main:telegram:dm:99"
        jarvis_key = "agent:jarvis:telegram:dm:99"
        store = self._store(
            tmp_path,
            {
                main_key: self._entry(main_key, "shared-session"),
                jarvis_key: self._entry(jarvis_key, "shared-session"),
            },
        )

        with pytest.raises(MultiplexSessionCollisionError) as exc:
            store.assert_no_cross_profile_session_aliases()

        assert "shared-session" not in str(exc.value)
        assert main_key not in str(exc.value)
        assert jarvis_key not in str(exc.value)

    def test_same_profile_aliases_are_allowed(self, tmp_path):
        first = "agent:jarvis:telegram:dm:99"
        second = "agent:jarvis:telegram:dm:99:topic"
        store = self._store(
            tmp_path,
            {
                first: self._entry(first, "same-profile-session"),
                second: self._entry(second, "same-profile-session"),
            },
        )

        store.assert_no_cross_profile_session_aliases()

    @pytest.mark.asyncio
    async def test_gateway_aborts_before_connecting_adapters(self, tmp_path):
        from gateway.run import GATEWAY_FATAL_CONFIG_EXIT_CODE, GatewayRunner

        home_token = set_hermes_home_override(tmp_path)
        try:
            runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
            collision_check = AsyncMock(
                side_effect=MultiplexSessionCollisionError(
                    "multiplex routing contains 1 active cross-profile "
                    "session collision(s)"
                )
            )
            runner._async_session_store = SimpleNamespace(
                _store=runner.session_store,
                assert_no_cross_profile_session_aliases=collision_check,
            )
            runner._abort_startup_if_shutdown_requested = AsyncMock(
                return_value=False
            )
            runner._start_loop_liveness_guards = Mock()
            runner._request_clean_exit = Mock()
            runner._create_adapter = Mock()

            with (
                patch("gateway.run.faulthandler.enable"),
                patch("gateway.status.write_runtime_status"),
                patch(
                    "agent.monitoring.gateway_health_export."
                    "start_gateway_health_export",
                    return_value=SimpleNamespace(enabled=False),
                ),
                patch(
                    "hermes_cli.security_advisories.detect_compromised",
                    return_value=[],
                ),
            ):
                result = await runner.start()

            assert result is True
            collision_check.assert_awaited_once()
            runner._create_adapter.assert_not_called()
            runner._request_clean_exit.assert_called_once()
            assert runner._exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
        finally:
            reset_hermes_home_override(home_token)

    def test_invariant_is_inert_outside_multiplex(self, tmp_path):
        store = self._store(tmp_path, {})
        store.config.multiplex_profiles = False
        first = "agent:main:telegram:dm:99"
        second = "agent:jarvis:telegram:dm:99"
        store._entries = {
            first: self._entry(first, "legacy-shared-session"),
            second: self._entry(second, "legacy-shared-session"),
        }

        store.assert_no_cross_profile_session_aliases()
