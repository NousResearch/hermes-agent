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
