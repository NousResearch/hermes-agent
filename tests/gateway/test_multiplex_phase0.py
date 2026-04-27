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
from unittest.mock import AsyncMock, patch
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
    def test_dm_with_thread(self, profile):
        s = _src(chat_id="99", chat_type="dm", thread_id="t1")
        assert build_session_key(s, profile=profile) == "agent:main:telegram:dm:99:t1"

    @pytest.mark.parametrize("profile", [None, "default"])
    def test_dm_without_chat_id_falls_back_to_user(self, profile):
        s = _src(chat_id="", chat_type="dm", user_id="jordan")
        assert build_session_key(s, profile=profile) == "agent:main:telegram:dm:jordan"

    @pytest.mark.parametrize("profile", [None, "default"])
    def test_group_per_user(self, profile):
        s = _src(platform=Platform.DISCORD, chat_id="g1", chat_type="group", user_id="alice")
        assert (
            build_session_key(s, profile=profile)
            == "agent:main:discord:group:g1:alice"
        )

    @pytest.mark.parametrize("profile", [None, "default"])
    def test_group_shared_when_disabled(self, profile):
        s = _src(platform=Platform.DISCORD, chat_id="g1", chat_type="group", user_id="alice")
        assert (
            build_session_key(s, group_sessions_per_user=False, profile=profile)
            == "agent:main:discord:group:g1"
        )


class TestSessionKeyNamespacedWhenOn:
    """A named profile occupies the namespace slot, isolating its sessions."""

    def test_named_profile_dm(self):
        s = _src(chat_id="99", chat_type="dm")
        assert build_session_key(s, profile="coder") == "agent:coder:telegram:dm:99"

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

    def test_positional_layout_preserved_for_parsers(self):
        """Downstream parsers split on ':' and read parts[2]=platform,
        parts[3]=chat_type, parts[4]=chat_id (see qqbot adapter
        _parse_gateway_session_key). The profile must occupy parts[1] only."""
        s = _src(platform=Platform.DISCORD, chat_id="g1", chat_type="group", user_id="alice")
        parts = build_session_key(s, profile="coder").split(":")
        assert parts[0] == "agent"
        assert parts[1] == "coder"  # namespace slot (was always 'main')
        assert parts[2] == "discord"  # platform — unchanged offset
        assert parts[3] == "group"  # chat_type — unchanged offset
        assert parts[4] == "g1"  # chat_id — unchanged offset

    def test_default_namespace_layout_matches_named(self):
        """Default and named keys differ ONLY in parts[1]."""
        s = _src(platform=Platform.SLACK, chat_id="c1", chat_type="channel", user_id="u1")
        d = build_session_key(s, profile="default").split(":")
        n = build_session_key(s, profile="coder").split(":")
        assert d[0] == n[0] == "agent"
        assert d[1] == "main" and n[1] == "coder"
        assert d[2:] == n[2:]  # everything after the namespace is identical

    @pytest.mark.asyncio
    async def test_adapter_internal_session_resolver_binds_profile(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        entry = object()
        runner._get_or_create_session_at_process_delivery_boundary = AsyncMock(
            return_value=entry
        )
        resolver = runner._make_profile_session_resolver("coder")
        source = _src(chat_id="99", chat_type="dm")

        assert await resolver(source) is entry
        assert source.profile == "coder"
        resolved_source = (
            runner._get_or_create_session_at_process_delivery_boundary.await_args.args[0]
        )
        assert build_session_key(
            resolved_source, profile=resolved_source.profile
        ) == "agent:coder:telegram:dm:99"


class TestMultiplexConfigFlag:
    """gateway.multiplex_profiles defaults off and round-trips."""

    def test_default_is_false(self):
        assert GatewayConfig().multiplex_profiles is False

    def test_to_dict_includes_flag(self):
        assert GatewayConfig().to_dict()["multiplex_profiles"] is False

    def test_from_dict_top_level(self):
        cfg = GatewayConfig.from_dict({"multiplex_profiles": True})
        assert cfg.multiplex_profiles is True

    def test_from_dict_nested_gateway(self):
        cfg = GatewayConfig.from_dict({"gateway": {"multiplex_profiles": True}})
        assert cfg.multiplex_profiles is True

    def test_from_dict_coerces_truthy_string(self):
        cfg = GatewayConfig.from_dict({"multiplex_profiles": "true"})
        assert cfg.multiplex_profiles is True

    def test_roundtrip(self):
        cfg = GatewayConfig.from_dict(GatewayConfig(multiplex_profiles=True).to_dict())
        assert cfg.multiplex_profiles is True

    def test_gateway_config_loader_honors_profile_runtime_scope(self, tmp_path, monkeypatch):
        """Multiplexed turns must resolve display settings from the routed profile."""
        import gateway.run as gateway_run

        root_home = tmp_path / "root"
        profile_home = tmp_path / "profiles" / "quiet"
        root_home.mkdir(parents=True)
        profile_home.mkdir(parents=True)

        (root_home / "config.yaml").write_text(
            yaml.safe_dump(
                {"display": {"tool_progress": "all", "interim_assistant_messages": True}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (profile_home / "config.yaml").write_text(
            yaml.safe_dump(
                {"display": {"tool_progress": False, "interim_assistant_messages": False}},
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", root_home)

        assert gateway_run._load_gateway_config()["display"]["tool_progress"] == "all"

        token = set_hermes_home_override(profile_home)
        try:
            scoped_config = gateway_run._load_gateway_config()
        finally:
            reset_hermes_home_override(token)

        assert scoped_config["display"]["tool_progress"] is False
        assert scoped_config["display"]["interim_assistant_messages"] is False


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

    def test_flag_off_resolve_profile_is_none(self, tmp_path):
        store = self._store(tmp_path)
        assert store._resolve_profile_for_key() is None

    def test_flag_on_uses_active_profile_namespace(self, tmp_path):
        store = self._store(tmp_path, multiplex_profiles=True)
        s = _src(chat_id="99", chat_type="dm")
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="coder"):
            assert store._generate_session_key(s) == "agent:coder:telegram:dm:99"

    def test_flag_on_default_profile_stays_legacy(self, tmp_path):
        store = self._store(tmp_path, multiplex_profiles=True)
        s = _src(chat_id="99", chat_type="dm")
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            assert store._generate_session_key(s) == "agent:main:telegram:dm:99"


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

    def test_flag_off_rejects_other_profile_peer_fallback(self, tmp_path):
        row = {
            "id": "sess-coder",
            "started_at": 1700000000,
            "session_key": "agent:coder:telegram:dm:99",
        }
        store = self._store_with_row(tmp_path, row)
        source = _src(chat_id="99", chat_type="dm")

        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            recovered = store._recover_session_from_db(
                session_key="agent:main:telegram:dm:99",
                source=source,
                now=datetime.fromtimestamp(1700000001),
            )

        assert recovered is None
        assert store._db.reopened == []

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
