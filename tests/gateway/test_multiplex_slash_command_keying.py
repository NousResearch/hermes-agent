"""Owner-routed /new must key ``agent:owner:*``, not process-default
``agent:main:*``, even when ``source.profile`` was never stamped.

``_handle_message`` stamps at ingress, but /new also runs from busy-path
and slash-confirm callbacks, and SessionStore falls back to
``get_active_profile_name()`` (the multiplexer default) when the stamp
is missing. That minted a customer session from an owner DM.
"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.profile_routing import ProfileRoute
from gateway.session import SessionEntry, SessionSource, SessionStore


OWNER_CHAT = "640466638"
OWNER_MAIN_KEY = f"agent:main:telegram:dm:{OWNER_CHAT}"
OWNER_OWNER_KEY = f"agent:owner:telegram:dm:{OWNER_CHAT}"


def _src(**kw) -> SessionSource:
    kw.setdefault("platform", Platform.TELEGRAM)
    kw.setdefault("chat_id", "99")
    kw.setdefault("chat_type", "dm")
    return SessionSource(**kw)


def _owner_route() -> ProfileRoute:
    return ProfileRoute(
        name="owner-dm",
        platform="telegram",
        profile="owner",
        chat_id=OWNER_CHAT,
    )


def _owner_source(*, profile=None) -> SessionSource:
    return _src(
        chat_id=OWNER_CHAT,
        user_id=OWNER_CHAT,
        user_name="owner",
        profile=profile,
    )


def _entry(session_key: str, session_id: str) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )


def _slash_key_runner(tmp_path, *, multiplex: bool, routes=None):
    """Bare GatewayRunner with a real SessionStore key generator."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        multiplex_profiles=multiplex,
        profile_routes=list(routes or []),
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=tmp_path, config=runner.config)
    store._db = None
    store._loaded = True
    store._save = lambda *a, **k: None
    runner.session_store = store
    runner.adapters = {Platform.TELEGRAM: MagicMock()}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._agent_cache = {}
    runner._reset_notice_session_info = lambda source: ""
    runner._telegram_topic_new_header = lambda source: None
    runner._is_telegram_topic_lane = lambda source: False
    return runner


def _served_homes(tmp_path):
    default_home = tmp_path / "default"
    owner_home = tmp_path / "profiles" / "owner"
    default_home.mkdir(parents=True, exist_ok=True)
    owner_home.mkdir(parents=True, exist_ok=True)
    return [("default", default_home), ("owner", owner_home)]


class TestMultiplexSlashCommandProfileKeying:
    def test_session_key_stamps_routed_profile_when_unset(self, tmp_path):
        runner = _slash_key_runner(
            tmp_path, multiplex=True, routes=[_owner_route()]
        )
        source = _owner_source(profile=None)

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=_served_homes(tmp_path),
        ), patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="default",
        ):
            key = runner._session_key_for_source(source)

        assert key == OWNER_OWNER_KEY
        assert source.profile == "owner"
        assert OWNER_MAIN_KEY not in (key,)

    def test_session_key_skips_reroute_when_already_stamped(self, tmp_path):
        runner = _slash_key_runner(
            tmp_path, multiplex=True, routes=[_owner_route()]
        )
        source = _owner_source(profile="owner")
        runner._profile_name_for_source = MagicMock(
            side_effect=AssertionError("must not re-match routes when stamped")
        )

        with patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="default",
        ):
            key = runner._session_key_for_source(source)

        assert key == OWNER_OWNER_KEY
        runner._profile_name_for_source.assert_not_called()

    def test_unmultiplexed_session_key_stays_legacy_main(self, tmp_path):
        runner = _slash_key_runner(
            tmp_path, multiplex=False, routes=[_owner_route()]
        )
        source = _owner_source(profile=None)

        with patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="owner",
        ):
            key = runner._session_key_for_source(source)

        assert key == OWNER_MAIN_KEY
        assert source.profile is None

    @pytest.mark.asyncio
    async def test_owner_new_resets_owner_key_not_main(self, tmp_path):
        runner = _slash_key_runner(
            tmp_path, multiplex=True, routes=[_owner_route()]
        )
        store = runner.session_store
        store._entries[OWNER_MAIN_KEY] = _entry(OWNER_MAIN_KEY, "sess-main")
        store._entries[OWNER_OWNER_KEY] = _entry(OWNER_OWNER_KEY, "sess-owner")
        main_before = store._entries[OWNER_MAIN_KEY].session_id
        owner_before = store._entries[OWNER_OWNER_KEY].session_id

        event = MessageEvent(text="/new", source=_owner_source(profile=None))

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=_served_homes(tmp_path),
        ), patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="default",
        ):
            await runner._handle_reset_command(event)

        assert store._entries[OWNER_MAIN_KEY].session_id == main_before
        assert store._entries[OWNER_OWNER_KEY].session_id != owner_before
        assert event.source.profile == "owner"

    @pytest.mark.asyncio
    async def test_owner_new_without_owner_row_does_not_mint_main(self, tmp_path):
        runner = _slash_key_runner(
            tmp_path, multiplex=True, routes=[_owner_route()]
        )
        store = runner.session_store
        store._entries[OWNER_MAIN_KEY] = _entry(OWNER_MAIN_KEY, "sess-main")
        main_before = store._entries[OWNER_MAIN_KEY].session_id
        created = []
        orig_goc = store.get_or_create_session

        def _track_goc(source, force_new=False):
            created.append(store._generate_session_key(source))
            return orig_goc(source, force_new=force_new)

        store.get_or_create_session = _track_goc
        event = MessageEvent(text="/new", source=_owner_source(profile=None))

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=_served_homes(tmp_path),
        ), patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="default",
        ):
            await runner._handle_reset_command(event)

        assert store._entries[OWNER_MAIN_KEY].session_id == main_before
        assert OWNER_OWNER_KEY in store._entries
        assert OWNER_MAIN_KEY not in created
        assert all(key.startswith("agent:owner:") for key in created) or (
            OWNER_OWNER_KEY in store._entries
            and store._entries[OWNER_OWNER_KEY].session_id != "sess-main"
        )

    @pytest.mark.asyncio
    async def test_unmultiplexed_new_still_uses_main_namespace(self, tmp_path):
        runner = _slash_key_runner(
            tmp_path, multiplex=False, routes=[_owner_route()]
        )
        store = runner.session_store
        store._entries[OWNER_MAIN_KEY] = _entry(OWNER_MAIN_KEY, "sess-main")
        main_before = store._entries[OWNER_MAIN_KEY].session_id
        event = MessageEvent(text="/new", source=_owner_source(profile=None))

        with patch(
            "hermes_cli.profiles.get_active_profile_name",
            return_value="owner",
        ):
            await runner._handle_reset_command(event)

        assert OWNER_OWNER_KEY not in store._entries
        assert store._entries[OWNER_MAIN_KEY].session_id != main_before
        assert event.source.profile is None

    def test_home_channel_probe_uses_routed_profile_scope(self, tmp_path):
        from gateway.run import _profile_runtime_scope

        homes = dict(_served_homes(tmp_path))
        runner = _slash_key_runner(
            tmp_path, multiplex=True, routes=[_owner_route()]
        )
        seen = []

        def fake_load():
            from hermes_constants import get_hermes_home

            home = Path(get_hermes_home()).resolve()
            seen.append(home)
            cfg = MagicMock()
            cfg.get_home_channel.return_value = (
                "set" if home == Path(homes["owner"]).resolve() else None
            )
            return cfg

        runner._resolve_profile_home_for_source = (
            lambda source: Path(homes["owner"])
        )
        runner.config.get_home_channel = lambda platform: None

        with patch(
            "gateway.config.load_gateway_config",
            side_effect=fake_load,
        ), patch(
            "agent.secret_scope.get_secret",
            return_value="",
        ), _profile_runtime_scope(Path(homes["default"])):
            configured = runner._home_channel_configured_for_source(
                _owner_source(profile=None)
            )

        assert configured is True
        assert seen, "probe must re-read config under the routed profile home"
        assert Path(homes["owner"]).resolve() in seen
        assert Path(homes["default"]).resolve() not in seen
