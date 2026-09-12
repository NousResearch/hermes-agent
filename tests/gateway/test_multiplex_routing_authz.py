"""Routing/authorization invariants for a multiplexed gateway (#104933, #103717).

Every test builds a bare ``GatewayRunner`` with stub adapters against a temp ``HERMES_HOME`` and
exercises the real resolvers (no patched predicates).
"""

import asyncio
import threading
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session import SessionSource


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


def _stub(platform, runner, label):
    adapter = _Stub.__new__(_Stub)
    adapter.platform, adapter.gateway_runner, adapter.label = platform, runner, label
    adapter._pending_messages, adapter._active_sessions = {}, {}
    return adapter


@pytest.fixture
def mux(tmp_path, monkeypatch):
    """Default home allows user 777; profile ``ops`` is a shared-bot satellite; ``team_b`` owns a bot."""
    from agent import secret_scope
    from gateway.run import GatewayRunner

    home = tmp_path / "hh"
    for name in ("ops", "team_b"):
        (home / "profiles" / name).mkdir(parents=True)
    (home / ".env").write_text("TELEGRAM_ALLOWED_USERS=777\n")
    (home / "profiles" / "team_b" / ".env").write_text("TELEGRAM_ALLOWED_USERS=72719239\n")
    (home / "profiles" / "ops" / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("TELEGRAM_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(key, raising=False)
    prev = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.config.profile_routes = parse_profile_routes(
        [{"name": "admin-dm", "platform": "telegram", "profile": "ops", "chat_id": "72719239"}])
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {}
    runner._primary_profile_name = "default"
    primary = _stub(Platform.TELEGRAM, runner, "PRIMARY")
    team_b = _stub(Platform.TELEGRAM, runner, "TEAM_B")
    team_b.set_owner_profile("team_b")
    runner.adapters = {Platform.TELEGRAM: primary}
    runner._profile_adapters = {"team_b": {Platform.TELEGRAM: team_b}, "ops": {}}
    served = [("default", home), ("ops", home / "profiles" / "ops"), ("team_b", home / "profiles" / "team_b")]
    with patch("hermes_cli.profiles.profiles_to_serve", return_value=served), \
            patch("hermes_cli.profiles.get_profile_dir", side_effect=lambda n: home / "profiles" / n), \
            patch("hermes_cli.profiles.profile_exists", return_value=True):
        yield SimpleNamespace(runner=runner, home=home, primary=primary, team_b=team_b)
    secret_scope.set_multiplex_active(prev)


def test_shared_bot_route_does_not_hijack_dedicated_secondary_bot(mux):
    """A chat_id route for the shared bot must not re-home the same user's DM with team_b's own bot;
    the same DM to the shared bot still routes to ``ops`` (#104933)."""
    via_team_b = mux.team_b.build_source(chat_id="72719239", chat_type="dm", user_id="72719239")
    assert via_team_b.profile == "team_b"
    assert mux.runner._session_key_for_source(via_team_b) == "agent:team_b:telegram:dm:72719239"
    via_primary = mux.primary.build_source(chat_id="72719239", chat_type="dm", user_id="72719239")
    assert via_primary.profile == "ops"


def test_secondary_busy_followup_authorized_against_its_own_allowlist(mux):
    """The busy-session handler of a secondary adapter must run under that profile's scope, so its
    owner (absent from the default allowlist) is admitted (#103717)."""
    from agent import secret_scope

    seen = {}

    async def _busy(event, session_key):
        seen["scoped"] = secret_scope.current_secret_scope() is not None
        return mux.runner._is_user_authorized(event.source)

    mux.runner._handle_active_session_busy_message = _busy
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="72719239", chat_type="dm", user_id="72719239")
    source._transport_adapter_ref = weakref.ref(mux.team_b)
    handler = mux.runner._make_profile_busy_session_handler("team_b")
    assert asyncio.run(handler(SimpleNamespace(source=source), "k")) is True
    assert seen["scoped"] is True


def test_mid_turn_authorization_reads_admitting_transport_allowlist(mux):
    """Inside a routed satellite's turn (its scope has no allowlist) ``_is_user_authorized_for_source``
    must admit the shared bot's owner without an ingress stamp — the seam /topic, sibling /stop, plugin
    injection, voice and auto-resume now use."""
    from gateway.run import _profile_runtime_scope

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="777", chat_type="dm", user_id="777", profile="ops")
    source._transport_adapter_ref = weakref.ref(mux.primary)
    with _profile_runtime_scope(mux.home / "profiles" / "ops"):
        assert mux.runner._is_user_authorized(source) is False  # the bug the call sites had
        assert mux.runner._is_user_authorized_for_source(source) is True
    stranger = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="1", profile="ops")
    stranger._transport_adapter_ref = weakref.ref(mux.primary)
    with _profile_runtime_scope(mux.home / "profiles" / "ops"):
        assert mux.runner._is_user_authorized_for_source(stranger) is False


def test_shared_bot_satellite_resolves_primary_transport_for_restored_sources(mux):
    """A routed satellite with no bot of its own drains through the primary for restored/cached
    sources (no transport ref); a disconnected secondary that owns a credential stays fail-closed."""
    restored = SessionSource(platform=Platform.TELEGRAM, chat_id="72719239", chat_type="dm", user_id="7", profile="ops")
    assert mux.runner._authorization_adapter(Platform.TELEGRAM, "ops") is mux.primary
    assert mux.runner._adapter_for_source(restored) is mux.primary
    assert mux.runner._resolve_injection_adapter("telegram", restored) is mux.primary
    mux.runner._profile_adapters["team_b"] = {}  # team_b's bot is down: never borrow the primary
    assert mux.runner._authorization_adapter(Platform.TELEGRAM, "team_b") is None


def test_completion_preflight_runs_in_target_profile_scope(mux):
    """An async-delegation completion for a secondary session must be classified against THAT
    profile's state.db (the watcher runs unscoped, where the row does not exist → ``terminal``)."""
    from gateway import run as run_module
    from hermes_state import SessionDB

    db_home = mux.home / "profiles" / "team_b"
    SessionDB(db_path=db_home / "state.db").create_session(
        session_id="sess-b", source="telegram", session_key="agent:team_b:telegram:dm:1", profile_name="team_b")
    runner = mux.runner
    runner._session_db_pinned = run_module._SESSION_DB_UNPINNED
    runner._session_db_handles, runner._session_db_handles_lock = {}, threading.Lock()
    runner.session_store, runner._session_sources = None, {}
    evt = {"type": "async_delegation", "session_key": "agent:team_b:telegram:dm:1", "parent_session_id": "sess-b"}

    async def _run():
        unscoped = await runner._classify_completion_target("sess-b")
        async with runner._completion_event_scope(evt):
            scoped = await runner._classify_completion_target("sess-b")
        return unscoped, scoped

    assert asyncio.run(_run()) == ("terminal", "deliver")
