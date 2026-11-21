"""An identity-less SECONDARY lane never delivers through the primary bot.

A reconnect (or a restored row without ``transport_profile``) can leave a source with no transport
identity. ``_delivery_adapter_for`` then falls back to the unique-owner heuristic. That heuristic
used ``source.profile`` only, which is empty on those rows, so a secondary lane's status bubbles,
long-running heartbeats and completion notices went out the DEFAULT bot — activity leaking into the
wrong chat, with no delivery-ledger row to find it later.

The fix reads the lane namespace from the session key when ``profile`` was never stamped: the lane's
own bot answers, and a lane whose bot is offline fails closed instead of borrowing the default one.
Satellites (a profile served through the default bot by ``profile_routes``) keep their egress.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session_identity import resolve_identity

CHAT = "100200300"


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


def _stub(platform, runner, label, owner=""):
    adapter = _Stub.__new__(_Stub)
    adapter.platform, adapter.gateway_runner, adapter.label = platform, runner, label
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._pending_messages, adapter._active_sessions = {}, {}
    if owner:
        adapter.set_owner_profile(owner)
    return adapter


def _runner(home, *, routes=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True, sessions_dir=home / "sessions")
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.config.profile_routes = parse_profile_routes(list(routes or []))
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {}
    runner._primary_profile_name = "default"
    primary = _stub(Platform.TELEGRAM, runner, "PRIMARY")
    studio = _stub(Platform.TELEGRAM, runner, "STUDIO", owner="studio")
    runner.adapters = {Platform.TELEGRAM: primary}
    runner._profile_adapters = {"studio": {Platform.TELEGRAM: studio}}
    return SimpleNamespace(runner=runner, home=home, primary=primary, studio=studio)


def _source(runner, *, chat_id=CHAT, profile=None, session_key=f"agent:studio:telegram:dm:{CHAT}"):
    """A source shaped like a restored/identity-less row: no transport adapter ref."""
    source = SimpleNamespace(
        platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm", user_id=chat_id,
        session_key=session_key,
    )
    if profile:
        source.profile = profile
    return source


@pytest.fixture
def mux(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    (home / "profiles" / "studio").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch("hermes_cli.profiles.profiles_to_serve",
               return_value=[("default", home), ("studio", home / "profiles" / "studio")]), \
            patch("hermes_cli.profiles.get_profile_dir",
                  side_effect=lambda n: home if n == "default" else home / "profiles" / n), \
            patch("hermes_cli.profiles.profile_exists", return_value=True):
        yield _runner(home)


def test_identityless_secondary_lane_delivers_through_its_own_bot(mux):
    """The regression: without the lane fallback this returns the PRIMARY adapter."""
    source = _source(mux.runner)
    assert mux.runner._delivery_adapter_for(source) is mux.studio


def test_identityless_secondary_lane_fails_closed_when_its_bot_is_offline(mux):
    """Bot offline is not a licence to answer from the default bot."""
    mux.runner._profile_adapters["studio"] = {}
    source = _source(mux.runner)
    assert mux.runner._delivery_adapter_for(source) is None


def test_identityless_primary_lane_still_uses_the_primary_bot(mux):
    """The default lane keeps its own egress — no fail-closed regression for main."""
    source = _source(mux.runner, session_key=f"agent:main:telegram:dm:{CHAT}")
    assert mux.runner._delivery_adapter_for(source) is mux.primary


def test_satellite_lane_keeps_default_bot_egress(tmp_path, monkeypatch):
    """A profile served THROUGH the default bot (``profile_routes``, no ``bot_profile``) must keep it."""
    routes = [{"name": "admin-dm", "platform": "telegram", "profile": "ops", "chat_id": "72719239"}]
    home = tmp_path / "hh2"
    (home / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch("hermes_cli.profiles.profiles_to_serve",
               return_value=[("default", home), ("ops", home / "profiles" / "ops")]), \
            patch("hermes_cli.profiles.get_profile_dir",
                  side_effect=lambda n: home if n == "default" else home / "profiles" / n), \
            patch("hermes_cli.profiles.profile_exists", return_value=True):
        rig = _runner(home, routes=routes)
    rig.runner._profile_adapters = {"ops": {}}
    source = _source(rig.runner, chat_id="72719239", session_key="agent:ops:telegram:dm:72719239")
    assert rig.runner._delivery_adapter_for(source) is rig.primary


def test_pinned_transport_still_wins_over_the_lane(mux):
    """A pinned identity (another bot receiving an ops lane) is untouched by the lane fallback."""
    team_b = _stub(Platform.TELEGRAM, mux.runner, "TEAM_B", owner="team_b")
    mux.runner._profile_adapters["team_b"] = {Platform.TELEGRAM: team_b}
    source = _source(mux.runner, profile="ops", session_key="agent:ops:telegram:dm:555")
    identity = resolve_identity(source, runner=mux.runner, transport_profile="team_b")
    assert identity.transport_profile == "team_b"
    assert mux.runner._delivery_adapter_for(source) is team_b
