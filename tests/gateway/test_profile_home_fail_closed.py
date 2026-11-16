"""A multiplexed gateway refuses a source naming a profile it does not serve (#154610).

Real profile predicates under a temporary HERMES_HOME: an explicit ``source.profile`` (webhook
prefix, goal wakeup, local recovery, API notification) or a route that names a missing, parked or
invalid profile must never resolve to the launch profile's home.
"""

from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.profile_routing import ProfileRouteRejected
from gateway.session import SessionSource
from gateway.session_identity import identity_of, restore_identity, resolve_identity


@pytest.fixture
def homes(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    served = default_home / "profiles" / "served"
    parked = default_home / "profiles" / "parked"
    for home in (served, parked):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (parked / "gateway.parked").write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home, served, parked


def _runner(multiplex=True):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    runner._primary_profile_name = "default"
    runner.adapters = {}
    runner._profile_adapters = {}
    return runner


def _src(profile=None):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="1", profile=profile)


def test_parked_fixture_is_installed_but_not_served(homes):
    from hermes_cli.profiles import profile_exists, profiles_to_serve
    assert profile_exists("parked")
    assert "parked" not in {name for name, _ in profiles_to_serve(True)}


@pytest.mark.parametrize("name", ["ghost", "parked", "Bad/Name"])
def test_direct_source_profile_is_refused_not_served_by_launch_home(homes, name):
    with pytest.raises(ProfileRouteRejected):
        _runner()._resolve_profile_home_for_source(_src(name))


def test_served_and_unnamed_sources_still_resolve(homes):
    default_home, served, _parked = homes
    runner = _runner()
    assert runner._resolve_profile_home_for_source(_src("served")) == served
    assert Path(runner._resolve_profile_home_for_source(_src())) == default_home


def test_routed_profile_missing_from_disk_is_refused(homes, monkeypatch):
    runner = _runner()
    monkeypatch.setattr(runner, "_profile_name_for_source", lambda _s: "ghost", raising=False)
    with pytest.raises(ProfileRouteRejected):
        runner._resolve_profile_home_for_source(_src())


def test_single_profile_gateway_keeps_launch_home_fallback(homes):
    default_home, _served, _parked = homes
    assert Path(_runner(multiplex=False)._resolve_profile_home_for_source(_src("ghost"))) == default_home


def test_ingress_identity_for_unserved_profile_is_unresolved(homes):
    source = _src("ghost")
    with pytest.raises(Exception) as exc_info:
        resolve_identity(source, runner=_runner(), transport_profile="default")
    assert type(exc_info.value).__name__ == "IdentityUnresolved"
    assert source.profile_route_rejected is True
    assert identity_of(source) is None


def test_restored_origin_for_deleted_profile_is_not_revived(homes):
    source = _src("ghost")
    assert restore_identity(source, runner=_runner(), transport_profile="default") is None
    assert source.profile_route_rejected is True
    assert identity_of(source) is None
