"""Satellite cron routes come from the LIVE multiplex host, not only the default root.

A multiplexing gateway may be launched by a NAMED profile (``hermes -p ops gateway run`` with
``gateway.multiplex_profiles: true``); its ``profile_routes`` then live in
``<root>/profiles/ops/config.yaml`` and the root ``config.yaml`` carries none. Reading only the
default root made every credentialless satellite's routed cron delivery a false
``not connected`` preflight block and left ``SharedRouteAdapters`` empty, so delivery failed
closed too. The route source must be the host gateway actually serving the satellite:

* inside the host process, its own launch home;
* from any other process, the live host identified by ``host_gateway_serving``;
* nothing identifiable -> the legacy default-root read, which fails closed when it has no routes.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cron.scheduler_preflight import (
    SharedRouteAdapters,
    _primary_profile_routes_for_current_home,
    _preflight_check_delivery,
)
from gateway.config import Platform
from gateway.host_attach import HostGateway
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

ROUTED = "C0BTH5ZQH61"
HOST_YAML = {
    "gateway": {
        "multiplex_profiles": True,
        "profile_routes": [
            {"name": "slack-sat", "platform": "slack", "chat_id": ROUTED, "profile": "sat"},
            {"name": "slack-other", "platform": "slack", "chat_id": "C0OTHER", "profile": "other"},
        ],
    }
}


def _gateway_config(connected_values):
    config = MagicMock()
    config.get_connected_platforms.return_value = [MagicMock(value=v) for v in connected_values]
    return config


@pytest.fixture
def named_host(tmp_path, monkeypatch):
    """Root config with NO routes; the multiplex host is the named profile ``host``."""
    root = tmp_path / "root"
    host_home = root / "profiles" / "host"
    sat_home = root / "profiles" / "sat"
    host_home.mkdir(parents=True)
    sat_home.mkdir(parents=True)
    (root / "config.yaml").write_text(yaml.safe_dump({"model": {"default": "x"}}), encoding="utf-8")
    (host_home / "config.yaml").write_text(yaml.safe_dump(HOST_YAML), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    token = set_hermes_home_override(str(sat_home))
    yield root, host_home, sat_home
    reset_hermes_home_override(token)


def _serving(host_home, served=("host", "sat")):
    def fake(profile, *, wait_for_channel=0.0):
        gw = HostGateway(4242, Path(host_home), tuple(served))
        return gw if gw.serves(profile) else None
    return fake


def _no_host(profile, *, wait_for_channel=0.0):
    return None


def _assert_routed(sat_routes):
    host_adapter = object()
    shared = SharedRouteAdapters({Platform.SLACK: host_adapter}, sat_routes)
    assert shared.get(Platform.SLACK, {"chat_id": ROUTED}) is host_adapter
    # Unrouted chat ids (including another satellite's bridge) stay refused.
    assert shared.get(Platform.SLACK, {"chat_id": "C0OTHER"}) is None
    assert shared.get(Platform.SLACK, {"chat_id": "C0UNROUTED"}) is None


class TestNamedMultiplexHost:
    def test_out_of_process_satellite_reads_live_host_routes(self, named_host):
        """CLI/cron process for the satellite: routes come from the live host's home."""
        _, host_home, _ = named_host
        with patch("gateway.host_attach.host_gateway_serving", _serving(host_home)):
            routes = _primary_profile_routes_for_current_home()
            assert [r.chat_id for r in routes] == [ROUTED]
            _assert_routed(routes)
            with patch("gateway.config.load_gateway_config", return_value=_gateway_config(set())):
                assert _preflight_check_delivery({"deliver": f"slack:{ROUTED}"}) is None
                blocked = _preflight_check_delivery({"deliver": "discord:123"})
                assert blocked is not None and "discord" in blocked

    def test_in_host_process_reads_own_launch_home(self, named_host, monkeypatch):
        """Inside the host gateway (process home = host) no probe is needed."""
        _, host_home, _ = named_host
        monkeypatch.setenv("HERMES_HOME", str(host_home))
        monkeypatch.setattr("gateway.status.owns_gateway_runtime_lock", lambda: True)

        def must_not_probe(*a, **k):
            raise AssertionError("host process must not dial its own control socket")

        with patch("gateway.host_attach.host_gateway_serving", must_not_probe):
            routes = _primary_profile_routes_for_current_home()
        assert [r.chat_id for r in routes] == [ROUTED]
        _assert_routed(routes)

    def test_multiplex_home_without_gateway_lock_is_not_a_route_source(self, named_host, monkeypatch):
        """A CLI merely launched under the host's home (no runtime lock) must probe the live host."""
        _, host_home, _ = named_host
        monkeypatch.setenv("HERMES_HOME", str(host_home))
        monkeypatch.setattr("gateway.status.owns_gateway_runtime_lock", lambda: False)
        with patch("gateway.host_attach.host_gateway_serving", _no_host):
            assert _primary_profile_routes_for_current_home() == []

    def test_no_live_host_fails_closed(self, named_host):
        with patch("gateway.host_attach.host_gateway_serving", _no_host):
            assert _primary_profile_routes_for_current_home() == []
            with patch("gateway.config.load_gateway_config", return_value=_gateway_config(set())):
                reason = _preflight_check_delivery({"deliver": f"slack:{ROUTED}"})
        assert reason is not None and "slack" in reason

    def test_host_not_serving_this_profile_fails_closed(self, named_host):
        _, host_home, _ = named_host
        with patch("gateway.host_attach.host_gateway_serving", _serving(host_home, served=("host",))):
            assert _primary_profile_routes_for_current_home() == []

    def test_host_probe_error_fails_closed(self, named_host):
        def boom(*a, **k):
            raise OSError("socket gone")

        with patch("gateway.host_attach.host_gateway_serving", boom):
            assert _primary_profile_routes_for_current_home() == []

    def test_non_multiplex_process_home_is_not_a_route_source(self, named_host, monkeypatch):
        """A non-multiplexing process home with stray routes never grants a satellite anything."""
        root, _, _ = named_host
        stray = root / "profiles" / "stray"
        stray.mkdir()
        cfg = yaml.safe_load(yaml.safe_dump(HOST_YAML))
        cfg["gateway"]["multiplex_profiles"] = False
        (stray / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(stray))
        monkeypatch.setattr("gateway.status.owns_gateway_runtime_lock", lambda: True)
        with patch("gateway.host_attach.host_gateway_serving", _no_host):
            assert _primary_profile_routes_for_current_home() == []

    def test_host_itself_has_no_satellite_routes(self, named_host):
        """Serving the host profile itself: nothing to consult."""
        _, host_home, _ = named_host
        token = set_hermes_home_override(str(host_home))
        try:
            with patch("gateway.host_attach.host_gateway_serving", _serving(host_home)):
                assert _primary_profile_routes_for_current_home() == []
        finally:
            reset_hermes_home_override(token)
