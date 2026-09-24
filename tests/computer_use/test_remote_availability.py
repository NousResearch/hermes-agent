"""Remote-availability wiring: check_fn, backend availability, discovery diagnosis, config defaults.

Behaviour contracts (not change-detectors):
- check_fn / backend is_available: an active remote CUA transport replaces the local
  cua-driver binary requirement; a broken remote config fails closed at the registry
  layer (tool stays hidden); inactive remote falls through to the local platform +
  binary rules unchanged.
- _empty_discovery_reason: remote-active reports the host-bridge hint and never
  probes the local loginctl/DISPLAY session; inactive engages the local probing path.
- DEFAULT_CONFIG ships computer_use.remote disabled (opt-in) with an empty URL.
- `hermes computer-use host-bridge` parses with safe defaults, requires --allowed-hosts,
  and dispatches exactly the kwargs run_host_bridge owns today.
"""
import argparse

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.subcommands import computer_use as cu_cli
from tools.computer_use import cua_backend
from tools.computer_use.cua_backend import CuaDriverBackend
from tools.computer_use.tool import check_computer_use_requirements, reset_backend_for_tests
from tools.computer_use.remote_provider import RemoteCuaProvider


@pytest.fixture(autouse=True)
def _selection(monkeypatch):
    reset_backend_for_tests()
    monkeypatch.setattr("tools.computer_use.tool._configured_provider_name", lambda: "remote")
    yield
    reset_backend_for_tests()


def _remote_stub(url: str = "https://bridge.example.com:8443"):
    return argparse.Namespace(url=url)  # structurally RemoteCuaConfig for the call sites under test


def _patch_backend_cfg(monkeypatch, resolve):
    """Patch transport resolution at the provider, not ambient backend state."""
    monkeypatch.setattr(RemoteCuaProvider, "_resolve_config", lambda self, *a: resolve())


class TestCheckFnRemoteAvailability:
    def test_remote_active_true_without_local_binary(self, monkeypatch):
        # Remote transport: no local cua-driver binary needed, even on a host that has none.
        monkeypatch.setattr("tools.computer_use.remote.resolve_remote_cua_config",
                            lambda *a, **k: _remote_stub())
        monkeypatch.setattr(RemoteCuaProvider, "_computer_use_cfg", staticmethod(lambda: {"remote": {"enabled": True}}))
        monkeypatch.setattr("tools.computer_use.cua_backend_driver.cua_driver_binary_available",
                            lambda: False)
        assert check_computer_use_requirements() is True

    def test_broken_remote_config_fails_closed(self, monkeypatch):
        def _broken(*a, **k):
            raise RuntimeError("HERMES_CUA_REMOTE_TOKEN must contain at least 32 bytes")
        monkeypatch.setattr("tools.computer_use.remote.resolve_remote_cua_config", _broken)
        monkeypatch.setattr(RemoteCuaProvider, "_computer_use_cfg", staticmethod(lambda: {"remote": {"enabled": True}}))
        assert check_computer_use_requirements() is False

    def test_no_remote_no_binary_false(self, monkeypatch):
        # Remote inactive: the local rules decide — no binary means the tool stays hidden.
        monkeypatch.setattr("tools.computer_use.tool._configured_provider_name", lambda: "local")
        monkeypatch.setattr("tools.computer_use.remote.resolve_remote_cua_config",
                            lambda *a, **k: None)
        monkeypatch.setattr(RemoteCuaProvider, "_computer_use_cfg", staticmethod(lambda: {"remote": {"enabled": True}}))
        monkeypatch.setattr("tools.computer_use.cua_backend_driver.cua_driver_binary_available",
                            lambda: False)
        assert check_computer_use_requirements() is False

    def test_no_remote_local_binary_true(self, monkeypatch):
        # Remote inactive: local availability is unchanged by the remote branch.
        monkeypatch.setattr("tools.computer_use.tool._configured_provider_name", lambda: "local")
        monkeypatch.setattr("tools.computer_use.remote.resolve_remote_cua_config",
                            lambda *a, **k: None)
        monkeypatch.setattr(RemoteCuaProvider, "_computer_use_cfg", staticmethod(lambda: {"remote": {"enabled": True}}))
        monkeypatch.setattr("tools.computer_use.cua_backend_driver.cua_driver_binary_available",
                            lambda: True)
        assert check_computer_use_requirements() is True


class TestBackendIsAvailable:
    def test_remote_active_true_without_local_binary(self, monkeypatch):
        monkeypatch.setattr(cua_backend, "cua_driver_binary_available", lambda: False)
        assert CuaDriverBackend(remote_config=_remote_stub()).is_available() is True

    def test_no_remote_no_binary_false(self, monkeypatch):
        monkeypatch.setattr(cua_backend, "cua_driver_binary_available", lambda: False)
        assert CuaDriverBackend().is_available() is False

    def test_no_remote_local_binary_true(self, monkeypatch):
        monkeypatch.setattr(cua_backend, "cua_driver_binary_available", lambda: True)
        assert CuaDriverBackend().is_available() is True


class TestRemoteCfgFailureScope:
    """Provider resolution errors may hide the tool, but never select a host backend."""

    def test_runtime_error_refuses_construction(self, monkeypatch):
        _patch_backend_cfg(monkeypatch, lambda: (_ for _ in ()).throw(
            RuntimeError("HERMES_CUA_REMOTE_TOKEN must contain at least 32 bytes")))
        with pytest.raises(RuntimeError, match="32 bytes"):
            RemoteCuaProvider().create_backend("s", "standard")
        assert not RemoteCuaProvider().is_available()

    def test_unexpected_exception_not_silenced(self, monkeypatch):
        _patch_backend_cfg(monkeypatch, lambda: (_ for _ in ()).throw(
            TypeError("config dict is actually a list")))
        with pytest.raises(TypeError, match="config dict is actually a list"):
            RemoteCuaProvider().create_backend("s", "standard")

    def test_unexpected_exception_does_not_select_local_available(self, monkeypatch):
        _patch_backend_cfg(monkeypatch, lambda: (_ for _ in ()).throw(KeyError("remote")))
        monkeypatch.setattr("tools.computer_use.cua_backend_driver.cua_driver_binary_available", lambda: True)
        assert check_computer_use_requirements() is False


class TestEmptyDiscoveryReason:
    def test_remote_active_skips_local_probing(self, monkeypatch):

        def _must_not_probe():
            raise AssertionError("local session probe must not run while remote transport is active")

        monkeypatch.setattr(cua_backend, "_linux_session_locked", _must_not_probe)
        reason = cua_backend._empty_discovery_reason(remote=True)
        assert "remote desktop returned no windows" in reason
        assert "host bridge" in reason

    def test_remote_inactive_engages_local_probing(self, monkeypatch):
        probed = []

        def _probe():
            probed.append(True)
            return None

        monkeypatch.setattr(cua_backend, "_linux_session_locked", _probe)
        assert "remote desktop returned no windows" not in cua_backend._empty_discovery_reason()
        assert probed  # the local diagnosis path is the one that ran


class TestDefaultConfigRemoteSection:
    def test_remote_off_by_default(self):
        # Opt-in contract: shipping the key disabled keeps every existing install in local mode.
        remote = DEFAULT_CONFIG["computer_use"]["remote"]
        assert remote["enabled"] is False
        assert remote["url"] == ""


class TestBridgeSubcommand:
    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        cu_cli.build_computer_use_parser(sub)
        return parser.parse_args(argv)

    def test_bridge_defaults(self):
        args = self._parse(["computer-use", "host-bridge", "--allowed-hosts", "desktop.example.com"])
        assert args.computer_use_action == "host-bridge"
        assert args.port == 8765
        assert args.bind == "127.0.0.1"
        assert args.allowed_origins == ""
        assert args.session_idle_timeout == 1800

    def test_allowed_hosts_required(self):
        # An empty allowlist must not be silently accepted at the CLI layer either.
        with pytest.raises(SystemExit):
            self._parse(["computer-use", "host-bridge"])

    def test_dispatch_passes_only_current_run_host_bridge_kwargs(self, monkeypatch):
        # Signature-drift guard: new bridge options default host-side, the CLI forwards five.
        captured = {}

        def _fake_bridge(**kw):
            captured.update(kw)

        monkeypatch.setattr("tools.computer_use.host_bridge_cli.run_host_bridge", _fake_bridge)
        args = argparse.Namespace(allowed_hosts="a.com,b.com", allowed_origins="",
                                  port=9000, bind="0.0.0.0", session_idle_timeout=1800)
        assert cu_cli._cu_host_bridge(args) == 0
        assert captured == {"allowed_hosts": ["a.com", "b.com"], "allowed_origins": [],
                            "port": 9000, "bind": "0.0.0.0", "session_idle_timeout": 1800}