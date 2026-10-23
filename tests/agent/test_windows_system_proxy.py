"""Regression: Windows system proxy auto-detect in the platform adapter layer.

Mirrors ``_detect_macos_system_proxy``: when no env vars and no per-platform
YAML proxy are set, a Windows system HTTP(S) proxy from the registry
(HKCU ``Internet Settings`` ProxyEnable/ProxyServer) is the last-resort probe
used by ``resolve_proxy_url``.

Platform arms run on their host: the registry-detection arms require a real
Windows runner (they exercise the real ``winreg``-injection seam, with the
module itself patched, not the host platform).
"""
from unittest import mock

import pytest

from gateway.platforms import base


class _FakeKey:
    """Context-managed fake HKEY whose QueryValueEx mirrors a registry view."""

    def __init__(self, values):
        self._values = values

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def QueryValueEx(self, name):
        if name not in self._values:
            raise FileNotFoundError(name)
        return self._values[name], 4  # REG_DWORD


def _winreg_module(values):
    key = _FakeKey(values)

    def _query(_key, name):  # module-level winreg.QueryValueEx delegates to the key
        return _key.QueryValueEx(name)

    return mock.Mock(
        HKEY_CURRENT_USER="HKCU",
        OpenKey=mock.Mock(return_value=key),
        QueryValueEx=_query,
    )


@pytest.mark.windows_only
def test_windows_system_proxy_enabled_returns_url():
    with mock.patch.dict("sys.modules", {"winreg": _winreg_module({"ProxyEnable": 1, "ProxyServer": "127.0.0.1:7890"})}):
        assert base._detect_windows_system_proxy() == "http://127.0.0.1:7890"


@pytest.mark.windows_only
def test_windows_system_proxy_disabled_returns_none():
    with mock.patch.dict("sys.modules", {"winreg": _winreg_module({"ProxyEnable": 0, "ProxyServer": "127.0.0.1:7890"})}):
        assert base._detect_windows_system_proxy() is None


@pytest.mark.windows_only
def test_windows_proxy_semicolon_list_takes_first_entry():
    with mock.patch.dict("sys.modules", {"winreg": _winreg_module({"ProxyEnable": 1, "ProxyServer": "10.0.0.1:8080;10.0.0.2:8080"})}):
        assert base._detect_windows_system_proxy() == "http://10.0.0.1:8080"


def test_resolve_proxy_url_falls_back_to_windows_detection():
    """No env / no macOS probe -> Windows registry detection completes the chain."""
    with mock.patch.object(base, "gateway_trust_env", return_value=True), \
         mock.patch.object(base, "first_proxy_env_value", return_value=None), \
         mock.patch.object(base, "_detect_macos_system_proxy", return_value=None), \
         mock.patch.object(base, "_detect_windows_system_proxy", return_value="http://127.0.0.1:7890"):
        assert base.resolve_proxy_url(None, target_hosts=None, configured=None) == "http://127.0.0.1:7890"
