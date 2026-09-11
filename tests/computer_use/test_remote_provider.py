"""Remote computer-use provider tests: registration, availability, and backend wiring.

These exercise the seam contract (`agent/computer_use_provider.py`):

- ``is_available`` is config-only — cheap, no network, no local cua-driver probe — and
  a raising resolve is an absent provider.
- ``create_backend`` hands the backend its resolved remote config; selection is the
  provider's, never the backend's (the registry's silent-selection rule).
- An operator who writes ``computer_use.provider: remote`` without remote config gets
  a dispatch-time error naming the fix, not a quiet fall back to the local desktop.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict
from unittest import mock

import pytest

from agent.computer_use_registry import _reset_for_tests as _reset_registry
from tools.computer_use.remote import RemoteCuaConfig


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register providers around each test: registration happens at import time."""
    import tools.computer_use.host_provider  # noqa: F401
    import tools.computer_use.remote_provider  # noqa: F401
    yield
    _reset_registry()
    importlib.reload(tools.computer_use.host_provider)
    importlib.reload(tools.computer_use.remote_provider)


def _provider():
    from tools.computer_use.remote_provider import RemoteCuaProvider

    return RemoteCuaProvider()


def _cfg(remote: Dict[str, Any]) -> Dict[str, Any]:
    """The computer_use config block itself (what _computer_use_cfg returns)."""
    return {"provider": "remote", "remote": remote}


def _raise():
    raise RuntimeError("HERMES_CUA_REMOTE_TOKEN must contain at least 32 bytes")


def _set_token(monkeypatch, token: str = "t" * 64) -> None:
    if token is None:
        monkeypatch.delenv("HERMES_CUA_REMOTE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HERMES_CUA_REMOTE_TOKEN", token)


def test_provider_registers_under_remote_name():
    from agent.computer_use_registry import list_providers

    names = {p.name for p in list_providers()}
    assert "remote" in names


def test_display_name_is_human_readable():
    assert _provider().display_name == "Remote desktop (MCP host bridge)"


def test_is_available_true_when_config_resolves(monkeypatch):
    monkeypatch.setattr("tools.computer_use.remote_provider.RemoteCuaProvider._computer_use_cfg",
                        staticmethod(lambda: _cfg({"enabled": True, "url": "https://example.test:8765/mcp"})))
    _set_token(monkeypatch)
    assert _provider().is_available() is True


def test_is_available_false_without_remote_config(monkeypatch):
    monkeypatch.setattr("tools.computer_use.remote_provider.RemoteCuaProvider._computer_use_cfg",
                        staticmethod(lambda: {"provider": "remote"}))
    assert _provider().is_available() is False


def test_is_available_false_when_config_raises(monkeypatch):
    monkeypatch.setattr("tools.computer_use.remote_provider.RemoteCuaProvider._computer_use_cfg",
                        staticmethod(_raise))
    assert _provider().is_available() is False


def test_is_available_false_on_short_token(monkeypatch):
    monkeypatch.setattr("tools.computer_use.remote_provider.RemoteCuaProvider._computer_use_cfg",
                        staticmethod(lambda: _cfg({"enabled": True, "url": "https://example.test:8765/mcp"})))
    _set_token(monkeypatch, "short")
    assert _provider().is_available() is False


def test_create_backend_attaches_remote_config(monkeypatch):
    monkeypatch.setattr("tools.computer_use.remote_provider.RemoteCuaProvider._computer_use_cfg",
                        staticmethod(lambda: _cfg({"enabled": True, "url": "https://example.test:8765/mcp"})))
    _set_token(monkeypatch)
    backend = _provider().create_backend("sess-1", "standard")
    assert isinstance(backend._remote_config, RemoteCuaConfig)
    assert backend._remote_config.url == "https://example.test:8765/mcp"


def test_create_backend_raises_when_remote_not_configured(monkeypatch):
    monkeypatch.setattr("tools.computer_use.remote_provider.RemoteCuaProvider._computer_use_cfg",
                        staticmethod(lambda: {"provider": "remote"}))
    with pytest.raises(RuntimeError, match="no remote transport is configured"):
        _provider().create_backend("sess-1", "standard")


def test_local_backend_stays_local_without_remote_config():
    """The registry's silent-selection rule: a local backend must not pick up remote config."""
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend(permission_mode="standard")
    assert backend._remote_config is None


def test_check_requirements_remote_provider(monkeypatch):
    """The seam's check_fn path: a non-host provider answers for its own runtime."""
    import tools.computer_use.tool as cu_tool

    monkeypatch.setattr("tools.computer_use.remote_provider.RemoteCuaProvider._computer_use_cfg",
                        staticmethod(lambda: _cfg({"enabled": True, "url": "https://example.test:8765/mcp"})))
    _set_token(monkeypatch)
    # Point the dispatcher at our provider without touching real config files.
    with mock.patch("tools.computer_use.tool._configured_provider_name", return_value="remote"):
        cu_tool.reset_backend_for_tests()
        try:
            assert cu_tool.check_computer_use_requirements() is True
        finally:
            cu_tool.reset_backend_for_tests()


def test_unknown_provider_names_surface_at_dispatch():
    """An operator typo (`provider: remot`) must error at dispatch, not fall back to the local desktop.

    Per the seam's contract the check keeps the tool (the model can then *say* what is wrong); the
    dispatcher is what raises ``UnknownComputerUseProvider``, so both behaviours are asserted.
    """
    import tools.computer_use.tool as cu_tool
    from agent.computer_use_registry import UnknownComputerUseProvider

    with mock.patch("tools.computer_use.tool._configured_provider_name", return_value="remot"):
        cu_tool.reset_backend_for_tests()
        try:
            assert cu_tool.check_computer_use_requirements() is True  # tool kept...
            with pytest.raises(UnknownComputerUseProvider, match="remot"):
                cu_tool.active_computer_use_provider()  # ...dispatch errors naming it
        finally:
            cu_tool.reset_backend_for_tests()