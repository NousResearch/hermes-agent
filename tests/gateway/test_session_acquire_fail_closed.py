"""Messaging admission must never run a turn without coordinated ownership."""
import asyncio
import builtins
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig
from hermes_cli import active_sessions
from tests.gateway.test_max_concurrent_sessions import (
    _make_event,
    _make_runner,
    _silence_global_gateway_hooks,
)


def _fail_registry(monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise OSError("private/path credential=synthetic-secret")

    if failure == "import":
        real_import = builtins.__import__

        def registry_import(name, *args, **kwargs):
            if name == "hermes_cli.active_sessions":
                raise ImportError("private/path credential=synthetic-secret")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", registry_import)
    else:
        monkeypatch.setattr(active_sessions, "_write_entries" if failure == "write" else "_FileLock", unavailable)


@pytest.mark.parametrize("failure", ["write", "lock", "import"])
def test_registry_write_failure_refuses_inbound_message(tmp_path, monkeypatch, caplog, failure):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _silence_global_gateway_hooks(monkeypatch)
    runner = _make_runner()
    runner._handle_message_with_agent = AsyncMock(return_value="unleased work executed")
    _fail_registry(monkeypatch, failure)

    # A handled refusal, not a raised exception: the inbound event loop stays usable.
    response = asyncio.run(runner._handle_message(_make_event()))

    assert getattr(response, "reason", None) == "SESSION_COORDINATION_UNAVAILABLE"
    assert "try again" in response.lower()
    assert "private/path" not in response and "synthetic-secret" not in response
    assert "synthetic-secret" not in caplog.text
    runner._handle_message_with_agent.assert_not_awaited()
    cast(MagicMock, runner.session_store).get_or_create_session.assert_not_called()
    assert runner._running_agents == {}


@pytest.mark.parametrize("reason", ["MAX_CONCURRENT_SESSIONS", "SESSION_NOT_OWNED"])
def test_known_refusal_is_returned_unchanged(tmp_path, monkeypatch, reason):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _silence_global_gateway_hooks(monkeypatch)
    runner = _make_runner()
    runner._handle_message_with_agent = AsyncMock()
    refusal = active_sessions.ActiveSessionRefusal("Existing controlled refusal", reason)
    monkeypatch.setattr(active_sessions, "try_acquire_active_session", lambda **kwargs: (None, refusal))

    response = asyncio.run(runner._handle_message(_make_event()))

    assert response is refusal
    assert isinstance(response, active_sessions.ActiveSessionRefusal)
    assert response.reason == reason
    runner._handle_message_with_agent.assert_not_awaited()
    cast(MagicMock, runner.session_store).get_or_create_session.assert_not_called()
    assert runner._running_agents == {}


@pytest.mark.parametrize("config", [GatewayConfig(), {}, SimpleNamespace(max_concurrent_sessions=None)])
def test_synthetic_config_still_acquires_real_ownership(tmp_path, monkeypatch, config):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = _make_runner()
    monkeypatch.setattr(runner, "config", config)

    lease, refusal = runner._claim_active_session_slot("test-session", _make_event().source)
    try:
        assert refusal is None
        assert lease is not None
        entries = active_sessions.active_session_registry_snapshot()
        assert len(entries) == 1
        assert entries[0]["session_id"] == "test-session"
    finally:
        if lease is not None:
            active_sessions.release_active_session(lease)
    assert active_sessions.active_session_registry_snapshot() == []
