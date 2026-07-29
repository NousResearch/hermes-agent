"""Gateway contract and live dispatch for /approvals."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _event(text: str = "/approvals") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="user-1",
            chat_id="chat-1",
            chat_type="dm",
        ),
    )


def _runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace(platforms={})
    runner.hooks = MagicMock(loaded_hooks=[])
    runner.hooks.emit = AsyncMock(return_value=[])
    runner._running_agents = {}
    runner._get_or_create_gateway_honcho = lambda _key: (None, None)
    runner._is_user_authorized = lambda _source: True
    runner.session_store = SimpleNamespace(get_or_create_session=lambda _source: None)
    return runner


@pytest.mark.asyncio
async def test_gateway_handler_uses_shared_persistent_logic_without_cache_eviction():
    runner = _runner()
    result = SimpleNamespace(message="Approval mode: manual (persistent profile setting).")
    runner._evict_cached_agent = MagicMock()

    with patch("hermes_cli.approval_mode.run_approval_mode_command", return_value=result) as run:
        output = await runner._handle_approvals_command(_event("/approvals manual"))

    assert output == result.message
    run.assert_called_once_with("manual")
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_rejects_non_admin_persistent_approval_change():
    runner = _runner()
    runner.config = SimpleNamespace(
        platforms={
            Platform.TELEGRAM: SimpleNamespace(
                extra={
                    "allow_admin_from": ["admin-1"],
                    "user_allowed_commands": ["approvals"],
                }
            )
        }
    )

    with patch("hermes_cli.approval_mode.run_approval_mode_command") as run:
        output = await runner._handle_approvals_command(_event("/approvals off"))

    assert "admin" in output.lower()
    run.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_live_dispatch_routes_and_persists_approvals_command(tmp_path, monkeypatch):
    runner = _runner()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "missing-managed"))
    from hermes_cli import managed_scope
    from hermes_cli.config import _LOAD_CONFIG_CACHE, _RAW_CONFIG_CACHE

    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()

    output = await runner._handle_message(_event("/approvals manual"))

    assert output == "Approval mode: manual (persistent profile setting)."
    assert yaml.safe_load((home / "config.yaml").read_text())["approvals"]["mode"] == "manual"
