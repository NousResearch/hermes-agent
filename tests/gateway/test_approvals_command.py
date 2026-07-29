"""Gateway contract and live dispatch for /approvals."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _event(
    text: str = "/approvals",
    *,
    user_id: str = "user-1",
    profile: str | None = None,
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id="chat-1",
            chat_type="dm",
            profile=profile,
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


def _clear_config_caches() -> None:
    from hermes_cli import managed_scope
    from hermes_cli.config import _LOAD_CONFIG_CACHE, _RAW_CONFIG_CACHE

    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()


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


@pytest.mark.asyncio
async def test_multiplex_approvals_change_persists_only_to_routed_profile(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    routed_home = tmp_path / "routed"
    default_home.mkdir()
    routed_home.mkdir()
    (default_home / "config.yaml").write_text(
        yaml.safe_dump({"approvals": {"mode": "manual"}}),
        encoding="utf-8",
    )
    (routed_home / "config.yaml").write_text(
        yaml.safe_dump({"approvals": {"mode": "smart"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "missing-managed"))
    _clear_config_caches()

    runner = _runner()
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda _source: routed_home

    output = await runner._handle_approvals_command(
        _event("/approvals off", profile="routed")
    )

    default_config = yaml.safe_load((default_home / "config.yaml").read_text())
    routed_config = yaml.safe_load((routed_home / "config.yaml").read_text())
    assert output == "Approval mode: off (persistent profile setting)."
    assert default_config["approvals"]["mode"] == "manual"
    assert routed_config["approvals"]["mode"] == "off"


def test_multiplex_slash_access_uses_routed_profile_admin_policy(
    tmp_path, monkeypatch
):
    default_home = tmp_path / "default"
    routed_home = tmp_path / "routed"
    default_home.mkdir()
    routed_home.mkdir()
    (default_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "telegram": {
                    "allow_admin_from": ["default-admin"],
                    "user_allowed_commands": [],
                }
            }
        ),
        encoding="utf-8",
    )
    (routed_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "telegram": {
                    "allow_admin_from": ["routed-admin"],
                    "user_allowed_commands": [],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "missing-managed"))
    _clear_config_caches()

    runner = _runner()
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = gateway_run.load_gateway_config().platforms
    runner._resolve_profile_home_for_source = lambda _source: routed_home

    denial = runner._check_slash_access(
        _event(user_id="routed-admin", profile="routed").source,
        "approvals",
    )

    assert denial is None
