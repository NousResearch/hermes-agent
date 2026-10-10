"""Same-profile iLink accounts isolate sessions, credentials and restored delivery."""

import copy
import asyncio
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.weixin import save_weixin_account
from gateway.platforms.weixin_group import WeixinAccountGroup, account_configs, resolve_outbound_account
from gateway.session import SessionSource, build_session_key


@pytest.fixture
def accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for account_id in ("a", "b"):
        save_weixin_account(str(tmp_path), account_id=account_id, token=f"token-{account_id}", base_url="https://example.test")
    config = PlatformConfig(enabled=True, extra={"account_id": "a", "default_account": "a", "accounts": {
        "a": {"dm_policy": "allowlist", "allow_from": ["user-a"]},
        "b": {"dm_policy": "allowlist", "allow_from": ["user-b"]},
    }})
    return tmp_path, config


def runner_for(group, config):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False, platforms={Platform.WEIXIN: config})
    runner.adapters = {Platform.WEIXIN: group}
    runner._profile_adapters = {}
    runner._primary_profile_name = "default"
    runner._voice_mode = {"weixin:same-peer": "voice_only"}
    group.gateway_runner = runner
    return runner


@pytest.mark.asyncio
async def test_live_and_restored_sources_use_the_receiving_account(accounts, monkeypatch):
    home, config = accounts
    monkeypatch.setattr("gateway.platforms.weixin_group.WeixinAdapter.connect", AsyncMock(return_value=True))
    monkeypatch.setattr("gateway.platforms.weixin_group.WeixinAdapter.disconnect", AsyncMock())
    group = WeixinAccountGroup(config)
    runner = runner_for(group, config)
    await group.apply_config(config)
    a, b = group.accounts["a"], group.accounts["b"]
    sa, sb = [child.build_source(chat_id="same-peer", user_id="user-b") for child in (a, b)]
    assert build_session_key(sa) != build_session_key(sb)
    assert runner._owning_profile(b, Platform.WEIXIN) == (True, None)
    assert runner._delivery_adapter_for(sb) is b
    restored = SessionSource.from_dict(sb.to_dict())
    assert restored.account_id == "b"
    assert runner._delivery_adapter_for(restored) is b
    assert "same-peer" in b._auto_tts_enabled_chats
    assert runner._is_user_authorized(sb) is True
    assert runner._is_user_authorized(sa) is False
    missing = SessionSource(Platform.WEIXIN, "same-peer", account_id="missing")
    assert runner._delivery_adapter_for(missing) is None
    group.accounts.pop("b")
    assert runner._delivery_adapter_for(restored) is None


@pytest.mark.asyncio
async def test_reconfiguration_waits_for_active_account_and_keeps_other_bot(accounts, monkeypatch):
    home, config = accounts
    connect = AsyncMock(return_value=True)
    disconnect = AsyncMock()
    monkeypatch.setattr("gateway.platforms.weixin_group.WeixinAdapter.connect", connect)
    monkeypatch.setattr("gateway.platforms.weixin_group.WeixinAdapter.disconnect", disconnect)
    group = WeixinAccountGroup(config)
    await group.apply_config(config)
    a, b = group.accounts["a"], group.accounts["b"]
    a._active_sessions["active-turn"] = object()
    updated = copy.deepcopy(config)
    updated.extra["accounts"]["a"]["dm_policy"] = "disabled"
    assert await group.apply_config(updated) is False
    assert group.accounts["a"] is a and group.accounts["b"] is b
    assert disconnect.await_count == 0
    a._active_sessions.clear()
    assert await group.apply_config(updated) is True
    assert group.accounts["a"] is not a and group.accounts["b"] is b
    assert group.accounts["a"]._dm_policy == "disabled"
    assert disconnect.await_count == 1
    updated.extra["accounts"]["a"]["enabled"] = False
    await group.apply_config(updated)
    assert "a" not in group.accounts and group.accounts["b"] is b


def test_direct_targets_resolve_only_profile_local_enabled_accounts(accounts):
    home, config = accounts
    selected, peer = resolve_outbound_account(config.extra, None, "b/wxid_peer")
    assert selected.token == "token-b" and peer == "wxid_peer"
    config.extra["accounts"]["b"]["enabled"] = False
    with pytest.raises(ValueError):
        resolve_outbound_account(config.extra, None, "b/wxid_peer")
    save_weixin_account(str(home / "other"), account_id="foreign", token="foreign-token", base_url="https://example.test")
    config.extra["accounts"]["foreign"] = {}
    assert "foreign" not in account_configs(config, home)


@pytest.mark.asyncio
async def test_disk_config_reload_is_scoped_and_does_not_change_process_env(accounts, monkeypatch):
    import os
    from hermes_cli.config import save_config
    home, config = accounts
    monkeypatch.setattr("gateway.platforms.weixin_group.WeixinAdapter.connect", AsyncMock(return_value=True))
    monkeypatch.setattr("gateway.platforms.weixin_group.WeixinAdapter.disconnect", AsyncMock())
    group = WeixinAccountGroup(config)
    await group.apply_config(config)
    save_config({"platforms": {"weixin": {"enabled": True, "extra": {"accounts": {
        "b": {"dm_policy": "disabled"}}, "default_account": "b"}}}}, strip_defaults=False)
    monkeypatch.setenv("WEIXIN_TOKEN", "ambient-other-profile")
    assert await group._reload_config() is True
    assert list(group.accounts) == ["b"]
    assert group.accounts["b"]._token == "token-b"
    assert group.accounts["b"]._dm_policy == "disabled"
    assert os.environ["WEIXIN_TOKEN"] == "ambient-other-profile"


@pytest.mark.asyncio
async def test_watcher_applies_disk_changes_and_invalid_yaml_keeps_live_accounts(accounts, monkeypatch):
    from hermes_cli.config import save_config
    home, config = accounts
    changed = asyncio.Event()

    async def disconnect(child):
        changed.set()

    monkeypatch.setattr("gateway.platforms.weixin_group.WeixinAdapter.connect", AsyncMock(return_value=True))
    monkeypatch.setattr("gateway.platforms.weixin_group.WeixinAdapter.disconnect", disconnect)
    group = WeixinAccountGroup(config)
    await group.apply_config(config)
    group._signature = group._config_signature()
    watch = asyncio.create_task(group._watch_config())
    try:
        save_config({"platforms": {"weixin": {"enabled": True, "extra": {
            "accounts": {"b": {"dm_policy": "allowlist", "allow_from": ["new-user"]}},
            "default_account": "b"}}}}, strip_defaults=False)
        await asyncio.wait_for(changed.wait(), 10)
        assert set(group.accounts) == {"b"}
        assert group.accounts["b"]._allow_from == ["new-user"]
        child = group.accounts["b"]
        (home / "config.yaml").write_text("platforms: [broken", encoding="utf-8")
        with pytest.raises(Exception):
            await group._reload_config()
        assert group.accounts["b"] is child
    finally:
        watch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watch


@pytest.mark.asyncio
async def test_cron_origin_metadata_delivers_through_the_original_account(accounts, monkeypatch):
    from gateway.platforms.base import SendResult
    from gateway.session_context import set_session_vars, clear_session_vars
    from tools.cronjob_job_args import _origin_from_env
    from cron.scheduler_delivery_origin import stamp_origin_discriminators
    home, config = accounts
    tokens = set_session_vars(platform="weixin", chat_id="wxid_peer", user_id="user-b", account_id="b")
    try:
        origin = _origin_from_env()
    finally:
        clear_session_vars(tokens)
    assert origin["account_id"] == "b"
    metadata, media_metadata = {}, {}
    target = SimpleNamespace(origin=origin, origin_target=True, is_relay=False, origin_user_id="user-b")
    stamp_origin_discriminators(target, metadata, media_metadata)
    assert metadata["account_id"] == media_metadata["account_id"] == "b"
    monkeypatch.setattr("gateway.platforms.weixin_group.WeixinAdapter.connect", AsyncMock(return_value=True))
    group = WeixinAccountGroup(config)
    await group.apply_config(config)
    b = group.accounts["b"]
    b.send = AsyncMock(return_value=SendResult(success=True, message_id="cron-delivery"))
    assert (await group.send("wxid_peer", "scheduled message", metadata=metadata)).success
    b.send.assert_awaited_once()
    assert not (await group.send("wxid_peer", "scheduled message", metadata={"account_id": "unknown"})).success
