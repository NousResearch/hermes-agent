import yaml
import pytest
from datetime import datetime
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.mark.asyncio
async def test_sethome_persists_feishu_platform_home_channel(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                extra={
                    "accounts": {
                        "corp": {
                            "app_id": "cli_corp",
                            "app_secret": "sec_corp",
                        },
                    },
                },
            ),
        },
    )

    event = MessageEvent(
        text="/sethome",
        source=SessionSource(
            platform=Platform.FEISHU,
            account_id="corp",
            chat_id="oc_c7a439d23a9a824e7c5b4352f802d660",
            chat_name="Team CLAIRE",
            chat_type="group",
            user_id="ou_owner",
            user_name="Ethan",
        ),
    )

    result = await runner._handle_set_home_command(event)

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    home = saved["platforms"]["feishu"]["home_channel"]

    assert home["account_id"] == "corp"
    assert home["chat_id"] == "oc_c7a439d23a9a824e7c5b4352f802d660"
    assert home["name"] == "Team CLAIRE"
    assert (
        runner.config.get_home_channel(Platform.FEISHU).chat_id
        == "oc_c7a439d23a9a824e7c5b4352f802d660"
    )
    assert runner.config.get_home_channel(Platform.FEISHU).account_id == "corp"
    assert "Home channel set to **Team CLAIRE**" in result


def test_has_home_channel_for_feishu_account_uses_platform_scoped_config():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                home_channel=HomeChannel(
                    platform=Platform.FEISHU,
                    chat_id="oc_dm_home",
                    name="Owner DM",
                    account_id="corp",
                ),
                extra={
                    "accounts": {
                        "corp": {
                            "app_id": "cli_corp",
                            "app_secret": "sec_corp",
                        },
                    },
                },
            ),
        },
    )

    source = SessionSource(
        platform=Platform.FEISHU,
        account_id="corp",
        chat_id="oc_group",
        chat_name="Team CLAIRE",
        chat_type="group",
        user_id="ou_owner",
        user_name="Ethan",
    )

    assert runner._has_home_channel_for_source(source) is True


def test_has_home_channel_for_feishu_account_accepts_legacy_account_scoped_fallback():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                extra={
                    "accounts": {
                        "corp": {
                            "app_id": "cli_corp",
                            "app_secret": "sec_corp",
                            "home_channel": {
                                "chat_id": "oc_dm_home",
                                "name": "Owner DM",
                            },
                        },
                    },
                },
            ),
        },
    )

    source = SessionSource(
        platform=Platform.FEISHU,
        account_id="corp",
        chat_id="oc_group",
        chat_name="Team CLAIRE",
        chat_type="group",
        user_id="ou_owner",
        user_name="Ethan",
    )

    assert runner._has_home_channel_for_source(source) is True


def test_has_home_channel_for_feishu_account_does_not_fallback_to_global_env(monkeypatch):
    monkeypatch.setenv("FEISHU_HOME_CHANNEL", "oc_global_home")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                extra={
                    "accounts": {
                        "corp": {
                            "app_id": "cli_corp",
                            "app_secret": "sec_corp",
                        },
                    },
                },
            ),
        },
    )

    source = SessionSource(
        platform=Platform.FEISHU,
        account_id="corp",
        chat_id="oc_group",
        chat_name="Team CLAIRE",
        chat_type="group",
        user_id="ou_owner",
        user_name="Ethan",
    )

    assert runner._has_home_channel_for_source(source) is False


def test_get_adapter_does_not_fallback_to_platform_default_for_account_bound_source():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    default_adapter = MagicMock()
    runner.adapters = {Platform.FEISHU: default_adapter}
    runner._adapters_by_binding = {Platform.FEISHU: default_adapter}

    adapter = runner._get_adapter(Platform.FEISHU, account_id="corp")

    assert adapter is None


@pytest.mark.asyncio
async def test_sethome_bypasses_running_agent_guard():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                extra={
                    "accounts": {
                        "corp": {
                            "app_id": "cli_corp",
                            "app_secret": "sec_corp",
                        },
                    },
                },
            ),
        },
    )

    adapter = MagicMock()
    adapter.account_id = "corp"
    adapter.send = AsyncMock()
    runner.adapters = {Platform.FEISHU: adapter}
    runner._adapters_by_binding = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    source = SessionSource(
        platform=Platform.FEISHU,
        account_id="corp",
        chat_id="oc_dm_home",
        chat_name="Owner DM",
        chat_type="dm",
        user_id="ou_owner",
        user_name="Ethan",
    )
    session_key = build_session_key(source)
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.FEISHU,
        chat_type="dm",
    )

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    running_agent = SimpleNamespace(
        interrupt=MagicMock(),
        get_activity_summary=lambda: {
            "seconds_since_activity": 0,
            "last_activity_desc": "test",
            "api_call_count": 1,
            "max_iterations": 8,
        },
    )
    runner._running_agents = {session_key: running_agent}
    runner._running_agents_ts = {session_key: time()}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._handle_set_home_command = AsyncMock(return_value="sethome ok")

    event = MessageEvent(text="/sethome", source=source, message_id="om_cmd")

    result = await runner._handle_message(event)

    assert result == "sethome ok"
    runner._handle_set_home_command.assert_awaited_once_with(event)
    running_agent.interrupt.assert_not_called()
