"""Code-flow grid sweep (NOUS-368, wave 1): three pre-existing automatic gateway diagnostics that
bypassed the warning boundary. absent/false = legacy bytes; true suppresses ONLY the diagnostic;
requested outcomes (an in-chat /restart's own ack, the requested background result) are untouched."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import HomeChannel, Platform
from gateway.pairing import PairingStore
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _configure(tmp_path, monkeypatch, setting):
    home = tmp_path / f"home-{setting}"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    display = {} if setting is None else {"suppress_warning_notifications": setting}
    (home / "config.yaml").write_text(json.dumps({"display": display}))
    return home


MODES = (None, False, True)


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", MODES)
async def test_shutdown_notice_to_active_chats_and_home_channel_honors_policy(tmp_path, monkeypatch, setting):
    _configure(tmp_path, monkeypatch, setting)
    runner, adapter = make_restart_runner()
    source = make_restart_source(thread_id="42")
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._cache_session_source(session_key, source)
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id="home-chat", name="Telegram Home")

    await runner._notify_active_sessions_of_shutdown()

    chats = [c for c, _m, _meta in adapter.sent_calls]
    if setting is True:
        assert chats == [], adapter.sent_calls
    else:
        assert sorted(chats) == sorted([source.chat_id, "home-chat"])
        assert all("Hermes is shutting down" in m for _c, m, _meta in adapter.sent_calls)


@pytest.mark.asyncio
async def test_in_chat_restart_ack_to_requester_is_never_suppressed(tmp_path, monkeypatch):
    _configure(tmp_path, monkeypatch, True)
    runner, adapter = make_restart_runner()
    source = make_restart_source(thread_id="42")
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._cache_session_source(session_key, source)
    restart_source = make_restart_source(thread_id="42")
    restart_source.message_id = "restart-command"
    runner._restart_requested = True
    runner._restart_command_source = restart_source
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id="home-chat", name="Telegram Home")

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent_calls) == 1
    chat_id, message, metadata = adapter.sent_calls[0]
    assert chat_id == source.chat_id and "Hermes is restarting" in message
    assert metadata["telegram_reply_to_message_id"] == "restart-command"


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", MODES)
async def test_unauthorized_owner_hint_honors_policy_and_stranger_stays_silent(tmp_path, monkeypatch, setting):
    _configure(tmp_path, monkeypatch, setting)
    runner, adapter = make_restart_runner()
    runner.pairing_store = PairingStore()
    runner.pairing_stores = {}
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM, chat_id="home-1", name="Ops")
    runner._hm_report_ignored_dm = GatewayRunner._hm_report_ignored_dm.__get__(runner, GatewayRunner)
    stranger = SessionSource(platform=Platform.TELEGRAM, chat_id="dm-777", user_id="777", user_name="Eve", chat_type="dm")

    await runner._hm_report_ignored_dm(stranger)
    await runner._hm_report_ignored_dm(stranger)

    chats = [c for c, _m, _meta in adapter.sent_calls]
    assert "dm-777" not in chats
    assert chats.count("home-1") == (0 if setting is True else 1)
    assert runner.pairing_store.list_pending("telegram") == []


def _bg_runner():
    from tests.gateway.test_background_command import _make_runner
    return _make_runner()


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", MODES)
async def test_background_task_failure_notice_honors_policy(tmp_path, monkeypatch, setting):
    _configure(tmp_path, monkeypatch, setting)
    runner = _bg_runner()
    from gateway.platforms.base import BasePlatformAdapter
    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.send = AsyncMock()
    adapter.emit_warning = BasePlatformAdapter.emit_warning.__get__(adapter, BasePlatformAdapter)
    adapter.warning_notifications_enabled = BasePlatformAdapter.warning_notifications_enabled.__get__(adapter, BasePlatformAdapter)
    adapter.platform = Platform.TELEGRAM
    runner.adapters[Platform.TELEGRAM] = adapter
    source = SessionSource(platform=Platform.TELEGRAM, user_id="1", chat_id="c1", user_name="u")

    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "k"}), \
         patch.object(runner, "_resolve_session_agent_runtime", side_effect=RuntimeError("BG_REAL_CRASH")):
        await runner._run_background_task("do the thing", source, "bg_test")

    if setting is True:
        adapter.send.assert_not_called()
    else:
        adapter.send.assert_called_once()
        args, kwargs = adapter.send.call_args
        content = kwargs.get("content") or (args[1] if len(args) > 1 else "")
        assert "failed before finishing" in content


@pytest.mark.asyncio
async def test_background_task_requested_result_is_never_suppressed(tmp_path, monkeypatch):
    """Control: a successful background task's requested result is delivered under true."""
    _configure(tmp_path, monkeypatch, True)
    from tests.gateway import test_background_command as tbc
    # Reuse the existing success-path test body under the suppressing config.
    t = tbc.TestRunBackgroundTask()
    await t.test_successful_task_sends_result()
