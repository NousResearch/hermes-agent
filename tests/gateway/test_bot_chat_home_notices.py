"""Gateway notices reach a profile's Bot Chat when it is the home (``gateway.home: bot-chat``).

Through the real authorities, Bot Chat door and SQLite on a multiplexed gateway with two profiles; only
the model call is stubbed, and it must never run: a notice is written into the Bot Chat as a note. The
profiles' Telegram home channels, which these notices went to before, receive nothing.
"""
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.run import GatewayRunner
from tests.waiting import async_wait_until


class _Telegram:
    """The Telegram adapter both profiles' home channels resolve to: records every send."""

    def __init__(self):
        self.sent = []

    async def send(self, chat_id, content, **_kwargs):
        self.sent.append((chat_id, content))
        return SimpleNamespace(success=True, message_id="1")


def _bot_chat_home_config(chat_id):
    config = GatewayConfig()
    config.home = "bot-chat"
    config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True, token="***", home_channel=HomeChannel(Platform.TELEGRAM, chat_id, "Home"))
    return config


@pytest_asyncio.fixture
async def gateway(tmp_path, monkeypatch):
    from pathlib import Path
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    import hermes_state

    root = tmp_path / ".hermes"
    homes = (("default", root), ("beta", root / "profiles" / "beta"))
    for name, home in homes:
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        (home / "config.yaml").write_text(json.dumps({"model": {
            "default": f"{name}-model", "provider": "custom", "base_url": "http://127.0.0.1:9/v1",
            "api_key": "loopback-only"}}))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    turns = []

    async def model(runner, message, *_args, **_kwargs):
        turns.append(message)
        return {"final_response": "ack", "messages": [], "api_calls": 1, "tools": []}

    monkeypatch.setattr(GatewayRunner, "_run_agent", model)
    config = _bot_chat_home_config("111")
    config.multiplex_profiles = True
    config._runtime_profile_homes = homes
    runner = GatewayRunner(config)
    telegram = _Telegram()
    runner.adapters = {Platform.TELEGRAM: telegram}
    # What adapter start leaves for a served secondary: its own config and bot.
    runner._served_profile_homes = {"beta": homes[1][1]}
    runner._profile_configs = {"beta": _bot_chat_home_config("222")}
    runner._profile_adapters = {"beta": {Platform.TELEGRAM: telegram}}
    process_ownership.reserve([home for _, home in homes])
    await initialize_gateway_runtime(runner)
    authorities = {name: runner.session_authorities.for_home(home) for name, home in homes}
    try:
        yield SimpleNamespace(runner=runner, authorities=authorities, telegram=telegram, turns=turns, root=root)
    finally:
        for authority in runner.session_authorities:
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


def _bot_chat_text(authority):
    row = authority.db.get_session_by_title("Bot Chat")
    if row is None:
        return ""
    return "\n".join(str(m.get("content")) for m in authority.db.get_messages(authority.db.get_compression_tip(row["id"])))


async def _noted(gateway, text):
    for name, authority in gateway.authorities.items():
        await async_wait_until(lambda: text in _bot_chat_text(authority), timeout=30, what=f"the note in {name}'s Bot Chat")
    assert gateway.turns == [], "a notice must be a note, never a model turn"
    assert gateway.telegram.sent == [], "the Telegram home channels are replaced"


@pytest.mark.asyncio
async def test_the_planned_restart_notice_is_a_note_in_each_profiles_bot_chat(gateway, monkeypatch):
    from agent.i18n import t
    from gateway import run as gateway_run

    marker = gateway.root / ".restart_pending.json"
    marker.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", gateway.root)

    await gateway.runner._replay_pending_planned_restart_notification()

    await _noted(gateway, t("gateway.startup.online"))
    assert not marker.exists()


@pytest.mark.asyncio
async def test_a_notice_reaches_a_bot_chat_that_already_exists(gateway):
    from gateway.bot_chat_home import post_note

    assert await post_note(gateway.root, "First notice")
    assert await post_note(gateway.root, "Second notice")

    authority = gateway.authorities["default"]
    await async_wait_until(lambda: "Second notice" in _bot_chat_text(authority), timeout=30,
                           what="the second notice in the existing Bot Chat")
    assert "First notice" in _bot_chat_text(authority)


@pytest.mark.asyncio
async def test_the_shutdown_ping_skips_the_replaced_home_channels(gateway):
    from gateway.run_shutdown_home import broadcast_home_shutdown_notice

    gateway.runner._draining = True
    await broadcast_home_shutdown_notice(gateway.runner, "Gateway shutting down for a test", set())

    assert gateway.telegram.sent == []
    assert all("shutting down" not in _bot_chat_text(a) for a in gateway.authorities.values())


@pytest.mark.asyncio
async def test_an_unauthorized_sender_notice_reaches_the_bot_chat_home(gateway):
    from gateway.run_inbound_unauthorized import UnauthorizedOwnerNotifier
    from gateway.session import SessionSource

    stranger = SessionSource(platform=Platform.TELEGRAM, chat_id="999", user_id="999", chat_type="dm")
    await UnauthorizedOwnerNotifier().notify(gateway.runner, stranger, "Stranger 999 was ignored")

    authority = gateway.authorities["default"]
    await async_wait_until(lambda: "Stranger 999 was ignored" in _bot_chat_text(authority), timeout=30,
                           what="the notice in the launch profile's Bot Chat")
    assert gateway.turns == [] and gateway.telegram.sent == []
