"""A picker card (/model, /reasoning, /fast) and the /update Yes/No prompt answer only to their opener.

Alice (the admin, ``group_allow_admin_from``) opens the card in a shared group thread; Bob is admitted
to the chat but may not run the admin-only command. Bob's tap gets "Only Alice can answer this
approval." and the card stays open; Alice's tap applies it. Bob's typed text while Alice's /update
prompt waits is an ordinary message, not the answer.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

EXTRA = {"group_allow_admin_from": ["alice"]}
REFUSAL = "Only Alice can answer this approval."


def _source(user_id):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", thread_id="7",
                         user_id=user_id, user_name=user_id.title())


def _runner(adapter):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="x", extra=EXTRA)})
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._running_agents = {}
    runner._pending_messages = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=[])
    runner._session_db = None
    runner._delivery_adapter_for = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None
    return runner


async def _open_reasoning_picker(adapter):
    """Alice runs the admin-only bare /reasoning; returns the runner and a 'what got applied' probe."""
    runner = _runner(adapter)
    assert runner._check_slash_access(_source("bob"), "reasoning") is not None  # Bob may not run it
    assert await runner._handle_reasoning_command(MessageEvent(text="/reasoning", source=_source("alice"))) is None
    key = runner._session_key_for_source(_source("alice"))
    return lambda: runner._session_state(key).conversation.reasoning_override


async def _tap_telegram(tapper, monkeypatch):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra=EXTRA))
    adapter._bot = AsyncMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(chat_id=-100, message_id=42, message_thread_id=7))
    adapter._app = MagicMock()
    applied = await _open_reasoning_picker(adapter)
    query = AsyncMock(data="cp:5", from_user=MagicMock(first_name=tapper.title(), id=tapper),
                      message=MagicMock(chat_id=-100, message_id=42, message_thread_id=7, chat=MagicMock(type="supergroup")))
    await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())
    alerts = [c.kwargs.get("text") for c in query.answer.call_args_list if c.kwargs.get("show_alert")]
    return applied(), alerts, "-100" in adapter._choice_picker_state


async def _tap_discord(tapper, monkeypatch):
    import plugins.platforms.discord.adapter as discord_mod

    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    sent = {}

    class _Adapter:  # the gateway's send_choice_picker gate checks the adapter TYPE
        async def send_choice_picker(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(success=True)

    applied = await _open_reasoning_picker(_Adapter())
    view = discord_mod.ChoicePickerView(choices=sent["choices"], on_choice_selected=sent["on_choice_selected"],
                                        allowed_user_ids=set(), allowed_role_ids=set())
    view.answer_extra = EXTRA  # bound by the adapter's _send_prompt
    response = SimpleNamespace(send_message=AsyncMock(), defer=AsyncMock(), edit_message=AsyncMock())
    await view._on_select(SimpleNamespace(user=SimpleNamespace(id=tapper, display_name=tapper), data={"values": ["xhigh"]},
                                          channel_id=1, guild=object(), channel=object(), response=response))
    alerts = [c.args[0] for c in response.send_message.call_args_list if c.kwargs.get("ephemeral")]
    return applied(), alerts, not view.resolved


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["telegram", "discord"])
@pytest.mark.parametrize("tapper", ["bob", "alice"])
async def test_admin_picker_answers_only_its_opener(platform, tapper, tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    tap = _tap_telegram if platform == "telegram" else _tap_discord
    applied, alerts, still_open = await tap(tapper, monkeypatch)
    if tapper == "bob":
        assert applied is None and alerts == [REFUSAL] and still_open
    else:
        assert applied == {"enabled": True, "effort": "xhigh"} and alerts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["typed", "telegram_button"])
@pytest.mark.parametrize("sender", ["bob", "alice"])
async def test_update_prompt_answers_only_its_opener(surface, sender, tmp_path, monkeypatch):
    """Alice's /update recorded her as the opener; only her reply (or an admin's) answers its prompt."""
    runner = _runner(MagicMock(send=AsyncMock()))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "checkout" / "gateway").mkdir(parents=True)
    (tmp_path / "checkout" / ".git").mkdir()
    runner._schedule_update_notification_watch = lambda: None
    with patch("gateway.slash_commands.project_root_str", lambda: str(tmp_path / "checkout")), \
            patch("gateway.slash_commands._spawn_detached_update"), \
            patch("gateway.run._resolve_hermes_bin", return_value=["hermes"]), \
            patch("hermes_cli.config.is_managed", return_value=False), \
            patch("hermes_cli.config.detect_install_method", return_value="git"):
        assert await runner._handle_update_command(MessageEvent(text="/update", source=_source("alice")))
    key = runner._session_key_for_source(_source("alice"))
    runner._session_state(key).persistent.update_prompt_pending = True
    response = tmp_path / ".update_response"

    if surface == "typed":
        typed = MessageEvent(text="yes", source=_source(sender))
        with patch("agent.estop.paused_reply", return_value="PAUSED"):  # `hermes pause` is on
            # Only the opener's reply is the prompt's answer; anyone else's is a new turn, still paused.
            assert (runner._hm_estop_gate(typed, typed.source, False) is None) is (sender == "alice")
        reply = runner._hm_update_prompt_reply(typed, key)
        assert (reply is not None) is (sender == "alice")
    else:
        from plugins.platforms.telegram.adapter import TelegramAdapter

        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra=EXTRA))
        query = AsyncMock(data="update_prompt:y", from_user=MagicMock(first_name=sender.title(), id=sender),
                          message=MagicMock(chat_id=-100, message_id=43, message_thread_id=7, chat=MagicMock(type="supergroup")))
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), MagicMock())
        if sender == "bob":
            assert query.answer.call_args.kwargs == {"text": REFUSAL, "show_alert": True}
    assert response.exists() is (sender == "alice")
    assert runner._session_state(key).persistent.update_prompt_pending is (surface == "telegram_button" or sender == "bob")


@pytest.mark.asyncio
async def test_second_update_while_one_runs_keeps_the_first_opener(tmp_path, monkeypatch):
    """Bob's /update while Alice's is still running is refused before it touches the marker, so
    Alice stays the only one (besides admins) who answers the running update's prompt."""
    from gateway.approval_actor import update_prompt_refusal

    runner = _runner(MagicMock(send=AsyncMock()))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "checkout" / "gateway").mkdir(parents=True)
    (tmp_path / "checkout" / ".git").mkdir()
    runner._schedule_update_notification_watch = lambda: None
    with patch("gateway.slash_commands.project_root_str", lambda: str(tmp_path / "checkout")), \
            patch("gateway.slash_commands._spawn_detached_update") as spawn, \
            patch("gateway.run._resolve_hermes_bin", return_value=["hermes"]), \
            patch("hermes_cli.config.is_managed", return_value=False), \
            patch("hermes_cli.config.detect_install_method", return_value="git"):
        await runner._handle_update_command(MessageEvent(text="/update", source=_source("alice")))
        reply = await runner._handle_update_command(MessageEvent(text="/update", source=_source("bob")))
    assert spawn.call_count == 1 and "already running" in reply
    assert json.loads((tmp_path / ".update_pending.json").read_text())["user_id"] == "alice"
    assert update_prompt_refusal("alice", extra={}) is None
    assert update_prompt_refusal("bob", extra={}) == "Alice"


def test_clarify_with_an_empty_authority_answers_to_anyone():
    """A clarify registered with ``authority={}`` records no restriction, so any admitted sender's
    reply answers it (a plugin's rewritten voice-note reply relies on this)."""
    from gateway.approval_actor import clarify_refusal
    from tools import clarify_gateway

    clarify_gateway.register("cl-open", "agent:main:telegram:dm:1", "Which one?", None, authority={})
    try:
        assert clarify_refusal("cl-open", "bob", extra=EXTRA) is None
    finally:
        clarify_gateway.clear_session("agent:main:telegram:dm:1")
