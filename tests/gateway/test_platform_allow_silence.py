"""``platforms.<platform>.allow_silence`` is the parent of every room's ``allow_silence``.

A room entry that sets ``allow_silence`` (true or false) wins along chat -> thread -> parent; an
entry that sets only other keys inherits; otherwise the platform value applies. Driven through the
real config loader and the real turn path (live, queued first response, crash recovery), with the
real Slack, WhatsApp and relay inbound builders where they decide ``reply_expected``.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.run_turn import _unexpected_silence_reply
from gateway.session import SessionSource
from gateway.session_identity import RoutingIdentity
from hermes_constants import get_hermes_home
from tests.gateway.test_active_turn_recovery import _close_store_db, _db_runner, _entry_for
from tests.gateway.test_gateway_silence_tokens import _runner
from tests.gateway.test_slack_ignore_other_user_mentions import BOT_USER_ID, CHANNEL_ID

THREAD = "1700000000.000010"
WA_GROUP = "120363001234567890@g.us"


def _write(home, raw):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps(raw), encoding="utf-8")


def _loaded(raw):
    """The launch profile's config exactly as the gateway loads it."""
    _write(get_hermes_home(), raw)
    return load_gateway_config()


def _slack_source(chat_id=CHANNEL_ID, thread_id=THREAD):
    return SessionSource(platform=Platform.SLACK, chat_id=chat_id, chat_type="group",
                         user_id="U_HUMAN", thread_id=thread_id)


def _wa_source(chat_id=WA_GROUP):
    return SessionSource(platform=Platform.WHATSAPP, chat_id=chat_id, chat_type="group",
                         user_id="6281234567890@s.whatsapp.net")


def _result(**changes):
    return {"final_response": "NO_REPLY", "failed": False, "api_calls": 1, "messages": [], **changes}


async def _live(runner, event, result=None):
    runner._run_agent = AsyncMock(return_value=result or _result())
    return await runner._handle_message_with_agent(event, event.source, "lane", 1)


async def _queued(runner, source, result=None, reply_expected=None):
    """Return the text the queued first-response branch would send, or None when it stays silent."""
    result = result or _result()
    runner._deliver_queued_first_response = AsyncMock(return_value=True)
    ctx = SimpleNamespace(session_key="lane", source=source, stream_consumer_holder=[None],
                          mute_notification_reply=False, persist_user_display_kind=None,
                          reply_expected=reply_expected, _status_thread_metadata=None,
                          event_message_id=None, inbound_message_id="42", run_generation=1)
    await runner._run_agent_deliver_first_response(ctx, None, result, result, None)
    calls = runner._deliver_queued_first_response.await_args_list
    return calls[0].args[0] if calls else None


async def _recovered(runner, event, tmp_path, platform):
    """Finish a real live turn, crash before delivery, restart: what recovery owes the chat."""
    from gateway.delivery_ledger import sweep_recoverable
    _, store = _db_runner(tmp_path)
    runner.session_store = store
    runner._run_agent = AsyncMock(return_value=_result(agent_persisted=False, messages=[
        {"role": "user", "content": "side chatter"}, {"role": "assistant", "content": "NO_REPLY"}]))
    try:
        await runner._handle_message_with_agent(event, event.source, store._generate_session_key(event.source), 1)
        _close_store_db(store)
        recovered, store = _db_runner(tmp_path)
        await recovered._recover_unclean_sessions()
        owed = sweep_recoverable(deliverable_platforms={platform})
        assert _entry_for(store, event.source).active_turn_token is None
        return owed
    finally:
        _close_store_db(store)


def _event(source, reply_expected=None):
    from gateway.platforms.base import MessageEvent
    return MessageEvent(text="side chatter", source=source, message_id="m1", reply_expected=reply_expected)


@pytest.fixture
def slack_adapter(tmp_path, monkeypatch):
    from plugins.platforms.slack.adapter import SlackAdapter
    from gateway.config import PlatformConfig
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache")
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = BOT_USER_ID
    adapter._running = True
    adapter.handle_message = AsyncMock()
    return adapter


async def _slack_followup_event(adapter):
    """The real Slack adapter's event for an unmentioned reply in a thread the bot is part of."""
    adapter._mentioned_threads.add(THREAD)
    message = {"channel": CHANNEL_ID, "channel_type": "channel", "user": "U_HUMAN",
               "text": "did you two sort the deck?", "ts": "1.4", "thread_ts": THREAD}
    with patch.object(adapter, "_resolve_user_name", new=AsyncMock(return_value="human")), \
            patch.object(adapter, "_fetch_thread_context", new=AsyncMock(return_value=None)), \
            patch.object(adapter, "_fetch_thread_parent_text", new=AsyncMock(return_value="")), \
            patch.object(adapter, "_has_active_session_for_thread", return_value=False):
        await adapter._handle_slack_message(message)
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_expected is None and event.source.platform is Platform.SLACK
    return event


async def _whatsapp_group_event():
    """The real WhatsApp adapter's event for group chatter in a free-response group."""
    from tests.gateway.test_whatsapp_group_gating import _group_message, _make_adapter
    wa = _make_adapter(require_mention=False, group_policy="open")
    wa.gateway_runner = None
    wa._owner_profile = None
    event = await wa._build_message_event(_group_message("did anyone book the venue"))
    assert event is not None and event.reply_expected is None
    assert event.source.platform is Platform.WHATSAPP and event.source.chat_id == WA_GROUP
    return event


# --- the customer paths ---------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["platforms", "top-level"])
@pytest.mark.parametrize("setting,silent", [(True, True), (False, False), (None, False)])
async def test_slack_thread_followup_follows_the_platform_setting(
        monkeypatch, tmp_path, slack_adapter, layout, setting, silent):
    event = await _slack_followup_event(slack_adapter)
    block = {} if setting is None else {"allow_silence": setting}
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"platforms": {"slack": block}} if layout == "platforms" else {"slack": block})

    response = await _live(runner, event)

    assert response == ("" if silent else _unexpected_silence_reply())


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["platforms", "top-level"])
@pytest.mark.parametrize("setting,silent", [(True, True), (False, False), (None, False)])
async def test_whatsapp_group_chatter_follows_the_platform_setting(monkeypatch, tmp_path, layout, setting, silent):
    event = await _whatsapp_group_event()
    block = {} if setting is None else {"allow_silence": setting}
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"platforms": {"whatsapp": block}} if layout == "platforms" else {"whatsapp": block})

    response = await _live(runner, event)

    assert response == ("" if silent else _unexpected_silence_reply())


@pytest.mark.asyncio
async def test_relay_fronted_slack_turn_uses_the_slack_platform_setting(monkeypatch, tmp_path):
    """Relay inbound carries the UNDERLYING platform, so ``platforms.slack`` governs it."""
    from gateway.relay.ws_transport import _event_from_wire
    event = _event_from_wire({"text": "lol same", "message_id": "m9", "source": {
        "platform": "slack", "chat_id": CHANNEL_ID, "chat_type": "group", "thread_id": THREAD,
        "user_id": "U_HUMAN"}})
    assert event.source.platform is Platform.SLACK and event.source.delivered_via_upstream_relay
    assert event.reply_expected is None
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"platforms": {"relay": {"enabled": True}, "slack": {"allow_silence": True}}})

    assert await _live(runner, event) == ""


@pytest.mark.asyncio
async def test_relay_block_does_not_opt_in_the_chats_platform(monkeypatch, tmp_path):
    """The relay's own block is not the chat's platform: it does not opt Slack chats in."""
    from gateway.relay.ws_transport import _event_from_wire
    event = _event_from_wire({"text": "lol same", "message_id": "m9", "source": {
        "platform": "slack", "chat_id": CHANNEL_ID, "chat_type": "group", "thread_id": THREAD}})
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"platforms": {"relay": {"enabled": True, "allow_silence": True}}})

    assert await _live(runner, event) == _unexpected_silence_reply()


# --- resolution order -------------------------------------------------------------------------


@pytest.mark.parametrize("platform_value,overrides,expected", [
    (True, {CHANNEL_ID: {"allow_silence": False}}, False),
    (True, {CHANNEL_ID: {"model": "openai/gpt-5-mini"}}, True),
    (True, {THREAD: {"system_prompt": "be brief"}, CHANNEL_ID: {"allow_silence": False}}, False),
    (None, {THREAD: {"model": "m"}, "C_PARENT": {"allow_silence": True}}, True),
    (False, {CHANNEL_ID: {"allow_silence": True}}, True),
    (True, {CHANNEL_ID: {"allow_silence": None}}, True),
    (None, {CHANNEL_ID: {"model": "m"}}, None),
    (None, {}, None),
], ids=["channel-false-beats-platform-true", "model-only-entry-inherits", "chat-entry-wins-over-thread",
        "model-only-thread-inherits-parent", "channel-true-beats-platform-false", "null-inherits",
        "unconfigured-with-model-entry", "unconfigured"])
def test_first_entry_that_sets_allow_silence_wins_then_the_platform(
        monkeypatch, tmp_path, platform_value, overrides, expected):
    block = {"channel_overrides": overrides}
    if platform_value is not None:
        block["allow_silence"] = platform_value
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"platforms": {"slack": block}})
    source = _slack_source()
    source.parent_chat_id = "C_PARENT"

    assert runner._room_allow_silence(source) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id,silent", [(WA_GROUP, False), ("120363000000000001@g.us", True),
                                            ("120363000000000002@g.us", True)],
                         ids=["explicit-false", "model-only-entry", "no-entry"])
async def test_whatsapp_group_entries_under_a_platform_true(monkeypatch, tmp_path, chat_id, silent):
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"whatsapp": {"allow_silence": True, "channel_overrides": {
        WA_GROUP: {"allow_silence": False}, "120363000000000001@g.us": {"model": "openai/gpt-5-mini"}}}})

    response = await _live(runner, _event(_wa_source(chat_id)))

    assert response == ("" if silent else _unexpected_silence_reply())


# --- unchanged guards -------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["normal", "queued"])
@pytest.mark.parametrize("changes", [
    {"failed": True}, {"partial": True}, {"interrupted": True},
    {"completed": False}, {"error": "provider error"}, {"final_response": ""},
])
async def test_platform_opt_in_does_not_hide_unhealthy_or_empty_results(monkeypatch, tmp_path, lane, changes):
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"platforms": {"slack": {"allow_silence": True}}})
    result = _result(**changes)
    if lane == "normal":
        assert await _live(runner, _event(_slack_source()), result)
    else:
        from gateway.run import _normalize_empty_agent_response
        result = {**result, "final_response": _normalize_empty_agent_response(result, result["final_response"])}
        assert await _queued(runner, _slack_source(), result, reply_expected=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("setting,sent", [(True, None), (None, "warning")])
async def test_queued_first_response_follows_the_platform_setting(monkeypatch, tmp_path, setting, sent):
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"platforms": {"whatsapp": {} if setting is None else {"allow_silence": setting}}})

    text = await _queued(runner, _wa_source())

    assert text == (None if sent is None else _unexpected_silence_reply())


@pytest.mark.asyncio
@pytest.mark.parametrize("setting,owed", [(True, False), (None, True)])
async def test_crash_recovery_honours_the_platform_decision(monkeypatch, tmp_path, setting, owed):
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"platforms": {"slack": {} if setting is None else {"allow_silence": setting}}})

    assert bool(await _recovered(runner, _event(_slack_source()), tmp_path, "slack")) is owed


@pytest.mark.asyncio
@pytest.mark.parametrize("source,reply_expected,silent", [
    (_slack_source(), None, False), (_slack_source(), True, False), (_wa_source(), None, False),
    (_slack_source(), False, True),
], ids=["slack-unknown", "slack-addressed", "whatsapp-unknown", "slack-unaddressed"])
async def test_unconfigured_profile_keeps_main_behaviour(monkeypatch, tmp_path, source, reply_expected, silent):
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"platforms": {"slack": {"channel_overrides": {CHANNEL_ID: {"model": "m"}}}}})

    response = await _live(runner, _event(source, reply_expected=reply_expected))

    assert response == ("" if silent else _unexpected_silence_reply())


# --- routed / multiplexed profiles ------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("served_on,launch_on,silent", [(True, False, True), (False, True, False)])
async def test_routed_profile_reads_its_own_platform_setting(monkeypatch, tmp_path, served_on, launch_on, silent):
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded({"slack": {"allow_silence": True} if launch_on else {}})
    runner.config.multiplex_profiles = True
    served = tmp_path / "served"
    _write(served, {"slack": {"allow_silence": True} if served_on else {}})
    source = _slack_source()
    source._identity = RoutingIdentity("default", "served", tmp_path, served)

    response = await _live(runner, _event(source))

    assert response == ("" if silent else _unexpected_silence_reply())


@pytest.mark.parametrize("raw", [
    {"platforms": {"slack": {"allow_silence": True}}},
    {"slack": {"allow_silence": True}},
    {"gateway": {"platforms": {"slack": {"allow_silence": True}}}},
    {"gateway": {"slack": {"allow_silence": True}}},
    {"platforms": {"slack": {"extra": {"allow_silence": True}}}},
    {"slack": {"allow_silence": False}, "platforms": {"slack": {"allow_silence": True}}},
    {"slack": {"require_mention": True}, "platforms": {"slack": {"allow_silence": True}}},
    {"slack": {"allow_silence": True, "channel_overrides": {CHANNEL_ID: {"model": "m"}}},
     "platforms": {"slack": {"channel_overrides": {CHANNEL_ID: {"allow_silence": False}}}}},
    {"platforms": {"slack": {"allow_silence": "yes"}}},
    {"slack": {"allow_silence": False}, "platforms": {"slack": {"extra": {"allow_silence": True}}}},
], ids=["nested", "top-level", "gateway-platforms", "gateway-platform", "explicit-extra",
        "top-level-wins", "top-level-without-the-key", "top-level-overrides-win", "non-bool",
        "authored-extra-beats-top-level"])
def test_routed_lookup_agrees_with_the_gateway_loader(monkeypatch, tmp_path, raw):
    runner = _runner(monkeypatch, tmp_path)
    runner.config = _loaded(raw)
    primary = runner._room_allow_silence(_slack_source())
    served = tmp_path / "served"
    _write(served, raw)
    source = _slack_source()
    source._identity = RoutingIdentity("default", "served", tmp_path, served)
    runner.config = GatewayConfig()

    assert runner._room_allow_silence(source) is primary
