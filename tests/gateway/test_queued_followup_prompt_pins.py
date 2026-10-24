"""Queued turns must execute and persist one prompt-input generation (#140257)."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource, SessionStore, build_session_context, build_session_context_prompt
from gateway.turn_context import TurnContext


@pytest.fixture
def prompt_turn(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    privacy = {"redact_pii": False}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"privacy": privacy})
    monkeypatch.setattr("gateway.session._slack_tools_loaded", lambda: False)
    config = GatewayConfig()
    stores = []

    def make_runner():
        runner = object.__new__(GatewayRunner)
        runner.config = config
        runner.session_store = SessionStore(tmp_path / "sessions", config)
        stores.append(runner.session_store)
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="queued")
        runner._run_agent_deliver_first_response = AsyncMock()
        runner._refresh_agent_cache_message_count = AsyncMock()
        runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": []})
        runner._delivery_adapter_for = lambda source: None
        runner._intake_adapter_for = lambda source: None
        return runner

    yield SimpleNamespace(make_runner=make_runner, privacy=privacy, config=config)
    for store in stores:
        if store._db is not None:
            store._db.close()


async def _seed_turn(runner, source):
    entry = runner.session_store.get_or_create_session(source)
    key = entry.session_key
    prompt = "" if source.platform == Platform.LOCAL else runner._pinned_session_context_prompt(
        build_session_context(source, runner.config, entry), False, key)
    runner._pinned_channel_inputs(key, "Channel A", source, internal=False)
    await runner._persist_prompt_pins(key, entry.session_id)
    return TurnContext(source=source, context_prompt=prompt, channel_prompt="Channel A",
                       session_key=key, session_id=entry.session_id, run_generation=1, history=[])


async def _followup(runner, turn, event, *, session_id=None, projected=False):
    result = {"final_response": "done", "messages": [], "session_id": None if projected else session_id}
    response = {"session_id": session_id} if projected else "done"
    await runner._run_agent_queued_followup(turn, None, "queued", event, response, result, None)
    return runner._run_agent.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["chat_topic", "chat_name", "user_name", "privacy"])
@pytest.mark.parametrize("compressed", [False, "raw", "projected"],
                         ids=["same-session", "raw-child", "projected-child"])
async def test_queued_human_executes_and_restores_the_same_new_prompt_pair(
    prompt_turn, change, compressed, monkeypatch,
):
    source = SessionSource(platform=Platform.TELEGRAM if change == "privacy" else Platform.SLACK,
                           chat_id="chat-123", user_id="user-456",
                           chat_name=None if change == "privacy" else "Ops A",
                           user_name=None if change == "privacy" else "Alice",
                           chat_topic="Topic A", chat_type="group")
    runner = prompt_turn.make_runner()
    turn = await _seed_turn(runner, source)
    live_session_id = turn.session_id
    if compressed:
        runner._session_db = None
        turn._run_still_current = lambda: True
        turn.agent_holder[0] = SimpleNamespace(session_id=turn.session_id + "-compressed")
        # Compression publishes a child route while TurnContext keeps the launch id.
        _, live_session_id, _ = TurnRunner(runner, turn)._sync_session_after_run([])
        assert runner.session_store.peek_session_id(turn.session_key) == live_session_id
        assert live_session_id != turn.session_id
    next_source = source if change == "privacy" else replace(source, **{change: "Updated B"})
    prompt_turn.privacy["redact_pii"] = change == "privacy"
    context = build_session_context(next_source, prompt_turn.config)
    expected = build_session_context_prompt(context, redact_pii=change == "privacy")
    assert expected != turn.context_prompt
    event = MessageEvent(text="queued", source=next_source, message_id="next", channel_prompt="Channel B")

    seen = await _followup(runner, turn, event, session_id=live_session_id, projected=compressed == "projected")
    durable = runner.session_store.get_prompt_pin(turn.session_key, expected_session_id=live_session_id)
    expected_key = runner._ephemeral_change_key(context, change == "privacy")
    assert (seen["context_prompt"], seen["channel_prompt"], durable["context_prompt"],
            durable["channel_prompt"], durable["context_key"], durable["redact_pii"]) == (
                expected, "Channel B", expected, "Channel B", expected_key, change == "privacy")
    assert seen["session_id"] == live_session_id

    # A new process reads the real routing store, rather than an in-memory snapshot double.
    restored = prompt_turn.make_runner()
    await restored._rehydrate_prompt_pins(turn.session_key, live_session_id)
    state = restored._peek_session_state(turn.session_key).conversation
    assert state.ephemeral_pin == (expected_key, expected, change == "privacy")
    assert state.channel_pin == ("Channel B", next_source.parent_chat_id)

    # The next idle human is a cache hit: it must not render the change one turn late.
    def unexpected_render(*args, **kwargs):
        pytest.fail("queued turn left a stale context key for the next human turn")

    monkeypatch.setattr("gateway.run_agent_cache.build_session_context_prompt", unexpected_render)
    assert restored._pinned_session_context_prompt(
        context, change == "privacy", turn.session_key) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["internal", "inherit", "eventless", "local"])
async def test_continuations_keep_their_established_prompt_contract(prompt_turn, kind):
    source = SessionSource(platform=Platform.LOCAL if kind == "local" else Platform.SLACK,
                           chat_id="chat-123", user_id="user-456", chat_name="Ops A",
                           user_name="Alice", chat_topic="Topic A", parent_chat_id="parent")
    runner = prompt_turn.make_runner()
    turn = await _seed_turn(runner, source)
    event = None if kind == "eventless" else MessageEvent(
        text="queued", source=replace(source, chat_name=None, user_name=None, chat_topic=None,
                                      parent_chat_id=None),
        channel_prompt="Channel B" if kind == "local" else None,
        internal=kind == "internal", inherit_channel_pin=kind == "inherit")

    seen = await _followup(runner, turn, event)
    assert seen["context_prompt"] == turn.context_prompt
    assert seen["channel_prompt"] == ("Channel B" if kind == "local" else "Channel A")
    if kind in {"internal", "inherit"}:
        assert seen["source"].parent_chat_id == "parent"
        durable = runner.session_store.get_prompt_pin(turn.session_key, expected_session_id=turn.session_id)
        assert durable["context_prompt"] == turn.context_prompt
        assert durable["channel_prompt"] == "Channel A"
