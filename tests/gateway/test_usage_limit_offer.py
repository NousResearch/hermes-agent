"""Messaging: after a usage-window refusal the gateway offers the plan's recommended model as a choice
button on platforms that already have them, else one line of text. Never a
credits button (anyone in the room could spend the owner's credits) and never a switch on its own."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform

LIMIT = {
    "gauge_id": "day", "scope": "all", "limit_name": "daily", "resets_at_ms": 4_000_000_000_000,
    "message": "You've hit your daily limit · resets 15:56",
    "suggestions": [{"model": "example/model-free:free", "label": "Model Free", "note": "free", "provider": "nous"}],
    "credits": {"action": "add_credits"},
}


class _TextAdapter:
    def __init__(self):
        self.send = AsyncMock()


class _ButtonAdapter(_TextAdapter):
    async def send_choice_picker(self, **kw):  # presence of the method is the platform gate
        raise AssertionError("routed through _try_send_choice_picker")


def _runner(adapter, *, prefix="/", picker_ok=True):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    runner._typed_command_prefix_for = lambda platform: prefix
    runner._session_key_for_source = lambda source: "chat:c1"
    runner._reply_metadata = lambda event: None
    runner._try_send_choice_picker = AsyncMock(return_value=picker_ok)
    return runner


def _event():
    return SimpleNamespace(source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="c1"),
                           raw_message=None, channel_prompt=None)


@pytest.mark.asyncio
async def test_a_button_platform_gets_one_choice_per_suggestion():
    adapter = _ButtonAdapter()
    runner = _runner(adapter)
    await runner._offer_usage_limit_switch(_event(), LIMIT)
    kw = runner._try_send_choice_picker.await_args.kwargs
    assert kw["title"] == "Keep going on another model:"
    assert kw["choices"] == [{"value": "example/model-free:free", "label": "Switch to Model Free (free)", "is_current": False}]
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_picker_falls_back_to_the_text_line():
    adapter = _ButtonAdapter()
    runner = _runner(adapter, picker_ok=False)
    await runner._offer_usage_limit_switch(_event(), LIMIT)
    text = adapter.send.await_args.args[1]
    assert text == "Switch to Model Free (free) with /model, or add credits in the Portal."


@pytest.mark.asyncio
async def test_a_text_platform_gets_one_line_with_its_own_prefix_and_no_topup():
    adapter = _TextAdapter()
    runner = _runner(adapter, prefix="!")
    await runner._offer_usage_limit_switch(_event(), {**LIMIT, "credits": {"action": "use_credits"}})
    runner._try_send_choice_picker.assert_not_awaited()
    text = adapter.send.await_args.args[1]
    assert text.startswith("Switch to Model Free (free) with !model")
    assert "Portal" in text and "/topup" not in text


@pytest.mark.asyncio
async def test_no_suggestion_still_says_how_to_change_model():
    adapter = _ButtonAdapter()
    runner = _runner(adapter)
    await runner._offer_usage_limit_switch(_event(), {**LIMIT, "suggestions": [], "credits": None})
    runner._try_send_choice_picker.assert_not_awaited()
    assert adapter.send.await_args.args[1] == "Pick another model with /model."


@pytest.mark.asyncio
async def test_the_offer_never_raises():
    runner = _runner(None)
    await runner._offer_usage_limit_switch(_event(), LIMIT)  # no adapter: nothing sent, no error
    await runner._offer_usage_limit_switch(SimpleNamespace(source=None), LIMIT)


def test_the_offer_waits_for_the_reply_to_land():
    """Registered as the adapter's post-delivery callback, so the button sits under the reply."""
    registered = {}

    class _Adapter(_ButtonAdapter):
        def register_post_delivery_callback(self, key, cb, *, generation=None):
            registered.update(key=key, cb=cb, generation=generation)

    runner = _runner(_Adapter())
    runner._track_background_task = lambda coro: (_ for _ in ()).throw(AssertionError("not deferred"))
    runner._schedule_usage_limit_offer(_event(), LIMIT, session_key="chat:c1", generation=7)
    assert registered["key"] == "chat:c1" and registered["generation"] == 7 and callable(registered["cb"])


@pytest.mark.asyncio
async def test_a_tap_switches_this_session_only_then_resends_like_retry(monkeypatch):
    from gateway import slash_commands_model as scm

    adapter = _ButtonAdapter()
    adapter.handle_message = AsyncMock()
    runner = _runner(adapter)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._session_model_overrides = {}
    runner._normalize_source_for_session_key = lambda source: source
    seen = {}

    async def _perform(ctx, target, provider, source):
        seen.update(target=target, provider=provider, persist_global=ctx.persist_global)
        return SimpleNamespace(new_model=target), None

    async def _commit(result, ctx, *, source, picker):
        seen["picker"] = picker
        runner._session_model_overrides[ctx.session_key] = {"model": result.new_model, "provider": "nous"}
        return "✓ Model switched to Model Free"

    tasks = []
    runner._perform_model_switch = _perform
    runner._commit_model_switch = _commit
    runner._track_background_task = lambda coro: tasks.append(coro)
    monkeypatch.setattr(scm._ModelSwitchContext, "read_config", lambda self: None)
    monkeypatch.setattr(scm._ModelSwitchContext, "apply_override", lambda self, override: None)

    reply = await runner._accept_usage_limit_switch(_event(), "example/model-free:free")
    assert reply.startswith("✓")
    assert seen == {"target": "example/model-free:free", "provider": "nous", "persist_global": False, "picker": True}
    assert len(tasks) == 1
    await tasks[0]
    resent = adapter.handle_message.await_args.args[0]
    assert resent.text == "/retry"


@pytest.mark.asyncio
async def test_a_failed_switch_does_not_resend(monkeypatch):
    from gateway import slash_commands_model as scm

    adapter = _ButtonAdapter()
    runner = _runner(adapter)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._session_model_overrides = {}
    runner._normalize_source_for_session_key = lambda source: source
    runner._perform_model_switch = AsyncMock(return_value=(None, "❌ Model not available"))
    tasks = []
    runner._track_background_task = lambda coro: tasks.append(coro)
    monkeypatch.setattr(scm._ModelSwitchContext, "read_config", lambda self: None)
    monkeypatch.setattr(scm._ModelSwitchContext, "apply_override", lambda self, override: None)

    assert await runner._accept_usage_limit_switch(_event(), "x/y") == "❌ Model not available"
    assert tasks == []


@pytest.mark.asyncio
async def test_a_switch_that_fails_at_commit_does_not_resend(monkeypatch):
    # The real error replies read "Error: …" (no ✗ glyph); the override is what says the switch landed.
    from gateway import slash_commands_model as scm

    adapter = _ButtonAdapter()
    runner = _runner(adapter)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._session_model_overrides = {}
    runner._normalize_source_for_session_key = lambda source: source
    runner._perform_model_switch = AsyncMock(return_value=(SimpleNamespace(new_model="x/y"), None))
    runner._commit_model_switch = AsyncMock(return_value="Error: agent swap failed")
    tasks = []
    runner._track_background_task = lambda coro: tasks.append(coro)
    monkeypatch.setattr(scm._ModelSwitchContext, "read_config", lambda self: None)
    monkeypatch.setattr(scm._ModelSwitchContext, "apply_override", lambda self, override: None)

    assert await runner._accept_usage_limit_switch(_event(), "x/y") == "Error: agent swap failed"
    assert tasks == []


def test_the_limit_line_and_the_heads_up_stay_off_messaging():
    """The reply already carries the sentence (and the button follows); a chat message cannot be cleared,
    so the limit notice would only pile up. The 90% heads-up is a desktop / terminal nudge. Every other
    credits notice still reaches the chat."""
    from unittest.mock import patch

    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.turn_context import TurnContext

    scheduled = []
    ctx = TurnContext(source=SessionSource(platform=Platform.TELEGRAM, chat_id="c1"), user_config={},
                      mute_notification_reply=False)
    holder = SimpleNamespace(_ctx=ctx, _status_live=lambda: True,
                             _schedule=lambda coro, *args: scheduled.append(coro),
                             _runner=SimpleNamespace(_deliver_platform_notice=lambda *args: "notice-send"))
    with patch("gateway.run.render_notice_line", lambda notice: notice.text):
        TurnRunner._notice_callback_sync(holder, SimpleNamespace(level="error", text="limit", key="credits.window_limit"))
        TurnRunner._notice_callback_sync(holder, SimpleNamespace(level="warn", text="91%", key="credits.window.day"))
        assert scheduled == []
        TurnRunner._notice_callback_sync(holder, SimpleNamespace(level="error", text="paused", key="credits.depleted"))
    assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_a_tap_after_a_newer_turn_changes_nothing():
    """A stale offer (the session ran another turn since the refusal) must not switch or /retry the newer turn."""
    adapter = _ButtonAdapter()
    runner = _runner(adapter)
    runner._session_state = lambda key: SimpleNamespace(persistent=SimpleNamespace(run_generation=8))
    runner._accept_usage_limit_switch = AsyncMock(return_value="switched")
    await runner._offer_usage_limit_switch(_event(), LIMIT, 7)
    on_choice = runner._try_send_choice_picker.await_args.kwargs["on_choice_selected"]
    assert await on_choice("c1", "example/model-free:free") != "switched"
    runner._accept_usage_limit_switch.assert_not_awaited()
    # The current turn's offer still works.
    await runner._offer_usage_limit_switch(_event(), LIMIT, 8)
    on_choice = runner._try_send_choice_picker.await_args.kwargs["on_choice_selected"]
    assert await on_choice("c1", "example/model-free:free") == "switched"


@pytest.mark.asyncio
async def test_a_turn_that_ran_during_the_switch_is_not_retried(monkeypatch):
    """The tap was current, but a newer turn ran while the switch awaited model resolution: the
    /retry would rewind that turn, so it is not sent."""
    from gateway import slash_commands_model as scm

    adapter = _ButtonAdapter()
    adapter.handle_message = AsyncMock()
    runner = _runner(adapter)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._session_model_overrides = {}
    runner._normalize_source_for_session_key = lambda source: source
    state = runner._session_state("chat:c1")
    state.persistent.run_generation = 7

    async def _perform(ctx, target, provider, source):
        state.persistent.run_generation = 8  # another message's turn ran during this await
        return SimpleNamespace(new_model=target), None

    async def _commit(result, ctx, *, source, picker):
        runner._session_model_overrides[ctx.session_key] = {"model": result.new_model, "provider": "nous"}
        return "✓ Model switched to Model Free"

    tasks = []
    runner._perform_model_switch = _perform
    runner._commit_model_switch = _commit
    runner._track_background_task = lambda coro: tasks.append(coro)
    monkeypatch.setattr(scm._ModelSwitchContext, "read_config", lambda self: None)
    monkeypatch.setattr(scm._ModelSwitchContext, "apply_override", lambda self, override: None)

    await runner._offer_usage_limit_switch(_event(), LIMIT, 7)
    on_choice = runner._try_send_choice_picker.await_args.kwargs["on_choice_selected"]
    assert (await on_choice("c1", "example/model-free:free")).startswith("✓")
    for task in tasks:
        await task
    adapter.handle_message.assert_not_awaited()


def test_a_messaging_refusal_points_at_the_portal_not_topup():
    from agent.plan_limits import portal_wording
    assert portal_wording("You've hit your daily limit · resets 15:56 · turn on Keep going in /topup") == (
        "You've hit your daily limit · resets 15:56 · turn on Keep going in the Portal")
    assert "/topup" not in portal_wording("You've hit your weekly limit · /topup to add credits")


def test_a_chat_refusal_is_worded_for_the_chat_and_other_replies_are_untouched():
    runner = _runner(_TextAdapter())
    refusal = {"usage_window_message": "x", "failure_resets_at": None}
    assert runner._usage_window_chat_wording(refusal, "Limit · /topup to add credits") == "Limit · add credits in the Portal"
    assert runner._usage_window_chat_wording({}, "use /topup to add credits") == "use /topup to add credits"


@pytest.mark.asyncio
async def test_the_limit_offer_follows_the_delivered_reply_only_for_a_limit():
    runner = _runner(_TextAdapter())
    runner._hmwa_deliver_turn_response = AsyncMock(return_value="sent")
    offers = []
    runner._schedule_usage_limit_offer = lambda event, limit, **kw: offers.append((limit, kw))
    args = ("src", "entry", "chat:c1", 7)
    for result in ({"usage_limit": LIMIT}, {}):
        assert await runner._deliver_turn_response_then_limit_offer(
            _event(), *args, result, [], "reply", "", False) == "sent"
    assert offers == [(LIMIT, {"session_key": "chat:c1", "generation": 7})]
