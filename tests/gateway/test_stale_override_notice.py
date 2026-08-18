import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import HomeChannel, Platform
from gateway.session import SessionSource
from gateway.stale_override_notice import (
    OverrideNoticeDecision,
    StaleOverrideNoticeConfig,
    reasoning_effort,
    reasoning_matches_policy,
    routes_differ,
    source_matches_channels,
)


class _Adapter:
    def __init__(self, picker_success=True):
        self.send = AsyncMock()
        self.handle_message = AsyncMock()
        self.picker_success = picker_success
        self.picker_call = None

    async def send_choice_picker(self, **kwargs):
        self.picker_call = kwargs
        return SimpleNamespace(success=self.picker_success)


def _source(platform=Platform.TELEGRAM, chat_id="123", thread_id=None):
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        chat_type="dm",
        user_id="u1",
        thread_id=thread_id,
    )


def test_config_defaults_disabled_and_normalizes_values():
    defaults = StaleOverrideNoticeConfig.from_dict(None)
    assert defaults.mode == "off"
    assert defaults.channels == ("home",)
    cfg = StaleOverrideNoticeConfig.from_dict({
        "mode": "INFO-ONLY",
        "idle_minutes": "75",
        "model": "off",
        "reasoning": "non-default",
        "channels": "Telegram:123",
    })
    assert cfg.mode == "info_only"
    assert cfg.idle_minutes == 75
    assert cfg.model == "off"
    assert cfg.reasoning == "non_default"
    assert cfg.channels == ("telegram:123",)


def test_invalid_config_falls_back_safely():
    cfg = StaleOverrideNoticeConfig.from_dict({
        "mode": "surprise",
        "idle_minutes": 0,
        "model": "all",
    })
    assert cfg.mode == "off"
    assert cfg.idle_minutes == 60
    assert cfg.model == "non_default"
    assert cfg.channels == ("home",)


def test_channel_scope_empty_wildcards_exact_and_home():
    source = _source(thread_id="9")
    home = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="123",
        thread_id="9",
        name="Home",
    )
    assert source_matches_channels(source, ())
    assert source_matches_channels(source, ("*",))
    assert source_matches_channels(source, ("telegram:*",))
    assert source_matches_channels(source, ("telegram:123",))
    assert source_matches_channels(source, ("telegram:123:9",))
    assert source_matches_channels(source, ("home",), home_channel=home)
    assert not source_matches_channels(source, ("discord:123",), home_channel=home)
    assert not source_matches_channels(
        source,
        ("home",),
        home_channel=HomeChannel(
            platform=Platform.TELEGRAM,
            chat_id="123",
            thread_id="10",
            name="Other",
        ),
    )


def test_routes_compare_provider_and_model_case_insensitively():
    assert not routes_differ("Model-A", "Provider", "model-a", "provider")
    assert routes_differ("model-a", "provider-a", "model-a", "provider-b")
    assert routes_differ("model-a", "provider", "model-b", "provider")


def test_reasoning_policy_above_non_default_and_disabled():
    medium = {"enabled": True, "effort": "medium"}
    high = {"enabled": True, "effort": "high"}
    maximum = {"enabled": True, "effort": "max"}
    ultra = {"enabled": True, "effort": "ultra"}
    low = {"enabled": True, "effort": "low"}
    disabled = {"enabled": False}

    assert reasoning_matches_policy("above_default", high, medium)
    assert reasoning_matches_policy("above_default", maximum, medium)
    assert reasoning_matches_policy("above_default", ultra, maximum)
    assert not reasoning_matches_policy("above_default", low, medium)
    assert reasoning_matches_policy("non_default", low, medium)
    assert reasoning_matches_policy("non_default", disabled, medium)
    assert not reasoning_matches_policy("off", high, medium)
    assert reasoning_effort(None) == "medium"
    assert reasoning_effort(disabled) == "none"


def test_decision_builds_axis_specific_choices_and_messages():
    both = OverrideNoticeDecision(
        model_stale=True,
        reasoning_stale=True,
        current_route="p/custom",
        default_route="p/default",
        current_reasoning="high",
        default_reasoning="medium",
    )
    assert [c["value"] for c in both.choices()] == [
        "continue",
        "default_model",
        "default_reasoning",
        "defaults",
    ]
    assert [c["label"] for c in both.choices()] == [
        "✅ Continue with current settings",
        "↩️ Restore default model",
        "↩️ Restore default reasoning",
        "↩️ Restore all defaults",
    ]
    assert "will not be sent" in both.message(61, held=True)
    assert "info-only" in both.message(61, held=False)

    model_only = OverrideNoticeDecision(model_stale=True)
    assert [c["value"] for c in model_only.choices()] == [
        "continue",
        "default_model",
    ]


def _runner(mode="info_only", *, picker_success=True):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        stale_override_notice=StaleOverrideNoticeConfig(
            mode=mode,
            idle_minutes=60,
            channels=(),
        ),
        get_home_channel=lambda _platform: None,
    )
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_session_metadata=AsyncMock(return_value=time.time() - 3700),
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(was_auto_reset=False)
        ),
        set_session_metadata=AsyncMock(),
        set_model_override=AsyncMock(),
    )
    runner._stale_override_pending = {}
    runner._background_tasks = set()
    runner._adapter = _Adapter(picker_success=picker_success)
    runner._adapter_for_source = lambda _source: runner._adapter
    runner._thread_metadata_for_source = lambda *_args: None
    runner._reply_anchor_for_event = lambda _event: None
    runner._stale_override_decision = MagicMock(
        return_value=OverrideNoticeDecision(
            model_stale=True,
            current_route="p/custom",
            default_route="p/default",
        )
    )
    runner._schedule_stale_override_prompt_expiry = MagicMock()
    runner._clear_stale_override_selection = AsyncMock()
    return runner


def _event(text="hello", *, internal=False):
    from gateway.platforms.base import MessageEvent

    return MessageEvent(text=text, source=_source(), internal=internal)


@pytest.mark.asyncio
async def test_info_only_notifies_and_does_not_hold_message():
    runner = _runner("info_only")
    handled, response = await runner._maybe_handle_stale_override_notice(
        _event(), "session-key"
    )

    assert (handled, response) == (False, None)
    runner._adapter.send.assert_awaited_once()
    assert "info-only mode" in runner._adapter.send.await_args.args[1]
    assert runner._adapter.picker_call is None


@pytest.mark.asyncio
async def test_pending_auto_reset_skips_misleading_override_prompt():
    runner = _runner("confirm")
    runner.async_session_store.get_or_create_session.return_value = SimpleNamespace(
        was_auto_reset=True
    )

    handled, response = await runner._maybe_handle_stale_override_notice(
        _event(), "session-key"
    )

    assert (handled, response) == (False, None)
    runner.async_session_store.get_or_create_session.assert_awaited_once()
    assert (
        runner.async_session_store.get_or_create_session.await_args.kwargs[
            "touch_activity"
        ]
        is False
    )
    runner._stale_override_decision.assert_not_called()
    assert runner._adapter.picker_call is None


@pytest.mark.asyncio
async def test_confirm_holds_then_resumes_only_after_explicit_choice():
    runner = _runner("confirm")
    event = _event()

    handled, response = await runner._maybe_handle_stale_override_notice(
        event, "session-key"
    )

    assert (handled, response) == (True, None)
    assert runner._adapter.picker_call is not None
    assert runner._adapter.picker_call["metadata"]["choice_layout"] == "vertical"
    runner._adapter.handle_message.assert_not_awaited()
    runner._schedule_stale_override_prompt_expiry.assert_called_once()

    callback = runner._adapter.picker_call["on_choice_selected"]
    result = await callback("123", "default_model")

    assert result.startswith("Override updated")
    runner._clear_stale_override_selection.assert_awaited_once_with(
        "session-key", model=True, reasoning=False
    )
    runner._adapter.handle_message.assert_awaited_once_with(event)
    assert event.metadata["_stale_override_notice_bypass"] is True
    assert "session-key" not in runner._stale_override_pending


@pytest.mark.asyncio
async def test_confirm_resumes_through_current_adapter_after_reconnect():
    runner = _runner("confirm")
    prompt_adapter = runner._adapter
    event = _event()

    held, _ = await runner._maybe_handle_stale_override_notice(event, "session-key")
    assert held is True
    callback = prompt_adapter.picker_call["on_choice_selected"]

    resume_adapter = _Adapter()
    runner._adapter_for_source = lambda _source: resume_adapter
    await callback("123", "continue")

    prompt_adapter.handle_message.assert_not_awaited()
    resume_adapter.handle_message.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_confirm_falls_back_to_prompt_adapter_when_route_is_temporarily_missing():
    runner = _runner("confirm")
    prompt_adapter = runner._adapter
    event = _event()

    held, _ = await runner._maybe_handle_stale_override_notice(event, "session-key")
    assert held is True
    callback = prompt_adapter.picker_call["on_choice_selected"]

    runner._adapter_for_source = lambda _source: None
    await callback("123", "continue")

    prompt_adapter.handle_message.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_reasoning_restore_is_observable_before_held_message_resumes(caplog):
    runner = _runner("confirm")
    runner._stale_override_decision = MagicMock(
        return_value=OverrideNoticeDecision(
            reasoning_stale=True,
            current_reasoning="xhigh",
            default_reasoning="medium",
        )
    )
    event = _event("test reasoning restore")

    held, _ = await runner._maybe_handle_stale_override_notice(event, "session-key")
    assert held is True

    callback = runner._adapter.picker_call["on_choice_selected"]
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        result = await callback("123", "default_reasoning")

    runner._clear_stale_override_selection.assert_awaited_once_with(
        "session-key", model=False, reasoning=True
    )
    runner._adapter.handle_message.assert_awaited_once_with(event)
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Stale-override selection accepted" in message
        and "choice=default_reasoning" in message
        and "reset_reasoning=True" in message
        for message in messages
    )
    assert any(
        "Stale-override held message re-dispatched" in message for message in messages
    )
    assert result.startswith("Override updated")


@pytest.mark.asyncio
async def test_model_store_failure_does_not_strand_reset_or_held_message():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    state = SimpleNamespace(
        conversation=SimpleNamespace(model_override={"model": "custom"})
    )
    runner._session_state = MagicMock(return_value=state)
    runner._async_session_store = SimpleNamespace(
        _store=object(),
        set_model_override=AsyncMock(side_effect=OSError("store unavailable")),
    )
    runner._set_session_reasoning_override = MagicMock()
    runner._evict_cached_agent = MagicMock()

    await runner._clear_stale_override_selection(
        "session-key", model=True, reasoning=True
    )

    assert state.conversation.model_override is None
    runner._set_session_reasoning_override.assert_called_once_with("session-key", None)
    runner._evict_cached_agent.assert_called_once_with("session-key")


@pytest.mark.asyncio
async def test_model_reset_cancels_pending_once_restore_and_switch_note():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    state = SimpleNamespace(
        conversation=SimpleNamespace(
            model_override={"model": "temporary"},
            one_turn_restore={
                "had_override": True,
                "override": {"model": "previous"},
            },
        )
    )
    runner._session_state = MagicMock(return_value=state)
    runner._pending_model_notes = {"session-key": "stale switch note"}
    runner._async_session_store = SimpleNamespace(
        _store=object(),
        set_model_override=AsyncMock(),
    )
    runner.session_store = runner._async_session_store._store
    runner._set_session_reasoning_override = MagicMock()
    runner._evict_cached_agent = MagicMock()

    await runner._clear_stale_override_selection(
        "session-key", model=True, reasoning=False
    )

    assert state.conversation.model_override is None
    assert state.conversation.one_turn_restore is None
    assert "session-key" not in runner._pending_model_notes
    runner._async_session_store.set_model_override.assert_awaited_once_with(
        "session-key", None
    )
    runner._evict_cached_agent.assert_called_once_with("session-key")


@pytest.mark.asyncio
async def test_conversation_boundary_makes_held_picker_inert():
    runner = _runner("confirm")
    event = _event()

    held, _ = await runner._maybe_handle_stale_override_notice(event, "session-key")
    assert held is True
    callback = runner._adapter.picker_call["on_choice_selected"]

    runner._peek_session_state = MagicMock(return_value=None)
    runner._clear_session_boundary_security_state = MagicMock()
    runner._clear_conversation_scope("session-key", reason="test-new")

    reply = await callback("123", "continue")
    assert "expired" in reply.lower()
    runner._adapter.handle_message.assert_not_awaited()
    assert "session-key" not in runner._stale_override_pending


@pytest.mark.asyncio
async def test_confirm_picker_failure_fails_open_without_losing_message():
    runner = _runner("confirm", picker_success=False)
    handled, response = await runner._maybe_handle_stale_override_notice(
        _event(), "session-key"
    )
    assert (handled, response) == (False, None)
    assert runner._stale_override_pending == {}


@pytest.mark.asyncio
async def test_pending_prompt_rejects_newer_message_without_replacing_original():
    runner = _runner("confirm")
    original = _event("first")
    await runner._maybe_handle_stale_override_notice(original, "session-key")
    handled, response = await runner._maybe_handle_stale_override_notice(
        _event("second"), "session-key"
    )
    assert handled is True
    assert "newer message was not sent" in response
    assert runner._stale_override_pending["session-key"]["event"] is original


@pytest.mark.asyncio
@pytest.mark.parametrize("event", [_event("/status"), _event(internal=True)])
async def test_commands_and_internal_messages_bypass_notice(event):
    runner = _runner("confirm")
    handled, response = await runner._maybe_handle_stale_override_notice(
        event, "session-key"
    )
    assert (handled, response) == (False, None)
    runner.async_session_store.get_session_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_replayed_held_event_consumes_one_shot_bypass():
    runner = _runner("confirm")
    event = _event()
    event.metadata["_stale_override_notice_bypass"] = True
    handled, response = await runner._maybe_handle_stale_override_notice(
        event, "session-key"
    )
    assert (handled, response) == (False, None)
    assert "_stale_override_notice_bypass" not in event.metadata


@pytest.mark.asyncio
async def test_completion_clock_only_advances_for_user_turns():
    runner = _runner("off")
    await runner._mark_stale_override_turn_completed("session-key", is_internal=False)
    runner.async_session_store.set_session_metadata.assert_awaited_once()
    args = runner.async_session_store.set_session_metadata.await_args.args
    assert args[0] == "session-key"
    assert args[1] == "stale_override_last_completed_at"
    assert isinstance(args[2], float)

    runner.async_session_store.set_session_metadata.reset_mock()
    await runner._mark_stale_override_turn_completed("session-key", is_internal=True)
    runner.async_session_store.set_session_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_completion_clock_is_deferred_until_platform_delivery():
    runner = _runner("info_only")
    registered = {}

    def register(session_key, callback, *, generation):
        registered.update(
            session_key=session_key,
            callback=callback,
            generation=generation,
        )

    runner._adapter_for_source = lambda _source: SimpleNamespace(
        register_post_delivery_callback=register
    )

    await runner._defer_stale_override_turn_completed(
        "session-key",
        source=_source(),
        run_generation=7,
        is_internal=False,
    )

    runner.async_session_store.set_session_metadata.assert_not_awaited()
    assert registered["session_key"] == "session-key"
    assert registered["generation"] == 7

    await registered["callback"]()
    runner.async_session_store.set_session_metadata.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_notice_does_not_register_or_write_completion_clock():
    runner = _runner("off")
    register = MagicMock()
    runner._adapter_for_source = lambda _source: SimpleNamespace(
        register_post_delivery_callback=register
    )

    await runner._defer_stale_override_turn_completed(
        "session-key",
        source=_source(),
        run_generation=7,
        is_internal=False,
    )

    register.assert_not_called()
    runner.async_session_store.set_session_metadata.assert_not_awaited()


def test_runner_decision_requires_explicit_override_and_uses_live_baselines():
    runner = _runner("confirm")
    # _runner stubs this method for dispatch-flow tests; this case exercises
    # the real decision method on the class.
    del runner._stale_override_decision
    state = SimpleNamespace(
        conversation=SimpleNamespace(
            model_override={"model": "custom", "provider": "p2"},
            reasoning_override={"enabled": True, "effort": "high"},
        )
    )
    runner._rehydrate_session_model_override = MagicMock()
    runner._peek_session_state = MagicMock(return_value=state)
    runner._resolve_session_agent_runtime = MagicMock(
        side_effect=lambda **kwargs: (
            ("default", {"provider": "p1"})
            if kwargs.get("include_session_override") is False
            else ("custom", {"provider": "p2"})
        )
    )
    runner._load_reasoning_config = MagicMock(
        return_value={"enabled": True, "effort": "medium"}
    )

    decision = runner._stale_override_decision(
        source=_source(),
        session_key="session-key",
        notice_config=runner.config.stale_override_notice,
    )

    assert decision.model_stale is True
    assert decision.reasoning_stale is True
    assert decision.default_route == "p1/default"
    assert decision.current_route == "p2/custom"

    state.conversation.model_override = None
    decision = runner._stale_override_decision(
        source=_source(),
        session_key="session-key",
        notice_config=runner.config.stale_override_notice,
    )
    assert decision.model_stale is False


@pytest.mark.asyncio
async def test_telegram_old_picker_message_cannot_resolve_new_chat_state():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    callback = AsyncMock(return_value="should not run")
    adapter._choice_picker_state = {
        ("123", ""): {
            "msg_id": 200,
            "choices": [{"value": "continue"}],
            "on_choice_selected": callback,
        }
    }
    query = SimpleNamespace(
        message=SimpleNamespace(message_id=100, message_thread_id=None),
        answer=AsyncMock(),
    )

    await adapter._handle_choice_picker_callback(query, "cp:0", "123")

    query.answer.assert_awaited_once_with(
        text="Picker expired — use the latest prompt."
    )
    callback.assert_not_awaited()
