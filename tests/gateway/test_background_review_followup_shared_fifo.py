"""A busy follow-up admitted into the shared runtime's durable FIFO fences the automatic review.

With a session authority installed, a message that arrives while the session is busy is
committed to ``session_admissions`` (``dispatch_shared_busy``) instead of the adapter's pending
slot. Review admission must see it exactly as it sees a slot write: the turn's non-consuming,
body-free probe answers True while the follower is queued (and only then), and the commit bumps
the follow-up epoch and runs the registered review cancel, so the review the live turn would
spawn never overlaps the follower. Real GatewayRunner, adapter, SessionAuthority, drain and
``execute_admission`` over SQLite; only the agent turn is a gated stand-in that marks the turn
active like ``_hmwa_prepare_turn`` and wires the probe like ``_wire_turn_agent_callbacks``.
"""

import asyncio
import types
from typing import cast

import pytest

from gateway.run_turn_runner import TurnRunner
from tests.gateway.test_shared_busy_receive_loop import Adapter, _statuses
from tests.waiting import async_wait_until


def _wire_probe(runner, adapter, source, session_key, run_generation):
    """The production probe for one turn (``_wire_turn_agent_callbacks``), on the real runner and adapter."""
    agent = types.SimpleNamespace()
    ctx = types.SimpleNamespace(
        progress_callback=None,
        native_tool_start_callback=None,
        voice_ack_callback=None,
        _voice_ack_guild=[None],
        _native_slack_task_cards=False,
        native_tool_complete_callback=None,
        _step_callback_sync=None,
        _hooks_ref=types.SimpleNamespace(loaded_hooks=[]),
        _status_callback_sync=None,
        _event_callback_sync=None,
        _status_adapter=adapter,
        _status_thread_metadata={},
        session_key=session_key,
        user_config={},
        source=source,
        mute_notification_reply=False,
        _thinking_enabled=False,
        agent_holder=[None],
        tools_holder=[None],
        process_task_id=None,
        process_baseline=None,
        run_generation=run_generation,
        voice_turn=False,
    )
    holder = types.SimpleNamespace(
        _ctx=ctx,
        _approval_owner=None,
        _runner=runner,
        _make_bg_review_callbacks=lambda: (lambda message: None, lambda: None),
        _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
        _clarify_callback_sync=lambda *a, **k: None,
        _notice_callback_sync=lambda *a, **k: None,
        _attach_session_title_callback=lambda agent, ctx: None,
    )
    TurnRunner._wire_turn_agent_callbacks(
        cast(TurnRunner, holder), agent, {}, None, None, None, False
    )
    return agent


@pytest.mark.asyncio
async def test_durable_fifo_followup_is_visible_to_the_review_probe_and_fences_the_review(
    monkeypatch,
):
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = Adapter()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(
        runner, profile_id="default", instance_id="owner"
    )
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(
        chat_id="busy-chat", chat_type="dm", user_id="fixture-user"
    )
    key = runner.session_store._generate_session_key(source)
    first_turn_wired, gate = asyncio.Event(), asyncio.Event()
    observed, cancels = {}, []

    async def turn(event, source, quick_key, run_generation):
        await runner._mark_durable_active_turn(event, quick_key)
        if event.text == "one":
            agent = _wire_probe(runner, adapter, source, quick_key, run_generation)
            adapter.register_followup_review_cancel(
                quick_key, lambda: cancels.append(event.text)
            )
            # The live turn's own started admission is not a follow-up.
            observed["probe_before"] = agent.followup_pending_callback()
            observed["epoch_before"] = adapter.followup_admission_state(quick_key).epoch
            first_turn_wired.set()
            await gate.wait()
            observed["probe_with_follower"] = agent.followup_pending_callback()
            observed["epoch_with_follower"] = adapter.followup_admission_state(
                quick_key
            ).epoch
        execution_result.get()["result"] = {
            "final_response": "reply to " + event.text,
            "messages": [],
        }
        return "reply to " + event.text

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)

    await adapter.handle_message(
        MessageEvent(text="one", source=source, message_id="m1")
    )
    await asyncio.wait_for(first_turn_wired.wait(), 5)
    assert observed["probe_before"] is False
    assert adapter.has_pending_message(key) is False
    assert cancels == []

    # The receive loop hands the follow-up over once it is committed to the FIFO, not when it runs.
    await asyncio.wait_for(
        adapter.handle_message(
            MessageEvent(text="two", source=source, message_id="m2")
        ),
        5,
    )
    assert _statuses(authority) == {"one": "started", "two": "queued"}
    assert adapter.has_pending_message(key) is True
    assert cancels == ["one"], (
        "the FIFO commit did not run the live turn's review fence"
    )
    assert adapter.followup_admission_state(key).epoch == observed["epoch_before"] + 1

    gate.set()
    await async_wait_until(
        lambda: "reply to two" in adapter.sent and not adapter._background_tasks,
        timeout=15,
    )
    assert observed["probe_with_follower"] is True
    assert observed["epoch_with_follower"] == observed["epoch_before"] + 1
    assert adapter.sent == ["reply to one", "reply to two"]
    # Nothing queued any more: the probe must not keep reporting a follower that already ran.
    assert _statuses(authority) == {"one": "terminal", "two": "terminal"}
    assert adapter.has_pending_message(key) is False
