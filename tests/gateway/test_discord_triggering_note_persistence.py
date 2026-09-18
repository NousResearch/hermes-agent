"""The Discord ``[Triggering message id: …]`` note is a model instruction: it rides the
API-bound user message but must never be persisted as the user row's ``content``
(every transcript surface rendered it as if the user typed it).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_inbound import discord_triggering_note
from gateway.session import SessionSource


def _runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=False)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    return runner


def _discord_source() -> SessionSource:
    return SessionSource(platform=Platform.DISCORD, chat_id="dm1", chat_type="dm", user_id="u1", user_name="Owen")


@pytest.mark.asyncio
async def test_discord_note_rides_model_text_but_not_persisted_content(monkeypatch):
    """Real inbound prep → persist seam: the model sees the note (outermost, ahead of the reply
    pointer); the durable row keeps the reply pointer and the authored text only."""
    monkeypatch.setattr("gateway.session._discord_tools_loaded", lambda: True)
    runner = _runner()
    source = _discord_source()
    event = MessageEvent(
        text="yes do that", source=source, message_id="1550380365858865157",
        reply_to_message_id="1550380365858865156", reply_to_text="Create a project plan for Q4",
    )

    model_text = await runner._prepare_inbound_message_text(event=event, source=source, history=[])
    message_text, persist_user_message, _ts = runner._hmwa_apply_message_timestamp(event, model_text)

    note = discord_triggering_note("1550380365858865157")
    assert message_text.startswith(f"{note}\n\n[Replying to: \"Create a project plan for Q4\"]")
    assert persist_user_message == '[Replying to: "Create a project plan for Q4"]\n\nyes do that'

    # Control: a turn without a platform message id (desktop relay) persists byte-identical text.
    relay = MessageEvent(text="plain relay turn", source=source, message_id=None)
    relay_text = await runner._prepare_inbound_message_text(event=relay, source=source, history=[])
    _, relay_persist, _ = runner._hmwa_apply_message_timestamp(relay, relay_text)
    assert relay_persist == "plain relay turn" == relay_text


@pytest.mark.asyncio
async def test_queued_followup_persists_authored_text():
    """The in-band queued follow-up runs the same inbound prep; its recursive ``_run_agent`` must
    carry the clean persist text too, or a queued Discord turn re-persists the note."""
    runner = _runner()
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": []})
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value="agent:main:discord:dm:dm1")
    note = discord_triggering_note("6002")
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=f"{note}\n\nthe follow-up")
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    source = _discord_source()
    turn_ctx = SimpleNamespace(
        source=source, session_id="sid", session_key="agent:main:discord:dm:dm1", run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata=None, context_prompt=None,
        result_holder=[None],
    )
    pending_event = SimpleNamespace(
        source=source, message_id="6002", channel_prompt=None, message_type=None, internal=False, metadata={},
    )

    await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="hi again", pending_event=pending_event,
        response="resp", result={"interrupted": True, "messages": []}, stream_task=None,
    )

    kwargs = runner._run_agent.await_args.kwargs
    assert kwargs["message"] == f"{note}\n\nthe follow-up"
    assert kwargs["persist_user_message"] == "the follow-up"
