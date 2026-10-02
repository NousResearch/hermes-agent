"""A plugin-injected gateway turn reaches ``run_conversation`` with the plugin as its author.

The dispatch restores the HUMAN's session source, so the source alone cannot tell a plugin's
instruction from something the user typed. ``TurnContext.turn_author_override`` is what carries
the difference into the agent call.

The author has to survive every path the same event can take into a turn: the first turn, a
follow-up drained mid-turn, and a pending-slot merge. A plugin event absorbed into a human-held
slot loses its author, and with it both guarantees this change makes — ``is_bot`` gates durable
profile writes, and the author id keys the a2a session.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_author import plugin_author_for_event
from gateway.platforms.base import BasePlatformAdapter, Platform, PlatformConfig, SendResult
from gateway.platforms.event import MessageEvent, MessageType


def _runner(turn_author_override):
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    ctx = TurnContext(
        source=SimpleNamespace(user_id="42", user_name="tester", is_bot=False),
        turn_author_override=turn_author_override,
        session_key="agent:main:telegram:dm:42",
        session_id="session-42",
        message="Read the room and call tool_x after you read it.",
    )
    runner = object.__new__(TurnRunner)
    runner._ctx = ctx
    runner._runner = SimpleNamespace(_consume_pending_native_image_paths=lambda key: [])
    return runner


def _captured_kwargs(turn_author_override) -> dict:
    """Run the real kwargs-building path and return what the agent was called with."""
    captured = {}

    def fake_run_conversation(message=None, *, conversation_history=None, task_id=None,
                              turn_author=None, **kwargs):
        captured.update(message=message, turn_author=turn_author, task_id=task_id,
                        history=conversation_history)
        return {"final_response": "ok", "messages": [], "api_calls": 0, "completed": True}

    agent = SimpleNamespace(run_conversation=fake_run_conversation)
    runner = _runner(turn_author_override)
    runner._approval_notify_sync = lambda *a, **k: None
    runner._run_conversation_with_approval(agent, [], None, None, None)
    return captured


PLUGIN_EVENT_AUTHOR = plugin_author_for_event(SimpleNamespace(metadata={
    "hermes_plugin_id": "room-wake", "hermes_plugin_injection": True,
}))


def test_a_plugin_injected_turn_is_authored_by_the_plugin():
    captured = _captured_kwargs(PLUGIN_EVENT_AUTHOR)
    # The regression: the old code built this from the restored HUMAN source, so a memory
    # provider stored the plugin's instruction as a durable fact about the user.
    assert captured["turn_author"] == {"id": "plugin:room-wake", "name": "room-wake", "is_bot": True}
    assert captured["turn_author"]["id"] != "42"


def test_an_ordinary_turn_is_still_authored_by_the_human():
    assert _captured_kwargs(None)["turn_author"] == {"id": "42", "name": "tester", "is_bot": False}


def test_a_bot_source_is_still_authored_by_the_bot():
    override = plugin_author_for_event(SimpleNamespace(metadata={
        "hermes_plugin_id": "x", "hermes_plugin_injection": True,
        "hermes_plugin_author": {"id": "bot:alpha", "name": "Alpha", "is_bot": True},
    }))
    assert _captured_kwargs(override)["turn_author"] == {"id": "bot:alpha", "name": "Alpha", "is_bot": True}


class TestPreparedTurnCarriesTheAuthor:
    def _prepared(self, metadata):
        from gateway.run import GatewayRunner

        return GatewayRunner._PreparedTurn(
            history=[], context_prompt="", message_text="read the room",
            persist_user_message=None, persist_user_timestamp=None,
            persist_user_display_kind="internal_notification",
            turn_author=plugin_author_for_event(SimpleNamespace(metadata=metadata)),
        )

    def test_prepare_turn_stamps_the_plugin_author_on_the_turn(self):
        prepared = self._prepared({
            "hermes_plugin_id": "room-wake", "hermes_plugin_injection": True,
        })
        assert prepared.turn_author == {"id": "plugin:room-wake", "name": "room-wake", "is_bot": True}

    @pytest.mark.parametrize("metadata", [None, {}, {"hermes_plugin_id": "x"}])
    def test_a_non_injected_event_yields_no_author(self, metadata):
        assert self._prepared(metadata).turn_author is None


def test_turn_context_carries_the_override():
    from gateway.turn_context import TurnContext

    ctx = TurnContext(turn_author_override={"id": "plugin:room-wake", "name": "room-wake", "is_bot": True})
    assert ctx.turn_author_override["is_bot"] is True
    # The default leaves a human turn exactly as it was.
    assert TurnContext().turn_author_override is None


# ---------------------------------------------------------------------------
# A plugin event that drains as a follow-up instead of starting the turn
# ---------------------------------------------------------------------------

class _PendingEventStub:
    """Enough of MessageEvent for the follow-up drain: no platform I/O happens on it."""

    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata
        self.internal = True
        self.message_type = MessageType.TEXT
        self.message_id = None
        self.channel_prompt = None
        self.reply_expected = None
        self.media_urls = []
        self.media_types = []
        self.source = SimpleNamespace(user_id="42", user_name="tester", is_bot=False)

    def get_command(self):
        return None


def _queued_followup_kwargs(metadata):
    """Drive the real ``_run_agent_queued_followup`` and capture the recursive call's kwargs.

    That recursive ``_run_agent`` is the same entry point the first turn goes through, so what it
    receives is what the agent ultimately gets.
    """
    import asyncio

    from gateway.run import GatewayRunner
    from gateway.turn_context import TurnContext

    captured = {}

    async def _noop(*args, **kwargs):
        return None

    async def _inbound_text(*, event, source, history, session_key):
        return event.text

    class _FollowupRunner(GatewayRunner):
        async def _run_agent(self, *args, **kwargs):
            captured.update(kwargs)
            return {"final_response": "ok", "messages": []}

    runner = _FollowupRunner.__new__(_FollowupRunner)
    runner._draining = False
    runner._MAX_INTERRUPT_DEPTH = 3
    runner.config = SimpleNamespace(group_sessions_per_user=True, thread_sessions_per_user=False)
    runner._status_action_label = lambda: "reload"
    runner._intake_adapter_for = lambda source: None
    runner._delivery_adapter_for = lambda source: None
    runner._refresh_agent_cache_message_count = _noop
    runner._prepare_profile_scoped_inbound_message_text = _inbound_text
    runner._session_key_for_source = lambda source: "agent:main:telegram:dm:42"
    runner._persist_prompt_pins = _noop

    turn_ctx = TurnContext(
        source=SimpleNamespace(user_id="42", user_name="tester", is_bot=False),
        session_key="agent:main:telegram:dm:42",
        session_id="session-42",
        context_prompt="",
        history=[],
        run_generation=1,
        _interrupt_depth=0,
    )
    turn_ctx._status_thread_metadata = {}
    turn_ctx.result_holder = [None]

    asyncio.run(runner._run_agent_queued_followup(
        turn_ctx=turn_ctx, adapter=None, pending="read the room",
        pending_event=_PendingEventStub("read the room", metadata),
        response=None, result={"messages": [], "interrupted": False}, stream_task=None,
    ))
    return captured


def test_a_plugin_event_drained_as_a_followup_keeps_the_plugin_author():
    captured = _queued_followup_kwargs({"hermes_plugin_id": "room-wake", "hermes_plugin_injection": True})
    # The regression: this call list omitted the override, so the recursive turn fell through to
    # the restored human in the runner and the plugin's words were stored as the user's.
    assert captured["turn_author_override"] == {"id": "plugin:room-wake", "name": "room-wake", "is_bot": True}


def test_a_human_event_drained_as_a_followup_keeps_no_override():
    # None must stay None so the human-source fallback downstream is untouched.
    assert _queued_followup_kwargs({})["turn_author_override"] is None


# ---------------------------------------------------------------------------
# A plugin event arriving while a turn is already running
# ---------------------------------------------------------------------------

class _SlotAdapter(BasePlatformAdapter):
    """A real adapter so the slot accessors under test are the production ones."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="msg-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _busy_runner(adapter):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queued_events = {}
    runner._delivery_adapter_for = lambda source: adapter
    return runner


def _event(text, *, internal=False, metadata=None, message_type=None):
    return MessageEvent(
        text=text, message_type=message_type or MessageType.TEXT,
        source=MagicMock(chat_id="42", platform=Platform.TELEGRAM),
        message_id=None, internal=internal, metadata=metadata or {},
    )


def test_a_plugin_event_is_not_absorbed_into_a_human_held_slot():
    """The Telegram grace path and the pending-sentinel path both merge with ``merge_text=True``.

    Merging appends the plugin's words to the human's message and keeps the human's metadata, so
    the combined turn runs with no plugin author at all — exactly what this change exists to
    prevent. The base adapter already refuses that merge; these two call sites did not.
    """
    adapter = _SlotAdapter()
    runner = _busy_runner(adapter)
    session_key = "agent:main:telegram:dm:42"
    human = _event("what is the weather?")
    adapter._pending_messages[session_key] = human

    plugin_event = _event(
        "read the room and call tool_x", internal=True,
        metadata={"hermes_plugin_id": "room-wake", "hermes_plugin_injection": True},
    )
    runner._hm_merge_pending_for_source(human.source, session_key, plugin_event, merge_text=True)

    # The human's message is untouched: no text smuggled in, no author overwritten.
    assert adapter._pending_messages[session_key] is human
    assert human.text == "what is the weather?"
    # And the plugin event is not dropped either — it waits behind the human's turn in the
    # FIFO, so it runs as its own turn carrying its own author.
    queued = runner._peek_session_state(session_key).conversation.queued_events
    assert [e.text for e in queued] == ["read the room and call tool_x"]
    promoted = runner._promote_queued_event(session_key, adapter, human)
    assert promoted is human
    assert adapter._pending_messages[session_key].text == "read the room and call tool_x"


def test_a_human_followup_still_merges_as_before():
    """The guard reads ``internal`` only, so ordinary human follow-ups are unaffected."""
    from gateway.platforms.base import _append_text

    adapter = _SlotAdapter()
    runner = _busy_runner(adapter)
    session_key = "agent:main:telegram:dm:42"
    human = _event("first half")
    adapter._pending_messages[session_key] = human

    runner._hm_merge_pending_for_source(human.source, session_key, _event("second half"), merge_text=True)
    assert adapter._pending_messages[session_key] is human
    assert human.text == _append_text("first half", "second half")


def test_two_plugin_events_with_different_authors_do_not_merge():
    """``_SECURITY_METADATA_KEYS`` decides which pending events may share one slot.

    Leaving the author out of that tuple made two injections from a single plugin id compare
    equal on every checked key, so they merged and the second one's text was attributed to the
    first one's author.
    """
    from gateway.run import GatewayRunner

    assert "hermes_plugin_author" in GatewayRunner._SECURITY_METADATA_KEYS

    adapter = _SlotAdapter()
    runner = _busy_runner(adapter)
    session_key = "agent:main:telegram:dm:42"
    # Photos are what make the merge branch reachable in this path at all.
    runner._queue_or_replace_pending_event(session_key, _event(
        "from the bridge", internal=True, message_type=MessageType.PHOTO,
        metadata={"hermes_plugin_id": "bridge", "hermes_plugin_injection": True,
                  "hermes_plugin_author": {"id": "person:a", "name": "A", "is_bot": False}},
    ))
    runner._queue_or_replace_pending_event(session_key, _event(
        "from someone else", internal=True, message_type=MessageType.PHOTO,
        metadata={"hermes_plugin_id": "bridge", "hermes_plugin_injection": True,
                  "hermes_plugin_author": {"id": "person:b", "name": "B", "is_bot": False}},
    ))

    stored = adapter._pending_messages[session_key]
    assert stored.metadata["hermes_plugin_author"]["id"] == "person:a"
    assert "someone else" not in (stored.text or "")