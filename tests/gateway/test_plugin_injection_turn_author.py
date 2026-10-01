"""A plugin-injected gateway turn reaches ``run_conversation`` with the plugin as its author.

The dispatch restores the HUMAN's session source, so the source alone cannot tell a plugin's
instruction from something the user typed. ``TurnContext.turn_author_override`` is what carries
the difference into the agent call.
"""

from types import SimpleNamespace

import pytest

from agent.turn_author import plugin_author_for_event


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