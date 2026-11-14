"""Synthetic /goal turns are recognised by a declared flag, not by their wording.

``goals.continuation_prompt`` lets an operator reword the continuation. The old text-prefix test in
``_is_goal_continuation_event`` then stops matching, so a queued continuation would be treated as
real user input: ``/goal pause``/``/goal clear`` could not drop it and the stale-goal check in
``run_turn`` would run it anyway. The event carries ``goal_continuation=True`` instead.
"""

from __future__ import annotations

from gateway.platforms.event import MessageEvent, MessageType

CUSTOM_PROMPT = "[Goal loop] Goal: ship the release\n\nTake the next concrete step, please."


def _event(text: str, **kwargs) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, **kwargs)


def test_flag_marks_a_continuation_whatever_its_wording():
    from gateway.run_busy import GatewayBusySessionMixin

    assert GatewayBusySessionMixin._is_goal_continuation_event(_event(CUSTOM_PROMPT, goal_continuation=True))
    assert not GatewayBusySessionMixin._is_goal_continuation_event(_event(CUSTOM_PROMPT))


def test_built_in_prefix_still_recognised_for_legacy_events():
    from gateway.run_busy import GatewayBusySessionMixin

    legacy = "[Continuing toward your standing goal]\nGoal: ship it"
    assert GatewayBusySessionMixin._is_goal_continuation_event(_event(legacy))
    assert GatewayBusySessionMixin._is_goal_continuation_event(legacy)  # bare string callers


def test_plain_user_message_is_not_a_continuation():
    from gateway.run_busy import GatewayBusySessionMixin

    assert not GatewayBusySessionMixin._is_goal_continuation_event(_event("please also update the docs"))


def test_synthetic_event_sets_the_flag():
    from gateway.run_goals import GatewayGoalsMixin

    class _Runner(GatewayGoalsMixin):
        pass

    from gateway.config import Platform
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.DISCORD, chat_id="chat-1", chat_type="channel",
                           user_id="u1")
    event = _Runner._synthetic_prompt_event(source, CUSTOM_PROMPT, goal_continuation=True)
    assert event.goal_continuation is True
    assert _Runner._synthetic_prompt_event(source, "heartbeat tick").goal_continuation is False


def test_continuation_flag_survives_dataclass_replace():
    """Ingress hooks rewrite events with dataclasses.replace(); the marker must survive."""
    import dataclasses

    event = _event(CUSTOM_PROMPT, goal_continuation=True)
    assert dataclasses.replace(event, text="rewritten").goal_continuation is True
