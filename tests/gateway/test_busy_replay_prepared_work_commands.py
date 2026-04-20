import json
from unittest.mock import MagicMock

from agent.continuation_engine import should_use_continuation_engine
from agent.intent_preclassifier import preclassify_intent
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import (
    _get_prepared_work_command,
    _is_discardable_pending_slash_command,
    _pending_event_replay_text,
)
from hermes_cli.work_command_adapter import prepare_work_command


def _handoff_payload() -> str:
    return json.dumps(
        {
            "task": "Resume the delegated task",
            "expected_outcome": "Implementation resumed with verification evidence",
            "required_skills": ["python", "testing"],
            "required_tools": ["read_file", "patch", "terminal"],
            "must_do": ["inspect current state before acting"],
            "must_not_do": ["discard the preserved contract"],
            "context": {"ticket": "w2-t09"},
        }
    )


def _raw_args_for(command_name: str) -> str:
    if command_name == "handoff":
        return _handoff_payload()
    if command_name == "init-deep":
        return "Investigate the migration surface"
    if command_name == "start-work":
        return "Continue the repo work"
    if command_name == "ralph-loop":
        return "Close the remaining work items"
    if command_name == "ulw-loop":
        return "Push the task to completion"
    raise KeyError(command_name)


def _event_text_for(command_name: str) -> str:
    if command_name == "handoff":
        return "/handoff Continue the repo work"
    return f"/{command_name} {_raw_args_for(command_name)}"


def _make_prepared_event(command_name: str = "handoff") -> MessageEvent:
    event = MessageEvent(
        text=_event_text_for(command_name),
        message_type=MessageType.TEXT,
        source=MagicMock(),
        message_id="m1",
    )
    event._prepared_work_command = prepare_work_command(
        command_name,
        raw_args=_raw_args_for(command_name),
        session_id="sess-1",
        cwd="/tmp",
    )
    return event


def test_pending_event_replay_text_uses_prepared_agent_message():
    for command_name in ("handoff", "init-deep", "start-work", "ralph-loop", "ulw-loop"):
        event = _make_prepared_event(command_name)

        prepared = _get_prepared_work_command(event)

        assert prepared is not None
        assert _pending_event_replay_text(event) == prepared.agent_message
        assert _pending_event_replay_text(event) != event.text
        assert f"[OMO command {command_name}]" in _pending_event_replay_text(event)


def test_pending_event_replay_preserves_continuation_semantics_for_work_commands():
    expected = {
        "handoff": ("default", False),
        "init-deep": ("default", False),
        "start-work": ("default", False),
        "ralph-loop": ("ralph", True),
        "ulw-loop": ("ultrawork", True),
    }

    for command_name, (expected_runtime_mode, should_continue) in expected.items():
        event = _make_prepared_event(command_name)
        prepared = _get_prepared_work_command(event)
        replay_text = _pending_event_replay_text(event)

        assert prepared is not None
        assert replay_text == prepared.agent_message

        classification = preclassify_intent({"message": prepared.task_contract["task"], "task_contract": prepared.task_contract})
        assert classification.inferred_runtime_mode == expected_runtime_mode
        assert should_use_continuation_engine(classification.inferred_runtime_mode, {
            "outcomeStatus": "interrupted",
            "activeTodos": [{"id": "todo-1", "content": "Finish the delegated task", "status": "in_progress"}],
        }) is should_continue
        if command_name == "start-work":
            assert "NAMED_WORKFLOW_JSON:" in replay_text
        elif should_continue:
            assert f'"runtime_mode": "{expected_runtime_mode}"' in replay_text
        else:
            assert '"runtime_mode": "ralph"' not in replay_text
            assert '"runtime_mode": "ultrawork"' not in replay_text


def test_prepared_work_command_event_is_exempt_from_slash_discard_guard():
    for command_name in ("handoff", "init-deep", "start-work", "ralph-loop", "ulw-loop"):
        event = _make_prepared_event(command_name)
        assert _is_discardable_pending_slash_command(event.text, event) is False


def test_plain_control_slash_command_still_discarded_from_pending_replay():
    event = MessageEvent(
        text="/stop",
        message_type=MessageType.TEXT,
        source=MagicMock(),
        message_id="m2",
    )

    assert _get_prepared_work_command(event) is None
    assert _is_discardable_pending_slash_command(event.text, event) is True


def test_plain_text_and_media_replay_behavior_is_unchanged():
    text_event = MessageEvent(
        text="follow-up question",
        message_type=MessageType.TEXT,
        source=MagicMock(),
        message_id="m3",
    )
    media_event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=MagicMock(),
        message_id="m4",
        media_urls=["/tmp/image.png"],
        media_types=["image/png"],
    )

    assert _pending_event_replay_text(text_event) == "follow-up question"
    assert _pending_event_replay_text(media_event) == "[User sent an image: /tmp/image.png]"
