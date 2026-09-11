"""Alternation repair must not turn a current-user override into history loss."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import repair_message_sequence_with_cursor
from agent.session_persistence import SessionPersistenceMixin, durable_user_row_content
from agent.turn_context_compaction import _reanchor


@pytest.mark.parametrize(
    ("bodies", "current_idx", "clean_current", "expected_clean"),
    [
        (["failed question", "resume question"], 1, "resume question", "failed question\n\nresume question"),
        (["earlier", "API guidance\n\nquestion"], 1, "question", "earlier\n\nquestion"),
        (["same", "same"], 1, "same", "same\n\nsame"),
        (["first", "second", "API guidance\n\nquestion"], 2, "question", "first\n\nsecond\n\nquestion"),
        (["earlier", "API guidance\n\nquestion", "follow-up"], 1, "question", "earlier\n\nquestion\n\nfollow-up"),
        (["API guidance\n\nquestion", "follow-up"], 0, "question", "question\n\nfollow-up"),
        (["earlier", "API guidance"], 1, "", "earlier"),
    ],
)
def test_merged_user_override_preserves_other_inputs_and_survivor_metadata(
    bodies, current_idx, clean_current, expected_clean,
):
    messages = [
        {"role": "user", "content": body, "timestamp": 10 + idx, "platform_message_id": f"event-{idx}"}
        for idx, body in enumerate(bodies)
    ]
    agent = SimpleNamespace(
        _persist_user_message_idx=current_idx,
        _persist_user_message_override=clean_current,
        _persist_user_message_timestamp=messages[current_idx]["timestamp"],
        _persist_user_message_platform_id=messages[current_idx]["platform_message_id"],
        _last_flushed_db_idx=0,
    )
    current_wire_text = messages[current_idx]["content"]
    expected_wire = "\n\n".join(bodies)

    assert repair_message_sequence_with_cursor(agent, messages) == len(bodies) - 1
    assert _reanchor(agent, messages, current_wire_text) == 0
    assert messages[0]["content"] == expected_wire

    clean, api = durable_user_row_content(agent, messages[0], messages[0]["content"], None)
    assert clean == expected_clean
    assert api == (expected_wire if expected_wire != expected_clean else None)
    if api is not None:
        messages[0]["api_content"] = api
    before_finalize = deepcopy(messages)

    SessionPersistenceMixin._apply_persist_user_message_override(agent, messages)

    assert messages[0]["content"] == expected_clean
    assert messages[0]["timestamp"] == 10
    assert messages[0]["platform_message_id"] == "event-0"
    assert messages[0].get("api_content") == before_finalize[0].get("api_content")
    assert repair_message_sequence_with_cursor(agent, messages) == 0
    SessionPersistenceMixin._apply_persist_user_message_override(agent, messages)
    assert messages[0]["content"] == expected_clean


def test_ordinary_user_override_still_cleans_current_input():
    messages = [
        {"role": "user", "content": "API guidance\n\nquestion", "timestamp": 10},
    ]
    agent = SimpleNamespace(
        _persist_user_message_idx=0, _persist_user_message_override="question",
        _persist_user_message_timestamp=20, _persist_user_message_platform_id="current-event",
        _last_flushed_db_idx=0,
    )
    assert repair_message_sequence_with_cursor(agent, messages) == 0
    SessionPersistenceMixin._apply_persist_user_message_override(agent, messages)
    assert messages[0]["content"] == "question"
    assert messages[0]["timestamp"] == 20
    assert messages[0]["platform_message_id"] == "current-event"


def test_native_user_blocks_are_not_merged_or_replaced_by_text_override():
    blocks = [{"type": "text", "text": "caption"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    messages = [{"role": "user", "content": "earlier"}, {"role": "user", "content": deepcopy(blocks)}]
    agent = SimpleNamespace(
        _persist_user_message_idx=1, _persist_user_message_override="caption",
        _persist_user_message_timestamp=20, _persist_user_message_platform_id=None,
        _last_flushed_db_idx=0,
    )
    assert repair_message_sequence_with_cursor(agent, messages) == 0
    SessionPersistenceMixin._apply_persist_user_message_override(agent, messages)
    assert messages[0]["content"] == "earlier"
    assert messages[1]["content"] == blocks
