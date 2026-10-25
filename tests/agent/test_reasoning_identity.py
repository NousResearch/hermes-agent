"""Native IDs are metadata for existing reasoning, never a new source of visible text."""

from copy import deepcopy
import json

import pytest

from agent.history_commentary import project_history_commentary
from agent.reasoning_identity import native_reasoning_items
from agent.stream_delivery import StreamDeliveryMixin


def summaries():
    return [
        {
            "type": "reasoning",
            "id": "rs_a",
            "encrypted_content": "opaque",
            "summary": [
                {"type": "summary_text", "text": "Inspect"},
                {"type": "summary_text", "text": "Check"},
            ],
        },
        {"type": "compaction", "encrypted_content": "checkpoint"},
        {
            "type": "reasoning",
            "id": "rs_b",
            "summary": [{"type": "summary_text", "text": "Verify"}],
        },
    ]


@pytest.mark.parametrize("encode", [lambda value: value, json.dumps])
def test_identity_decorates_only_exact_displayed_reasoning_without_changing_replay(
    tmp_path, encode
):
    (tmp_path / "config.yaml").write_text("display:\n  show_commentary: false\n")
    row = {
        "role": "assistant",
        "content": "Done.",
        "reasoning": "Inspect\nCheck\n\nVerify\n\nPublic.",
        "codex_reasoning_items": encode(summaries()),
        "codex_message_items": encode([
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Public."}],
            }
        ]),
    }
    original = deepcopy(row)
    projected = project_history_commentary([row], home=tmp_path)[0]
    assert projected["display_commentary"] == []
    assert projected["display_reasoning"] == "Inspect\nCheck\n\nVerify"
    assert [item["id"] for item in projected["display_reasoning_items"]] == [
        "rs_a",
        "rs_b",
    ]
    assert "encrypted_content" not in json.dumps(projected["display_reasoning_items"])
    assert row == original
    assert (
        native_reasoning_items(
            summaries(), "Inspect\nCheck\n\nVerify\n\nOther analysis"
        )
        == []
    )
    malformed = summaries()
    malformed[-1]["id"] = "rs_a"
    assert native_reasoning_items(malformed, "Inspect\nCheck\n\nVerify") == []


def test_identified_delta_keeps_writer_fencing_legacy_fallback_and_observer_once():
    agent = StreamDeliveryMixin()
    structured, legacy, hooks = [], [], []
    agent.reasoning_event_callback = lambda *args: structured.append(args)
    agent.reasoning_callback = legacy.append
    agent._stream_reasoning_hooks_enabled = True
    agent._enqueue_stream_hook = lambda *args, **kwargs: hooks.append((args, kwargs))
    agent._claim_stream_writer()
    for phase, text in [("start", ""), ("delta", "Inspect"), ("end", "")]:
        agent._fire_reasoning_event(phase, "rs:summary:0", text)
    assert structured == [
        ("start", "rs:summary:0", ""),
        ("delta", "rs:summary:0", "Inspect"),
        ("end", "rs:summary:0", ""),
    ]
    assert legacy == []
    assert len(hooks) == 1 and hooks[0][1]["delta"] == "Inspect"
    # Another writer has claimed the sink; this thread's token is now stale.
    agent._stream_writer_token += 1
    for phase in ("start", "delta", "end"):
        agent._fire_reasoning_event(phase, "stale", "Must not appear")
    assert len(structured) == 3 and len(hooks) == 1
    agent._claim_stream_writer()

    def broken(*_args):
        raise RuntimeError("display unavailable")

    agent.reasoning_event_callback = broken
    agent._fire_reasoning_event("delta", "rs:summary:1", "Fallback")
    assert legacy == ["Fallback"]
    assert len(hooks) == 2


def test_native_reasoning_obeys_muted_notification_turn_and_restores_delivery():
    from agent.notification_presentation import (
        event_presentation_muted,
        notification_turn,
    )

    agent = StreamDeliveryMixin()
    seen = []

    def callback(*args):
        seen.append(args)

    agent.reasoning_event_callback = callback
    agent.reasoning_callback = lambda text: seen.append(("legacy", text))
    agent._stream_reasoning_hooks_enabled = False
    agent._claim_stream_writer()
    phases = [("start", ""), ("delta", "Diagnostic only"), ("end", "")]
    with notification_turn(agent, muted=True, session_id="owner"):
        for phase, text in phases:
            agent._fire_reasoning_event(phase, "rs_muted", text)
        assert seen == []
        for phase, _ in phases:
            assert event_presentation_muted("reasoning." + phase, "owner")
            assert not event_presentation_muted("reasoning." + phase, "other")
        assert not event_presentation_muted("notification.clear", "owner")
    assert agent.reasoning_event_callback is callback
    for phase, text in phases:
        agent._fire_reasoning_event(phase, "rs_restored", text)
        assert not event_presentation_muted("reasoning." + phase, "owner")
    assert seen == [(phase, "rs_restored", text) for phase, text in phases]
