"""REST session transcripts must not ship untyped gateway scaffold notices as user rows.

A ``[System: …]`` ``role=user`` row persisted without a ``display_kind`` is model-facing
recovery scaffolding (e.g. the stream-timeout nudge appended by the turn-truncation path).
The gateway's own history projection (``tui_gateway.session_history._history_to_messages``)
drops these rows, but this REST projection feeds the Desktop's transcript prefetch, which
addresses VISIBLE user rows by durable row id — and a scaffold row can never resolve as a
rewind/regenerate target (the gateway truncation resolver refuses it fail-closed), so
shipping it as a user row dead-ends every retry.  Typed notices (``model_switch``, …) must
keep flowing: the Desktop renders them as timeline rows.
"""

from hermes_cli.web_routers.sessions import _project_for_display


def test_rest_projection_hides_untyped_gateway_notices_and_keeps_typed_ones():
    messages = [
        {"role": "user", "id": 1, "content": "hello", "timestamp": 1.0},
        {
            "role": "user",
            "id": 2,
            "content": (
                "[System: Your previous tool call (terminal) was too large and the stream "
                "timed out before it could be delivered. Do NOT retry the same tool call "
                "with the same large content.]"
            ),
            "timestamp": 2.0,
        },
        {
            "role": "user",
            "id": 3,
            "content": "[System: The active model for this chat has changed to example.]",
            "display_kind": "model_switch",
            "timestamp": 3.0,
        },
        {"role": "assistant", "id": 4, "content": "ok", "timestamp": 4.0},
    ]

    projected = _project_for_display(messages)
    by_id = {message["id"]: message for message in projected}

    # No row is dropped: offsets/counts stay stable for pagination and "Show earlier".
    assert len(projected) == len(messages)
    # The untyped scaffold notice is hidden (the Desktop collapses `hidden` rows).
    assert by_id[2]["display_kind"] == "hidden"
    # A human user row and a typed timeline notice remain untouched.
    assert by_id[1].get("display_kind") is None
    assert by_id[3].get("display_kind") == "model_switch"
    # The payload itself is preserved — only the display kind is added.
    assert by_id[2]["content"] == messages[1]["content"]
    # Input rows are not mutated in place.
    assert "display_kind" not in messages[1]


def test_rest_projection_still_projects_compaction_summaries():
    from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

    summary = {
        "role": "user",
        "id": 10,
        "content": "[CONTEXT COMPACTION — REFERENCE ONLY] earlier turns were compacted",
        "timestamp": 5.0,
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }

    projected = _project_for_display([summary])

    assert len(projected) == 1
    # A pure handoff compacts to `hidden` — the pre-existing behavior this edit sits in.
    assert projected[0]["display_kind"] == "hidden"
