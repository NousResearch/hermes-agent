"""REST session transcripts must not ship untyped gateway scaffold notices as user rows.

A ``[System: …]`` ``role=user`` row persisted without a ``display_kind`` is model-facing
recovery scaffolding (e.g. the stream-timeout nudge appended by the turn-truncation path).
The gateway's own history projection (``tui_gateway.session_history._history_to_messages``)
drops these rows, but this REST projection feeds the Desktop's transcript prefetch, which
addresses VISIBLE user rows by durable row id — and a scaffold row can never resolve as a
rewind/regenerate target (the gateway truncation resolver refuses it fail-closed), so
shipping it as a user row dead-ends every retry.  Typed notices (``model_switch``, …) must
keep flowing: the Desktop renders them as timeline rows.

A stored ``role=user`` ``display_kind=steer`` row must ship the user's own words as
``display_content`` (the same unwrap ``session.resume`` applies) — marker-wrapped content
never pairs with the optimistic plain-text copy, so Desktop renders the correction twice
and reorders it (#121978).
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


def test_rest_projection_unwraps_steer_rows_for_display():
    """A stored steer row displays the user's words, not the marker wrapper (#121978).

    The gateway's session.resume projection unwraps the model-facing steer markers; the
    REST projection must agree, or the persisted row ships marker-wrapped and never pairs
    with the optimistic plain-text copy Desktop already holds.
    """
    from agent.prompt_builder import STEER_MARKER_CLOSE, STEER_MARKER_OPEN

    steer_row = {
        "role": "user",
        "id": 21,
        "content": f"{STEER_MARKER_OPEN}\nfocus on the parser bug\n{STEER_MARKER_CLOSE}",
        "display_kind": "steer",
        "timestamp": 6.0,
    }
    plain_row = {"role": "user", "id": 22, "content": "unrelated text", "timestamp": 7.0}

    projected = _project_for_display([steer_row, plain_row])

    assert len(projected) == 2  # no row dropped: pagination offsets stay stable
    by_id = {message["id"]: message for message in projected}
    # The steer row ships its own words as display_content; the marker wrapper stays in
    # `content` for export/inspection consumers.
    assert by_id[21]["display_content"] == "focus on the parser bug"
    assert by_id[21]["content"] == steer_row["content"]
    assert by_id[21]["display_kind"] == "steer"
    # A plain user row is untouched — no display_content sidecar.
    assert "display_content" not in by_id[22]
    # Input rows are not mutated in place.
    assert "display_content" not in steer_row


def test_rest_projection_keeps_steer_row_without_marker_as_is():
    """A steer row whose content lost the marker wrapper (legacy persistence) falls through
    to the normal path instead of emitting an empty display_content."""
    steer_row = {"role": "user", "id": 31, "content": "plain legacy text", "display_kind": "steer",
                 "timestamp": 8.0}

    projected = _project_for_display([steer_row])

    assert len(projected) == 1
    assert "display_content" not in projected[0]
    assert projected[0]["content"] == "plain legacy text"
