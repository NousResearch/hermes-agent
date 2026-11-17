"""The compressor's in-flight-task restatement is model scaffolding, not a user bubble.

After a mid-turn compaction Hermes re-appends the still-running request as a ``role=user``
row whose content leads with ``[STILL IN PROGRESS — …]`` and whose ``display_metadata``
carries ``compaction_inflight_task_restatement: True`` (``agent.message_metadata``). The row
exists so the model keeps the active task across the handoff boundary; a person never typed
it. Neither display projection checked the flag, so Desktop (``session.resume``'s
``display_projection``) and the Web UI (``GET /api/sessions/<id>/messages``) rendered it as a
normal prompt bubble (#155495).

The flag — not the header text — is the signal: a human pasting that header carries no
``display_metadata`` and must still render, which the negative case below pins.
"""

from agent.message_metadata import INFLIGHT_TASK_RESTATEMENT
from hermes_cli.web_routers.sessions import _project_for_display
from tui_gateway.server import _history_to_messages

HEADER = (
    "[STILL IN PROGRESS — this is the active request, restated after the "
    "compaction boundary because it was not finished yet. Continue it; do not "
    "start over.]"
)


def _stored_rows(*, stamped: bool) -> list[dict]:
    """The stored shape: the real opener, the compressor's replay row, the reply."""
    replay = {"role": "user", "id": 2, "message_uid": "a" * 32,
              "content": HEADER + "\nplease rename the widget", "timestamp": 2.0}
    if stamped:
        replay["display_metadata"] = {INFLIGHT_TASK_RESTATEMENT: True}
    return [
        {"role": "user", "id": 1, "message_uid": "a" * 32,
         "content": "please rename the widget", "timestamp": 1.0},
        replay,
        {"role": "assistant", "id": 3, "content": "done", "timestamp": 3.0},
    ]


def test_gateway_projection_drops_the_stamped_restatement():
    projected = _history_to_messages(_stored_rows(stamped=True))

    assert [message["text"] for message in projected] == ["please rename the widget", "done"]


def test_rest_projection_hides_the_stamped_restatement():
    rows = _stored_rows(stamped=True)

    projected = _project_for_display(rows)

    # Rows are kept (pagination offsets and "Show earlier" stay stable); the replay is typed hidden
    # the way every other display-only scaffold row is, so the client collapses it.
    assert len(projected) == len(rows)
    assert projected[0].get("display_kind") is None
    assert projected[1]["display_kind"] == "hidden"
    # The payload survives for inspect/export, and the input rows are not mutated in place.
    assert projected[1]["content"] == rows[1]["content"]
    assert "display_kind" not in rows[1]


def test_an_unstamped_row_with_the_same_header_still_renders():
    """A human pasting the header text (or a row persisted before the stamp existed) is user input."""
    rows = _stored_rows(stamped=False)

    assert [message["text"] for message in _history_to_messages(rows)] == [
        "please rename the widget",
        HEADER + "\nplease rename the widget",
        "done",
    ]
    projected = _project_for_display(rows)
    assert len(projected) == len(rows)
    assert projected[1].get("display_kind") is None


def test_active_only_keeps_the_inflight_request_without_its_frame(tmp_path):
    """A committed compaction archives the original; the replay is the only active ask."""
    from agent.context_compressor import (
        COMPRESSED_SUMMARY_METADATA_KEY, SUMMARY_PREFIX, ContextCompressor, _SUMMARY_END_MARKER,
    )
    from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_read import transcript_rows
    from tests.fixtures.storage_seed import seed_messages, seed_session

    store = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(store, "active-only", "desktop")
        seed_messages(
            store,
            "active-only",
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "please rename the widget"},
        )
        original = transcript_rows(store, "active-only")[-1]
        assert original.get("message_uid")
        carrier = {
            "role": "assistant", COMPRESSED_SUMMARY_METADATA_KEY: True,
            "content": f"{SUMMARY_PREFIX}\n## Summary\nprior work\n\n{_SUMMARY_END_MARKER}",
        }
        compressor = object.__new__(ContextCompressor)
        compressor.quiet_mode = True
        window = compressor._reappend_inflight_user_task([carrier], {**original})
        store.messages.compact("active-only", [NewMessage.from_live(message) for message in window])

        active = transcript_rows(store, "active-only", TranscriptRead(view=TranscriptView.ACTIVE))
        assert sum(
            bool((row.get("display_metadata") or {}).get(INFLIGHT_TASK_RESTATEMENT))
            for row in active
        ) == 1
        expected = ["please rename the widget"]
        assert [
            row["content"] for row in _project_for_display(active)
            if row["role"] == "user" and row.get("display_kind") != "hidden"
        ] == expected
        assert [
            row["text"] for row in _history_to_messages(active) if row["role"] == "user"
        ] == expected

        lineage = transcript_rows(store, "active-only", TranscriptRead(view=TranscriptView.DISPLAY))
        assert sum(
            row.get("content") == expected[0] and row.get("display_kind") != "hidden"
            for row in _project_for_display(lineage, lineage=True)
        ) == 1
    finally:
        store.close()
