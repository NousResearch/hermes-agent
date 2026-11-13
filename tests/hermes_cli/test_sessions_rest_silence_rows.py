"""Transcript reloads must not render intentional-silence replies as chat bubbles.

A completed turn whose whole reply is ``[SILENT]`` / ``NO_REPLY`` is suppressed at delivery by the
gateway (and blanked for live Bot Chat completions), but the row stays persisted for role
alternation. Both history projections must therefore hide it, while real replies, replies that merely
mention the marker, and assistant rows that carry tool calls keep rendering.
"""

from hermes_cli.web_routers.sessions import _project_for_display
from tui_gateway.server import _history_to_messages


def _rows():
    return [
        {"role": "user", "id": 1, "content": "[System: background process proc_1 completed]",
         "display_kind": "process_complete", "timestamp": 1.0},
        {"role": "assistant", "id": 2, "content": "[SILENT]", "timestamp": 2.0},
        {"role": "assistant", "id": 3, "content": " no_reply. ", "timestamp": 3.0},
        {"role": "assistant", "id": 4, "content": "Done. I would normally reply [SILENT] here.",
         "timestamp": 4.0},
        {"role": "user", "id": 5, "content": "status?", "timestamp": 5.0},
        {"role": "assistant", "id": 6, "content": "All green.", "timestamp": 6.0},
    ]


def test_rest_projection_hides_bare_silence_markers():
    rows = _rows()
    projected = _project_for_display(rows)
    by_id = {m["id"]: m for m in projected}
    assert len(projected) == len(rows)  # marked, never dropped: pagination offsets stay stable
    assert by_id[2]["display_kind"] == "hidden"
    assert by_id[3]["display_kind"] == "hidden"
    assert by_id[4].get("display_kind") is None  # prose that mentions the marker still renders
    assert by_id[6].get("display_kind") is None
    assert by_id[2]["content"] == "[SILENT]"  # payload preserved
    assert "display_kind" not in rows[1]  # input not mutated


def test_rest_projection_keeps_tool_call_rows_and_typed_rows():
    rows = [
        {"role": "assistant", "id": 1, "content": "[SILENT]",
         "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "assistant", "id": 2, "content": "NO_REPLY", "display_kind": "some_kind"},
    ]
    by_id = {m["id"]: m for m in _project_for_display(rows)}
    assert by_id[1].get("display_kind") is None
    assert by_id[2]["display_kind"] == "some_kind"


def test_resume_projection_skips_bare_silence_markers():
    texts = [m.get("text") for m in _history_to_messages(_rows())]
    assert "[SILENT]" not in texts
    assert not any(isinstance(t, str) and t.strip().lower().startswith("no_reply") for t in texts)
    assert "Done. I would normally reply [SILENT] here." in texts
    assert "All green." in texts
