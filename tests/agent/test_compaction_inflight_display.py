"""Display projections show a compaction's re-stated in-flight request once (#131104).

When compaction leaves an unfinished request before its handoff, the compressor re-states it
after the summary, framed by ``_INFLIGHT_TASK_REPLAY_HEADER``, either as a standalone user row
or merged onto the handoff carrier. The original row stays in the display lineage (archived on
commit), so a projection that also paints the restatement shows the request twice, the second
copy wrapped in a model-only frame. Every display surface goes through the shared projection:
the Desktop's REST transcript and the gateway resume history.
"""

from __future__ import annotations

from agent.compaction_display import project_compaction_message_for_display
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    ContextCompressor,
    _INFLIGHT_TASK_REPLAY_HEADER,
    _SUMMARY_END_MARKER,
)
from hermes_cli.web_routers.sessions import _project_for_display
from tui_gateway.server import _history_to_messages

ASK = "please migrate the billing tables and tell me when it is done"


def _compressor() -> ContextCompressor:
    compressor = object.__new__(ContextCompressor)
    compressor.quiet_mode = True
    return compressor


def _carrier(role: str) -> dict:
    return {
        "role": role,
        "content": f"{SUMMARY_PREFIX}\n## Summary\nran migrations.\n\n{_SUMMARY_END_MARKER}",
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }


def _display_lineage(carrier_role: str) -> list[dict]:
    """The rows a display read sees: the archived original, then the compacted window.

    An assistant carrier ends the window on assistant, so the restatement is appended as a
    standalone user row; a user carrier gets it merged after its end marker.
    """
    original = {"role": "user", "content": ASK, "message_uid": "ask-uid"}
    window = _compressor()._reappend_inflight_user_task([_carrier(carrier_role)], {**original})
    assert any(_INFLIGHT_TASK_REPLAY_HEADER in str(m.get("content")) for m in window)
    return [original, *window, {"role": "assistant", "content": "migrated."}]


def _visible_texts(rows: list[dict]) -> list[str]:
    return [
        str(m.get("display_content", m.get("content")))
        for m in rows
        if m.get("display_kind") != "hidden" and m.get("role") == "user"
    ]


def _check(lineage: list[dict]) -> None:
    for surface in (_visible_texts(_project_for_display(lineage)),
                    [m.get("text") or str(m.get("content")) for m in _history_to_messages(lineage)
                     if m.get("role") == "user"]):
        assert sum(ASK in text for text in surface) == 1, surface
        assert not any(_INFLIGHT_TASK_REPLAY_HEADER in text for text in surface), surface


def test_a_standalone_restatement_is_shown_once_without_its_frame():
    lineage = _display_lineage("assistant")
    assert any(m.get("role") == "user" and str(m["content"]).startswith(_INFLIGHT_TASK_REPLAY_HEADER)
               for m in lineage)
    _check(lineage)


def test_a_restatement_merged_onto_the_carrier_is_shown_once_without_its_frame():
    _check(_display_lineage("user"))


def test_a_live_ask_on_the_carrier_without_the_frame_stays_visible():
    # The _force_user_leading layout carries the real live request after the end marker.
    carrier = {**_carrier("user")}
    carrier["content"] += f"\n\n{ASK}"
    projected = project_compaction_message_for_display(carrier)
    assert projected is not None and ASK in projected["content"]
