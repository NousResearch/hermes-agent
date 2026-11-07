"""A compaction carrier whose tail is a producer notice projects as that notice, not a user bubble.

``ContextCompressor._merge_summary_into_tail_row`` prepends the summary onto the tail row when no
standalone summary row can alternate. When that tail row is an async-delegation or background-process
notice, the persisted carrier has no ``display_kind`` (it is a DB sidecar, not part of the message
dict), so the display projection used to hand the raw producer envelope to clients as a plain user
message.
"""

from __future__ import annotations

from agent.compaction_display import project_compaction_message_for_display
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _SUMMARY_END_MARKER,
)
from tools.process_registry_notifications import (
    async_delegation_display_text,
    format_process_notification,
    process_completion_display_text,
)


def _carrier(tail: str, **extra) -> dict:
    content = f"{SUMMARY_PREFIX}\n\n{HISTORICAL_TASK_HEADING}\nold work\n\n{_SUMMARY_END_MARKER}\n\n{tail}"
    return {"role": "user", "content": content, COMPRESSED_SUMMARY_METADATA_KEY: True, **extra}


_BATCH_EVENT = {
    "type": "async_delegation",
    "delegation_id": "deleg_test",
    "is_batch": True,
    "goals": ["Summarize the release notes. Return JSON.", "List the failing checks. Return JSON."],
    "results": [
        {"task_index": 0, "status": "completed", "summary": '{"items": [1, 2, 3]}', "duration_seconds": 4},
        {"task_index": 1, "status": "failed", "error": "boom", "duration_seconds": 2},
    ],
    "completed_at": 1_700_000_000,
}

_SINGLE_EVENT = {
    "type": "async_delegation",
    "delegation_id": "deleg_one",
    "goal": "Check the build log",
    "status": "completed",
    "summary": '{"ok": true}',
    "completed_at": 1_700_000_000,
}

_PROCESS_EVENT = {
    "type": "completion",
    "session_id": "proc_test",
    "command": "pytest -q tests/unit",
    "exit_code": 1,
    "output": '{"passed": 3, "failed": 1}',
}


def test_async_batch_tail_projects_as_delegation_notice():
    envelope = format_process_notification(_BATCH_EVENT)
    assert envelope.startswith("[ASYNC DELEGATION BATCH COMPLETE")

    projected = project_compaction_message_for_display(_carrier(envelope))

    assert projected is not None
    assert projected["display_kind"] == "async_delegation_complete"
    assert projected["content"] == envelope
    metadata = projected["display_metadata"]
    assert metadata["display_text"] == async_delegation_display_text(_BATCH_EVENT)
    assert metadata["task_count"] == 2


def test_single_async_delegation_tail_projects_as_delegation_notice():
    envelope = format_process_notification(_SINGLE_EVENT)
    assert envelope.startswith("[ASYNC DELEGATION COMPLETE")

    projected = project_compaction_message_for_display(_carrier(envelope))

    assert projected["display_kind"] == "async_delegation_complete"
    assert projected["content"] == envelope
    assert projected["display_metadata"]["display_text"] == async_delegation_display_text(_SINGLE_EVENT)


def test_background_process_tail_projects_as_process_notice():
    envelope = format_process_notification(_PROCESS_EVENT)
    assert envelope.startswith("[IMPORTANT: Background process ")

    projected = project_compaction_message_for_display(_carrier(envelope))

    assert projected is not None
    assert projected["display_kind"] == "process_complete"
    assert projected["content"] == envelope
    assert projected["display_metadata"]["display_text"] == process_completion_display_text([_PROCESS_EVENT])


def test_process_batch_tail_projects_as_process_notice():
    second = {**_PROCESS_EVENT, "session_id": "proc_two", "exit_code": 0}
    envelope = "\n\n".join((
        "[IMPORTANT: 2 background processes completed. Treat these results as one batch and give one "
        "consolidated response; preserve failures and actionable results.]",
        format_process_notification(_PROCESS_EVENT),
        format_process_notification(second),
    ))

    projected = project_compaction_message_for_display(_carrier(envelope))

    assert projected["display_kind"] == "process_complete"
    assert projected["display_metadata"]["display_text"] == process_completion_display_text(
        [_PROCESS_EVENT, second])


def test_real_user_ask_after_marker_stays_a_plain_user_row():
    ask = "Please rerun the [ASYNC DELEGATION BATCH COMPLETE] parser against the fixtures"

    projected = project_compaction_message_for_display(_carrier(ask))

    assert projected is not None
    assert projected["content"] == ask
    assert "display_kind" not in projected
    assert "display_metadata" not in projected


def test_carrier_keeps_a_producer_kind_it_already_has():
    """Compatible with sidecar recovery (#136422): a carrier that already names its producer kind keeps it."""
    envelope = format_process_notification(_BATCH_EVENT)
    metadata = {"display_text": "Subagent Tasks Completed: recovered label (2 tasks)", "task_count": 2}

    projected = project_compaction_message_for_display(
        _carrier(envelope, display_kind="async_delegation_complete", display_metadata=metadata))

    assert projected["display_kind"] == "async_delegation_complete"
    assert projected["display_metadata"] == metadata


def test_carrier_without_producer_envelope_still_drops_internal_kind():
    projected = project_compaction_message_for_display(_carrier("a live ask", display_kind="hidden"))

    assert projected is not None
    assert "display_kind" not in projected


def test_dashboard_route_paints_the_notice_not_a_user_bubble():
    """The dashboard session route the Desktop reads keeps the recovered notice kind and its label."""
    from hermes_cli.web_routers.sessions import _project_for_display

    envelope = format_process_notification(_BATCH_EVENT)
    (projected,) = _project_for_display([{"id": 7, **_carrier(envelope)}])

    assert projected["display_kind"] == "async_delegation_complete"
    assert projected["display_content"] == envelope
    assert projected["display_metadata"]["task_count"] == 2
