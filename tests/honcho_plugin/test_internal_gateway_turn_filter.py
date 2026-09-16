"""The internal-turn filter must reject every machine envelope, not just completions.

``process_registry_notifications`` emits three ASYNC DELEGATION envelope headers. The
filter's ASYNC branch used to require the word COMPLETE, so the early-warning
``[ASYNC DELEGATION TASK FAILED — <id>, task i/n]`` notice (one child of a fan-out
failed while its siblings keep running) passed as a genuine user turn and leaked
delegation traffic into the user peer as durable memory. These tests build every
envelope through the real formatter and pin that each is rejected, while a human
discussing the marker mid-message stays valid input.
"""

from plugins.memory.honcho import _is_internal_gateway_turn
from tools.process_registry_notifications import format_process_notification


def _task_failure_evt() -> dict:
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_x",
        "task_failure_notice": True,
        "results": [
            {
                "task_index": 1,
                "goal": "b",
                "status": "error",
                "error": "401 authentication_error",
                "duration_seconds": 12.5,
            }
        ],
        "goals": ["a", "b", "c"],
        "n_tasks": 3,
    }


def _batch_complete_evt() -> dict:
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_x",
        "is_batch": True,
        "results": [{"task_index": 0, "status": "completed", "summary": "ok"}],
        "goals": ["a"],
        "dispatched_at": 1.0,
        "role": "leaf",
        "model": "m",
    }


def _single_complete_evt() -> dict:
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_x",
        "status": "completed",
        "summary": "done",
        "dispatched_at": 1.0,
        "role": "leaf",
        "model": "m",
    }


def test_every_async_delegation_envelope_is_rejected() -> None:
    for evt in (_task_failure_evt(), _batch_complete_evt(), _single_complete_evt()):
        text = format_process_notification(evt)
        assert text is not None
        assert _is_internal_gateway_turn(text), (
            f"envelope leaked past the filter: {text.splitlines()[0]}"
        )


def test_task_failure_header_shape_is_pinned() -> None:
    # Pin the exact header the formatter produces so the filter regex stays in sync with it.
    text = format_process_notification(_task_failure_evt())
    assert text.startswith("[ASYNC DELEGATION TASK FAILED — deleg_x, task 2/3]")


def test_human_discussing_the_marker_mid_message_stays_valid_input() -> None:
    assert not _is_internal_gateway_turn(
        "What does [ASYNC DELEGATION TASK FAILED — deleg_x, task 2/3] mean?"
    )
    assert not _is_internal_gateway_turn(
        "I saw an [ASYNC DELEGATION BATCH COMPLETE — deleg_x] notice earlier"
    )
