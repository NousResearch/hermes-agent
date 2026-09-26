r"""The internal-turn filter must reject every machine envelope, not just completions.

``process_registry_notifications`` emits three ASYNC DELEGATION envelope headers. The
filter's ASYNC branch used to require the word COMPLETE, so the early-warning
``[ASYNC DELEGATION TASK FAILED — <id>, task i/n]`` notice (one child of a fan-out
failed while its siblings keep running) passed as a genuine user turn and leaked
delegation traffic into the user peer as durable memory. These tests build every
envelope through the real formatter and pin that each is rejected, while a human
discussing the marker mid-message stays valid input.

The ``[IMPORTANT: ...]`` family had two sibling gaps: the watch-pattern branch matched
``\d+`` while real session ids are ``proc_`` + 12 hex chars (so it could never fire),
and neither the multi-completion batch header nor the watch-disabled/overflow notices
were listed at all. The bare ``[IMPORTANT: Background process `` prefix now matches the
convention already used by ``hermes_state_timeline`` and ``conversation_compression``.
"""

from plugins.memory.honcho import _is_internal_gateway_turn
from tools.process_registry_notifications import (
    ProcessNotificationBatch,
    format_process_notification,
)


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


def _watch_match_evt() -> dict:
    return {
        "type": "watch_match",
        "session_id": "proc_4dae56ca81f6",
        "pattern": "ERROR",
        "command": "tail -f build.log",
        "output": "ERROR: compile failed",
    }


def _completion_evt(session_id: str) -> dict:
    return {
        "type": "completion",
        "session_id": session_id,
        "command": "make test",
        "exit_code": 0,
        "output": "all green",
    }


def _watch_breaker_evt(evt_type: str, message: str) -> dict:
    return {"type": evt_type, "message": message}


def test_every_important_envelope_is_rejected() -> None:
    events = (
        _watch_match_evt(),
        _completion_evt("proc_0123456789ab"),
        _watch_breaker_evt(
            "watch_disabled",
            "Watch patterns disabled for process proc_4dae56ca81f6 — reached the "
            "lifetime cap of 3 delivered matches. Falling back to notify_on_complete "
            "semantics; you'll get exactly one notification when the process exits.",
        ),
        _watch_breaker_evt(
            "watch_overflow_tripped",
            "Watch-pattern overflow: >20 notifications in 60s across all processes. "
            "Suppressing further watch_match events for 120s.",
        ),
        _watch_breaker_evt(
            "watch_overflow_released",
            "Watch-pattern notifications resumed. 7 match event(s) were suppressed "
            "during the flood.",
        ),
    )
    for evt in events:
        text = format_process_notification(evt)
        assert text is not None
        assert _is_internal_gateway_turn(text), (
            f"envelope leaked past the filter: {text.splitlines()[0]}"
        )


class _NeverConsumed:
    @staticmethod
    def is_completion_consumed(session_id: str) -> bool:
        return False


def test_multi_completion_batch_header_is_rejected() -> None:
    sids = ("proc_0123456789ab", "proc_cdef01234567")
    batch = ProcessNotificationBatch(
        tuple(
            ({"session_id": sid}, format_process_notification(_completion_evt(sid)))
            for sid in sids
        )
    )
    text = batch.render(_NeverConsumed)
    assert text is not None
    assert text.startswith("[IMPORTANT: 2 background processes completed.")
    assert _is_internal_gateway_turn(text)


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
    assert not _is_internal_gateway_turn(
        'Why did [IMPORTANT: Background process proc_4dae56ca81f6 matched '
        'watch pattern "ERROR"] fire twice?'
    )
