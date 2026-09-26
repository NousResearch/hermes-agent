"""Background-process notifications must never be ingested as durable user memory.

``_INTERNAL_GATEWAY_TURN_RE`` matched only the ``matched watch pattern`` shape, so every
other envelope the same emitter produces — plain completions above all — reached
``sync_turn`` as a genuine user turn and was written to the user peer, where the backend
derived "facts" about the human from machine output.

The branch is now keyed on the envelope PREFIX rather than on the status verb that follows
it. ``_completion_status`` renders six different verbs today and ``_REASON_STATUS`` is
designed to grow; an alternation of verbs leaks every status added after it is written.
This is also the classification ``agent/context_compressor.py`` already uses
(``_BACKGROUND_PROCESS_NOTIFICATION_PREFIX``).

Every expectation here is produced by the real emitters — ``format_process_notification``,
``ProcessNotificationBatch.render`` and the gateway's batch headers — never by a
hand-written literal, so a change to the envelope shape fails these tests instead of
silently drifting away from them.
"""

import pytest

from plugins.memory.honcho import _is_internal_gateway_turn
from tools.process_registry_notifications import (
    ProcessNotificationBatch,
    _completion_status,
    format_process_notification,
)

# `proc_` + 12 hex chars, the shape tools/process_registry.py actually mints. The previous
# `\d+` in the watch-pattern branch could never match a real session id.
SESSION_ID = "proc_4dae56ca81f6"
OTHER_SESSION_ID = "proc_e51ff36350af"


def _completion_event(**overrides) -> dict:
    return {
        "type": "completion",
        "session_id": SESSION_ID,
        "command": "make test",
        "exit_code": 0,
        "output": "all green",
        **overrides,
    }


# One event per branch of _completion_status(), so a new status in _REASON_STATUS shows up
# here as an uncovered case rather than as a silent memory leak.
_COMPLETION_STATUS_EVENTS = {
    "completed normally": _completion_event(exit_code=0),
    "exited": _completion_event(exit_code=1),
    "failed to start": _completion_event(completion_reason="failed_start"),
    "marked lost because the process backend disappeared": _completion_event(
        completion_reason="lost", exit_code="?"
    ),
    "terminated by Hermes": _completion_event(completion_reason="killed", exit_code=-15),
    "terminated by a subagent (sa-7)": _completion_event(
        completion_reason="killed", termination_source="a subagent (sa-7)", exit_code=-15
    ),
}


class _NeverConsumed:
    """Registry stub: nothing has been consumed, so render() keeps every notification."""

    @staticmethod
    def is_completion_consumed(session_id: str) -> bool:
        return False


@pytest.mark.parametrize("status,event", sorted(_COMPLETION_STATUS_EVENTS.items()))
def test_every_completion_status_is_filtered(status: str, event: dict) -> None:
    # Pin the fixture against the real status renderer: if _completion_status stops
    # producing this verb, the test is stale and says so instead of passing vacuously.
    assert _completion_status(event) == status
    text = format_process_notification(event)
    assert text is not None
    assert _is_internal_gateway_turn(text), f"completion envelope leaked: {text.splitlines()[0]}"


def test_watch_pattern_notification_is_filtered() -> None:
    text = format_process_notification({
        "type": "watch_match",
        "session_id": SESSION_ID,
        "pattern": "ERROR",
        "command": "tail -f build.log",
        "output": "ERROR: compile failed",
    })
    assert text is not None
    assert _is_internal_gateway_turn(text)


def test_heartbeat_notification_is_filtered() -> None:
    """Heartbeats are the one envelope with no ``IMPORTANT:`` marker at all."""
    text = format_process_notification({
        "type": "heartbeat",
        "session_id": SESSION_ID,
        "seq": 3,
        "elapsed": 180.0,
        "interval": 60,
        "command": "make test",
        "output": "still compiling",
    })
    assert text is not None
    assert _is_internal_gateway_turn(text)


@pytest.mark.parametrize("event_type,message", [
    ("watch_disabled",
     f"Watch patterns disabled for process {SESSION_ID} — reached the lifetime cap of 3 "
     "delivered matches. Falling back to notify_on_complete semantics."),
    ("watch_overflow_tripped",
     "Watch-pattern overflow: >20 notifications in 60s across all processes. "
     "Suppressing further watch_match events for 120s."),
    ("watch_overflow_released",
     "Watch-pattern notifications resumed. 7 match event(s) were suppressed during the flood."),
])
def test_watch_breaker_notifications_are_filtered(event_type: str, message: str) -> None:
    text = format_process_notification({"type": event_type, "message": message})
    assert text is not None
    assert _is_internal_gateway_turn(text)


def test_registry_batch_header_is_filtered() -> None:
    """Several completions in one turn are rendered under a batch header."""
    batch = ProcessNotificationBatch(tuple(
        ({"session_id": sid}, format_process_notification(_completion_event(session_id=sid)))
        for sid in (SESSION_ID, OTHER_SESSION_ID)
    ))
    text = batch.render(_NeverConsumed)
    assert text is not None
    # render() only emits the header for more than one live notification.
    assert text != format_process_notification(_completion_event())
    assert _is_internal_gateway_turn(text)


def test_gateway_coalesced_completion_header_is_filtered() -> None:
    """The gateway coalesces completions under its own header, distinct from the registry's."""
    from gateway.run_notifications import GatewayNotificationsMixin

    entries = [
        (format_process_notification(_completion_event(session_id=sid)),
         _completion_event(session_id=sid), None)
        for sid in (SESSION_ID, OTHER_SESSION_ID)
    ]
    text = GatewayNotificationsMixin._format_coalesced_process_completions(entries)
    assert _is_internal_gateway_turn(text)


def test_gateway_delegation_group_header_is_filtered() -> None:
    """Grouped subagent delegations reuse the batch-header shape with a different noun.

    The header is built inline in ``_deliver_async_delegation_group_scoped``, which needs a
    live gateway to reach, so the source line is read from that function rather than
    retyped — a literal here would drift from the emitter it is supposed to pin.
    """
    import inspect
    import re

    from gateway.run_notifications import GatewayNotificationsMixin

    source = inspect.getsource(
        GatewayNotificationsMixin._deliver_async_delegation_group_scoped
    )
    match = re.search(r'f"(\[IMPORTANT: \{len\(blocks\)\} background subagent delegations )"', source)
    assert match, "delegation group header moved; update this test against the emitter"
    header = match.group(1).replace("{len(blocks)}", "3") + "completed for this session.]"
    assert _is_internal_gateway_turn(header)


def test_human_prose_mentioning_a_notification_is_not_filtered() -> None:
    """The ``^`` anchor is load-bearing: real questions about these strings are real input."""
    completion_text = format_process_notification(_completion_event())
    assert not _is_internal_gateway_turn(f"Why did this fire twice? {completion_text}")
    assert not _is_internal_gateway_turn(
        "A background process completed normally but the artifact is missing — can you look?"
    )
    assert not _is_internal_gateway_turn(
        f"The [Background process {SESSION_ID} heartbeat lines are spamming me, can we turn them off?"
    )


def test_bracketed_question_without_a_session_id_is_not_filtered() -> None:
    """A user can open a message with the bracketed phrase and still mean it literally.

    This is why the branch matches the prefix plus the session-id token rather than the
    bare prefix: the envelope always names a process, a question about the envelope does
    not. ``tests/plugins/test_honcho_startup_fail_open.py`` pins the same two strings.
    """
    assert not _is_internal_gateway_turn("[IMPORTANT: Background process — what does that mean?]")
    assert not _is_internal_gateway_turn(
        "IMPORTANT: Background process — can you explain what that means?"
    )


def test_skill_invocation_scaffold_is_not_filtered() -> None:
    """A sibling ``[IMPORTANT: ...]`` message that this filter must not swallow.

    Skill invocations are stripped upstream by ``MemoryManager._strip_skill_scaffolding``,
    which keeps the user's real words. Matching them here would drop the whole turn.
    """
    from agent.skill_commands import _SKILL_INVOCATION_PREFIX

    assert not _is_internal_gateway_turn(
        f'{_SKILL_INVOCATION_PREFIX}"pdf" skill. Its content follows.]\nsplit this PDF'
    )


@pytest.mark.parametrize("text", [
    "[ASYNC COMPLETE — deleg_x]",
    "[ASYNC DELEGATION BATCH COMPLETE — deleg_x]",
    "[CONTEXT COMPACTION: older turns summarized]",
    "[CONTEXT SUMMARY]: previous context",
    "[PRIOR CONTEXT from an earlier session]",
    "[Your active task list was preserved across context compression]",
    "A background fan-out of 3 subagent(s) you dispatched earlier has finished.",
    "A background subagent you dispatched earlier has finished.",
])
def test_preexisting_branches_still_filter(text: str) -> None:
    assert _is_internal_gateway_turn(text)
