"""A failed async delegation wakes its unviewed local parent as a turn instead of becoming a note.

#134889 LC-7 writes a completion into a local session no viewer is watching as a note, with no model
call. That is right for a result the parent will read next time, but a failed subagent leaves the
parent's plan stalled until a human returns: a provider refusal (``content_policy_blocked``) stopped a
background subagent and the parent sat on the note for 48 minutes. A final delegation result with a
failed child now runs as the parent's turn, so it can re-dispatch (on another model) or report, as a
successful one does (test_unviewed_completion_resumes_parent.py). The early per-task notice stays a
note. So do a manual cron run's result, a failure in a chat its viewer closed, and a failure past the
unattended-wake limit.
"""
import pytest

from tests.gateway.test_completion_admission import pending
from tests.gateway.test_session_end_delegations import _admissions, _local
from tests.gateway.test_unviewed_completion_notes import _human, _target, _turns
from tools import async_delegation as delegation


@pytest.fixture(autouse=True)
def _no_delegations():
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


_REFUSAL = ("content_policy_blocked: This content was flagged for possible cybersecurity risk. "
            "If this seems wrong, try rephrasing your request.")


def _batch(route, name, *statuses):
    """A finished one-unit batch whose children ended in *statuses*, persisted like a real dispatch."""
    results = [{"task_index": i, "status": status, "summary": None if status != "completed" else f"RESULT_{name}_{i}",
                "error": _REFUSAL if status == "error" else (None if status == "completed" else status),
                "failure_reason": "content_policy_blocked" if status == "error" else None, "api_calls": 7}
               for i, status in enumerate(statuses)]
    evt = {"type": "async_delegation", "is_batch": True, "session_key": route, "delegation_id": name,
           "goals": [f"RESULT_{name} goal {i}" for i in range(len(statuses))], "results": results,
           "status": "completed", "dispatched_at": 1700000000.0, "completed_at": 1700000060.0}
    delegation._persist_dispatch(evt)
    delegation._persist_completion(evt, {"results": results})
    return evt


def _notes(authority, target, name):
    return [m for m in authority.db.get_messages(target)
            if name in str(m.get("content")) and (m.get("display_metadata") or {}).get("completion_note")]


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses,turn", [
    (("error",), True),                     # the reported case: the only child refused
    (("timeout",), True),
    (("completed", "failed"), True),        # one failed sibling is enough
    (("completed",), True),                 # a finished result resumes the parent too
    (("interrupted",), True),               # unbounded like a success: a stop is not a failure
    (("cancelled", "completed"), True),
], ids=["refused", "timeout", "mixed", "success", "interrupted", "cancelled"])
async def test_unviewed_failed_delegation_runs_as_a_turn(tmp_path, monkeypatch, statuses, turn):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        assert not authority.sessions[ref.session_id].subscribers, "no viewer is watching"
        evt = dict(_batch(route, "deleg_fail", *statuses), parent_session_id=target)
        assert await runner._deliver_async_delegation_group([evt]) is True
        seen = _turns(runner, authority, target, "deleg_fail")
        await authority._drain(ref)
        assert len(seen) == int(turn)
        assert len(_notes(authority, target, "deleg_fail")) == int(not turn)
        if turn:
            assert "[ASYNC DELEGATION BATCH COMPLETE — deleg_fail]" in seen[0][0]
        assert [r["outcome"] for r in _admissions(authority, ref)] == ["completed"]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_failed_sibling_in_a_coalesced_group_wakes_the_parent(tmp_path, monkeypatch):
    """Two units of one session coalesce into one admission: the failure of the non-primary decides too."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        ok = dict(pending(route, "deleg_ok"), parent_session_id=target)
        bad = dict(_batch(route, "deleg_bad", "error"), parent_session_id=target)
        assert await runner._deliver_async_delegation_group([ok, bad]) is True
        seen = _turns(runner, authority, target, "deleg_")
        await authority._drain(ref)
        assert len(seen) == 1 and "deleg_ok" in seen[0][0] and "deleg_bad" in seen[0][0]
        assert len(_admissions(authority, ref)) == 1
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_crashed_batch_with_no_child_results_wakes_the_parent(tmp_path, monkeypatch):
    """A runner that raises finalizes the unit as ``status=error`` with ``results=[]`` (``_batch_crash``):
    no child reports a failure, the unit itself does."""
    from tools.async_delegation import _batch_crash
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        evt = {"type": "async_delegation", "is_batch": True, "session_key": route, "delegation_id": "deleg_crash",
               "goals": ["RESULT_deleg_crash goal"], "status": "error", "dispatched_at": 1700000000.0,
               "completed_at": 1700000060.0, **_batch_crash("RuntimeError: batch runner crashed", 1.0)}
        delegation._persist_dispatch(evt)
        delegation._persist_completion(evt, _batch_crash("RuntimeError: batch runner crashed", 1.0))
        assert await runner._deliver_async_delegation_group([dict(evt, parent_session_id=target)]) is True
        seen = _turns(runner, authority, target, "deleg_crash")
        await authority._drain(ref)
        assert len(seen) == 1 and "did not complete successfully" in seen[0][0]
        assert _notes(authority, target, "deleg_crash") == []
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_early_task_failure_notice_stays_a_note(tmp_path, monkeypatch):
    """The per-task notice is an interim diagnostic; the batch's final result is the one that wakes."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        notice = dict(_batch(route, "deleg_early", "error"), parent_session_id=target,
                      task_failure_notice=True, status="running")
        notice.pop("completed_at")
        assert await runner._deliver_completion_notification("[early] RESULT_deleg_early", notice) is True
        seen = _turns(runner, authority, target, "deleg_early")
        await authority._drain(ref)
        assert seen == []
    finally:
        authority.db.close()


def test_refused_child_block_names_the_recovery():
    """The parent learns the refusal is deterministic for that model and what to do about it."""
    from tools.process_registry_notifications import format_process_notification
    text = format_process_notification(_batch_event("error")) or ""
    assert "content_policy_blocked" in text
    assert "different model" in text and "delegate_task" in text
    assert "do not re-dispatch it unchanged" in text.lower()
    assert "different model" not in (format_process_notification(_batch_event("completed")) or "")


def _batch_event(status):
    return {"type": "async_delegation", "is_batch": True, "delegation_id": "deleg_x", "goals": ["triage"],
            "results": [{"task_index": 0, "status": status, "summary": "done" if status == "completed" else None,
                         "error": _REFUSAL if status == "error" else None,
                         "failure_reason": "content_policy_blocked" if status == "error" else None}],
            "dispatched_at": 1700000000.0, "completed_at": 1700000060.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("stall,failed", [({"stalled_after_quiet_seconds": 600.0}, True), ({}, False)],
                         ids=["stall-monitor", "user-stop"])
async def test_unit_the_stall_monitor_stopped_wakes_the_parent(tmp_path, monkeypatch, stall, failed):
    """A stuck child that honours the stall monitor's stop reports ``interrupted``, as a user stop does:
    the unit's stall context is what makes it a failure to recover from (bounded, and told why). Both
    resume the parent."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        evt = dict(_batch(route, "deleg_stall", "interrupted"), parent_session_id=target, **stall)
        assert await runner._deliver_async_delegation_group([evt]) is True
        seen = _turns(runner, authority, target, "deleg_stall")
        await authority._drain(ref)
        assert len(seen) == 1 and _notes(authority, target, "deleg_stall") == []
        assert ("stall monitor stopped this unit after 600s" in seen[0][0]) is failed
        assert authority.sessions[ref.session_id].unattended_wakes == int(failed)
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_failed_manual_cron_run_stays_a_note(tmp_path, monkeypatch):
    """``cronjob(action='run')`` reports through the same event with ``role=cron_run``. Its outcome is
    finished text, as a scheduled run's is: a turn over it would only spend a model call."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        evt = {"type": "async_delegation", "session_key": route, "delegation_id": "deleg_cron", "role": "cron_run",
               "goal": "Manual run of cron job 'RESULT_deleg_cron'", "status": "error", "summary": None,
               "error": "RESULT_deleg_cron failed", "dispatched_at": 1700000000.0, "completed_at": 1700000060.0}
        delegation._persist_dispatch(evt)
        delegation._persist_completion(evt, {"status": "error", "error": evt["error"]})
        assert await runner._deliver_async_delegation_group([dict(evt, parent_session_id=target)]) is True
        seen = _turns(runner, authority, target, "deleg_cron")
        await authority._drain(ref)
        assert seen == []
        assert len(_notes(authority, target, "deleg_cron")) == 1
    finally:
        authority.db.close()


def test_cron_run_coalesced_with_a_failed_delegation_still_wakes():
    """One admission carries the batch: a cron run as its primary must not hide a failed sibling."""
    from tools.process_registry_notifications import delegation_wakes_parent
    cron = {"type": "async_delegation", "role": "cron_run", "status": "error", "error": "job failed"}
    assert not delegation_wakes_parent(cron)
    assert delegation_wakes_parent(dict(cron, _sibling_failed=True))
    assert not delegation_wakes_parent(dict(cron, status="completed", summary="ok"))


async def _fail(runner, authority, ref, route, target, name):
    evt = dict(_batch(route, name, "error"), parent_session_id=target)
    assert await runner._deliver_async_delegation_group([evt]) is True
    await authority._drain(ref)


@pytest.mark.asyncio
async def test_unattended_wakes_stop_at_the_limit_until_the_user_returns(tmp_path, monkeypatch):
    """A woken parent that re-dispatches into the same failure (billing, a usage limit) would wake
    itself without end. After ``UNATTENDED_WAKE_LIMIT`` wakes in a row nobody watched, the next failure
    is a note; the user's own message starts the count again."""
    from hermes_agent.gateway.session_completion_notes import UNATTENDED_WAKE_LIMIT
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        seen = _turns(runner, authority, target, "deleg_loop")
        for i in range(UNATTENDED_WAKE_LIMIT + 2):
            await _fail(runner, authority, ref, route, target, f"deleg_loop{i}")
        woken = [text for text, _ in seen]
        assert len(woken) == UNATTENDED_WAKE_LIMIT, "the loop is bounded"
        assert all(f"deleg_loop{i}" in woken[i] for i in range(UNATTENDED_WAKE_LIMIT))
        assert len(_notes(authority, target, "deleg_loop")) == 2, "later failures are kept as notes"
        _human(authority, ref, "back", "what happened?")
        await authority._drain(ref)
        await _fail(runner, authority, ref, route, target, "deleg_loop_after")
        assert len(seen) == UNATTENDED_WAKE_LIMIT + 2 and "deleg_loop_after" in seen[-1][0]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_failure_in_a_chat_its_viewer_closed_stays_a_note(tmp_path, monkeypatch):
    """Exit or /new ends the session's background work and refuses new dispatches: a wake there would
    run its re-dispatched child inside the wake turn of a chat nobody will reopen. A viewer attaching
    lifts the refusal, and the next failure wakes again."""
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route, target = authority.sessions[ref.session_id].route, _target(runner, authority, ref)
        token = (authority.profile_id, ref.session_id)
        delegation.refuse_for_session(token, session_key=route, parent_session_ids=(ref.session_id, target))
        seen = _turns(runner, authority, target, "deleg_closed")
        await _fail(runner, authority, ref, route, target, "deleg_closed0")
        assert seen == [] and len(_notes(authority, target, "deleg_closed0")) == 1
        delegation.allow_for_session(token)
        await _fail(runner, authority, ref, route, target, "deleg_closed1")
        assert len(seen) == 1 and "deleg_closed1" in seen[0][0]
    finally:
        authority.db.close()


@pytest.mark.parametrize("reason,told", [("billing", True), ("auth", True), ("model_not_found", True),
                                         ("rate_limit", False), ("interpreter_shutdown", False), (None, False)])
def test_deterministic_failure_tells_the_parent_not_to_retry_unchanged(reason, told):
    """A failure that repeats for the same task on the same model gets the don't-retry line a refusal
    gets; a transient one (a rate limit, a gateway restart) does not."""
    from tools.process_registry_notifications import format_process_notification
    evt = _batch_event("error")
    evt["results"][0].update(failure_reason=reason, error="Insufficient credits. Add credits, then retry.")
    text = (format_process_notification(evt) or "").lower()
    assert ("do not re-dispatch it unchanged" in text) is told
    assert (f"this failure ({reason})" in text) is told
