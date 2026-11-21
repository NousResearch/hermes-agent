"""Kanban card notifications carry the whole summary / block reason, not a 160-200 char stub."""

from types import SimpleNamespace

from gateway.kanban_watchers_notifier import _DISPLAY_LIMIT, _EVENT_FORMATTERS


def _n():
    return SimpleNamespace(head="[default] @worker Kanban t_x", title="Some card", task_id="t_x", task=None)


def _ev(**payload):
    return SimpleNamespace(payload=payload)


LONG = ("Posted and read-back verified the report-only plan for node-3. Live before-state: 15 failed units, "
        "all session-*.scope; no non-session failed unit was present. The plan explicitly declares "
        "`irreversible: yes` and names its rollback.\nSecond line with the rollback note.")


def test_completed_message_shows_full_summary_but_wake_stays_short():
    msg, wake, _ = _EVENT_FORMATTERS["completed"](_ev(summary=LONG), _n())
    assert "names its rollback." in msg
    assert "Second line with the rollback note." in msg
    assert wake == LONG.splitlines()[0][:200]


def test_blocked_reason_keeps_continue_and_cancel_lines():
    reason = "⏸ reviewer t_x — clearance\n" + "x" * 300 + '\nTo continue: "Do it."\nTo cancel:   "Cancel it."'
    msg, _, _ = _EVENT_FORMATTERS["blocked"](_ev(reason=reason), _n())
    assert 'To continue: "Do it."' in msg and 'To cancel:   "Cancel it."' in msg


def test_triage_reason_not_cut_at_160():
    reason = "y" * 400 + " END"
    msg, _, _ = _EVENT_FORMATTERS["block_loop_detected"](_ev(reason=reason, recurrences=2), _n())
    assert " END" in msg


def test_overlong_text_is_shortened_with_ellipsis():
    msg, _, _ = _EVENT_FORMATTERS["blocked"](_ev(reason="z" * (_DISPLAY_LIMIT * 3)), _n())
    assert "…" in msg
    assert len(msg) < _DISPLAY_LIMIT + 300
