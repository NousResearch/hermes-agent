"""Tests for the ``stale_resource_wait`` diagnostic.

A ``resource`` block says a peer holds a lease or a rate-limited seat. It is
exempt from the unblock-loop breaker, so nothing escalates it on its own; these
tests pin the window, the escalation, and -- the case that motivated the rule --
that a task narrating its own wait does not silence its own alarm.
"""

from __future__ import annotations

import json
import time

from hermes_cli import kanban_diagnostics as kd


def _task(**overrides):
    base = {
        "id": "t_res001",
        "title": "waiting on a figma write lease",
        "assignee": "vx-design",
        "status": "blocked",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _block_event(ts, *, block_kind="resource", reason="figma write lease held by t_other", json_payload=False):
    payload = {"reason": reason}
    if block_kind is not None:
        payload["kind"] = block_kind
    return {
        "kind": "blocked",
        "created_at": int(ts),
        "payload": json.dumps(payload) if json_payload else payload,
    }


def _event(kind, ts):
    return {"kind": kind, "created_at": int(ts), "payload": None}


def _only(diags, kind):
    return [d for d in diags if d.kind == kind]


def test_fires_after_the_default_one_hour_window():
    now = int(time.time())
    diags = kd.compute_task_diagnostics(_task(), [_block_event(now - 3600 * 2)], [], now=now)
    hits = _only(diags, "stale_resource_wait")
    assert len(hits) == 1
    assert hits[0].severity == "warning"
    assert hits[0].data["age_hours"] >= 2
    assert "figma write lease" in hits[0].data["block_reason"]


def test_quiet_inside_the_window():
    now = int(time.time())
    diags = kd.compute_task_diagnostics(_task(), [_block_event(now - 600)], [], now=now)
    assert _only(diags, "stale_resource_wait") == []


def test_escalates_past_four_windows():
    now = int(time.time())
    diags = kd.compute_task_diagnostics(_task(), [_block_event(now - 3600 * 5)], [], now=now)
    assert _only(diags, "stale_resource_wait")[0].severity == "error"


def test_the_tasks_own_comment_does_not_silence_it():
    """The regression this rule exists for.

    ``_rule_stuck_in_blocked`` treats any comment as evidence someone is on the
    case, but a card queueing for a lease comments on itself while it waits. On
    a real board four such waits sat 10-13h, each with the lease already free,
    and no diagnostic was ever raised.
    """
    now = int(time.time())
    events = [
        _block_event(now - 3600 * 6),
        _event("commented", now - 3600 * 5),
        _event("commented", now - 60),
    ]
    assert _only(kd.compute_task_diagnostics(_task(), events, [], now=now), "stale_resource_wait")


def test_an_unblock_clears_it():
    now = int(time.time())
    events = [_block_event(now - 3600 * 6), _event("unblocked", now - 3600)]
    assert _only(kd.compute_task_diagnostics(_task(), events, [], now=now), "stale_resource_wait") == []


def test_ignores_other_block_kinds():
    now = int(time.time())
    for kind in ("needs_input", "capability", "transient", "dependency", None):
        events = [_block_event(now - 3600 * 30, block_kind=kind)]
        diags = kd.compute_task_diagnostics(_task(), events, [], now=now)
        assert _only(diags, "stale_resource_wait") == [], kind


def test_ignores_tasks_that_are_not_blocked():
    now = int(time.time())
    events = [_block_event(now - 3600 * 6), _event("unblocked", now - 3600 * 5)]
    diags = kd.compute_task_diagnostics(_task(status="ready"), events, [], now=now)
    assert _only(diags, "stale_resource_wait") == []


def test_reads_a_json_encoded_payload():
    """Real rows carry ``payload`` as a JSON string, not a dict."""
    now = int(time.time())
    events = [_block_event(now - 3600 * 3, json_payload=True)]
    assert _only(kd.compute_task_diagnostics(_task(), events, [], now=now), "stale_resource_wait")


def test_stuck_in_blocked_does_not_also_report_a_resource_wait():
    """Both rules would otherwise fire on an old resource block, the second one
    a day later and under the wrong remedy ("waiting for human input")."""
    now = int(time.time())
    events = [_block_event(now - 3600 * 48)]
    diags = kd.compute_task_diagnostics(_task(), events, [], now=now)
    assert _only(diags, "stuck_in_blocked") == []
    assert _only(diags, "stale_resource_wait")


def test_stuck_in_blocked_still_fires_for_everything_else():
    now = int(time.time())
    events = [_block_event(now - 3600 * 48, block_kind="needs_input")]
    diags = kd.compute_task_diagnostics(_task(), events, [], now=now)
    assert _only(diags, "stuck_in_blocked")


def test_window_is_configurable_and_zero_disables():
    now = int(time.time())
    events = [_block_event(now - 3600 * 3)]
    assert _only(kd.compute_task_diagnostics(
        _task(), events, [], now=now, config={"resource_wait_stale_hours": 8}),
        "stale_resource_wait") == []
    assert _only(kd.compute_task_diagnostics(
        _task(), events, [], now=now, config={"resource_wait_stale_hours": 0}),
        "stale_resource_wait") == []

