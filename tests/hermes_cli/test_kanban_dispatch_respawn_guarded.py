"""Tests for respawn_guarded observability in dispatch output and diagnostics.

Covers:
- dispatch text output includes "Deferred (respawn guard): ..." line
- dispatch --json includes "respawn_guarded" key with task_id + reason
- kanban_diagnostics._rule_stranded_in_ready distinguishes a guarded card
  from a genuinely stranded one, and surfaces the guard reason + expiry
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from io import StringIO
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Helpers shared across test sections
# ---------------------------------------------------------------------------

def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
        "claim_lock": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _make_dispatch_result(**overrides):
    """Return a DispatchResult with all lists empty except what's in overrides."""
    r = kanban_db.DispatchResult()
    for k, v in overrides.items():
        setattr(r, k, v)
    return r


def _run_cmd_dispatch(result, json_mode=False):
    """Invoke _cmd_dispatch with a fake dispatch_once that returns result.

    Returns captured stdout as a string.
    """
    from hermes_cli import kanban as kb_cli

    fake_config = {"kanban": {}}
    args = argparse.Namespace(dry_run=False, max=None, failure_limit=2, json=json_mode)

    buf = StringIO()
    with (
        patch("hermes_cli.config.load_config", return_value=fake_config),
        patch.object(kanban_db, "dispatch_once", return_value=result),
        patch("sys.stdout", buf),
    ):
        kb_cli._cmd_dispatch(args)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# dispatch text output — respawn_guarded
# ---------------------------------------------------------------------------


def test_dispatch_text_shows_respawn_guarded_line():
    """When dispatch defers tasks via respawn guard, the text output must print
    a 'Deferred (respawn guard): ...' line so the operator isn't left wondering
    why Spawned: 0."""
    result = _make_dispatch_result(
        respawn_guarded=[("t_aaa", "active_pr"), ("t_bbb", "recent_success")],
    )
    out = _run_cmd_dispatch(result)
    assert "Deferred (respawn guard):" in out, (
        f"Expected 'Deferred (respawn guard):' in output; got:\n{out}"
    )
    assert "t_aaa (active_pr)" in out
    assert "t_bbb (recent_success)" in out


def test_dispatch_text_omits_respawn_guarded_line_when_empty():
    """When no tasks are guarded the line must NOT appear (no noise)."""
    result = _make_dispatch_result(respawn_guarded=[])
    out = _run_cmd_dispatch(result)
    assert "respawn guard" not in out, (
        f"Empty respawn_guarded should produce no line; got:\n{out}"
    )


def test_dispatch_text_shows_all_guard_reasons():
    """All four guard reasons must be legible in the text line."""
    reasons = ["active_pr", "recent_success", "blocker_auth", "rate_limit_cooldown"]
    result = _make_dispatch_result(
        respawn_guarded=[(f"t_{r}", r) for r in reasons],
    )
    out = _run_cmd_dispatch(result)
    for reason in reasons:
        assert reason in out, f"Reason {reason!r} missing from dispatch text:\n{out}"


# ---------------------------------------------------------------------------
# dispatch --json output — respawn_guarded key
# ---------------------------------------------------------------------------


def test_dispatch_json_includes_respawn_guarded_key():
    """--json output must include a 'respawn_guarded' key as an array of
    {task_id, reason} objects."""
    result = _make_dispatch_result(
        respawn_guarded=[("t_aaa", "active_pr")],
    )
    out = _run_cmd_dispatch(result, json_mode=True)
    data = json.loads(out)
    assert "respawn_guarded" in data, (
        f"'respawn_guarded' key missing from --json output; keys={list(data.keys())}"
    )
    guarded = data["respawn_guarded"]
    assert isinstance(guarded, list)
    assert len(guarded) == 1
    assert guarded[0] == {"task_id": "t_aaa", "reason": "active_pr"}


def test_dispatch_json_respawn_guarded_empty_list_when_none():
    """When no tasks are guarded, 'respawn_guarded' must be an empty list
    (not absent, not null)."""
    result = _make_dispatch_result(respawn_guarded=[])
    out = _run_cmd_dispatch(result, json_mode=True)
    data = json.loads(out)
    assert "respawn_guarded" in data
    assert data["respawn_guarded"] == []


def test_dispatch_json_multiple_guarded_tasks():
    """Multiple guarded tasks must all appear in the JSON array."""
    guarded = [
        ("t_aaa", "active_pr"),
        ("t_bbb", "recent_success"),
        ("t_ccc", "rate_limit_cooldown"),
    ]
    result = _make_dispatch_result(respawn_guarded=guarded)
    out = _run_cmd_dispatch(result, json_mode=True)
    data = json.loads(out)
    assert len(data["respawn_guarded"]) == 3
    ids = {g["task_id"] for g in data["respawn_guarded"]}
    assert ids == {"t_aaa", "t_bbb", "t_ccc"}


# ---------------------------------------------------------------------------
# diagnostics — guarded cards vs genuinely stranded cards
# ---------------------------------------------------------------------------


def test_diagnostics_stranded_shows_guard_reason_when_respawn_guarded_event_present():
    """A ready task with a recent respawn_guarded event should fire
    stranded_in_ready with the guard reason in the title (not 'no worker')."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 6 * 3600),
        _event("respawn_guarded", ts=now - 60, reason="active_pr"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1, "Should still fire a stranded_in_ready diagnostic"
    d = stranded[0]
    assert "active_pr" in d.title, (
        f"Guard reason should appear in title; got: {d.title!r}"
    )
    assert "respawn guard" in d.title.lower(), (
        f"Title should mention 'respawn guard'; got: {d.title!r}"
    )
    assert "no worker" not in d.title.lower(), (
        f"Guarded task must NOT say 'no worker' in title; got: {d.title!r}"
    )
    assert d.data.get("guard_reason") == "active_pr"


def test_diagnostics_stranded_shows_expiry_for_rate_limit():
    """For rate_limit_cooldown, the diagnostic detail should mention
    the expiry time (guard is still in-window)."""
    now = 100_000
    guard_ts = now - 120  # guard fired 2 min ago, cooldown = 300s, 3 min left
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 2 * 3600),
        _event("respawn_guarded", ts=guard_ts, reason="rate_limit_cooldown"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    d = stranded[0]
    assert "rate_limit_cooldown" in d.title
    assert "expires" in d.detail.lower() or "3m" in d.detail or "~3m" in d.detail, (
        f"Expected expiry hint in detail; got: {d.detail!r}"
    )


def test_diagnostics_stranded_no_guard_shows_original_message():
    """Without a respawn_guarded event the diagnostic uses the original
    'no worker' message (no regression)."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [_event("created", ts=now - 6 * 3600)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert "no worker" in stranded[0].title.lower(), (
        f"Non-guarded task should use 'no worker' title; got: {stranded[0].title!r}"
    )
    assert "guard_reason" not in stranded[0].data


def test_diagnostics_stranded_guard_event_ts_used_for_last_seen_at():
    """When a guard event is present, last_seen_at should reflect the guard
    event timestamp, not the original ready transition."""
    now = 100_000
    guard_ts = now - 30  # very recent
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 4 * 3600),
        _event("respawn_guarded", ts=guard_ts, reason="recent_success"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].last_seen_at == guard_ts


def test_diagnostics_stranded_latest_guard_event_wins():
    """When multiple respawn_guarded events exist (across ticks), the
    most-recent one's reason is surfaced."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 6 * 3600),
        _event("respawn_guarded", ts=now - 600, reason="blocker_auth"),
        _event("respawn_guarded", ts=now - 120, reason="active_pr"),  # wins
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].data["guard_reason"] == "active_pr"


def test_diagnostics_stranded_guard_data_contains_expected_keys():
    """The data dict for a guarded stranded task should include
    guard_reason and guard_event_ts for downstream consumers."""
    now = 100_000
    guard_ts = now - 60
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 3 * 3600),
        _event("respawn_guarded", ts=guard_ts, reason="active_pr"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    d = stranded[0]
    assert "guard_reason" in d.data, f"guard_reason missing from data: {d.data}"
    assert "guard_event_ts" in d.data, f"guard_event_ts missing from data: {d.data}"
    assert d.data["guard_event_ts"] == guard_ts
    assert d.data["guard_reason"] == "active_pr"
    # Core stranded fields still present
    assert "age_seconds" in d.data
    assert "assignee" in d.data
