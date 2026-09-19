from __future__ import annotations

import json

from claude_selfimprove import notify, paths


def _read_queue_lines():
    path = paths.self_improvement_queue_path()
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_notify_applied_writes_compatible_event_schema(sandbox):
    ok = notify.notify_applied(
        artifact="/Users/george/.claude/rules/never-force-push.md",
        label="never-force-push",
        improvement="Never force push to the main branch.",
        session_id="sess-1",
    )
    assert ok
    lines = _read_queue_lines()
    assert len(lines) == 1
    event = lines[0]
    assert set(event.keys()) == {
        "v", "ts", "trigger", "kind", "action", "artifact", "label",
        "improvement", "status", "reason", "session_id", "id",
    }
    assert event["trigger"] == "claude-selfimprove"
    assert event["kind"] == "claude_pipeline"
    assert event["action"] == "apply"
    assert event["status"] == "success"
    assert event["reason"] == ""


def test_notify_conflict_blocked_sets_failure_status_with_reason(sandbox):
    notify.notify_conflict_blocked(
        artifact="/x/rules/foo.md", label="foo", reason="contradicts existing rule bar",
    )
    event = _read_queue_lines()[0]
    assert event["action"] == "conflict_blocked"
    assert event["status"] == "failure"
    assert "contradicts" in event["reason"]


def test_notify_rollback_and_failure_events(sandbox):
    notify.notify_rollback(artifact="/x/rules/foo.md", label="foo", reason="post-write verification failed")
    notify.notify_failure(artifact="/x/rules/bar.md", label="bar", reason="disk full")
    lines = _read_queue_lines()
    assert [e["action"] for e in lines] == ["rollback", "failure"]


def test_events_never_contain_raw_transcript_text(sandbox):
    # improvement/label are the only free-text fields, and callers only ever
    # pass already-summarized lesson text - verify the redaction/truncation
    # path is actually applied end to end.
    notify.notify_applied(
        artifact="/x", label="l" * 500,
        improvement="contact george@flexslot.gg " + ("y" * 500),
    )
    event = _read_queue_lines()[0]
    assert "george@flexslot.gg" not in event["improvement"]
    assert len(event["label"]) <= 81
    assert len(event["improvement"]) <= 161


def test_id_is_stable_for_identical_events():
    from claude_selfimprove.notify import _make_event

    e1 = _make_event(action="apply", artifact="/a", label="l", improvement="i", status="success")
    e2 = _make_event(action="apply", artifact="/a", label="l", improvement="i", status="success")
    assert e1["id"] == e2["id"]


def test_id_differs_for_different_status():
    from claude_selfimprove.notify import _make_event

    e1 = _make_event(action="apply", artifact="/a", label="l", improvement="i", status="success")
    e2 = _make_event(action="apply", artifact="/a", label="l", improvement="i", status="failure", reason="x")
    assert e1["id"] != e2["id"]


def test_enqueue_appends_multiple_events_as_separate_lines(sandbox):
    notify.notify_applied(artifact="/a", label="a", improvement="first")
    notify.notify_applied(artifact="/b", label="b", improvement="second")
    lines = _read_queue_lines()
    assert len(lines) == 2
    assert lines[0]["artifact"] == "/a"
    assert lines[1]["artifact"] == "/b"


def test_enqueue_never_raises_when_queue_dir_is_unwritable(sandbox, monkeypatch):
    import os as _os

    def boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(_os, "open", boom)
    ok = notify.notify_applied(artifact="/a", label="a", improvement="x")
    assert ok is False
