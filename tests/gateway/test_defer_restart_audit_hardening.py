"""G2-R4 hardening for the audit appender (gate 2 closure): short writes must
never count as success, a crashed writer's incomplete tail must be isolated
before the next append, and the audit file must be opened without following
symlinks with owner/type/permission verification (create-mode 0600 only governs
new files)."""

from __future__ import annotations

import json
import logging
import os
import stat

import pytest

from gateway import defer_restart as dr


FAKE_GENERATION = {"pid": 4242, "start_time": 1_760_000_000}
FAKE_IDENTITY = {
    "kind": "launchd",
    "service_id": "hermes-gateway",
    "hermes_home": "/tmp/hermes-test",
}


@pytest.fixture()
def sandbox(monkeypatch, tmp_path):
    requests_dir = tmp_path / "restart_requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dr, "requests_dir", lambda: requests_dir)
    monkeypatch.setattr(dr, "process_generation", lambda: dict(FAKE_GENERATION))
    monkeypatch.setattr(dr, "compute_service_identity", lambda: dict(FAKE_IDENTITY))
    return requests_dir


def _audit_lines(requests_dir):
    path = requests_dir / "audit.jsonl"
    return path.read_bytes().splitlines(keepends=True)


def test_partial_tail_isolated_and_next_event_readable(sandbox, caplog):
    """A crashed half-written record must not corrupt earlier events nor the
    next appended event: the tail is isolated with a bare newline and the
    following record parses on its own line (single-writer condition)."""
    audit = sandbox / "audit.jsonl"
    # one complete event, then a crashed writer: half a JSON record, no newline
    audit.write_bytes(b'{"request_id": "old-ok", "outcome": "accepted"}\n{"request_id": "half')
    with caplog.at_level(logging.WARNING):
        dr.append_audit("req-new", "duplicate", actor_ref="parent=1")
    lines = _audit_lines(sandbox)
    assert len(lines) == 3
    assert json.loads(lines[0])["request_id"] == "old-ok"          # earlier event intact
    assert not lines[1].rstrip(b"\n").endswith(b"}")                # partial tail, now newline-terminated
    assert lines[1].endswith(b"\n")                                 # isolated — not glued to next event
    assert json.loads(lines[2])["request_id"] == "req-new"          # next event independently readable
    assert "incomplete tail record" in caplog.text


def test_short_write_is_completed_not_trusted(sandbox, monkeypatch):
    """A short os.write return must not be treated as success: the appender
    completes the record through its write loop, and the event parses whole."""
    real_write = os.write
    state = {"split_done": False}

    def _splitting_write(fd, data):
        # simulate a short write on the first large (record) write only
        if not state["split_done"] and len(data) > 16:
            state["split_done"] = True
            return real_write(fd, data[: len(data) // 2])
        return real_write(fd, data)

    monkeypatch.setattr(dr.os, "write", _splitting_write)
    dr.append_audit("req-1", "accepted", actor_ref="a=1")
    lines = _audit_lines(sandbox)
    assert len(lines) == 1
    assert json.loads(lines[0])["request_id"] == "req-1"  # complete record, no truncation
    assert state["split_done"] is True  # the injected short write actually happened


def test_pre_existing_open_file_is_tightened(sandbox):
    """create-mode 0600 governs NEW files only: a pre-existing over-open file
    must be tightened to owner-only before use, and the event must be readable."""
    audit = sandbox / "audit.jsonl"
    audit.write_text("")
    audit.chmod(0o666)
    dr.append_audit("req-1", "accepted")
    mode = stat.S_IMODE(os.stat(audit).st_mode)
    assert mode == 0o600
    assert json.loads(_audit_lines(sandbox)[0])["request_id"] == "req-1"


def test_symlink_audit_path_is_rejected(sandbox, caplog):
    """The audit file must not be opened through a symlink (O_NOFOLLOW): the
    append is refused and reported, and the link target stays untouched."""
    target = sandbox / "elsewhere.jsonl"
    target.write_text("")
    audit = sandbox / "audit.jsonl"
    audit.symlink_to(target)
    with caplog.at_level(logging.WARNING):
        dr.append_audit("req-1", "accepted")
    assert target.read_text() == ""          # nothing written through the link
    assert "audit append failed" in caplog.text
