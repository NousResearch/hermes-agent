"""Accepted-stop evidence the pause record cannot read or could not write: the debt stays owed.

The gateway is this test process (a live incarnation that accepted the stop and is draining); the
updater is gone (the record is unowned); every file is real and every refusal is a real one (mode
000, a read-only directory), standing in for a Windows sharing violation or ACL refusal.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path

import pytest

from hermes_cli import update_pause_record as pause_record

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes stand in for Windows refusals"),
    pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes"),
]


def _accepted_then_orphaned(tmp_path: Path, monkeypatch, *, home_writable: bool) -> tuple[Path, str]:
    """Record a pause of this process, request its stop, let it accept the request while the record
    refuses the consumer's checkpoint (read-only record directory), then age the request past its
    TTL and orphan the record. Returns ``(marker, pause_id)``."""
    from gateway import status
    root, home = tmp_path / "root", tmp_path / "root" / "profiles" / "p"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    pid = os.getpid()
    marker = status._get_planned_stop_marker_path()
    token = pause_record.record_pause({"resume_needed": True, "profiles": {"p": pid},
                                       "identities": {str(pid): pause_record.identity(pid)["ct"]}}, None, [])
    pause_record.mark_stop_requested(token, [pid], markers={pid: marker})
    assert status.write_planned_stop_marker(pid)
    root.chmod(0o555)
    if not home_writable:
        home.chmod(0o555)
    try:
        assert status.consume_planned_stop_marker_for_self() is True, "premise: the stop was accepted"
    finally:
        home.chmod(0o755)
        root.chmod(0o755)
    saved = pause_record.read()["token"]
    assert saved["stop_sent"] == [], "premise: no checkpoint landed in the record"
    body = json.loads(marker.read_text(encoding="utf-8"))
    body["written_at"] = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()  # past the TTL
    marker.write_text(json.dumps(body), encoding="utf-8")
    pause_record.write(saved, owner=pause_record.UNOWNED)  # the updater died
    return marker, saved["pause_id"]


def _owed_after_recovery(pause_id: str) -> dict | None:
    """Run a launch's recovery; the profiles still owed by the obligation, ``None`` once retired."""
    pause_record.recover(["status"])
    path = pause_record.record_path()
    retired = path.with_suffix(".retired")
    ids = json.loads(retired.read_text(encoding="utf-8"))["ids"] if retired.exists() else []
    carriers = [pause_record.read(src) for src in (path, *pause_record._claims(path)) if src.exists()]
    owed = [body["token"]["profiles"] for body in carriers if body["token"]["pause_id"] == pause_id]
    assert bool(owed) != (pause_id in ids), "an obligation is either owed or retired, never both or neither"
    return owed[0] if owed else None


@pytest.mark.parametrize("receipt", ["refused", "malformed", "missing"])
def test_an_unreadable_accepted_stop_receipt_keeps_the_draining_gateway_owed(tmp_path, monkeypatch, receipt):
    """R1: a receipt that exists but cannot be read or parsed is unknown evidence: the live draining
    gateway stays owed (never restarted over, never retired). A receipt that is genuinely absent
    beside an expired, unconsumed request is the control: that stop was never accepted."""
    marker, pause_id = _accepted_then_orphaned(tmp_path, monkeypatch, home_writable=True)
    accepted = pause_record._accepted_path(marker)
    assert accepted.exists(), "premise: the consumer left its receipt"
    if receipt == "refused":
        accepted.chmod(0)
    elif receipt == "malformed":
        accepted.write_text("{trunc", encoding="utf-8")
    else:
        accepted.unlink()
    try:
        owed = _owed_after_recovery(pause_id)
    finally:
        if accepted.exists():
            accepted.chmod(0o644)
    if receipt == "missing":
        assert owed is None, "a request nobody accepted must still expire"
        return
    assert owed == {"p": os.getpid()}, f"a {receipt} receipt retired a draining gateway's restart debt"
    # Unknown is judged again, never resolved into "sent": once readable it is a receipt like any other.
    assert str(os.getpid()) not in pause_record.read(pause_record._claims(pause_record.record_path())[0])["token"]["stop_sent"]
    assert _owed_after_recovery(pause_id) == {"p": os.getpid()}


def test_an_accepted_stop_whose_receipt_cannot_be_written_stays_owed(tmp_path, monkeypatch):
    """Thread 6: the gateway home refuses new files too, so neither the record checkpoint nor the
    receipt beside the request can be created; the consumer still accepts the stop. The kept
    request itself must carry the acceptance, so the debt survives the request's TTL."""
    marker, pause_id = _accepted_then_orphaned(tmp_path, monkeypatch, home_writable=False)
    assert not pause_record._accepted_path(marker).exists(), "premise: the receipt could not be created"
    assert _owed_after_recovery(pause_id) == {"p": os.getpid()}, "an accepted stop lost its restart debt"
