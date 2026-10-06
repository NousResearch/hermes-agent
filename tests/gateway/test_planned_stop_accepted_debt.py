"""The gateway's planned-stop consumer and the accepted-stop evidence a Windows update pause relies on.

The gateway is this test process; every file is real and every refusal a real one (a read-only
directory standing in for a Windows ACL or sharing refusal). Recovery is the real
``update_pause_record.recover``.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gateway import status
from hermes_cli import update_pause_record as pause_record

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes stand in for Windows refusals"),
    pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes"),
]


def _home(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    root, home = tmp_path / "root", tmp_path / "root" / "profiles" / "p"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return root, home


def _accepted_unrecorded(tmp_path: Path, monkeypatch, *, home_writable: bool) -> tuple[Path, dict]:
    """A paused gateway (this process) accepts its update's stop request while the record refuses the
    checkpoint; the request then ages past its TTL and the updater dies. The acceptance survives only
    beside the request: the ``.accepted`` receipt, or (home refusing new files) the stamped request."""
    root, home = _home(tmp_path, monkeypatch)
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
    body = json.loads(marker.read_text(encoding="utf-8"))
    assert body.get("accepted") is (None if home_writable else True), "premise: receipt vs stamped request"
    body["written_at"] = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    marker.write_text(json.dumps(body), encoding="utf-8")
    saved = pause_record.read()["token"]
    assert saved["stop_sent"] == [], "premise: no checkpoint landed in the record"
    pause_record.write(saved, owner=pause_record.UNOWNED)
    return marker, saved


def _owed(pause_id: str) -> dict | None:
    pause_record.recover(["status"])
    path = pause_record.record_path()
    bodies = [pause_record.read(src) for src in (path, *pause_record._claims(path)) if src.exists()]
    owed = [b["token"]["profiles"] for b in bodies if b["token"]["pause_id"] == pause_id]
    return owed[0] if owed else None


@pytest.mark.parametrize("second_look", ["second_signal", "watcher_probe"])
def test_a_stamped_accepted_request_outlives_its_ttl_while_the_pause_is_owed(tmp_path, monkeypatch, second_look):
    """A second shutdown signal (another Ctrl+C) or a watcher probe after the request's TTL must not
    delete the stamped request: it is the only durable trace that this draining gateway accepted the
    stop, and recovery would otherwise retire its restart debt while it still drains."""
    marker, saved = _accepted_unrecorded(tmp_path, monkeypatch, home_writable=False)
    if second_look == "second_signal":
        assert status.consume_planned_stop_marker_for_self() is False, "an expired request matches nobody"
    else:
        assert status.planned_stop_marker_targets_self() is False
    assert marker.exists(), f"the {second_look} deleted the stamped accepted request"
    assert _owed(saved["pause_id"]) == {"p": os.getpid()}, "an accepted stop lost its restart debt"
