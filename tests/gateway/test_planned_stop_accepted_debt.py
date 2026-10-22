"""The gateway's planned-stop consumer and the accepted-stop evidence a Windows update pause relies on.

The gateway is this test process; every file is real and every refusal a real one (a read-only
directory standing in for a Windows ACL or sharing refusal). Recovery is the real
``update_pause_record.recover``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone, UTC
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
    body["written_at"] = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
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


@pytest.mark.parametrize("home_writable", [True, False], ids=["receipt", "stamped_request"])
def test_accepted_stop_evidence_is_removed_once_its_pause_is_settled_and_never_before(
        tmp_path, monkeypatch, home_writable):
    """``.accepted`` receipts and stamped requests are cleaned up by the next consume once no pause of
    this checkout is on disk (recovery or the updater retired it), and kept while one still is."""
    marker, saved = _accepted_unrecorded(tmp_path, monkeypatch, home_writable=home_writable)
    evidence = pause_record._accepted_path(marker) if home_writable else marker
    status.consume_planned_stop_marker_for_self()
    assert evidence.exists(), "accepted-stop evidence removed while its pause is still owed"
    pause_record.discharge(saved)
    assert not pause_record.record_path().exists(), "premise: the pause is settled"
    status.consume_planned_stop_marker_for_self()
    assert not pause_record._accepted_path(marker).exists(), "a settled receipt was left behind"
    assert not marker.exists(), "a settled stamped request was left behind"


def test_with_no_pause_on_disk_the_consumer_never_waits_on_the_pause_lock(tmp_path, monkeypatch):
    """No pause record (every non-Windows host, or a settled pause): a plain ``hermes gateway stop`` is
    consumed without the pause lock, so another holder costs it no wait and leaves no request or
    ``.accepted`` receipt behind."""
    _home(tmp_path, monkeypatch)
    held, release = threading.Event(), threading.Event()

    def holder():
        with pause_record._mutex():
            held.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(5), "premise: another thread holds the pause lock"
        marker = status._get_planned_stop_marker_path()
        assert status.write_planned_stop_marker(os.getpid())
        started = time.monotonic()
        assert status.consume_planned_stop_marker_for_self() is True
        elapsed = time.monotonic() - started
    finally:
        release.set()
        thread.join()
    assert elapsed < 1.0, f"the consumer waited {elapsed:.2f}s on a pause lock no pause needed"
    assert not marker.exists(), "the consumed request was left on disk"
    assert not pause_record._accepted_path(marker).exists(), "an .accepted receipt was written with no pause"
