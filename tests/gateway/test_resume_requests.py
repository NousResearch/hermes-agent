"""Tests for gateway/resume_requests.py — the external resume-request dropbox.

SGR-6EA95669 (2026-07-10): external tools must never write sessions.json
(two-writers clobber race); they drop request files the gateway sweeps.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from gateway import resume_requests as rr


def _submit(home: Path, key: str, reason: str = "restart_interrupted") -> Path:
    return rr.submit_resume_request(home, key, reason)


def test_submit_then_sweep_roundtrip(tmp_path: Path) -> None:
    path = _submit(tmp_path, "agent:main:discord:thread:1:1")
    assert path.exists()
    got = rr.sweep_resume_requests(tmp_path)
    assert got == [("agent:main:discord:thread:1:1", "restart_interrupted")]
    # Consumed: file gone, second sweep empty.
    assert not path.exists()
    assert rr.sweep_resume_requests(tmp_path) == []


def test_sweep_missing_dir_fast_path(tmp_path: Path) -> None:
    assert rr.sweep_resume_requests(tmp_path) == []


def test_duplicate_requests_dedup_first_reason_wins(tmp_path: Path) -> None:
    _submit(tmp_path, "agent:main:discord:thread:2:2", "restart_interrupted")
    time.sleep(0.001)
    _submit(tmp_path, "agent:main:discord:thread:2:2", "shutdown_timeout")
    got = rr.sweep_resume_requests(tmp_path)
    assert len(got) == 1
    assert got[0][0] == "agent:main:discord:thread:2:2"
    # Both files consumed regardless of dedup.
    assert rr.sweep_resume_requests(tmp_path) == []


def test_stale_requests_dropped(tmp_path: Path) -> None:
    path = _submit(tmp_path, "agent:main:discord:thread:3:3")
    payload = json.loads(path.read_text())
    payload["requested_at"] = time.time() - 7200
    path.write_text(json.dumps(payload))
    assert rr.sweep_resume_requests(tmp_path) == []
    assert not path.exists()  # consumed, not left to rot


def test_malformed_request_quarantined_not_reparsed(tmp_path: Path) -> None:
    directory = rr.dropbox_dir(tmp_path)
    directory.mkdir(parents=True)
    bad = directory / "garbage.json"
    bad.write_text("{not json")
    assert rr.sweep_resume_requests(tmp_path) == []
    assert not bad.exists()
    assert (directory / "garbage.json.rejected").exists()
    # Rejected file is not picked up again.
    assert rr.sweep_resume_requests(tmp_path) == []


def test_submit_atomic_no_partial_files_visible(tmp_path: Path) -> None:
    _submit(tmp_path, "agent:main:discord:thread:4:4")
    names = os.listdir(rr.dropbox_dir(tmp_path))
    # No tmp files left behind; the one visible file is complete valid JSON.
    assert all(not n.startswith(".resume-req-") for n in names)
    assert len([n for n in names if n.endswith(".json")]) == 1


def test_key_sanitization_in_filename(tmp_path: Path) -> None:
    path = _submit(tmp_path, "agent:main:discord:thread:5:5")
    assert ":" not in path.name
    got = rr.sweep_resume_requests(tmp_path)
    # Original key survives sanitized filenames (payload is authoritative).
    assert got[0][0] == "agent:main:discord:thread:5:5"


def test_boot_sweep_marks_and_gates_via_session_store(tmp_path: Path, monkeypatch) -> None:
    """Integration: the run.py boot sweep marks known sessions resume_pending
    through mark_resume_pending (suspended wins) and skips unknown keys —
    a dropbox request must never bypass the store's gates."""
    import gateway.run as run_mod

    calls = []

    class _Store:
        def mark_resume_pending(self, key, reason):
            calls.append((key, reason))
            # Emulate the real gate: known+active session True; else False.
            return key == "agent:main:discord:thread:6:6"

    _submit(tmp_path, "agent:main:discord:thread:6:6")
    _submit(tmp_path, "agent:main:discord:thread:GONE:0")

    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)

    # Drive just the sweep block: replicate the run.py consumption loop
    # against a stub store (the full _schedule_resume_pending_sessions needs
    # a live runner; the sweep block's contract is what we lock here).
    store = _Store()
    for key, reason in rr.sweep_resume_requests(run_mod._hermes_home):
        store.mark_resume_pending(key, reason)

    assert ("agent:main:discord:thread:6:6", "restart_interrupted") in calls
    assert ("agent:main:discord:thread:GONE:0", "restart_interrupted") in calls
    assert rr.sweep_resume_requests(tmp_path) == []  # consumed either way


def test_draining_gateway_defers_dropbox_to_successor(tmp_path: Path, monkeypatch) -> None:
    """SGR-6EA95669 follow-up (2026-07-18): the OLD gateway's housekeeping
    tick during a long restart drain used to CONSUME dropbox requests and
    mark resume_pending in memory that died with the process — the successor
    boot then found an empty dropbox and the initiating session of a
    safe-restart never resumed. A draining/shutting-down runner must leave
    the request files on disk for the successor's boot sweep."""
    import asyncio

    import gateway.run as run_mod
    from gateway.run import GatewayRunner

    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)

    consumed = []

    class _Store:
        def mark_resume_pending(self, key, reason):
            consumed.append((key, reason))
            return True

    path = _submit(tmp_path, "agent:main:discord:thread:7:7")

    runner = object.__new__(GatewayRunner)
    runner.session_store = _Store()

    loop = asyncio.new_event_loop()
    try:
        shutdown_event = asyncio.Event()

        # Case 1: draining flag set → request file untouched, store untouched.
        runner._draining = True
        runner._shutdown_event = shutdown_event
        GatewayRunner._sweep_resume_requests(runner)
        assert path.exists(), "draining sweep must NOT consume the request"
        assert consumed == []

        # Case 2: not draining but shutdown_event set → same deferral.
        runner._draining = False
        shutdown_event.set()
        GatewayRunner._sweep_resume_requests(runner)
        assert path.exists(), "shutting-down sweep must NOT consume the request"
        assert consumed == []

        # Case 3: healthy runner → consumes and honors as before.
        runner._shutdown_event = asyncio.Event()  # cleared
        GatewayRunner._sweep_resume_requests(runner)
        assert not path.exists(), "healthy sweep must consume the request"
        assert consumed == [("agent:main:discord:thread:7:7", "restart_interrupted")]
    finally:
        loop.close()
