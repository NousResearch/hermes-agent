"""`hermes update --status`: read-only health report over marker, tail and receipt."""
import json
import os
import time

import pytest

import hermes_cli.update_cmd_status as status_mod
from hermes_cli.update_cmd_status import _cmd_update_status
from hermes_cli.update_lock import (
    STALE_MARKER_QUARANTINE_AGE_SECONDS,
    UPDATE_MARKER_MAX_AGE_SECONDS,
    update_marker_path,
)

DEAD_PID = 2**30  # never a live pid on any host


@pytest.fixture
def status_home(tmp_path, monkeypatch):
    """Temp hermes home as both HERMES_HOME and the fake project root."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)

    class _FakeMain:
        PROJECT_ROOT = home

    monkeypatch.setattr(status_mod, "_m", lambda: _FakeMain())
    return home


def _write_marker(payload: bytes) -> None:
    update_marker_path().write_bytes(payload)


def _dead_marker_bytes(age_seconds: float) -> bytes:
    started_at = int(time.time() - age_seconds)
    return f"{DEAD_PID}\n{started_at}\nrun:test-run\n".encode()


def _write_receipt(home, record: dict) -> None:
    receipts = home / "logs" / "update_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "latest.json").write_text(json.dumps(record), encoding="utf-8")


def test_pristine_home_is_idle_and_healthy(status_home, capsys):
    assert _cmd_update_status() == 0
    out = capsys.readouterr().out
    assert "none — no update in flight" in out
    assert "none yet" in out
    assert "idle and healthy" in out


def test_successful_receipt_is_reported_not_buried(status_home, capsys):
    _write_receipt(status_home, {
        "outcome": "success",
        "started_at": "2026-10-20T10:00:00+00:00",
        "finished_at": "2026-10-20T10:09:00+00:00",
        "pre_update": {"sha": "abc123", "short_sha": "abc123"},
        "steps": [{"name": "sibling_profile_snapshots", "ok": True, "detail": "x"}],
    })
    assert _cmd_update_status() == 0
    out = capsys.readouterr().out
    assert "completed" in out


def test_failed_receipt_names_the_reason(status_home, capsys):
    _write_receipt(status_home, {
        "outcome": "failed",
        "started_at": "2026-10-20T10:00:00+00:00",
        "pre_update": {"sha": "abc123"},
        "steps": [],
        "stop_reason": "RuntimeError: gateway never came back",
    })
    assert _cmd_update_status() == 3
    out = capsys.readouterr().out
    assert "gateway never came back" in out
    assert "attention needed" in out


def test_dead_marker_is_stale_and_quarantine_eligible(status_home, capsys):
    _write_marker(_dead_marker_bytes(STALE_MARKER_QUARANTINE_AGE_SECONDS + 60))
    assert _cmd_update_status() == 3
    out = capsys.readouterr().out
    assert "STALE" in out
    assert "quarantines it automatically" in out


def test_young_dead_marker_is_stale_but_not_yet_eligible(status_home, capsys):
    _write_marker(_dead_marker_bytes(60))
    assert _cmd_update_status() == 3
    out = capsys.readouterr().out
    assert "STALE" in out
    assert "quarantines it automatically" not in out


def test_live_marker_reports_owner_not_death(status_home, capsys):
    from hermes_cli.update_lock import process_create_time

    ct = process_create_time(os.getpid())
    assert ct is not None
    started_at = int(time.time() - 30)
    _write_marker(f"{os.getpid()}\n{started_at}\nct:{ct:.3f}\nrun:test-run\n".encode())
    assert _cmd_update_status() == 0
    out = capsys.readouterr().out
    assert "owned by this process" in out


def test_ceiling_constants_keep_their_ordering():
    # The watchdog must never fire inside the liveness window: eligibility is
    # strictly beyond the age at which any live updater is still trusted.
    assert STALE_MARKER_QUARANTINE_AGE_SECONDS > UPDATE_MARKER_MAX_AGE_SECONDS
