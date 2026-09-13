"""An unclean gateway death must trigger a state.db integrity check.

Regression for the 2026-08-31 incident. ``state.db`` was corrupt from
2026-08-26 evening (a SIGKILL landed on a gateway mid-WAL-checkpoint during a
``--replace`` restart storm), but nothing checked the file. The damage sat in
old, rarely-read session rows for 3.5 days until a Desktop read tripped over
it on 2026-08-30 17:15 and surfaced as "Session not found".

``record_startup`` already detects the unclean exit and logs "SIGKILL / OOM /
VM death" — it just never looked at the database that death may have torn.
The check is gated on the unclean exit precisely because it costs ~2s on a
500MB store; a clean boot must not pay it.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from gateway.lifecycle_ledger import (
    check_state_db_integrity,
    get_lifecycle_sentinel_path,
    record_startup,
)

_DEAD_PID = 2 ** 22 + 12345  # beyond default pid_max; never alive


def _write_sentinel(home: Path, phase: str = "running") -> None:
    path = get_lifecycle_sentinel_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "phase": phase,
            "pid": _DEAD_PID,
            "start_time": 1000.0,
            "started_at": "2026-08-26T23:56:45+00:00",
        }),
        encoding="utf-8",
    )


def _make_state_db(home: Path, *, corrupt: bool) -> Path:
    """Build a real SQLite file, optionally with a genuinely torn b-tree page."""
    path = home / "state.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany(
        "INSERT INTO sessions (v) VALUES (?)", [(f"row-{i}" * 40,) for i in range(4000)]
    )
    conn.commit()
    conn.close()
    if corrupt:
        with open(path, "r+b") as handle:
            handle.seek(4096 * 6)
            handle.write(b"\xEF" * 4096)
    return path


def _exit_diag_records(home: Path) -> list:
    log = home / "logs" / "gateway-exit-diag.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


# ── the checker itself ──────────────────────────────────────────────────────


def test_checker_passes_a_healthy_store(tmp_path: Path) -> None:
    _make_state_db(tmp_path, corrupt=False)
    assert check_state_db_integrity(home=tmp_path) == "ok"


def test_checker_reports_a_torn_btree_page(tmp_path: Path) -> None:
    _make_state_db(tmp_path, corrupt=True)
    verdict = check_state_db_integrity(home=tmp_path)
    assert verdict != "ok"
    assert "btreeInitPage" in verdict or "malformed" in verdict.lower()


def test_checker_tolerates_a_missing_store(tmp_path: Path) -> None:
    assert check_state_db_integrity(home=tmp_path) == "absent"


# ── wiring into the unclean-exit path ───────────────────────────────────────


def test_unclean_exit_records_the_corruption_verdict(tmp_path: Path) -> None:
    _make_state_db(tmp_path, corrupt=True)
    _write_sentinel(tmp_path)

    evidence = record_startup(home=tmp_path)

    assert evidence is not None
    assert evidence["state_db_integrity"] != "ok"
    record = _exit_diag_records(tmp_path)[0]
    assert record["state_db_integrity"] != "ok"


def test_unclean_exit_on_a_healthy_store_records_ok(tmp_path: Path) -> None:
    _make_state_db(tmp_path, corrupt=False)
    _write_sentinel(tmp_path)

    evidence = record_startup(home=tmp_path)

    assert evidence is not None
    assert evidence["state_db_integrity"] == "ok"


def test_clean_exit_does_not_pay_for_the_check(tmp_path: Path, monkeypatch) -> None:
    """A clean boot must not scan the store — that is the whole cost gate."""
    _make_state_db(tmp_path, corrupt=True)
    _write_sentinel(tmp_path, phase="exited")

    called = []
    import gateway.lifecycle_ledger as ledger

    monkeypatch.setattr(
        ledger, "check_state_db_integrity", lambda **kw: called.append(1) or "ok"
    )
    record_startup(home=tmp_path)

    assert not called, "integrity check ran on a clean boot"


# ── bounded startup check ──────────────────────────────────────────────────
# A 3 GB store made quick_check hold startup for 16-29 minutes after unclean
# deaths. Startup now waits a bounded time and finishes the check off-path.


def test_a_check_over_budget_is_deferred_not_judged(tmp_path: Path, monkeypatch) -> None:
    import gateway.lifecycle_ledger as ledger

    _make_state_db(tmp_path, corrupt=False)
    monkeypatch.setattr(ledger, "_INTEGRITY_PROGRESS_OPS", 1)
    verdict = check_state_db_integrity(home=tmp_path, budget_s=-1.0)
    assert verdict.startswith("check-deferred: over")
    assert check_state_db_integrity(home=tmp_path, budget_s=60.0) == "ok"


def test_startup_does_not_wait_for_a_slow_check(tmp_path: Path, monkeypatch) -> None:
    import gateway.lifecycle_ledger as ledger

    _make_state_db(tmp_path, corrupt=True)
    _write_sentinel(tmp_path)
    started = []
    real = ledger._finish_integrity_check_later
    monkeypatch.setattr(ledger, "UNCLEAN_INTEGRITY_STARTUP_BUDGET_S", -1.0)
    monkeypatch.setattr(ledger, "_INTEGRITY_PROGRESS_OPS", 1)
    monkeypatch.setattr(ledger, "_finish_integrity_check_later",
                        lambda home: started.append(real(home)) or started[-1])
    evidence = record_startup(home=tmp_path)
    assert evidence is not None
    assert evidence["state_db_integrity"].startswith("check-deferred")
    started[0].join(timeout=30)
    tags = [record.get("tag") for record in _exit_diag_records(tmp_path)]
    assert "gateway.state_db_integrity_deferred" in tags
