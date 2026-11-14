"""Regression: doctor's state.db repair hints must order "stop the writer" before `--fix` (#110054).

Item 3 of #110054 fixed this for the large-WAL warning: a bare ``run 'hermes doctor --fix'`` printed
while a live Desktop/gateway holds state.db is the nudge that turns the user into the second writer the
deleted-WAL guard then fires on. The FTS / malformed-schema repair hints and the >1 GB write-health probe
skip print the same bare command — and ``repair_state_db_schema`` refuses a live holder outright
(its WAL-holder preflight), so the suggested command cannot succeed until the writer is stopped.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli.doctor_report import Finding
from hermes_cli.doctor_state import _repair_state_db, _write_health_reason


def _db(tmp_path) -> Path:
    db = tmp_path / "state.db"
    db.write_bytes(b"")
    return db


def _hold(monkeypatch, held: bool) -> None:
    import hermes_state_repair

    monkeypatch.setattr(hermes_state_repair, "_live_writer_holds_db", lambda *a, **k: held)


def test_fts_repair_hint_orders_stop_before_fix_while_a_writer_may_hold_the_store(tmp_path, monkeypatch):
    """Held (or unprovable): the hint must order "stop the profile's gateway" before the command, and
    must not read as a plain "run --fix" that the repair path would refuse."""
    _hold(monkeypatch, True)
    finding = Finding()

    _repair_state_db(finding, False, _db(tmp_path), "fts")

    assert len(finding.issues) == 1 and not finding.fixed
    hint = finding.issues[0]
    assert hint.index("stop the profile's gateway") < hint.index("hermes doctor --fix")
    assert "cannot prove state.db is quiet" in hint


def test_schema_repair_hint_orders_stop_before_fix_while_a_writer_may_hold_the_store(tmp_path, monkeypatch):
    _hold(monkeypatch, True)
    finding = Finding()

    _repair_state_db(finding, False, _db(tmp_path), "schema")

    assert len(finding.issues) == 1 and not finding.fixed
    hint = finding.issues[0]
    assert hint.index("stop the profile's gateway") < hint.index("hermes doctor --fix")
    assert "cannot prove state.db is quiet" in hint


def test_repair_hints_are_unchanged_when_the_store_is_provably_quiet(tmp_path, monkeypatch):
    """No holder: today's hint, which does not make the user stop anything first."""
    _hold(monkeypatch, False)
    finding = Finding()

    _repair_state_db(finding, False, _db(tmp_path), "fts")

    assert finding.issues == [
        "state.db FTS write corruption — run 'hermes doctor --fix' "
        "(or 'hermes sessions repair') to rebuild the FTS index"
    ]


def test_repair_hint_fails_closed_when_the_holder_scan_raises(tmp_path, monkeypatch):
    """An unprovable holder is treated as held: the stop-first hint, never the bare command."""
    import hermes_state_repair

    def boom(*_a, **_k):
        raise RuntimeError("holder scan unavailable")

    monkeypatch.setattr(hermes_state_repair, "_live_writer_holds_db", boom)
    finding = Finding()

    _repair_state_db(finding, False, _db(tmp_path), "fts")

    assert "stop the profile's gateway" in finding.issues[0]


def test_oversized_write_probe_skip_orders_stop_before_fix(tmp_path, monkeypatch, capsys):
    """The >1 GB held-store probe skip names the writer to stop instead of a bare `run --fix`."""
    _hold(monkeypatch, True)
    db = _db(tmp_path)
    with open(db, "ab") as handle:  # sparse: only the size check reads it
        handle.truncate((1 << 30) + 1)

    assert _write_health_reason(db, should_fix=False) is None
    out = capsys.readouterr().out
    assert "stop the profile's gateway" in out
    assert out.index("stop the profile's gateway") < out.index("hermes doctor --fix")
