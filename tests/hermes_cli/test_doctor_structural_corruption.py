"""#88587 — doctor must name STRUCTURAL state.db corruption honestly.

The write-health probe's failure used to be reported as "FTS write corruption" unconditionally,
routing operators to `--fix` / `sessions repair` (FTS rebuilds that cannot repair canonical-table
damage) and to the .malformed-backup beside the DB (a snapshot of the same corrupt file). The
discriminator maps integrity_check damage through sqlite_master.rootpage and keeps the FTS path
only when every damaged object is a Hermes FTS shadow.
"""

import contextlib
import io
import sqlite3

from hermes_cli.doctor_report import Finding
from hermes_cli.doctor_state import _state_db_health
from hermes_state import SessionDB
from hermes_state_repair import integrity_damage_is_structural, state_db_has_structural_damage


def test_integrity_damage_classifier_maps_tree_ids_through_rootpage():
    """Field mappings from #88587: tree 5 -> sessions and tree 15 -> gateway_routing are
    structural; a damaged messages_fts shadow tree is not; a lookalike foreign object,
    a canonical index named in a "missing from index" line, and the freelist are."""
    fts_only = [
        "Tree 12 page 9: btreeInitPage() returns error code 11",
        "row 3 missing from index messages_fts_trigram_idx",
    ]
    master = [(12, "table", "messages_fts_data"), (5, "table", "sessions"),
              (15, "table", "gateway_routing"), (40, "table", "archive_fts_data")]
    assert integrity_damage_is_structural(fts_only, master) is False
    assert integrity_damage_is_structural(["Tree 5  page 421385: btreeInitPage() returns error code 11"], master)
    assert integrity_damage_is_structural(["Tree 15 page 15 cell 0: 2nd reference to page 5453"], master)
    assert integrity_damage_is_structural(["Tree 40 page 40: btreeInitPage() returns error code 11"], master)
    assert integrity_damage_is_structural(["row 1 missing from index sqlite_autoindex_delivery_obligations_1"], master)
    assert integrity_damage_is_structural(["Freelist: invalid page number 167772160"], master)
    # Unparseable / unknown-tree lines keep the FTS wording (incomplete, never wrong).
    assert integrity_damage_is_structural(["Tree 999 page 1: garbage", "*** in database main ***"], master) is False


def _seed(tmp_path, rows=120):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", source="cli")
    for i in range(rows):
        db.append_message("s1", "user", f"hello {i} alpha beta " + "lorem " * 30)
    db.close()
    raw = sqlite3.connect(db_path)
    raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    page_size = raw.execute("PRAGMA page_size").fetchone()[0]
    root = raw.execute("SELECT rootpage FROM sqlite_master WHERE name='sessions'").fetchone()[0]
    raw.close()
    return db_path, page_size, root


def _run_doctor(db_path, should_fix):
    finding = Finding()
    with contextlib.redirect_stdout(io.StringIO()):
        _state_db_health(finding, should_fix, db_path, "~/x")
    return finding


def test_doctor_routes_structural_damage_to_recover_not_fts_rebuild(tmp_path, monkeypatch):
    """Real torn ``sessions`` b-tree: doctor --fix must not run the FTS repair ladder (no
    .malformed-backup, nothing fixed) and must point at `hermes sessions recover` for THIS
    database with the profile pinned; a real FTS-only stomp still takes the FTS path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path, page_size, root = _seed(tmp_path)
    with open(db_path, "r+b") as f:
        f.seek((root - 1) * page_size + 8)
        f.write(b"\xff\xff" * 8)
    assert state_db_has_structural_damage(db_path) is True

    finding = _run_doctor(db_path, should_fix=True)
    assert finding.fixed == 0 and finding.issues == []
    (issue,) = finding.manual_issues
    assert "structural" in issue and "sessions recover" in issue and str(db_path) in issue
    assert "FTS write corruption" not in issue and "restore from the backup" not in issue
    assert not list(tmp_path.glob("state.db.malformed-backup-*"))

    fts_path = tmp_path / "fts" / "state.db"
    fts_db = SessionDB(db_path=fts_path)
    fts_db.create_session("s1", source="cli")
    for i in range(40):
        fts_db.append_message("s1", "user", f"hello {i} alpha")
    fts_db.close()
    raw = sqlite3.connect(fts_path)
    raw.execute("UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
    raw.commit()
    raw.close()
    assert state_db_has_structural_damage(fts_path) is False
    fts_finding = _run_doctor(fts_path, should_fix=False)
    assert fts_finding.manual_issues == []
    assert any("FTS" in i for i in fts_finding.issues)
