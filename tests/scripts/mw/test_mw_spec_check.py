"""Tests for scripts/mw-spec-check.py §13.1 traceability check (MW-P3-I3).

Loads the checker by file path (its name has a dash, not importable normally),
exercises check_g_testid_map with synthetic fixtures, plus a smoke test that the
real repo passes with the §13.1 tally in the output.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_CHECK = REPO_ROOT / "scripts" / "mw-spec-check.py"

_spec = importlib.util.spec_from_file_location("mw_spec_check", SPEC_CHECK)
assert _spec and _spec.loader
msc = importlib.util.module_from_spec(_spec)
sys.modules["mw_spec_check"] = msc
_spec.loader.exec_module(msc)


def _grow(code: str, verdict: str = "[G] §10-2") -> str:
    """A master-table row line the checker accepts (verdict cell = cells[4])."""
    return f"| {code} | desc | ref | {verdict} |"


def _prep(monkeypatch, tmp_path, map_rows, test_defs, raw_body=None):
    """Point module globals at fixtures; map_rows = list of (code, tool, tid, status).

    map_rows entries may be raw pre-joined strings (for duplicate/edge fixtures).
    raw_body: if given, used verbatim as the test file body instead of test_defs.
    """
    lines = ["| row | tool | test_id | status |", "|---|---|---|---|"]
    for r in map_rows:
        lines.append(r if isinstance(r, str) else "| " + " | ".join(r) + " |")
    mp = tmp_path / "map.md"
    mp.write_text("\n".join(lines), encoding="utf-8")

    tdir = tmp_path / "tests"
    tdir.mkdir()
    if raw_body is not None:
        body = raw_body
    else:
        body = "\n".join(f"def {t}():\n    pass\n" for t in test_defs)
    (tdir / "test_x.py").write_text(body, encoding="utf-8")

    monkeypatch.setattr(msc, "MAP_FILE", mp)
    monkeypatch.setattr(msc, "TESTS_DIR", tdir)


def test_all_mapped_ok(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [("I1-05", "§10-2", "test_pg", "mapped")], ["test_pg"])
    errors = []
    tally = msc.check_g_testid_map([_grow("I1-05")], errors)
    assert errors == []
    assert "mapped(verified) 1" in tally


def test_missing_from_map_fails(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [], [])
    errors = []
    msc.check_g_testid_map([_grow("I1-05")], errors)
    assert any("ไม่มีใน mw-g-testid-map" in e for e in errors)


def test_mapped_test_not_exist_fails(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [("I1-05", "§10-2", "test_nope", "mapped")], [])
    errors = []
    msc.check_g_testid_map([_grow("I1-05")], errors)
    assert any("ไม่พบ test" in e for e in errors)


def test_unknown_external_tag_fails(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [("I1-05", "§10-2", "weird-tool", "external")], [])
    errors = []
    msc.check_g_testid_map([_grow("I1-05")], errors)
    assert any("external tag" in e for e in errors)


def test_allowed_external_tag_ok(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [("I4-02", "§10-9", "gitleaks", "external")], [])
    errors = []
    tally = msc.check_g_testid_map([_grow("I4-02", "[G] §10-9")], errors)
    assert errors == []
    assert "external(unverified-here) 1" in tally


def test_stale_map_row_fails(tmp_path, monkeypatch):
    _prep(
        monkeypatch,
        tmp_path,
        [("I1-05", "§10-2", "test_pg", "mapped"), ("I9-99", "§10-2", "test_pg", "mapped")],
        ["test_pg"],
    )
    errors = []
    msc.check_g_testid_map([_grow("I1-05")], errors)
    assert any("stale" in e for e in errors)


def test_pending_i2e_counted_not_failed(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [("I1-01", "§10-8", "mw-backend-check", "pending-i2e")], [])
    errors = []
    tally = msc.check_g_testid_map([_grow("I1-01", "[G] §10-8")], errors)
    assert errors == []
    assert "pending-i2e 1" in tally


def test_r_style_row_is_detected(tmp_path, monkeypatch):
    # regression: "I3-R7" style codes must be caught (were silently skipped before)
    _prep(monkeypatch, tmp_path, [("I3-R7", "§10-4", "test_lock", "mapped")], ["test_lock"])
    errors = []
    tally = msc.check_g_testid_map([_grow("I3-R7", "[G] §10-4")], errors)
    assert errors == []
    assert "1 [G]" in tally


# --- hardening after GPT-5 review (5 false-pass paths) ---------------------

def test_duplicate_map_row_fails(tmp_path, monkeypatch):
    _prep(
        monkeypatch,
        tmp_path,
        [
            "| I1-05 | §10-2 | test_pg | mapped |",
            "| I1-05 | §10-8 | mw-backend-check | pending-i2e |",
        ],
        ["test_pg"],
    )
    errors = []
    msc.check_g_testid_map([_grow("I1-05")], errors)
    assert any("แถวซ้ำ" in e for e in errors)


def test_skipped_test_is_not_valid_proof(tmp_path, monkeypatch):
    body = "import pytest\n\n@pytest.mark.skip\ndef test_skipped():\n    pass\n"
    _prep(monkeypatch, tmp_path, [("I1-05", "§10-2", "test_skipped", "mapped")], [], raw_body=body)
    errors = []
    msc.check_g_testid_map([_grow("I1-05")], errors)
    assert any("skip" in e for e in errors)


def test_comment_or_string_def_not_counted(tmp_path, monkeypatch):
    body = 'x = "def test_fake(): pass"\n# def test_fake():\n'
    _prep(monkeypatch, tmp_path, [("I1-05", "§10-2", "test_fake", "mapped")], [], raw_body=body)
    errors = []
    msc.check_g_testid_map([_grow("I1-05")], errors)
    assert any("ไม่พบ test" in e for e in errors)


def test_g_outside_verdict_column_flagged(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [("I1-05", "§10-2", "test_pg", "mapped")], ["test_pg"])
    # [G] wrongly placed in the 2nd column (desc), verdict has none
    bad = "| I1-05 | [G] desc | ref | plain |"
    errors = []
    msc.check_g_testid_map([bad], errors)
    assert any("นอกช่องคำตัดสิน" in e for e in errors)


def test_pending_non_backend_tool_fails(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [("I1-05", "§10-2", "mw-backend-check", "pending-i2e")], [])
    errors = []
    msc.check_g_testid_map([_grow("I1-05")], errors)
    assert any("pending ได้เฉพาะ §10-8" in e for e in errors)


def test_pending_makes_tally_incomplete(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [("I1-01", "§10-8", "mw-backend-check", "pending-i2e")], [])
    errors = []
    tally = msc.check_g_testid_map([_grow("I1-01", "[G] §10-8")], errors)
    assert errors == []
    assert "INCOMPLETE" in tally


def test_map_tool_mismatch_master_fails(tmp_path, monkeypatch):
    # master row is §10-2 but map claims §10-8 pending → mismatch error
    _prep(monkeypatch, tmp_path, [("I1-05", "§10-8", "mw-backend-check", "pending-i2e")], [])
    errors = []
    msc.check_g_testid_map([_grow("I1-05", "[G] §10-2")], errors)
    assert any("ไม่ตรงเครื่องมือในตารางแม่" in e for e in errors)


def test_g_after_verdict_column_flagged(tmp_path, monkeypatch):
    _prep(monkeypatch, tmp_path, [("I1-05", "§10-2", "test_pg", "mapped")], ["test_pg"])
    # [G] appears in an EXTRA column after the verdict cell
    bad = "| I1-05 | desc | ref | plain | [G] extra |"
    errors = []
    msc.check_g_testid_map([bad], errors)
    assert any("นอกช่องคำตัดสิน" in e for e in errors)


def test_strict_mode_fails_on_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("MW_SPEC_REQUIRE_G13_COMPLETE", "1")
    _prep(monkeypatch, tmp_path, [("I1-01", "§10-8", "mw-backend-check", "pending-i2e")], [])
    errors = []
    msc.check_g_testid_map([_grow("I1-01", "[G] §10-8")], errors)
    assert any("strict" in e for e in errors)


def test_default_mode_pending_not_fail(tmp_path, monkeypatch):
    monkeypatch.delenv("MW_SPEC_REQUIRE_G13_COMPLETE", raising=False)
    _prep(monkeypatch, tmp_path, [("I1-01", "§10-8", "mw-backend-check", "pending-i2e")], [])
    errors = []
    msc.check_g_testid_map([_grow("I1-01", "[G] §10-8")], errors)
    assert errors == []


def test_real_repo_passes_with_g13_tally():
    out = io.StringIO()
    with redirect_stdout(out):
        rc = msc.main()
    text = out.getvalue()
    assert rc == 0, text
    assert "§13.1" in text
    assert "pending-i2e" in text
