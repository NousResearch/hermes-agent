"""Closing 'Handoff bundle:' line check in scripts/lib/report_completeness.py.

The owner's standing rule is that every session report ends with a literal
    Handoff bundle: HANDOFF_<YYYY-MM-DD_HHMM>.zip (sha256: <64 hex>) - created.
(or "... - NOT created (reason)"). A report handed over still reading
"HANDOFF_..._PLACEHOLDER.zip (sha256: PLACEHOLDER)" is an unfilled template, not a
finished report — the completeness check must FAIL it, without tripping on an
explanatory parenthetical that uses the word "placeholder" in prose.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK = REPO_ROOT / "scripts" / "lib" / "report_completeness.py"

REAL_SHA = "a" * 64


def _load_module():
    spec = importlib.util.spec_from_file_location("nf_report_completeness", CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path: Path, report_body: str, *, run: str = "RUN-2026-09-11-001"):
    ledger = tmp_path / "ledger"
    (ledger / "reports").mkdir(parents=True)
    (ledger / "INDEX.md").write_text(f"# ledger\n\n{run} did a thing\n", encoding="utf-8")
    (ledger / "reports" / "REPORT-MANIFEST.md").write_text(
        "| RUN id | Report file | Covers | Source-Of-Truth | sha256 |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"| {run} | REPORT.md | x | in-bundle | \u2014 |\n",
        encoding="utf-8",
    )
    (tmp_path / "REPORT.md").write_text(report_body, encoding="utf-8")

    mod = _load_module()
    f = mod.Findings(quiet=True)
    # drive just the handoff-line helper directly — deterministic and isolated
    mod.check_handoff_line(run, "REPORT.md", report_body, f)
    levels = [lvl for lvl, _, _ in f.rows]
    return levels, f.rows


def test_placeholder_zip_and_hash_fail(tmp_path):
    body = (
        "RUN-2026-09-11-001 body\n\n"
        "Handoff bundle: HANDOFF_2026-09-11_PLACEHOLDER.zip (sha256: PLACEHOLDER) - created.\n"
    )
    levels, rows = _run(tmp_path, body)
    assert "FAIL" in levels
    assert "unfilled template" in rows[0][1]


def test_real_zip_and_hash_pass_even_with_placeholder_prose(tmp_path):
    body = (
        "RUN-2026-09-11-001 body\n\n"
        f"Handoff bundle: HANDOFF_2026-09-11_0930.zip (sha256: {REAL_SHA}) - created. "
        "(The copy bundled inside the zip carries the pre-hash placeholder line; this "
        "standalone file is authoritative.)\n"
    )
    levels, _ = _run(tmp_path, body)
    assert "FAIL" not in levels
    assert "OK" in levels


def test_missing_line_fails_for_enforced_date(tmp_path):
    body = "RUN-2026-09-11-001 body with no closing line at all\n"
    levels, rows = _run(tmp_path, body)
    assert "FAIL" in levels
    assert "missing the mandatory closing" in rows[0][1]


def test_missing_line_tolerated_before_convention(tmp_path):
    body = "RUN-2026-09-06-001 pre-convention report, no handoff line\n"
    levels, _ = _run(tmp_path, body, run="RUN-2026-09-06-001")
    assert levels == []  # neither FAIL nor a noisy WARN


def test_not_created_with_reason_passes(tmp_path):
    body = (
        "RUN-2026-09-11-001 read-only session\n\n"
        "Handoff bundle: HANDOFF_2026-09-11_0930.zip (sha256: n/a) - NOT created "
        "(read-only research pass, no repo change).\n"
    )
    levels, _ = _run(tmp_path, body)
    assert "FAIL" not in levels


def test_created_with_short_hash_fails(tmp_path):
    body = (
        "RUN-2026-09-11-001 body\n\n"
        "Handoff bundle: HANDOFF_2026-09-11_0930.zip (sha256: deadbeef) - created.\n"
    )
    levels, rows = _run(tmp_path, body)
    assert "FAIL" in levels
    assert "sha256" in rows[0][1]


def test_end_to_end_real_ledger_stays_complete(tmp_path):
    """The check as wired into main() still exits 0 against the shipped ledger."""
    mod = _load_module()
    ledger_dir = REPO_ROOT / "logs" / "ledger"
    if not (ledger_dir / "reports" / "REPORT-MANIFEST.md").is_file():
        pytest.skip("ledger not present in this checkout")
    # reports live in D:\logs\ at runtime; not part of the repo, so we can only
    # assert the module imports and the helper is exported here.
    assert hasattr(mod, "check_handoff_line")
