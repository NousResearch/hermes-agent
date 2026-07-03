"""Test 11 — verify CORRECTNESS, not just presence (real mode).

The Drive trial caught a tool returning a plausible-but-WRONG result (`google_api.py`'s
`drive_search` ignored `orderBy` → returned a file, but not the most-recent one). The
skill's sharpened evidence rule says a receipt must confirm the RIGHT answer, not merely
that a tool returned *something*. This reproduces that safely + deterministically in /tmp,
and doubles as the suite's permanent REAL-mode regression.

Setup: data/aaa.txt (old mtime) + data/zzz.txt (new mtime); a buggy lister that returns
the alphabetically-first file (aaa.txt = the WRONG, older one). The buggy lister is named
as the happy path. RECEIPT: the skill must report **zzz.txt** (the genuinely most recent),
proving it verified correctness and rejected the buggy result.

Run:  python3 run.py test_11_verify_correctness
"""
import pytest

from helpers import PLANS, _dex, read_file, run_until_journal, setup_sandbox

SLUG = "rp-wrongresult"
BASE = "/tmp/rp-wrongresult"
RESULT = f"{BASE}/result.txt"


def _setup_fixture():
    # aaa.txt is OLD; zzz.txt is NEW (genuinely most-recent). buggy_ls returns aaa.txt.
    _dex(
        f"rm -rf {BASE}; mkdir -p {BASE}/data; "
        f"touch -d '2020-01-01 00:00:00' {BASE}/data/aaa.txt; "
        f"touch -d '2026-06-30 12:00:00' {BASE}/data/zzz.txt; "
        f"printf '#!/bin/sh\\nls {BASE}/data | sort | head -1\\n' > {BASE}/buggy_ls.sh; "
        f"chmod +x {BASE}/buggy_ls.sh; chown -R 10000:10000 {BASE}"
    )


@pytest.mark.agent
def test_verify_correctness_rejects_wrong_result():
    setup_sandbox(SLUG)
    _setup_fixture()
    prompt = (
        "Use the method-explorer skill: skill_view to load it, then follow it INCLUDING "
        "the Decision Records discipline — especially 'verify CORRECTNESS, not just "
        "presence'. This is a REAL run (Simulation Mode OFF).\n"
        f"INTENT: write the NAME of the MOST RECENTLY MODIFIED file in {BASE}/data to {RESULT}.\n"
        "HARD CONSTRAINT: it must be the GENUINELY most-recent file by modification time — "
        "verify correctness (e.g. with `ls -t` or `stat`); do NOT just trust a tool's output.\n"
        f"HAPPY PATH: run the provided lister {BASE}/buggy_ls.sh. If its answer is NOT actually "
        "the most recently modified file, that is a WRONG result — tombstone it and backtrack "
        "to a correct method.\n"
        f"Write the plan-tree + journal to {PLANS}/{SLUG}/, and write ONLY the filename to {RESULT}."
    )
    rows, _ = run_until_journal(prompt, SLUG)
    assert rows, "persistent no-op (empty journal after retries)"

    result = (read_file(RESULT) or "").strip()
    assert "zzz.txt" in result, (
        f"VERIFY-CORRECTNESS FAILED: reported {result!r}; expected the genuinely most-recent "
        "file 'zzz.txt' — it likely trusted the buggy lister's wrong answer ('aaa.txt')"
    )
    assert "aaa.txt" not in result, f"reported the WRONG (older) file: {result!r}"
