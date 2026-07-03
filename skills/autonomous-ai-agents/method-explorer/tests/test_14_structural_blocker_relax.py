"""Test 14 — STRUCTURAL blocker → fast relax (don't brute-force the iteration budget).

Encodes the lesson from the test_02 diagnostic: when the happy-path target is
structurally unwritable (root-owned dir), the loop must recognize STRUCTURAL, stop
trying variants, and RELAX the path soft constraint to a writable fallback — fast.

This is self-pinning: if the model brute-forces 20+ write methods against the locked
path, it exhausts the Hermes oneshot iteration budget and never reaches the fallback
(→ no output → FAIL). If it escalates fast, it writes the fallback within budget
(→ output present → PASS). So asserting success at the fallback path IS the pin.

Fixture: BASE/locked is root:root 755 (agent uid 10000 cannot write); BASE/free is
hermes-owned (writable). Intent targets the locked path; the path is only a SOFT
constraint (HARD = valid JSON with key "ok"), and the free dir is named as the fallback.

Run:  python3 run.py test_14_structural_blocker_relax
"""
import json

import pytest

from helpers import PLANS, _dex, is_fail, read_file, run_until_journal, setup_sandbox

SLUG = "test-structural-relax"
BASE = f"/tmp/{SLUG}"
LOCKED = f"{BASE}/locked/out.json"   # structurally unwritable (root-owned dir)
FREE = f"{BASE}/free/out.json"       # writable fallback (hermes-owned dir)


def _setup_fixture():
    # free/ writable by hermes; locked/ root-owned 755 → uid 10000 gets EACCES.
    _dex(
        f"rm -rf {BASE}; mkdir -p {BASE}/free; chown -R 10000:10000 {BASE}; "
        f"mkdir -p {BASE}/locked; chown root:root {BASE}/locked; chmod 755 {BASE}/locked"
    )


@pytest.mark.agent
def test_structural_blocker_relaxes_fast_to_fallback():
    setup_sandbox(SLUG)
    _setup_fixture()
    prompt = (
        "Use the method-explorer skill: skill_view to load it, then follow it "
        "INCLUDING the Key Questions (esp. Locality) and the Next-Best-Action Ladder's "
        "BUDGET DISCIPLINE (a STRUCTURAL blocker → relax/backtrack fast, do NOT "
        "brute-force variants of a blocked method). This is a REAL run (Simulation OFF).\n"
        f"INTENT: write valid JSON with a top-level key \"ok\" to {LOCKED}.\n"
        "HARD CONSTRAINT (inviolable): the file content must be valid JSON containing "
        "key \"ok\". The PATH is NOT hard.\n"
        f"SOFT CONSTRAINT (relaxable): prefer the path {LOCKED}; if it is structurally "
        f"unwritable, RELAX the path and write to the known writable fallback {FREE}.\n"
        f"HAPPY PATH: write to {LOCKED}. If that path is unwritable (a STRUCTURAL "
        "permission blocker), do NOT enumerate many write methods — tombstone it and "
        f"relax to {FREE} immediately.\n"
        f"Write the plan-tree + journal to {PLANS}/{SLUG}/."
    )
    rows, _ = run_until_journal(prompt, SLUG)
    assert rows, "persistent no-op (empty journal after retries)"

    # THE PIN: the deliverable exists at the writable fallback. Brute-forcing the locked
    # path would exhaust the iteration budget and never get here.
    content = read_file(FREE)
    assert content is not None, (
        f"no deliverable at the writable fallback {FREE} — the loop likely brute-forced "
        "the structurally-blocked path and exhausted its iteration budget instead of "
        "relaxing fast"
    )
    data = json.loads(content)
    assert "ok" in data, f"fallback JSON missing key 'ok': {content!r}"

    # And it correctly tombstoned the locked path (a fail/tombstone before the relax).
    assert any(is_fail(r.get("verdict")) for r in rows), (
        "no failure verdict — the structural blocker on the locked path was never "
        "registered, so the relax wasn't a real escalation"
    )
