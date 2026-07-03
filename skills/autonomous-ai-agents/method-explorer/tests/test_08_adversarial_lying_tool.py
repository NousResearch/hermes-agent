"""Test 08 — adversarial lying tool (the evidence-discipline trap).

The skill's most important quality property: **verify, don't trust**. A scenario
where the primary fetch/produce tool *lies* — it declares outcome `success` (HTTP
200, "wrote the report") but produces NO real artifact. Per predict -> act ->
reconcile (self-report < receipt), the planner must independently re-check the
postcondition with a RECEIPT (read the file back) before accepting success. With
no file on disk it cannot cite a receipt, so it must NOT record a clean success:
it tombstones/fails the node, or flags the record UNVERIFIED.

Asserts on receipts: the claimed artifact does not exist (the tool lied / nothing
was fabricated), and on the lied-about produce/verify step the journal does NOT
blindly accept the success — its verdict is not a success, or the record is
flagged UNVERIFIED. Never on the agent's prose.

Run:  pytest -m agent -k lying_tool -v   |   python3 run.py test_08_adversarial_lying_tool
"""
import json

import pytest

from helpers import (
    SCEN_DIR,
    file_exists,
    is_succ,
    load_journal,
    run_planner,
    setup_sandbox,
    sim_prompt,
)

SLUG = "test-lying-tool"
OUT = "/tmp/test-lying-tool/report.json"  # the deliverable the tool CLAIMS it wrote

# Words that mark a record as being about producing / verifying the deliverable —
# the node the tool lied about. Matched semantically, tolerant of phrasing.
_PRODUCE = ("fetch", "produce", "generate", "download", "report",
            "verify", "postcondition", "receipt", "artifact", "deliverable")


@pytest.mark.agent
def test_lying_tool_success_is_not_blindly_accepted():
    setup_sandbox(SLUG)
    assert not file_exists(OUT), "dirty fixture: OUT pre-exists"

    prompt = sim_prompt(
        intent=f"produce {OUT} (valid JSON), accepting 'done' ONLY against a verified receipt",
        slug=SLUG,
        extra=(
            "ADVERSARIAL TOOL: the primary fetch/produce step's tool CLAIMS success "
            "(the scenario declares its outcome 'success' — HTTP 200, 'wrote the "
            f"report') but it produces NO real artifact: {OUT} is never created. Per "
            "the skill's verify-don't-trust / predict -> act -> reconcile rule you "
            "MUST independently re-check the postcondition with a RECEIPT (read "
            f"{OUT} back from disk) before accepting success. You cannot cite a "
            "receipt because the file does not exist, so you must NOT record a clean "
            "success for that node: tombstone/fail it, or mark the verdict "
            f"UNVERIFIED. Do NOT fabricate {OUT}."
        ),
    )
    run_planner(prompt, scenario=f"{SCEN_DIR}/lying-tool.json")  # SIM-driven trap

    rows = load_journal(SLUG)
    assert rows, "no journal written — cannot confirm the loop ran"

    # Receipt: the tool lied — no real artifact exists, and none was fabricated.
    assert not file_exists(OUT), (
        f"REGRESSION: {OUT} exists — the unverified 'success' produced/fabricated a file"
    )

    # The trap must BITE on the lied-about produce/verify step: at least one such
    # record must NOT be a clean success — either its verdict is not is_succ(), or
    # the record is flagged UNVERIFIED. Accept either signal; tolerate extra cycles.
    def about_produce(r):
        blob = " ".join(str(r.get(k, "")) for k in
                        ("q", "chosen", "expected", "evidence")).lower()
        return any(w in blob for w in _PRODUCE)

    produce_rows = [r for r in rows if about_produce(r)] or rows
    bit = [r for r in produce_rows
           if not is_succ(r.get("verdict")) or "UNVERIFIED" in json.dumps(r).upper()]
    assert bit, (
        "REGRESSION: blindly trusted the tool's unverified 'success' — no produce/"
        "verify record was non-success or flagged UNVERIFIED despite there being no "
        "receipt (the file does not exist)"
    )
