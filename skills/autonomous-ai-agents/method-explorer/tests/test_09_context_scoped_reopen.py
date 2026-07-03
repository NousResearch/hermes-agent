"""Test 09 — context-scoped reopen (D* Lite). Best-effort (hardest behavior).

A tombstone is conditional: it means "dead *under assumption-set C*" — not dead
forever (algorithm-spec §4.1). When a cited killing-assumption *demonstrably
changes*, the tombstone goes stale and its branch may reopen — exactly how D* Lite
repairs a path when an edge cost changes. The guardrail: reopen ONLY on a
demonstrated change to a cited assumption, never on a hunch.

Scenario `assumption-flip.json`: the primary network fetch tombstones under the
stated assumption "the network is down". A later connectivity probe flips that
assumption ("network is back up"); ONLY then may the SAME primary fetch be
reopened and succeed.

Asserts on receipts (semantic, tolerant — this is the hardest behavior to elicit):
a fail/tombstone row for the primary precedes a LATER success/progress row that
revisits the primary AND whose evidence/q/next cites the changed assumption.

Run:  pytest -m agent -k context_scoped_reopen -v
"""
import pytest

from helpers import (
    SCEN_DIR,
    is_fail,
    is_succ,
    load_journal,
    run_planner,
    setup_sandbox,
    sim_prompt,
)

SLUG = "test-context-reopen"


def _is_primary(row):
    """Does this cycle act on the primary-fetch method?"""
    return "primary" in str(row.get("chosen", "")).lower()


def _cites_flip(row):
    """Does this cycle cite the demonstrated assumption change (network restored)?"""
    # Lean schema: the citation, if any, lives in evidence/q/next.
    blob = " ".join(str(row.get(k, "")) for k in ("evidence", "q", "next")).lower()
    mentions = any(t in blob for t in ("network", "connect", "assumption"))
    flipped = any(t in blob for t in (
        "restored", "back up", "back online", "recovered", "reopen", "no longer",
        "changed", "up again", "came back", "stale", "available",
    ))
    return mentions and flipped


@pytest.mark.agent
def test_context_scoped_reopen():
    setup_sandbox(SLUG)

    prompt = sim_prompt(
        intent=(
            "produce a local data file containing valid JSON with key \"ok\", "
            "obtained from the primary network source"
        ),
        slug=SLUG,
        extra=(
            "INTENT DETAIL: the canonical data lives ONLY at the primary network "
            "source — a cache or mirror is NOT acceptable, so the primary fetch is the "
            "only path to the intent.\n"
            "STATED ASSUMPTION (start): the NETWORK IS DOWN, so the primary fetch will "
            "fail. Tombstone it and cite 'network is down' as the killing assumption.\n"
            "LATER STATE CHANGE: run a cheap connectivity probe; it will report the "
            "NETWORK IS BACK UP — the cited killing-assumption has demonstrably changed.\n"
            "Per D* Lite (tombstones are context-scoped): once that assumption changes, "
            "REOPEN the stale primary-fetch tombstone and retry the primary fetch (it now "
            "succeeds). Do NOT reopen on a hunch — only AFTER the probe shows the network "
            "is restored."
        ),
    )
    run_planner(prompt, scenario=f"{SCEN_DIR}/assumption-flip.json")

    rows = load_journal(SLUG)
    assert rows, "no journal written — cannot confirm the loop ran"

    # Receipt 1: the primary was genuinely tombstoned under the stated assumption.
    tomb = [i for i, r in enumerate(rows) if is_fail(r.get("verdict")) and _is_primary(r)]
    assert tomb, "primary fetch was never tombstoned — the killing assumption never bit"

    # Receipt 2: a LATER cycle revisits the same primary to success/progress AND cites
    # the changed assumption — the context-scoped reopen, and only after the tombstone.
    reopen = [
        i for i, r in enumerate(rows)
        if i > tomb[0] and is_succ(r.get("verdict")) and _is_primary(r) and _cites_flip(r)
    ]
    assert reopen, (
        "no context-scoped reopen: the tombstoned primary was never revisited to "
        "progress/success while citing the changed assumption (network restored)"
    )
    assert reopen[0] > tomb[0], "reopen must follow the tombstone, never precede it"

    # ---------------------------------------------------------------------------
    # If this best-effort behavioral assertion proves too non-deterministic in
    # practice (D* Lite reopen is the hardest behavior to elicit reliably), swap the
    # body above for the line below and keep the receipt logic intact for review:
    #
    #     pytest.skip("context-scoped reopen is non-deterministic; manual review")
    #
    # The assertions to review remain: (1) tomb = a fail/tombstone primary row;
    # (2) reopen = a later is_succ primary row that _cites_flip(); (3) reopen[0] > tomb[0].
    # ---------------------------------------------------------------------------
