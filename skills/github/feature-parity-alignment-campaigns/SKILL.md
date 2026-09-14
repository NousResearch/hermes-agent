---
name: feature-parity-alignment-campaigns
description: Govern platform parity campaigns with verifiable ledgers.
version: 2.0.0
author: Axl Ibiza (andrexibiza), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Campaign, Parity, Platforms, Interlock, Evidence, Release]
    related_skills: [github-issues, github-pr-workflow, codebase-inspection]
---

# Feature Parity & Alignment Campaigns

Use this skill for a platform parity campaign, a campaign reconciliation, or any
claim that a campaign capability is implemented, shipped, superseded, or done.

The campaign is not a pile of issues, branches, packets, or green tests. It is
one versioned semantic contract whose rows have exactly one current product
meaning, one publication authority, explicit runtime consumers, and terminal
release evidence.

## Required sequence

1. Pin current upstream `main`, official provider documentation, and the live
   issue/PR graph.
2. Recover the canonical capability IDs, names, and product dispositions from
   the approved specification. Never repurpose a row ID.
3. Write or update the campaign ledger described in
   `references/ledger-contract.md`.
4. Run `scripts/ci/validate_feature_parity_ledger.py <ledger.json>` before any
   implementation or publication work.
5. Reconcile duplicates and stale candidates. Every active capability has
   exactly one authoritative publication route; all other routes are
   dependencies or explicitly superseded.
6. Start implementation from current main or an approved predecessor. Record a
   behavioral RED, the smallest complete GREEN, focused and related tests, and
   runtime consumer wiring.
7. Keep feature code out of forbidden god files. Extract a stable seam first or
   land a new bounded module plus an accepted consumer.
8. Update the ledger at each state transition. Artifact evidence never advances
   delivery state by itself.
9. Call a row `released` only after an exact merged commit, head-bound CI, live
   platform receipt where the contract requires it, and two independent
   approvals.
10. Re-run the ledger validator after merge and whenever a linked issue, PR,
    product disposition, consumer, or release receipt changes.

## Hard invariants

- Canonical row identity is `(id, name, product_state)` and is digest-locked.
- A packet, patch, branch, or focused suite is evidence, not delivery.
- One authoritative PR cannot own multiple capability rows.
- `candidate_open`, `on_main_unverified`, and `released` require a real runtime
  consumer, not merely a request builder or dormant module.
- Rejected capabilities cannot carry implementation paths.
- Conditional, deferred, rejected, and pair-gap rows require an explicit
  decision.
- No feature code grows a declared forbidden surface.
- `released` requires terminal evidence; no actor certifies its own work.

## Outputs

A complete campaign produces:

- an executable ledger;
- a human-readable current-state report derived from that ledger;
- one authoritative issue/PR route per active capability;
- immutable provenance and supersession links;
- CI and live-system receipts tied to exact commits;
- an explicit terminal condition with no ambiguous or artifact-only rows.
