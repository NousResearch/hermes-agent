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

Build and reconcile platform parity campaigns without confusing candidate
artifacts with delivered behavior. The campaign ledger records mutable work;
the external contract registry preserves immutable product meaning.

## When to Use

- Building a new platform Feature Parity & Alignment campaign.
- Reconciling a campaign after competing branches, packets, or specifications.
- Evaluating whether a capability is blocked, wired, on main, or released.
- Repairing publication authority, provenance, or supersession topology.

Do not use this workflow to certify live behavior from prose, local files, or a
focused test result alone.

## Prerequisites

- A pinned upstream `main` commit.
- The approved capability specification and source anchors.
- The live issue and pull-request graph for the target repository.
- Official provider documentation for the product surface being evaluated.

## Procedure

1. **Pin the evidence surface.** Record upstream SHA, UTC capture time, official
   documentation anchors, and the live publication graph. Completion criterion:
   every factual campaign claim resolves to a pinned source.
2. **Recover canonical capability identity.** Preserve ordered `id`, `name`,
   `product_state`, and `source_anchor` tuples. Never reuse an ID for a new
   meaning. Completion criterion: the computed contract digest matches a
   registered revision in
   `docs/architecture/feature-parity/contracts.json`.
3. **Write the ledger.** Follow `references/ledger-contract.md`; include every
   required field even when its value is an empty list. Completion criterion:
   no capability is absent, duplicated, or reordered.
4. **Reconcile publication authority.** Give each active row exactly one open or
   merged authoritative pull request. Mark related work as a dependency and
   obsolete work as superseded. Completion criterion: no PR owns two rows in
   this ledger or another repository ledger.
5. **Separate candidate states.** Use `candidate_blocked` for an explicit gate,
   `candidate_unwired` for tested code without a runtime consumer, and
   `candidate_open` only when implementation, tests, consumer, and exact head
   SHA are present. Completion criterion: artifact evidence alone never
   advances delivery state.
6. **Protect product decisions.** Conditional, deferred, rejected, and pair-gap
   rows require an explicit decision and cannot advance through a product gate
   by acquiring code. Completion criterion: rejected and deferred rows own no
   production, test, or consumer paths.
7. **Preserve bounded architecture.** Declare forbidden growth paths and reject
   non-canonical, traversal, absolute, or Windows-style ledger paths.
   Completion criterion: no accepted implementation grows a declared god-file
   surface.
8. **Verify terminal evidence.** A released row requires one exact merged SHA,
   a commit-bound Actions receipt, a hashed in-repository live receipt, and at
   least two distinct exact-head reviews independent of the PR author.
   Completion criterion: repository validation resolves every receipt and hash.
9. **Validate the repository.** Run:

   ```text
   terminal(command="python scripts/ci/validate_feature_parity_ledger.py --repository-root .")
   ```

   Completion criterion: the command reports `VALID` with no suppressed rows.
10. **Publish current state.** Derive human-readable reports from the validated
    ledger and update supersession links. Completion criterion: public status
    matches the machine-readable state exactly.

## Hard Invariants

- Canonical semantics live in the external append-only registry, not in a
  self-authenticating ledger.
- A packet, patch, branch, or focused suite is evidence, not delivery.
- One authoritative pull request cannot own multiple capability rows.
- Candidate and terminal states require the exact publication state they claim.
- `candidate_open`, `on_main_unverified`, and `released` require a real runtime
  consumer.
- A blocked row names its blocker; an unwired row names its wiring gap.
- No actor certifies their own release work.
- Repository-wide validation owns cross-ledger collisions and receipt hashes.

## Pitfalls

- Recomputing a ledger hash after redefining a row does not authorize the new
  meaning; append a reviewed registry revision instead.
- Non-empty strings are not terminal evidence. Use the structured CI, receipt,
  and review objects from the contract reference.
- Closed packet-era PRs remain provenance, not current publication authority.
- Request builders and dormant modules are not runtime consumers.
- Keep the current-state report generated from the ledger; hand-maintained
  counts drift quickly.

## Verification

Run the validator and focused tests through `terminal`:

```text
terminal(command="python scripts/ci/validate_feature_parity_ledger.py --repository-root .")
terminal(command="python -m pytest -q tests/scripts/test_feature_parity_ledger.py tests/scripts/test_feature_parity_ledgers_repository.py")
terminal(command="git diff --check")
```

Verification is complete only when all three commands succeed against the exact
head proposed for review.
