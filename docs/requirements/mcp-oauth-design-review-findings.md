# MCP OAuth Credential Store - Design Review Findings

Status: Draft
Audience: Hermes maintainers and contributors
Scope: Triage of external design-review feedback on the MCP OAuth credential store
Source: GitHub PR NousResearch/hermes-agent#100308, review comment by @webtecnica, 2026-09-01
Comment URL: https://github.com/NousResearch/hermes-agent/pull/100308#issuecomment-5494992847
Related: mcp-oauth-credential-store-requirements.md, ../architecture/mcp-oauth-credential-store-architecture.md, ../design/mcp-oauth*, ../plans/2026-09-01-mcp-oauth-chunk-1-implementation-plan.md

## 1. Purpose

The collaborator (@webtecnica) reviewed the full architecture document and Chunks 1, 3, 4, and 5
while the design is still draft. They have direct experience with credential-pool rotation races,
cross-process concurrent refresh/401 handling, and cross-profile credential isolation.

This document records their feedback verbatim-in-substance, assigns each item a disposition, and
identifies the architecture/design sections that must change. It does not itself modify those
documents.

## 2. Points affirmed by the review (no action required)

The reviewer explicitly endorsed the following decisions. Capture as validation; do not revisit.

- Removing rollback rather than fixing it. The staged in-memory adapter (Chunk 3) is correct
  because the old snapshot/remove/restore path cannot know which files belong to which flow under
  concurrency. Invariant "the active bundle is never touched during a flow" is easier to reason
  about and test.
- CAS with opaque 128-bit revisions not derived from token material, plus the explicit
  `replace_authorized` vs `compare_and_swap` split (admin lock required for the former). The
  operation matrix in architecture §8.3 is precise about lock acquisition per op.
- Fail-closed backend selection (§2.6, §11): no silent Keychain-to-plaintext fallback; `auto`
  reports its choice in diagnostics.
- Atomic write protocol (§9.3): `O_EXCL` temp file at `0600`, `fsync` + parent-dir fsync,
  `os.replace`. Matches existing Hermes practice.
- Idempotent migration with read-back verification and manual conflict resolution (§13);
  timestamps never win.
- Phased rollout with exit criteria (§18); the Phase 1 legacy-compatible backend means a release
  rollback does not orphan credentials.
- `auto` on Windows/Linux selecting the file backend (§2.6) is honest and documented; the
  versioned configuration migration is the right forward mechanism. No change.

## 3. Substantive concerns

### F-1. Keychain subprocess cost on the hot path

Every `load` / revision-probe through `security find-generic-password` is a subprocess spawn. The
in-process provider cache mitigates repeated loads, but a revision probe per auth-sensitive request
could still spawn `security` per request.

- Reviewer ask: specify how often the revision is probed, and whether there is an in-memory
  TTL/backoff for the probe. A cached bundle plus a slow revision check is usually a better trade
  than a subprocess per request.
- Disposition: Accept. Tuning decision, not structural.
- Action: Architecture must state a revision-probe cadence and an in-memory probe TTL/backoff
  policy. Add the probe-cadence contract to the Chunk 1 provider-cache design.

### F-2. Probe failure policy conflates auth rejection with transient transport failure

Commit requires "the configured authentication probe" to succeed (§6.3, §16). A probe that fails on
a network timeout (server briefly down) forces the user to redo the entire browser flow even
though the token is valid.

- Reviewer ask: distinguish probe outcomes - auth rejection (401 -> abort) vs transient transport
  failure (retry, or commit-and-let-runtime-discover). Current rule is safe but frustratingly
  strict.
- Disposition: Accept. Tuning decision.
- Action: Architecture §6.3 and §16 must define a probe-outcome taxonomy (reject / transient /
  success) and the commit behavior for each. Add test cases for both failure classes.

### F-3. `profile_id` canonicalization under symlinks

`profile_id` is derived deterministically from the canonical profile-scoped Hermes home with
`strict=False` expansion. If CLI and gateway processes reach the home through different symlink
spellings, they may not resolve to the same canonical form; the identity digest then changes and
credentials orphan silently.

- Reviewer ask: an explicit `realpath` / canonicalization rule, plus a contract test that two
  processes resolving the same profile through different path spellings agree on the digest.
- Disposition: Accept. Potential silent-orphan bug.
- Action: Architecture must specify the exact canonicalization (e.g. `os.path.realpath` after
  expansion) for `profile_id` derivation. Add the two-spelling agreement contract test to the
  Chunk 1 test matrix.

### F-4. Legacy-mode reader-incoherence window between Chunk 3 and Chunk 5

Chunk 3's compatibility backend still writes 3-4 separate files with "commit orders to minimize
inconsistency", so the reader-incoherence problem this project exists to eliminate persists in that
mode until Chunk 5 lands.

- Reviewer ask: state explicitly that Phase 2's exit criterion guarantees only "no destructive
  failure", not "coherent reads", until Phase 3.
- Disposition: Accept. Documentation gap, not a code change.
- Action: Architecture §18 (phased rollout / exit criteria) must add this qualification to the
  Phase 2 exit criterion.

### F-5. Wall-clock step between load and request shifts expiry classification

`expires_at = accepted_at_utc + expires_in` uses wall-clock UTC for persistence and classification.
Monotonic time is correctly reserved for in-process waits. A wall-clock step (NTP correction,
manual change) between load and request shifts classification. §7.1 recalculates state on every
load, which is the right mitigation.

- Reviewer ask: note the residual risk that a large clock step makes a token look valid longer
  than it is; consider whether `expires_in` should be re-anchored on reload when remaining lifetime
  is unknown.
- Disposition: Accept as documentation; re-anchoring is open (see §5).
- Action: Architecture §7.1 must document the residual clock-step risk. Decide and record whether
  reload re-anchors `expires_in`.

### F-6. Keychain duplicate-item ambiguity missing from the test matrix

Keychain write verification is read-back (§10.3) - endorsed. The design already maps
duplicate-item ambiguity (two items matching service+account) to a typed error, but the test
matrix in §17.1 does not list that case.

- Reviewer ask: add the duplicate-item case to the test matrix.
- Disposition: Accept. Test-coverage gap only.
- Action: Add duplicate-item ambiguity to architecture §17.1 test matrix.

## 4. Minor items

### F-7. Bound the revision prefix allowed in diagnostics

§15 prohibits full revisions in logs. The reviewer wants prefixes long enough to brute-force the
128-bit space also prohibited (4-hex prefix fine, 32-hex not), so the rule is testable.

- Disposition: Accept.
- Action: Architecture §15 must state a maximum logged revision-prefix length and a test asserting
  it.

### F-8. Chunk 4 transitional revision envelope - crash between legacy write and manifest write

The Chunk 4 "revision alongside legacy state" envelope is the trickiest part of the plan. The
manifest must be written atomically with the state it revisions, or a revision/state mismatch
results.

- Disposition: Accept.
- Action: Chunk 4 design must specify atomic manifest+state write ordering and add a dedicated test
  for a crash between the legacy write and the manifest write.

## 5. Open questions

- F-5 re-anchoring: should `expires_in` be re-anchored on reload when remaining lifetime is
  unknown? Not yet decided. Needs a maintainer decision before Chunk 1 finalizes clock handling.
- F-2 transient-failure behavior: retry vs commit-and-let-runtime-discover - pick one. Needs a
  maintainer decision.

## 6. Collaboration offered by reviewer

- Review Chunk 3's staged-adapter implementation against the MCP SDK's actual storage contract
  when it lands.
- Help write the cross-process CAS/concurrency test matrix (Chunk 4), especially the two-worker
  one-winner demonstration.
- Test the Keychain backend on the headless/gateway path once Phase 4 exists.

## 7. Overall assessment (reviewer)

The design is coherent and the staging approach is the right direction. The open questions are
mostly about how much work the runtime hot path does per request and how strict the probe/commit
policy should be - both tuning decisions, not structural risks.

## 8. Disposition summary

| ID  | Item | Type | Target section(s) | Disposition |
|-----|------|------|-------------------|-------------|
| F-1 | Keychain subprocess cost / revision-probe cadence | Tuning | Arch (new), Chunk 1 design | Accept |
| F-2 | Probe failure policy taxonomy | Tuning | Arch §6.3, §16 | Accept; behavior TBD |
| F-3 | `profile_id` canonicalization under symlinks | Correctness | Arch (profile_id derivation), Chunk 1 tests | Accept |
| F-4 | Legacy incoherence window Chunk 3->5 | Docs | Arch §18 | Accept |
| F-5 | Wall-clock step vs expiry classification | Docs + decision | Arch §7.1 | Accept; re-anchor TBD |
| F-6 | Keychain duplicate-item ambiguity test | Test coverage | Arch §17.1 | Accept |
| F-7 | Bound logged revision-prefix length | Minor | Arch §15 | Accept |
| F-8 | Chunk 4 manifest/state atomic write + crash test | Test coverage | Chunk 4 design | Accept |

## 9. Next steps

1. Maintainer decisions on F-2 and F-5 open questions.
2. Apply F-1, F-2, F-3, F-4, F-5, F-7 edits to
   `../architecture/mcp-oauth-credential-store-architecture.md`.
3. Apply F-6 and F-8 to the relevant design/test-matrix documents.
4. Fold F-1 probe-cadence contract and F-3 canonicalization contract test into the Chunk 1
   implementation plan (currently stopped at Chunk 1, Task 6).
