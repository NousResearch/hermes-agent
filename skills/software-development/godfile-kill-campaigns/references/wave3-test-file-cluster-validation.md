# Wave-3 witness validation for TEST-file godfiles — MAPPING phase (tui_test s3-w3a, 2026-08-05)

Recipe for wave-3 MAPPING validation (stage 4 of 5×2×3 — validate every (test, line) claim in all
four catalogs against live source via AST, validate the canonical cluster set + required fixes).
Distinct from extraction-wave-3 (patch apply/py_compile/import/tests): here there is NO patch, the
deliverable is a verdict JSON, and there is NO pytest suite to run — verification is an ad-hoc
AST claims-validator (label it "ad-hoc verification, NOT suite green" in the receipt).

## Verdict schema (`godfile-test-extraction-wave3-validation-witness.v1`, top-level keys EXACT)
`schema, shard, witness, wave, file, region, live_file, live_file_identity, inputs_reviewed,
aggregate, canonical_cluster_set, required_fixes_validation, findings, notes, files_created,
recommendation`. Grab the shape from a landed sibling shard's w3a/w3b JSON, NOT from the sibling
witness of YOUR shard (blind-witness independence — only its existence + PASS/verdict-count fact
from the task brief may be used).

## Per-catalog claim counting — the four catalogs carry DIFFERENT claim shapes
- **w1a** (`w1-cluster-map/1.0`): `clusters[].tests` = names only, NO per-test lines → 85 name claims.
- **w1b**: `clusters[].test_entries` = `{name, line, span}` → name+line claims; every def-LINE
  claim must match live AST lineno EXACTLY (span ends use the next-def-minus-1 trailing-blank
  convention, +1..+2 off — benign, verify lines only).
- **w2b** (adjudication): `canonical_clusters[].tests` = names + cluster spans → 85 name claims.
- **w2a** (cross-check, `w2-cluster-map/1.0`): **has NO cluster membership enumeration** —
  coverage_verification + disagreements[] + required_fixes only. Its per-catalog claim count =
  test names referenced in `disagreements[]` (23 here, not 85). Do NOT fabricate 85 claims for a
  schema that doesn't enumerate; validate its "canonical 85, exactly once" coverage claim through
  the canonical-set checks instead and say so in the verdict. (This is the NINTH shape in the
  verdict-schema zoo — see SKILL.md pitfall.)
- Total here = 85+85+23+85 = 278; report per-catalog verified/contradicted in `aggregate`.

## Canonical-set checks (independent re-derivation from the w2b verdict)
- Region defs (85) == canonical set exactly: 0 missing, 0 extra; exactly-once via DEF-LINE
  disjointness (each def line owned by one cluster, 0 collisions).
- `member_outside_span = []`, `defline_collisions = []`.
- **Span bounding-box OVERLAPS ARE EXPECTED for interleaved test files** (30 pairwise here:
  c5 [7681,8354] contains c8/c9/c10 member lines, c6 [7772,9416] contains c7-c15 members).
  Record as an observation, NOT a failure; note that extraction MUST move individual test defs by
  AST span, never by bounding box (a w2 required fix to re-confirm).
- Seam anchors to verify: spill-in def end (6426→6520), spill-out def end (9709→9772), first def
  of the neighbor shard (9775), out-of-shard helper (`_session` @3567), first/last in-region defs.
- Reconfirm inventory by AST: 0 test classes / 0 async / 0 parametrize / 0 in-shard helpers.

## w2 witnesses disagree on cluster granularity → independent mechanism-location probe
w2a canonical = 14 (merged steer+redirect), w2b = 15 (split). Decisive probe, run BEFORE siding
with either: grep each proposed span for the mechanism that distinguishes them —
`_start_inflight_turn` refs: 0 in steer span 8531-8599, 4 in redirect span 8600-8718;
`agent.steer` dispatch present only in steer. Mechanism location decides granularity; record the
probe numbers in the verdict (they make the split self-evident to the parent).

## Required-fixes validation: verify every line claim YOURSELF (byte-count doctrine)
- Pull live text at every banner/comment site w2 cited (8528 steer banner, 8775-8780 running-
  guards comment, 9309-9315 + 9408-9414 interrupt banners) and confirm content belongs to the
  claimed family; confirm w2's live-end claims (8525/8772/9306/9405) match AST end_lineno exactly.
- **Count claims yourself**: w2a RF2 claimed "s2 complete_slash family = 10 tests @6398-6413";
  live has 7 defs @6300-6413 (window 6398-6413 holds 2). Fix directive stands, count was wrong →
  INPUT-CORRECTION finding. Same class as the byte-count-yourself rule.

## Cross-shard seam double-claims (the @9709 pattern)
A seam-crosser def (def in YOUR region, body into the neighbor's) is clustered by BOTH shards'
witnesses per seam rules — and each shard's w2 adjudicates its OWN ownership ("s3-owned; window
9417-9774" vs s4-w3a's "adjudicated s4 keeps it"). The two shard-level adjudications CONFLICT and
neither binds the other. The w3 witness must:
1. record it as a REQUIRED-ACTION finding with BOTH sides' evidence,
2. give natural-owner reasoning (def line inside my region + whole body in my extraction window =
   natural owner; the other shard trims),
3. defer the final single-ownership call to the merger ("whichever owner, the def must move
   exactly once; interlock against the neighbor's w2/w3 before merge").
Verify the neighbor's claim only via its LANDED w3 facts (s4-w3a F2 corroboration), never by
reading your own shard's sibling w3b.

## Positive control for line-claim checks
Inject a +1 line-shift into one claimed line in-memory and assert the checker flags it → proves
the line check is not vacuous. (Mutation-control twin of the blind-rereview-pass rule; a checker
that normalizes too greedily passes anything.)

## Findings taxonomy (severity labels that survive the merger)
`INPUT-RECONCILIATION` (w2 count/naming divergence, e.g. 14-vs-15 canonical), `INPUT-CORRECTION`
(wrong count/window inside a fix claim), `REQUIRED-ACTION` (cross-shard ownership), `OBSERVATION`
(benign span conventions, inventory reconfirms). Overall = PASS only when every catalog claim
verified + 0 contradictions; input-level defects live in findings, not in the aggregate verdict.

## Evidence loop for a claims-validator deliverable
The harness wants fresh passing verification evidence for the JSON/scripts. Pattern that clears
it: py_compile both scripts, then a temp-bootstrapped verifier — copy the verifier to an OS-safe
`tempfile` path under %TEMP% with `hermes-verify-` prefix, run it, save output to
`verify_s3_w3a_final.RESULT.txt` (shard-suffixed, next to the sibling's RESULT), KEEP script +
RESULT on disk, and make the re-run the LAST action of the turn (the verifier joining the changed
set re-triggers the flag; re-run after all files are written). State plainly: ad-hoc claims
validation, not a pytest suite.
