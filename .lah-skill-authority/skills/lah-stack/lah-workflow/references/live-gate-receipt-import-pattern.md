# Live Gate Receipt Import (PROMOTION_ONLY with live-operator-gate execution)

Established during `CLOE_LIVING_CAPABILITY_EVIDENCE_SYSTEM_V1_LIVE_OPERATOR_GATE_EXECUTION_V1` (2026-08-04):
promoted `4450f09d` over `7299e4f0` (2 promotions tested / 0 demotion), hot-reload verified
(container `c374050e74aa`, StartedAt unchanged), then imported 6 live receipts (3 PASS gate
verdicts × 2 capabilities), rebuilt the final candidate `f24eed48` — dry-run only, NOT promoted.

## When this applies
A PROMOTION_ONLY mission whose spec includes an explicit **Gate 4 — LIVE RECEIPTS** step
("Importer uniquement les receipts structurés certifiés PASS"). The BASE promotion-only
pattern adds NO evidence receipts; the live-operator-gate variant DOES — via the canonical
live-import collector, and ONLY for gates actually executed and passed in this run.
No fake receipts: every verdict must be grounded in real observed tool output.

## Import path (canonical — no new code in the repo)
1. Construct one `cloe_live_gate_verdict_v1` per PASSED live gate (e.g. hot-reload gate,
   canonical audit gate, comparison gate). Required fields (validated by
   `validateLiveGateVerdict`): `gate_id`, `test_case`, `expected_assertions[]`,
   `observed_assertions[]`, `pass` (boolean), `source_sha`, `deployed_sha`, `graph_hash`,
   `runtime_identity` (e.g. `container:<id>@sha256:<digest>`), `certification_authority`
   (`'OPERATOR'` — the only allowed authority), `certified_at` (ISO-8601),
   `raw_receipt_digest` (sha256 of the raw gate record), `capability_ids[]` (non-empty).
2. `importLiveGateVerdict({ verdict, allowedAuthorities: ['OPERATOR'], intendedSourceSha,
   ledgerRoot })` → raw receipt inputs (ONE per capability_id, dimension `live_verified`,
   result `VERIFIED` for `pass:true`). Passing `intendedSourceSha` rejects verdicts bound to
   another SHA (`SOURCE_SHA_MISMATCH`). The importer NEVER self-decides a pass — it only maps
   an explicitly certified verdict; `pass:false` produces NOT_VERIFIED negative evidence.
3. `appendReceipt(ledgerRoot, input)` — canonicalizes (content-addressed sha256 receipt_id),
   atomic tmp+fsync+rename into `receipts/<hash>.json`, auto-rebuilds `indexes/` +
   `ledger-manifest.json`, idempotent on duplicates.
4. Verify: `verifyLedger()` ok (corrupt 0, drift 0), count 176 → 182 (6 added),
   `listReceipts(ledgerRoot, { dimension: 'live_verified' })` shows the new VERIFIED receipts.

Run the import from a `/tmp` script with absolute `file://` imports of the repo modules
(relative imports inside the modules resolve against the module file). Keeps the worktree
clean on READ_ONLY missions.

## Policy requirements for live_verified acceptance
`evidence-policy.v1.json`: `live_verified` is high-trust → environment `'production'`,
max_age_hours 168; allowed collector `'collector-live-import'`; certification authorities
`['OPERATOR']` (policy.live); `min_observations` 1 → ONE PASS receipt per capability = VERIFIED.

## Effect on the candidate
After import, the NEXT publisher dry-run builds a NEW candidate (live_verified now VERIFIED on
the imported capabilities) — that is the FINAL candidate. In the stop-before-final-promotion
variant: report it, do NOT promote.

## proof_state_hash extraction (pitfall)
- `proof-state.json` does NOT carry `proof_state_hash` (top keys: schema, policy_version,
  source_sha, deployed_sha, environment, capabilities; entries keyed `capability_id` +
  `dimensions`). Querying it for proof_state_hash returns undefined.
- Per-node `proof_state_hash` is computed by `applyProofStateToGraph` (sha256 over
  {evidence_level, dimensions state + accepted_receipt_ids, conflicts}) and stored on the
  GRAPH NODES → staged candidate `capability-graph.json` → `node.proof_state_hash`.
  A mission's expected "Proof-state hash" target must be checked against the candidate's
  graph nodes, not proof-state.json.
- Publisher dry-run JSON fields: `graph_hash`, `receipt_set_hash`, `previous`
  (= on-disk current — proves the dry-run is side-effect-free), `diff`
  {changed, promotions[], demotions[]}, `validation` {ledger, receipts_used, ...},
  `stagingDir`, `message` — NO top-level proof_state_hash.

## Container identity verification quirk (this VPS's docker)
`docker inspect -f '{{.State.RestartCount}}'` fails with "template parsing error: map has no
entry for key RestartCount" — RestartCount is a TOP-LEVEL field:
`docker inspect -f 'StartedAt={{.State.StartedAt}} RestartCount={{.RestartCount}}' <container>`

## Operator-denied verification command
If the operator denies even a read-only verification command mid-mission: do NOT retry, do
NOT rephrase, do NOT achieve the same outcome via another command. Finish the report on
already-captured facts, disclose the exact missing datum, and hand the operator the file
path to verify manually (e.g. `<stagingDir>/capability-graph.json` → node.proof_state_hash).
