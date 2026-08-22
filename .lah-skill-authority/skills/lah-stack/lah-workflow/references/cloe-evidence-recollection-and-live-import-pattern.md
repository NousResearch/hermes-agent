# CLOE Evidence Recollection + Live-Gate Import Pattern

Established during CLOE Lots 1-3 (2026-08-04/05), CORRECTED during Lot 3 resume
(2026-08-05): several documented steps proved WRONG in practice (pilot rerun
appends; ledger layout duality; pure-builder live import). Read the corrections
below before any recollection/promotion.

## Why recollection is needed (SHA-rebind)

Receipts bind to `source_sha`. When the deployed SHA moves (new PR merged +
deployed), old VERIFIED receipts are rejected (`SOURCE_SHA_MISMATCH`) →
mechanical demotions. Fix: re-run the evidence pilot from a CLEAN worktree at
the deployed SHA so new receipts carry the deployed source_sha.

**All previously-certified live gate receipts are ALSO rejected on SHA-rebind**
(not just the current lot's): every prior lot's `live_verified` dim demotes to
UNKNOWN. The plan MUST re-import EVERY prior lot's certified live gate verdict
at the new SHA (Lot 3 resume needed tracking AND provider-statistics-read
re-imports), otherwise the publisher dry-run shows demotions on OTHER
capabilities. Check the full dry-run diff, not just the current lot's
capability.

## Steps

1. **Clean worktree at deployed SHA** (collector enforces `allowDirty:false`):
   ```bash
   cd <canonical-repo> && git worktree add /tmp/<mission>-evidence-collect <DEPLOYED_SHA>
   cp -al <known-good-worktree>/lah-openclaw-mvp/node_modules /tmp/<mission>-evidence-collect/lah-openclaw-mvp/node_modules
   ```
   `cp -al` hardlinks node_modules — same package.json → no npm install, fast,
   DEPS_OK. A deps-less worktree produces FALSE `TEST_ASSERTIONS_FAILED`
   receipts (`ERR_MODULE_NOT_FOUND` for dotenv/zod) — root cause of the Lot 1
   business-tested demotions.

   **Worktree root == repo root**: `git worktree add /tmp/<x> <SHA>` at the
   openclaw-runtime TOP LEVEL makes `lah-openclaw-mvp` a SUBDIR. `git status`
   therefore reports untracked SIBLINGS (snapshot dirs, out files) as dirt.
   Clean the whole worktree root between runs, not just the app dir.

2. **Run the pilot** (runtime+shadow collectors need the admin key — pass via
   process env from a wrapper (docker exec container `echo "$ADMIN_API_KEY"` or
   read .env with split-string key name), never print it, never on argv):
   ```bash
   node tools/cloe-evidence-pilot.mjs \
     --app-root <wt>/lah-openclaw-mvp \
     --ledger-root /home/deploy/cloe-self-audit-evidence/living-evidence-v1 \
     --graph-staging /tmp/<mission>-graph-staging \
     --deployed-sha <DEPLOYED_SHA> \
     --deployed-graph-hash <FULL-SERVED-GRAPH-HASH>
   ```
   **MANDATORY flags — the pilot's hardcoded defaults are STALE**: without
   `--deployed-sha` the pilot binds runtime/shadow receipts (and the candidate
   graph-meta `deployed_sha`) to the stale default (observed: 949795a…), and
   `--deployed-graph-hash` defaults to an old graph. Always pass both; derive
   the served graph hash from `<graph-dir>/current-manifest.json` (FULL
   64-hex, never truncated).
   **--graph-staging must be OUTSIDE the repo**: the pilot writes candidate
   snapshot dirs (`cloe-snapshot-<hash>-<rand>/`) as SIBLINGS of the staging
   dir; inside the repo they dirty the worktree for the next run. Use
   `/tmp/...` or the evidence root's graph-staging.
   Expected: ~71 receipts appended, promotions > 0, demotions = 0.

3. **Supersession edges** — after recollection, the SAME cap/dim can have two
   active receipts (old NOT_VERIFIED + new VERIFIED) → CONFLICTED. Collectors
   NEVER emit `evidence.supersedes_receipt_ids` edges (evidence-system gap, S5
   test in `cloe-evidence-supersession.test.mjs` fail-closed). Fix: import the
   real new VERIFIED receipts + add explicit edges over the old receipts
   (mirrors `collector-live-import.mjs`). Never fabricate evidence — copy real
   receipts, only add edges.

4. **Live gate import** — `importLiveGateVerdict({ verdict, intendedSourceSha,
   ledgerRoot, now })` from `collector-live-import.mjs`. Verdict schema
   (`cloe_live_gate_verdict_v1`) requires: `schema`, `gate_id`, `test_case`,
   `capability_ids` (non-empty array), `raw_receipt_digest` (non-empty string),
   `expected_assertions`, `observed_assertions`, `pass` (boolean), `source_sha`,
   `deployed_sha`, `graph_hash`, `runtime_identity`, `certification_authority`
   (must be in DEFAULT_ALLOWED_AUTHORITIES — OPERATOR works), `certified_at`
   (ISO-8601). One receipt per capability_id, dimension `live_verified`.

   **importLiveGateVerdict is a PURE BUILDER — it does NOT persist.** The
   caller must append each returned receipt (`appendReceipt`) and rebuild
   indexes (`rebuildIndexes`). A script that only prints the result reports
   `imported: 1 / VERIFIED` and writes NOTHING (observed in Lot 3 resume — the
   ledger stayed at 71 files while the script claimed success).
   **Idempotency**: pass `now: new Date(gate.certified_at)` so `observed_at`
   is deterministic → repeated execution produces the SAME receipt (run 1:
   appended=1; run 2: duplicate=1). Without it, every run appends a fresh
   receipt.
   **raw_receipt_digest must be REAL** — never a synthetic placeholder
   (observed: a5f3c2b1…, b0b111c2…). Use the sha256 of the proof artifact or
   the canonical raw verdict object. For deterministic pure-function proofs:
   re-run the proof script (read-only) and hash
   `JSON.stringify(result-without-digest)` — the first 24 hex chars must match
   the proof script's own truncated digest (validation). For provider proofs
   (non-deterministic): hash the saved proof script artifact file.
   **graph_hash must be the FULL 64-hex served hash** from
   current-manifest.json, not the truncated prefix.

5. **Rebuild candidate — DO NOT rerun the pilot.** The pilot appends a FULL
   new batch every run (receipts carry fresh `observed_at` → new hashes; the
   earlier doc claim "rerun the pilot, no new receipts appended" is WRONG for
   this pilot). Observed: ledger 72 → 143 on the second run (71 duplicates).
   The PUBLISHER rebuilds the candidate from the ledger itself (no appends),
   so run the pilot ONCE after all imports, then go straight to the publisher
   dry-run/promote. If a second pilot run is unavoidable, expect +N receipts
   (same results, different hashes — proof-state still fine, ledger just
   grows) and check for conflicts.

6. **Promote** — `bin/cloe-evidence-publisher.mjs --promote --graph-dir
   <graph> --ledger-root <ledger>` → expect PROMOTED (previous archived), then
   re-invoke → ALREADY_CURRENT (idempotent no-op). Verify chain integrity
   (`manifest-chain-integrity.mjs` → ok:true, violations:[]) and publication
   receipt count +1. Hot reload: container polls current-manifest every 15s;
   container ID/StartedAt/RestartCount unchanged proves no restart.
   Publisher dry-run FIRST: diff vs the SERVED graph (previous = current
   manifest identity). REQUIRE: source_sha == deployed_sha == deployed SHA,
   zero demotions, zero rejected receipts, zero invalid supersession edges,
   prior certified capabilities preserved. ANY demotion (even a mechanical
   live_verified SHA-rebind on ANOTHER lot's capability) → STOP, surface the
   exact gate, no promotion. Re-importing another capability's live gate is
   outside the authorized import script's scope — operator decision required.

## Pitfalls

- **Ledger layout duality — TWO stores**: the CURRENT evidence-ledger.mjs
  (deploy-cap era, 131f3506+) writes `<ledgerRoot>/receipts` +
  `<ledgerRoot>/indexes` + `<ledgerRoot>/ledger-manifest.json`. Older runs
  (pre-deploy-cap) wrote `<ledgerRoot>/ledger/receipts` (legacy). The legacy
  store can show 661 files while the canonical store has 71 — READ THE CODE
  (`appendReceipt`/`loadAllReceipts`) to know which store the passed
  `--ledger-root` maps to before counting. `ledger-manifest.json
  receipt_count` is the canonical count.
- **Pilot stdout counts refer to ITS ledger root**: match the reported
  `ledger.receipt_count` to the store you counted, or you will misread 661
  (legacy) vs 71 (canonical).
- **Dirty worktree → collectors SKIPPED** (local/wiring/tests REPOSITORY_DIRTY)
  → business caps demote. Always recollect from a fresh clean worktree. NOTE:
  the runtime/shadow probes recreate `data/` files DURING the run (after the
  allowDirty check passes), so the worktree is dirty again right after — clean
  it before any NEXT pilot run (move untracked files aside; restore tracked
  ones with `git checkout --`).
- **Deployed-SHA ambiguity after route-fix PRs**: verify runtime GIT_COMMIT via
  `docker inspect` before binding receipts; operator may deploy manually.
- **Bounded live proofs with provider**: use the canonical clean campaign
  (8308460), single-day window; expect rows=0 honest empty state; record
  live_sent=false, writes_performed=false.
- **Live import receipts must be idempotent** (now=certified_at) and carry a
  real digest + full graph hash — operator review gate (2026-08-05) rejects
  synthetic digests and truncated hashes.
- **Ledger counts**: recollection adds ~71; live import +1 per capability.
  Append-only, verify_ok, 0 rejected. Duplicate evidence batches from repeated
  pilot runs are semantically harmless (same results, no conflicts) but inflate
  the ledger — avoid by running the pilot exactly once.
