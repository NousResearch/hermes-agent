# Evidence Promotion-Only Mission Pattern (PROMOTION_ONLY)

Established during `CLOE_HOT_RELOAD_AND_PUBLISHER_IDEMPOTENCY_REPAIR_V1` promotion (2026-08-04):
revalidated and promoted the enriched candidate `7299e4f0…` against deployed repair SHA `949795a…`
in `leanframeworklab/openclaw-runtime` (PR #681 merge), hot-reload + idempotency verified on the
running container `c374050e74aa`.

## Trigger

Mission text authorizes ONLY: revalidate an already dry-run-validated enriched candidate and, if
unchanged, promote it via the canonical publisher. Prohibited: code changes, new PR, image rebuild,
deploy, container restart/recreate, new evidence generation, manual edits to manifests/snapshots/
receipts/ledgers, `lah-tools-runtime` changes, workflow/CI reruns. The mission spec provides all
routing metadata (repo, PR, merge SHA, container, paths) → Gate 0 router is replaced by the spec
(SAME exception as user-provided repo metadata).

## Pre-flight (before ANY mutation)

1. **session_search the prior mission** — the prior final report carries: expected candidate identity,
   expected promotions/demotions (and dimension breakdown), receipt_set_hash, established manifest
   polling interval (this system: 15 s; proof waited 18 s), and the exact next command block.
2. **Locate the code authority worktree** — the publisher MUST run from the worktree whose HEAD == the
   merged/deployed SHA. Check all candidates:
   `git rev-parse --short HEAD` in the deploy worktree, PR worktree, canonical checkout, workspace clone.
   Only the deploy worktree (e.g. `deploy-evidence-v1`) matches the merged SHA; the PR worktree still
   sits at the pre-merge head, the canonical checkout and workspace clone lag on other branches.
3. **Capture baseline (read-only):**
   - Container: `docker ps --filter id=<c>` + `docker inspect -f 'StartedAt={{.State.StartedAt}} RestartCount={{.RestartCount}}' <c>`
   - `graph-dir/current-manifest.json` + `previous-manifest.json`, `snapshots/` listing
   - Ledger: `ls <ledger-root>/receipts/ | wc -l` (evidence receipts only)
   - Sources file: `sha256sum` + `stat -c '%y'` (must predate the mission)
   - Live served state: `GET /self-audit/query/summary` with header `x-admin-api-key: <key>` on
     `127.0.0.1:4000` → `summary.graph_hash` + `summary.source_sha`. Admin key: `docker exec <c> sh -c 'echo "$ADMIN_API_KEY"'`.

## CLI facts (cloe-evidence-publisher.mjs)

- **No `--help` flag.** Unknown flags are ignored; a run without `--graph-dir` returns
  `{"ok":false,"blocked":"MISSING_GRAPH_DIR"}`. Inspect the flag set by grepping `parseArgs` in the
  source instead of probing flags blindly.
- Flags: `--dry-run | --promote`, `--rollback <graph_hash>`, `--graph-dir`, `--ledger-root`,
  `--source-sha`, `--deployed-sha`, `--environment`, `--sources-file`.
- **Dry-run is repeatable and side-effect-free**: temp staging dir, final message
  `DRY_RUN: no canonical mutation performed`. Safe to run twice (the idempotency check IS a second dry-run).
- Candidate identity appears as `graph_hash` in the JSON output; the staging dir name carries the
  candidate prefix (e.g. `cloe-snapshot-7299e4f08395-*`).
- Promotion (`--promote`, promoted:true): rewrites current/previous manifests, creates
  `snapshots/<hash>/`, and creates exactly ONE publication receipt under
  `<graph-dir>/../publication-receipts/publication-<hash>-<epoch-ms>.json`. **NO new evidence
  receipts** — `ledger/receipts/` count is unchanged (promotions reuse existing evidence receipts).
- Publisher internals (verified from source 2026-08-05): `runPublication` REBUILDS the
  candidate from the ledger at publication time via `buildCandidateSnapshot` (stagingRoot =
  dirname(graphDir)); the candidate snapshot dir `cloe-snapshot-<hash>-<rand>/` appears NEXT
  TO graphDir and is renamed into `graphDir/snapshots/<hash>/` by
  `publishSnapshotToGraphDir`. Chain-integrity (`validateManifestChain`) gates BEFORE any
  mutation; the ALREADY_CURRENT no-op check runs AFTER loadability validation. The pilot's
  own build + dry-run is only a PREVIEW — the publisher's rebuild from the ledger is the
  authoritative candidate, so the ledger must be complete (all intended receipts appended)
  before `--promote`, and `--ledger-root`/`--source-sha`/`--deployed-sha` must match what
  the pilot used.

## Chain-integrity validator — pure function, NO CLI

`src/self-audit/evidence/manifest-chain-integrity.mjs` exports `validateManifestChain(graphDir)` only.
Invoke from the deploy worktree root:

```bash
node -e "import('./src/self-audit/evidence/manifest-chain-integrity.mjs').then(m => console.log(JSON.stringify(m.validateManifestChain('<graph-dir>'), null, 2)))"
```

Expect `{ ok:true, schema:"cloe_manifest_chain_integrity_v1", violations:[] }` with current ≠ previous,
previous snapshot present, valid source SHA binding.

## Gate checklist BEFORE promotion (every line must match the mission spec)

- candidate identity (dry-run `graph_hash`) == expected, exact 64-hex
- promotions == expected count AND dimension breakdown (e.g. runtime_reachable ×5 + shadow_verified ×5)
- demotions == 0 (no verified capability downgraded)
- `source_sha` == `deployed_sha` == repair SHA
- current manifest unchanged by dry-run (dry-run `.previous.graph_hash` == on-disk current)
- chain integrity ok:true / violations []
- sources-file sha256 + mtime unchanged
- container ID + StartedAt unchanged
- ANY mismatch → STOP, report the exact gate, no promotion.

## Promotion → verify sequence

1. Run the EXACT `--promote` command from the mission spec; capture full JSON.
2. Verify disk: current-manifest (new hash), previous-manifest (old current, archived_at set),
   `snapshots/<new-hash>/` exists, evidence-receipt count unchanged, 1 publication receipt created.
3. Wait strictly > the established polling interval (15 s; this system proved at 18 s). NO restart.
4. Query the live endpoint → served `graph_hash` must equal the promoted hash (hot reload).
5. Verify container ID + StartedAt IDENTICAL (proves no restart occurred).
6. Run chain integrity again (current = promoted hash, previous = old current).
7. Duplicate dry-run → message contains `ALREADY_CURRENT`, `promoted:false`, `diff.changed:0`,
   manifests byte-identical (mtime unchanged).

## Ledger semantics for the report (two separate stores)

- `ledger/receipts/` = EVIDENCE receipts (append-only; grow only on evidence-generation missions,
  promotions reuse existing evidence receipts → count constant). **Exception — live-operator-gate
  missions with an explicit Gate 4 ("import certified PASS live receipts") DO grow the ledger via
  `collector-live-import.mjs`: see `references/live-gate-receipt-import-pattern.md`.**
- `graph-dir/../publication-receipts/` = PUBLICATION receipts (1 per promotion, timestamped filename).
- Report "receipt count created" = publication receipts created; never conflate the two stores.
  Historical publication receipts accumulate here (multiple entries per graph hash are normal).

## Preserved unresolved states

Extract `UNKNOWN_OR_NOT_PROVEN` / `NOT_VERIFIED` entries from the dry-run JSON (walk the diff/graph
for objects with `state` in those values) and list them in the final report. Never convert them
without new evidence — the mission explicitly forbids generating new evidence.

## Stop conditions

- BEFORE promotion: any gate mismatch → stop, precise blocked verdict naming the gate.
- AFTER promotion: container identity/StartedAt changed, chain integrity fails, or duplicate
  dry-run is not ALREADY_CURRENT → stop, report. No improvised remediation, no rollback unless the
  promotion itself corrupts chain integrity.

## Final verdict strings

Success: `CLOE_LIVING_CAPABILITY_EVIDENCE_SYSTEM_V1_ENRICHED_RUNTIME_SHADOW_EVIDENCE_PROMOTED_AND_HOT_RELOAD_VERIFIED`
(only when every gate passes). Otherwise a precise blocked/failed verdict naming the exact gate.
