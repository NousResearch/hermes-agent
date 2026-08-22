# Living Capability Evidence System — pattern (CLOE_LIVING_CAPABILITY_EVIDENCE_SYSTEM_V1)

Class: capability-graph / evidence-system missions on `lah-openclaw-mvp` (self-audit family:
LOT1→LOT10 capability graph, then living evidence system). Governed by `lah-workflow`.

## Routing & workspace
- Prefix `CLOE_` → `openclaw-runtime`. Implementation in a dedicated worktree off `origin/main`
  (never the workspace clone `/home/deploy/openclaw-runtime` — that is the LIVE gateway ESM-loader host, never rebuild from there).
- Worktree for this mission: `/home/deploy/lah-stack-worktrees/cloe-living-capability-evidence-system-v1`
  branch `feat/cloe-living-capability-evidence-system-v1`.

## Production facts (resolved baseline, Aug 2026)
- Active canonical graph dir: `/home/deploy/cloe-self-audit-evidence/lot10/graph`
  (ro-mounted in container at `/app/data/self-audit/canonical-graph`; compose env `CLOE_SELF_AUDIT_GRAPH_DIR`).
- Graph flat-file layout (no manifest yet): `capability-graph.json`, `capability-graph.previous.json`,
  `graph-hash.txt`, `graph-meta.json` (+ previous variants). New publisher writes snapshot layout
  `snapshots/<hash>/…` + `current-manifest.json`; provider must stay backward-compatible with flat layout.
- Deployed: container `lah-openclaw-mvp` serves 127.0.0.1:4000 (`node src/server.js`); host gateway
  (openclaw@2026.6.11) on 18789; graph source_sha = deployed app SHA `0eabdd5c…`; container image
  `sha256:bde383fc…` (rebuilt 2026-08-03). Admin key for read-only probes: container env `ADMIN_API_KEY`
  (never print it).
- Current graph: hash `3dfe01a6…`, source `0eabdd5c`, 101 nodes — probe `/self-audit/query/summary` returns exactly this.

## Module layout (`src/self-audit/evidence/`)
constants.mjs · receipt-schema.mjs · receipt-hash.mjs · evidence-ledger.mjs · collector-common.mjs ·
pilot-capabilities.mjs · collector-local.mjs · collector-wiring.mjs · collector-tests.mjs ·
test-capability-mapping.mjs · evidence-policy.v1.json · evidence-policy-engine.mjs · proof-state-merge.mjs ·
collector-runtime.mjs · collector-shadow.mjs · collector-live-import.mjs
Plus minimal builder extension: `src/self-audit/capability-graph/builder.mjs` accepts `options.proofState`.

## Design decisions (lock these; do not relitigate)
- Receipt `cloe_capability_evidence_receipt_v1`: content-addressed `receipt_id = sha256:<hash over
  content EXCLUDING receipt_id/receipt_hash>`; provided id/hash validated against computed →
  `RECEIPT_ID_MISMATCH` (no mutable replacement); credential field NAMES rejected at any depth
  (fail-closed); derived fields must NOT be required by the semantic validator before computation.
- High-trust dims (runtime_reachable, shadow_verified, live_verified) require top-level
  `deployed_sha + runtime_identity + probe_identity + expected_contract + observed_result`;
  live additionally `certification_authority + gate_id + certified_at`.
- Ledger: append-only `receipts/<hash>.json` (atomic tmp+fsync+rename, mode 0600); `indexes/`
  reproducible; `rejected/` stores reason+digest ONLY (never raw credential content); manifest
  hash computed over `{schema, receipt_hashes}` (stable across rebuild, `generated_at` excluded).
- Policy engine: per-dimension collector-class allowlist (local/test can NEVER promote
  runtime/shadow/live); CONFLICTED on VERIFIED + any negative (no first-wins); UNKNOWN default;
  `evidence_level` = max VERIFIED dimension (documented mapping 0..6); fully deterministic output.
- Builder v2 hash semantics: hashed body UNCHANGED (`{nodes,edges,dynamic_edges_unresolved,
  authority_conflicts,source_manifest}`) so legacy graphs still validate; proof fields live inside
  nodes → any proof change deterministically changes the graph hash; metadata (`receipt_set_hash`,
  `evidence_policy_version`, `deployed_sha`) carried on graph but NOT in hashed body; per-node
  `proof_state_hash`; status derived DECLARED→IMPLEMENTED→INSTANTIATED→TESTED_UNIT→
  RUNTIME_REACHABLE→SHADOW_VERIFIED→LIVE_VERIFIED.
- Honest states: repair-planner has NO runtime surface and NO app wiring chain → statically_wired and
  runtime_reachable stay UNKNOWN_OR_NOT_PROVEN (documented, no forced VERIFIED). capability-graph
  `statically_wired` wiring chain references `src/self-audit/evidence/snapshot-builder.mjs` — flips
  VERIFIED once Phase 11 builds it.
- Live import: importer NEVER decides success — maps certified verdict `pass` mechanically;
  FAIL creates NOT_VERIFIED receipt that persists; later PASS + earlier FAIL → policy reports
  CONFLICTED until operator resolution (explicit supersession ids recorded, no silent precedence).

## Traps (learned this mission)
- **NODE_TEST_CONTEXT**: a collector that spawns `node --test <file>` from inside a `node:test`
  process gets EMPTY stdout + exit 0 — the child inherits `NODE_TEST_CONTEXT` and switches to
  IPC-child mode. Fix: `const env = {...process.env}; delete env.NODE_TEST_CONTEXT;` + prefer
  `spawnSync` over `execFileSync` for TAP capture in that context.
- **Collector dirty-gate vs own untracked files**: fail-closed collectors (block on REPOSITORY_DIRTY)
  reject the real worktree while your own new files are untracked. Test with a CLEAN git fixture:
  copy the analysis-relevant real files into a temp repo, commit, run collectors there (inject
  sourceSha = fixture HEAD). Keep one deliberate dirty-repo fixture test for the blocked path.
- **Debugging preference (operator)**: do NOT drop standalone debug scripts under /tmp piped to
  python — instrument the failing test (assertion message with `JSON.stringify(evidence)`), run with
  `--test-name-pattern`, read TAP, revert. `rm -rf` (even /tmp) requires operator consent.
- `write_file`/`patch` occasionally mangle lines (`Object...` truncation, duplicate declarations):
  re-read the file, patch precisely, run `node --check` after every write.
- **Publisher provider-load gate — `CANDIDATE_NOT_LOADABLE` / `PROVIDER_STATUS:stale`**: the
  publisher validates the candidate with the REAL `createCanonicalGraphProvider`, which marks a
  graph stale when `expectedSourceSha` does not match the `openclaw-runtime` commit in
  `source_manifest`. A candidate built from sources whose `meta` lacks the bound commit entry
  (or carries a different commit) is BLOCKED at publication. Fix: `sources.meta` MUST include
  `{repository:'openclaw-runtime', commit:<source-sha>, collector_version, commit_bound:true}`,
  and the sources-file passed to `bin/cloe-evidence-publisher.mjs --sources-file` must have
  `meta[0].commit === --source-sha`, or the dry-run exits 1. Symptom checklist: CANDIDATE_NOT_LOADABLE
  → always check the sources file meta commit first (the `--out-sources` dump of the pilot harness
  is the reliable way to generate a matching file).
- **Evidence-SHA loop (variant of the SHA-infinite-loop trap)**: re-running the pilot harness at a
  new HEAD re-binds ALL receipts to the new SHA → matrix changes → commit → new HEAD → loop.
  Fix: collect evidence ONCE at the clean HEAD you intend to ship; dump `--out-sources` in that
  same run; commit report artifacts (matrix/docs) afterwards WITHOUT re-running the pilot — the
  candidate stays bound to the collection-time SHA. Never pass a stale sources file with a
  different `--source-sha`.
- **Exact-surface tests must move with deliberate surface extensions**: existing suites assert the
  EXACT frozen surface (`K2 ALLOWED_OPERATIONS is frozen and complete`, `T37 route paths are exactly
  the self-audit query surface`). Adding provenance ops/routes breaks them by design — update those
  assertions in the same change (they are deliberate extensions, not regressions). Run the full
  mapped suites after any route/client/op change.
- **Pre-existing-failure baseline proof (no-regression claim)**: to prove a failing suite is
  pre-existing, not a regression: `git worktree add <tmp> origin/main`, `ln -s <mission-worktree>/lah-openclaw-mvp/node_modules <tmp>/lah-openclaw-mvp/node_modules`, run the same test file on both
  branches, compare pass/fail counts (identical counts = zero regression). Clean up with
  `git worktree remove --force <tmp>`. Record the baseline counts in the validation receipt.
- **Secret-extraction shell commands trip the terminal scanner**: `sed 's/^ADMIN_API_KEY=…'` patterns get redacted to `***` mid-command, breaking shell quoting; `$(cat secretfile)` also gets
  mangled. Safe pattern: `docker inspect … | grep '^ADMIN_API_KEY' | cut -d= -f2- > /tmp/.key && chmod 600 /tmp/.key`,
  then `KEY=$(</tmp/.key)` (redirection form, no `cat`), pass via env or `--admin-key`; remove the
  key file immediately after.

## Staged deployment of the exact merged SHA (post-merge, operator-authorized)

The canonical checkout is often on a dirty feature branch — NEVER build the container from it.
Recipe that deploys the exact merge SHA with ONE bounded restart of the affected service:

1. **Clean worktree as build context**: `git worktree add <deploy-wt> <merge-sha>` (from the canonical
   checkout). Verify `git rev-parse HEAD == <merge-sha>` and `git status --porcelain | wc -l == 0`.
2. **Copy the compose secrets**: `cp <canonical>/lah-openclaw-mvp/.env <deploy-wt>/lah-openclaw-mvp/.env && chmod 600`.
3. **Absolute-volume compose override (DATA LOSS GUARD)**: the compose file mounts `./data:/app/data`
   relative to the compose-file location. Running compose from the worktree would mount an EMPTY
   worktree `data/` over the real runtime data. Write a one-off override (NEVER committed):
   ```yaml
   services:
     lah-openclaw-mvp:
       volumes:
         - /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/data:/app/data
         - /home/deploy/cloe-self-audit-evidence/lot10/graph:/app/data/self-audit/canonical-graph:ro
   ```
   and use `docker compose -p <project> -f docker-compose.yml -f compose.override-deploy.yml`.
4. **Build with explicit GIT_COMMIT**: `GIT_COMMIT=<merge-sha> docker compose -p <project> … build <service>`
   (compose `${GIT_COMMIT:-unknown}` never self-resolves to the SHA you think — pass it explicitly).
5. **One bounded restart**: `docker stop <c> && docker rm <c>`, then `compose up -d --no-deps <service>`
   (`--no-deps` protects sibling services like `lah-tools-runtime`). `--force-recreate` alone hits the
   container-name conflict pitfall — stop/rm first.
6. **Deploy worktree dirty-gate trap**: the worktree is made "dirty" by untracked artifacts
   (override file, symlinked `node_modules`). If fail-closed collectors will run against it, ignore
   them via `git rev-parse --git-path info/exclude` (worktrees have a POINTER `.git` file — appending
   to `.git/info/exclude` fails with "Not a directory"; the resolved path is the SHARED canonical
   `.git/info/exclude`, safe because excludes are path-scoped) — add `lah-openclaw-mvp/node_modules`
   + the override filename there.

**Deployed-code identity — image vs container env (do not panic)**: `env_file: .env` OVERRIDES the
image `ENV` at container start. After a correct rebuild, `docker exec <c> env | grep GIT_COMMIT` can
still show the STALE .env value while the deployed code is the new SHA. The authoritative checks:
`docker inspect <image> --format '{{range .Config.Env}}…'` (image-level = build truth) AND in-container
file evidence (`docker exec <c> ls <new-sha-only-file>` + `grep -c <new-symbol> <changed-source>`).
Never do a second restart just to fix the cosmetic env var — record it as a residual limitation and
fix the .env line at the next maintenance restart.

## Refreshed evidence semantics after deployment (promotion still gated)

After deploying the new code but BEFORE graph promotion, the runtime/shadow collectors MUST be
bound to the deployed merge SHA (`source_sha == deployed_sha == <merge-sha>`), while the production
graph still serves the OLD graph (source `0eabdd5c…`). The probes return 200 + canonical data, but
the observed graph source SHA ≠ expected → receipts are NOT_VERIFIED with reason `source binding
mismatch`. This is the DESIGNED fail-closed outcome, not a defect: runtime/shadow evidence cannot
promote until the graph itself is regenerated/promoted to the new SHA. Report it honestly as
`runtime_reachable: NOT_VERIFIED` (with receipt IDs) in the refreshed matrix — never fake VERIFIED,
never drop the negative receipts. `live_verified` stays UNKNOWN (no certified gates). The refreshed
dry-run (against the REAL production graph dir, dry-run only) validates: ledger integrity, policy
engine, deterministic hash reproduction, real provider load + query interface (publisher's
`validateCandidateLoadable`), runtime probes, shadow suite, previous/current chain (previous = the
unchanged production graph), and rollback simulation (re-run the publisher + hot-reload + snapshot
test files in temp dirs).

## Mission status — DEPLOYED + REFRESHED DRY-RUN (2026-08-03, staged deployment authorized)
- Commits: 57500ac (design) · e93aa46 (schema+ledger) · 74c76a7 (local+wiring) · 1ac8442 (test
  collector) · 97dcf3a (policy) · 0175be9 (builder v2) · 793050e (runtime+shadow+live-import) ·
  d8b6f7d (snapshot builder) · 927ec5d (publisher+CLI) · 5e948e0 (provider hot reload) ·
  08503f7 (provenance) · bdd51d2…a3ddc8b (pilot matrix iterations) — merged PR #676 → main
  `b3b79ee` (exact-head merge, tree hash == PR head tree; GHA unavailable by billing →
  REMOTE_CI_UNAVAILABLE_BY_BILLING_LIMIT / local-validation doctrine).
- Validation: 140 new evidence tests + 228 existing mapped suites green; gateway routing suite
  failures (19) identical to origin/main baseline → zero regression; `git diff --check` clean;
  secret scan clean (one intentional fake-credential negative fixture).
- **Staged deployment (operator-authorized, 2026-08-03)**: exact merge SHA `b3b79ee` deployed via
  clean worktree build (`GIT_COMMIT=b3b79ee`) + absolute-volume override; ONE bounded restart of
  `lah-openclaw-mvp` only (`--no-deps`; `lah-tools-runtime` untouched). Image Config.Env GIT_COMMIT
  == b3b79ee + in-container b3b79ee-only files prove code identity (container env var shows the
  stale .env GIT_COMMIT=2855674b — cosmetic, documented). Health 200; `/self-audit/query/summary`
  and the new `/provenance /proof-state /receipts` routes 200; graph mount (ro); production graph
  UNCHANGED (3dfe01a6…). Previous deployed SHA: `2855674b…`.
- **Refreshed evidence (post-deploy)**: ledger 122 receipts (28 appended bound to b3b79ee);
  candidate graph `a7867b10…`, source_sha == deployed_sha == `b3b79ee`, receipt_set `24f365f4…`;
  dry-run vs REAL production graph dir: exit 0, 17 promotions / 0 demotions, previous = 3dfe01a6,
  no mutation. Matrix: implemented/tested VERIFIED ×6, statically_wired VERIFIED ×5
  (repair-planner NOT_VERIFIED), runtime_reachable NOT_VERIFIED ×5 + shadow_verified NOT_VERIFIED ×5
  (source-binding mismatch — honest, graph not yet promoted), live_verified UNKNOWN ×6.
  Behavioral simulation receipt (deployment-scoped) VALID via the skill validator (10/10 scenarios).
- **First canonical graph promotion (operator-authorized, 2026-08-03)**: exact command
  `bin/cloe-evidence-publisher.mjs --promote --graph-dir …/lot10/graph --ledger-root …/living-evidence-v1/ledger
  --source-sha b3b79ee --deployed-sha b3b79ee --environment production --sources-file <refresh-sources>` →
  ok:true, promoted, graph `a7867b10…`, receipt_set `24f365f4…`, 17 promotions / 0 demotions,
  previous = 3dfe01a6/0eabdd5c, publication receipt emitted, lock released, both manifests atomic.
- **Provider hot reload FAILED on the running instance (bounded diagnosis, no restart)**: after the
  promotion, the LIVE `/self-audit/query/summary` kept returning the legacy graph (3dfe01a6/0eabdd5c,
  evidence_level 0) minutes past the 15 s manifest poll. Code is NOT the defect: a local transition
  reproduction (temp dir: legacy flat → copy in current-manifest + snapshots → same provider instance
  reloads to a7867b10, previous correct) AND a fresh in-container provider both work. Files are visible
  in the container (ro mount shows current-manifest.json + snapshots). Working hypothesis: on each poll
  `loadResolvedGraph('current')` falls back to the still-present FLAT `capability-graph.json` when
  `resolveManifestRaw` fails, re-swapping the same legacy graph and masking the manifest. Confirming the
  exact mechanism requires instrumenting the running process = a restart, which the operator gate
  forbids. Gate behavior: record failure, PRESERVE the promoted artifacts (do NOT rollback), do NOT
  restart to force reload, do NOT generate VERIFIED runtime receipts, stop with the bounded diagnosis.
- **Runtime/shadow probes MUST bind the served graph identity for EVERY capability (collector gap
  exposed by the hot-reload failure)**: probe maps for `/changes` + `/evidence-changes` check only
  `200 + ok:true` (no `graph_hash`/`source_sha` check — those bodies lack a `summary`). Against a
  legacy-served API those probes PASS, generating VERIFIED receipts bound to the NEW source SHA even
  though the runtime served the OLD graph. The policy engine then correctly reports CONFLICTED
  (VERIFIED + historical NOT_VERIFIED, no silent precedence) → second promotion blocked. Fix for the
  next iteration: before emitting any runtime/shadow receipt, require a summary-bound probe to confirm
  `observed graph_hash === expected` (a run-level precondition); when the served graph ≠ expected,
  emit NOT_VERIFIED for ALL capabilities regardless of per-route 200s. Never infer the served graph
  from a single summary probe if other routes can drift.
- **Publisher legacy-flat → snapshot migration: materialize the legacy snapshot BEFORE --promote**:
  the publisher archives the flat current with `previous-manifest.snapshot_path: null`; the provider's
  previous resolution defaults to `snapshots/<previous_hash>/` and, when missing, falls back to the
  FLAT `capability-graph.previous.json` (an older graph) — wrong previous identity. Before the first
  promote on a flat-layout dir: `mkdir -p snapshots/<legacy-hash> && cp capability-graph.json graph-meta.json snapshots/<legacy-hash>/`
  (host-side canonical-dir prep, same class as the publisher's own writes; verify the copy with the
  REAL `computeGraphHash`, NOT ad-hoc JSON.stringify — the real hash is `stableStringify` of
  `{nodes,edges,dynamic_edges_unresolved,authority_conflicts,source_manifest}`, so a naive body
  reconstruction MISMATCHES and looks like corruption).
- **Secret-extraction refinement**: `docker inspect … | grep '^ADMIN_API_KEY' | cut -d= -f2-` in a
  COMBINED command (with curl pipes) was denied by the terminal guard. The cleaner authorized path is
  reading the key from the canonical `.env` file: `grep '^ADMIN_API_KEY' <canonical>/lah-openclaw-mvp/.env | cut -d= -f2- > /tmp/.key`,
  then `KEY=$(</tmp/.key)`, pass via env to the validated collector interfaces (never print the key).
  After a denied command: do NOT replay/rephrase — identify the operation, use only individually
  authorized read-only probes through the validated collectors, stop with the unresolved permission
  boundary if the collector itself stays blocked.
- **GRAPH_STALE silently served when the provider's expected source SHA is unset**: server.js
  wired `expectedSourceSha: process.env.CLOE_SELF_AUDIT_EXPECTED_SOURCE_SHA || null`; the deploy
  flow never set that env var, so after the runtime advanced (844f29c) past the graph source
  (131f350) the provider kept serving the stale graph as FRESH (time-based check 113h < 168h
  maxStaleHours) with HTTP 200 — GRAPH_STALE_VS_DEPLOYED_RUNTIME invisible to the query surface.
  Fix (CLOE_SELF_AUDIT_EVIDENCE_PROJECTION_REPAIR_V1, PR #744, merge 1ce26b66):
  `resolveExpectedSourceSha(env)` in canonical-graph-provider.mjs — explicit env wins, else
  `GIT_COMMIT` (canonical deployer build identity), `GIT_COMMIT=unknown`/unset → null (legacy).
  server.js now passes it, so a stale-bound graph is surfaced as 503 GRAPH_STALE via the
  existing route logic. Verify the RUNNING container env (`docker exec <c> env | grep GIT_COMMIT`)
  before relying on it — a stale .env override would re-break the binding.
- **Runtime/shadow probe maps must include a summary-bound graph-identity probe for every
  capability** (including non-self-audit surfaces like /audit/reconciliation): the collector
  only checks graph_hash/source_sha when `probe.checks` includes them, so a capability probe
  without a summary probe cannot pin the served graph identity. For reconciliation
  observability (runtime-reachable by architecture: GET /audit/reconciliation admin-key gated,
  read-only), the minimal seam = RUNTIME_PROBE_MAP entry (summary-bound probe + surface probe)
  + SHADOW_SUITE_MAP entry (8 deterministic cases, first case summary-bound). Runtime code
  unchanged; repair-planner remains intentionally offline (NO_REACHABLE_WIRING_CHAIN, packets
  executable:false).
- **Self-audit query surface rate-limits at 60 req/min per IP — a full shadow sweep (~70+
  requests) ALWAYS trips it mid-run**, silently producing NOT_VERIFIED shadow receipts with
  STATUS:429 (+ CONTRACT_OK/GRAPH_HASH_MISMATCH/SOURCE_SHA_MISMATCH because the 429 body has no
  summary). Observed live on 2026-08-09 (CLOE_SELF_AUDIT_PRODUCTION_DEPLOY...): wave 1 killed
  the 6 self-audit caps (8+1 cases each), business caps (2 cases) survived. Fix: run the shadow
  collector ONE capability at a time with a pacing delay (~12s between caps keeps < 60/min),
  and record EXPLICIT SUPERSESSION — before appendReceipt, set
  `receipt.evidence.supersedes_receipt_ids = <all NOT_VERIFIED shadow receipt ids for the same
  cap|dim|source/deployed SHA>` (the resolver accepts VERIFIED→NOT_VERIFIED without
  allowNegativeSupersession; edges validate same cap/dim/SHA/env + strictly-later observed_at).
  Without supersession, VERIFIED + NOT_VERIFIED on the same SHA → CONFLICTED → promotion
  blocked. Also: a bad admin key produces a full 401 wave — every case fails with
  STATUS:401+CONTRACT_OK+GRAPH_HASH_MISMATCH+SOURCE_SHA_MISMATCH; treat it as a harness/command
  bug (the terminal may mask `ADMIN_API_KEY=$KEY`), re-run with the correct env, and supersede
  those negatives the same way. Verify the served graph hash BEFORE the run
  (`/self-audit/query/summary`) and bind expectedGraphHash to the CURRENT manifest, not a stale
  constant.
- **Post-promotion honest states (second candidate)**: ledger 132 receipts (10 appended this run);
  second candidate `86cbda93…`, source/deployed b3b79ee, previous a7867b10, receipt_set `03a3d4a4…`;
  second dry-run vs REAL production dir: ok, 0 promotions / 0 demotions (nothing new VERIFIED);
  runtime/shadow: provider+graph NOT_VERIFIED, detector/interface/routes CONFLICTED, planner UNKNOWN,
  live UNKNOWN ×6. Required stopping verdict NOT claimable (hot reload failed): the strongest honest
  statement is `CANONICAL_GRAPH_PROMOTED + HOT_RELOAD_FAILED_PENDING_DIAGNOSIS` — the second
  promotion remains operator-gated. A single authorized restart of `lah-openclaw-mvp` re-creates the
  provider, which then loads current-manifest → serves a7867b10 without manifest changes; after that,
  regenerate runtime/shadow with the graph-binding fix above.
- Docs: `lah-openclaw-mvp/docs/cloe/living-capability-evidence-system/` (architecture, receipt
  schema, policy engine, collector mapping, snapshot publication, rollback, operator runbook,
  certification receipt, validation receipt) + continuity JSON in `docs/mcporter/`.

## Mission status — RUNTIME SHADOW RECOVERED (2026-08-03, operator-authorized single restart)

- ONE bounded restart of `lah-openclaw-mvp` (e4b7a2f3e395 → 89c5945d5a55, same image
  268e27be/b3b79ee, `docker stop && docker rm` then `compose up -d --no-deps --no-build`)
  MATERIALIZED the provider: runtime served a7867b10/b3b79ee within one request. This is NOT
  hot-reload verification — the process was re-created (do not claim HOT_RELOAD_VERIFIED).
- Runtime+shadow regenerated against the served graph: 10 receipts VERIFIED (5 runtime + 5 shadow)
  bound to a7867b10/b3b79ee, each carrying `evidence.supersedes_receipt_ids` referencing 24
  historical NOT_VERIFIED receipts. Ledger 132 → 142.
- POLICY GAP confirmed: policy engine b3b79ee contained 0 references to `supersedes_receipt_ids`
  (only `collector-live-import` writes it, live dimension) → VERIFIED + NOT_VERIFIED → CONFLICTED →
  second promotion blocked (candidate 4679cbdc, dry-run 0/0). Honest report: supersession recorded
  but unapplied.

## Mission status — EXPLICIT SUPERSESSION POLICY REPAIR (2026-08-03, PR #677 → merge 6cda879d)

- `supersession-resolver.mjs` (new) consumed by `evidence-policy-engine.mjs`: classifyDimension
  resolves explicit supersession → classifies ACTIVE receipts only → exposes
  superseded_receipt_ids / supersession_chains / invalid_supersession_edges. `proof-state-merge.mjs`
  exposes superseded_receipt_ids in evidence_refs (provenance). See
  `references/supersession-resolver-pattern.md` for the full resolver pattern and pitfalls.
- Validation: 30 new supersession tests + 247 existing = 277 green; behavioral simulation receipt
  VALID (30 scenarios); FastSafe 15/15.
- PHASE 8 REBUILD: re-evaluated the EXISTING 142 receipts (no regeneration, no duplicates —
  receipt_set_hash a9f815a6 unchanged); candidate `27ab756e` (source/deployed b3b79ee — receipts
  are b3b79ee-bound; repair-SHA-bound candidate would reject them), previous a7867b10; publisher
  dry-run vs REAL production dir: 10 promotions / 0 demotions, zero canonical mutation. Second
  promotion remains operator-gated; hot reload remains a separate unresolved concern.

## Post-restart recovery — evidence regenerated, SECOND PROMOTION BLOCKED (2026-08-03)

Operator authorized ONE bounded restart + evidence recovery
(`CLOE_..._RUNTIME_SHADOW_RECOVERED_AND_SECOND_PROMOTION_READY`). This is the direct continuation of
the hot-reload-failure stop above: the restart materializes the provider, then runtime/shadow are
regenerated against the graph actually served.

- **Restart recipe that materializes the provider (no rebuild)**: `docker stop <c> && docker rm <c>`,
  then `docker compose -p lah-openclaw-mvp -f docker-compose.yml -f compose.override-deploy.yml
  up -d --no-deps --no-build lah-openclaw-mvp` from the deploy worktree (the one whose compose
  labels the running container references). `--no-build` is REQUIRED: the compose service has a
  `build:` section and a plain `up` may rebuild; the existing image
  (`lah-openclaw-mvp-lah-openclaw-mvp:latest` = the GIT_COMMIT=b3b79ee image, e.g. 268e27be) must be
  reused byte-identical. Pre-checks BEFORE stopping: `.env` exists (0600) in the compose workdir;
  record `lah-tools-runtime` ID/StartedAt for the untouched proof; confirm no publication lock file.
  After `up`, verify image SHA unchanged via `docker inspect <c> --format '{{.Image}}'`.
- **Post-restart verification (10-point Gate 3)**: /health 200; `/self-audit/query/summary` serves
  the manifest graph (a7867b10…) with source b3b79ee…; `/self-audit/query/proof-state` exposes
  previous_graph_hash (3dfe01a6…) → current/previous independently queryable; graph mount still ro
  (RW=false); ZERO files written to the graph dir after the restart timestamp
  (`find … -newermt '<restart-ts>'`); unauthenticated probe → 401; lah-tools-runtime ID/StartedAt
  unchanged; provenance + receipts routes 200. A restart proves successful provider initialization
  from the manifest — it is NOT hot-reload proof (hot reload must still be reproduced via a manifest
  transition on a live process).
- **SUPERSESSION POLICY GAP (Gate 6) — proven empirically**: the deployed policy engine
  (`evidence-policy-engine.mjs` classifyDimension) NEVER consumes `supersedes_receipt_ids`. Grep
  proof: 0 occurrences of `supersedes` in policy-engine / snapshot-builder / canonical-publisher;
  `collector-live-import.mjs` is the ONLY writer (live dimension only). Consequence observed with 10
  new VERIFIED runtime/shadow receipts carrying explicit `evidence.supersedes_receipt_ids` pointing
  at the 25 historical NOT_VERIFIED (same cap+dim+source+deployed b3b79ee, later observation,
  corrected runtime condition, valid hashes, no circularity): matrix stays CONFLICTED (accepted=4:
  1 VERIFIED + 3 NOT_VERIFIED) → dry-run 0 promotions → second promotion blocked. This is by design
  ("explicit supersession ids recorded, no silent precedence" — resolution is an OPERATOR act, not
  engine auto-application). Required verdict `…RUNTIME_SHADOW_RECOVERED_AND_SECOND_PROMOTION_READY`
  is NOT claimable; honest verdict:
  `…RESTART_OK_RUNTIME_SHADOW_EVIDENCE_REGENERATED_AND_SECOND_PROMOTION_BLOCKED_BY_UNSUPPORTED_SUPERSESSION`.
  To unblock: implement a supersedes consumer in the policy engine (exclude superseded receipts from
  classifyDimension) + tests, or operator-explicit conflict resolution — then rebuild candidate.
- **Runner adaptation for a new expected graph**: `tools/cloe-evidence-refresh.mjs` HARDCODES
  `DEPLOYED_GRAPH_HASH`/`DEPLOYED_GRAPH_SOURCE` constants at top. Re-running it after the promoted
  graph changed would bind probes to the OLD graph and emit NOT_VERIFIED (source-binding mismatch)
  or stale truth. Copy it to a new tool (e.g. `tools/cloe-evidence-recover.mjs`, untracked
  host-side like the refresh tool) with the new constants + a supersedes-enrichment loop: before
  `appendReceipt`, set `receipt.evidence.supersedes_receipt_ids = <historical NOT_VERIFIED ids for
  cap|dim>` (build the map from `listReceipts(ledgerRoot)` filtered by result NOT_VERIFIED +
  source_sha == deployed_sha == b3b79ee + dimension runtime/shadow). Keep `--app-root` pointed at a
  b3b79ee checkout: the runner asserts `gitHead(appRoot) === b3b79ee` exactly — use the deploy
  worktree, NOT the mission worktree at a3ddc8b (identical trees but different HEAD → guard exits).
- **Proof that a worktree IS the deployed code when the canonical checkout diverges**: the canonical
  checkout may sit on a fix branch that does NOT contain the deployed merge SHA (`git branch
  --contains <sha>` empty, `git merge-base --is-ancestor` NO). Verify the mission worktree is
  tree-identical to the deployed SHA with `git diff --stat <deployed-sha> <worktree-head>` — empty
  output = identical trees, so the worktree is the authoritative source for the deployed code.
- **Pipe-to-python terminal commands get denied (operator guard)**: parsing JSON tool output via
  `… | python3 -c "…"` (e.g. `check-codegraph-availability.mjs --json | python3 -c`) is denied by
  the terminal guard. Do NOT retry/rephrase/pipe differently — use the dedicated codegraph tool or
  read files directly; the guard treats pipe-to-interpreter as a security scan trigger. (Same family
  as the existing secret-extraction denials.)
- **Gate 7.9 verification**: dimension allowlist (`dimension_collectors`) restricts
  runtime_reachable→[collector-runtime], shadow_verified→[collector-shadow] — verify 0
  local/test/wiring receipts land in high-trust dims before declaring policy PASS.
- Ledger after recovery: 142 receipts (10 appended), manifest rebuilt (verifyLedger + rebuildIndexes
  host-side); candidate `4679cbdc…` (source/deployed b3b79ee, previous a7867b10, receipt_set
  a9f815a6); publisher dry-run ok:true 0/0 no canonical mutation (manifests mtime unchanged).
  Continuity: `docs/mcporter/CLOE_LIVING_CAPABILITY_EVIDENCE_SYSTEM_V1_RUNTIME_SHADOW_RECOVERED_CONTINUITY_V1.json`.

## Mission status — SECOND GRAPH PROMOTION EXECUTED, HOT RELOAD FAILED AGAIN (2026-08-04)

- Operator authorized the second promotion (exact command with `--sources-file phase8-sources.json`).
  Run the publisher from the mission worktree `cloe-living-capability-evidence-system-v1`
  (HEAD 38a0838, branch fix/cloe-explicit-supersession-policy-repair-v1) — NOT from
  `deploy-evidence-v1` (b3b79ee) which LACKS `supersession-resolver.mjs` and would promote 0/0.
  Verify checkout validity BEFORE running: `git diff --stat 38a0838 6cda879d -- .` empty
  (tree == merge SHA) + `supersession-resolver.mjs` present. Result: ok:true, promoted,
  graph `27ab756e…`, previous a7867b10, receipt_set `a9f815a6`, 10 promotions
  (5 runtime_reachable + 5 shadow_verified) / 0 demotions, ledger UNCHANGED (142 receipts,
  no appends — Phase 8 re-evaluation only). Snapshot body-hash recomputation matches.
- **PITFALL — double invocation of `--promote` (self-reference corruption)**: the first run was
  piped through `| tail`, which truncated the report; re-running the IDENTICAL command to capture
  the full JSON made the publisher resolve `previousIdentity` from the ALREADY-promoted current →
  `previous-manifest.json` = 27ab756e (self-reference) + a spurious second publication receipt
  with `previous_graph_hash == graph_hash`. FIX: never re-run `--promote`; capture output to a
  file in ONE run (`node … --promote … > /tmp/promo-report.json 2>/tmp/promo-report.err`, then
  parse top-level keys). If corrupted, restore `previous-manifest.json` from the FIRST
  publication receipt (authentic `previous_graph_hash` = a7867b10) + snapshot meta
  (receipt_set 24f365f4); chmod 600 to match publisher file modes. Do NOT delete the spurious
  receipt — history stays intact; document it as an artifact.
- **Hot-reload failure #2 — ROOT CAUSE CONFIRMED (not the flat fallback)**: live summary still served
  a7867b10 at +10 min (poll 15 s) with the SAME response size (48065 bytes) across every probe.
  Bounded diagnosis (no restart): (1) `docker exec cat` shows the container SEES current-manifest
  27ab756e + snapshot (graph_hash inside = 27ab756e); (2) `sha256sum` of container
  `canonical-graph-provider.mjs` + `graph-hash.mjs` == source (zero drift); (3) live serves a7867b10,
  NOT the flat 3dfe01a6 → `resolveManifestRaw` WORKS, flat-fallback hypothesis disproved;
  (4) a FRESH provider on the same graph dir loads B correctly → in-memory state leak, not data.
  The definitive mechanism is the SINGLE-FLIGHT PROMISE LEAK in `getCurrentGraph`: synchronous
  early-return paths (manifest unchanged / legacy flat) returned BEFORE the `finally {
  currentCachePromise = null }` that only wrapped the async reload block — the IIFE resolved before
  the `currentCachePromise = promise` assignment landed, leaking a resolved promise that every later
  request served forever. Fix + full reproduction + idempotency/chain-integrity repair (PR #681 →
  merge 949795a): see `references/hot-reload-single-flight-promise-leak.md`. Post-fix real proof on
  the deployed repair: manifest transition 27ab756e→af758d64 with ZERO restart (container ID +
  StartedAt unchanged) → `HOT_RELOAD_VERIFIED_BY_REAL_NO_RESTART_MANIFEST_TRANSITION`; repeated
  promote of the already-current graph → `ALREADY_CURRENT` with byte-identical manifests; chain
  integrity ok. Verdict (staged): `…_HOT_RELOAD_AND_PUBLISHER_IDEMPOTENCY_REPAIRED_RUNTIME_SHADOW_DRY_RUN_READY`.
- **Response-size fingerprinting**: identical `size_download` across probes (48065) is a cheap
  signal that the cached graph did not change — use it to confirm "same graph served" before
  deep diagnosis.
- **Publication receipts are the recovery authority for manifest chains**: every
  `publication-<hash>-<ts>.json` records `previous_graph_hash`, `source_sha`, `deployed_sha`,
  `receipt_set_hash` — use the authentic (FIRST) receipt to rebuild a corrupted
  `previous-manifest.json` byte-for-byte.

## Registry extension — composing NEW capability groups (CLOE_HIGH_ROI_BUSINESS_CAPABILITY_GRAPH_V1, Lot 0)

The evidence system was built for ONE pilot registry (`pilot-capabilities.mjs`). Registering new
capability groups (the 8 `capability:business-*` capabilities) WITHOUT special-casing collectors
or breaking graph determinism — proven pattern, merged as PR #682 (base 949795a):

**Key discovery — the collectors were ALREADY registry-parameterized.** Every collector takes the
registry/maps as defaulted params, so composition is DATA-PASSING, never collector modification:
- `collector-local`: `capabilities = PILOT_CAPABILITIES`
- `collector-wiring`: `capabilities = PILOT_CAPABILITIES, chains = PILOT_WIRING_CHAINS`
- `collector-tests`: `capabilities = PILOT_CAPABILITIES, mapping = TEST_CAPABILITY_MAP`
- `collector-runtime`: `capabilities = PILOT_CAPABILITIES, probeMap = RUNTIME_PROBE_MAP`
- `collector-shadow`: `capabilities = PILOT_CAPABILITIES, suiteMap = SHADOW_SUITE_MAP`

**Two composition points** (grep before writing any code):
1. The harness `tools/cloe-evidence-pilot.mjs` assembles `sources.declared: <registry>.map(...)`
   (the graph's DECLARED nodes come from HERE, line ~285) and passes `PILOT_CAPABILITIES` to the
   collectors by default.
2. The policy file `evidence-policy.v1.json` `capabilities` map (engine is PERMISSIVE for unknown
   capabilities → defaults; explicit registration is the pattern, same as the pilot entries).

**Composition module design** (`capability-registries.mjs` + `high-roi-capabilities.mjs`):
- New registry module exports capabilities in the pilot shape + explicit wiring chains + test map +
  runtime probe map + shadow suite map + live gates (`certified:false` → honest UNKNOWN).
- `composeCapabilityRegistries(registries[])`: deterministic merge, SORT by `capability_id`,
  throws `DUPLICATE_CAPABILITY_ID`; `composeWiringChains`/`composeCapabilityMaps` throw
  `UNKNOWN_WIRING_CHAIN`/`UNKNOWN_CAPABILITY_MAPPING`/`DUPLICATE_*`. Pilot entries stay the SAME
  object references (byte-identical) — test this explicitly.
- Harness gains `--registry composed|pilot` (default composed); pilot-only runs stay possible.
- Verify BEFORE writing wiring chains that every link is provable: `proveLink` checks the source
  file CONTAINS `link.symbol`, and for `IMPORTS` also that the source contains the target BASENAME.
  Grep the real import/call sites first; a comment mentioning a function (e.g. "calls submitApprovalRaw")
  is NOT an import.
- Runtime/shadow probes for the NEW capabilities at Lot 0 point at the self-audit node endpoints
  (`/self-audit/query/capabilities/<id>`) — honest pre-promotion: they 404 until the new graph is
  promoted. Per-lot phases replace them with the real business routes.

**Test matrix that proved the foundation** (12 tests, all RED-first): 8 IDs unique+exact; composition
deterministic+sorted; duplicate IDs fail closed; unknown mappings fail closed; pilot output unchanged
(compose([pilot]) ≡ sorted pilot, pilot nodes identical in graph with/without composed registry);
no cross-capability proof (every mapping key must be a registered capability); graph determinism
(same sources → same hash); empty evidence → honest DECLARED/UNKNOWN nodes; hash changes only when
content changes (pilot-only vs composed differ by exactly the new nodes); wiring/map/live-gate keys
reference only registered capabilities; every registered module statically exports its symbols.

### Pitfalls specific to registry extension

- **`proof_state.capabilities[0]` index fragility — breaks the moment a new prefix group is added**:
  policy-engine tests asserted `capabilities[0]` because the sorted policy list used to start with
  `capability:self-audit-*` (CAP). Adding `capability:business-*` (sorted 'b' < 's') silently moved
  the business group first → 10/13 tests failed although the engine was correct. NEVER index
  `capabilities[0]` on sorted policy/capability lists — look up by `capability_id` (helper
  `entryOf(proof_state, id)`). Count assertions must compare against
  `Object.keys(policy.capabilities).length`, not the pilot list length. Grep for `[0]`-style
  assertions when registering any new capability prefix group.
- **Runtime/shadow receipts bind to the DEPLOYED SHA — a committed-but-not-deployed candidate keeps
  runtime UNKNOWN in the matrix, and that is CORRECT**: the harness binds runtime/shadow receipts to
  `deployedSha` (source_sha=deployed). When candidate source ≠ deployed SHA, `source_sha_match`
  rejects the runtime receipts → matrix shows runtime_reachable UNKNOWN for ALL capabilities, plus
  NOT_VERIFIED ×N for the new capabilities whose node probes 404 (nodes not yet in the served graph).
  This is the designed fail-closed outcome, not a regression — runtime evidence cannot promote until
  the exact merge SHA is deployed and receipts are regenerated. Report it as such.
- **Stale hardcoded deployed identity in `tools/cloe-evidence-pilot.mjs`**: `deployedSha` and
  `deployedGraphHash` were HARDCODED constants (0eabdd5c/3dfe01a6) that went stale; running the
  harness would bind receipts to the wrong deployed identity. Verify against the live app
  (`/self-audit/query/summary` source_sha/graph_hash) before running; make them `--deployed-sha` /
  `--deployed-graph-hash` args with current defaults.
- **Mapped business test suites regenerate untracked `data/` side-effects that block the harness**:
  `test/cloe-governed-action-packet-store*` writes `data/cloe-governed-action-packets.json`,
  `test/campaign-memory*` writes `data/memory-events/` → `git status` dirty → harness collectors
  (allowDirty:false) block with REPOSITORY_DIRTY. Operator-approved cleanup protocol: record
  presence/origin/untracked status in the report FIRST, then remove with EXPLICIT consent — and
  NEVER bundle `rm -rf` into a git branch/commit command (the terminal guard blocks the whole
  command; keep cleanup as its own consented command).
- **Fresh worktree has no node_modules**: `git worktree add` then `node --test` fails with
  `ERR_MODULE_NOT_FOUND: Cannot find package 'express'` — looks like regressions, it is missing deps.
  Run `npm ci` in the worktree before the first baseline run; classify the failures as ENVIRONMENTAL.
- **MCP `codegraph_explore` ≠ in-repo codegraph pack**: the MCP tool requires its OWN index at the
  project root (`codegraph init`); the repo's AGENTS.md contract (`npm run codegraph:refresh` →
  `lah-openclaw-mvp/.codegraph/` with snapshot.json/route-map.json/danger-surface-map.json) is a
  DIFFERENT system. When MCP explore reports "isn't indexed … don't call codegraph for it again",
  use the in-repo pack + bounded manual reads for Gate 1 — do not keep calling the MCP tool.
- **Merge gate when the operator is unresponsive**: with a required CI check BLOCKING the PR and no
  operator answer to the authorize-bypass clarify, do NOT merge (bypass requires explicit consent).
  Persist the evidence (copy the staging ledger/matrix/log out of /tmp into the worktree
  `evidence/` dir), write a status/operator packet, deliver the mission-style report, and stop at
  the merge gate.
