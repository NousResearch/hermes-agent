# Evidence Authority Verification — Multi-Lot Capability Programs

Pattern for starting a program that will produce/promote evidence (CLOE capability graph, living evidence system): establish the TRUE authority before touching anything. Established during CLOE_HIGH_ROI_BUSINESS_CAPABILITY_GRAPH_V1 (2026-08-04).

## Host layout (openclaw-runtime / lah-openclaw-mvp evidence system)

- Active evidence root: `/home/deploy/cloe-self-audit-evidence/living-evidence-v1/`
  - `ledger/ledger-manifest.json` → receipt_count + receipt_hashes (182+ immutable receipts)
  - `ledger/receipts/` (content-addressed receipts), `ledger/indexes/`, `ledger/rejected/`
  - `graph-staging/` (host-side publisher staging)
- Canonical graph mount (app reads RO): `/home/deploy/cloe-self-audit-evidence/lot10/graph/`
  - `capability-graph.json` + `capability-graph.previous.json`
  - `current-manifest.json` / `previous-manifest.json` ← AUTHORITATIVE for graph_hash, source_sha, deployed_sha, receipt_set_hash, snapshot_path
  - `snapshots/<graph_hash>/`, `publication-receipts/`
- Container mounts: `lot10/graph → /app/data/self-audit/canonical-graph` (ro) + data dir → `/app/data` (rw)

## Live verification commands (never print secrets)

```bash
# Running app's own authority — key fetched from container env, used in-place, never echoed
KEY=$(docker exec lah-openclaw-mvp printenv ADMIN_API_KEY)
curl -s -H "x-admin-api-key: $KEY" http://127.0.0.1:4000/self-audit/query/summary
# → graph_hash, source_sha, nodes[] (101 capabilities), stats
# Ledger count + receipt-set hash + deployed SHA:
cat /home/deploy/cloe-self-audit-evidence/lot10/graph/current-manifest.json
# Container identity:
docker inspect lah-openclaw-mvp --format '{{.State.StartedAt}} {{.Image}}'
```

## Traps

1. **Container env GIT_COMMIT is a STALE build-arg, not the deployed SHA.** Compose `${GIT_COMMIT:-…}` resolves from the build shell env, not the canonical repo. Observed 2026-08-04: env GIT_COMMIT=2855674b (PR #649 merge) while the app declared source_sha/deployed_sha=949795a (current-manifest.json + self-audit API agree). Bind ALL evidence to the app-declared SHA; document the env discrepancy in operator packets; pass `GIT_COMMIT=$(git -C <canonical> rev-parse HEAD)` explicitly on the next rebuild.
2. **graph-hash.txt / graph-hash.previous.txt in the graph dir can be STALE** (dated at graph build, not at last promotion). The manifests (current/previous) + the live API are the authority — never the .txt files.
3. **Fresh worktree baseline: `npm ci` FIRST.** `node --test` in a fresh worktree fails with `ERR_MODULE_NOT_FOUND: Cannot find package 'express'` on route/conversation tests. `ls node_modules | wc -l` → 0 confirms environmental cause. Classify pre-`npm ci` failures ENVIRONMENTAL; re-run after `npm ci` before calling anything a regression.
4. **CodeGraph tooling lives in the worktree at the exact SHA** (`tools/codegraph/freshness-check.js --repo openclaw`). A diverged/dirty canonical checkout on another branch may lack `tools/codegraph/` → MODULE_NOT_FOUND. Run repo tooling from the program worktree; if index missing → `npm run codegraph:refresh` (local gitignored artifacts in `.codegraph/`).
5. **Stale local refs/heads/main ≠ origin/main.** The checkout can sit on an unrelated feature branch while local `main` points at an old SHA. Authority = `git fetch origin && git rev-parse origin/main`. Create the program worktree from the FULL origin/main SHA (`git worktree add <path> <full-sha>`) so the base can never drift.

## Reuse-audit shortcut for capability programs

Build the mission's "existing implementation map" fast:
- For each lot's named files: `[ -f path ] && echo EXISTS || echo MISSING`
- Grep exports of the 2-3 key files to classify: REUSE (exists + wired), REPAIR (exists but gap), CREATE (absent)
- Cross-check conversation wiring with `grep -rln "<module>" src/routes/ src/cognitive/ src/services/cloe-*conversation*` — zero hits means NOT wired into the conversational path (a REPAIR gap even if the module exists, e.g. campaign-memory is wired via unified-retrieval-gateway, not a dedicated conversation service).
