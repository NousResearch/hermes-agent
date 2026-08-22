# Source-SHA Boundary Promotion Pattern (evidence system)

When a mission promotes a candidate whose **source SHA differs from the previously
deployed SHA** (i.e. the first promotion after deploying new code), the proof-state
engine mechanically rejects every high-trust receipt (`runtime_reachable`,
`shadow_verified`, `live_verified`) bound to the OLD deployed SHA. Result: the
candidate shows **demotions of previously VERIFIED capabilities** (e.g. a pilot
`LIVE_VERIFIED` lvl 6 capability drops to lvl 3). This is by design (source_sha_match
+ deployed_sha_match), NOT a bug — but it trips the "pilot proof states unchanged /
0 unexpected demotion" gates on every SHA-crossing promotion.

Established 2026-08-04 during CLOE_HIGH_ROI_BUSINESS_CAPABILITY_GRAPH_V1 Lot 0
(949795a → 5d6e9cb).

## Root cause: the probe chicken-and-egg

- Runtime/shadow collectors compare the **served** graph's `source_sha` (from the
  mounted canonical-graph manifest, e.g. `current-manifest.json`) against
  `expectedSourceSha`.
- The served manifest only carries the new SHA **after** a promotion.
- Therefore probes for source_sha-checked dims CANNOT pass before the first promotion
  at the new SHA, yet the first promotion candidate shows them as NOT_VERIFIED
  (demotion).

Non-source_sha-checked probes (checks list without `source_sha`, e.g. capability-graph
runtime probe) DO pass pre-promotion.

## The two-phase pattern (operator-approved remedy)

1. **Phase 1 — promote the honest candidate** at the new SHA. The demotions are
   mechanical and must be DOCUMENTED as expected (capability, dimension, from/to,
   reason = SHA-rebind). `current` becomes the phase-1 graph, `previous` = the old graph.
2. **Re-collect high-trust receipts at the new SHA** — now the served manifest carries
   the new SHA so the probes pass:
   - runtime + shadow collectors against the live app (expectedGraphHash = phase-1 hash)
   - re-certify live gates (see below)
3. **Resolve VERIFIED↔NOT_VERIFIED conflicts**: the ledger now holds BOTH the
   pre-promotion NOT_VERIFIED receipts AND the post-promotion VERIFIED receipts, all
   bound to the same new SHA → every such dimension becomes CONFLICTED (not restored).
   Fix: append **superseding copies** of the VERIFIED receipts whose
   `evidence.supersedes_receipt_ids` targets the pre-promotion NOT_VERIFIED receipt ids.
   The supersession resolver then marks the negatives superseded → state VERIFIED.
4. **Phase 2 — re-promote** the new candidate. Verify: 0 demotions vs the original
   graph, pilot/unchanged capabilities byte-identical proof states, business
   capabilities at their intended level.

Chain: old_graph → phase1_graph → final_graph. `previous` after phase 2 = phase1 graph
(operator accepted this in Option A).

## Constructing superseding receipts (the CANON_FAIL trap)

`canonicalizeReceipt` recomputes `receipt_id`/`receipt_hash` from content and REJECTS
mismatched provided values:
- strip BOTH `receipt_id` AND `receipt_hash` from the receipt before adding
  `evidence.supersedes_receipt_ids` (RECEIPT_ID_MISMATCH then RECEIPT_HASH_MISMATCH
  otherwise);
- supersedes edges must be same-capability (the resolver rejects cross-capability /
  cross-dimension hiding);
- targets must already exist in the ledger (INVALID_SUPERSEDES_TARGET).

## Re-certifying live gates at a new deployed SHA

`importLiveGateVerdict` (collector-live-import.mjs) requires: gate_id, test_case,
expected_assertions[], observed_assertions[], pass, source_sha, deployed_sha,
graph_hash, runtime_identity, certification_authority (OPERATOR), certified_at (ISO),
capability_ids[], raw_receipt_digest. Optional supersedes_receipt_ids.

TWO gotchas:
1. **One verdict PER capability** — the importer copies the same `supersedes_receipt_ids`
   list onto every capability_id in the verdict. A multi-capability verdict therefore
   creates cross-capability supersession edges → rejected. Use `capability_ids: [single]`.
2. **The importer does NOT append** — it returns `{ ok, receipts, imported, result }`;
   the caller must `appendReceipt(ledgerRoot, receipt)` each returned receipt.

Re-attest each gate against the LIVE runtime (container identity + StartedAt + the
actual API responses you observed) — never fabricate.

## Publisher requires the real sources

`bin/cloe-evidence-publisher.mjs --dry-run/--promote` WITHOUT `--sources-file` builds a
candidate with `sources = {}` → fails `CANDIDATE_NOT_LOADABLE`. Generate the sources
with the harness: `node tools/cloe-evidence-pilot.mjs ... --out-sources /tmp/sources.json`
then pass `--sources-file /tmp/sources.json` to the publisher.

## Harness stale defaults + REPOSITORY_DIRTY (2026-08-04 Lot 1)

`tools/cloe-evidence-pilot.mjs` has STALE hardcoded defaults: `--deployed-sha` defaults to
`949795a...` and `--deployed-graph-hash` to `f24eed48...`. Every run MUST pass
`--deployed-sha <new sha> --deployed-graph-hash <currently served hash>` or the
runtime/shadow probes compare against the wrong identity and produce NOT_VERIFIED.

Run the harness from a FRESH `git worktree add /tmp/<name> <merge-sha>` — the deploy
worktree carries untracked compose override files that trip `allowDirty: false`.
The harness APPENDS receipts to the ledger itself (Lot 1: 296 → 367) and runs a dry-run
against staging; read `--out-matrix` for the per-capability proof matrix and
`--out-sources` for the publisher sources file. Mapped test suites write side-effect files
into `data/` (~300 ms in) which block every SUBSEQUENT harness run with REPOSITORY_DIRTY
(pre-clean `data/memory-events` / `data/cloe-governed-action-packets.json` with operator
consent, or use a fresh worktree). Full session detail:
`references/campaign-memory-lot1-proof-pattern.md`.

## Dry-run demotions are SOURCE_SHA_MISMATCH (expected, not regressions)

The real dry-run on the canonical graph dir (`--graph-dir <lot10>/graph`) after bootstrap
collect showed promotions 16 / demotions 15 / changed 31, with ALL 15 rejection reasons
being `SOURCE_SHA_MISMATCH`. Verify demotions are 100% mechanical before phase 1 promote.
Watch-out: 4 business caps (creative-inventory, governed-campaign-creation, business-health,
tracking-attribution) drop `tested` VERIFIED→NOT_VERIFIED because HIGH_ROI_TEST_MAP maps
campaign-memory but not those caps — plan re-attestation during the recollect phase.

## Passing ADMIN_API_KEY without the secret-masker mangling the command

The Hermes secret masker replaces `$(docker exec … printenv …)` and `read < keyfile`
patterns with `***` IN THE EXECUTED command → silent `syntax error` / exit 1 with no
output. Reliable workaround:

```bash
set -a && . /home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp/.env && set +a
node tools/cloe-evidence-pilot.mjs ...   # harness reads process.env.ADMIN_API_KEY
```

(the harmless `fg: no job control` warning from sourcing is expected).

## Harness REPOSITORY_DIRTY from mapped-test side-effects

The test collector RUNS the mapped test files. Some mapped suites write into
`data/` (e.g. governed-action-packet-store → `data/cloe-governed-action-packets.json`,
campaign-memory → `data/memory-events/`) ~300 ms into the run. These are untracked
side-effects that:
- appear AFTER the local/wiring/tests dirty checks in a clean-start run (run succeeds);
- block EVERY SUBSEQUENT harness run with `REPOSITORY_DIRTY` on local/wiring/tests
  (only runtime/shadow receipts are then produced — a tell-tale 26-receipt ledger).

Fix: `rm -f data/cloe-governed-action-packets.json && rm -rf data/memory-events` before
each harness run. These files are env-overridable in the store
(`CLOE_GOVERNED_ACTION_PACKETS_FILE`), so a durable fix = point tests at a temp path.

## Deployment pitfalls at the new merge SHA

1. **GIT_COMMIT in `.env` beats the image build arg.** The container env comes from
   `env_file: .env` — a stale `GIT_COMMIT=…` line in the copied `.env` overrides even a
   correct `docker compose build --build-arg`. Fix BOTH: edit the deploy `.env`
   GIT_COMMIT line AND pass `--build-arg GIT_COMMIT=<sha>`. Verify the baked value:
   `docker run --rm --entrypoint sh <image>:latest -c 'echo $GIT_COMMIT'`.
2. **Non-default override files need explicit `-f`.** `compose.override-deploy.yml` is
   NOT auto-loaded (only `compose.override.yml` is). Without `-f docker-compose.yml -f
   compose.override-deploy.yml`, the relative `./data` volume resolves inside the fresh
   deploy worktree → mounts an empty dir → the app loses its runtime data. Always
   override data + graph mounts with ABSOLUTE canonical host paths, and always invoke
   compose with both `-f` flags. Verify mounts after `up`:
   `docker inspect <c> --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{"\n"}}{{end}}'`.
3. Recreate = `docker stop <c> && docker rm <c>` then `docker compose -f … -f … up -d
   --no-deps <service>` (the --force-recreate name-conflict pitfall).
4. Only recreate the app container; leave other services (e.g. lah-tools-runtime) untouched.

## ALREADY_CURRENT / hot-reload verification checklist

After each promotion:
- re-invoke `--promote` with the same args → `promoted: false`,
  `message: ALREADY_CURRENT … manifests untouched`;
- `sha256sum` current/previous manifests identical before/after; no new snapshot dir;
  exactly ONE `PROMOTED` publication receipt per graph hash;
- current != previous, no self-reference, publication lock released (no `*.lock`);
- poll `GET /self-audit/query/summary` (x-admin-api-key): served graph_hash transitions
  to the new hash; `docker inspect` container ID + StartedAt unchanged, restarts=0
  (hot reload without restart).

## Policy-growth-robust evidence tests

Tests that index `proof_state.capabilities[0]` break the moment the policy registers
new capabilities (`capability:business-*` sorts BEFORE `capability:self-audit-*`).
Pattern: `entryOf(proofState, capabilityId)` lookup by id; P8-style count assertions
should compare against `Object.keys(policy.capabilities).length` and assert the honest
UNKNOWN state for newly registered capabilities with no receipts.
