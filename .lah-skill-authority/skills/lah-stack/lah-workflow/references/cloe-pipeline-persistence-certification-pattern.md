# CLOE Pipeline Persistence Certification Pattern

Established during CLOE_LAHB_AUTONOMOUS_AFFILIATE_RUNTIME_E2E_V1 (LOT 8, 2026-08-08).
Applies to MIXED certification missions that must prove a conversational/agent pipeline
both produces a business artifact AND persists it durably.

## 1. Certify against the DEPLOYED SHA, never the local working tree

The canonical checkout is often on an unrelated branch (e.g. `fix/cloe-p2-fallback-dashboard-scope`
at b65cc67) while the deployed container runs the mission SHA (5275b0a). `git diff` against the
deployed SHA shows hundreds of spurious deletions — red herrings.

Proof the container matches the repo before certifying anything:

```bash
# container file hash vs repo SHA hash — MUST match
docker exec <container> sha256sum /app/src/services/<file>.js
git -C <canonical_checkout> show <deployed_sha>:<path> | sha256sum
```

Extract ALL files under test at the deployed SHA into a scratch dir, and read/certify from
those copies — never from the working tree, never from a worktree at a different SHA.

## 2. Persistence-seam proof: before/after data-layer inspection

A perfect assistant answer is NOT proof of persistence. Cloé can answer PAUSED_READY with a
grounded candidate while persisting nothing.

- Snapshot counts + mtimes of every data layer BEFORE the E2E turn:
  `decision-records/`, `memory-events/`, `cloe-governed-action-packets.json`,
  `execution-receipts.json`, `campaign-ledger.json`.
- Re-check AFTER the turn. Zero delta = seam absent, regardless of answer quality.
- Grep for the writer's callers to prove it is unwired:
  `grep -rln "createRecord" src/ | grep -v test` → if only the writer file itself matches,
  the seam is unwired.
- `MEMORY_APPEND_WRITE` unset means `appendMemoryFact` runs in read_only/dry-run mode and
  writes nothing — an env gate that silently disables persistence.

Minimal repair (1 cycle, reuse existing writers — do NOT build a new framework):
- Call the existing writer (e.g. `createRecord` from `src/decision/cloe-v5-operational-decision-record.js`)
  in the pipeline block that produces the artifact, only on the success path (PAUSED_READY),
  NOT on the blocked path (fail-closed: no record when BLOCKED).
- Non-blocking try/catch: if persistence fails, the conversational pipeline continues.
- Expose `decision_record_id` + status in `cognitiveContextPack.available_items[...].metadata`
  and in `compact_summary` so Cloé can cite the record in its answer.

## 3. Pipeline test-harness trap: provider env param required

Direct module call returns READY_PAUSED but the full pipeline (`buildBrainAskResponse`)
returns `LOCAL_FALLBACK_ANSWER_READY` and your patched block never executes.

Root cause: `resolveBrainProviderConfig(env, sessionKey)` reads the `env` PARAM. Passing
`env: {}` → `enabled:false` → early return `buildLocalFallbackBrainAskResponse` BEFORE the
block under test (the generate block at ~1722 sits after the fallback returns at ~1417).

Fix: pass the provider config in the env param of the pipeline call:

```js
env: {
  OPENCLAW_BRAIN_PROVIDER: 'deepseek',
  DEEPSEEK_API_KEY: '...',
  DEEPSEEK_BASE_URL: 'https://api.deepseek.com',
  DEEPSEEK_MODEL: 'deepseek-v4-flash'
}
```

Asymmetry to remember: the LAH Brain bridge (`business-runtime-bridge.js`) reads `process.env`
(`getBaseUrl()` → `LAHB_URL`, `getSecret()` → `LAHB_ADMIN_API_KEY`), so those go in
process.env — NOT the env param. Provider config comes from the env param. Set both.

Diagnostic sequence that found it: add a console.error on the proposalResult line inside the
block + in the catch — neither fired → block never reached → traced the early-return fallbacks
at ~1416-1453 → provider disabled because env param was `{}`.

## 4. Secret-masking corrupts hand-written token-extraction lines

Rebuilding a curl script that extracts `ADMIN_API_KEY` via awk mangles the line under
secret-masking: `sub(/^ADMIN_API_KEY=*** ...` — the KEY line is corrupted and the token
comes back empty at runtime.

Fix: `cp` a previously working script (e.g. a prior lot's cert script) and `sed -i` ONLY the
changing parts (conversation_id prefix, prompt text). Never hand-write the KEY extraction
line. Verify with `bash -n` before running.

## 5. Worktree for the repair cycle

Base the repair worktree on `origin/main` (which contains the deployed SHA as ancestor), not
on the deployed SHA itself:

```bash
git worktree add <path> origin/main
git -C <path> checkout -b fix/<mission>-<seam>-v1
```

Confirm the deployed SHA is an ancestor of origin/main first:
`git merge-base --is-ancestor <deployed_sha> origin/main`.
