---
name: lah-workflow-small-model
description: "Compact gated LAH Stack workflow for small/fast models (Ling 3 Flash, etc.). Same role as lah-workflow; optimized for explicit decision boundaries, bounded discovery, certified runbook fast path, and strong STOP semantics. Do not modify lah-workflow."
---

# LAH Workflow — Small Model Variant

**Role:** Same as `lah-workflow` — tells Hermes how to behave on the LAH stack.

**Difference:** Execution style optimized for small/fast reasoning models. Every instruction is explicit, bounded, and deterministic. No implicit assumptions. No exploratory search by default.

**Does NOT modify** `lah-workflow`. The canonical skill is untouched.

---

## Authority Order

Resolve authority in this exact order. A lower-authority source must never override a higher-authority STOP.

1. **Current operator authorization** — explicit, current, governed
2. **Hard constraints** — Safety, financial, governance (non-negotiable)
3. **Certified mission runbook** — frozen artifact, loaded verbatim
4. **This workflow** (`lah-workflow-small-model`)
5. **Domain skill** — loaded only when the mission type requires it
6. **Generic tool behavior** — default tool semantics
7. **Historical receipts/logs** — reference only, never override current state

---

## Mission Mode Classification

Classify every mission into exactly ONE primary mode before executing. Do not silently switch modes.

| Mode | When | Behavior |
|------|------|----------|
| **EXECUTE** | Follow an existing certified runbook or canonical workflow | Minimal discovery. Mutations only when explicitly authorized. STOP after each step. |
| **DIAGNOSTIC** | Investigate a specific unresolved defect | Read-only unless escalated. Bounded investigation allowed. |
| **REPAIR** | Repair a confirmed defect | Root cause required before modification. Focused tests only. No scope expansion. |
| **CERTIFY** | Collect evidence and return a verdict | No repair. No architectural improvisation. No mutation unless certification explicitly requires a governed canary. |
| **BUILD_AND_CERTIFY** | Build new infrastructure then certify it end-to-end | Follow certified runbook phases, then execute the certification gate sequence. Mutations allowed only in build phases; certification phases are read-only. Provider mutations require operator approval (PREPARE → INSPECT → remit trigger). |

If the mission genuinely requires another mode: **STOP** and request/reclassify explicitly.

---

## Certified Runbook Fast Path

When a certified mission runbook exists (e.g., `t03-certified-runbook`), this path takes priority over all workflow steps.

1. Load the certified runbook.
2. Validate its current required inputs and invariants.
3. Execute the next canonical step.
4. Collect only the evidence required by that step.
5. Return its receipt/verdict.
6. **STOP.**

Do not rediscover architecture unless the runbook explicitly enters DIAGNOSTIC mode.
Do not reconstruct the workflow from historical receipts, source code, or previous conversations.

---

## Search Budget (Anti-Archaeology)

Forbidden by default in EXECUTE and CERTIFY modes:

- Broad recursive grep
- Broad find
- Filesystem archaeology
- Worktree archaeology
- Port scanning
- Route guessing
- Reading unrelated implementation files
- Historical receipt archaeology
- Repeated searches for the same fact
- Constructing alternate execution paths
- Ad-hoc provider calls when canonical wrappers exist

Allowed narrow lookups only when required to resolve a named canonical artifact.

| Mode | Max discovery actions |
|------|----------------------|
| EXECUTE | 1 narrow lookup before STOP/UNRESOLVED |
| CERTIFY | 2 narrow evidence-resolution actions |
| DIAGNOSTIC | Bounded investigation (mission-scoped) |
| REPAIR | Bounded investigation around confirmed root cause |

### Anti-Archaeology Counter Gate (BUILD_AND_CERTIFY)

When certifying a deterministic pipeline, enforce that all archaeology counters are zero:

- `grep_count` = 0
- `find_count` = 0
- `repo_search_count` = 0
- `swagger_discovery_count` = 0
- `web_research_count` = 0
- `ad_hoc_payload_count` = 0
- `ad_hoc_tracking_url_count` = 0
- `manual_provider_contract_resolution_count` = 0

If ANY counter is non-zero, the certification fails with `FAST_PATH_NO_ARCHAEOLOGY`. The pipeline must consume only already-known explicit authorities — no runtime discovery.

---

## Evidence Contract

Use the correct evidence class for each assertion. Lower-class evidence cannot override higher-class evidence.

| Class | Requirement |
|-------|-------------|
| **RUNTIME STATE** | Prefer canonical health/state/readback. Source code is not runtime evidence. |
| **PROVIDER STATE** | Prefer canonical provider reader. Historical receipt is not current provider state. |
| **AUTHORIZATION** | Require current governed authorization evidence. Do not infer from budget or historical approval. |
| **SAFETY** | Require current Supervisor/Governor canonical state. Do not infer Safety state from implementation source. |
| **DEPLOYMENT** | Require actual runtime provenance. Git HEAD alone may be insufficient unless runtime is proven to execute that checkout. |
| **CERTIFIED HISTORY** | Use frozen certified artifacts for workflow semantics/history, not for current mutable state. |

---

## Provider Canary Pattern (CERTIFY / BUILD_AND_CERTIFY)

When certification requires a real provider call (e.g., CREATE_PAUSED to prove the pipeline):

1. **PREPARE** — Build the governed packet (CLOE gouverné + preflight + normalisation zéro-write). Do NOT trigger the provider mutation.
2. **INSPECT** — Verify the packet against all contracts (selection, compile, invariants, drift, safety, attribution).
3. **REMIT** — Hand the trigger point (approve LAHB) to the operator. The operator explicitly approves the canary execution.

Hermes NEVER triggers provider mutations without explicit operator approval. The operator executes the canary, not Hermes.

Canary constraints:
- Zero-spend only (CREATE_PAUSED, never PLAY)
- New test identity (never reuse T04/T05 or production campaigns)
- Only canonical path: selection → validate → compile → COMPILE_READY → CREATE_PAUSED → variation materialization → provider readback → P6 certification → Safety verification → CREATED_PAUSED_READY_TO_PLAY
- Forbidden: manual payload, manual affiliate URL, ad-hoc launch script, grep/find during launch, Swagger rediscovery, manual variation construction outside canonical path

---

## Stop Contract

When any STOP condition triggers, halt immediately. Do not improvise fallback paths.

| Condition | Action |
|-----------|--------|
| UNKNOWN_CANONICAL_STATE | BLOCKED_SAFE |
| CANONICAL_ROUTE_UNAVAILABLE | BLOCKED_SAFE |
| AUTHORIZATION_UNCLEAR | BLOCKED_SAFE |
| INDEPENDENT_BLOCKER | STOP |
| CERTIFICATION_FAILURE | RECERTIFICATION_REQUIRED |
| NEW_DEFECT_DURING_EXECUTE | STOP — do NOT silently switch to REPAIR |
| NEW_DEFECT_DURING_CERTIFY | STOP — do NOT repair |
| REPAIR_REQUIRES_SCOPE_EXPANSION | STOP |

---

## LCM Context Discipline

Optimize for Hermes LCM long-running sessions.

- Prefer compact facts over raw log replay.
- Do not reload full historical missions when a certified receipt/runbook exists.
- Do not paste large source files into working context unnecessarily.
- Preserve canonical IDs/hashes/verdicts exactly.
- Summarize completed phases into compact state.
- Do not repeatedly reconstruct already-certified facts.
- Treat frozen certified artifacts as references.
- Treat current runtime state as fresh evidence.

Internal working state (keep compact):

```
MODE
OBJECTIVE
CURRENT_STEP
REQUIRED_FACTS
FACTS_OBSERVED
BLOCKER
ALLOWED_ACTION
NEXT_ACTION
```

---

## Gate Sequence (Compact)

The canonical gate sequence from `lah-workflow` is preserved. Small-model execution uses the same gates but with tighter bounds:

1. **Gate 0** — Repo routing preflight (MANDATORY). Load `lah-repo-router`.
2. **Gate 0.5** — Mission decomposition (MANDATORY). Load `mission-decomposer`.
3. **Gate 1** — CodeGraph (resolved repo only, maxFiles: 12).
4. **Gate 2** — AutoResearch (read-only). Use `research` skill if needed.
5. **Gate 3** — Superpowers Plan. Use `grill-me`/`grill-with-docs` only if design is fuzzy.
6. **Gate 3.5** — Design Authority Gate. Before TDD begins, compare pre-TDD delta against SPEC/plan. Fix any contradictions (wrong types, wrong field names, missing params, wrong formulas). Classify delta as PRE_TDD_UNCOMMITTED_DELTA. Do not start TDD until contradictions are resolved.
7. **Gate 4** — FastSafe Gate (15 checks). Batch when possible.
8. **Gate 5** — Implementation (varies by mission type).
9. **Gate 6** — Tests & Verification.
10. **Gate 7** — Operator Packet.
11. **Gate 8** — Commit (ciblé staging, mission-code prefix).
12. **Gate 9** — PR & Merge.
    - Merge contract: a tranche is NOT complete until PR is merged and remote main reflects the merge. PR creation alone is insufficient. After merge, fetch/reconcile main, run post-merge smoke, record merge SHA, then proceed.
    - API contract: GET endpoints MUST be observational and non-destructive. They MUST NOT mutate state, clear buffers, or delete data. If explicit clearing is needed, use a separate mutation boundary (e.g., DELETE endpoint).
13. **Gate 10** — Memory Lock.
14. **Gate 11** — Continuity JSON.

For mission-type specifics (READ_ONLY_AUDIT, DESIGN_ONLY, MIXED, PROMOTION_ONLY), consult `lah-workflow`. This small-model variant does not duplicate those branches — it defers to the canonical skill for mission-type branching logic.

---

## API Route Conventions

### Non-Destructive GET

GET endpoints MUST be observational and non-destructive:
- MUST NOT mutate state, clear buffers, delete data, or trigger side effects
- MUST return the same data on repeated identical calls
- If explicit state mutation is needed (e.g., clearing a cache), use a separate DELETE or POST endpoint as the mutation boundary

### POST/PUT/DELETE

POST/PUT endpoints may mutate state. DELETE endpoints are the explicit mutation boundary for clearing or removing resources.

---

### Pitfalls

#### ES Module `__dirname` Unavailable

When converting CommonJS modules to ES modules (`.mjs` or `"type": "module"`), `__dirname` is not defined by default. Node.js throws `ReferenceError: __dirname is not defined`.

**Fix:** Replace `__dirname` with `path.dirname(new URL(import.meta.url).pathname)`:

```js
const __dirname = path.dirname(new URL(import.meta.url).pathname);
```

This must be done BEFORE any `path.resolve(__dirname, ...)` calls.

#### Cross-Repo Path Resolution

When a module in repo A needs to import from repo B (e.g., `lah-openclaw-mvp` importing from `lah-brain`), relative paths must account for the actual filesystem layout, not the logical project structure.

**Rule:** Count directory levels from the importing file to the target file on disk. Use `path.resolve(__dirname, '<relative-path>')` and verify the resolved path exists before shipping.

**Common mistake:** Using `../../../` when the actual path requires `../../../../../` (5 levels). Always verify with `node -e "console.log(path.resolve('...'))"` after writing the path.

#### Fingerprint-Based Contract Stability

When computing file fingerprints for drift detection, use `SHA-256` of file content at load time. Do NOT use `grep`/`find`/`search` during runtime — fingerprints must be deterministic and computed from explicit authority sources only. If any fingerprint changes, the aggregate `CAMPAIGN_CONTRACT_FINGERPRINT` changes and Fast Path becomes ineligible.

#### Contract/Schema Names Are Not Function Names

Canonical LAH contract names (e.g., `lah_campaign_selection_v1`) are SCHEMA/CONTRACT identifiers, not necessarily function names. Do not declare the architecture missing merely because no standalone function literally matches the contract name. The actual implementations are:

- **Selection adapter** (`lah_campaign_selection_v1` contract): `selectCampaignFactoryStrategy()` in `campaign-factory-routing.js` — takes `{ campaign_count, format }`, returns `{ strategy, template_id, details }`
- **Compiler/draft builder** (`lah_campaign_compile_v1` contract): `buildCampaignCreationDraft()` in `campaign-creation-draft.js` — takes `{ source, campaigns, memory_context }`, returns a draft with `draft_id`, `draft_hash`, `status`
- **LAHB approval submission**: `POST /approvals/submit` with `action_type: CAMPAIGN_CREATE_PAUSED` — calls `submitApproval(db, actionType, payload, correlationId)` in `approval-queue.js`, returns `{ id, action_type, status: "PENDING", created_at }`

If the certified Fast Launch entrypoint cannot directly invoke these functions, return `BLOCKED_SELECTION_ADAPTER_WIRING` with exact evidence. Do not perform broad archaeology to find them.

#### CLI Boolean Flag Parsing

When a CLI flag like `--json` is parsed by a custom `parseCliFlags` function, the flag without an `=` value returns boolean `true`, not the string `"true"` or `"1"`. Any downstream check that compares against string values only (`=== "true" || === "1"`) will silently fail and the JSON output path will never be taken.

**Fix**: Always include `|| options.json === true` in the check, or normalize boolean flags to strings in `parseCliFlags`:

```js
// In parseCliFlags, normalize boolean flags:
acc[rawKey] = rawValue === undefined ? "true" : rawValue;
```

Or in the check:
```js
if (options.json === "true" || options.json === "1" || options.json === true) {
```

This pitfall applies to any Node.js CLI that uses a custom flag parser rather than a standard library like `yargs` or `commander`.

#### Provider Canary — Approval Existence Verification

Before executing a provider canary (CREATE_PAUSED), the approval must exist in the LAHB `approval_queue` and be in `PENDING` status. The approval is verified via `GET /approvals/:id` against the LAHB API (URL from `LAHB_URL` env var, default `https://leanframeworklab.com`).

**If the approval does not exist at the LAHB API, STOP immediately with `BLOCKED_APPROVAL_STATE`.** Do not create a new approval — the mission explicitly forbids creating another approval. Do not proceed to Step 2 (operator approval) or any execution step.

**Verification sequence:**
1. `GET /approvals/:approval_id` against LAHB API
2. Confirm `status === "PENDING"` (not `NOT_FOUND`, not `APPROVED`, not any other state)
3. If `NOT_FOUND` → `BLOCKED_APPROVAL_STATE`
4. If terminal status (`EXECUTED`, `FAILED`, `REJECTED`) → `BLOCKED_APPROVAL_STATE`
5. If `APPROVED` → STOP (already approved, do not re-approve)

#### Provider Canary — LAHB API Auth Fallback

If the LAHB API returns 401 (Unauthorized), the `LAHB_ADMIN_API_KEY` env var is not set or is invalid in the current session. Do NOT attempt to create a new approval or bypass the auth check.

**Fallback:** Use the certified mission resume packet as the fallback authority for approval state. The resume packet contains the `approval_id`, `operator_approved` status, and `authorization_state` from the last verified state.

**When to use this fallback:**
- LAHB API returns 401 or connection error
- `LAHB_ADMIN_API_KEY` is not set in the current environment
- The resume packet contains a valid `approval_id` with `operator_approved: false`

**When NOT to use this fallback:**
- The resume packet is missing or stale (fingerprint has changed)
- The approval_id in the resume packet is unknown or was never verified
- The mission requires fresh authorization evidence (per the Evidence Contract)

**Verification:** After using the resume packet fallback, confirm the resume packet's `canonical_context_fingerprint` matches the current `LAH_ARCHITECTURE_FINGERPRINT`. If they differ, the resume packet may be stale — do not use it.

#### Provider Canary — SESSION_LOADER_GAP (LAHB Credential Not Available)

The canonical credential-loading mechanism is `getSecret('LAHB_ADMIN_API_KEY')` from `secret-accessor.js`, which reads from `process.env.LAHB_ADMIN_API_KEY`. If this throws `Missing required secret: LAHB_ADMIN_API_KEY`, the credential may exist in the deploy `.env` file but not be loaded into the Hermes session's environment.

**Root cause:** The Hermes session and the deployed runtime container (`lah-openclaw-mvp`) have separate environment contexts. The container loads its `.env` file automatically; the Hermes session does not.

**Canonical secret source:** `/home/deploy/openclaw-runtime/lah-openclaw-mvp/.env` (the deploy `.env`).

**Smallest correction:** Source the deploy `.env` into the session environment before using the canonical loader:
```bash
cd /home/deploy/openclaw-runtime/lah-openclaw-mvp && set -a && . ./.env && set +a
```
Then retry the canonical loader (`getSecret('LAHB_ADMIN_API_KEY')`).

**Do NOT:**
- Duplicate the secret into another file
- Export it globally in the shell
- Introduce fallback credentials
- Inspect ambient environment as authority (use `getSecret`, not `process.env` directly)

**Verification:** After sourcing, confirm `getSecret('LAHB_ADMIN_API_KEY')` succeeds without throwing. Then proceed with the live approval readback.

#### Provider Canary — Approval State Machine

Approvals can be in multiple states beyond PENDING:
- **PENDING**: Awaiting operator approval. Usable for CREATE_PAUSED.
- **APPROVED**: Already approved by operator. Do not re-approve. STOP.
- **FAILED**: Execution attempt failed (e.g., NORMALIZATION_FAILED, OPENCLAW_REJECTED). The approval cannot be reused for a different format or after the root cause is not repaired.
- **EXECUTED**: Successfully executed. Do not re-execute.
- **REJECTED**: Explicitly rejected. Do not reuse.

**Critical pitfall:** A FAILED approval for the wrong format (e.g., banner when the canary scope is popunder) cannot be reused for a popunder CREATE_PAUSED. The approval's `payload.target.format` must match the canary's candidate format. If it doesn't match, the approval is effectively BLOCKED for the current canary scope.

**Verification:** After live readback, confirm `approval.status === "PENDING"` AND `approval.payload.target.format` matches the canary's candidate format. If either check fails, STOP with the appropriate blocker.

#### Provider Canary — FORMAT_MAP Normalization Gap

The ExoClick normalizer (`exoclick-normalizer.js`) maintains a `FORMAT_MAP` that maps catalog format names to ExoClick `advertiser_ad_type` IDs and media storage templates. If a format is catalog-selectable (has a template in `TEMPLATE_REGISTRY`) but missing from `FORMAT_MAP`, the normalizer rejects it with `NORMALIZATION_FAILED / BLOCKED_INVALID_FIELDS` at the OpenClaw boundary — after operator approval has already been granted.

#### Provider Canary — Capability Contract Gap (E2E Executability)

A format present in `FORMAT_MAP` is NOT sufficient for Fast Path eligibility. Each format must pass the full E2E capability chain:

```
SELECTABLE → CAMPAIGN_CREATABLE → CREATIVE_MATERIALIZABLE → PROVIDER_READBACK_CERTIFIABLE → P6_ELIGIBLE
```

The `FORMAT_END_TO_END_EXECUTABILITY_VERIFIED` invariant (in `capability-contract.js`) enforces this at three boundaries:
1. `validateFormatExecutable()` in `exoclick-normalizer.js` — blocks non-E2E formats from catalog selection
2. `selectCampaignFactoryStrategy()` in `campaign-factory-routing.js` — returns `ok: false` with `format_capability_error`
3. `validateDraftCampaigns()` in `campaign-creation-draft.js` — adds E2E error to draft validation

**Critical**: Banner, native, and video formats may be in `FORMAT_MAP` and `TEMPLATE_REGISTRY` but still lack `CREATIVE_MATERIALIZABLE` capability. They MUST be blocked at the compiler boundary before provider mutation. The planner must not know provider mechanics. See `governed-decision-gate` → `references/capability-contract-gate-pattern.md` for the full implementation pattern.

## Deployment Synchronization Pitfall (discovered 2026-08-19)

The source code repo (`/home/deploy/openclaw-runtime/`) and the deployed runtime (`/opt/lah-goes/runtime/lah-openclaw-mvp/`) can diverge. The source code may already contain a FORMAT_MAP fix while the deployed runtime still runs a stale version. This causes the canary to fail at execution even though the source code is correct.

**Detection**: After repairing FORMAT_MAP in the source, verify the deployed runtime's `exoclick-normalizer.js` matches. Compare `FORMAT_MAP` keys in both locations.

**Repair**: Update the deployed runtime's `exoclick-normalizer.js` and restart `lah-governed-operator-executor.service`:
```
sudo systemctl restart lah-governed-operator-executor.service
```

**Verification**: After restart, confirm the runtime's FORMAT_MAP includes all catalog-selectable formats before resubmitting a canary approval.

## Fresh Approval After FAILED State

When an execution attempt causes the current approval to become FAILED (e.g., due to a FORMAT_MAP normalization gap):

1. **Preserve** the failed approval as historical evidence (do not delete or reuse it).
2. **Repair** the demonstrated root cause (e.g., add missing FORMAT_MAP entry).
3. **Verify** the repair with focused regression tests.
4. **Restart** the affected runtime if required (`systemctl restart lah-governed-operator-executor.service`).
5. **Prepare** a fresh approval via `POST /approvals/submit` with the same experiment family, business intent, and financial envelope, with `play_authority: false`.
6. **STOP** at `OPERATOR_AUTHORIZATION_REQUIRED` — the fresh approval has a new ID and requires explicit operator consent before execution can proceed.

The fresh approval follows the same canonical path as the original: `APPROVAL → governed execution → CREATE_PAUSED → variation materialization → provider readback → P6 → Safety → CREATED_PAUSED_READY_TO_PLAY`. Never PLAY.

**Deployment Synchronization**: The source code repo (`/home/deploy/openclaw-runtime/`) may already contain the FORMAT_MAP fix while the deployed runtime (`/opt/lah-goes/runtime/lah-openclaw-mvp/`) runs a stale version. After repairing FORMAT_MAP in the source, always verify the deployed runtime's copy matches. If they diverge, update the deployed runtime's `exoclick-normalizer.js` and restart `lah-governed-operator-executor.service` via `systemctl restart lah-governed-operator-executor.service`.

**Reference**: See `references/format-catalog-reconciliation.md` for the full reconciliation pattern and cross-format test matrix.

## Fresh Approval After FAILED State

When an execution attempt causes the current approval to become FAILED (e.g., due to a FORMAT_MAP normalization gap):

1. **Preserve** the failed approval as historical evidence (do not delete or reuse it).
2. **Repair** the demonstrated root cause (e.g., add missing FORMAT_MAP entry).
3. **Verify** the repair with focused regression tests.
4. **Restart** the affected runtime if required (`systemctl restart lah-governed-operator-executor.service`).
5. **Prepare** a fresh approval via `POST /approvals/submit` with the same experiment family, business intent, and financial envelope, with `play_authority: false`.
6. **STOP** at `OPERATOR_AUTHORIZATION_REQUIRED` — the fresh approval has a new ID and requires explicit operator consent before execution can proceed.

The fresh approval follows the same canonical path as the original: `APPROVAL → governed execution → CREATE_PAUSED → provider readback → variation materialization → provider readback → P6 → Safety → CREATED_PAUSED_READY_TO_PLAY`. Never PLAY.

See `references/canary-execution-flow.md` for the full canary execution flow including the OpenClaw execution backend gap and anti-archaeology constraints.

#### Authority Duplication — lah-workflow vs lah-workflow-small-model

`lah-workflow` (827+ uses) and `lah-workflow-small-model` (63 uses) can both be loaded for LAH missions, causing authority duplication where `lah-workflow` overrides `lah-workflow-small-model` as the primary orchestrator.

**Symptom**: `lah-workflow` is selected as the primary workflow even for small/fast model missions, causing unnecessary discovery and archaeology.

**Fix**: Pin `lah-workflow-small-model` as the primary workflow using the Hermes curator CLI:

```
hermes curator pin lah-workflow-small-model
```

This sets `pinned: true` in `~/.hermes/skills/.usage.json`, preventing auto-transitions and ensuring `lah-workflow-small-model` is the primary orchestrator for governed LAH missions. `lah-workflow` remains as LEGACY_REFERENCE only — consult it for mission-type branching logic but never as the primary workflow.

**Verification**: Check `~/.hermes/skills/.usage.json` — `lah-workflow-small-model.pinned` must be `true` and `lah-workflow.pinned` should remain `false`.

#### Cold-Start Discovery Trap (CRITICAL)

When starting a new governed mission, the agent MUST follow the certified startup sequence and MUST NOT fall into the trap of broad discovery (searching for repos, .env files, capability-contract files, or provider endpoints manually).

**Symptom**: Agent does `codegraph init` on the home directory, runs `find` for .env files, searches for repo paths manually, or inspects arbitrary directories before routing. This wastes context, violates anti-archaeology, and produces stale or wrong results.

**Root cause**: The agent skips Gate 0 (lah-repo-router) and Gate 0.5 (mission decomposition), jumping directly to CodeGraph or file discovery.

**Fix**: The certified startup sequence is MANDATORY and non-negotiable:
1. **Gate 0**: Run `lah-repo-router` to resolve the canonical repo path. Accept AMBIGUOUS only if resolved by explicit_target.
2. **Gate 0.5**: Run mission decomposition if the mission exceeds ~50 lines.
3. **Gate 1**: Run CodeGraph ONLY on the routed repo path from Gate 0. Never initialize CodeGraph on a parent directory.
4. **Convergence Governor**: Activate before any discovery. Blocks ALL discovery until CodeGraph bootstrap completes.

If CodeGraph is missing on the routed repo, return `BLOCKED_CODEGRAPH_PROJECT_SCOPE`. Do NOT initialize arbitrary parent directories.

If the router returns AMBIGUOUS, resolve it by providing an explicit_target — do NOT accept AMBIGUOUS as a working state.

**Anti-pattern commands** (never run these at mission start):
- `codegraph init /home/deploy` — wrong scope, home directory is not a project
- `find /home/deploy -name ".env"` — env archaeology, use routed repo .env only
- `find /home/deploy -name "capability-contract*"` — broad search, use CodeGraph on routed repo
- `find /home/deploy -type d -name "openclaw-runtime"` — repo discovery, use lah-repo-router
- `grep -rn "LAHB_URL" /home/deploy` — broad search, use routed repo only

**Verification**: After Gate 0, the repo path must come from lah-repo-router output. After Gate 1, CodeGraph must be initialized and synced on that exact path. If either condition is not met, STOP with the appropriate BLOCKED code.

#### CodeGraph Mandatory Bootstrap — Startup Sequence Integration

CodeGraph bootstrap is a mandatory gate in the startup sequence for all governed LAH missions. It must run AFTER `LOAD_RESUME_PACKET` and BEFORE `IDENTIFY_NEXT_ACTION`.

**Startup sequence** (do not reorder):
1. LOAD_CONTEXT — Load Certified Architecture Context
2. VERIFY_FINGERPRINT — Compare `LAH_ARCHITECTURE_FINGERPRINT`
3. LOAD_RESUME_PACKET — Load mission resume packet
4. CODEGRAPH_BOOTSTRAP — Run `lah_context_resolve()` to check freshness, refresh if stale, load mission context packet
5. IDENTIFY_NEXT_ACTION — Determine next action from checkpoint or resume packet
6. EXECUTE — Execute the next action directly

**The `lah_context_resolve()` function** (in `scripts/startup-orchestrator.js`) is the semantic primitive for CodeGraph bootstrap. It:
- Checks CodeGraph freshness via `freshness-check.js`
- Refreshes the pack if stale/missing via `refresh-pack.js`
- Loads the mission context packet via `mission-context-pack.js`
- Returns a structured receipt: `{ phase: "CODEGRAPH_BOOTSTRAP", can_proceed: boolean, receipt?: object, error?: string }`

**Convergence Governor Gate**: The convergence governor blocks ALL discovery actions before CodeGraph bootstrap completes. The stop reason is `CODEGRAPH_BOOTSTRAP_REQUIRED`. After bootstrap completes, discovery is allowed (subject to other convergence checks).

**Activation**: Before any discovery, run the convergence governor's bootstrap gate activator:
```bash
node scripts/activate-bootstrap-gate.js <routed-repo-path>
```
This verifies CodeGraph is initialized on the routed repo path and activates the governor's discovery block.

**Pitfall**: If `lah_context_resolve()` returns `can_proceed: false`, the startup orchestrator must block mission execution and return `CODEGRAPH_BOOTSTRAP_FAILED`. Do not proceed to `IDENTIFY_NEXT_ACTION` or `EXECUTE` when bootstrap fails.

**Verification**: Run `node tests/codegraph-bootstrap-tests.js` from the lah-workflow-small-model directory. All 12 tests (T01-T12) must pass.

---

## Communication

Caveman levels by phase (loaded from `caveman` skill):

- **NORMAL** — routing preflight, arch/design/plan/risk (Gates 0-3)
- **LITE** — FastSafe, progress, operator tests, memory lock (Gates 4-6, 9.5, 10)
- **FULL** — tests, PR, merge, continuity (Gates 6-9, 11)

---

## Certified Architecture Context & Resume Packet (P1-P15)

### Startup Sequence (NEW)

Every mission startup now follows this order:

1. **LOAD_CONTEXT** — Load the Certified Architecture Context from `scripts/certified-architecture-context.js`
2. **VERIFY_FINGERPRINT** — Compare `LAH_ARCHITECTURE_FINGERPRINT` with stored value
3. **LOAD_RESUME_PACKET** — Load mission resume packet
4. **CODEGRAPH_BOOTSTRAP** — Run `lah_context_resolve()` to check CodeGraph freshness, refresh if stale, load mission context packet
5. **IDENTIFY_NEXT_ACTION** — Determine the next action from checkpoint or resume packet
6. **EXECUTE** — Execute the next action directly

Only if:
- Context is missing
- Context drift detected
- Blocking unknown not represented in resume packet

may ORIENT/DISCOVER begin.

### Certified Architecture Context (P1+P2)

A machine-readable canonical operational context registry captures certified facts including:

- canonical repo ownership
- LAHB approval authority and URL
- openclaw-runtime governed execution authority
- OPENCLAW_INTERNAL_URL (preferred execution URL)
- port 4000 (governed runtime)
- port 18789 (OpenClaw browser gateway, NOT campaign execution)
- campaign execution transport (LAHB → runtime → compiler → ExoClick)
- compiler, launcher, P6, ExoClick, tracking, Safety, format authorities
- approval execution seam
- canonical runtime services

Each fact includes: value, source_authority, fingerprint, certified_at, staleness_policy.

### Architecture Fingerprint (P3)

`LAH_ARCHITECTURE_FINGERPRINT` is a SHA-256 hash of all certified fact fingerprints combined deterministically.

At mission startup:
- If fingerprint unchanged → `CONTEXT_VALID`
- If changed → `CONTEXT_DRIFT` (only changed components may be rediscovered)

### Mission Resume Packet (P4+P8)

Persistent mission resume packets capture the exact state needed to continue a mission. Required fields:

- mission_id, mission_type, current_checkpoint, next_action
- known_facts, resolved_blockers, blocking_unknowns
- authorization_state, provider_state, approval_ids, campaign_ids, compiled_packet_ids
- canonical_context_fingerprint, forbidden_rediscovery
- last_verified_at

A mission continuation MUST load this packet before any discovery.

### No-Rediscovery Gate (P5+P6)

`CERTIFIED_FACT_REDISCOVERY_BLOCKED` blocks attempts to rediscover certified facts when:
- Context fingerprint unchanged AND
- Fact not stale AND
- No contradictory runtime evidence exists

Examples of BLOCKED actions:
- Re-probing what port 18789 is
- Searching OPENCLAW_INTERNAL_URL precedence again
- Finding LAHB URL again
- Searching campaign execution architecture again

Allowed startup checks (P6):
- One LAHB health/status check
- One governed runtime health check
- Exact approval readback
- Exact campaign readback when relevant
- Architecture fingerprint comparison

### Startup Budget (P7)

Maximum 3 orientation actions for a resumed known architecture mission:
1. Load context
2. Verify fingerprint
3. Read exact current object

Then execute. If more are attempted without CONTEXT_DRIFT → `STARTUP_ARCHAEOLOGY_BLOCKED`.

### Context Update Policy (P9)

After a successful repair, update ONLY impacted certified facts. Do NOT invalidate the entire architecture because one authority changed. Context evolves incrementally.

### Contradiction Handling (P10)

If runtime evidence contradicts certified context: emit `CERTIFIED_CONTEXT_CONTRADICTION` and investigate ONLY that fact.

### Convergence Governor Integration (P12)

Certified facts are automatically inserted into the Evidence Ledger as `RESOLVED_CANONICAL_FACT`. They cannot become blocking_unknowns again without contradictory evidence. This prevents post-compaction AND post-mission rediscovery.

### Startup Metrics (P14)

Added metrics for the certified startup sequence:
- `startup_tool_calls` — total tool calls during startup
- `certified_fact_rediscovery_attempts` — attempts to rediscover certified facts
- `certified_fact_rediscovery_blocked` — blocked rediscovery attempts
- `architecture_context_load_ms` — time to load architecture context
- `fingerprint_check_ms` — time to verify fingerprint
- `resume_packet_load_ms` — time to load resume packet
- `targeted_drift_discovery_calls` — targeted discovery calls after drift detection

Target resumed mission:
- `startup_tool_calls <= 3`
- `certified_fact_rediscovery_attempts = 0`
- Architecture rediscovery = 0

The Hermes Convergence Governor is a reusable component integrated into this workflow that prevents Hermes from continuing repository/runtime discovery after sufficient evidence exists.

### When It Activates

The convergence governor is active during all mission modes. It is especially critical in:

- **EXECUTE** — prevents unnecessary archaeology after the canonical route is found
- **DIAGNOSTIC** — bounds investigation and forces convergence when evidence is sufficient
- **BUILD_AND_CERTIFY** — enforces zero-archaeology during certification (P2 anti-archaeology gate)

### Components

| Component | P# | What It Does |
|-----------|-----|--------------|
| Command Fingerprinter | P1 | Normalizes tool/terminal actions into semantic fingerprints; detects equivalent commands |
| Discovery Budget | P2 | Per-phase ceilings with NEW_EVIDENCE_JUSTIFICATION requirement for overages |
| Evidence Ledger | P3 | Structured hypothesis tracking with blocking unknowns |
| Evidence Sufficiency Gate | P4 | `evaluateEvidenceSufficiency()` returns INSUFFICIENT/SUFFICIENT_TO_FORM_HYPOTHESIS/SUFFICIENT_TO_IMPLEMENT/SUFFICIENT_TO_REPORT |
| Loop Detector | P5 | EXACT_REPEAT_DETECTED and SEMANTIC_REPEAT_DETECTED with DISCOVERY_LOOP_DETECTED emission |
| Information Gain Scorer | P6 | HIGH/MEDIUM/LOW/ZERO scoring; FORCE_CONVERGENCE_REVIEW after 3 consecutive low-value actions |
| Side-Quest Detector | P7 | Maps discovery families to BLOCKING_UNKNOWN_ID; blocks unmapped exploration |
| Mutation Escalation Guard | P8 | Blocks unauthorized live/provider mutation probes in DIAGNOSTIC/PREPARE/READ_ONLY/OFFLINE_CERTIFICATION modes |
| Context Compaction Continuity | P9 | Persists and restores evidence ledger across context compaction; NO_POST_COMPACTION_REDISCOVERY invariant |
| Workflow State Machine | P10 | ORIENT→DISCOVER→HYPOTHESIS_READY→IMPLEMENT→VERIFY→REPORT; broad discovery prohibited at IMPLEMENT+ |
| Force Convergence | P11 | Automatic convergence check output when loop detected; transitions immediately if no blocking unknown |
| Stop Conditions | P12 | 5 explicit STOP DISCOVERY conditions that override generic "explore thoroughly" |
| Hard Safety Limit | P15 | 50 discovery action emergency ceiling with HARD_CONVERGENCE_TRIGGER |

### Integration Points

1. **Gate 1 (CodeGraph)** — After CodeGraph resolves the repo, the convergence governor records the canonical authority and evaluates whether further discovery is needed.
2. **Gate 4 (FastSafe)** — The governor's evidence sufficiency gate is consulted; if SUFFICIENT_TO_IMPLEMENT, broad discovery is frozen.
3. **Gate 5 (Implementation)** — The governor transitions to IMPLEMENT state; broad discovery is prohibited.
4. **Gate 6 (Tests & Verification)** — If verification exposes a new blocker, the governor allows narrow DISCOVER_BLOCKER transition (not full mission reset).
5. **Gate 9 (PR & Merge)** — The governor's metrics are included in the operator packet.
6. **Gate 10 (Memory Lock)** — The governor persists state for context compaction continuity.

### Key Principles

- **SEARCH FOR EVIDENCE, NOT FOR CONFIDENCE.** Do not keep searching simply because certainty could theoretically increase. Required threshold: enough evidence to make the next governed decision safely.
- **NO_POST_COMPACTION_REDISCOVERY** unless source fingerprint changed or evidence became stale.
- **STOP DISCOVERY** when canonical authority is identified AND exact defect is localized, OR when smallest repair is clear, OR when remaining unknowns do not affect repair correctness, OR when mission acceptance criteria can already be evaluated, OR when repeated searches provide zero new evidence.
- **MUTATION ESCALATION GUARD** — discovery loops must never silently escalate into mutation/probing.

### Metrics Exposed in Receipt

- `total_tool_calls`, `discovery_tool_calls`, `exact_repeat_count`, `semantic_repeat_count`, `zero_information_calls`, `broad_search_count`, `forced_convergence_count`, `side_quest_block_count`, `unauthorized_probe_block_count`, `post_compaction_rediscovery_count`, `hard_convergence_trigger_count`

### Target Metrics for Normal Repair Mission

- `exact_repeat_count` = 0
- `semantic_repeat_count` <= 1
- `zero_information_calls` <= 2
- `post_compaction_rediscovery_count` = 0

### P16 — Convergence Governor Enforcement Proof

The convergence governor must distinguish between three distinct enforcement states. These are NOT interchangeable.

| State | Meaning | Satisfies Runtime Enforcement? |
|-------|---------|-------------------------------|
| `RUNTIME_ENFORCED` | An actual tool-dispatch gate is active and blocking unauthorized actions | YES |
| `BEHAVIORAL_ONLY` | The LLM is following instructions voluntarily | NO |
| `NOT_ACTIVE` | No governor is active | NO |

**Machine-verifiable proof is required for RUNTIME_ENFORCED.** The proof must include:

- `enforcement_module` — Name of the enforcement module (e.g., `model_tools.py:handle_function_call`)
- `dispatch_boundary` — The exact dispatch boundary where enforcement occurs
- `session_id` or `mission_id` — The session this proof applies to
- `bootstrap_state` — The CodeGraph bootstrap state at proof time
- `blocked_action_count` — Count of actions blocked by the governor
- `runtime_gate_version` — Version of the runtime gate
- `runtime_gate_fingerprint` — SHA-256 fingerprint of the gate code

**Pitfall: BEHAVIORAL_ONLY ≠ RUNTIME_ENFORCED**

A governor that is merely followed voluntarily (BEHAVIORAL_ONLY) is NOT the same as a governor that blocks at the tool-dispatch boundary (RUNTIME_ENFORCED). Hermes must never report `CONVERGENCE_GOVERNOR_ACTIVE=true` or `isRuntimeEnforced=true` unless the actual enforcement boundary provides machine-verifiable evidence that the governor is active for the current mission.

**Pitfall: Objective Satisfaction Must Block Discovery at Runtime**

After the mission objective becomes OBJECTIVE_SATISFIED or OPERATOR_AUTHORIZATION_REQUIRED, discovery tools must be rejected at the runtime boundary with `BLOCKED_OBJECTIVE_ALREADY_SATISFIED`. This is not an LLM instruction — it is a tool-dispatch gate. The denial must include:

- `mission_state` — The current objective state
- `objective` — What the objective was
- `satisfied_by` — What achieved satisfaction
- `blocked_action` — The discovery action that was blocked
- `reason` — Why it was blocked

**Pitfall: Router Ambiguity Must Be Machine-Blocking**

The repo router must produce a machine-consumable result with `status` field. Allowed statuses: `RESOLVED`, `BLOCKED_AMBIGUOUS`, `BLOCKED_UNKNOWN`. If `status != RESOLVED`, mission execution must STOP. The LLM must not repair routing ambiguity by reading mapping files manually, guessing from directory names, or choosing the "most likely" repository. The router ambiguity gate must be enforced at the model_tools.py dispatch boundary, not at the instruction level.

**Verification:** Run the runtime enforcement regression corpus (`tests/runtime-enforcement-regression-tests.js`). All 16 tests (T01-T16) must pass.

### P16 — Convergence Governor Enforcement Proof

The convergence governor must distinguish between three distinct enforcement states. These are NOT interchangeable.

| State | Meaning | Satisfies Runtime Enforcement? |
|-------|---------|-------------------------------|
| `RUNTIME_ENFORCED` | An actual tool-dispatch gate is active and blocking unauthorized actions | YES |
| `BEHAVIORAL_ONLY` | The LLM is following instructions voluntarily | NO |
| `NOT_ACTIVE` | No governor is active | NO |

**Machine-verifiable proof is required for RUNTIME_ENFORCED.** The proof must include:

- `enforcement_module` — Name of the enforcement module (e.g., `model_tools.py:handle_function_call`)
- `dispatch_boundary` — The exact dispatch boundary where enforcement occurs
- `session_id` or `mission_id` — The session this proof applies to
- `bootstrap_state` — The CodeGraph bootstrap state at proof time
- `blocked_action_count` — Count of actions blocked by the governor
- `runtime_gate_version` — Version of the runtime gate
- `runtime_gate_fingerprint` — SHA-256 fingerprint of the gate code

**Pitfall: BEHAVIORAL_ONLY ≠ RUNTIME_ENFORCED**

A governor that is merely followed voluntarily (BEHAVIORAL_ONLY) is NOT the same as a governor that blocks at the tool-dispatch boundary (RUNTIME_ENFORCED). Hermes must never report `CONVERGENCE_GOVERNOR_ACTIVE=true` or `isRuntimeEnforced=true` unless the actual enforcement boundary provides machine-verifiable evidence that the governor is active for the current mission.

**Pitfall: Objective Satisfaction Must Block Discovery at Runtime**

After the mission objective becomes OBJECTIVE_SATISFIED or OPERATOR_AUTHORIZATION_REQUIRED, discovery tools must be rejected at the runtime boundary with `BLOCKED_OBJECTIVE_ALREADY_SATISFIED`. This is not an LLM instruction — it is a tool-dispatch gate. The denial must include:

- `mission_state` — The current objective state
- `objective` — What the objective was
- `satisfied_by` — What achieved satisfaction
- `blocked_action` — The discovery action that was blocked
- `reason` — Why it was blocked

**Pitfall: Router Ambiguity Must Be Machine-Blocking**

The repo router must produce a machine-consumable result with `status` field. Allowed statuses: `RESOLVED`, `BLOCKED_AMBIGUOUS`, `BLOCKED_UNKNOWN`. If `status != RESOLVED`, mission execution must STOP. The LLM must not repair routing ambiguity by reading mapping files manually, guessing from directory names, or choosing the "most likely" repository. The router ambiguity gate must be enforced at the model_tools.py dispatch boundary, not at the instruction level.

**Verification:** Run the runtime enforcement regression corpus (`node tests/runtime-enforcement-regression-tests.js`). All 16 tests (T01-T16) must pass.

#### Runtime Enforcement Boundaries

The convergence governor enforcement proof is consumed at the tool-dispatch boundary in `model_tools.py`. Three gates must all pass before a discovery tool is allowed:

1. **Router Ambiguity Gate** — `is_router_ambiguous()` must return False. If True, `BLOCKED_AMBIGUOUS_ROUTER` is returned.
2. **Objective Satisfaction Gate** — `is_discovery_blocked()` must return False. If True, `BLOCKED_OBJECTIVE_ALREADY_SATISFIED` is returned.
3. **Governor Truthfulness Gate** — The governor must have `RUNTIME_ENFORCED` proof. If `BEHAVIORAL_ONLY` or `NOT_ACTIVE`, `BLOCKED_GOVERNOR_NOT_ACTIVE` is returned.

These are runtime-enforced gates, not LLM instructions. The LLM cannot override them.

#### Frozen Dataclass Checkpoint Advancement (CRITICAL)

MissionCheckpoint is a frozen dataclass (`@dataclass(frozen=True)`). It cannot be mutated in place.

**Symptom:** `session_db.write_mission_checkpoint(updated_checkpoint)` raises `MissionCheckpointIntegrityError: checkpoint parent is not current checkpoint` or `TypeError: cannot set attribute`.

**Root cause:** Two distinct issues:
1. Attempting to mutate a frozen dataclass instance directly (e.g., `cp.phase = "P1"`).
2. Creating a new checkpoint without setting `parent_checkpoint_id` to the current checkpoint's ID, violating the deterministic lineage chain.

**Fix:** Use `dataclasses.replace()` to create a new frozen instance with updated fields, and set `parent_checkpoint_id` to the current checkpoint's `checkpoint_id`:

```python
from dataclasses import replace

cp = session_db.load_mission_checkpoint(mission_id)
cp_next = replace(cp,
    phase="P1_MISSION_BOUNDARY",
    completed_steps=cp.completed_steps + ["P0_PREFLIGHT"],
    pending_steps=[s for s in cp.pending_steps if s != "P1_MISSION_BOUNDARY"],
    next_action="P2_ACTION_BOUNDARY: ...",
    checkpoint_id=f"{mission_id}:checkpoint:1",
    parent_checkpoint_id=cp.checkpoint_id,
    state_version=CHECKPOINT_SCHEMA_VERSION,
)
session_db.write_mission_checkpoint(cp_next)
```

**Verification:** After writing, reload and confirm `checkpoint.parent_checkpoint_id` matches the previous checkpoint's ID and `phase` reflects the new phase.

#### hermes chat CLI Timeout for Ling Model Turns (PROVIDER QUIRK)

The `hermes chat` CLI defaults to a 60-second timeout, which is insufficient for Ling 3 Flash turns that involve real tool execution and context compilation. Turns may time out silently or raise `KeyboardInterrupt`.

**Symptom:** `hermes chat` commands exit with `KeyboardInterrupt` or return no output after 60 seconds, even though the Ling model is still reasoning.

**Fix:** Use `timeout 120` (or higher) on the shell command and keep prompts concise:

```bash
timeout 120 hermes chat -q "Short, focused question" --model inclusionai/ling-3.0-flash --provider zenmux 2>&1 | tail -20
```

For longer prompts, break them into shorter sub-questions. The Ling model responds faster on focused, single-topic prompts (19s for a 19-character prompt vs 90s+ for complex multi-part prompts).

**Verification:** A successful turn should return within 120s with a substantive response. If it consistently times out, check ZenMux provider health before assuming a model defect.

## References
## References
- `references/popunder-create-paused-approval-submission.md` — POPUNDER CREATE_PAUSED approval submission flow: routing resolution, LAHB API config, old vs new approval IDs, scope verification, and pitfalls (2026-08-20)
- `references/tracking-contract-audit-pattern.md` — 7-phase diagnostic for auditing affiliate tracking URL contracts against canonical LAH/CrakRevenue sources
- `references/t05-tracking-contract-repair-and-certification.md` — T05 repair workflow: FormData API quirk, macro mapping, repair pattern, postback attribution chain (exoclick-campaign-operations skill)
- `references/t05-forensic-analysis-workflow.md` — T05 forensic analysis template: 12-section business forensic structure, data unit normalization, Popunder CPM tracking semantics, zone decomposition, arm analysis, statistical negative evidence, economic value assessment, action gate
- `references/merge-contract-and-api-conventions.md` — merge contract (Gate 9) and non-destructive GET API convention
- `references/small-model-variant-pattern.md` — pattern for creating small-model variants of LAH Stack workflow skills
- `references/contract-drift-detection.md` — fingerprint-based contract drift detection pattern (P9): 10 authority fingerprints, CAMPAIGN_CONTRACT_FINGERPRINT aggregate, drift status values, no-archaeology constraint
- `references/provider-canary-pattern.md` — zero-spend CREATE_PAUSED canary pattern (P9-P12): canonical flow, operator approval gate, P6 verification gates, constraints
- `references/format-catalog-reconciliation.md` — format catalog/runtime reconciliation pattern: FORMAT_MAP gap detection, cross-format test matrix, SELECTABLE_FORMAT_MUST_BE_RUNTIME_EXECUTABLE invariant
- `references/format-map-deployment-drift.md` — concrete FORMAT_MAP divergence pattern: source vs deployed runtime diff, detection command, repair steps, impact on canary execution
- `references/canary-execution-flow.md` — canary execution flow: anti-archaeology constraints, OpenClaw execution backend gap, approval prerequisites, timing targets, required verdicts
- `references/convergence-governor-pattern.md` — reusable convergence governor (P1-P16): command fingerprinting, discovery budgets, evidence sufficiency, loop detection, force convergence, mutation guard, context compaction continuity, workflow state machine, hard safety limits, enforcement proof (P16)
- `references/behavioral-certification-pattern.md` — SNAPSHOT -> DECISION -> RECEIPT pattern for certifying agent behavior on campaign analysis tasks, with forbidden-action constraints
- `references/campaign-snapshot-data-quality-authority.md` — shared rule: campaign snapshot data_quality.status as campaign fact authority, PASS/PARTIAL/FAIL semantics, no re-query guarantees
- `references/compression-canary-test-pattern.md` — pattern for certifying that real context compression preserves mission continuity via MISSION_STATE + CompressionRecoveryGate
- `references/real-runtime-benchmark-pattern.md` — REAL_RUNTIME_MEASURE_ONLY benchmark execution pattern: telemetry surface, per-turn metrics capture via step_callback, durable mission checkpoint system requirements, common pitfalls (compiled_context_tokens=0, SessionDB binding, iteration budget)
- `references/canary-v3-real-mission-readiness-gate.md` — V3 canary execution pattern: durable mission setup with SessionDB, agent construction with canonical ZenMux credential resolution, DurableBenchmarkHarness turn execution, P1/P2/P3 metrics collection, receipt format, and known pitfalls
- `references/crakrevenue-postback-and-attribution-join.md` — CrakRevenue postback chain, outbound/inbound attribution join, event vs paid conversion gap (session 2026-08-20)
