# OpenClaw Runtime — Mission Execution Pattern

## Scope

This reference covers the **openclaw-runtime** repo (`/home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp`), which has a different workflow than `lah-stack-tools` or `cartelogic-v2`.

Key differences:
- **PR-based workflow** for all changes (MEMORY_LOCK_PR_POLICY_V1)
- **Admin merge with exception receipts** when CI can't run
- **Mission Runtime API extension pattern** (core module → API function → CLI command → adapter)
- **Memory lock via dedicated PR** not direct push
- **CodeGraph** with explicit projectPath

---

## 1. CodeGraph Analysis

Always use explicit `projectPath`:

```javascript
codegraph_explore({
  query: "symbols or files to explore",
  projectPath: "/home/deploy/lah-stack-repos/openclaw-runtime/lah-openclaw-mvp",
  maxFiles: 15
});
```

---

## 2. Governed Standard Sequence

The mission execution sequence for this repo:

```
1. CodeGraph analysis (projectPath explicit)
2. Implementation (direct or via sub-agents)
3. Tests (node:test, .mjs files, --test-concurrency=1)
4. Operator packet (docs/operator/)
5. Commit + push to feature branch
6. Open PR
7. Merge (standard PR) or admin merge with receipt
8. Post-merge verify
9. Memory lock via dedicated PR (per MEMORY_LOCK_PR_POLICY_V1)
```

---

## 3. Mission Runtime API Extension Pattern

When adding new capabilities to the Mission Runtime API, use this layered architecture:

```
Layer 1: Core generators (pure functions, no side effects)
  scripts/pr-autopilot/core/<feature>.mjs
  → buildXxx(), validateXxx() — pure data transformations

Layer 2: File writer (disk I/O, uses ArtifactStore)
  scripts/pr-autopilot/core/<feature>.mjs
  → writeXxxArtifacts() — creates files in Mission Runtime layout

Layer 3: API function (fail-closed, standard response shape)
  scripts/mission-runtime/api.mjs
  → createXxx(input) — validates, delegates to Layer 2, returns { ok, paths, metadata }

Layer 4: API contract (schemas, exit codes)
  scripts/mission-runtime/api-contract.mjs
  → Contract entry with input/output schemas

Layer 5: CLI command (user-facing)
  scripts/pr-autopilot.mjs
  → runXxx({ context, output, jsonMode }) — parses args, calls Layer 3 or 2

Layer 6: Hermes adapter method (agent-facing)
  scripts/hermes-adapter.mjs
  → generateXxx(missionId, options) — delegates to Layer 3, returns paths + metadata
```

### Import rules

- Core generators import from `../../mission-runtime/artifact-store.mjs` and `../../mission-runtime/api-contract.mjs`
- API functions import from `../pr-autopilot/core/<feature>.mjs`
- CLI imports from `./pr-autopilot/core/<feature>.mjs`
- Hermes adapter imports from `./mission-runtime/api.mjs`

### Response shape convention

```javascript
// Success
return { ok: true, exit_code: 0, paths: { ... }, metadata: { ... } };

// Failure
return { ok: false, exit_code: 65, error: { code: 'INVALID_INPUT', message: '...', details: { ... } } };
```

---

## 4. PR Workflow

### Standard PR merge

```bash
git checkout -b feat/my-feature
git push origin feat/my-feature
gh pr create --base main --head feat/my-feature --title "..." --body "..."
gh pr checks <N> --watch
gh pr merge <N> --squash --delete-branch
```

### LOCAL_CI_VERIFIED merge (when GitHub Actions is unavailable)

When GitHub Actions is unavailable due to billing/quota exhaustion, use LOCAL_CI_VERIFIED_MERGE_POLICY_V1 instead of waiting for CI.

**Conditions (all must be true):**
1. GitHub Actions explicitly unavailable (state the cause: billing exhaustion until date)
2. All targeted local tests pass
3. `git diff --check` passes
4. A LOCAL_CI_VERIFIED receipt is created BEFORE merge
5. No forbidden changes (live action, provider write, execute, campaign, scheduler)
6. Post-merge verify is planned
7. Memory lock still goes through PR when possible

**Procedure:**

```bash
# 1. Create feature branch from main HEAD
git branch feat/my-feature main_sha  # or from current main worktree
git checkout feat/my-feature

# 2. Implement, test, commit
git add <intentional-files>
git commit -m "feat: add CLOE feature description"
git push origin feat/my-feature

# 3. Open PR
gh pr create --base main --head feat/my-feature --title "feat: ..." --body "..."

# 4. Create LOCAL_CI_VERIFIED receipt at docs/operator/receipts/
#    Receipt must include: policy name, reason (billing exhaustion), test results,
#    changed files, forbidden change scan, conditions met checklist, safety note.

# 5. Merge via main worktree (since main is used by /tmp/openclaw-main-merge)
cd /tmp/openclaw-main-merge/lah-openclaw-mvp
git pull origin main
git merge feat/my-feature --no-edit
git push origin main

# 6. Post-merge verify on merged main
node --test test/<targeted-tests>
git diff --check
git status --short

# 7. Comment on PR with merge record, receipt ref, post-merge results
gh pr comment <N> --body "## Merge Record\n\nMerged under: LOCAL_CI_VERIFIED_MERGE_POLICY_V1\n..."
```

**Willow:** The `gh pr merge` command works even when GH Actions is unavailable — it skips the GH Actions check requirement on push. If the push itself is blocked by branch protection, the admin merge exception path below applies instead. But the receipt is the canonical merge record either way.

**Receipt format** (store at `docs/operator/receipts/local-ci-verified-<feature>-<timestamp>.json`):

```json
{
  "receipt_id": "receipt-local-ci-verified-<feature>-<timestamp>",
  "receipt_type": "local_ci_verified_merge",
  "policy": "LOCAL_CI_VERIFIED_MERGE_POLICY_V1",
  "created_before_merge": true,
  "timestamp": "<ISO-8601>",
  "reason": "github_actions_unavailable_billing_exhausted",
  "reason_description": "GitHub Actions unavailable because billing is exhausted. Local targeted tests passed.",
  "github_actions_status": { "available": false, "cause": "billing_exhaustion" },
  "local_validation": {
    "targeted_tests_passed": true,
    "targeted_test_commands": ["node --test test/..."],
    "targeted_test_results": { "test_name": "PASS (N/M)" },
    "git_diff_check_passed": true,
    "json_receipt_parse_passed": true
  },
  "conditions_met": { /* all booleans true */ },
  "forbidden_change_scan": { /* all booleans false */ },
  "changed_files": ["path/to/files..."],
  "_safety": { "note": "..." }
}
```

### Memory lock PR (per MEMORY_LOCK_PR_POLICY_V1)

Allowed paths: `lah-openclaw-mvp/docs/mcporter/**`, `lah-openclaw-mvp/docs/operator/**`

```bash
git checkout -b docs/memory-lock-<name>
git add docs/mcporter/<continuity-file>.json
git commit -m "docs: memory lock <name>"
git push origin docs/memory-lock-<name>
gh pr create --base main --head docs/memory-lock-<name>
gh pr checks <N> --watch
gh pr merge <N> --squash --delete-branch
```

### Admin merge exception (when CI can't run)

Required when branch protection blocks merge (e.g. `[skip ci]` commit prevents CI trigger).

**Conditions (must all be true):**
1. CI is green OR failure is docs-only / non-applicable
2. Operator gives explicit approval
3. The reason is one of the 4 approved categories (ADMIN_MERGE_EXCEPTION_POLICY_V1 §4)
4. No forbidden change types

**Procedure:**

```bash
# 1. Create receipt before merge
mkdir -p receipts
[write receipt JSON to file]

# 2. Try admin merge
gh pr merge <N> --admin --squash --delete-branch

# IF stuck ("Merge already in progress"):
git fetch origin feat/my-branch
git checkout main
git merge --squash origin/feat/my-branch
git commit -m "feat: description (#<N>)"
git push origin main
gh pr close <N>

# 3. Post-merge verify
git pull --ff-only
# rerun tests, git diff --check, git status

# 4. Commit receipt
git add receipts/
git commit -m "docs: add admin merge exception receipt for PR #<N> [skip ci]"
git push origin main
```

---

## 5. Tests

```bash
# Run specific test file
node --test --test-concurrency=1 test/my-test.test.mjs

# Run full suite
node --test --test-concurrency=1 test/mission-runtime.test.mjs test/mission-runtime-api.test.mjs test/mission-runtime-api-contract.test.mjs test/mission-runtime-mcp-adapter.test.mjs test/hermes-adapter.test.mjs test/mission-runtime-api-usage-harding.test.mjs test/repair-packet.test.mjs

# Duplication scan
node scripts/check-mission-runtime-duplication.mjs

# Whitespace check
git diff --check
```

---

## 6. Key Directories

| Path | Purpose |
|---|---|
| `scripts/mission-runtime/` | Shared Mission Runtime API (11 modules) |
| `scripts/pr-autopilot/` | PR Autopilot CLI and core modules |
| `scripts/pr-autopilot/core/` | Feature modules (diagnostics, repair-packet, etc.) |
| `scripts/hermes-adapter.mjs` | Hermes-safe read-only adapter |
| `test/` | Tests (.mjs files, node:test) |
| `docs/governance/` | Governance policies |
| `docs/operator/` | Operator packets (mission closeout docs) |
| `docs/mcporter/` | Memory lock continuity files |
| `docs/runbooks/` | Usage runbooks |
| `tmp/pr-autopilot/` | PR Autopilot runtime artifacts |
| `receipts/` | Admin merge exception receipts |

---

## 9. CLOE Cognitive Architecture Mission Pattern

A recurring class of mission for adding new cognitive capabilities to CLOE (e.g. CLOE_COGNITIVE_BEHAVIOR_MODEL_V1, CLOE_MEMORY_ARCHITECTURE_V1, CLOE_MEMORY_RUNTIME_V1). Each follows the same deliverable pattern.

### 3-phase CLOE implementation sequence

CLOE missions build on each other in a defined dependency order. Subsequent missions may reference or import from previous ones:

| Phase | Mission | Module path | Purpose |
|-------|---------|-------------|---------|
| 1 | Behavior Model | `src/cognitive/cloe-behavior-capabilities.js` | Define how CLOE reasons and behaves: principles, domains, response modes |
| 2 | Memory Architecture | `src/cognitive/cloe-memory-layers.js` | Define what memory exists: layers, rules, epistemology, source of truth hierarchy |
| 3 | Memory Runtime | `src/cognitive/cloe-memory-*.js` (schemas/backends/runtime) | Implement governed memory interface with pluggable backends |

Phase N may import from Phase N-1 (e.g. memory schemas import layer IDs from memory layers). Do NOT skip phases — each provides foundation for the next.

### Deliverable checklist

| # | Deliverable | Path pattern | Required |
|---|-------------|-------------|----------|
| 1 | Architecture document | `docs/architecture/CLOE_<FEATURE>_V1.md` | Yes |
| 2 | Frozen data registry | `src/cognitive/cloe-<feature>.js` | Yes |
| 3 | Optional context integration | `src/cognitive/<feature>-integration.js` | If safe+bounded |
| 4 | Test suite | `test/cloe-<feature>.test.js` | Yes |
| 5 | Operator packet | `docs/operator/CLOE_<FEATURE>_V1_OPERATOR_PACKET.md` | Yes |
| 6 | LOCAL_CI_VERIFIED receipt | `docs/operator/receipts/local-ci-verified-<feature>-<date>.json` | Yes |
| 7 | Continuity JSON | `docs/mcporter/CLOE_<FEATURE>_V1_CONTINUITY.json` | Yes |
| 8 | Commit + PR + merge | `feat: add CLOE <feature>` | Yes |
| 9 | Post-merge verify | Rerun tests on merged main | Yes |
| 10 | Memory lock | Update memory with mission completion | Yes |

### Architecture document structure pattern

Each architecture doc follows a consistent structure:
- Status and scope header
- Table of contents
- Why the spec exists (problem statement)
- Architectural relationships to existing docs
- Core definitions (principles, domains, layers, etc.)
- For each defined entity: expected behavior, prohibited behavior, governance
- Relationship to CLOE behavior model or other CLOE layers
- Explicit position on controversial/ambiguous topics (e.g. mem0 NOT wired)
- Current state vs target state

### Frozen registry pattern (src/cognitive/)

Each data registry module follows this structure:

```javascript
import { deepFreeze } from './deep-freeze.js'; // or inline helper

const DATA = deepFreeze([ /* frozen entries */ ]);

export function getter() {
  return Object.freeze({
    version: '1.0.0',
    data: DATA,
    readOnly: true,
    providerWrite: false,
    filesystemWrite: false,
    gateMutation: false,
    campaignMutation: false,
    schedulerMutation: false
  });
}

export function summarize(mode = 'brief') {
  // Returns bounded string guaranteed < 6000 chars in full mode
}
```

Key rules:
- `deepFreeze` must freeze nested objects, not just the array
- Safety flags (`readOnly`, `providerWrite`, etc.) must be explicitly set
- Summary formatter must accept `'brief'` and `'full'` modes
- Brief mode must be < 1500 chars; full mode must be < 6000 chars
- No mem0 dependency, no filesystem access, no provider calls

### Test suite pattern

Each test suite covers:
1. Module structure (frozen, data-only, read-only)
2. All expected entries present (by ID)
3. Required fields on each entry
4. Safety flags (no provider write, no execute, no mutation)
5. Summary bounded
6. No dangerous operations
7. Version and metadata

### Branch and commit conventions

| Convention | Value |
|-----------|-------|
| Branch name | `feat/cloe-<feature-name>-v1` |
| Commit message | `feat: add CLOE <feature description>` |
| PR title | Same as commit message |
| Merge method | Fast-forward (preferred) or squash |
| Merge record | Comment on PR with full validation record |

### Common pitfalls

- **Main used by worktree**: openclaw-runtime main is checked out at `/tmp/openclaw-main-merge`. To merge, `cd /tmp/openclaw-main-merge && git pull origin feat/my-branch && git merge feat/my-branch --no-edit && git push origin main`. You cannot `git checkout main` in the dev worktree.
- **Stale stash before branch switch**: stash before switching branches, pop after returning. Forgetting the pop leaves work stranded.
- **Multiple worktrees**: the openclaw-runtime repo has at least two worktrees at `/home/deploy/lah-stack-repos/openclaw-runtime` (dev worktree) and `/tmp/openclaw-main-merge` (main worktree). A previous stash from the dev worktree may reappear on the new branch. Use `git stash drop` after popping if no longer needed.
- **`gh push` bypass warning**: `git push origin main` when GH Actions is unavailable produces `remote: Bypassed rule violations for refs/heads/main` — this is expected and normal under LOCAL_CI_VERIFIED policy. The receipt is the canonical merge record, not the CI status.
- **Memory full on `memory()` call**: OpenClaw memory has a 2200-char limit. The OpenClaw entry must be consolidated to fit new mission completions. If `memory(action='add')` fails with "would exceed the limit", use `action='replace'` on the existing OpenClaw entry to consolidate shorter.
- **Fast-forward vs squash merge**: fast-forward keeps history clean. Only squash when the PR contains multiple granular commits that don't make sense independently.
- **`gh pr close` on already-merged PR**: returns `Pull request can't be closed because it was already merged` — this is a non-fatal error. The merge succeeded; the close is redundant. Use `gh pr comment` instead for the merge record.
- **JavaScript shorthand property bug with snake_case variables**: when building response objects, `writes_allowed` (shorthand) looks for a variable literally named `writes_allowed`. If the source variable is `writesAllowed` (camelCase), this produces a ReferenceError at runtime. Always use explicit key-value pairs (`writes_allowed: writesAllowed`) when the key is snake_case and the value is camelCase.

---

## 10. CLOE Memory Runtime Implementation Pattern (Phase 3)

When implementing a governed memory runtime for CLOE (e.g. CLOE_MEMORY_RUNTIME_V1), follow this 3-module triad:

### Module 1: Memory Schemas — `src/cognitive/cloe-memory-schemas.js`

Pure validators with no dependencies beyond the layer registry. Exports:

| Export | Purpose |
|--------|---------|
| `validateLayer(id)` | Check layer ID is valid per architecture |
| `validateMemoryRecordForLayer(record, layer)` | Record shape + data allowed in given layer |
| `validateMemoryQuery(query)` | Query structure (keywords, tags, limit, threshold) |
| `validateGovernanceMetadata(meta)` | Write approval metadata (operator_approved, approved_by, approval_timestamp) |
| `validateMemoryRecordSafety(record)` | Reject secrets (api_key, token, password) and dangerous markers (provider_write, live_action) |
| `createEpistemology(status, opts)` | Build epistemology block with source, timestamp, layer |
| `createSafetyEnvelope()` | Standard safety envelope (read_only, provider_write, execute_called, etc.) |
| `isSecretField(key)` / `hasDangerousMarker(val)` | Low-level pattern checks |

Key rules:
- No schema library — plain functions returning `{ ok, reason }`
- `validateMemoryQuery` accepts `threshold` field (0-1) for future semantic search — does not use it yet in V1
- `validateGovernanceMetadata` requires `operator_approved: true` for write ops — this is the governance gate
- `validateTtlMetadata` is optional (TTL is not enforced in V1); validates `session/turn/explicit/persistent/stable` type strings

### Module 2: Backend Factory — `src/cognitive/cloe-memory-backends.js`

Pluggable backends implementing a standard interface. Each backend provides:

```javascript
{
  id, name, description,          // Identity
  isProduction, supportsWrites,    // Capability flags
  mem0Enabled,                     // Always false in V1
  add(record),                     // Write — blocked by default
  search(query),                   // Read — deterministic keyword+tag match
  get(id),                         // Read single record
  getAll(filter),                  // Read all (optionally filtered by layer)
  update(input),                   // Write — requires governance metadata
  delete(input),                   // Write — requires governance metadata
  history(filter)                  // Event log
}
```

**Built-in backends (3):**

| Backend | Production | Writes | Use case |
|---------|-----------|--------|----------|
| `null_backend` | YES | NEVER | Safe default — always returns empty/unavailable |
| `static_backend` | YES | NEVER | Data-only read backend for tests/docs — accepts `records: [...]` in config |
| `in_memory_test_backend` | NO | Configurable | Only for tests; pass `writes_allowed: true` in config to enable add |

**Every method returns a structured envelope:**

```javascript
{
  ok,                  // boolean
  mode,                // 'read_only' | 'write'
  read_only,           // boolean
  writes_allowed,      // boolean
  layer,               // string|null
  epistemology,        // { status, source, stored_at, layer }
  result / results,    // single record | array of records
  total,               // number (for search/getAll)
  error,               // string|null
  provenance,          // { backend, source, id, ... }
  safety,              // createSafetyEnvelope()
  history              // event array
}
```

**Registration:**

```javascript
import { registerBackend, createBackend, listBackends, hasBackend, getMem0BackendStatus } from './cloe-memory-backends.js';

registerBackend('null_backend', createNullBackend); // done at module load
const backend = createBackend('null_backend', { defaultLayer: 'operational_memory' });
```

**mem0 placeholder** — `getMem0BackendStatus()` returns `{ installed: false, enabled: false, wired: false, planned: true }`. No mem0 client exists yet.

### Module 3: Runtime API — `src/cognitive/cloe-memory-runtime.js`

Thin wrapper around a backend that adds governance checks and structured envelopes.

```javascript
import { createCloeMemoryRuntime } from './cloe-memory-runtime.js';

// Safe default (null backend, read-only)
const rt = createCloeMemoryRuntime();

// With test backend
const rt = createCloeMemoryRuntime({
  backend: 'in_memory_test_backend',
  backendConfig: { writes_allowed: true },
  defaultLayer: 'operational_memory'
});
```

Exposes: `addMemory`, `searchMemory`, `getMemory`, `getAllMemory`, `updateMemory`, `deleteMemory`, `history`, `getStatus`.

Governance enforcement in the runtime:
- `addMemory` validates via `validateMemoryRecordForLayer` before delegating to backend
- `updateMemory` / `deleteMemory` require governance metadata (operator_approved, approved_by, approval_timestamp) — blocked by default
- `searchMemory` / `getMemory` / `getAllMemory` / `history` pass through without governance checks (read-only)
- Every response includes a safety envelope

### Common pitfalls

- **Shorthand property bug**: `writes_allowed,` inside an object literal references a variable named `writes_allowed`. If your variable is `writesAllowed` (camelCase from `config.writes_allowed`), use `writes_allowed: writesAllowed` explicitly.
- **`node -e` ESM imports need `./` prefix**: `import('./src/cognitive/...')` works, but `import('src/cognitive/...')` fails because bare specifiers are not relative paths. Always use `./` for local imports in eval.
- **Default backend fallback**: `createCloeMemoryRuntime()` with no args defaults to `null_backend`. If an unknown backend name is passed, it falls back to `null_backend` silently — always check `getStatus().backend.name` to confirm the expected backend loaded.

---

## 11. CLOE Operational Assistant Stack Implementation Pattern

A recurring mission that builds 5 interdependent slices in one governed mission, each independently testable, collectively forming the "assistant memory stack" that makes CLOE useful as an operational assistant.

### Architecture

The stack builds bottom-up: Operational Memory → Workflow Memory → Session Continuity → Next Move Engine → Self Knowledge. Each slice imports from the previous layer or accepts it via dependency injection (DI).

### Execution strategy: parallel slice creation with delegate_task

Each slice is a self-contained module + test + doc. Create slices in parallel (max 2-3 per batch) using `delegate_task`. Each sub-agent receives full context about existing modules, expected API shapes, envelope structures, and output file paths. Context completeness is critical — sub-agents have zero access to the parent session's memory.

### Slice interface contracts

**Operational Memory** (`createCloeOperationalMemory`): 5 record types (active_mode, temporary_instruction, active_constraint, blocker, environment_condition). Fields: id, type, scope (system|session|task|agent|user), value, reason, source, provenance, priority (0-100), expires_at/ttl, status (active|expired|superseded|revoked), governance_metadata, epistemology. Add requires governance_metadata (operator_approved, approved_by, approval_timestamp). Auto-supersedes conflicts on same type+scope. Accepts `reference_time` for deterministic test expiration.

**Workflow Memory** (`createCloeWorkflowMemory`): 7 workflow records (pr_standard active, memory_lock_pr active, local_ci_verify active, admin_merge_exception experimental, post_merge_verify active, receipt_processing active, deprecated_workflow_placeholder deprecated). Keyword scoring against `when_to_use` fields for situation matching.

**Session Continuity** (`createCloeSessionContinuity`): Two modes — static snapshot (pre-built object for tests) or live (derives from operational memory + workflow memory via DI). Never invents missing state — absent fields return `null`/`[]`.

**Next Move Engine** (`createCloeNextMoveEngine`): Composite scoring = impact(35%) + inverted_risk(25%) + inverted_cost(20%) + inverted_dependency(20%). Blockers prioritize stabilization over feature work. GitHub Actions unavailable → local_ci_verify candidate. Continuity required → memory_lock_pr candidate.

**Self Knowledge** (`createCloeSelfKnowledge`): Answers 8 question types (capabilities, identity, memory, mem0_status, limits, workflows, next_move, blocked). All envelopes report `read_only=true, provider_write=false`. mem0 explicitly NOT wired. Limits include `tool_invention` prohibition.

### Testing patterns

Each slice test: module exports, factory defaults, each public method (valid + fail-closed), safety envelope verification, no provider write / no execute / no secret leakage. Run all slice tests together, then existing suites for regression.

---

## 12. CLOE Assistant Capabilities Batch Pattern

After the operational assistant stack (section 11) is complete, subsequent capability batches add more slices following the same parallel-execution pattern. Each batch is 3 slices, independently testable, built on the stack's foundation.

### Batch A: Tool Awareness + Governed Action Handoff + Decision Memory

**Slice 7 — Tool Awareness** (`src/cognitive/cloe-tool-awareness.js`):
13-tool registry with status categories (available|unavailable|degraded|forbidden|unknown). Methods: `listTools()`, `findToolByTask(task)`, `recommendToolForTask(task)`, `explainUnavailability(toolId)`, `getToolById(toolId)`. Never claims execution ability if only handoff is allowed. Forbidden tools include safety constraints explaining why. Each tool has: tool_id, name, category, status, allowed_actions, forbidden_actions, required_approval, best_for, not_for, known_failure_modes, safety_constraints, source_refs, last_known_state.

**Slice 8 — Governed Action Handoff** (`src/cognitive/cloe-governed-action-handoff.js`):
8 handoff types: codex_prompt, hermes_prompt, operator_packet, receipt_draft, validation_plan, rollback_note, memory_lock_plan, pr_autopilot_packet. Each handoff has: handoff_id, target_agent_or_tool, objective, repo_path, context_summary, constraints, allowed_actions, forbidden_actions, validation_commands, expected_outputs, approval_required, risk_tier, rollback_note, provenance, no_execution: true. Fail-closed: provider_write, live_gate, campaign, scheduler, admin_merge, bypass_governance, auto_execute all rejected. Approval required: admin_merge_exception, memory_lock_plan, pr_autopilot_packet, operator_packet.

**Slice 9 — Decision Memory** (`src/cognitive/cloe-decision-memory.js`):
20-field decision records with governance-gated writes. Fields: decision_id, title, decision, reason, alternatives_considered, alternatives_rejected, scope (global|repo|mission|workflow|architecture|policy), status (active|superseded|revoked|proposed), priority, source_refs, provenance, governance_metadata, decided_at, supersedes, expires_at, impact, risks, related_policies, related_workflows. Methods: `addDecision(record)` (governance required), `searchDecisions(query)`, `getDecision(id)`, `listActiveDecisions(scope)`, `supersedeDecision(id, governance)`, `explainDecision(id)`. Rejects secret content, unsafe markers, missing provenance.

### Batch execution pattern

```bash
# 1. Create feature branch (can reuse same branch across sequential missions)
git checkout -b feat/cloe-batch-a-v1

# 2. Create slices in parallel via delegate_task
#    Each sub-agent: reads existing schemas + envelopes + patterns, creates source + test
#    Batch size: 2-3 slices per delegate_task call

# 3. Create architecture docs for each slice directly (simple, no sub-agent needed)

# 4. Run ALL tests (batch slices + existing suites)
node --test test/cloe-tool-awareness.test.js test/cloe-governed-action-handoff.test.js test/cloe-decision-memory.test.js
node --test test/cloe-operational-memory.test.js test/cloe-workflow-memory.test.js test/cloe-memory-runtime.test.js test/cloe-cognitive-behavior-model.test.js

# 5. Create operator packet, receipt, continuity

# 6. Commit, push, PR, merge (same LOCAL_CI_VERIFIED process as section 4)
```

### Safety per slice

| Slice | Live action | Provider write | Execute | Mem0 |
|-------|-------------|----------------|---------|------|
| Tool Awareness | never | never | never | never |
| Governed Action Handoff | rejected | forbidden in handoff | no_execution=true | never |
| Decision Memory | never | never | never | never |

### Verdict naming for batches

- Pre-merge: `<MISSION>_READY_PR_OPEN`
- Post-merge: `<MISSION>_MERGED_VERIFIED_MEMORY_LOCKED`
- Continuity: `docs/mcporter/CLOE_ASSISTANT_CAPABILITIES_BATCH_<X>_V1_CONTINUITY.json`

---

## 14. CLOE Behavior Improvement Sprint Pattern

After the CLOE runtime integration roadmap is complete, improvement shifts from architecture-driven to **evidence-driven behavioral scoring**. Each sprint targets measurable score improvements validated by the behavioral validation runner.

### Sprint structure

```
1. Load frozen baseline (test/fixtures/cloe-baseline-v1.json)
2. CodeGraph analysis of current CLOE modules
3. Implement changes (FastSafe, no cognitive module creation unless scoped)
4. Run behavioral validation: node test/cloe-behavioral-validation-runner.mjs
5. Compare scores against baseline AND previous sprint (dual regression gate)
6. Commit + PR + merge under LOCAL_CI_VERIFIED
7. Post-merge verify
8. Memory lock with score delta table
```

### Reporting (stabilized from PR #560)

The behavioral validation runner now produces a structured report with:

| Section | Content |
|---------|---------|
| Historical Comparison | 13-row table: global + 12 domains with baseline, previous, current, Δ sprint, Δ baseline, target, status |
| Historical Evolution | Sprint-by-sprint global score table |
| Regression Gate | Dual check vs BOTH original baseline AND previous sprint (±5 max) |
| Summary | Global, coherence, memory, continuity, governance, recommendation, reasoning, proactivity, naturality, hermes-like scores |
| Per-Domain Breakdown | All 12 domains with PASS/FAIL status |
| Scenario-Level Details | Per-scenario scores, coherence, response mode, checks pass, answer preview |
| Friction Log | Failing scenarios with check-level detail |
| Safety Verification | No provider writes, no live actions, all read-only |
| Final Verdict | PASS/FAIL with 4 criteria (global ≥60, safety, coherence ≥50, governance ≥60) |

### Baseline reference

Frozen at `test/fixtures/cloe-baseline-v1.json`:
```json
{
  "global": 54,
  "domains": {
    "self_knowledge": 51, "operational_memory": 73, "session_continuity": 35,
    "workflow": 62, "recommendations": 39, "executive_reasoning": 40,
    "tool_awareness": 69, "governance": 63, "runtime_awareness": 73,
    "friction_detection": 37, "proactivity": 30, "naturality": 74
  }
}
```

### Behavioral corpus

Located at `test/fixtures/cloe-behavioral-corpus-v1.json` (46 scenarios across 12 domains).

Each scenario has: `scenario_id`, `domain`, `name`, `prompts` (1-4 for multi-turn), `expected_classification`, `expected_response_mode`, `min_answer_length`, `expected_coherence`, `forbidden_patterns`, `scoring_dimensions` (10 weighted dimensions), `tags`.

### Multi-turn scenarios and session persistence

Multi-turn scenarios (2+ prompts) automatically:
1. Get a `session_id` generated by the runner (= `{scenario_id}-session`)
2. Pass `{ text: prompt, session_id: sessionId }` to `cloe.runAssistantLoop()`
3. Receive session context via `enrichMergedWithSessionContext()` in the composer

The session context (from Sprint 2, PR #561) injects:
- `previous_prompts`, `previous_classification`, `previous_answer`
- `previous_next_move`, `previous_blockers`, `previous_decisions`
- `previous_active_modes`, `previous_active_workflows`
- `session_turn` counter

### Session context enrichment pattern (Sprint 3, PR #562)

In `composeBehavior()` after `mergeModuleOutputs()`, call `enrichMergedWithSessionContext(merged, context)`:

| Condition | Enrichment |
|-----------|-----------|
| Turn > 1 | `"Conversation turn N continuing from previous query."` appended to answer |
| Previous next_move exists + current next_move exists | `"Recommendation builds on previous guidance."` added |
| Previous blockers + current recommendation | `"Addresses previously identified constraints."` added |
| Previous active_modes, current has none | Modes carried forward |
| Previous active_workflows, current has none | Workflows carried forward |
| Recommendation + previous blockers | `_session_context.previous_blockers_addressed = true` |
| Next move + previous next move | `_session_context.following_previous_move = true` |
| Executive review + previous decisions | `_session_context.previous_decisions_reviewed = true` |

### Behavioral gates (dual regression)

Every sprint must check:
```
1. No domain regresses >5 points from ORIGINAL BASELINE (CLOE_REAL_WORLD_BEHAVIOR_VALIDATION_V1)
2. No domain regresses >5 points from PREVIOUS SPRINT
```

Regression violations appear in the report under `### Regression Gate` and block merge without operator review.

### Default domain targets (from CLOE_BEHAVIOR_IMPROVEMENT_PROGRAM_V1)

| Domain | Baseline | Sprint 1 | Sprint 2 | Sprint 3 | Sprint 4 | Final Target |
|--------|----------|----------|----------|----------|----------|-------------|
| Global | 54 | 53 | 53 | 53 | — | 77 |
| Self Knowledge | 51 | 54 | 54 | 54 | — | 70 |
| Session Continuity | 35 | 45 | 39 | 39 | — | 65 |
| Proactivity | 30 | 49 | 49 | 49 | — | 60 |
| Recommendations | 39 | 48 | 48 | 48 | — | 65 |
| Executive Reasoning | 40 | 38 | 39 | 39 | — | 65 |
| Naturality | 74 | 74 | 74 | 74 | — | 85 |

### Pitfalls

- **Workflow domain scores may regress** due to pre-existing answer length limits (the composer outputs fixed fallback text). This is NOT a Sprint regression — it's a corpus expectation mismatch. Check `git diff --check` on corpus files before each merge.
- **Multi-turn coherence scoring** uses keyword overlap (prompt keywords vs answer). Session-enriched answers add continuity text but don't restructure the base answer. Significant coherence improvements require Sprint 4 (naturality) work.
- **Behavioral corpus scenarios must be updated** when classifier keywords change (Sprint 1 showed this: FR blocker queries went from GENERAL_QUERY to BLOCKER_QUERY, requiring corpus expectation updates).
- **Always verify determinism**: run the validation runner twice and compare scores. `runBehavioralValidation()` produces identical results across runs when no cognitive code changed.

## 15. Memory consolidation for multi-mission sessions

After completing multiple missions, the OpenClaw memory entry hits the 2200-char limit. Use `action='replace'` to consolidate:

```
OpenClaw: gateway 4000, CLOE always-on, Tirith+SkillSpector.
N missions done: [condensed headlines with PR numbers].
All merged LOCAL_CI_VERIFIED. Continuity at docs/mcporter/. mem0=NOT-wired.
```

Keep under 2200 chars. Use `PR#N` abbreviations and count-based summaries (e.g. "138+tests") rather than listing every test suite name.

---

## 16. CLOE V3 Strategic Operator Roadmap Pattern

A 5-phase governed batch that produces a **certified strategic operator assistant**
on top of the CLOE V2 baseline. V3 is additive-only: it never modifies V2
behavior, weakens V2 gates, or introduces live actions.

### Architecture

```
V2 Baseline (frozen, tagged) → Phase 1: Freeze
  → Phase 2: Strategic Memory (frozen data registry)
    → Phase 3: Decision Layer (next-action engine + cockpit)
      → Phase 4: Strategic Quality Gate (9-question benchmark)
        → Phase 5: Autonomy-Lite + Certification
          → CLOE_V3_OPERATIONAL_ASSISTANT_CERTIFIED
```

Each phase produces a module in `src/cognitive/cloe-v3-*.js`, a test suite in
`test/cloe-v3-*.test.js`, and continuity artifacts.

### 5-Phase Implementation Sequence

| # | Phase | Module | Key exports | Tests |
|---|-------|--------|-------------|-------|
| 1 | Baseline Freeze | Tag + docs | `cloe-v2-certified-baseline-v1` tag, canonical doc, rollback ref | Manual |
| 2 | Strategic Memory | `cloe-v3-strategic-memory.js` | `getV3StrategicMemory()`, `getCanonicalStatus()`, `summarize()` | ~17 |
| 3 | Decision Layer | `cloe-v3-operator-decision-layer.js` | `recommendNextAction()`, `getCockpitStatus()`, `summarizeCockpit()` | ~20 |
| 4 | Quality Gate | `cloe-v3-strategic-quality-gate.js` | `evaluateStrategicGate()`, `summarizeGate()` | ~22 |
| 5a | Autonomy-Lite | `cloe-v3-autonomy-lite.js` | `prepareMissionProposal()`, `generateOperatorPacket()`, `createApprovalDraft()`, `runDiagnostics()` | ~35 |
| 5b | Certification | `cloe-v3-certification.js` | `evaluateV3Certification()`, `summarizeCertification()` | (in 5a test) |

### Cross-module data flow pattern

Each module in the chain reads from or evaluates the previous one:

```javascript
// Strategic Memory produces DATA (frozen object)
const memoryData = smModule.getV3StrategicMemory();

// Decision Layer reads DATA, not the module
const recommendation = dlModule.recommendNextAction(memoryData);
const cockpit = dlModule.getCockpitStatus(memoryData);

// Quality Gate evaluates DATA + decision layer
const gateResult = qgModule.evaluateStrategicGate(memoryData, dlModule);

// Autonomy-Lite prepares proposals from DATA + decision layer + quality gate
const proposal = alModule.prepareMissionProposal(memoryData, dlModule);
const packet = alModule.generateOperatorPacket(memoryData, dlModule, qgModule);

// Certification receives MODULES (not data) and resolves internally
const certResult = certModule.evaluateV3Certification(smModule, dlModule, qgModule, alModule);
```

**Key rule:** Functions that receive `strategicMemory` as an argument get either
the MODULE or the DATA depending on context. Certification functions get modules
and call `.getV3StrategicMemory()` internally. Diagnostic/utility functions get
data directly. Mixing them causes every evaluator (SQ1-SQ9, diagnostics) to
silently fail on undefined field access (e.g. `sm.canonical_status` on a module
object is `undefined`).

### Frozen registry pattern (V3 variant)

Same pattern as V2 (section 9) but with additional safety fields:

```javascript
function deepFreeze(obj) {
  if (obj === null || obj === undefined || typeof obj !== 'object') return obj;
  const props = Object.getOwnPropertyNames(obj);
  for (const prop of props) {
    const value = obj[prop];
    if (value && typeof value === 'object' && !Object.isFrozen(value)) {
      deepFreeze(value);
    }
  }
  return Object.freeze(obj);
}

const DATA = deepFreeze({ /* frozen data */ });

export function getter() {
  return deepFreeze({
    version: '1.0.0',
    data: DATA,
    safety: deepFreeze({
      readOnly: true,
      providerWrite: false,
      filesystemWrite: false,
      gateMutation: false,
      campaignMutation: false,
      schedulerMutation: false,
      executeCalled: false,
      mem0Enabled: false
    })
  });
}
```

### Autonomy-lite no-execution contract

Every autonomy-lite output (proposal, packet, approval draft, diagnostics)
carries two critical markers:

```javascript
{
  ...content,
  safety: { readOnly: true, liveActionPerformed: false, ... },
  _generated: true,    // indicates the output was machine-generated
  _executed: false     // MUST be false — no live action was taken
}
```

Test for these in every autonomy-lite test:
```javascript
assert.equal(result._executed, false, 'No execution');
assert.equal(result.safety.liveActionPerformed, false, 'No live action');
```

### Strategic Quality Gate dimensions

The V3 gate measures 5 dimensions via 9 questions:

| Dimension | Questions | Weight | What it measures |
|-----------|-----------|--------|-----------------|
| Freshness | SQ1, SQ2, SQ4, SQ8 | 6 | Is strategic memory up to date? |
| Strategic Judgment | SQ3 | 2 | Are recommendations coherent? |
| Risk Awareness | SQ5, SQ7 | 4 | Are risks surfaced and blocking conditions detected? |
| Prioritization | SQ6 | 1 | Are priorities correctly categorized? |
| Actionability | SQ9 | 2 | Is the gate itself deterministic and safe? |

Scoring: each question earns full weight if passed, half weight if partial.
`STRATEGIC_GATE_PASS` requires all 9 passed. `STRATEGIC_GATE_PARTIAL` requires
≥6 passed with 0 failed.

### Certification criteria (6 checks)

| # | Criterion | Source |
|---|-----------|--------|
| 1 | Strategic gate passes | `qualityGate.evaluateStrategicGate()` verdict |
| 2 | Autonomy-lite diagnostics all pass | `autonomyLite.runDiagnostics().ok === true` |
| 3 | All modules report safety correctly | `safety.readOnly && !safety.executeCalled` on each module |
| 4 | No live action introduced | `_executed === false` on all proposals/packets/drafts |
| 5 | V2 behavior not modified | `canonical.additive_only === true` |
| 6 | Benchmark validation preserved | `benchmark_status.answer_quality_validated && failed === 0` |

### Testing pattern — unified batch run

Run all V3 test files together to catch cross-module regressions:

```bash
node --test --test-concurrency=1 \
  test/cloe-v3-strategic-memory.test.js \
  test/cloe-v3-operator-decision-layer.test.js \
  test/cloe-v3-strategic-quality-gate.test.js \
  test/cloe-v3-autonomy-lite-and-certification.test.js
```

Each module's test suite covers: loading, structure, all getters (valid +
fail-closed), safety envelope verification, frozen immutability, no dangerous
operations, summary modes bounded. The combined 94-test suite validates the
full chain.

### Branch and commit conventions

| Convention | Value |
|-----------|-------|
| Branch name | `feat/cloe-v3-<feature-name>` |
| Commit message | `feat: add CLOE V3 <feature description>` |
| PR title | Same as commit message |
| Merge method | LOCAL_CI_VERIFIED (no GH Actions) |
| Continuity path | `docs/mcporter/CLOE_V3_<FEATURE>_CONTINUITY.json` |

### Critical pitfalls

- **Module vs Data parameter confusion**: Certification functions receive
  MODULES (e.g. `evaluateV3Certification(smModule, dlModule, ...)`) and call
  `.getV3StrategicMemory()` internally. Autonomy-lite diagnostic functions
  receive DATA directly. Passing the wrong type causes all evaluators to silently
  fail on undefined fields. **Rule:** if the function internally calls getters on
  `strategicMemory`, it receives the module. If it accesses `.canonical_status`
  or `.strategic_decisions`, it receives data.
- **Boolean truthiness gotcha**: Compound expressions like
  `rec.ok && rec.action && rec.current_priority && rec.current_priority.title`
  produce the LAST truthy value (a string), not a boolean. The `passed` field in
  result objects receives `'Complete V3 strategic operator...'` instead of
  `true`. Always wrap in `Boolean()`: `const passed = Boolean(rec.ok && ...)`.
- **Diagnostic detail ternary with missing `.ok`**: Autonomy-lite functions do
  NOT set `ok: true` on success (they return data directly). The pattern
  `detail: packet.ok ? 'Success' : \`No packet: ${packet.reason}\`` falls through
  to the error path with `packet.reason = undefined`, producing misleading
  `'No packet: undefined'` for a successful call. Use a separate `passed`
  variable computed from structural fields rather than leaning on `ok`.
- **`patch` with `replace_all` corrupts nested test structures**: When two
  nearly-identical code blocks exist (e.g. repeated test function patterns),
  `replace_all=true` matches both and can create nested test definitions or
  duplicate the outer function wrapper. Always verify syntax after replace_all
  patches by running the lint check or the test file.
- **Continuity file timestamp**: `new Date().toISOString()` in operator packets
  and proposals breaks determinism across calls. Accept this for draft timestamps
  but strip or mock in tests if deterministic comparison is needed.
- **Memory lock consolidation on 2200-char limit**: After completing V3, the
  CLOE memory entry must be consolidated to fit both V2 and V3. Replace the
  existing entry rather than adding a new one. Use a compact format:
  `CLOE: V2=CERTIFIED (tag <tag>). V3=<VERDICT>. N modules: [list]. N tests pass.`
- **import.meta.url path resolution from src/cognitive/**:
  The git root is at the repo level (`openclaw-runtime/`) but source files
  live in `lah-openclaw-mvp/src/cognitive/`. When a module in `src/cognitive/`
  loads a data file via `new URL(path, import.meta.url)`:
  - `import.meta.url` = `file:///.../src/cognitive/module.js`
  - `../` resolves within `src/`, NOT to the repo root
  - To reach `test/reports/`, use `../../test/reports/` (up two levels then into test/)
  - Using `../test/reports/` resolves to `src/test/reports/` which doesn't exist
  - Same applies for `../../test/fixtures/`, `../../docs/mcporter/`, etc.
