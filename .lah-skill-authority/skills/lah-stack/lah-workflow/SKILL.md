---
name: lah-workflow
description: "Use when executing a gated LAH Stack mission: Repository routing preflight → CodeGraph → AutoResearch → plan → FastSafe → implémentation → tests → PR → merge → verify → memory lock. Supports CODE_CHANGE, READ_ONLY_AUDIT, DESIGN_ONLY, MIXED. Gate 0 repo routing is MANDATORY."
---

# LAH Workflow — Gated Orchestration

**Persona missions:** the human gate judges personality, not tests — full untruncated old-vs-candidate comparisons, never auto-verdict; canonical files can SUPERSEDE a code fix. See `references/canonical-persona-and-parity-patterns.md`.

A **gate** is a checkable step. Read-only audits: show exact command first; see `references/readonly-audit-scan-protocol.md`.
Canonical exact-SHA deployment: `references/canonical-exact-sha-deployment-capability.md`. Recollect: `references/cloe-evidence-recollection-and-live-import-pattern.md`; outcome: `references/affiliate-outcome-semantics-contract.md`. Live-cert/admin-merge/harness pitfalls: `references/cloe-live-certification-and-admin-merge.md`, `references/lahb-route-test-harness-pitfalls.md`. Focused test-harness pitfalls + live certification probes for lah-brain/openclaw (JS default-param trap, mock-fetch defaults, fixture display names, secret-masking mutilation of `VARNAME=` lines, CERT A/B/C shapes, deployer first-run IMAGE_GIT_COMMIT_MISMATCH): `references/lahb-test-harness-and-live-cert-patterns.md`. Bounded in-process live-gate proofs for business capabilities (module map per capability, fail-closed assertions, determinism + real digest, operator import protocol, Option C read-only provider replay, batching multiple gates into one promotion): `references/cloe-live-gate-bounded-proof-pattern.md`.

---

## Prerequisites

- **Repos:**
  - `lah-stack-tools` → `/home/deploy/lah-stack-repos/lah-stack-tools`
  - `lah-stack-biz-assets` → `/home/deploy/lah-stack-repos/lah-stack-biz-assets`
  - `cartelogic-v2` → `/home/deploy/lah-stack-repos/cartelogic-v2`
- **Router:** `skill_view(name='lah-repo-router')` + `read_file(repo_mappings.json)` — load BEFORE first repository-dependent action
- **CodeGraph:** `node tools/codegraph/check-codegraph-availability.mjs --json` (in lah-stack-tools)
- **Branch:** `git status --short && git branch --show-current && git rev-parse --short HEAD`

---

## Repository Routing Preflight (Gate 0)

**This step is MANDATORY. It runs before any other gate.**

The `lah-repo-router` skill is the routing authority. It decides where the work belongs. Do not skip it. Do not guess a repository. Do not use the shell's current working directory as authority.

**Exception — user-provided explicit repo metadata replaces Gate 0 routing:** When the mission spec explicitly provides ALL of the following — canonical authority, worktree path, branch name, current HEAD SHA, and required ancestry — treat that metadata as the routing receipt. Do NOT run the router. Proceed directly to Gate 0 verification (Step 4: verify the workspace, detect worktree-vs-clone, verify HEAD matches). Rationale: the user's spec is more authoritative than the router's prefix-based matching, and the router cannot resolve data it doesn't contain (e.g. non-canonical repos, worktree-specific branches). This is the routing equivalent of the Gate 3 "user-provided detailed spec replaces plan" pitfall. If any element is missing (e.g. no explicit SHA, no worktree path), fall back to the router.

### Step 1 — Load the router

```bash
skill_view(name='lah-repo-router')
read_file(path='~/.hermes/skills/software-development/lah-repo-router/references/repo_mappings.json')
```

### Step 2 — Invoke the routing engine

Pass the mission text to the deterministic router.
**Do NOT use `echo "$MISSION" | node ...`** — the pipe-to-node pattern silently returns 0 resolved
(see router pitfall). Use the file-based shell wrapper:

```bash
# Write mission to temp file (newline-terminated for reliable parsing)
echo "$MISSION" > /tmp/lah-mission.txt
bash ~/.hermes/skills/software-development/lah-repo-router/scripts/dry-run-route.sh \
  ~/.hermes/skills/software-development/lah-repo-router/references/repo_mappings.json \
  /tmp/lah-mission.txt
```

If the router returns `Resolved: 0` despite a valid mission prefix, apply the manual fallback
(see Step 3a below).

Parse the JSON receipt from the output. Extract:
- `decision` — RESOLVED | AMBIGUOUS | UNRESOLVED
- `decision_source` — ROUTER | MULTI_ROLE_PARSE | CODEGRAPH
- `repository_authority` — the resolved repo
- `implementation_repo` — where code changes go
- `execution_workspace` — where the agent operates
- `memory_repo` — where memory/decisions go (may differ from implementation)
- `context_repos` — read-only repositories
- `write_allowed_repos` — repos writable under routing policy
- `write_forbidden_roots` — paths that must not be mutated
- `codegraph_used` — whether CodeGraph escalation was triggered
- `codegraph_evidence` — structural evidence if CodeGraph was used

### Step 3 — Handle the decision

#### RESOLVED

Save the full receipt. It is referenced by every later gate.

Set:
```
ROUTING_RECEIPT=<parsed JSON>
ROUTING_REPO=<repository_authority>
ROUTING_WORKSPACE=<execution_workspace>
ROUTING_IMPL=<implementation_repo>
ROUTING_MEMORY=<memory_repo>
ROUTING_CONTEXT=<context_repos>
ROUTING_WRITE_ALLOWED=<write_allowed_repos>
ROUTING_FORBIDDEN=<write_forbidden_roots>
ROUTING_DECISION_SOURCE=<decision_source>
```

Continue to **Gate 0.5 — Mission Decomposition (MANDATORY)** with the resolved repository as context.

#### AMBIGUOUS

**STOP.** Do not proceed to implementation or mutation.

Surface:
```
⚠️  ROUTING AMBIGUOUS
    Reason: <reason_code>
    Candidates: <repository_candidates>
    CodeGraph evidence: <codegraph_evidence>
    Required: operator decision
```

Do not guess. Do not auto-select the first candidate. Do not proceed.

#### UNRESOLVED

**STOP. Fail-closed.**

```
❌  ROUTING FAILED — UNRESOLVED
    Reason: <reason_code>
    No matching repository found.
    Cannot continue. Mission blocked.
```

#### Protected root or write-policy conflict

**STOP.** Block the workflow before any mutation.

```
⛔  ROUTING BLOCKED — WRITE POLICY VIOLATION
    Forbidden roots: <write_forbidden_roots>
    Cannot write to protected path. Mission blocked.
```

### Step 3a — Manual fallback (router returns 0 resolved or wrong resolution)

When the router returns `Resolved: 0` despite having a mission with a valid prefix,
OR when the router returns RESOLVED but to a wrong repository with high confidence:

1. **Check `canonicalMissionRepoMap`** in `repo_mappings.json` — does the mission prefix match an entry?
2. **Check the mission's explicit metadata** — does the user specify a working directory, expected branch, or expected HEAD SHA?
3. **Verify the expected repo**:
   ```bash
   cd <expected_canonical_checkout> && \
     git rev-parse --show-toplevel && \
     git rev-parse --short HEAD
   ```
4. **If only one repo matches** the explicit metadata and the prefix map, resolve manually:
   ```
   decision: RESOLVED
   decision_source: MANUAL_OVERRIDE (router returned 0 resolved; mission metadata matched)
   ```
5. **If no repo matches** explicit metadata, treat as UNRESOLVED.

Known causes of 0-resolved or wrong resolution:
- Pipe-to-node EOF race (mitigated by Step 2's file-based approach)
- `CLOE_` prefix matching `openclaw` — the router matches prefix literal `CLOE` which is NOT `openclaw`. The `canonicalMissionRepoMap` entry `"cloe":"openclaw-runtime"` is for the alias system, not the prefix matcher. If the router hasn't been updated, the prefix search for `CLOE_` finds no match.
- **Router resolves `CLOE_` to `openclaw-runtime` with HIGH confidence, but actual repo is `cloe-diagnostic-orchestrator`** — when `cloe` IS in `canonicalMissionRepoMap`, the router returns `RESOLVED` to `openclaw-runtime` without ambiguity. But `cloe-diagnostic-orchestrator` is a separate repo (not in the canonical mapping) with its own worktrees. Verify by checking mission metadata (lot directory, worktree path, HEAD SHA) against `/home/deploy/lah-stack-repos/cloe-diagnostic-orchestrator/`.
- A mission prefix that exists in prefixPriority but not in a form the router's Phase 1 can match deterministically

Apply manual resolution only when the evidence is definitive (HEAD SHA matches the mission spec, branch matches, remote matches). Never guess.

### Step 4 — Verify the workspace

```bash
cd <ROUTING_WORKSPACE or ROUTING_REPO canonical_checkout> && \
  git rev-parse --show-toplevel && \
  git remote -v
```

If the command fails or the remote does not match the expected canonical remote, **STOP. Workspace mismatch.**

### Step 4b — Identify workspace vs canonical checkout identity

The LAH Stack maintains **two separate clones** of `openclaw-runtime` and potentially other repos:
- **Workspace clone** (e.g. `/home/deploy/openclaw-runtime`) — execution workspace `openclaw-runtime-dev`, remote `github.com` (no `-lah-stack` suffix). Used for active work.
- **Canonical checkout** (e.g. `/home/deploy/lah-stack-repos/openclaw-runtime`) — canonical authority, remote `github.com-lah-stack`. References `repo_mappings.json`.

These are **separate filesystem entries with different inodes** — not symlinks, not worktrees derived from the same `.git` directory. A file written to one does NOT appear in the other. After routing, verify which clone you are in:

```bash
# Check if multiple clones exist by comparing inodes
inode1=$(stat --format="%i" /home/deploy/openclaw-runtime 2>/dev/null)
inode2=$(stat --format="%i" /home/deploy/lah-stack-repos/openclaw-runtime 2>/dev/null)
if [ "$inode1" != "$inode2" ] && [ -n "$inode1" ] && [ -n "$inode2" ]; then
  echo "DIFFERENT CLONES — separate filesystems"
fi

# Verify git roots match the expected remote
cd <path> && git rev-parse --show-toplevel && git remote -v
```

**Critical rule for deliverables:** After writing an architecture plan, documentation, or mission artifact to the workspace clone, verify whether the canonical checkout needs the same file. If the deliverable should survive workspace changes (e.g. the workspace branch is a temporary feature branch), it must be committed to or synced to the canonical checkout. A deliverable that exists only in the workspace clone and not in the canonical checkout is not durable — treat it as ephemeral.

### Step 4c — Distinguish worktree from separate clone

A **Git worktree** (e.g. `/home/deploy/lah-stack-worktrees/cloe-x402-lot-d`) is NOT a separate clone — it shares the canonical checkout's `.git` directory. This is distinct from the separate-clone case above and has important implications.

**How to detect a worktree:**

```bash
# Check the git common directory — if it points outside the worktree, it's a worktree
cd /path/to/suspected/worktree
git rev-parse --git-common-dir
# Worktree output: /home/deploy/lah-stack-repos/openclaw-runtime/.git
# Standalone repo output: .git

# Also check git worktree list on the canonical checkout
cd /home/deploy/lah-stack-repos/openclaw-runtime && git worktree list
```

**Implications of a worktree (shared `.git`):**

| Property | Worktree | Separate clone |
|----------|----------|---------------|
| `stat --format="%i"` vs canonical checkout | Different inode (different dir) | Different inode |
| `git rev-parse --git-common-dir` | Points to canonical `.git` | `.git` (local) |
| `git status` shows canonical checkout's dirty files | **YES** — dirty files from the canonical checkout appear | No |
| `git log` shows only worktree's branch | Yes | Clone's branch only |
| Staging/committing | Modifies canonical checkout's refs | Local only |
| Untracked files shown | Includes canonical checkout's untracked | Local only |

**Pitfall — worktree `git status` contaminated by canonical checkout dirt:** A worktree whose own branch is clean will still show dirty files and untracked files in `git status` because the shared `.git` reflects the canonical checkout's 155 dirty files. This session's worktree at `cloe-x402-lot-d` showed 3 modified and 38 untracked files — all from the canonical checkout, none from the Lot D commit.

To see only the worktree's OWN changes, use:

```bash
# Compare the worktree's branch to its parent/base commit
git diff c13cc39 HEAD --name-only

# Or scope status to the specific directory you care about
git status --porcelain path/to/your/directory/

# Check git log for only the worktree branch's own commits
git log --oneline --first-parent
```

Do NOT rely on `git status --porcelain` alone to assess worktree cleanliness when the worktree shares `.git` with a dirty canonical checkout.

---

## Gate 0.5 — Mission Decomposition (MANDATORY)

**This step is MANDATORY. It runs after Gate 0 (routing) and before any other gate.**

Long specs overwhelm context; gates get acknowledged then lost; sub-agent parallelism needs atomic phases.

### Step 1 — Load the decomposer

```bash
skill_view(name='mission-decomposer')
```

### Step 2 — Generate the phase plan

```bash
echo "$MISSION_TEXT" > /tmp/mission.txt
node /home/deploy/.hermes/skills/software-development/mission-decomposer/scripts/decompose-mission.mjs \
  /tmp/mission.txt \
  <REPO_PATH> \
  <MISSION_TYPE>
```

### Step 3 — Review the plan

Phase fields: `id` (unique), `goal` (task for sub-agent), `context_fields` (what it needs), `toolsets` (restricted), `gate_pass` (verification), `artifacts_out` (extract+propagate), `critical` (blocks rest).

### Step 4 — Execute phases in parallel batches

**Batch 1 — Discovery** (3 lanes via `delegate_task`):

| Lane | Focus | Output |
|------|-------|--------|
| **A — Architecture** | CodeGraph on resolved repo | Module map, integration points |
| **B — Asset & Scope** | Inventory of registries, states | Priority catalog |
| **C — Safety & Quality** | Secret scan, prohibited content | Security checklist |

**Context rule:** each sub-agent gets routing context, exact paths, safety invariants, output format; zero parent memory.

### Step 5 — Verify inter-phase consistency

1. Combined tests pass (no regression)
2. Commit scope clean
3. No credentials or network activated

### Gate-pass conditions

- [ ] `decompose-mission.mjs` produced a valid JSON plan
- [ ] Each phase has a concrete `gate_pass` check
- [ ] Batch 1 sub-agents ran in parallel (3 concurrent lanes)
- [ ] All phases report `gate_pass: true`
- [ ] Combined tests pass

### When to skip

Do NOT skip this gate.

**Exception — user-provided phase-by-phase spec satisfies gate.** If the user provides 10+ explicit phases with gate-pass conditions, treat that as decomposition. Do NOT run `decompose-mission.mjs`. (Parallel to Gate 3 user-provided-spec rule.)

### Known pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| **Sub-agent writes to wrong repo** | Files outside allowed scope | Pass `SAFETY: write only to <ROUTING_WORKSPACE>` |
| **Orchestrator balloons** | All phase results in context | Extract only `artifacts_out` fields |
| **Parallelism underutilized** | 1 sub-agent instead of 3 | Always fill Batch 1 with 3 independent lanes |

## Gate 0.5b — Existing-Infrastructure Audit (before Gate 1)

The routing receipt is propagated to all later phases. Later phases must not independently reselect the repository.

The receipt is available as a JSON object with at minimum:

```json
{
  "repository_authority": "string",
  "implementation_repo": "string",
  "execution_workspace": "string",
  "memory_repo": "string | null",
  "context_repos": [],
  "write_allowed_repos": [],
  "write_forbidden_roots": [],
  "decision_source": "ROUTER | MULTI_ROLE_PARSE | CODEGRAPH"
}
```

### Where each field is used

| Later phase | Uses |
|-------------|------|
| Gate 1 — CodeGraph | `repository_authority` as CodeGraph projectPath |
| Gate 2 — AutoResearch | `context_repos` for read-only inspection |
| Gate 5 — Implementation | `implementation_repo` + `execution_workspace` + `write_allowed_repos` |
| Gate 6 — Tests | `execution_workspace` for test execution |
| Gate 8 — Commit | `implementation_repo` for commit target |
| Gate 9 — PR/Merge | `implementation_repo` for PR creation |
| Gate 10 — Memory Lock | `memory_repo` for memory writes |
| Gate 11 — Continuity JSON | All fields for the continuity record |

---

## CodeGraph ordering

CodeGraph is used in TWO distinct places, in a specific order:

### 1. Targeted CodeGraph inside the router (if required)

When the router encounters ambiguity, it calls the `codegraph-evidence.cjs` adapter on the candidate repositories only. This is STRUCTURAL evidence — it scores ownership, finds symbols, and helps disambiguate.

This happens automatically inside the router. You do not need to call it separately.

### 2. Normal CodeGraph analysis (after routing)

After the router has resolved the repository, use `codegraph_explore` with the resolved project path:

```javascript
codegraph_explore({
  query: "concept or symbol name",
  projectPath: "/absolute/path/to/resolved/repo",  // from routing receipt
  maxFiles: 12
});
```

**IMPORTANT:** Do not call general CodeGraph analysis before the router. Do not scan all repositories by default. The correct order is:

```text
lah-repo-router
→ targeted CodeGraph escalation inside router (if needed)
→ repository selected
→ normal CodeGraph analysis inside the selected repository only
```

---

## Repository context enforcement

Before any file operation or agent launch:

```bash
# Verify the execution workspace exists
test -d "<ROUTING_WORKSPACE or canonical_checkout>" || { echo "WORKSPACE_MISSING"; exit 1; }

# Verify git root
cd "<path>" && git rev-parse --show-toplevel || { echo "GIT_ROOT_MISMATCH"; exit 1; }

# Verify remote metadata
cd "<path>" && git remote get-url origin || { echo "REMOTE_MISSING"; exit 1; }

# Reject workspace mismatch
# If the resolved path is under a forbidden root, block
forbidden_roots=("${ROUTING_FORBIDDEN[@]}")
for root in "${forbidden_roots[@]}"; do
  if [[ "$resolved_path" == "$root"* ]]; then
    echo "FORBIDDEN_ROOT: $root"
    exit 1
  fi
done

# Verify the implementation repo is writable (from routing policy)
if [[ " ${ROUTING_WRITE_ALLOWED[@]} " =~ " ${ROUTING_IMPL} " ]]; then
  echo "Write allowed to: $ROUTING_IMPL"
else
  echo "READ_ONLY: $ROUTING_IMPL (requires_explicit_scope or always_forbidden)"
fi
```

Before memory writes:

```bash
# Use memory_repo — NOT the implementation repo
if [[ -n "$ROUTING_MEMORY" ]]; then
  echo "Memory writes to: $ROUTING_MEMORY"
  # cd to that repo
fi
```

---

## Mission Types

Determine the mission type FIRST. This affects which gates apply and how Gate 5 works.

| Type | Description | SKIP gates |
|------|-------------|------------|
| **CODE_CHANGE** | Feature, fix, refactor, CLI addition, test addition | None |
| **READ_ONLY_AUDIT** | Analysis, mapping, gap analysis, inventory, policy design. No code modification. No thresholds changed. | Gate 5 (standard impl), Gate 8 (commit), Gate 9 (PR/merge). Replace Gate 5 with Audit Execution (sub-agents + docs) |
| **DESIGN_ONLY** | Plan, architecture design, spec writing. No code. | Gate 5, 6, 8, 9. Gate 7-10-11 still apply if the plan is the deliverable. |
| **MIXED** | READ_ONLY_AUDIT baseline → identifies gap → bounded CODE_CHANGE repair → runtime proof against real provider. Used for integration proof missions where you need to prove end-to-end behavior through the canonical pipeline. | Gate 5 splits into 3 phases (Audit → Repair → Proof). Gate 6 includes runtime scenario validation. Gate 8/9 commit repair + proof artifacts. |
| **PROMOTION_ONLY** | Revalidate + promote an already dry-run-validated enriched candidate (evidence system). No code, no PR, no rebuild, no restart, no new evidence. Sequence: fresh dry-run identity match → canonical promote → hot-reload verify (same container, StartedAt unchanged) → duplicate dry-run ALREADY_CURRENT → final report + memory lock. | Gate 0 routing (mission metadata replaces it), Gates 1-9 (code/audit/impl/test/commit/PR/merge), Gate 11 continuity JSON when the mission says stop after report. Gate 10 memory lock applies. See `references/evidence-promotion-only-mission-pattern.md`; recollection/supersession repair: `references/evidence-recollection-and-supersession-pattern.md`. |

**Pitfall — user-provided detailed spec replaces Gate 3 plan:** When the user provides a self-contained mission spec with explicit sections (mission objective, scope, architecture, non-negotiables, migration strategy, tests, verification, commit strategy, stop conditions, verdict format, and report structure), treat it as the Gate 3 deliverable. Do NOT write a separate plan document — the spec IS the plan. Skip `grill-me`/`grill-with-docs` unless the user explicitly asks for it. Proceed directly to Gate 4 (FastSafe) or Gate 5 (Implementation) depending on the mission type.

---

## Skills Branch Table

These skills are available in the Hermes session — **load them at the right gate** when the trigger fires:

| Branch | Skill | Gate | Trigger |
|--------|-------|------|---------|
| **Mission decomposition** | `mission-decomposer` | **Gate 0.5 — MANDATORY** | **Always executes after Gate 0 (routing).** Splits spec into atomic phases for parallel sub-agent execution. Required for missions of any length to prevent context loss and enable parallelism. |
| **Repo routing** | `lah-repo-router` | Gate 0 (Preflight) | MANDATORY — resolve repository before any action. Always load at start. |
| **Design sharpening** | `grill-me` / `grill-with-docs` | Before Gate 3 (Plan) | Design is fuzzy, scope needs tightening, trade-offs unclear |
| **Background research** | `research` | At Gate 2 (AutoResearch) | Complex context needed — delegate reading to a background agent |
| **Test-first** | `test-driven-development` (Matt Pocock's `/tdd`) | At Gate 5 (Impl) | Function has clear success criteria, prefer RED→GREEN→REFACTOR |
| **Bug diagnosis** | `diagnosing-bugs` | At Gate 6 (Tests) | Tests fail and the bug resists a first glance |
| **Code review** | `code-review` (Matt Pocock's 2-axis) | At Gate 6 (Vérification) | A diff to review against spec + coding standards |
| **Skill selection** | `ask-matt` | Any | Uncertain which skill fits — `ask-matt` names the right one |
| **Session handoff** | `handoff` | Any | Context nearing token limit — compact and continue in a fresh thread |

Superpowers plugin skills are also available as `superpowers:<name>` (see Gate 3).

---

## Gate 0.5 — Existing-Infrastructure Audit (before Gate 1)

**Gate-pass:** audit existing LAH infrastructure in the **resolved repository** for compose-vs-build before design.

### Canonical-repo survey

Select the resolved repo from the routing receipt. If the resolved repo is `lah-stack-tools`, survey these directories:

| Module | Purpose | Reuse pattern |
|--------|---------|--------------|
| `maintenance-authority/` | 10-layer MEP pipeline (L1→L10): collection, normalisation, correlation, diagnosis, risk, remediation, decision, execution, validation, receipt. Schemas, probes, pipeline orchestration. | Most missions can push observations through this pipeline instead of building a new event model |
| `git-workspace/` | Workspace manager: inspectRepository, classifyWorkingTree, createMissionWorkspace, WORKSPACE_STATES state machine. | Git baseline classification, isolation worktrees |
| `git-policy/` | Quarantine manager: non-destructive copy-aside, quarantineMatching, restoreQuarantine. | Safe rollback for mutation missions |
| `batch-runner/src/gates/` | Pre-execution gates, gate evaluator, gate result, approval-gate-enforcement-dry-run. | Gate infrastructure for approval-required steps |
| `operator-validation/` | 5-gate trial engine (safety, factual, experience, runtime, provider), evidence writer, hybrid evaluator. | Operator trial / certification for any new capability |
| `codegraph/` | codegraph-probe, check-codegraph-availability, repository-registry, mission-context-pack. | CodeGraph discovery |
| `controlled-run/` | controlled-run-schema, validateControlledRunInput. | Schema validation patterns |
| `release-authority/` | release-activator, release-rollback, release-verifier, activation-ledger. | Rollback + receipt patterns |
| `repo-hygiene-authority/` (NEW) | Repository hygiene: candidate evidence schema (16 classifications), gate registry, cleanup recipes, collectors, CLI, **reversible-trial.mjs** (governed removal trials: neutralization, rollback, operator scenarios, per-candidate isolation). | Hygiene/cleanup missions + governed reversible removal trials |

### Upstream-dependency survey

When the defect trace points to an installed npm package rather than canonical-repo source, ALSO survey:
- The installed package at `.npm-global/lib/node_modules/<name>/dist/` — these are generated bundles, not acceptable fix locations, but MUST be inspected to understand the bug
- The package's `package.json` `repository` field — to identify the upstream source
- The package's `package.json` `version` — to know what's deployed
- Whether the canonical repo contains a fork of the upstream package (check remote, changelog, fork relationship)
- The running process (ps aux + ls -la /proc/<pid>/cwd) to confirm which version is actually loaded

Example: `openclaw@2026.6.11` bug lived upstream; `openclaw-runtime` held only the app layer — fix was an ESM-loader runtime wrapper on the gateway process, not a canonical-repo source change.

For each module, verify it loads cleanly before writing the plan:
```javascript
try { await import('../tools/maintenance-authority/schema.mjs'); console.log('OK'); }
catch { console.log('MISSING — will need to build'); }
```

**Pitfall:** Do not assume infrastructure exists from documentation alone. A `tools/` subdirectory may exist but be unimportable (broken imports, missing dependencies). Verify with an actual `import()` attempt before committing to a compose-vs-build decision.

---

## Gate 1 — CodeGraph (Cartographie)

**Gate-pass:** Every impacted module explored with `codegraph_explore` (explicit `projectPath` from routing receipt). Architecture, dependencies, and integration points mapped. No module skipped by approximation.

Routing preflight must already have resolved the repository. Gate 1 uses the resolved repository path.

```javascript
// Resolved repository path from routing receipt
const resolvedRepoPath = ROUTING_REPO_PATH;  // e.g. /home/deploy/lah-stack-repos/lah-stack-tools

codegraph_explore({
  query: "concept or symbol name",
  projectPath: resolvedRepoPath,
  maxFiles: 12
});
```

**Note:** The router may already have triggered targeted CodeGraph escalation on candidate repos if routing was ambiguous. Gate 1's CodeGraph analysis is for the *resolved* repository only — deep exploration of its architecture for implementation planning.

---

## Gate 2 — AutoResearch (Contexte read-only)

**Gate-pass:** Internal context (past sessions, assets) and external context (web, niche) collected. Risks and constraints identified. No live mutations. Results documented and available for the plan.

**Branch — `research`:** If the question spans primary sources (API docs, specs, competitor sites, academic papers), load `research` — it delegates reading to a background agent and leaves a cited Markdown file. Keep working while it reads.

**Pattern extraction via sub-agents:** When the mission requires analyzing external repos for architectural patterns, use `delegate_task` with `toolsets=['web']` and a goal like "Extract architectural patterns from <repo>. Focus on: (1)... (2)... Produce structured JSON extraction matrix with: source_repo, subsystem, extracted_principle, target_lah_component, adaptation_needed, implementation_risk, license_status, dependency_decision per principle." Run up to 3 repos per batch, 2 batches for 6-8 repos. Each sub-agent gets the exact file paths, safety invariants, and expected output format — it has zero access to your memory.

**External protocol research** — for missions that require understanding an external protocol (payment protocol, API standard, federated system), use the multi-source protocol research pattern:
   - **Lane A** — spec (official repo, IETF draft)
   - **Lane B** — vendor docs (quickstarts, SDKs, pricing)
   - **Lane C** — security research (arXiv, NDSS, known attacks)
   - **Lane D** — ecosystem/standards (adoption, partnerships)
   - See `references/protocol-research-pattern.md` for full pattern and pitfalls.

Methods:
- **Web search** for market, competition, regulation
- **Session search** for past decisions and patterns
- **Docs** for existing assets

---

## Gate 3 — Superpowers Plan (Spécification)

**Gate-pass:** Plan written at `docs/superpowers/plans/YYYY-MM-DD-mission-name.md`. Scope, target modules, tests, and FastSafe invariants documented. Continuity JSON template included. Validated against AutoResearch results.

**Branch — `grill-me` / `grill-with-docs`:** If the design is still fuzzy — scope unclear, trade-offs unresolved, edge cases unnamed — load `grill-me` first. `grill-with-docs` leaves ADRs and a glossary; use it when this repo should retain the design rationale. Only then write the plan.

The plan includes:
- **Mission** — name and objective
- **Scope** — in-scope AND out-of-scope (from AutoResearch results)
- **CodeGraph** — modules to inspect
- **Sub-agents** — if >3 modules, plan parallel lanes (max 3 per batch)
- **FastSafe** — flags that must stay `false`
- **Tests** — how many, key assertions
- **Continuity JSON** — template for the final record
- **For MIXED missions**: also document the baseline capture method, the expected proof artefacts, and the runtime scenarios to run

**Superpowers skills:** Cat `~/.hermes/plugins/superpowers/skills/<name>/SKILL.md` — not `skill_view`.

---

## Gate 4 — FastSafe Gate (Sécurité)

**Gate-pass:** 15 checks executed individually. Zero failures. Report signals `FASTSAFE_PASS` or lists blocked checks with reasons.

**Efficient batch execution:** When all 15 checks are quick grep/runs with no interdependencies, batch them in a single `execute_code` call with a Python loop (multi-file `grep -c` parsing + config-constant false positives: `references/fastsafe-batch-execution-pitfalls.md`).

| # | Check | Enforce |
|---|-------|---------|
| 1 | No public publish | `publish_allowed` is `false` on all assets |
| 2 | No WordPress publish status mutation | No code calling `wp.publish()` or `status: "publish"` |
| 3 | No BIZ17 publication mission | Verify scope — don't execute publication missions |
| 4 | No deploy | No deploy scripts/commands |
| 5 | No scheduler activation | No cron/scheduler config |
| 6 | No Telegram live calls | No Telegram API calls |
| 7 | No provider calls | No paid provider calls |
| 8 | No LLM generation calls | No non-essential LLM — read-only provider inference for benchmark validation is ALLOWED under no-live-action governance |
| 9 | No ad spend | No budget references |
| 10 | No live affiliate link injection | No live affiliate links |
| 11 | No force push | Standard `git push` only |
| 12 | No secrets printed | Verify files have no hardcoded secrets |
| 13 | No `git add .` | Staging ciblé only |
| 14 | No memory overwrite/collapse | Append-only to `operational_memory.jsonl`, no `rm`/`truncate`/`wipe` |
| 15 | No destructive migration | No migration scripts that alter existing memory stores |

---

## Gate 5 — Implementation (Construction)

**Gate-pass depends on mission type:**

### CODE_CHANGE missions:
BR28 preflight passed (dry-run). Tests written (RED). Code implemented (GREEN). No live mutations, no provider calls. Simulated rollback verifiable.

1. **Load BR28 preflight:** `skill_view(name='br28-implementation-orchestration')` — executes dry-run safety, progressive gates, local evidence, simulated rollback.

2. **Branch — `test-driven-development`:** If the function has clear success criteria, load `tdd` (RED→GREEN→REFACTOR). Each slice is a single behaviour, tested before code.

3. **Safety constants** — use the strict pattern:
   ```javascript
   const SAFETY = Object.freeze({
     public_publish: false, wordpress_mutation: false,
     provider_calls: false, telegram_send: false,
     scheduler_enabled: false, approval_bypass: false, fail_closed: true,
   });
   ```

4. **Kill switch OC29** — must be `enabled: false` for tests:
   ```bash
   mkdir -p tools/control-plane/data
   echo '{"enabled":false,"reason":"operator_seeded_disabled","updated_at":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' > tools/control-plane/data/autonomy-kill-switch.jsonl
   ```

5. **Module structure** — load `pattern-based-codebase-extension` and `lah-stack-cli-cockpit` for conventions.

6. **Test structure** — each test file: `cleanAll()` then `seedKillSwitch()`:
   ```javascript
   function cleanAll() {
     try { rmSync(DATA_DIR, { recursive: true, force: true }); } catch {}
     try { rmSync(join(__dirname, '..', 'runs'), { recursive: true, force: true }); } catch {}
     seedKillSwitch();
   }
   function seedKillSwitch() {
     // writes autonomy-kill-switch.jsonl with enabled:false
   }
   ```

7. **CLI integration points:** import → handler → help text → switch case — see `pattern-based-codebase-extension` for the convention.

### READ_ONLY_AUDIT missions:

No code changes. Gate 5 becomes **Audit Execution**:

1. Define the analysis framework (doctrine, criteria, taxonomy) before collecting data
2. Run case-level audits via sub-agents (max 3 parallel) — **see constraint below on file-writing inside audited repos**
3. Run architecture gap mapping via sub-agent
4. Write all findings to `docs/superpowers/plans/` — **only the parent agent writes to repo directories; sub-agents must write to the external evidence directory instead**
5. Verify internal consistency: all case audits reference the same framework, all gap findings trace to specific cases
6. FastSafe is auto-pass (read-only by definition — still verify no secrets, no mutations)
7. Skip Gate 8 (Commit) and Gate 9 (PR/Merge) — documentation is committed as part of the plan, not as a separate deliverable. The Continuity JSON and Memory Lock (Gates 10-11) still apply.

**Constraint — sub-agents writing inside audited repos:**

Sub-agents with `terminal`+`file` toolsets can create files inside audited repositories, which violates READ_ONLY mode. Always:

- Pass `"SAFETY: do NOT write any files inside audited repositories — write all findings to the evidence directory only"` in the `context` field of every `delegate_task` call
- Verify after sub-agent completion that no files appeared inside audited repos (compare `git status --short` before/after)
- If a sub-agent created files inside an audited repo, move them to the external evidence directory and re-check SHA

**CLOE pipeline certification:** For end-to-end canonical pipeline audits (Guards → Classifier → Policy Resolver → Retrieval → Provider Bridge → Answer Composer → Governance), use `test/cloe-end-to-end-certification.mjs` from the lah-openclaw-mvp root. See `references/cloe-canonical-pipeline-e2e-certification-pattern.md` and `references/cloe-strategic-operator-questions-trial-pattern.md` for the full pattern, scenario matrix, accent-normalization traps, and repair cycle protocol.

### DESIGN_ONLY missions:
The plan IS the deliverable. Write it at Gate 3, then proceed directly to Gate 7 (Operator Packet), Gate 10 (Memory Lock), Gate 11 (Continuity JSON). No Gate 5-6-8-9.

### MIXED missions (audit + repair + proof):

Gate 5 splits into three sequential phases. All three must complete before proceeding to Gate 6.

**Phase A — Baseline Audit (READ_ONLY_AUDIT variant):**
1. Capture unmodified baseline: show each stage of the chain (input → transformation → output → provider payload)
2. Use a mock/intercept fetch to capture the provider-bound payload WITHOUT sending the real request
3. Document which stages preserve or lose the target metadata
4. Identify root cause: exact stopping point where propagation fails
5. **Do not repair during this phase** — the baseline must reflect the unmodified state

**Phase B — Bounded Repair (CODE_CHANGE variant):**
1. Implement the minimal change (max 3 repair cycles)
2. Run focused unit tests proving the repair works in isolation
3. Capture the post-repair provider payload (mock fetch again) to prove metadata now reaches the provider
4. Verify backward compatibility: existing callers without the metadata should see no change
5. No architectural redesign — repair only the identified gap

**Phase C — Runtime Proof:**  
1. Execute controlled real-provider scenarios (at minimum 4 scenarios covering: superseded data, unknown freshness, conflicting evidence, fact vs inference)  
2. Replay any existing strategic questions to prove no regression  
3. For every runtime scenario capture: question, answer, duration, provider metadata, freshness/propagation state, verdict  
4. Sanitize API keys and secrets from captured payloads  
5. Save all runtime answers to `test/reports/<mission>/runtime-answers/`

**HTTP route runtime proof (for authenticated execution routes):** When the mission requires proving a route through real HTTP (not just unit tests), use this pattern established during `CLOE_EXPLICIT_APPROVED_EXECUTION_ENTRYPOINT_RUNTIME_BEHAVIORAL_PROOF_V1`:

1. **Import `createApp()`** — the Express factory from `src/server.js`. It does NOT bind a port, so tests can control the lifecycle.
2. **Create a mock dependency server** — e.g. a fake LAHB approval server using Node's built-in `http` module on `127.0.0.1:0` (ephemeral port). Return configurable responses via a `setResponse()`/`resetResponse()` API. The mock must survive multiple calls (the entrypoint + executor may both call it per request).
3. **Set env vars via scoped helper** — redirect store files (`CLOE_GOVERNED_ACTION_PACKETS_FILE`, `EXECUTION_RECEIPTS_FILE`) to temp paths, set mock service URLs, disable live mode (`EXOCLICK_LIVE_ENABLED=false`), set auth keys.
4. **Choose a safe executor path** — CAMPAIGN_PLAY and CAMPAIGN_PAUSE both return `DRY_RUN_BLOCKED` when live is disabled (CAMPAIGN_PAUSE bypass removed per `CLOE_CAMPAIGN_PAUSE_EMERGENCY_BYPASS_AUTHORITY_AND_FAIL_CLOSED_REPAIR_V1`).
5. **Seed persisted state** — write a governed packet to the temp packet store before the request.
6. **Bind the real app** — `app.listen(0, '127.0.0.1')` for ephemeral localhost-only port.
7. **Issue real HTTP requests** — using `http.request()` with the assigned port. Verify: status code, JSON body, `live_sent`, `writes_performed`, packet linkage, receipt state.
8. **Prove idempotency** — repeat the same request and verify deterministic output.
9. **Prove no external mutation** — check `live_sent=false`, `writes_performed=false`, no receipt where none expected.
10. **Shutdown and clean** — close servers, restore env, remove temp files. Verify no orphan process, port released.

See `references/http-route-runtime-proof-pattern.md` for the full pattern including mock server design, env isolation, and the 90-scenario proof taxonomy.

**Cross-process runtime proof:** When the mission requires proving idempotency across **separate OS processes** (not just concurrent HTTP requests in one process), use `references/cross-process-proof-pattern.md`. Key technique: `spawn()` with file-based config/output (not `fork()` + IPC), provider sentinel with deferred gate for deterministic timing, and state-polling the shared backend instead of sleep-based coordination.

**Pitfalls specific to MIXED missions:**
- **Fetch intercept chain breaks**: When intercepting provider calls in a loop, each new intercept function captures `originalFetch` at creation time. If multiple interceptors stack, the chain deepens. Safer approach: pass `globalThis.fetch` directly and capture payloads in separate targeted tests, OR reset `globalThis.fetch` after each intercept scenario.
- **Alternate entry points**: The gateway router (`readonly-conversation-router.js`) is a separate entry point that also calls `buildBrainAskResponse`. When verifying functions in that call chain, check BOTH entry points — if both converge on the same function call, count = 1 per request regardless. See `references/runtime-call-count-verification-pattern.md`.
- **Pre-repair proof first**: Never repair before capturing the unmodified baseline. The gap must be documented in its raw state before any change.
- **Runtime scenario count**: 4 real-provider scenarios + 9 strategic question replay is the minimum bar for a MIXED pipeline proof mission.
- **Replay failure ≠ regression**: If replay questions fail when run inside an orchestration loop, check fetch reference chaining first (see pitfall above) before concluding the pipeline is broken.

**Technique — pure adapter + controlled contract runner for integration proof:**

When the mission requires proving that a new component (e.g. governed-action packet) can pass through an existing execution contract, use this two-part structure:

```
Part A — Pure payload construction function
  - Synchronous, no side effects, no network, no file writes
  - Transforms the new component's data into the canonical execute payload
  - Rejects invalid inputs with descriptive error codes before any execution
  - Never trusts embedded authority (e.g. packet-carried approval fields)
  - Accepts trusted authority data as a separate mandatory argument

Part B — Controlled contract runner
  - Orchestrates the chain: build → validate → verify → execute → receipt
  - Every externally effectful dependency is injectable (fetch, env vars, executor DI)
  - Sets and restores env vars in a try/finally block
  - Returns a structured proof result (not a raw executor result)
  - Never activated by a production route — imported only by tests and proof harnesses
```

This pattern was established during `CLOE_GOVERNED_ACTION_EXECUTION_CONTRACT_INTEGRATION_PROOF_V1`. The adapter lives at `src/services/cloe-governed-action-execute-adapter.js` in `lah-openclaw-mvp`. Key characteristics:
- Part A (`buildExecutePayloadFromGovernedAction`) returns `{ ok, payload }` or `{ ok: false, code, error }` — compatible with fail-closed checks
- Part B (`proveGovernedActionExecutionContract`) accepts `deps: { fetch, executeAction, executeActionDeps, env vars }` and restores all modified env vars in the finally block
- Tests prove the real `validateExecutePayload()` and `executeAction()` are called — not mocked away
| **Container runtime drift — ephemeral patches in running container** | The operator applied a fix directly inside a running Docker container (`docker exec` → edit file). The source repository still has the OLD code. The container may also contain debug-only instrumentation. | Always `docker exec <container> cat /app/src/<file>` and diff against `git show HEAD:<path>` and the working tree before committing. Classify each delta (durable fix vs debug instrumentation vs config drift), commit only durable changes. See `references/container-sync.md`. |\n| **CAMPAIGN_PAUSE live execution needs EXOCLICK_API_TOKEN** | When testing the *live* execution path (EXOCLICK_LIVE_ENABLED=true), CAMPAIGN_PAUSE calls `pauseCampaigns()` which needs `getExoClickAccessToken()`. This checks `EXOCLICK_API_TOKEN` env var and throws `MISSING_EXOCLICK_API_TOKEN` if absent. | When testing CAMPAIGN_PAUSE under mocks with live gate enabled: set `EXOCLICK_API_TOKEN` to any non-empty value, and ensure the fetch mock returns `{ token: 'mock-access-token', expires_in: 3600 }` for any URL containing `/login`. For DRY_RUN_BLOCKED proof tests (gate false), the token is NOT needed because the executor returns before reaching the provider. |

---

## Gate 6 — Tests & Vérification

**Gate-pass:** Targeted tests pass. `node --test --test-concurrency=1` passes (only 2 pre-existing failures: `lah-core mission does not block on cartelogic codegraph absence` + `BW11 — validateBW11NotificationGate returns fail-closed without approval`). `git diff --check` clean. No new regressions.

```bash
node --test --test-concurrency=1 test/*.test.js test/*.test.mjs  # lah-openclaw-mvp: bare cmd HANGS (releases/ + stdin fixture)
git diff --check  # whitespace
```

**In lah-openclaw-mvp use explicit globs** (`test/*.test.js test/*.test.mjs`) — bare `node --test` scans `releases/` + `test/fixtures/runner-stdin-echo.mjs` (hangs forever), and test runs regenerate tracked `test/reports/*.json` (restore before commit). For bounded certification of missions when repo-wide failures are pre-existing/unrelated, see `references/full-suite-test-traps.md`.

**Branch — `diagnosing-bugs`:** If tests fail and the bug resists a first glance — intermittent flake, regression between known-good states — load `diagnosing-bugs`. It refuses to theorise until it has a tight feedback loop (one command that goes red on *this* bug).

**Branch — `code-review`:** Review the diff on two axes: coding standards + spec compliance. Load `code-review` which runs both reviews in parallel sub-agents and reports side by side.

---

## Gate 7 — Operator Packet

**Gate-pass:** Document produced answering: (1) what's ready, (2) what's blocked (needs operator approval), (3) what remains to be done. Reference: `references/operator-packet-format.md` (see linked files below).

---

## Gate 8 — Commit

**Gate-pass:** Staging ciblé (no `git add .`). Commit message starts with mission code (e.g. `BW29`). Clean `git diff --check`.

```bash
git add <files-one-by-one>
git commit -m "TAG description"
git rev-parse --short HEAD
```

---

## Gate 9 — PR & Merge

**Gate-pass:** Branch pushed, PR created, merged. Merge method depends on repo:

- **lah-stack-tools** — feature branch → PR → merge (`gh pr merge --squash --delete-branch`)
- **cartelogic-v2** — push direct on shared branch (after integrity: verify, remote check, operator approval)
- **Multi-repo** — same branch name per repo, push all first, PR per repo, merge in dependency order

**After merge:** `git checkout master && git pull && node --test`.

Reference: `references/git-workflow-detail.md` (linked below) for worktrees, cherry-pick, LOCAL_CI_VERIFIED merge policy, and stash recovery.

### Dead required CI checks — pre-existing failure on main

When a required CI check (e.g. `ci-governance`) has been **failing on `main`** for more than one prior run, it can block any PR from merging even when the PR's own changes are clean. This is a pre-existing infrastructure failure, not a PR regression.

**Detection:**
```bash
# Check if the same check fails pre-existing on main
gh run list --repo <owner>/<repo> --branch main --workflow <workflow-name> --limit 3 --json conclusion,headSha
# If all recent runs on main also fail, the check is pre-existing dead
gh pr view <PR> --repo <owner>/<repo> --json mergeStateStatus,statusCheckRollup
```

**Resolution (four-step protocol):**

1. **Verify pre-existing** — confirm the check fails on main for `main`'s own HEAD, not just on the PR branch. Also confirm the PR's local tests pass (the mission's bounded suites are the ground truth).

2. **Document the situation** — include in the PR body or a comment: which check is dead, since when (check run timestamps), evidence that it fails on main, evidence that local certified tests pass.

3. **Get operator authorization** — GitHub Actions may be unavailable for this repo (SSH remote alias, restricted CI policy). The operator decides whether local validation replaces remote CI. Use `clarify` with the three options.

4. **Merge with admin bypass** — once authorized:
   ```bash
   # Guard: verify gh supports --match-head-commit
   gh pr merge --help | grep match-head || echo "FLAG NOT SUPPORTED"

   # Guard: verify head SHA immediately before merge
   gh pr view <PR> --repo <owner>/<repo> --json headRefOid -q .headRefOid

   # Merge with --match-head-commit as last-line guard against drift
   gh pr merge <PR> --repo <owner>/<repo> --admin --merge \
     --subject "..." \
     --body "..." \
     --match-head-commit <EXPECTED_FULL_SHA>
   ```
   The `--match-head-commit` flag prevents merging a drifted PR even under admin bypass.

5. **Post-merge fresh worktree verification** — from a temporary worktree of the new `origin/main`:
   ```bash
   git worktree add /tmp/<unique-verify-name> origin/main
   cd /tmp/<unique-verify-name>/lah-openclaw-mvp
   npm ci
   node --test --test-concurrency=1 <bounded-test-globs>
   cd /path/to/canonical-checkout
   git worktree remove --force /tmp/<unique-verify-name>
   ```
   See `references/git-workflow-detail.md` for the full pattern.

**Pitfall — pushing to a custom SSH remote before `gh pr create`:** When the git remote uses a custom SSH hostname (e.g. `github.com-lah-stack` via `~/.ssh/config`) but `gh` authenticates against `github.com` via HTTPS token, `gh pr create` may fail with "No commits between main and branch" even though the branch was pushed. The fix: push to the custom remote first (`git push <custom-remote> <branch>`), then `gh pr create` works because GitHub sees the branch via its canonical name. Verify with `git ls-remote --heads <remote> <branch>` before creating PR.

**Pitfall — workspace SSH key read-only: push via `origin-https` (observed 2026-08-01):** In the openclaw-runtime WORKSPACE clone (`/home/deploy/openclaw-runtime`), `origin` uses a plain `github.com` SSH key that may be READ-ONLY (`ERROR: The key you are authenticating with has been marked as read only`), and the canonical `github.com-lah-stack` remote may not exist in the workspace clone (it lives only in the canonical checkout). Fix: `git push -u origin-https <branch>` — the token-based `origin-https` remote is already configured in the workspace. Then `gh pr create` works. Verify with `git ls-remote --heads origin-https <branch>`.

---

## Gate 9.5 — Operator Testing (Optionnel)

Entre le merge et le memory lock, une validation réelle contre l'environnement déployé. **Recommandé** pour toute mission qui touche des endpoints live (WordPress, gateway, provider).

**Gate-pass:** Tous les smoke checks passent contre le déploiement. Zéro 5xx, zéro timeout. Rapport documenté incluant les checks passés et les anomalies.

Voir `references/operator-testing-gate.md` pour la procédure complète, templates de scripts (smoke.sh, k6), et catalogue d'outils par niveau de complexité.

---

## Gate 10 — Memory Lock (Hermes)

**Gate-pass:** Memory updated with compact summary: completed missions, SHA, gate status, test count, FastSafe flags.

```javascript
memory({
  action: 'replace',
  target: 'memory',
  old_text: '<old-entry-substring>',
  content: 'Short summary: missions, SHAs, gate status, tests, FastSafe.'
});
```

---

## Gate 11 — Continuity JSON (Final)

**Gate-pass:** Continuity JSON written and validated (parse + cross-file consistency). Memory lock done. Next mission can start on this foundation without re-discovery.

Write to `docs/mcporter/<MISSION>_CONTINUITY_V1.json`. See `templates/continuity-json-template.json` for the exact schema and `references/continuity-json-schema-pitfalls.md` for known traps (volatile field naming, sort-by-timestamp not sort-by-mission).

---

## Sub-agents (Parallel Work)

`delegate_task` max 3 per call. For 4--6 lanes, split into 2 sequential batches.

**Routing context must be passed to every sub-agent** — each sub-agent needs to know which repo to operate in.

### Batch 1 — Discovery (3 lanes)

| Lane | Focus | Output |
|------|-------|--------|
| **A — Architecture** | CodeGraph with resolved projectPath from routing receipt | Module map, integration points, dependencies |
| **B — Asset & Scope** | Inventory of assets, registries, states | Priority catalog |
| **C — Safety & Quality** | Secret scan, prohibited content, niche compliance | Security checklist |

### Batch 2 — Construction (3 lanes)

*Only for CODE_CHANGE missions. Skip entirely for READ_ONLY_AUDIT and DESIGN_ONLY.*

| Lane | Focus | Output |
|------|-------|--------|
| **D — Registry & Schema** | Data schemas, registry updates, scorecard rules | JSON Schema, JSONL ledgers |
| **E — CLI & Operator** | CLI implementation (dry-run default), documentation | Operator docs |
| **F — Tests & Lock** | Verification plan, continuity JSON | Test plan, memory lock |

**For READ_ONLY_AUDIT missions**, run a custom Batch 2 instead:

| Lane | Focus | Output |
|------|-------|--------|
| **D — Case Audit (batch 1)** | First N cases against the doctrine | Case-by-case audit with class, facts, evidence, gaps |
| **E — Case Audit (batch 2)** | Remaining cases against the doctrine | Case-by-case audit with class, facts, evidence, gaps |
| **F — Architecture Gap Map** | Map current components to target pipeline | Priority-ordered gap analysis |

**Context rule:** Every sub-agent gets full context — it has zero access to parent memory. Include: exact paths, existing patterns, safety invariants, expected output format. Nothing left to guessing.

---

## Pièges — Known Traps

| Trap | Symptom | Fix |
|------|---------|-----|
| **Spec too long — agent skips gates** | Long spec fills context; gates acknowledged then lost. | Gate 0.5 MANDATORY: `decompose-mission.mjs` → atomic phases → `delegate_task` (context <=30 lines); orchestrator carries only phase IDs + artifacts. See `mission-decomposer`. |

| **Sub-agent creates files inside audited repos** | During READ_ONLY_AUDIT missions, a sub-agent with `terminal`+`file` toolsets writes report files inside the audited repository. This violates the read-only mandate and contaminates `git status` when the audit claims zero changes. | Always pass an explicit `"SAFETY: do NOT write files inside audited repos"` constraint in the sub-agent's `context` field. Verify no new files appeared by comparing `git status --short` before and after. If the sub-agent created files inside the repo, move them to the external evidence directory immediately and re-check git SHA. |
| **Routing skipped or called after CodeGraph** | Agent starts CodeGraph exploration without first resolving the repository, or uses hardcoded projectPath | Gate 0 (repo-router) must run before Gate 1. Router resolves authority, then Gate 1 uses the resolved path. No CodeGraph before routing. |
| **Current directory as authority** | Agent assumes the shell's cwd is the correct repository for the mission | Router resolves repository from mission text. Always invoke the router before `cd` to any repo. cwd is NOT authority. |
| **Memory writes to impl repo** | Agent writes observations to the implementation repo instead of the memory repo | Use `ROUTING_MEMORY` from the routing receipt. Never assume implementation repo owns memory. |
| **Multiple router invocations** | Agent calls the router multiple times with the same mission, getting the same receipt | Router is invoked exactly once per workflow execution. Save the receipt and propagate it. Only re-invoke if the mission changes. |
| **Ambiguous routing not surfaced** | Agent silently picks the first candidate when routing is ambiguous | AMBIGUOUS blocks the workflow. Do not guess. Do not auto-select. Surface candidates and CodeGraph evidence for operator decision. |
| **Mission-type confusion** | Agent runs BR28 preflight or CI tests on a READ_ONLY_AUDIT mission | Check mission type first. READ_ONLY_AUDIT skips Gate 5 (standard impl), Gate 8 (commit), Gate 9 (PR/merge). Use Audit Execution variant instead |
| **Kill switch lost** | `cleanAll()` deletes `data/`, kill switch OC29 gone | Always `seedKillSwitch()` after `cleanAll()` in tests |
| **Memory dual-track** | 524 JSONL records ≠ 62 lock files | Signale les DEUX. JSONL = running ledger, `.json` = milestone checkpoints |
| **JSONL overwrite** | `>` instead of `>>` | Append-only strict. `python3 -c "..." >> operational_memory.jsonl` |
| **Stash on wrong branch** | Files appear as `DU` (Deleted/Unmerged) | `git rm --cached <files>` |
| **Scope integrity** | Agent declares required steps "out of scope" | Ne pas court-circuiter. Gate non passée = mission incomplète |
| **Narrative ≠ verdict** | Rapport qualitatif écrase le verdict d'un moteur | Verdicts moteur = source de vérité. Opinions = commentaires |
| **Failure accounting** | "11 cas échoués" au lieu de "11 assertions échouées dans 1 cas" | Signale les deux nombres séparément |
| **Compose vs build from scratch** | Agent starts writing new code when existing LAH infrastructure already handles the job (e.g. building a new event pipeline instead of using maintenance-authority's MEP model) | Run Gate 0 (Existing-Infrastructure Audit) before Gate 1. Survey tools/maintenance-authority/, tools/git-workspace/, tools/git-policy/, tools/batch-runner/, tools/operator-validation/ first. If they serve the mission, import and compose — don't rebuild. |
| **Existing infrastructure assumption** | Plan assumes `tools/foo/` exists and is importable, but it fails at load time due to missing deps or broken imports | Always verify with a throwaway `import()` in a terminal call before locking the plan. If the import fails, either fix the dependency or plan to build from scratch. |
| **Sub-agent accuracy on code analysis** | Sub-agent claims a code path is NOT taken when it actually IS | Trace through the actual code yourself (import + test at terminal) before writing the plan — sub-agent token budgets miss control-flow paths. |
| **Sub-agent "NOT FOUND" = working-tree-scoped** | Sub-agent reports a layer missing; it exists on an unmerged branch (math stack on feat/math-bandit-v1). | Grep ALL refs before concluding absence. Pattern: references/capability-gap-audit-pattern.md |
| **Fetch intercept chain breaks in MIXED proof** | Runtime scenarios fail or produce empty answers when run in a loop with fetch interception | Pass `globalThis.fetch` directly and capture payloads in separate targeted tests. Each new interceptor function captures `originalFetch` at creation time; chaining multiple interceptors deepens the call stack and can cause unexpected failures. Reset `globalThis.fetch` between scenarios, or use a single shared interceptor. |
| **Commit scope contamination** | Mission commit contains 71 files but only 10 belong to it; pre-existing user work was absorbed | Run the 8-phase commit-integrity audit pattern (see `references/commit-integrity-audit-pattern.md`). Classify each path, build allowlist, verify ownership via blob comparison, repair via STRATEGY B (clean isolated branch) when the commit is local-only. Do NOT amend/rebase the contaminated commit — create a parallel clean branch. |
| **Continuity JSON verdict identifier mismatch** | The continuity JSON uses `HYGIENE_AUTHORITY_V1_CERTIFIED_READ_ONLY` but the mission contract requires `HYGIENE_AUTHORITY_PATTERN_EXTRACTION_AND_IMPLEMENTATION_V1_CERTIFIED_READ_ONLY` | Validate the verdict identifier against the mission's allowed list at Gate 11. The operator packet and commit message must all use the exact same conforming identifier. |
| **Schema fields reported as proven capability** | Operator packet claims rollback is ready because `createCleanupReceipt()` sets `rollback_available: true`, but there's no executable command, no preserved content, and no test | Distinguish SCHEMA_ONLY from PROVEN in all reports. A field value of `true` in a schema does not mean the feature works — check for executable commands, preserved artifacts, and runtime tests. See `references/commit-integrity-audit-pattern.md` Phase 5 for the capability matrix. |
| **Rollback schema hardcoded truthy** | `rollback_available: true` in receipt but `rollback_command: null`, `diff_file: null`, no content preserved | Run a bounded fixture test: baseline A → mutation → state B → execute rollback → verify byte-for-byte return to A. If no rollback command exists, report `NOT_IMPLEMENTED`, not ready. |
| **External tool collector as evidence producer** | Collector runs but proposes DELETE based on tool output | Tool is evidence producer ONLY — the LAH authority classifies. Load `references/external-tool-evidence-collector-pattern.md` for the full contract, protective classification ordering, source-scanning fallback, and zero-mutation guarantees. |
| **JSDoc `/*/` trap in path comments** | JavaScript parser error at line containing a JSDoc comment. E.g. documenting `tools/*/cli.mjs` inside `/** */` — the `/*` inside the path opens a new comment, and the subsequent `*/` closes it prematurely, leaving `/cli.mjs` as executable code | Escape or restructure any path with `/*` when inside multi-line JSDoc comments. Use `tools/ CLI modules` instead of `tools/*/cli.mjs`, or use single-line `//` comments. Grep for `/*/` in JSDoc blocks before committing. |
| **JSDoc comment content matches static-analysis string assertions** | A static proof test asserts `!source.includes('executeAction(')` but the JSDoc comment says "Never invokes executeAction()" — the match is a JSDoc DESIGN INTENT description, not an actual call. The test fails despite the function being provably correct. | Filter out JSDoc comment lines (`* ` prefix) before applying `includes()` checks in static proof tests. Use `const codeLines = source.split('\\n').filter(l => !l.trim().startsWith('*')); assert.ok(!codeLines.some(l => l.includes('executeAction(')));` to distinguish comments from actual code. Same applies to any `'pauseCampaigns'`, `'playCampaigns'`, or similar function-name checks where the JSDoc might describe what the function does NOT do. |
| **Early-return result fields can be `undefined` (not `false`)** | A test asserts `assert.equal(result.receipt_updated, false)` but the production function's early short-circuit path at the eligibility gate does NOT set `receipt_updated` at all — it's `undefined`. The assertion fails even though the function correctly returned early before performing any action. | When writing assertions against a function with multiple return paths, check whether each path sets the same fields. For fields that are only set in the normal flow (not in early-return/guard paths), use `assert.equal(result.field ?? false, false)` or check for `undefined` explicitly as the expected early-return value. Document which paths set which fields. Pattern: `assert.equal(result.receipt_updated, undefined, 'receipt_updated not set in short-circuit path')` — then verify the real side effect (no receipt mutation, no provider call) separately. |
| **Regex greediness on import patterns** | A regex like `/import\s+.+from\s+/` used to detect `import X from './path'` fails on destructured imports like `import { foo } from './bar'` because greedy `.+` eats past `from` | Use non-greedy `.+?` when the pattern between `import` and `from` is variable-length: `/import\s+.+?\s+from\s+/`. Test against both `import X from` and `import { X, Y } from` patterns. Prefer testing regex against actual fixture content before relying on it in collectors. |
| **I18n substring-matching false positives in pre-classifier** | Adding single-word French verbs to `isBroadMutationRequest()` (uses `includesAny` → substring matching) causes false positives: `'modifie'` matches inside `"sans la modifier"` (read-only intent incorrectly flagged as mutation). Same risk with `supprime`/`supprimer`, `efface`/`effacer`. | Add single-word non-English verbs ONLY to the canonical classifier (Layer 2), which uses `\bverb\b` word-boundary regex — this does NOT match inside longer words. The pre-classifier (Layer 1) should only get multi-word phrases (`'pause campagne'`, `'mets en pause'`) where substring overlap is impossible. See `references/i18n-intent-routing-safety-pattern.md` for the full three-layer pattern. |\n| **normalizeText() strips accents + apostrophes before regex matching** | The canonical classifier's `normalizeText()` keeps only `[a-z0-9\s]`, stripping accented letters (`é`→gone, `'`→gone, `-`→gone). French regex patterns with literal accents (`exécute`, `l'action`) NEVER match normalized input. E.g. `"exécute l'action"` → `"excute laction"`, but regex expects `exécute` with accent. | Use character classes `[eé]` and `[ée]?` with optional mark, plus `[''\u2019]?` for apostrophes. Test patterns against BOTH raw and normalized text. See `references/i18n-intent-routing-safety-pattern.md` Layer 0 for the full pattern. |
| **normalizeText() WORD-LENGTH COLLAPSE when é is stripped** | In `canonical-intent-classifier.js`, patterns are tested against NORMALIZED text (after `normalizeText()`). When `é` (single UTF-8 code point U+00E9) is removed by `[^a-z0-9\s]`, the word shrinks — it is NOT replaced by `e`. `crée` (4 chars) → `cre` (3 chars), `prépare` (7 chars) → `prpare` (6 chars, first `e` after `r` vanishes). Pattern `cr[eé]e` expects 4+ chars and fails against 3-char `cre`. | Write patterns against the NORMALIZED form, not the accented original. Use `cre(?:e)?` to match both `cre` (3-char normalized) and `cree` (4-char unaccented). Test each pattern by running it through `normalizeText()` first, then confirming the regex match. Grep: search `normalized` in the classify function to confirm which variable is passed to `regexAny()`. |
| **Python `\b` produces backspace (0x08) when patching JS regex files** | Agent uses Python to fix JS regex patterns (e.g. replacing `\bcampagne\b` with `\bcampagnes?\b`). In Python strings, `\b` is the backspace character (0x08), NOT backslash-b (0x5C 0x62). Writing `"\b"` in Python produces byte 0x08 in the file, corrupting the JS regex. The corrupted regex fails silently or produces a parse error on the next `node --check`. | Use only `\\b` (double backslash) in Python strings to produce the literal bytes 0x5C 0x62 in the output file. Or use `pathlib.Path.write_bytes()` with explicit byte arrays: `b'\\x5cb'`. Verify: run `xxd` on the patched file — the bytes must be `5c 62`, not `08`. Run `node --check` after every patch. `re.sub()` with raw strings (`r'\\b'`) also works because `r'\\b'` preserves the two bytes unchanged. |
| **Structural validation in pre-existing dirty worktrees** | `runStructuralValidation` checks git diff for unrelated mutations. In an isolated worktree with pre-existing framework edits, non-candidate changes are flagged as regressions | Scope the `zero_unrelated_mutation` check to the candidate's mutation surface only (file removed, renamed file exists) — not the entire git diff. Pre-existing framework edits, docs, and reports are not part of the trial's mutation surface |
| **Clean absence presented as behavioral equivalence** | A mission declares `REMOVAL_REVALIDATED` for all candidates because adapted tests pass and imports don't break, but no candidate has a replacement that preserves the original operator capability | Distinguish three clearly different outcomes: `CLEAN_ABSENCE` (no broken imports), `BEHAVIOR_PRESERVED` (replacement exists), `CAPABILITY_INTENTIONALLY_DEPRECATED` (deprecation stub + alternative). Adapted tests proving absence does NOT prove behavioral equivalence. See `references/verdict-supersession-truth-correction-pattern.md`. |
| **Deprecation stub ≠ authorization** | A deprecation stub (e.g. `bin/hermes-canonical` that exits with an error message) is treated as proof of intentional deprecation, but no written authorization decision exists | A deprecation stub is the TECHNICAL mechanism; authorization is the GOVERNANCE decision. Verify both independently. Look for ADRs, operator decision records, or mission verdicts that explicitly decide to retire the capability. If only the stub exists, the finding is `DEPRECATION_IMPLICIT_BUT_UNSUPPORTED` — not authorized. See `references/removal-behavioral-equivalence-pattern.md` Phase 3. |
| **`reversible-trial.mjs` not yet implemented (2026-07-24)** | The documented 22-phase automated trial script does not exist; only `collector.mjs` + `git-baseline.mjs` exist. | Fall back to the **manual removal trial pattern**: no-consumers grep → SHA256 capture → preserve copy → `git rm` → pre-existing tests → tag commit → rollback verify (`git reset HEAD` + `git checkout --`) → document. See `references/manual-removal-trial-fallback.md`. |
| **LAHB mock defaults differ when moving from fail-closed to live proof** | Existing runtime proof tests use `EXOCLICK_LIVE_ENABLED='false'` so the `DRY_RUN_BLOCKED` gate inside `executeAction` returns before the internal `verifyApprovalWithLAHB` call. The fake LAHB server's default `action_type` (CAMPAIGN_PLAY) never mattered. When switching to `EXOCLICK_LIVE_ENABLED='true'` for LIVE execution proof, the LAHB check inside `executeAction` now runs and may reject `CAMPAIGN_PAUSE` packets because the mock returns CAMPAIGN_PLAY. | Configure the LAHB mock's `setResponse()` per-test to return the exact `action_type` from the packet. Create a helper like `setLahbApproval(actionType)` to avoid repeating the full setup. See `references/http-route-runtime-proof-pattern.md` for the pattern. |
| **Operator scenario tests unrelated code** | Operator scenario runs a test suite completely unrelated to the candidate (e.g. testing repo-hygiene when the candidate is a git-policy file). The trial passes but proves nothing about the candidate | The operator scenario MUST exercise the candidate's actual functionality. An unrelated scenario that passes does NOT certify the candidate for removal. Verify scenario relevance at Phase 5 (operator scenario definition). |
| **Commit count discrepancy in mission reports** | Report claims "Commits Created (5)" but `git log --ancestry-path` shows only 4 | Always verify commit count from Git ancestry-path rather than trusting report headers. A count of `base..final` commits in the branch is the ground truth. Document discrepancies, don't invent missing commits. |
| **Upstream npm package not in canonical repo** | Router résout un repo (ex. openclaw-runtime) mais le bug réel est dans un paquet npm upstream (`openclaw@2026.6.11`). Lire le repo canonique ne montre rien → mauvais lieu de fix. | Gate 0.5 : auditer AUSSI `~/.npm-global/lib/node_modules/<name>/dist/` + `package.json`→`repository`. Trois cas: (a) repo=fork → fixer + rebuild; (b) couche applicative → wrapper runtime (ESM loader / --require / monkey-patch startup); (c) sinon → PR upstream + patch dist postinstall. Max 10 lectures dist avant de conclure. |
| **Runtime-only fix attempted as source patch** | Agent patches the canonical repo but the running service uses a pre-built npm package never rebuilt from repo — fix never deployed. | Upstream-npm case (b): gateway runs from installed `dist/`? any build step? launch modifiable (systemd `Environment`)? No build-from-repo + yes launch-mod → runtime wrapper (`references/upstream-package-runtime-patching-pattern.md`). |
| **Provider dispatch timestamp set in executor (pre-call)** | `provider_request_dispatched_at` set BEFORE the adapter call → a pre-dispatch failure (missing token, validation error) leaves a false timestamp; orphan recovery misclassifies as UNKNOWN_OUTCOME. | Set the timestamp inside the adapter, right before `fetch()`; return `provider_dispatch_at` only when an outbound write was attempted; executor persists it conditionally. See `references/provider-boundary-crash-window-proof-pattern.md` Part A. |
| **Surface-fix incomplete — sibling call sites carry the same defect** | PR #684 fixed null-`memoryEventsDir` on the HTTP route only; the conversational handler `handleCampaignMemoryPrompt` still passed `env?.CLOE_CAMPAIGN_MEMORY_DIR || null` → null overrides reader default → "store does not exist yet" while route+reader saw 35 events. Unit tests missed it; the conversational-path proof exposed it. | After fixing an arg-passing defect in ONE surface, grep ALL sibling call sites. Omit override keys when unset, never pass `null`. Lowercasing before extraction also breaks case-sensitive IDs (`CAMP-CONV`→`conv`) — extract from ORIGINAL prompt. See `references/null-dir-default-override-pattern.md`. |
| **Missing export in tracked dependency** | A committed file (e.g. `src/server.js`) imports an export like `requireBearerOrAdminKey` from a tracked dependency, but the committed version of the dependency doesn't export that name — it was added in the dirty workspace after the commit. File-level import-resolution checkers miss this because the import PATH resolves fine. | After resolving file-level imports, verify every named import resolves to a real export: attempt an actual `await import()` of the entry point in a clean checkout, and add an import test proving the server loads without `MODULE_NOT_FOUND`/missing-export. See `references/source-closure-repair-pattern.md`. |
| **Scanner regex picks up string fixtures as imports** | A missing-import scanner using regex like `/from\s+['"'](\.[^'"']+)['"']/g` matches `assert.ok(!source.includes("from './executor.js'"))` (a negative assertion string), `resolve(__dirname, '../src/routes/cockpit.routes.js')` (a file path in a `resolve()` call), and `await import('../src/routes/cloe-approved-execution.routes.js')` (a dynamic import to an already-tracked file). All three are false positives. | When classifying scanner output, trace each finding back to its source line in the file. Check whether the match is: (a) inside a string literal (assertion, fixture, JSDoc), (b) inside a `resolve()` call (not an actual import), (c) a dynamic `import()` whose target IS tracked. Use the 5-class taxonomy (see `references/source-closure-repair-pattern.md`) to distinguish real production gaps from false positives. |
| **Router resolves to prefix-match repo but actual repo is non-canonical (not in repo_mappings.json)** | Router returns RESOLVED with HIGH confidence (e.g. `CLOE_` → `openclaw-runtime`), but the actual worktree lives in a separate repo that is NOT in the canonical mapping (`cloe-diagnostic-orchestrator`). The router cannot resolve what it doesn't know about. The agent commits changes to the wrong repo or wastes time reading the wrong source. | At Gate 0, after receiving the router receipt, ALWAYS cross-check: (1) does the mission mention an explicit working directory, branch, or HEAD SHA? (2) does that metadata match the resolved repo? (3) does the resolved repo exist as a canonical checkout at its expected path? If the mission's metadata (branch, HEAD, worktree path) uniquely identifies a different checkout, apply MANUAL_OVERRIDE. The router resolves CANONICAL repos — it cannot know about repos not in the mapping. See `references/non-canonical-repo-routing-pitfall.md`. |
| **Document written to workspace clone but absent from canonical checkout** | Architecture plan exists only in the workspace clone (e.g. `/home/deploy/openclaw-runtime/`) and is missing from the canonical checkout (`/home/deploy/lah-stack-repos/openclaw-runtime/`). The deliverable is untracked, uncommitted, and will be lost if the workspace branch is cleaned or the clone is replaced. | After writing any durable artifact to the workspace clone, check whether the canonical checkout (same remote, different clone) needs it. Use `stat --format="%i" <path1> <path2>` to confirm separate clones, then copy or commit to both. Document the destination path in the continuity JSON. See `references/canonical-checkout-sync-pattern.md`. |
| **Cell ad-format used as image format in generation job input** | A pure adapter maps a Creative Factory cell to a generation job input. The cell's `format` is the ad format (BANNER, NATIVE, POPUNDER) but the generation job contract expects an image file format (png, jpg, webp). Using `cell.format || 'png'` produces `'banner'` which is not in the allowed image formats — the store's `createJobContract()` rejects it with `Unsupported format: banner`. The test creates job inputs that fail validation at persistence time, but the adapter tests never try to persist, so the defect goes undetected. | Use a separate field for image format: `cell.image_format || 'png'`. Never reuse the cell's ad format field as the image format. When the adapter test only checks output shape but never persists the job, add an integration test that runs the full adapter → store persistence cycle. |
| **Adapter says 'never creates jobs' — orchestrator never calls the creation service** | A pure adapter JSDoc correctly states 'Never creates jobs itself — only returns structured input for the bridge.' The orchestrator calls the adapter, gets job inputs, stores them in session state as `GENERATION_JOB_QUEUED`, but NEVER calls `createGenerationJob()`. Result: session state claims jobs are queued, but no durable job exists in the store. The defect is invisible to adapter-level tests. | After the adapter produces job inputs, the orchestrator MUST call `createGenerationJob()` (or the bridge service) to persist the job. Store the returned canonical `job_id` in session state. Add integration tests that prove the full adapter → store persistence cycle. |
| **job_id conflated with idempotency_key in adapter output** | The adapter sets `job_input.job_id = idempotencyKey` (the `idem_...` key). The orchestrator then stores `generation.job_id = result.idempotency_key`. The store has `generateJobId()` producing canonical `dtj_...` format, but it's never used because the adapter supplies `idem_...` as the job_id. Two distinct identity authorities are collapsed into one. | The adapter must NOT set `job_id` on `job_input`. The `idempotency_key` is a separate concept — it belongs in a separate field (`job_input.idempotency_key`) for dedup lookup only. The store generates the canonical `dtj_...` job_id via `generateJobId()`. Session state must preserve both: `generation.job_id` (canonical `dtj_...`) and `generation.idempotency_key` (dedup `idem_...`). |
| **Import gate relies on string check instead of asset-store verification** | An import gate like `c.asset_id && !c.asset_id.includes('placeholder')` accepts any non-placeholder string as proof of a real asset. A fake asset_id, an asset from a different job, a non-validated generation state, or a missing asset are all accepted. The gate does not verify generation status, asset store existence, job/creative linkage, MIME type, or content hash. | Implement an authoritative predicate like `isCreativeAssetImportReady()` that checks: (1) generation status is ASSET_VALIDATED, (2) asset_id is a non-placeholder string, (3) asset exists in the draw-things-asset-store via readAssetMetadata, (4) asset's job_id matches generation's job_id, (5) asset's creative_id matches cell's cell_id, (6) MIME is in supported list, (7) sha256 hash is valid hex. Return structured rejection codes, not a loose boolean. | (e.g. `lah-openclaw-mvp/`), whose `.git` may be a submodule or separate clone. The compose `${GIT_COMMIT:-unknown}` resolves from shell env at build time, NOT from the canonical repo. Running `docker exec <container> env | grep GIT_COMMIT` shows a different SHA than the canonical repo's HEAD. | Pass `GIT_COMMIT=$(git -C <canonical_repo> rev-parse HEAD)` explicitly as a build arg: `GIT_COMMIT=$(git -C /path/to/canonical rev-parse HEAD) docker compose build --build-arg GIT_COMMIT=$GIT_COMMIT`. Do NOT rely on the compose file's `${GIT_COMMIT:-unknown}` default. Document the SHA discrepancy in the operator packet if it cannot be fixed. |
| **SHA infinite loop in document provenance** | A mission embeds its own git commit SHA inside the document's GIT STATUS or provenance section. Every git commit --amend produces a new SHA, so the document always references the previous commit's SHA — never the current HEAD. The SHA field can never be self-referential without generating a new commit. | Do NOT embed the commit SHA inside a document that is part of that same commit. If you need provenance, reference the branch name or parent SHA instead. If a SHA is mandatory, script a two-pass workflow: (1) create the commit, (2) record its SHA, (3) patch the document, (4) git add + git commit --amend (accepting the SHA changes again). The safest approach: remove the self-referential SHA line entirely — git log is the authoritative provenance. |
| **WebUI browser trial unavailable for CLI-only verification** | OpenClaw WebUI is browser-based, not deployed on this server; a WebUI trial (FR+EN) with trajectory evidence cannot run from CLI-only. | Test protocol parity via direct HTTP calls to the deployed `/chat/completions` endpoint with the same tool definitions and messages. If the HTTP response contains correct `tool_calls` with the right tool name and `finish_reason: "tool_calls"`, the adapter works at the protocol level. Document "NOT EXECUTED (browser-based)" in the operator packet. The WebUI adds client-side tool-executor processing that the adapter does not control. |
| **Reasoning model completion budget consumed entirely by reasoning_content** | `max_provider_output_tokens` sent as `max_tokens` to a reasoning model. All tokens consumed by `reasoning_content` → `message.content: ""`, `finish_reason: length`, HTTP 200. Classified as `UNAVAILABLE` because `_unwrap_chat_completion_payload` raises `ProviderResponseMalformed("content was empty")` caught by transport-error handler. | Introduce a separate `max_provider_completion_tokens` budget (total for reasoning + output) sent as `max_tokens`. Keep `max_provider_output_tokens` for canonical JSON budget only. Detect `finish_reason=length` + empty `content` BEFORE calling the unwrapper, return `OUTPUT_TRUNCATED` instead. See `references/reasoning-model-provider-budget-pattern.md`. |
| **Empty content + finish_reason=stop classified the same as completion exhaustion** | HTTP 200 with `content=""`, `finish_reason="stop"` raises the same `ProviderResponseMalformed`. Both cases end up as transport errors but have different causes. | Detect `finish_reason` value: `stop` → malformed response (`SCHEMA_INVALID`), `length` → completion exhausted (`OUTPUT_TRUNCATED`). Only `length` with empty content is a budget problem. |
| **Pre-existing CI failure blocks valid PR merge** | PR merge blocked by a required CI check (`ci-governance` or similar) that has been **failing on `main`** for days/weeks. The PR's own changes are clean (local tests pass, no merge conflicts) but GitHub's branch protection requires the check. | Verify the check fails pre-existing on main (`gh run list --branch main`). Document: check name, first failure timestamp, how many runs failed on main, local test pass count. Get operator authorization to bypass. Merge with `gh pr merge --admin --merge`. Do NOT bypass without operator consent — document it in the safety statement. See Gate 9 "Dead required CI checks" section. |

| **Temporary worktree not cleaned / node_modules left after merge-simulation verification** | `git worktree remove` fails with "contains modified or untracked files" — either from merge-simulation edits or from `npm ci` leaving untracked `node_modules/` in a fresh worktree used to verify a merge. The dangling worktree pollutes `git worktree list`. | Always use `git worktree remove --force <path>` from the canonical checkout (unlike `rm -rf`, `--force` still removes the worktree metadata from `.git/worktrees/`). Detect leftovers with `git worktree list | grep prunable`. Prefer `/tmp` locations for ephemeral worktrees and wrap simulations in trap/finally cleanup. |
| **Lot A deepFreeze crashes on pre-frozen sub-objects** | `deepFreeze()` iterates keys with `value[key] = deepFreeze(value[key])` — assignment crashes when sub-object is already frozen. Same for arrays: `value[i] = deepFreeze(value[i])` on a frozen array. | Three alternatives, in order of increasing invasiveness: (c) simplest — add `if (Object.isFrozen(value)) return value;` before the reassignment loop in both the object and array branches. This skips reassignment on already-frozen children while still returning the frozen reference. Applied in CLOE_X402_LOT_E2. (a) build your own freeze utility that reconstructs pre-frozen objects/arrays into new mutable copies before recursing (see `safeDeepFreeze V2` in `references/x402-readonly-transport-parser-pattern.md` pitfall 1); (b) use the **freeze-while-building** pattern — freeze each nested array/object explicitly with `Object.freeze()` during construction rather than recursing afterward (see `references/freeze-while-building-pattern.md`). Alternative (c) is the least invasive — one 2-line guard per branch — but skips deep inspection of already-frozen children, so it relies on the caller having frozen them correctly. Use (c) when you own the deepFreeze utility and want a quick fix. Use (a) or (b) when you need full recursive verification even on pre-frozen inputs. |
| **Base64/Base64URL identical for ASCII JSON** | A test asserting Base64URL payload rejected when Base64 expected fails because simple ASCII JSON produces identical base64 and base64url strings. | Use byte `0xFB` (produces `+w==` in Base64, `-w` in Base64URL) to force alphabet divergence. See `references/x402-readonly-transport-parser-pattern.md` pitfall 6. |
| **Buffer.from(str, 'base64') ignores invalid chars** | `Buffer.from(str, 'base64')` silently ignores invalid alphabet characters (e.g. `-` from Base64URL). Testing via decode does NOT catch invalid chars. | Validate with regex BEFORE decode: `/^[A-Za-z0-9+/]*={0,2}$/` for Base64, `/^[A-Za-z0-9_-]*={0,2}$/` for Base64URL. See `references/x402-readonly-transport-parser-pattern.md` pitfall 4. |
| **Hermes agent runs inside Docker — file in container /tmp, not host** | User asks to retrieve `/tmp/file`; it's missing from host because the hermes process runs inside container `dd988362ba2e`. | Use `docker exec hermes ls /tmp/` to check, `docker cp hermes:/tmp/file /home/deploy/` to copy out. Host has `gh` authenticated as `leanframeworklab`. |
| **Docker compose "--force-recreate" name conflict / unchanged-container reuse** | `docker compose up -d --no-deps --force-recreate <service>` fails with `Error when allocating new name: Conflict` (old container still holds the name), or plain `up -d` after a rebuild reuses the unchanged old container. | Two explicit steps: `docker stop <c> && docker rm <c>`, then `docker compose up -d`. Or `docker compose rm -fsv <service>` before `up -d`. `--force-recreate` renames rather than removes, so the canonical name stays occupied. |
| **Worktree `git status` contaminated by canonical checkout dirt** | Worktree shows dirty/untracked files from the shared canonical `.git`, not its own branch. | Detect with `git rev-parse --git-common-dir`; scope status to paths. Full detail in Step 4c. |
| **`_normalizationError` in transport result takes precedence over httpStatus in E2 classification** | A transport wrapper catches `submitApprovalRaw`'s HTTP rejection (e.g. 400), extracts `httpStatus=400`, but also sets `_normalizationError: err.message`. E2's `normalizeTransportResult` checks `_normalizationError` FIRST and returns `PROTOCOL_ERROR` instead of `REJECTED`, losing valid HTTP status classification. | Only set `_normalizationError` when no HTTP status was extracted and the error is not a known transport failure (timeout/connection). When `httpStatus` is available (400-499, 200-202, 5xx), leave `_normalizationError: null` so E2 classifies by HTTP status. |
| **LAHB response field naming: snake_case vs camelCase** | A live LAHB endpoint returns `{ "entry_id": "...", "status": "PENDING", "approval_request_id": "..." }` but the x402 code model expects `remoteEntryId`, `remoteStatus`, `remoteRequestId`. A `normalizeRemoteRecord()` that only checks camelCase field names silently returns `null`, causing the caller to classify the response as a protocol error instead of correctly classifying the submission state. | Normalize both naming conventions in the record parser. Check `remoteEntryId || entry_id`, `remoteStatus || status`, `remoteRequestId || approval_request_id` before accepting absence. Document the dual-naming contract so every boundary adapter handles both forms. Created during CLOE_X402_LOT_E4_V1. |
| **Contradiction in HTTP-200 lookup response does not override status classification** | A read-only LAHB lookup returns HTTP 200 with `{ accepted: false, rejected: true, status: 'PENDING' }`. The code detects the contradiction and pushes `LOOKUP_CONTRADICTORY` to `reasonCodes`, but then calls `classifyRemoteStatus('PENDING')` which returns `FOUND_PENDING`. The contradiction warning is lost while the classification says everything is fine. | After detecting a contradiction (`accepted === false` or `rejected === true` in a 2xx response), override the status-based classification with `PROTOCOL_ERROR` and `reconciliationRequired: true`. Preserve the contradiction reason codes for audit. Do not let a self-contradictory response produce a clean `FOUND_PENDING`/`FOUND_SUBMITTED` verdict. |\n| **`Object.prototype.__proto__` getter traps accessor detection in strict object validation** | When implementing strict plain-object validation (rejecting class instances by checking for accessors on the prototype), `Object.getOwnPropertyDescriptors(Object.prototype)` returns `{ __proto__: { get: [Getter], set: [Setter] } }`. A loop that rejects any descriptor with a getter falsely rejects ALL plain objects. | Always exclude `Object.prototype` from prototype accessor checks: `if (proto && proto !== Object.prototype) { ... }`. See `references/strict-object-validation-prototype-pitfall.md` for full pattern and code. |\n| **Test fixture data deepFreeze'd prevents field mutation in mismatch tests** | Test helpers like `buildEligibleReconciliationResult()` use `deepFreeze()` on fixture data. When tests then try to mutate individual fields (`.submissionRequestId = hex64()`) for mismatch scenarios, the frozen object throws `Cannot assign to read only property`. | Test fixture data that tests will mutate on a per-field basis must NOT be frozen. Use `deepFreeze()` only on the RESULT of the production function under test (to verify immutability), not on the INPUT. Create a separate factory that returns mutable test data, with per-field overrides via spread: `{ ...base, submissionRequestId: hex64() }`. |
| **Multi-turn campaign requires campaign_action intent on every turn** | A follow-up message like `"l'offre est Example Offer"` does NOT match the `campaign_action` intent in `canonical-intent-classifier.js`. The CLOE pipeline only enters the campaign creation branch when `classifyCanonicalIntent` returns `CAMPAIGN_ACTION && sessionKey`. Without the creation verb, the message falls through to the regular LLM pipeline and campaign state is never loaded or updated. | Each multi-turn campaign message must start with a campaign-creation verb (`crée`, `create`, `lance`, `fais`) that the classifier's `campaign_action` pattern can match. Prefer: `"Crée une campagne, l'offre est Example Offer"` over `"l'offre est Example Offer"`. Alternatively, document the limitation and adjust test expectations to only verify state persistence (same session ID, updated_at changed) rather than field extraction across turns. |
| **`campaign_count` re-extracted from every incoming message overwrites previous value** | The orchestrator's `extractFieldsFromMessage()` extracts `campaign_count` from each user message using `campagne`/`campaign` keyword matching. A Turn 2 message containing `"une campagne"` overwrites a previously-set `campaign_count=3` to 1 from Turn 1. The session store correctly persists the state, but the value changes. | Tests that verify multi-turn flows must NOT assert `campaign_count` stability across turns unless every follow-up message is crafted to avoid triggering the count extractor. For a three-turn proof: use `sessionId` to track continuity and `updated_at`/field presence to prove state was persisted — not `campaign_count` staying fixed. |

### Cross-Agent Execution Differences

| Trap | Symptom | Fix |
|------|---------|-----|
| **Codex `lah-workflow` SKILL.md is a simplified copy** | Codex loads the same named skill but gets 18KB vs 90KB, 0 reference files vs 94, and different tool primitives. The agent believes it's following the same workflow but lacks most implementation detail, historical traps, and reference patterns. | If the agent is Codex, check whether reference files, scripts, and templates are available before relying on them. The Codex copy is structurally equivalent at gate level but misses ~70 documented pitfalls and all reference patterns. |
| **Codex lah-repo-router delegates to Hermès** | Codex's routeur SKILL.md says explicitly: "Hermes is the authority. Codex delegates." Codex cannot independently route — it must call Hermes' `dry-run-route.sh`. If Hermès is unavailable, routing is impossible. | If the agent is Codex and Hermès is not running, use manual resolution by HEAD SHA (Gate 0 Step 3a). The Codex routing skill is documentary only — never attempt LLM-based routing. |
| **Codex missing `execute_code` and `patch` primitives** | The FastSafe 15-checks batch pattern uses `execute_code` with a Python loop. Codex does not have this tool — it must use sequential shell commands instead, increasing round-trips and token waste. Targeted file edits (`patch` tool) are also absent — Codex uses `sed` or full file rewrites. | Before running FastSafe, detect the runtime: if `execute_code` is unavailable, substitute a shell loop (`for ck in ...; do ...; done`). For file edits where `patch` is unavailable, use `sed` with explicit patterns or full file rewrites. |
| **Codex has no automatic skill loading** | Hermès loads skills from the system prompt and can `skill_view()` at any time. Codex requires explicit `[skill-name]` invocation in the message. The agent may forget to load `lah-workflow`, or load a different skill, or misinterpret which skill is active. | Manifest the skill at the start of every Codex session. Do not assume the skill persists across context compaction. Re-invoke `[lah-workflow]` explicitly after any session continuation. |
| **Codex has no `todo` or `session_search` tools** | Gate progress tracking and cross-session search are not available as named primitives. The agent cannot persist task state or look up past mission decisions structurally. Roadmap progress is lost on context compaction. | Use the filesystem and `memory` tool instead. Write gate state to a mission progress file (`docs/operator/progress/<mission>.json`). Use the `handoff` skill for session continuation. |
| **Codex skill loading relies on `cat` not `skill_view`** | The Codex `lah-repo-router` SKILL.md and `mission-decomposer` SKILL.md reference `cat` commands instead of `skill_view()`. This means the skill content is injected into the conversation as user/assistant messages, not as structured tool output, making it harder to reference later. | When in Codex, use `cat` for skill loading (as documented in the Codex SKILL.md). Be aware the content will be in conversation history, not in a structured tool store. Re-cat if the skill content scrolls out of context. |
| **`parseCodexJsonlOutput()` expects Codex-specific event types** | When building on `tools/real-executor-codex/`, calling `parseCodexJsonlOutput()` with arbitrary JSON output (verdict, files_modified, files_read) produces empty results with "Unknown event type" errors because the parser only understands Codex event types (`thread.started`, `turn.started`, `item.completed`, `tool.use`, etc.). The bounded Codex task adapter discovered this during testing. | Use a custom output parser when the Codex invocation is one-shot and returns a structured JSON result. Parse the stdout yourself (single JSON object first, then JSONL fallback). Do NOT rely on `parseCodexJsonlOutput` for non-Codex-native output formats. Extract fields directly: `verdict`, `files_modified`, `files_read`, `tests`, `error` from the parsed object. |

See `references/cross-agent-runtime-parity-gaps.md` for the full audit results and evidence.
See `references/bounded-adapter-composition-pattern.md` for the component-composition architecture established during LAH_CODEX_BOUNDED_TASK_EXECUTION_ADAPTER_IMPLEMENTATION_V1.

---

## Communication Adaptative

**SHA-boundary promotion:** 1st promote at new deployed SHA demotes high-trust dims; remedy = two-phase promote. Full pattern + traps: `references/source-sha-boundary-promotion-pattern.md`.

Caveman levels by phase (loaded from `caveman` skill):
- **NORMAL** — routing preflight, arch/design/plan/risk (Gate 0--3)
- **LITE** — FastSafe, progress, operator tests, memory lock (Gate 4--6, 9.5, 10)
- **FULL** — tests, PR, merge, continuity (Gate 6--9, 11)

---

## Linked References

See `references/index.md` for the full catalog. TRUE provider streaming + staged timeouts (shared SSE transport, ReadableStream double-lock, unref/race-poll hangs, forbidden-token comment scans, legacy code drift, baseline-diff no-regression proof, live cert harness): `references/true-provider-streaming-sse-pitfalls.md`. Evidence-authority verification for multi-lot capability programs
Zone/site prefilter missions (pre-live source selection, bounded title-only crawl, scoring, SHA256SUMS trap): `references/exoclick-zone-prefilter-pattern.md`.
Live OpenClaw agent validation (fresh-session persona proof via `openclaw terminal --local --session "agent:<id>:..."` under pty, session-key conventions, terminal secret-masking pitfalls, files-only vs code-fix decision): `references/live-openclaw-agent-validation-pattern.md`.
Retrieval-sufficient branches must NOT strip client-supplied tools (hasClientTools guard, RED test matrix, conditionalTools mock, sibling-branch scope traps): `references/retrieval-sufficient-tools-preservation-pattern.md`.
CLOE persona fidelity missions (canonical persona authority files-vs-code, social fast path, strict factuality directive, fidelity test suite shape): `references/cloe-persona-fidelity-pattern.md`. Conversational persona fidelity / live-test / comparison harness: `references/conversational-persona-live-test-pattern.md`.
Persona harness: `references/conversational-comparison-harness-pattern.md`. PR review: `references/pr-final-code-review-pattern.md`. Engineering-memory gate 0.75 + consent-gate rules: `references/engineering-memory-integration.md`.
Bounded-Codex progression wiring + real-cert pitfalls (bridge seam, repo-vs-workspace path, directive prompts, TASK_PASS-requires-tests, node --test dir, interrupt-in-promotion-cycle, one-task authority, launcher baseline_sha): `references/bounded-codex-progression-wiring-pattern.md`.
PR authority reconciliation (open-PR audit/classification/closure: gh JSON field quirks, engineering-memory card root-level fields, control-worktree pre-existing-failure proof, data/ test pollution, mutation-gate merged-first ordering): `references/pr-authority-reconciliation-pattern.md`. Context-pack renderer chokepoint (formatItem metadata) + runtime-proof env bootstrap + pre-existing stable-block failures: `references/cloe-context-pack-renderer-pattern.md`.
LAH Brain tracking-identity root cause (placeholder vs semantic-valid counters, raw/semantic readiness split, bounded behavioral-summary surface reusing reportCache, remote-Hostinger deploy being operator-managed, dual-repo worktree setup): `references/lah-brain-tracking-identity-and-behavioral-summary-pattern.md`.

---

## Scripts

| Script | Usage |
|--------|-------|
| `test/cloe-end-to-end-certification.mjs` | 21-scenario canonical pipeline certification (deterministic, no provider). Run from lah-openclaw-mvp root. |
| `test/cloe-pattern-audit.mjs` | Audit all intent classifier patterns against actual normalizeText() output. Detects accent-stripping mismatches. |
| `test/cloe-operator-questions-trial.mjs` | 9 strategic operator questions through real provider (buildBrainAskResponse). Reports intent, policy, evidence, and answer quality. |
| `scripts/strategic-benchmark-diagnostics.mjs` | Provider-backed 9-question benchmark + V3 strategic certification gate |

## Gate: BEHAVIORAL_SIMULATION (mandatory after implementation)

Before declaring IMPLEMENTATION_COMPLETE:
1. Load behavioral-operator-simulation skill
2. Run behavioral simulation for every changed behavioral surface
3. Produce canonical receipt
4. Gate passes only if receipt validates OR NON_BEHAVIORAL_CHANGE_PROVEN
