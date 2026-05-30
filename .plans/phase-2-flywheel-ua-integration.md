# Phase 2: UA Flywheel Integration — Orchestration Layer

> **Parent doc:** `.plans/ua-incorporation-strategy.md`
> **Prerequisite:** Phase 1 (Foundation) ✓ complete — committed `24356edcd`, 80 tests pass, scan scripts verified on test-bed repos.
> **Status:** D1-D3 complete, verified, reviewed, committed, and pushed (`5a39c7fc7`); D4 remains deferred pending explicit JC approval.
> **Execution beads:** Defined in `.beads/phase2-d1-extract-imports.md`, `.beads/phase2-d2-code-scan-skill.md`, `.beads/phase2-d3-validation-gate-skill.md`, `.beads/phase2-d4-review-integration-deferred.md`.
>
> **These bead files are the authoritative execution units.** This plan describes intent and scope; the beads contain exact functions, schemas, test contracts, verification commands, and allowed/forbidden file lists. When executing, dispatch coder subagents using the bead files as the sole implementation spec.

---

## Objective

Build the JIT skill layer that makes the Phase 1 scan scripts actionable by agents. Phase 2 delivers two skills (code-scan + validation-gate), the `extract_imports.py` script, and optional integration with `requesting-code-review`.

**Context budget:** ≤100 total lines when both skills are JIT-loaded (each SKILL.md ≤80 lines).

---

## Approval Scope

Approving Phase 2 authorizes Hermes to execute the full phase as a review-branch workstream, with slice-by-slice local verification and reviewer checks before any commit. It does **not** authorize merge, deployment, publishing, or production mutation.

### Included

1. Implement `scripts/code-scan/extract_imports.py` and tests.
2. Add `skills/code-analysis/code-scan/SKILL.md` and lightweight supporting references only if needed.
3. Add `skills/code-analysis/validation-gate/SKILL.md` and deterministic validation contract tests.
4. D4 deferred by default: prepare optional `requesting-code-review` integration as a documented follow-up only; execute it only if approval explicitly includes D4.

### Excluded / Deferred

- No dashboard, React UI, Vite server, graph visualization, or web endpoint.
- No automatic prompt/context injection and no always-on scanning.
- No new runtime dependency unless JC approves a dependency exception.
- No tree-sitter or WASM; regex/stdlib extraction only in Phase 2.
- No SQLite/summary store or `flywheel scan` CLI command; the older Phase 2 summary/CLI concept is deferred.
- No commit/push/merge/deploy beyond the local/branch checkpoint JC explicitly approves.

### Owners and Review

| Role | Responsibility |
|---|---|
| Hermes | Coordinates, maintains docs/state, verifies outputs, presents approval gates |
| coder subagent | Implements each approved slice; no commit/push authority |
| reviewer subagent | Reviews spec compliance, quality/security, context-budget, and scope preservation |
| JC | Approves full Phase 2 execution and separately approves commit/push/merge/deploy gates |

---

## Prerequisites

Phase 2 must not start until Phase 1 is complete and verified:

- `scripts/code-scan/scan_project.py` exists and emits stable JSON.
- `scripts/code-scan/language_registry.py` exists and is imported by `scan_project.py`.
- `scripts/code-scan/graph_schema.py` exists and can validate node/edge contracts.
- `.hermesignore` exists with default exclusions.
- Phase 1 tests pass on the selected test-bed repos.

---

## Test-Bed Repos

Use these local repos unless JC substitutes others before approval:

| Tier | Repo | Purpose |
|---|---|---|
| Small | `/home/jarrad/.hermes/hermes-agent/cass_memory_system` | Python package-scale scan/import extraction |
| Medium | `/home/jarrad/.hermes/hermes-agent/mission-control` | TypeScript/Node project with frontend-style imports |
| Large/current | `/home/jarrad/.hermes/hermes-agent` | Real Hermes repo smoke/performance guardrail |

If any test-bed path is moved outside this worktree before execution, update this table before starting Phase 2.

---

## Rollback / Off-Switch Plan

- Phase 2 is explicit invocation only: agents load `code-scan` / `validation-gate` only when requested or when a bead explicitly requires it.
- If a skill misbehaves, remove or revert only `skills/code-analysis/<skill>/` and keep Phase 1 scripts intact.
- If `extract_imports.py` fails on a language, return warnings and incomplete import coverage; do not block scan summaries unless JSON output is invalid.
- If performance exceeds budget, disable the optional D4 code-review integration first, then narrow language coverage before changing Phase 1 scanner behavior.
- No persistent project artifacts are written in Phase 2 except temporary scan/import JSON under an ignored output path agreed during implementation.

---

## Deliverables

### D1: `scripts/code-scan/extract_imports.py`

**Purpose:** Extract import/dependency maps from the scan_project.py output. Reads scan JSON, parses import statements, returns import map JSON.

**Authoritative execution bead:** `.beads/phase2-d1-extract-imports.md` contains exact function signatures, regex patterns, output schema contract, test fixtures, and verification commands.

**Scope:**
- Read `scan_project.py` JSON output
- Regex-based import extraction (start with regex; tree-sitter is Phase 4)
- Support: Python (`import X`, `from X import Y`), JavaScript/TypeScript (`import X from 'Y'`, `require()`), Rust (`use X::Y`), Go (`import "pkg"`), shell/bash (`source`, `.`)
- Output: structured import map JSON following the schema contract below
- No LLM involvement

**Output schema contract (exact):**
```json
{
  "schema_version": "1.0.0",
  "source_scan": {
    "project_root": "/path/to/project",
    "total_files": 423
  },
  "generated_at": "2026-05-30T12:00:00Z",
  "files": {
    "src/main.py": {
      "imports": ["os", "sys", "pathlib"],
      "warnings": []
    }
  },
  "totals": {
    "files_with_imports": 156,
    "files_without_imports": 267,
    "unique_modules": 89,
    "total_warnings": 3
  }
}
```

Required top-level keys: `schema_version` (string, always `"1.0.0"`), `source_scan` (object with `project_root` + `total_files`), `generated_at` (ISO 8601), `files` (map of relative_path → `{ imports: string[], warnings: string[] }`), `totals` (object with `files_with_imports`, `files_without_imports`, `unique_modules`, `total_warnings`).

**Required functions:** `load_scan_output`, `iter_scanned_files`, `extract_python_imports`, `extract_js_ts_imports`, `extract_rust_imports`, `extract_go_imports`, `extract_shell_imports`, `extract_imports_for_file`, `build_import_map`, `main`.

**Acceptance criteria:**
- Runs standalone: `python scripts/code-scan/extract_imports.py <scan_output.json>`
- Correctly extracts imports from ≥5 supported languages
- Output is valid JSON matching the schema contract above
- Zero new runtime dependencies
- All test cases in `.beads/phase2-d1-extract-imports.md` pass (RED → GREEN → FULL)

---

### D2: `skills/code-analysis/code-scan/SKILL.md`

**Purpose:** JIT skill that orchestrates the scan pipeline. Agent loads this skill when analyzing a codebase.

**Authoritative execution bead:** `.beads/phase2-d2-code-scan-skill.md` contains exact frontmatter content, all 7 orchestration steps, output format, line budget constraints, and contract tests.

**Frontmatter:** Must follow existing SKILL.md convention with `hermes.tags` including `on-demand`.

**Behavior:**
1. On activation, confirms target project directory
2. Runs `scan_project.py` against the project
3. Runs `extract_imports.py` on scan output
4. Reads the JSON artifacts
5. Uses LLM to synthesize non-deterministic fields: project name, one-line description, framework narrative
6. Renders structured summary to user

**Acceptance criteria:**
- SKILL.md ≤80 lines (enforced by contract test in the bead)
- Skill loads via `agent/skill_commands.py` as user message (not system prompt)
- End-to-end scan of a 50-file project completes in <5s (script execution only; LLM synthesis excluded from timing)
- Agent produces correct scan output without hallucinating file structures
- All contract tests in `.beads/phase2-d2-code-scan-skill.md` pass

---

### D3: `skills/code-analysis/validation-gate/SKILL.md`

**Purpose:** Two-phase validation skill. Phase 1 runs a deterministic validation script; Phase 2 reads results and renders approval/rejection.

**Authoritative execution bead:** `.beads/phase2-d3-validation-gate-skill.md` contains exact frontmatter, two-phase behavior, validation checks, output format, line budget, graph_schema.py contract tests, and RED/GREEN/FULL evidence requirements.

**Behavior:**
1. Accepts a target artifact (graph JSON, scan output, or analysis result)
2. Runs a Python validation script against it (uses `graph_schema.py` for schema validation)
3. Reads the JSON validation results
4. Renders: APPROVED / WARNING / REJECTED with structured notes
5. If REJECTED with critical issues, triggers revision gate

**Acceptance criteria:**
- SKILL.md ≤80 lines (enforced by contract test in the bead)
- Validation script runs in <2s on typical outputs
- Warnings don't block; only critical issues trigger REJECTED
- Maps to existing Revision gate in gates taxonomy
- All contract tests in `.beads/phase2-d3-validation-gate-skill.md` pass

---

### D4: Integration with `requesting-code-review` — DEFERRED by default

**⚠️ DEFAULT STATUS: DEFERRED.** This deliverable is **not included** in the standard Phase 2 approval scope. It will only execute if JC **explicitly** includes D4 in the Phase 2 approval.

**Authoritative execution bead:** `.beads/phase2-d4-review-integration-deferred.md` contains the scoped implementation outline, constraints, and deferred verification plan. **No implementation work until JC grants explicit approval.**

**Purpose (if approved):** Extend the existing `requesting-code-review` skill to optionally run scan + validation gate on changed files in a PR/diff context.

**Scope (if approved):**
- Add an optional code-scan step when reviewing code changes
- Run `scan_project.py --changed` (or parse scan output for changed files only)
- Feed results into review skill context
- No new files — update existing skill only

**Acceptance criteria (if approved):**
- Only activates when the user requests code analysis as part of review
- Does not slow down normal review flow when code-scan is not requested
- Existing `requesting-code-review` tests still pass
- Modified SKILL.md still ≤80 lines total (may require condensing existing content)

**Why deferred by default:**
1. Modifying a production-critical skill introduces regression risk
2. Line budget may be tight
3. Value of scan-enriched review is unproven until D1/D2/D3 ship in isolation
4. JC can approve D4 after seeing D1/D2/D3 results

---

## Verification Plan

| Test | Command / Method | Pass Criteria |
|---|---|---|
| Unit: extract_imports.py | `pytest tests/tools/test_extract_imports.py` | All tests pass |
| Fixture coverage | Python, JS/TS, Rust, Go, shell fixture files | Expected imports match golden JSON |
| Integration: full scan pipeline | Agent loads code-scan skill, scans `cass_memory_system` and `mission-control` | Produces correct JSON, no hallucination |
| Large-repo smoke | Run scan/import pipeline against this Hermes repo | Completes without scanning ignored giant/vendor dirs; produces valid JSON |
| Integration: validation gate | Feed known-good and known-bad graph JSON to validation skill | Correct APPROVED/WARNING/REJECTED verdicts |
| Context budget check | Measure loaded SKILL.md total lines | ≤100 lines for both skills |
| Scope guardrail | Search diff for dashboard/React/tree-sitter/SQLite/auto-injection additions | No excluded feature enters Phase 2 |
| Existing tests | Targeted tests touched by Phase 2 plus any affected skill tests | No regression |

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Import regex extraction misses edge cases | Acceptable for MVP; Phase 4 adds tree-sitter |
| Skill files exceed 80-line limit during iteration | Enforce in review; split into sub-skills if needed |
| Scan scripts become a maintenance burden | Keep scripts focused; each <200 LOC |
| Validation gate false positives | Separate warnings from critical issues; warnings never block |

---

## Phase 2 Deliverables Checklist

- [x] Prerequisite: Phase 1 verified and committed (`24356edcd`, 80 tests pass)
- [x] D1: `scripts/code-scan/extract_imports.py` + unit tests → `.beads/phase2-d1-extract-imports.md` — complete; committed/pushed in `5a39c7fc7`
- [x] D2: `skills/code-analysis/code-scan/SKILL.md` (≤80 lines) → `.beads/phase2-d2-code-scan-skill.md` — complete; 39 lines; committed/pushed in `5a39c7fc7`
- [x] D3: `skills/code-analysis/validation-gate/SKILL.md` (≤80 lines) → `.beads/phase2-d3-validation-gate-skill.md` — complete; 48 lines; committed/pushed in `5a39c7fc7`
- [ ] D4: `requesting-code-review` integration — **DEFERRED** unless JC explicitly approves → `.beads/phase2-d4-review-integration-deferred.md`
- [x] Verification: tests pass, context budget met, scope guardrails pass
- [x] Reviewer: spec compliance + quality/security + scope preservation PASS (`.hermes/handoffs/2026-05-30-0648-phase2-d1-d3-review-pass.md`)
- [x] Approval: JC approved D1-D3 autonomous execution; D4 deferred

### JC Approval Wording (copy-paste template)

```
Phase 2 UA Flywheel Integration — Approval Decision

I approve Phase 2 UA Flywheel Integration for autonomous execution on branch `docs/ua-flywheel-phase1-phase2-plan`.

Approving:
  ☐ D1 (extract_imports.py) — import extraction script + tests
  ☐ D2 (code-scan SKILL.md) — JIT scan orchestration skill
  ☐ D3 (validation-gate SKILL.md) — two-phase reviewer skill
  ☐ D4 (requesting-code-review integration) — optional, deferred by default

Scope limits:
  - Explicit invocation / JIT only: no dashboard, no React UI, no auto-injection,
    no SQLite store, no CLI command, no tree-sitter/WASM, no new runtime deps.
  - No commit/push/merge/deploy beyond the local branch checkpoint.
  - Coder subagents have no commit/push authority.
  - Forbidden files (skills_sync.py, test_skills_sync.py) must remain untouched.
  - SKILL.md files must not exceed 80 lines.

Verifier to run after execution: see Verification table below.
Reviewer subagent must return explicit PASS on spec compliance, scope preservation,
context budget, quality/security, and forbidden-file integrity.
```
