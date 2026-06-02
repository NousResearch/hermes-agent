# Phase 3: Incremental Analysis — Fingerprint Persistence + Graph Assembly

> **Parent doc:** `.plans/ua-incorporation-strategy.md`
> **Prerequisite:** Phase 1 (Foundation) ✓ complete — committed `24356edcd`, 80 tests pass. Phase 2 (Orchestration) ✓ complete — D1-D3 evaluated 11/11 PASS, merged to `jc-fork/main` at HEAD `24e9fe65a`.
> **Status:** ✅ D1-D3 COMPLETE — merged to local main at `0133a0a4b` via PR #6 squash merge. CI: Tests ✅, Lint ✅, Nix ✅. D4 deferred by default.
> **Execution beads:** Defined in `.beads/phase3-d1-fingerprint-model.md`, `.beads/phase3-d2-incremental-scan.md`, `.beads/phase3-d3-assemble-graph.md`, `.beads/phase3-d4-skill-integration-deferred.md`.
>
> **These bead files are the authoritative execution units.** This plan describes intent and scope; the beads contain exact functions, schemas, test contracts, verification commands, and allowed/forbidden file lists. When executing, dispatch coder subagents using the bead files as the sole implementation spec.

---

## Objective

Add two capabilities that make the Phase 1/2 scan pipeline suitable for iterative use on the same project:

1. **Fingerprint persistence** — Store per-file content hashes and structural fingerprints (function names, class names, import sources) in `.hermes/code-state/fingerprints.json` (git-ignored). A comparison layer classifies each file as UNCHANGED / COSMETIC / STRUCTURAL between scans.
2. **Incremental scan mode** — A `--incremental` flag on `scan_project.py` that reads the previous fingerprint file, performs the normal deterministic file walk so the output schema remains fresh, classifies files as UNCHANGED / COSMETIC / STRUCTURAL, persists updated fingerprints, and reports the changed subset for downstream heavy analysis. Phase 3 does not preserve stale scan records from fingerprints.
3. **Graph assembly** — `scripts/code-scan/assemble_graph.py` merges batch subagent analysis outputs (scan JSON + import maps) into a unified dependency graph with deduplication, ID normalization, and edge merging. Adapts UA's `merge-batch-graphs.py` mechanical fixes into a stdlib-only Python script.

**Context budget:** Zero for fingerprint logic and graph assembly (pure scripts/modules). The existing Phase 2 skills (`code-scan`, `validation-gate`) gain optional new flags (`--incremental`) but no SKILL.md rewrite required — the skills reference the new flags only when the user requests incremental mode.

---

## Approval Scope

Approving Phase 3 authorizes Hermes to execute the full phase as a review-branch workstream, with slice-by-slice local verification and reviewer checks before any commit. It does **not** authorize merge, deployment, publishing, or production mutation.

### Included

1. **D1:** Define `.hermes/code-state/fingerprints.json` persistence format and implement fingerprint extraction + comparison logic as a new module `scripts/code-scan/fingerprints.py`.
2. **D2:** Add `--incremental` flag to `scan_project.py` that uses fingerprints for change detection, skips UNCHANGED/COSMETIC files, and merges results to produce full-schema output.
3. **D3:** Implement `scripts/code-scan/assemble_graph.py` that merges batch outputs into a unified graph with deduplication, ID normalization, and edge merging.
4. **D4 (Deferred by default):** SKILL.md updates to surface the `--incremental` flag in the code-scan skill. Documented for future work but not executed as part of Phase 3.

### Excluded / Deferred

- No dashboard, React UI, Vite server, graph visualization, or web endpoint.
- No automatic prompt/context injection and no always-on scanning.
- No new runtime dependency unless JC approves a dependency exception.
- No tree-sitter or WASM; regex/stdlib extraction only (fingerprint extraction reuses Phase 2 regex patterns).
- No SQLite/summary store or `flywheel scan` CLI command.
- No CLI command beyond `--incremental` and `--full` flags on `scan_project.py`.
- No commit/push/merge/deploy beyond the local/branch checkpoint JC explicitly approves.
- D4 skill-integration is a documented follow-up bead; execute only if JC explicitly includes it.

### Owners and Review

| Role | Responsibility |
|---|---|
| Hermes | Coordinates, maintains docs/state, verifies outputs, presents approval gates |
| coder subagent | Implements each approved slice; no commit/push authority |
| reviewer subagent | Reviews spec compliance, quality/security, context-budget, and scope preservation |
| JC | Approves full Phase 3 execution and separately approves commit/push/merge/deploy gates |

---

## Prerequisites

Phase 3 must not start until Phase 1 and Phase 2 D1-D3 are complete and verified:

- `scripts/code-scan/scan_project.py` exists and emits stable JSON (Phase 1, committed).
- `scripts/code-scan/language_registry.py` exists (Phase 1, committed).
- `scripts/code-scan/graph_schema.py` exists and validates node/edge contracts (Phase 1, committed).
- `scripts/code-scan/extract_imports.py` exists and emits import map JSON (Phase 2 D1, merged).
- `.hermesignore` exists with default exclusions (Phase 1, committed).
- Phase 1 tests (80) and Phase 2 tests (111/111) pass.
- `skills/code-analysis/code-scan/SKILL.md` exists (Phase 2 D2, merged).
- `skills/code-analysis/validation-gate/SKILL.md` exists (Phase 2 D3, merged).

---

## Test-Bed Repos

Use these local repos unless JC substitutes others before approval:

| Tier | Repo | Purpose |
|---|---|---|
| Small | `/home/jarrad/work/testbeds/ua-flywheel/cass_memory_system` | Fingerprint extraction, incremental smoke test |
| Medium | `/home/jarrad/work/testbeds/ua-flywheel/mission-control` | Import-based fingerprinting, graph assembly on mixed language |
| Large/current | `/home/jarrad/.hermes/hermes-agent` | Performance guardrail, deduplication on large batch outputs |

---

## Fingerprint Format (D1 contract)

### `.hermes/code-state/fingerprints.json` schema

```json
{
  "schema_version": "1.0.0",
  "project_root": "/path/to/project",
  "captured_at": "2026-05-30T12:00:00Z",
  "files": {
    "src/main.py": {
      "content_hash": "sha256:abc123...",
      "line_count": 150,
      "size_bytes": 4200,
      "functions": ["parse_config", "main", "_helper"],
      "classes": ["ConfigParser"],
      "imports": ["os", "sys", "json"],
      "change_level": "STRUCTURAL"
    }
  }
}
```

Required top-level keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `schema_version` | string | yes | Always `"1.0.0"` for Phase 3 |
| `project_root` | string | yes | Absolute path to scanned project |
| `captured_at` | string | yes | ISO 8601 timestamp |
| `files` | object | yes | Map of relative_path → fingerprint record |

Required per-file keys:

| Key | Type | Required | Description |
|---|---|---|---|
| `content_hash` | string | yes | SHA-256 hex digest of file content |
| `line_count` | int | yes | Physical line count (matches scan) |
| `size_bytes` | int | yes | File size in bytes |
| `functions` | list[str] | yes | Extracted function names (regex) |
| `classes` | list[str] | yes | Extracted class names (regex) |
| `imports` | list[str] | yes | Extracted import module names (reuse extract_imports patterns) |
| `change_level` | string | no | Only present after comparison: `UNCHANGED`, `COSMETIC`, or `STRUCTURAL` |

### Change-level classification

| Level | Criteria |
|---|---|
| `UNCHANGED` | `content_hash` matches previous scan exactly |
| `COSMETIC` | `content_hash` changed but `functions`, `classes`, and `imports` lists are identical (whitespace/comment/docstring-only edits) |
| `STRUCTURAL` | Any of `functions`, `classes`, or `imports` changed, OR file is new/deleted |

---

## Graph Assembly Format (D3 contract)

### `assemble_graph.py` output schema

```json
{
  "schema_version": "1.0.0",
  "generated_at": "2026-05-30T12:00:00Z",
  "source_files": ["batch1.json", "batch2.json"],
  "nodes": [
    {
      "node_id": "file:src/main.py",
      "node_type": "file",
      "filePath": "src/main.py",
      "label": "main.py",
      "language": "python",
      "functions": ["parse_config", "main"],
      "classes": ["ConfigParser"],
      "imports": ["os", "sys"]
    }
  ],
  "edges": [
    {
      "source": "file:src/main.py",
      "target": "module:os",
      "edge_type": "imports",
      "meta": {}
    }
  ],
  "summary": {
    "total_nodes": 42,
    "total_edges": 156,
    "deduplicated_nodes": 3,
    "deduplicated_edges": 7,
    "orphan_nodes": 5
  }
}
```

### ID normalization rules

- File nodes: `file:<relative_path>` (e.g., `file:src/main.py`)
- Module nodes: `module:<module_name>` (e.g., `module:os`, `module:react`)
- Function nodes: `func:<relative_path>:<function_name>` (e.g., `func:src/main.py:main`)
- Class nodes: `class:<relative_path>:<class_name>` (e.g., `class:src/main.py:ConfigParser`)

### Deduplication rules

1. Nodes with identical `node_id` are merged — preserve first-seen attributes, append unique `functions`/`classes`/`imports` from duplicates.
2. Edges with identical `(source, target, edge_type)` tuples are deduplicated — keep the first instance.
3. Orphan nodes (no edges) are flagged in `summary.orphan_nodes` but NOT removed.

---

## Rollback / Off-Switch Plan

- `.hermes/code-state/fingerprints.json` is git-ignored and written to a per-project dot-directory. If it becomes stale or corrupt, delete it and re-run a full scan.
- The `--incremental` flag is opt-in: running `scan_project.py` without `--incremental` performs a full scan as before (Phase 1/2 behavior unchanged).
- If fingerprint extraction fails for a language, emit a warning and classify as STRUCTURAL (safer than incorrect UNCHANGED).
- `assemble_graph.py` is a standalone script — it reads JSON inputs and produces JSON output. If it fails, the original batch inputs remain untouched.
- If the graph assembly produces incorrect results, run `scan_project.py --full` + re-assemble to regenerate.

---

## Deliverables

### D1: Fingerprint Model — `scripts/code-scan/fingerprints.py`

**Purpose:** Extract, compare, and persist file fingerprints. New module.

**Authoritative execution bead:** `.beads/phase3-d1-fingerprint-model.md`.

**Scope:**
- Extract content hash (SHA-256), function names, class names, and imports per file.
- Compare against persisted `.hermes/code-state/fingerprints.json`.
- Classify each file as UNCHANGED / COSMETIC / STRUCTURAL.
- Load and save fingerprint files.
- Zero new runtime dependencies; stdlib only.

**Acceptance criteria:**
- `fingerprints.py` module with extraction, comparison, and persistence functions.
- Tests cover hash matching, structural detection, cosmetic-only detection, new/deleted files.
- Output matches fingerprint format contract above.
- All tests in bead pass (RED → GREEN → FULL).

---

### D2: Incremental Scan — `--incremental` flag on `scan_project.py`

**Purpose:** Enable incremental re-scanning using fingerprints. Modifies `scan_project.py` only (adds flag + conditional logic).

**Authoritative execution bead:** `.beads/phase3-d2-incremental-scan.md`.

**Scope:**
- Add `--incremental` argparse flag to `scan_project.py`.
- When `--incremental` is set: load previous fingerprints from `.hermes/code-state/fingerprints.json`, perform the normal fresh file walk, build current fingerprints, classify files, and expose the STRUCTURAL subset for downstream heavy analysis.
- Emit the same fresh scan output schema as full mode; do not merge stale file records from fingerprints.
- Add `--full` flag to explicitly ignore any existing fingerprints and force a full scan (no-op when no fingerprints exist yet).
- First run with `--incremental` (no prior fingerprints) behaves identically to `--full`.

**Acceptance criteria:**
- Output JSON schema identical to Phase 1 full scan regardless of incremental mode.
- Incremental re-scan on an unchanged project produces the same data fields as a full scan on the same current tree, excluding timestamps and incremental warning metadata.
- Touching one file marks only that file STRUCTURAL while the normal scan output remains fresh for every file.
- Existing Phase 1 tests still pass.
- New tests for incremental behavior pass.

---

### D3: Graph Assembly — `scripts/code-scan/assemble_graph.py`

**Purpose:** Merge batch analysis outputs into a unified dependency graph.

**Authoritative execution bead:** `.beads/phase3-d3-assemble-graph.md`.

**Scope:**
- Read multiple scan/import JSON inputs (from batch subagent runs).
- Build nodes and edges with ID normalization (`file:`, `module:`, `func:`, `class:` prefixes).
- Deduplicate nodes by `node_id` — merge attributes.
- Deduplicate edges by `(source, target, edge_type)`.
- Validate final graph using `graph_schema.py`.
- Output unified graph JSON following the graph assembly format contract.

**Acceptance criteria:**
- Merged graph contains all unique nodes and edges from all inputs.
- No duplicate nodes or edges in output.
- `graph_schema.py` validation passes on output.
- Orphan nodes are counted and reported in summary.
- All tests in bead pass (RED → GREEN → FULL).

---

### D4: Skill Integration — DEFERRED by default

**⚠️ DEFAULT STATUS: DEFERRED.** Not included in standard Phase 3 approval scope. Execute only if JC explicitly includes D4.

**Authoritative execution bead:** `.beads/phase3-d4-skill-integration-deferred.md`.

**Purpose (if approved):** Update `skills/code-analysis/code-scan/SKILL.md` to optionally reference `--incremental` mode. When the user asks for a re-scan on a project that already has fingerprints, the skill should prefer `--incremental` by default and offer `--full` as an override.

**Scope (if approved):**
- Update `code-scan/SKILL.md` to mention `--incremental` and `--full` flags.
- Add conditional logic: check for existing fingerprints before choosing scan mode.
- Line budget: must still keep SKILL.md ≤80 lines total (may require condensing).

**Why deferred by default:**
1. SKILL.md line budget is tight (code-scan is already 39 lines).
2. The skill works correctly without incremental awareness; the flag is CLI-only for now.
3. Value of skill-layer incremental awareness is unproven until D1/D2/D3 ship as CLI features.

---

## Verification Plan (Summary)

| Test | Command / Method | Pass Criteria |
|---|---|---|
| D1: Fingerprint extraction | `pytest tests/code_scan/test_fingerprints.py -v` | All tests pass |
| D1: Format contract | Schema validator on fingerprints.json output | All required keys present, types correct |
| D1: Change classification | Known-good/bad file diffs → compare change levels | UNCHANGED/COSMETIC/STRUCTURAL classified correctly |
| D2: Incremental full scan | `scan_project.py <dir> --incremental` with no prior fingerprints | Output identical to `scan_project.py <dir>` without flag |
| D2: Incremental re-scan | Touch one file, re-run `--incremental`, compare classifications | Only touched file is STRUCTURAL; output schema remains identical to full scan |
| D2: --full override | `scan_project.py <dir> --full --incremental` exists or `--full` alone | Forces full re-scan regardless of fingerprints |
| D2: Existing tests | `pytest tests/code_scan/test_scan_project.py -v` | No regression |
| D3: Graph assembly | `pytest tests/code_scan/test_assemble_graph.py -v` | All tests pass |
| D3: ID normalization | Graph output validation | All node_ids follow normalization rules |
| D3: Deduplication | Merge inputs with intentional duplicates | No duplicate nodes or edges in output |
| D3: Schema validation | `graph_schema.py` validate on graph output | Zero issues |
| Full suite regression | `pytest tests/code_scan/ -q` | All tests pass (prior 111 + new tests) |
| Large-repo smoke | Run incremental + assembly on hermes-agent repo | Completes; produces valid JSON; incremental detects actual changes |
| Scope guardrail | Pattern search for excluded features | Zero matches |

---

## Phase 3 Deliverables Checklist

- [x] **D1:** Fingerprint model module + tests → `.beads/phase3-d1-fingerprint-model.md` — ✅ Merged at `0133a0a4b` via PR #6
- [x] **D2:** Incremental scan flag + tests → `.beads/phase3-d2-incremental-scan.md` — ✅ Merged at `0133a0a4b` via PR #6
- [x] **D3:** Graph assembly script + tests → `.beads/phase3-d3-assemble-graph.md` — ✅ Merged at `0133a0a4b` via PR #6
- [ ] **D4:** Skill integration (deferred) → `.beads/phase3-d4-skill-integration-deferred.md` — deferred by default
- [x] Verification: tests pass, fingerprint format contract met, graph output validated
- [x] Reviewer: spec compliance + scope preservation + quality/security + forbidden-file integrity
- [x] Approval: JC approved full Phase 3 execution (D1-D3; D4 deferred) — 2026-05-30T15:36:59Z

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Fingerprint false UNCHANGED (missed structural change) | STRUCTURAL is the safe default: only classify UNCHANGED on exact hash match. COSMETIC requires identical functions/classes/imports lists. |
| `--incremental` produces stale or divergent scan data | Contract test: run full and incremental on identical current input, compare core JSON data excluding timestamps and incremental warning metadata. |
| Graph assembly double-counts nodes from overlapping batches | Deduplication by normalized `node_id` is mandatory; contract test supplies intentionally overlapping inputs. |
| `.hermes/code-state/` directory not created | Script auto-creates the directory with `os.makedirs(exist_ok=True)` before writing. |
| Fingerprints become stale across branches | `project_root` in fingerprints.json anchors to a specific working directory; branch switches change file contents which triggers STRUCTURAL detection on next scan. |
| SKILL.md line budget exceeded if D4 is approved | D4 must condense existing skill content to accommodate incremental mode references; reviewer enforces ≤80 lines. |

---

## JC Approval Wording (copy-paste template)

```
Phase 3 UA Flywheel Incremental Analysis — Approval Decision

I approve Phase 3 UA Flywheel Incremental Analysis for autonomous execution
on branch `docs/ua-flywheel-phase3-plan`.

Approving:
  ☐ D1 (fingerprint model) — extraction, comparison, persistence module
  ☐ D2 (incremental scan) — --incremental/--full flags on scan_project.py
  ☐ D3 (graph assembly) — assemble_graph.py with deduplication
  ☐ D4 (skill integration) — optional, deferred by default

Scope limits:
  - Explicit invocation / JIT only: no dashboard, no React UI, no auto-injection,
    no SQLite store, no CLI command beyond --incremental/--full flags,
    no tree-sitter/WASM, no new runtime deps.
  - No commit/push/merge/deploy beyond the local branch checkpoint.
  - Coder subagents have no commit/push authority.
  - Forbidden files (skills_sync.py, test_skills_sync.py) must remain untouched.
  - Existing Phase 1/2 files may be modified only as authorized per bead
    (scan_project.py for --incremental flag; all other Phase 1/2 files forbidden).
  - D4 skill modification must keep SKILL.md ≤80 lines.

Verifier to run after execution: see Verification table above.
Reviewer subagent must return explicit PASS on spec compliance, scope preservation,
context budget, quality/security, and forbidden-file integrity.
```
