# Understand-Anything Incorporation Strategy

> **Status:** Phase 1 complete / verified / committed (`24356edcd`). Phase 2 D1-D3 complete / reviewed / committed / pushed (`5a39c7fc7`). D4 is deferred by default and requires explicit JC approval.
> **Related:** See `understand-anything-to-flywheel-review.md` for the full risk analysis and artifact inventory.
> **Last updated:** 2026-05-30

## Scope

Adopt high-value ideas from Understand-Anything into Hermes/Flywheel without context bloat. No always-on giant prompts. No dashboard. No TypeScript rebuild. Python native. JIT-loaded skills only.

## What To Adopt (ranked by value/cost)

### 1. Script-First / LLM-Second Analysis Pattern

**Value:** High. Deterministic scripts handle file enumeration, language detection, import maps, and category assignment. LLM only synthesizes narrative fields (description, framework story). Drastically reduces token spend and hallucination risk.

**Form:** A `scripts/code-scan/` module under the repos root (or `.hermes/` tool scripts). Two scripts:
- `scan_project.py` — walks project, applies `.hermesignore` filtering, assigns language/category per extension tables, counts lines, estimates complexity. Returns JSON.
- `extract_imports.py` — reads the scan output, parses imports via regex/tree-sitter (reuse existing LSP infrastructure where possible), returns import map JSON.

The LLM orchestration skill reads these JSON artifacts + manifests and synthesizes only what's non-deterministic (project name, description, framework narrative).

**Context cost:** Near-zero. The agent sees a 20-40 line JSON summary, not the scan logic.

### 2. Schema-Validated Graph with Type Alias Normalization

**Value:** Medium-High. UA's Zod schema (schema.ts) plus NODE_TYPE_ALIASES map is excellent quality control. When graph-like structures appear in Flywheel (dependency maps, module relationships), validate them.

**Form:** A lightweight Python schema module `scripts/code-scan/graph_schema.py` with:
- Node types enum (reuse 16 existing UA types, extend only as needed)
- Edge types enum (reuse subset relevant to code analysis — `imports`, `contains`, `calls`, `tested_by`, `configures`, `documents`)
- Alias map (`func`→`function`, `fn`→`function`, `doc`→`document`, etc.)
- Validation function that runs in code (no LLM) and returns issues list

**Context cost:** Zero — runs as a utility, never loaded into context.

### 3. Fingerprint-Based Incremental Re-Analysis

**Value:** High for large repos; low for small ones. Content hash + structural fingerprint = skip unchanged files. UA implements this in `fingerprint.ts` with `ChangeLevel` (NONE/COSMETIC/STRUCTURAL).

**Form:** A `.hermes/code-state/fingerprints.json` file (git-ignored) that stores per-file fingerprints: content hash, function names, class names, import sources. On re-analysis, compare fingerprints and only re-analyze files with STRUCTURAL changes.

**Context cost:** Zero — persistence layer only. Fingerprint comparison is code, not context.

### 4. Graph Reviewer / Validation Gate

**Value:** High. UA's graph-reviewer.md pattern: Phase 1 writes and executes a deterministic validation script; Phase 2 reads the script's JSON results and renders approval/rejection. Two-phase keeps the LLM out of counting/verification.

**Form:** Adapt as a **gate skill** (JIT-loaded, not always-on): `skills/code-analysis/validation-gate/SKILL.md`. The skill:
1. Writes a Python validation script (or loads a pre-bundled one)
2. Runs it against the graph/analysis output
3. Reads the JSON results
4. Renders approval/rejection + structured notes

Maps cleanly onto the existing **Revision gate** in the gates taxonomy. Critical issues→revision; warnings→notes; approved→proceed.

**Context cost:** ~50 lines when loaded (the skill definition). Deterministic checks live in script files, not in the prompt.

### 5. Language/Framework Registry

**Value:** Medium. Systematic mapping from file extensions, manifest patterns, and infrastructure file presence → language/framework/category. UA hardcodes these tables in `scan-project.mjs`.

**Form:** A static Python data file `scripts/code-scan/language_registry.py` — dictionaries of extension→language, manifest key→framework pattern, filename→category. Deterministic lookup, no LLM.

**Context cost:** Zero — utility module only. Agent uses the results, not the tables.

### 6. Assembly Reviewer Pattern (Post-Merge Validation)

**Value:** Medium. After parallel subagent analysis produces batch results, a reviewer checks: recovered dropped items, cross-batch gaps, sanity-checks mechanical fixes.

**Form:** Not a new artifact — just extend Subagent-Driven Development's existing review step to include "cross-batch edge recovery" as a reviewer checklist item when the plan involves parallelized analysis tasks. No new file needed.

**Context cost:** 4-6 lines appended to existing review prompts when parallel analysis is active.

---

## What To Avoid

| UA Feature | Reason to Skip |
|---|---|
| Dashboard (React, React Flow, dark theme) | Hermes is CLI/Discord-first; dashboard adds 20K+ lines of React/TS for niche use |
| Tour Builder / Onboarding Tours | Solves a different problem; Hermes has `agent/onboarding.py` and `mission-control/` already |
| Full Knowledge Graph as always-on data structure | Too heavy. Flywheel's local state protocol (`.hermes/PROJECT_STATE.md`) is the right granularity |
| Semantic Batching / Neighbor Maps | Only justified for >500-file repos; add as a Phase 3 optimization if needed, not upfront |
| Embedding Search (`embedding-search.ts`) | Not in Hermes's scope; memory system covers semantic retrieval |
| Always-on prompt loading | Violates JIT principle; the entire point of this strategy |
| Staleness tracking (`staleness.ts`) as separate module | Merge into fingerprint module; no need for separate concern |

---

## Artifact Boundaries

All adopted artifacts follow these boundaries:

| Artifact | Location | Loaded Into Context? | Trigger |
|---|---|---|---|
| `scripts/code-scan/scan_project.py` | Repo root `scripts/` | No (runs as subprocess) | Validation gate skill loads it |
| `scripts/code-scan/extract_imports.py` | Same | No | Same |
| `scripts/code-scan/graph_schema.py` | Repo root `scripts/` | No (utility) | Imported by validation scripts |
| `scripts/code-scan/language_registry.py` | Repo root `scripts/` | No (utility) | Imported by scan script |
| `.hermes/code-state/fingerprints.json` | Per-project `.hermes/` | No (disk only) | Written/read by scan + fingerprint code |
| `skills/code-analysis/validation-gate/SKILL.md` | Skills dir | Yes (~50 lines) | JIT-loaded when plan includes graph/structure verification step |
| `.hermesignore` | Project root | No | Read by scan script |

**No artifact exceeds 60 lines in context.** All deterministic logic lives in scripts/modules that run as code.

---

## JIT Loading Strategy

Skills are loaded via `agent/skill_commands.py` as user messages, not system prompt additions. The pattern:

1. **Scan skill** — loaded only when user asks "analyze this repo", "map this codebase", "understand this project", or when a plan bead requires code structure understanding.
2. **Validation gate skill** — loaded only when a bead/stage requires verification of graph-like output (post-scan review, pre-commit structure check).
3. **Fingerprint logic** — never loaded into context. Runs as a utility. The skill references it; the agent sees only the output.

Skills should use the existing SKILL.md frontmatter pattern with `hermes.tags` including `on-demand` for discoverability in the skill manifest.

---

## Phased Implementation

### Phase 1: Foundation (Week 1-2) — *Highest ROI, minimal risk*

**Deliverables:**
1. `scripts/code-scan/scan_project.py` — deterministic file enumeration, language detection, category assignment, line counts, complexity estimation, `.hermesignore` support
2. `scripts/code-scan/language_registry.py` — extension-to-language table, category priority rules, framework detection patterns
3. `scripts/code-scan/graph_schema.py` — node/edge type enums, alias map, validation function
4. `.hermesignore` default rules file (reuse existing patterns: `node_modules/`, `.git/`, `__pycache__/`, `dist/`, `build/`, `*.pyc`, etc.)

**Verification:** Run `scan_project.py` against 3 repos of different sizes/small/medium/300+ files. Output matches UA's `scan-project.mjs` results within 5%. No LLM involved.

**Context impact:** Zero. These are scripts that run as subprocesses.

### Phase 2: Orchestration (Week 3-4) — *Make it accessible to agents*

**Approval package:** `.plans/phase-2-flywheel-ua-integration.md` is the authoritative execution plan for Phase 2.

**Deliverables:**
1. `scripts/code-scan/extract_imports.py` — import extraction script (stdlib regex fallback; tree-sitter remains Phase 4)
2. `skills/code-analysis/code-scan/SKILL.md` — skill that orchestrates scan_project.py + extract_imports.py + bounded LLM narrative synthesis
3. `skills/code-analysis/validation-gate/SKILL.md` — two-phase reviewer (deterministic validation script + LLM decision rendering)
4. Optional integration with existing `requesting-code-review` skill to run scan + validation gate on changed files only when explicitly requested

**Owner:** Hermes coordinates; coder implements each approved slice; reviewer must review Phase 2 docs/skill/script diffs before commit.

**Execution gates:** Phase 1 deliverables must exist and pass first. No automatic context injection, no dashboard, no CLI command, no new runtime dependency, and no code-review integration unless separately approved inside the Phase 2 approval.

**Verification:** Agent running subagent-driven-development with a code-scan bead produces correct scan output and passes validation gate without hallucination. Phase 2 also requires targeted unit tests, context-budget checks (both skills ≤100 loaded lines total), and a smoke run against the agreed test-bed repos.

**Context impact:** ~100 lines total (two SKILL.md files loaded JIT).

### Phase 3: Incremental Analysis (Week 5-6) — *Optimize for re-analysis*

**Deliverables:**
1. `.hermes/code-state/fingerprints.json` persistence format
2. Fingerprint extraction + comparison logic (append to scan_project.py or separate module)
3. `--incremental` flag on scan that skips COSMETIC/UNCHANGED files
4. `scripts/code-scan/assemble-graph.py` — merges batch outputs into unified graph with deduplication, ID normalization, edge merging (adapt UA's mechanical fixes from merge-batch-graphs.py)

**Verification:** Re-running scan on a touched-file repo only re-analyzes changed files; fingerprint file updates correctly; assembled graph matches full-scan results.

**Context impact:** Zero. Persistence layer only.

### Phase 4: Optional — *Only if Phase 1-3 prove valuable*

**Deliverables:**
1. Import extraction using tree-sitter via existing LSP infrastructure
2. Cross-batch edge recovery skill (extends validation gate)
3. Neighbor map generation for >500-file repos
4. Staleness detection (compare git diff against fingerprints)

**No commit to Phase 4 until Phase 1-3 are in use and validated.**

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Scripts drift from UA canonical tables | Version pin the extension/category tables in comments; test against UA's output |
| Import extraction regex misses edge cases | Phase 3 uses tree-sitter via LSP (already in Hermes agent) |
| Validation gate false positives | Script outputs warnings vs issues separately; warnings don't block |
| Skills add context bloat | Each SKILL.md capped at 80 lines; deterministic logic in scripts |
| `.hermes/code-state/` becomes stale | Add staleness check in Phase 4; document in skill that user can force full re-scan with `--full` flag |

---

## Decision Log

| Decision | Rationale |
|---|---|
| Python, not TypeScript | Hermes is Python-native; no build chain changes needed |
| Scripts in `scripts/`, not `skills/` | Skills are prompts; scripts are code. Separate concerns |
| Schemas in `scripts/code-scan/`, not bundled per-skill | Shared utilities, loaded as modules, not context |
| Fingerprint format is JSON, not SQLite | Simple enough for a file; hermes_state.py already handles SQLite |
| Graph schema uses subset of UA types, not all 16 | Only `imports`, `contains`, `calls`, `tested_by`, `configures`, `documents` are useful in Flywheel context; extend as needed |
| No dashboard | Out of scope; Hermes has Discord + CLI for projection |
