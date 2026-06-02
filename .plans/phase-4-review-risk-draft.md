# Phase 4: Enhanced Analysis — Review & Risk Draft

> **Created:** 2026-06-01
> **Status:** DRAFT — for JC/reviewer assessment, not yet approved
> **Prerequisite:** Phase 1 ✅, Phase 2 D1-D3 ✅, Phase 3 D1-D3 ✅ (merged at `0133a0a4b`)
> **Parent strategy:** `.plans/ua-incorporation-strategy.md` §Phase 4
> **Review doc:** `understand-anything-to-flywheel-review.md`
> **This document does NOT authorize any work.** It is a reviewer/risk assessment to be reviewed by JC before Phase 4 bead drafting.

---

## 1. Scope Overview

Phase 4 adds seven targeted enhancements to the existing scan/fingerprint/graph pipeline. All remain **JIT/explicit-invocation only**, **stdlib-only** (no new runtime deps, no tree-sitter/WASM), **no dashboard/UI**, **no SQLite store**, **no auto-injection**. These enhancements consume and augment Phase 1–3 outputs without modifying their contracts.

### Origin

The seven deliverables come from the UA review's "patterns worth extracting" (§4) plus risk-mitigation signals identified in the existing scan pipeline. UA patterns are adapted to Flywheel's Python-native, JIT-loaded constraint.

### Beads Summary

| Bead | Description | Dependencies | Risk | Tier |
|---|---|---|---|---|
| D1 | Import classification (local / stdlib / third-party) | Phase 2 `extract_imports.py` | Low | T1 |
| D2 | Entrypoint detection | Phase 1 `language_registry.py` | Low | T1 |
| D3 | Orphan warning triage | Phase 3 `assemble_graph.py` orphan counting | Low | T1 |
| D4 | Architectural hub ranking | D1 (classified imports) | Moderate | T2 |
| D5 | Delta reporting | Phase 3 fingerprints + D1 classification | Moderate | T2 |
| D6 | Scan-to-report artifact | D1–D5 outputs | High | T3 |
| D7 | Non-LLM semantic extraction | Phase 1 `scan_project.py` | Moderate | T2 |

---

## 2. Bead Definitions

### D1: Import Classification (T1)

**What:** Extend `extract_imports.py` to classify each imported module as `stdlib`, `third-party`, or `local` (project-internal). Output adds a `classification` field to each import map entry.

**How:**
- stdlib: maintain a curated set of stdlib module names per language (Python `sys.modules` subset, plus known stdlib list; JS `fs`, `path`, `crypto`, etc.)
- local: module path matches a path within the scanned project tree
- third-party: anything else (not stdlib, not found locally)

**Scope boundary:** Data-only addition to existing output schema. No new script files. Classification tables are static data in `language_registry.py` or a new `stdlib_modules.py` data file.

**Context cost:** Zero — classification is embedded in import map JSON.

**Regression risk:** LOW — classification field is additive; consumers that ignore extra fields continue to work. Existing import extraction logic is untouched.

---

### D2: Entrypoint Detection (T1)

**What:** Detect file entrypoints (files containing `if __name__ == "__main__"`, `main()` calls exposed via setup.py/pyproject.toml `[project.scripts]`, index.js default exports, etc.) and flag them in the scan output.

**How:** Regex patterns applied during `scan_project.py` file walk. Results added as `entrypoints: [...]` in per-file JSON records.

**Scope boundary:** Detection is regex-only. No execution or import simulation. No LLM summarization of entrypoint purpose.

**Context cost:** Zero — embedded in scan output JSON.

**Regression risk:** LOW — additive field. If regex misses an entrypoint, it's a false negative (safe). False positives are informational only.

---

### D3: Orphan Warning Triage (T1)

**What:** Enhance `assemble_graph.py` orphan detection to classify orphans by severity:
- **INFO:** Config files, docs, test fixtures that are structurally isolated by design.
- **WARNING:** Source files with no imports and nothing imports them — possible dead code.
- **CRITICAL:** Files that *should* be connected (e.g., referenced in manifest but absent from graph).

**How:** Extend the existing `summary.orphan_nodes` count into a structured list with severity classification. Uses language/category context from scan output.

**Scope boundary:** Classification rules are deterministic and configurable. No LLM involved.

**Context cost:** Zero — embedded in graph assembly JSON output.

**Regression risk:** LOW — extends an existing summary field. Backward-compatible if consumers treat `orphan_nodes` as a scalar count (JSON schema should allow both scalar and structured for transition).

---

### D4: Architectural Hub Ranking (T2)

**What:** Rank files by architectural centrality — compute a simple hub score (in-degree + out-degree, or weighted variant) based on classified imports. Output a ranked list of "hub files" (files that import many others and are imported by many).

**How:** Post-process the assembled graph. Uses D1 classifications to weight edges (local imports count more than third-party for hub ranking). Top N files flagged in a `hubs` section of the report.

**Scope boundary:** Scoring is a simple deterministic formula. No LLM ranking. No visualization. Top N is configurable (default: 10). Score computation is standalone — can be run without the full graph.

**Context cost:** Zero — output is a ranked list in JSON.

**Regression risk:** MODERATE — adds a new output section. Dependent on D1 classification accuracy (incorrect classification skews hub scores). Must have its own unit tests with known-good graph inputs.

---

### D5: Delta Reporting (T2)

**What:** Compare two scan/fingerprint snapshots (e.g., "before" and "after" a git commit or branch) and produce a structured diff report: files added/removed, structural changes, import changes, hub score shifts, new orphans.

**How:** New script `scripts/code-scan/delta_report.py` that takes two fingerprint JSON files (or scan JSON files), diffs them, and outputs a delta report. Uses Phase 3 fingerprint `change_level` classification for file-level delta, plus D1 classification for import-level delta.

**Scope boundary:** Reads fingerprint files that already exist on disk. Does not write to them. Does not trigger scans. Pure comparison logic.

**Context cost:** Zero — output is a structured delta report in JSON.

**Regression risk:** MODERATE — new module, but read-only. Must handle edge cases: missing fingerprint files, schema version changes, project root mismatches.

---

### D6: Scan-to-Report Artifact (T3)

**What:** A single structured report that aggregates scan output, import classifications, hub rankings, orphan warnings, and delta insights into a unified artifact. This is the "one document" that gives the agent comprehensive project understanding without loading the full graph.

**How:** New script `scripts/code-scan/report.py` that reads scan JSON, graph JSON, fingerprints, and (optionally) delta reports, then produces a consolidated JSON report. The report is structured with sections: project overview, file inventory, import summary, hub files, orphan warnings, and (if available) delta insights.

**Scope boundary:** Aggregation only. No analysis logic. All data comes from existing scripts. Report size cap: 500 KB (enforced by summarization if inputs are larger).

**Context cost:** The report itself is designed to be ~5–20 KB — small enough to be JIT-loaded into agent context as a project overview, but only when the user explicitly requests it.

**Regression risk:** HIGH — this is the integration point for everything. If report generation is slow or produces bloated output, it undermines the entire value proposition. Must have strict size caps and performance guards.

---

### D7: Non-LLM Semantic Extraction (T2)

**What:** Extract non-import semantic signals from source files: docstrings, type annotations, exported symbols, decorator usage, and simple dependency patterns (e.g., "this class inherits from X", "this function is decorated with @auth_required").

**How:** Regex-based extraction during scan, parallel to existing fingerprint extraction. Results stored in a new `semantic_signals` field in per-file records. No LLM summarization.

**Scope boundary:** Regex-only extraction patterns per language. No type inference, no AST traversal, no type-checker integration.

**Context cost:** Zero — embedded in per-file scan records.

**Regression risk:** MODERATE — new extraction logic adds CPU time to file walk. Must be bounded per-file (e.g., max 50 signals per file) and have a timeout guard.

---

## 3. Sequencing Plan

### Critical Path

```
Phase 1-3 (complete)
    │
    ├── T1: D1 (import classification)  ─┐
    ├── T1: D2 (entrypoint detection)    │
    ├── T1: D3 (orphan triage)           │ (can execute in parallel)
    └────────────────────────────────────┘
                    │
    ├── T2: D4 (hub ranking)  ← requires D1 classified imports
    ├── T2: D5 (delta report) ← requires Phase 3 fingerprints + D1
    └── T2: D7 (semantic extraction)    (independent of D1–D4)
                    │
    └── T3: D6 (scan-to-report) ← requires T1 + T2 outputs
```

### Sequencing Pitfalls

1. **Parallel T1 execution risk:** D1, D2, D3 modify different existing scripts (`extract_imports.py`, `scan_project.py`, `assemble_graph.py`). If executed in parallel, merge conflicts are likely in shared utility functions or test fixtures. **Mitigation:** Serialize T1 execution in order D1 → D3 → D2 (D1 has the broadest impact, do it first and let D2/D3 adapt).

2. **D4 depends on D1 accuracy:** Hub ranking quality is directly proportional to import classification accuracy. If D1's stdlib list is incomplete or project root matching is buggy, hub scores will be wrong. **Mitigation:** D1 must pass a known-good classification test against all three test-bed repos before D4 starts. D4 should accept a "classification confidence" metric and degrade gracefully on low-confidence inputs.

3. **D6 integration bottleneck:** D6 aggregates outputs from all upstream beads. If any upstream bead changes its output schema, D6 breaks. **Mitigation:** Define D6's input schema contract in its bead before T2 starts. T2 beads must produce output matching that contract. Add a schema validation step in D6 that fails fast on mismatched inputs.

4. **D5 fingerprint dependency drift:** Phase 3 fingerprints are in `.hermes/code-state/fingerprints.json`. D5 needs to compare snapshots, but fingerprint files only capture the current state. **Mitigation:** D5 should support reading from two sources: (a) the current fingerprint file, and (b) a user-supplied "before" snapshot (copied by the agent before making changes). D5 does not need to maintain snapshot history itself.

5. **D7 per-CPU cost:** Semantic extraction adds regex processing per-file. On repos >1,000 files, this could add seconds to scan time. **Mitigation:** D7 must have a per-file timeout (e.g., 100ms max per file) and a total scan budget (e.g., semantic extraction cannot add >20% to base scan time). If budget exceeded, D7 degrades silently (emits partial results with a flag).

---

## 4. Scope Creep Risks

| Creep Vector | How It Manifests | Prevention |
|---|---|---|
| "Just add LLM summaries" | The line between deterministic extraction and LLM summarization is tempting once the infrastructure exists. | Every bead explicitly forbids LLM calls. SKILL.md updates for Phase 4 must restate this. |
| "Tree-sitter would be more accurate" | D1/D2/D7 regex limitations will reveal edge cases. | If tree-sitter is desired, it requires a separate dependency approval outside Phase 4 scope. |
| "The report should also include..." | D6 aggregation naturally invites additional sections (e.g., recent git log, TODO comments). | D6 output schema is fixed at bead definition time. New sections require a new bead or Phase 5. |
| "Auto-detect on every scan" | Once the pipeline produces useful reports, there will be pressure to auto-generate them. | JIT-only constraint remains. D6 only produces a report when the skill explicitly invokes it. |
| "Store reports for later" | Persistent report storage could create bloat. | Reports are transient. No SQLite. If agent needs to reference a prior report, it re-runs the scan. |
| "CLI command for delta" | Exposing delta reporting as `hermes delta` or similar. | No new CLI commands. Delta report is produced by running the script via skill invocation. |

---

## 5. Validation Strategy

### Per-Bead Verification

| Bead | Unit Tests | Integration Tests | Performance Guard |
|---|---|---|---|
| D1 | Classification accuracy on known imports (stdlib, third-party, local) against all 3 test-bed repos | Existing Phase 2 E2E tests still pass with classification field added | No measurable regression on extract_imports.py runtime |
| D2 | Regex precision/recall on known entrypoint patterns (10+ patterns per supported language) | Scan output includes `entrypoints` field for repos with known entrypoints | Regex timeout per-file: 50ms |
| D3 | Orphan classification on synthetic graphs with known orphan scenarios | Graph assembly output backward-compatible (scalar + structured `orphan_nodes`) | No measurable regression on assemble_graph.py runtime |
| D4 | Hub ranking accuracy on synthetic graphs with known centrality | Hub scores align with manual inspection on test-bed repos | Scoring O(n log n) where n = node count; < 1s on hermes-agent graph |
| D5 | Delta report accuracy on before/after snapshots with known changes | Round-trip: delete file → run delta → verify file listed as removed; restore → verify change reverted | Comparison of two fingerprints.json: < 2s for hermes-agent |
| D6 | Report completeness (all sections present), size cap (≤500 KB) | Report from all 3 test-bed repos validates against output schema | Report generation: < 5s on hermes-agent repo |
| D7 | Semantic extraction precision on known source files with docstrings, decorators, type annotations | Extracted signals match manual inspection on 20+ sample files across 3 languages | Per-file timeout: 100ms; total budget: +20% of base scan time |

### Regression Suite

All existing tests must pass after Phase 4 integration:
- Phase 1: 80 tests (`tests/code_scan/test_scan_project.py`, `test_language_registry.py`, `test_graph_schema.py`)
- Phase 2: 111 tests (including `test_extract_imports.py`)
- Phase 3: 132 tests (including `test_fingerprints.py`, `test_assemble_graph.py`)

**Total regression baseline:** 323 tests. Any failing test blocks Phase 4 progression.

### Scope Guardrail Check

After each bead, run pattern search for forbidden features:
```bash
# No tree-sitter imports
grep -r "tree_sitter\|web-tree-sitter\|wasm" scripts/code-scan/

# No new runtime dependencies
grep -r "^import\|^from" scripts/code-scan/*.py | sort -u  # compare against Phase 3 baseline

# No SQLite
grep -r "sqlite\|SQLite" scripts/code-scan/

# No dashboard references
grep -r "dashboard\|vite\|react\|frontend" scripts/code-scan/

# No auto-injection logic
grep -r "auto.*inject\|always.*on\|background.*scan" scripts/code-scan/

# No forbidden file modifications
git diff skills_sync.py test_skills_sync.py  # must return empty
```

---

## 6. Regression Risk Matrix

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| D1 classification breaks existing import map consumers | Low | High | Classification field is additive; existing fields unchanged. Full regression suite required. |
| D2 regex false positives pollute entrypoint data | Medium | Low | False positives are informational; agents should treat entrypoints as hints, not facts. |
| D3 orphan classification changes output schema | Medium | Medium | Support both scalar and structured `orphan_nodes` during transition; deprecate scalar only after downstream consumers adapt. |
| D4 hub score inaccuracies mislead agent architecture decisions | Medium | Medium | Hub scores include confidence metric; scores with <80% classification coverage are marked "low confidence." |
| D5 delta report schema incompatible with fingerprint v1 | Low | Medium | D5 explicitly checks fingerprint `schema_version` and fails with clear error on unsupported versions. |
| D6 report size exceeds budget on large repos | Medium | High | Hard cap at 500 KB with summarization fallback; fail open with truncated report + warning if cap exceeded. |
| D7 semantic extraction slows scan by >20% | Low | Medium | Per-file timeout + total budget enforcement; degrade gracefully under budget pressure. |
| Combined Phase 4 CPU overhead exceeds scan budget | Medium | Medium | Aggregate budget: Phase 4 additions cannot increase total scan time by >30% over Phase 3 baseline. |
| Merge conflicts between T1 parallel beads | Medium | Low | Serialize T1 execution as D1 → D3 → D2. |

---

## 7. Approval Gates

### Gate 0: Phase 4 Scope Approval (BEFORE any bead work)

**Trigger:** JC reviews this draft document and approves/rejects Phase 4 scope.

**Requirements:**
- JC confirms Phase 4 scope aligns with project priorities
- JC confirms T2/T3 bead risk levels are acceptable
- JC confirms performance budgets are enforced
- JC confirms forbidden constraints remain in force

**Outcome:** Approved scope → proceed to Gate 1. Rejected scope → revise and re-submit.

---

### Gate 1: T1 Completion Review (D1 + D2 + D3)

**Trigger:** All T1 beads complete, pass unit tests, pass regression suite.

**Requirements:**
- ✅ D1 unit tests pass; classification accuracy ≥95% on test-bed repos
- ✅ D2 unit tests pass; entrypoint detection precision ≥90%
- ✅ D3 unit tests pass; orphan classification backward-compatible
- ✅ Full regression suite (323+ existing tests) passes
- ✅ Scope guardrail check passes (zero forbidden pattern matches)
- ✅ No new runtime dependencies introduced
- ✅ Forbidden files unchanged

**Signoff:** Reviewer subagent must explicitly PASS on:
1. Spec compliance (each T1 bead matches its contract)
2. Scope preservation (no features outside D1–D3 scope)
3. Regression safety (all Phase 1–3 tests pass)
4. Quality/security (no path leakage, no secret exposure in outputs)

**Outcome:** Reviewer PASSED → jc approves commit of T1 work → proceed to Gate 2. Reviewer FAILED → fix and re-verify.

---

### Gate 2: T2 Completion Review (D4 + D5 + D7)

**Trigger:** All T2 beads complete, pass unit tests, pass regression suite.

**Prerequisites:** Gate 1 PASSED and T1 changes committed.

**Requirements:**
- ✅ D4 hub scores validated; confidence metrics present
- ✅ D5 delta reports accurate on known before/after pairs
- ✅ D7 semantic extraction within performance budget
- ✅ Full regression suite passes (323 + T1 + T2 tests)
- ✅ Scope guardrail check passes
- ✅ Performance guardrails verified on all 3 test-bed repos

**Signoff:** Reviewer subagent must explicitly PASS on:
1. Spec compliance
2. Scope preservation
3. Performance budget compliance
4. Data accuracy (hub scores, delta reports, semantic signals match ground truth on test repos)

**Reviewer signoff REQUIRED for Gate 2** — T2 beads have higher integration risk and produce data used by downstream agents.

**Outcome:** Reviewer PASSED → JC approves commit → proceed to Gate 3.

---

### Gate 3: T3 Completion Review (D6)

**Trigger:** D6 complete, passes unit tests, passes regression suite.

**Prerequisites:** Gate 2 PASSED and T1+T2 changes committed.

**Requirements:**
- ✅ D6 report complete for all 3 test-bed repos
- ✅ Report size ≤500 KB for all repos
- ✅ Report schema validated
- ✅ Full regression suite passes (all tests)
- ✅ Scope guardrail check passes
- ✅ Report generation within performance budget (<5s on hermes-agent)

**Signoff:** Reviewer subagent must explicitly PASS on:
1. Spec compliance
2. Scope preservation
3. Size cap enforcement
4. Integration correctness (report reflects upstream data faithfully)

**JC approval REQUIRED for Gate 3** — D6 is the highest-risk bead and produces the artifact most likely to be loaded into agent context.

**Outcome:** Reviewer PASSED + JC approves → Phase 4 complete. Commit and merge per JC instructions.

---

## 8. Tiering Rationale

### Why T1 (D1, D2, D3)

These are **foundational, low-risk, additive** enhancements:
- Each modifies a single existing script without changing its output contract
- All are pure data additions (new fields in existing JSON structures)
- All are deterministic, no LLM involvement
- All can be rolled back by simply removing the new fields
- No cross-bead dependencies within T1

**Reviewer signoff:** Standard. Required at Gate 1 completion, but T1 beads themselves do not require individual signoff before execution.

### Why T2 (D4, D5, D7)

These introduce **analytic logic** that produces derived data:
- D4 computes hub scores from classified imports (depends on D1 accuracy)
- D5 performs cross-snapshot comparisons (depends on fingerprint schema stability)
- D7 adds new extraction patterns (CPU budget impact)
- Incorrect outputs could mislead agent decisions

**Reviewer signoff:** Required individually before commit. Each T2 bead must be reviewed after unit test pass, before integration into the branch.

### Why T3 (D6)

This is the **integration bead** that aggregates all upstream outputs:
- Highest complexity (consumes outputs from D1–D5)
- Highest context impact (report may be loaded into agent context)
- Highest risk of size/performance overruns
- The most valuable — but also the most dangerous if flawed

**Reviewer signoff:** Required. JC approval required. D6 cannot proceed to commit until Gate 3 signoff from both reviewer and JC.

---

## 9. Recommended Execution Order

```
Week 1:
  └── D1 (import classification) ──→ Gate 1 checkpoint 1/3
  └── D3 (orphan triage)           ──→ Gate 1 checkpoint 2/3
  └── D2 (entrypoint detection)    ──→ Gate 1 checkpoint 3/3 → REVIEWER SIGNOFF → Gate 1 PASS

Week 2:
  └── D4 (hub ranking)             ──→ REVIEWER SIGNOFF
  └── D7 (semantic extraction)     ──→ REVIEWER SIGNOFF
  └── D5 (delta reporting)         ──→ REVIEWER SIGNOFF → Gate 2 PASS

Week 3:
  └── D6 (scan-to-report)          ──→ REVIEWER SIGNOFF → Gate 3 PASS → JC MERGE APPROVAL
```

Total estimated: 3 weeks for beads, 1 week for reviews/gates = 4 weeks.

---

## 10. Open Questions for JC

1. **Performance budget priority:** Should scan time budget be preserved at Phase 3 levels (no increase), or is a 20–30% increase acceptable given the added signal value?

2. **D7 scope:** The original UA review suggested tree-sitter for semantic extraction. This draft specifies regex-only. If regex proves insufficient, should tree-sitter integration be a separate Phase 5 rather than a Phase 4 exception?

3. **D6 context loading:** The scan-to-report artifact could be loaded into agent context JIT. Should there be a token budget cap (e.g., 3,000 tokens max) on report loading, in addition to the file size cap?

4. **D4 deferred option:** Hub ranking may produce noisy results early on. Should D4 be drafted as deferred-by-default (like D4 in Phases 2 and 3) and executed only after D1–D3 have proven valuable in production use?

5. **Test-bed expansion:** Should a fourth test-bed repo (non-Python, non-TypeScript — e.g., Go or Rust) be added to validate import classification and semantic extraction across more languages?
