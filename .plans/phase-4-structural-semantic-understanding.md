# Phase 4 — Structural/Semantic Understanding Layer

> **Status:** Approved plan for execution (JC 2026-06-01T03:05:36Z); D1 active; D2-D7 pending; no commit/push/merge without evidence and approval.
> **Project:** Hermes Understand-Anything / code-scan Flywheel integration
> **Prerequisites:** Phase 1 complete; Phase 2 D1-D3 complete; Phase 3 D1-D3 merged; Phase 2/3 D4 skill-integration beads remain deferred unless separately approved.
> **Source review:** Live UA review on `mcp_agent_mail` plus meta-review of current UA skills.

## Goal

Upgrade Understand-Anything from a deterministic structural scanner into a stronger deterministic codebase-understanding layer that can tell Hermes not only **what files exist and import each other**, but also:

- which files are architectural hubs;
- where execution likely starts;
- which imports are local vs standard library vs third-party;
- which orphan graph nodes are expected vs suspicious;
- which modules expose useful semantic signals such as public symbols, classes, functions, decorators, and docstrings;
- what changed between two scans; and
- what compact report should be handed to Hermes/subagents before deeper reading.

The phase remains deterministic, JIT-invoked, read-only against target repositories, and stdlib-only.

## Non-Goals / Guardrails

These are explicitly out of scope for Phase 4 unless JC separately approves them:

- No dashboard, React UI, Vite app, browser visualizer, or always-on service.
- No automatic context injection, background scan, watcher, cron, or persistent indexing.
- No SQLite/vector store/report database.
- No tree-sitter, WASM parser, LSP integration, or new runtime dependency.
- No LLM summarization inside scanner scripts.
- No edits to `tools/skills_sync.py` or `tests/tools/test_skills_sync.py`.
- No commit, push, merge, deploy, or release without explicit JC approval.

## Architecture

Phase 4 extends the existing pipeline:

```text
scan_project.py
  -> extract_imports.py
  -> assemble_graph.py
  -> validation-gate
```

with optional deterministic enrichers:

```text
scan.json + imports.json -> classify_imports.py       -> classified-imports.json
scan.json                -> detect_entrypoints.py     -> entrypoints.json
graph.json + scan.json   -> triage_orphans.py         -> orphan-triage.json
graph.json               -> hub_ranking.py            -> hubs.json
scan.json                -> semantic_extract.py       -> semantic-signals.json
old scan/fp + new scan/fp-> delta_report.py           -> delta.json
all artifacts            -> report_builder.py         -> UA_REPORT.md / JSON summary
```

Each script is standalone and can be invoked explicitly by the `code-scan` skill or a future Phase 4 skill-integration bead. The scanner core remains usable if none of the enrichers are run.

## Output Contracts

### Import Classification

`classified-imports.json` groups every raw import into:

> Compatibility note: `source_scan` / upstream provenance fields should echo the upstream artifact shape rather than replacing it with a plain string. If `extract_imports.py` emits `source_scan` as a structured object, D1 preserves that object and adds classification provenance separately.

- `local`
- `stdlib`
- `third_party`
- `relative`
- `unknown`

The output must include per-file classifications and global totals.

### Entrypoints

`entrypoints.json` lists detected startup/control-flow anchors with:

- file path;
- language;
- entrypoint type;
- signals matched;
- confidence; and
- source of evidence.

### Orphan Triage

`orphan-triage.json` groups graph orphan nodes into:

- `expected` — docs, tests, config, assets, fixtures;
- `entrypoint_candidate` — executable files likely to be standalone;
- `suspicious` — source files that are neither imported nor plausible entrypoints;
- `unknown` — insufficient metadata.

### Hub Ranking

`hubs.json` ranks important files using graph degree and project-local dependency signals. Scores are hints, not truth. The report must include confidence/coverage notes.

### Semantic Signals

`semantic-signals.json` extracts deterministic signals such as:

- Python classes/functions/docstrings/decorators via `ast` where practical;
- JS/TS exported symbols and route-ish patterns via bounded regex;
- CLI/web/data-model signal hints;
- capped per-file signal counts.

### Delta Report

`delta.json` compares two scan/fingerprint snapshots and reports:

- added/removed files;
- language/category count deltas;
- framework deltas;
- structural/cosmetic/unchanged fingerprint summaries when available;
- import/hub/orphan deltas if optional artifacts are supplied.

### Scan-to-Report Artifact

`UA_REPORT.md` is a compact, human/agent-readable summary designed for JIT handoff. It should include:

- deterministic scan facts;
- likely entrypoints;
- architectural hubs;
- import classification totals;
- orphan triage summary;
- semantic hotspots;
- delta summary when provided;
- suggested reading plan;
- validation verdict and caveats.

Target size: concise by default, with hard caps and truncation warnings for large repos.

## Performance Budget

Recommended default for approval if JC does not override it:

- Individual enrichers should each complete in under 2 seconds on hermes-agent-scale repos under normal VPS conditions.
- Combined Phase 4 overhead should stay at or below +30% relative to the Phase 3 scan/import/graph baseline.
- D5 semantic extraction must enforce per-file signal caps and degrade with warnings rather than blocking large scans.
- D7 report generation should stay under 5 seconds and enforce a 500 KB artifact cap unless JC approves a larger artifact.

## Bead Index

Execution order is intentionally serialized to avoid overlapping script/test changes and to preserve clean checkpoints.

1. `.beads/phase4-d1-import-classification.md`
2. `.beads/phase4-d2-entrypoint-detection.md`
3. `.beads/phase4-d3-orphan-triage.md`
4. `.beads/phase4-d4-hub-ranking.md`
5. `.beads/phase4-d5-semantic-extraction.md`
6. `.beads/phase4-d6-delta-reporting.md`
7. `.beads/phase4-d7-scan-report-artifact.md`

## Dependency Map

```text
D1 import classification ─┐
D2 entrypoints ───────────┼─> D7 scan-to-report artifact
D3 orphan triage ─────────┤
D4 hub ranking ───────────┤
D5 semantic extraction ───┤
D6 delta reporting ───────┘
```

D4 can run after graph assembly exists; it does not strictly require D1, but D7 benefits from both. D7 is the integration bead and must run last.

## Complexity / Routing

| Bead | Tier | Executor | Reviewer |
|---|---:|---|---|
| D1 import classification | T2 | hermes-coder | Required |
| D2 entrypoint detection | T2 | hermes-coder | Required |
| D3 orphan triage | T1/T2 | hermes-coder | Required due graph-validation implications |
| D4 hub ranking | T2 | hermes-coder | Required |
| D5 semantic extraction | T2 | hermes-coder | Required |
| D6 delta reporting | T2 | hermes-coder | Required |
| D7 scan-to-report artifact | T3 unless split | hermes-coder, checkpointed | Required + JC approval before execution |

## Test Strategy

Every implementation bead must use TDD-style evidence:

- RED focused test or explicit reason RED is not applicable;
- GREEN focused pass;
- FULL code-scan regression pass;
- scope guardrail check;
- forbidden-file diff check;
- reviewer PASS before commit approval.

Baseline regression command for most beads:

```bash
python -m pytest tests/code_scan/ -v --tb=short
```

Scope guardrail command for every bead:

```bash
git diff --name-only -- tools/skills_sync.py tests/tools/test_skills_sync.py
python - <<'PY'
from pathlib import Path
for path in Path('scripts/code-scan').glob('*.py'):
    text = path.read_text(errors='ignore')
    forbidden = ['tree_sitter', 'web-tree-sitter', 'wasm', 'sqlite3', 'chromadb', 'openai', 'anthropic', 'litellm']
    hits = [needle for needle in forbidden if needle.lower() in text.lower()]
    if hits:
        raise SystemExit(f'Forbidden dependency/scope term in {path}: {hits}')
print('SCOPE GUARD PASS')
PY
```

## Approval Gates

### Gate 0 — Plan Approval

JC approved the Phase 4 plan on 2026-06-01T03:05:36Z:

> I approve Phase 4 Understand-Anything Structural/Semantic Understanding for autonomous planning-to-execution on a new branch. Approved scope: D1-D7 as written, with D7 checkpointed if needed. Guardrails: JIT-only, no dashboard/UI, no auto-injection, no SQLite/vector store, no tree-sitter/WASM/new runtime dependencies, no LLM summaries inside scanner scripts, no edits to `tools/skills_sync.py` or `tests/tools/test_skills_sync.py`, and no commit/push/merge without evidence and my approval.

Execution branch: `feat/ua-phase4-structural-semantic`. D1 active; D2-D7 pending. No commit/push/merge without evidence and JC approval.

### Gate 1 — D1-D3 Foundation

After D1-D3: verify imports, entrypoints, and orphan triage on hermes-agent plus at least one external repo scan artifact.

### Gate 2 — D4-D6 Insight Enrichment

After D4-D6: verify hub ranking, semantic extraction, and delta report remain fast, deterministic, and bounded.

### Gate 3 — D7 Report Artifact

D7 requires reviewer PASS and JC approval before commit/push because it becomes the human/subagent-facing report contract.

## Open Decisions for JC

1. Should D7 be split into `D7a JSON report` and `D7b Markdown renderer`, or kept as one checkpointed T3 bead?
2. Should Phase 4 include only scripts/tests first, leaving `code-scan` skill integration as Phase 4 D8/deferred?
3. What performance budget should we enforce for hermes-agent-scale repos: +20%, +30%, or explicit second limits?
4. Should entrypoint and semantic extraction be Python-first in D2/D5, then JS/TS/Go/Rust later?

## Recommended Default

Approve D1-D6 as implementation scope, keep D7 as checkpointed but included, and defer `code-scan` SKILL.md wiring until D7 output stabilizes. That preserves the deterministic tool layer before changing agent-facing behavior.
