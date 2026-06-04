---
name: code-scan
hermes.tags: [on-demand, code-analysis, project-mapping]
---

# Code Scan — JIT Orchestration Skill

## Purpose
Load on-demand to run the code-scan pipeline against any target directory.
Never auto-injected; triggered explicitly by the user.

## Mode Selection

Use the **UA-005 mode router** (`scripts/code-scan/run_ua.py`) for all new code-scan
runs.  It replaces the manual orchestration steps below with explicit mode selection.

```bash
python scripts/code-scan/run_ua.py --target <dir> --out <bundle_dir> [--mode <mode>]
```

### Available Modes

| Mode | Pipeline Stages | Use When… |
|---|---|---|
| **inventory** (default: no) | scan + imports only | Quick inventory of files and dependencies. |
| **structure** (default) | scan + imports + graph + validation | Default — balanced structural analysis with graph validation. |
| **review** | structure + analytics + context envelope + report | Deep review with analytics, subagent context, and markdown report. |
| **delta** | incremental scan + delta summary against prior manifest | Detect changes since a prior run (use `--prior-manifest`). |
| **preflight** | structure + entrypoints/hubs + subagent context | Subagent handoff preparation with structure and context. |
| **full** | all available deterministic enrichers | Everything available — analytics, context, report. |
| **security-review** | review-mode artifacts + security evidence-gap report data | Planning/preflight mode for static security-assurance follow-up gaps; never a certification. |

### Mode Routing Guarantees

- **Validation failures are never hidden.** Any mode that runs graph assembly
  also runs the validation gate and writes `validation.json`.
- **Quick modes avoid unnecessary graph analytics.** `inventory` and `structure`
  do not produce `analytics.json` or `subagent-context.json`.
- **Missing optional enrichers are recorded as skipped.** When an optional deterministic
  enricher is unavailable in the current environment, its artifact is omitted and noted
  in `artifacts_missing` metadata — nothing is fabricated.
- **Default mode is `structure`**, preserving backward compatibility with the
  existing pipeline behavior (scan → imports → graph → validation).

## Orchestration Steps (Legacy Manual Path)

> **Prefer** `run_ua.py` mode selection above.  The manual steps below remain
> documented for compatibility but are superseded by the mode router.

1. **Confirm target:** Ask user for target directory or use working directory.
2. **Check fingerprints & run scan:**
   - If `.hermes/code-state/fingerprints.json` exists → use `--incremental` for a fast diff scan.
   - If fingerprints are absent → `--incremental` is safe and behaves like a full scan.
   - If user requests a forced full rescan → use `--full` instead.
   - Command: `python scripts/code-scan/scan_project.py <target_dir> --incremental --output <temp_scan.json>`
3. **Run import extraction:** `python scripts/code-scan/extract_imports.py <temp_scan.json> > <temp_imports.json>`
4. **Read artifacts:** Read both JSON outputs.
5. **Synthesize (LLM-only):** From scan data produce:
   - Project name (from directory or package.json/pyproject.toml)
   - One-line description (inferred from frameworks + files, not hallucinated)
   - Framework/stack narrative (from detected frameworks + language distribution)
6. **Render summary:** Output as structured markdown (format below).
7. **Clean up:** Temp files can be deleted; they are not tracked artifacts.

## Canonical Run Bundle Shape

> **Reference:** Full mode definitions, bundle layout, and usage details are documented in [references/ua-run-bundle-and-modes.md](references/ua-run-bundle-and-modes.md).

Every UA run bundle produced by `run_ua.py` is a directory containing:

| Artifact | Description |
|----------|-------------|
| `manifest.json` | Run metadata: `run_id`, `status` (`"complete"` or `"failed"`), `mode`, `timestamp`, `target_path`, `target_git_head`, `bundle_dir`, `command_flags`, `artifact_paths`, `script_versions`, `target_mutation_allowed`, target cleanliness fields (`target_dirty_before`, `target_dirty_after`, `target_dirty_files_before`, `target_dirty_files_after`, `target_cleanliness_status`, `unexpected_target_changes`), `artifacts_missing` (when upstream enrichers are absent), `project_state_recorded`, `ledger_path`, `project_state_append_status`, `project_state_append_error`. |
| `scan.json` | Deterministic file inventory and language distribution. |
| `imports.json` | Deterministic import map with schema version and totals. |
| `graph.json` | Dependency graph (`nodes` + `edges`) — present in all modes except `inventory` and `delta`. |
| `validation.json` | Deterministic validation results (`issues`, `warnings`, `severity_summary`, `severity_classified_warnings`) — present whenever graph is built. |
| `runtime-readiness.json` | Runtime/toolchain readiness detection (`detected_stacks`, `required_commands`, `suggested_verification`, `verification_status`, `blockers`). Only checks tool availability — never runs build or test commands. Present in `structure`, `review`, `preflight`, and `full` modes. |
| `analytics.json` | Graph analytics (centrality, hubs, etc.) — present only when `analyze_graph` is available and mode produces analytics (`review`, `full`). Recorded in `artifacts_missing` otherwise. |
| `subagent-context.json` | Subagent context envelope — present only when `build_context_bundle` is available and mode produces context (`review`, `preflight`, `full`). Recorded in `artifacts_missing` otherwise. |
| `summary.json` | Compact aggregate summary of all produced stages. |
| `REPORT.md` | Human-readable markdown report — present only in `review` and `full` modes. |
| `runtime-readiness.md` | Human-readable runtime readiness report (counterpart to runtime-readiness.json). Same availability as runtime-readiness.json. |
| `cache/` (optional) | External fingerprint cache directory when `--external-cache-dir` is used. |

**Read-only target guarantee:** The target directory is **never mutated** by the UA pipeline. All artifacts are written exclusively to the `--out` bundle directory. The only exception is when `--in-repo-cache` is explicitly passed, which writes fingerprints inside the target repo's `.hermes/code-state/`. By default, fingerprints use an external cache under `<bundle>/cache/`, leaving the target fully read-only.

**Deterministic boundary:** Deterministic scripts (`scan_project.py`, `extract_imports.py`, `assemble_graph.py`, `graph_schema.py`, and optional enrichers) produce **facts** — JSON artifacts with stable schemas. The LLM's role is to **interpret** those facts (e.g., synthesizing project name, description, and framework narrative from scan data). Never allow LLM intuition to override deterministic validation results, graph structure, or severity classifications.

## Phase 5 Trust Boundaries and Closeout Checklist

UA validation means the graph/artifact is structurally usable; it does not prove security, deployment readiness, RLS correctness, or runtime correctness.

Runtime readiness lists tool availability and suggested/external gate status; UA does not execute project gates unless a separate user-approved runner exists. Suggested commands in `runtime-readiness.json`, `runtime-readiness.md`, or `REPORT.md` must be treated as `suggested_not_run` unless a separate external gate explicitly records an executed result.

Use reviewer/researcher as targeted critics; Hermes owns final assessment. `subagent-context.json` critic packs are bounded handoff aids, not authority transfer, and deterministic UA facts must remain separate from Hermes interpretation.

Canonical Phase 5 artifact checks for `review`/`full` bundles:

- Verify `manifest.json` before trusting a bundle: `status`, `target_mutation_allowed`, target cleanliness fields, `unexpected_target_changes`, `provenance`, and `artifact_integrity`.
- Read `validation.json` for deterministic issues/warnings and severity-classified orphan findings.
- Read `runtime-readiness.json`/`.md` for detected stacks, blockers, and verification gate statuses without assuming commands ran.
- Read `domain-surfaces.json` as deterministic inventory only; it is not semantic, security, RLS, runtime, or deployment validation.
- Read `REPORT.md` for confidence labels and the explicit “What UA proves / What UA does not prove” boundary section.
- Read `subagent-context.json` for scoped `reviewer_critic`, `researcher_scout`, and `coder_preflight` packs when using targeted critics.

## Constraints
- Never hallucinate file structures — only report what scan scripts return.
- If scan fails, report the error; do not guess.
- .hermesignore rules are already enforced by scan_project.py.
- Only synthesize name, description, and framework fields. Everything else is deterministic.
- `--incremental` relies on `.hermes/code-state/fingerprints.json`; omit or use `--full` to force a complete rescan.

## Opt-In Project-State Integration (UA-006)

After a UA bundle run, you may optionally record a compact summary in the
project's `.hermes/PROJECT_STATE.md` ledger **if it already exists**:

**CLI (automated):**

```bash
python scripts/code-scan/run_ua.py --target <project> --out <bundle> --record-project-state [--project-root <path>]
```

- `--record-project-state` enables the hook (disabled by default).
- `--project-root` explicitly sets the project root; if omitted, `--target` is used.

**CLI (manual, standalone):**

```bash
python scripts/code-scan/project_state_append.py <project_root> --manifest <bundle>/manifest.json
```

- The ledger is **appended only** — existing content is never overwritten.
- When the ledger is absent, no state is written and the manifest reports
  `project_state_recorded: false` with `ledger_path: null`.  UA runs are not affected by this.
- Only compact deterministic facts are recorded: run ID, mode, target path,
  artifact bundle path (linked, not embedded), validation verdict, issue/warning
  counts, file count, top-5 languages, graph node/edge count, next recommended
  action.
- No LLM/ML judgement or large JSON blobs are written to the ledger.
- Programmatic usage: `from project_state_append import append_project_state`

## Output Format

## Project: <name>
- **Description:** <one-line>
- **Languages:** <detected language distribution>
- **Frameworks:** <detected frameworks array>
- **Structure:** <top-level dirs + key files>
- **Import map:** <top 5 most-imported modules>
- **Files:** <total_files> total, <files_with_imports> with imports

## Phase 6 Trustworthy Handoff Additions

- `subagent-context.json` includes relevance-ranked recommended files and a bounded `must_read_map` for Hermes/subagent attention routing. These are navigation aids, not authority transfer or security proof.
- `runtime-readiness.json` includes a runtime gate inventory with statuses such as `suggested_not_run`; UA lists tool availability and candidate commands without executing target build/test/lint/audit gates.
- `security-review` mode emits deterministic security evidence gaps from static UA facts, domain surfaces, must-read map, orphan summaries, runtime gate inventory, and critic packs. Every gap must be labelled as `outside_ua_scope`, `static_analysis_finding`, `suggested_verification_not_run`, or `executed_external_gate` when externally supplied evidence exists.
- UA validation does not prove security, deployment readiness, RLS correctness, runtime correctness, CI success, or production behavior. Reviewer/researcher outputs are targeted critic evidence; Hermes owns final synthesis and final assessment.
