# UA Run Bundle and Modes Reference

> This document describes the implemented UA-001 through UA-006 pipeline behaviors.
> All descriptions below reflect currently-shipped code — no deferred or speculative features.

## Canonical Run Bundle Shape (UA-001)

A UA run bundle is a directory created by `run_ua.py` (or the lower-level `run_bundle.py`)
containing all artifacts from a single pipeline execution.

### Required Artifacts (Always Present)

| File | Source | Description |
|------|--------|-------------|
| `manifest.json` | `run_ua.py` | Run metadata: `run_id`, `mode`, `timestamp`, `target_path`, `target_git_head`, `bundle_dir`, `command_flags`, `artifact_paths`, `script_versions`, `target_mutation_allowed`, `artifacts_missing`, `project_state_recorded`, `ledger_path` |
| `scan.json` | `scan_project.py` | File inventory: `total_files`, `total_lines`, `languages`, `frameworks`, per-file metadata |
| `imports.json` | `extract_imports.py` | Import map: `schema_version`, `files_with_imports`, `unique_modules`, per-file import lists |
| `summary.json` | `run_ua.py` | Compact aggregate of all stages |

### Conditional Artifacts (Mode-Dependent)

| File | Modes | Source | Description |
|------|-------|--------|-------------|
| `graph.json` | `structure`, `review`, `preflight`, `full` | `assemble_graph.py` | Dependency graph with `nodes` and `edges` arrays, `summary` object |
| `validation.json` | `structure`, `review`, `preflight`, `full` | `graph_schema.py` | Validation results: `issues`, `warnings`, `severity_summary`, `severity_classified_warnings` |
| `analytics.json` | `review`, `full` | `analyze_graph.py` (when available) | Graph analytics: centrality, hubs, cluster data |
| `subagent-context.json` | `review`, `preflight`, `full` | `build_context_bundle.py` (when available) | Subagent context envelope for handoff |
| `REPORT.md` | `review`, `full` | `run_ua.py` | Human-readable markdown report |

### Missing-Enricher Tracking

When upstream beads are not present (e.g., `analyze_graph.py` or `build_context_bundle.py`
do not exist on disk), requested artifacts are **omitted** and their names are recorded
in `manifest.json` → `artifacts_missing`. Nothing is fabricated.

## Read-Only Target Mode (UA-001)

The **target directory is never mutated** by the UA pipeline. All artifacts are written
exclusively to the `--out` bundle directory.

The only exception is when `--in-repo-cache` is explicitly passed, which writes
fingerprints inside the target repo's `.hermes/code-state/`. By default, the
pipeline uses an external cache under `<bundle>/cache/`, leaving the target fully
read-only.

## Explicit Modes (UA-005)

Modes are enforced by the `RunUA` class in `scripts/code-scan/run_ua.py`. Invalid or
unspecified mode defaults to `structure` for backward compatibility.

| Mode | Pipeline Stages | Artifacts Produced |
|------|----------------|--------------------|
| `inventory` | scan → imports | `scan.json`, `imports.json`, `summary.json`, `manifest.json` |
| `structure` (default) | scan → imports → graph → validation | `scan.json`, `imports.json`, `graph.json`, `validation.json`, `summary.json`, `manifest.json` |
| `review` | structure → analytics → context → report | All structure artifacts + `analytics.json`, `subagent-context.json`, `REPORT.md` |
| `delta` | incremental scan → delta vs prior manifest | `scan.json`, `imports.json`, `manifest.json` (with `delta_summary`) |
| `preflight` | structure → context | Structure artifacts + `subagent-context.json` (no analytics, no report) |
| `full` | all available deterministic enrichers | All structure artifacts + analytics + context + report (when available) |

### Mode Routing Guarantees

- **Validation failures are never hidden.** Any mode that runs graph assembly also
  runs the validation gate and writes `validation.json`.
- **Quick modes avoid unnecessary graph analytics.** `inventory` and `structure`
  do not produce `analytics.json` or `subagent-context.json`.
- **Default mode is `structure`**, preserving backward compatibility with the
  existing pipeline (scan → imports → graph → validation).

## Validation Severity Taxonomy (UA-002)

Deterministic severity classification in `graph_schema.py` — never LLM-assigned:

| Severity | Meaning | Action |
|----------|---------|--------|
| `INFO` | Non-blocking informational notices | No action required |
| `MINOR` | Low-priority orphan assets | Awareness only |
| `MODERATE` | Potentially disconnected code worth reviewing | Worth investigating |
| `MAJOR` | Reserved for future suspicious-pattern heuristics | Never assigned by LLM intuition |

The validation gate skill produces three verdicts:
- **APPROVED** — no issues, no warnings
- **WARNING** — no issues, but warnings present (render with severity breakdown)
- **REJECTED** — issues present (blockers; request changes)

## Graph Analytics Interpretation Boundaries (UA-003)

Deterministic scripts produce **facts**; LLMs **interpret** facts.

- `analyze_graph.py` computes centrality scores, hub detection, cluster analysis.
  These are numerical facts written to `analytics.json`.
- The LLM's job is to describe what centrality or hub patterns mean for the
  project's architecture — never to recalculate or override the numbers.
- `artifacts_missing` in the manifest records when analytics are unavailable.
  The LLM should not fabricate analytics data.

## Subagent Context Envelope (UA-004)

- `build_context_bundle.py` produces `subagent-context.json` — a structured envelope
  designed for reliable subagent handoff.
- Context includes: project topography, key files, dependency hotspots, entry points,
  and recommended focus areas.
- Generated in `review`, `preflight`, and `full` modes when the enricher is available.
- The LLM uses this envelope to brief downstream subagents; it does not regenerate
  or modify context data.

## Project-State Recording (UA-006)

- Opt-in via `--record-project-state` flag (disabled by default).
- Appends a compact UA section to `.hermes/PROJECT_STATE.md` **only if the ledger
  already exists** — no file is created if absent.
- Appended data is append-only: existing ledger content is never overwritten.
- Recorded facts: run ID, mode, target path, artifact bundle path (linked, not
  embedded), validation verdict, issue/warning counts, file count, top-5 languages,
  graph node/edge count, next recommended action.
- **No LLM/ML judgement or large JSON blobs** are written to the ledger.
- Manifest always includes `project_state_recorded` (boolean) and `ledger_path`
  (string or null) regardless of opt-in status.

## Deterministic / LLM Boundary

```
┌─────────────────────┐     ┌─────────────────────┐
│  Deterministic      │     │  LLM Interpretation │
│  Scripts (Facts)    │────▶│  (Narrative)         │
│                     │     │                      │
│  scan_project.py    │     │  Project name        │
│  extract_imports.py │     │  Description         │
│  assemble_graph.py  │     │  Framework narrative │
│  graph_schema.py    │     │  Architecture summary│
│  analyze_graph.py   │     │  Recommendations     │
│  build_context_*.py │     │                      │
│  run_ua.py          │     │                      │
└─────────────────────┘     └─────────────────────┘
```

- **Scripts produce facts.** JSON artifacts with stable schemas — file counts,
  import maps, node/edge graphs, validation results, centrality scores.
- **LLMs interpret facts.** Generate human-readable descriptions, synthesize
  project narratives from detected frameworks, recommend actions based on
  validation verdicts.
- **Scripts never fabricate.** If data is absent, it is recorded as missing,
  not guessed.
- **LLMs never override facts.** Validation verdicts, severity classifications,
  and graph structure come from deterministic scripts — LLM rendering is
  advisory only.

## Path Fallback Guidance

When a target directory is not explicitly provided, use the current working
directory as fallback. If the scan cannot resolve a target, report the error
and do not proceed — do not guess paths or hallucinate file listings.

## Testing Warning

**Do not stop the UA pipeline before graph assembly and validation when testing.**
Stopping after scan or import extraction alone will not produce `graph.json` or
`validation.json`, which are required artifacts for the validation gate and most
downstream consumers. Use `--mode structure` at minimum for meaningful test results.

## Skill Cross-References

- **code-scan skill** (`skills/code-analysis/code-scan/SKILL.md`): Full orchestration
  guide, mode selection, legacy manual path, project-state integration.
- **validation-gate skill** (`skills/code-analysis/validation-gate/SKILL.md`): Phase 1
  deterministic validation, Phase 2 LLM rendering, severity taxonomy, verdict format.
