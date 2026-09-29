---
name: validation-gate
hermes.tags: [on-demand, code-analysis, quality-gate]
---

# Validation Gate — Two-Phase Reviewer Skill

**JIT/on-demand only.** Never auto-injected. Load when a bead stage requires verification of graph-like or analysis artifacts.

## Phase 1 — Deterministic Validation

1. Accept a target: path to graph JSON, scan output, or analysis result (must have `{"nodes": ..., "edges": ...}` shape).
2. Run `scripts/code-scan/graph_schema.py` by importing `validate_graph()`:
   ```python
   import importlib.util
   spec = importlib.util.spec_from_file_location('graph_schema', 'scripts/code-scan/graph_schema.py')
   mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
   result = mod.validate_graph(graph)  # → {"issues": [...], "warnings": [...], "severity_summary": {...}, "severity_classified_warnings": [...]}
   ```
3. Parse results: `issues` list, `warnings` list, `severity_summary` dict, and `severity_classified_warnings` list.

## Phase 2 — LLM Rendering

Interpret results and render a verdict:

- **APPROVED** — `issues` is empty and `warnings` is empty. No action needed.
- **WARNING** — `issues` is empty but `warnings` is non-empty. Render warnings as structured notes with severity breakdown from `severity_summary`. Proceed with caution.
- **REJECTED** — `issues` is non-empty. Render issues as structured blockers. Trigger revision gate (request changes before proceeding).

**Constraint:** Do NOT validate using LLM intuition — only report what the validation script returns. Do NOT assign severity to warnings using LLM intuition — severity is determined deterministically by `graph_schema.py` heuristics.

## Warning Severity Taxonomy (UA-002)

Warnings are classified into four deterministic severity levels:

| Severity | Meaning | Typical Examples |
|----------|---------|------------------|
| **INFO** | Non-blocking informational notices | Orphan documentation files (`docs/`, `README.md`, `CHANGELOG.md`, `.txt`, `.rst`) |
| **MINOR** | Low-priority orphan assets | Orphan fixtures, test data, images, config files (`tests/fixtures/`, `assets/`, `.json`, `.yaml`) |
| **MODERATE** | Potentially disconnected code worth reviewing | Orphan source files not matched by INFO/MINOR heuristics |
| **MAJOR** | Requires attention if triggered | Reserved for future suspicious-pattern heuristics; never assigned by LLM intuition |

The `severity_summary` field in the validation result provides counts per severity:
```json
{
  "severity_summary": {"info": 2, "minor": 1, "moderate": 0, "major": 0}
}
```

The `severity_classified_warnings` field provides individual entries:
```json
{
  "severity_classified_warnings": [
    {"severity": "info", "message": "Orphan node: 'docs/README.md' is not referenced by any edge"},
    {"severity": "moderate", "message": "Orphan node: 'src/legacy.py' is not referenced by any edge"}
  ]
}
```

## Validation Checks

The skill delegates to `graph_schema.py` which enforces: valid `NodeType`/`EdgeType` enums (or aliases), `node_id` present + string, `filePath` present, `edge_type` resolves, `source`/`target` reference known node IDs, no self-referencing edges, orphan nodes flagged as severity-classified warnings.

## Integration with the UA Run Bundle

The validation gate is normally invoked as part of a **UA run bundle** (via `run_ua.py` mode router, see `skills/code-analysis/code-scan/SKILL.md`). In any mode that assembles the dependency graph (`structure`, `review`, `preflight`, `full`), `validation.json` is written alongside `graph.json` in the bundle directory. The validation gate skill can also be loaded standalone to inspect an existing bundle's validation results.

When the UA run includes **subagent context** generation (`review`, `preflight`, `full` modes), the validation verdict is recorded in the `PROJECT_STATE.md` ledger if `--record-project-state` was set (UA-006). See the **Opt-In Project-State Integration** section in the code-scan skill for details.

## Output Format

```
## Validation Gate: [APPROVED | WARNING | REJECTED]
- **Issues:** <count> critical
- **Warnings:** <count> non-blocking
  - INFO: <count>
  - MINOR: <count>
  - MODERATE: <count>
  - MAJOR: <count>
### Issues (if any)
- <issue 1>
### Warnings (if any)
- <warning 1> [severity]
### Recommendation
<Proceed / Fix critical issues / Request changes>
```
