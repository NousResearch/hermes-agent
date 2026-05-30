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
   result = mod.validate_graph(graph)  # → {"issues": [...], "warnings": [...]}
   ```
3. Parse results: `issues` list and `warnings` list.

## Phase 2 — LLM Rendering

Interpret results and render a verdict:

- **APPROVED** — `issues` is empty. Warnings may exist but are presented as non-blocking notes.
- **WARNING** — `issues` is empty but `warnings` is non-empty. Render warnings as structured notes. Proceed with caution.
- **REJECTED** — `issues` is non-empty. Render issues as structured blockers. Trigger revision gate (request changes before proceeding).

**Constraint:** Do NOT validate using LLM intuition — only report what the validation script returns.

## Validation Checks

The skill delegates to `graph_schema.py` which enforces: valid `NodeType`/`EdgeType` enums (or aliases), `node_id` present + string, `filePath` present, `edge_type` resolves, `source`/`target` reference known node IDs, no self-referencing edges, orphan nodes flagged as warnings.

## Output Format

```
## Validation Gate: [APPROVED | WARNING | REJECTED]
- **Issues:** <count> critical
- **Warnings:** <count> non-blocking
### Issues (if any)
- <issue 1>
### Warnings (if any)
- <warning 1>
### Recommendation
<Proceed / Fix critical issues / Request changes>
```
