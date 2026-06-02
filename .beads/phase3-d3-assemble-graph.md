---
id: phase3-d3-assemble-graph
title: Phase 3 D3 — assemble_graph.py: merge batch outputs into unified dependency graph
status: completed
executor: delegate-coder
parallel_safe: false
base_branch: docs/ua-flywheel-phase3-plan
allowed_files:
  - scripts/code-scan/assemble_graph.py
  - tests/code_scan/test_assemble_graph.py
  - tests/code_scan/fixtures/graph-batch/
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - scripts/code-scan/scan_project.py
  - scripts/code-scan/extract_imports.py
  - scripts/code-scan/fingerprints.py
  - scripts/code-scan/language_registry.py
  - scripts/code-scan/graph_schema.py
  - .hermesignore
  - skills/
depends_on:
  - phase1-code-scan-completion-fix
  - phase2-d1-extract-imports
verification:
  - python -m pytest tests/code_scan/test_assemble_graph.py -v
  - python scripts/code-scan/assemble_graph.py tests/code_scan/fixtures/graph-batch/batch1.json tests/code_scan/fixtures/graph-batch/batch2.json --output /tmp/assembled.json
risk: medium
---

# Phase 3 D3 — assemble_graph.py: Merge Batch Outputs into Unified Dependency Graph

## Context & Intent

**Why this bead exists.** When a project is analyzed in parallel by multiple subagents (one per batch of files), each produces a partial scan/import result. Phase 3 needs a script that merges these batch outputs into a single, coherent dependency graph with proper deduplication, ID normalization, and edge merging. This adapts UA's `merge-batch-graphs.py` mechanical fixes into a stdlib-only Python script that integrates with Hermes's `graph_schema.py` for validation.

**Authority docs.** `.plans/phase-3-incremental-analysis.md` (D3 section + graph assembly format contract) defines the high-level scope. `.plans/ua-incorporation-strategy.md` (§1. Script-First / LLM-Second Analysis Pattern) mandates deterministic scripts for graph assembly.

**Intent.** Create `scripts/code-scan/assemble_graph.py` as a stdlib-only script that:
1. Reads multiple JSON inputs (scan outputs, import maps, or partial graph fragments from batch subagent runs).
2. Builds normalized node IDs for all entities (files, modules, functions, classes).
3. Builds edges from import relationships and scan metadata.
4. Deduplicates nodes by `node_id` — merges attributes from duplicates.
5. Deduplicates edges by `(source, target, edge_type)`.
6. Validates the final graph using `graph_schema.py`.
7. Outputs a unified graph JSON following the graph assembly format contract.

**Non-goals.** No tree-sitter/WASM. No new runtime dependencies. No dashboard, React UI, visualization, web endpoint. No SQLite. No CLI command beyond script flags. No modifications to existing Phase 1/2 files.

## Implementation Details

### Target files

| File | Purpose | Max LOC |
|---|---|---|
| `scripts/code-scan/assemble_graph.py` | Script: reads batch inputs, builds unified graph, deduplicates, validates, outputs | ~250 |
| `tests/code_scan/test_assemble_graph.py` | Unit + fixture-driven tests for all functions | ~200 |
| `tests/code_scan/fixtures/graph-batch/` | Batch input fixtures: overlapping batches, duplicate nodes/edges, clean inputs | varies |

### Required functions (exact naming)

| Function | Signature | Purpose |
|---|---|---|
| `load_batch_inputs` | `(paths: List[str]) -> List[dict]` | Read and parse all input JSON files. Returns list of parsed dicts. Raises `ValueError` on invalid JSON or missing file. |
| `normalize_node_id` | `(node_type: str, identifier: str) -> str` | Normalize a node ID using the ID prefix scheme: `file:<path>`, `module:<name>`, `func:<path>:<name>`, `class:<path>:<name>`. Lowercases the type prefix (unknown types get `unknown:<identifier>`). |
| `build_nodes_from_scan` | `(scan_data: dict) -> List[dict]` | Extract file-level nodes from a scan output. Each file in `scan_data["files"]` becomes a node: `{ "node_id": "file:<relative_path>", "node_type": "file", "filePath": relative_path, "label": basename, "language": language }`. Optionally includes `functions`, `classes`, `imports` if available. |
| `build_nodes_from_imports` | `(import_data: dict) -> List[dict]` | Extract module nodes from import data. For each unique imported module across all files, create a node: `{ "node_id": "module:<module_name>", "node_type": "module", "filePath": null, "label": module_name }`. Also creates `func:<path>:<name>` and `class:<path>:<name>` nodes if scan data is merged. |
| `build_edges_from_imports` | `(import_data: dict) -> List[dict]` | Extract import edges from import data. For each file's imports, create edges: `{ "source": "file:<relative_path>", "target": "module:<module_name>", "edge_type": "imports", "meta": {} }`. |
| `deduplicate_nodes` | `(nodes: List[dict]) -> tuple[List[dict], int]` | Deduplicate nodes by `node_id`. Merges attributes: if two nodes have the same `node_id`, keep first-seen values for `node_type`, `filePath`, `label`, `language`, and APPEND unique items from `functions`, `classes`, `imports` lists. Returns `(deduplicated_list, count_of_removed_duplicates)`. |
| `deduplicate_edges` | `(edges: List[dict]) -> tuple[List[dict], int]` | Deduplicate edges by `(source, target, edge_type)`. Keep first instance. Returns `(deduplicated_list, count_of_removed_duplicates)`. |
| `merge_nodes` | `(batch_nodes_lists: List[List[dict]]) -> List[dict]` | Merge nodes from multiple batches. Concatenate all node lists, then deduplicate. Returns merged node list. |
| `merge_edges` | `(batch_edges_lists: List[List[dict]]) -> List[dict]` | Merge edges from multiple batches. Concatenate all edge lists, then deduplicate. Returns merged edge list. |
| `assemble_graph` | `(scans: List[dict], imports_list: Optional[List[dict]]) -> dict` | Full assembly pipeline. Build nodes from all scans and imports, build edges from imports, merge and deduplicate, validate with graph_schema.py, return graph dict with summary. |
| `count_orphans` | `(nodes: List[dict], edges: List[dict]) -> int` | Count nodes that are not referenced as source or target in any edge. |
| `build_summary` | `(nodes: List[dict], edges: List[dict], dedup_nodes: int, dedup_edges: int, orphans: int) -> dict` | Build the summary section of the output graph. Returns `{ "total_nodes": int, "total_edges": int, "deduplicated_nodes": int, "deduplicated_edges": int, "orphan_nodes": int }`. |
| `main` | `() -> int` | CLI entry point. Parse args: `assemble_graph.py <input1.json> [input2.json ...] [--output file.json]`. Write JSON to stdout or file. Return exit code. |

### Graph output schema (exact contract)

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

Required top-level keys: `schema_version` (string, always `"1.0.0"`), `generated_at` (string, ISO 8601), `source_files` (list of input file paths used), `nodes` (list of node dicts), `edges` (list of edge dicts), `summary` (object with aggregate counts).

Required node keys: `node_id` (string, normalized), `node_type` (string, valid per graph_schema.py), `filePath` (string or null), `label` (string). Optional: `language`, `functions`, `classes`, `imports`.

Required edge keys: `source` (string, node_id), `target` (string, node_id), `edge_type` (string, valid per graph_schema.py), `meta` (object, can be empty `{}`).

### ID normalization rules (exact)

| Entity Type | Normalized ID Format | Example |
|---|---|---|
| File | `file:<relative_path>` | `file:src/main.py` |
| Module | `module:<module_name>` | `module:os` |
| Function | `func:<relative_path>:<function_name>` | `func:src/main.py:main` |
| Class | `class:<relative_path>:<class_name>` | `class:src/main.py:ConfigParser` |
| Unknown | `unknown:<identifier>` | `unknown:some_raw_id` |

Normalization rules:
- Type prefix is always lowercase.
- `relative_path` uses forward slashes (normalized from any OS separator).
- `module_name` is the raw import string as extracted (e.g., `os`, `react`, `./components/Header`).
- Path separators in node IDs must be forward slashes regardless of OS.

### Deduplication rules (exact)

1. **Node deduplication:** Two nodes with identical `node_id` are merged. Keep first-seen values for `node_type`, `filePath`, `label`, `language`. For list-valued fields (`functions`, `classes`, `imports`): concatenate lists from all duplicates, deduplicate, and sort.
2. **Edge deduplication:** Two edges with identical `(source, target, edge_type)` tuples are considered duplicates. Keep the first instance. The `meta` field from the first instance is preserved.
3. **Orphan counting:** A node is "orphan" if its `node_id` does not appear as a `source` or `target` in ANY edge. Orphans are NOT removed — they are counted in `summary.orphan_nodes`.

### Graph schema validation integration

After deduplication, the assembled graph must be validated using `graph_schema.py`:

```python
from graph_schema import validate_graph
result = validate_graph(graph_dict)
if result["issues"]:
    # Log issues but do not fail — produce the graph anyway with issues in meta
    for issue in result["issues"]:
        print(f"Graph validation issue: {issue}", file=sys.stderr)
```

Validation issues are printed to stderr but do not cause the script to fail (exit code 0 unless a fatal error occurs). This allows the graph to be produced even with warnings — the downstream validation-gate skill is responsible for deciding whether to block on issues.

### CLI interface

```
Usage:
    python assemble_graph.py <input1.json> [input2.json ...] [--output file.json] [--verbose]

Options:
    --output file.json    Write output to file instead of stdout
    --verbose             Print progress to stderr
```

Inputs can be any combination of scan_project.py outputs and extract_imports.py outputs. The script auto-detects input type based on schema (scan has a `files` array; import map has a `files` object with `imports` and `warnings`).

## Complexity Tier

**T3** — Multi-input merging with deduplication and schema validation. The merge logic requires careful attribute handling (append lists, deduplicate, sort). Graph schema integration adds a layer of indirection. Requires fixture design for overlapping/duplicate cases. Estimated 10-12 subagent iterations. Requires coder subagent + Hermes verification + reviewer signoff before presentation for commit approval.

## Execution Engine

**Executor:** `delegate-coder` — Hermes dispatches a coder subagent via `delegate_task` with full plan context.

**Process:**
1. Coder subagent creates fixture files (batch inputs with intentional overlaps/duplicates), then writes tests that fail (RED).
2. Implement `assemble_graph.py` to pass tests (GREEN).
3. Hermes runs `pytest tests/code_scan/test_assemble_graph.py -v` — all must pass (FULL).
4. Hermes runs end-to-end: assemble graph from multiple batch inputs, validate with graph_schema.py.
5. Reviewer subagent validates: spec compliance, deduplication correctness, ID normalization, schema validation integration, scope guardrails.
6. Hermes presents evidence bundle to JC — this bead authorizes implementation only, not commit/push.

**Subagent reliability preflight:**
- Task shape: merge/deduplicate script + schema validation + graph output
- Expected artifacts: 1 source file, 1 test file, fixture directory with ≥3 batch input pairs
- `max_iterations`: 15 per subagent dispatch
- File-write: YES. Run-test: YES. Commit: NO (await JC approval).

## Required Inline Context

### Project context

- **Repo:** `/home/jarrad/.hermes/hermes-agent`
- **Current branch:** `docs/ua-flywheel-phase3-plan`
- **Parent beads:** `phase1-code-scan-completion-fix` (scan output schema), `phase1-graph-schema` (graph_schema.py for validation), `phase2-d1-extract-imports` (import map schema)
- **No new pip dependencies** — stdlib only (`os`, `pathlib`, `json`, `argparse`, `datetime`, `sys`, `typing`, `collections`)
- **D3 import of graph_schema.py:** Must add `scripts/code-scan` to sys.path (same pattern as scan_project.py) to import graph_schema for validation

### Workspace cleanliness requirement

Phase 3 execution starts from a clean worktree for each bead. The executor must capture `git status --short` before and after the bead. Any file outside `allowed_files` is a scope violation and must be reverted before review.

### Scope guardrails

This bead may ONLY create or modify files listed in `allowed_files`. No existing Phase 1/2 files are modified. `graph_schema.py` is imported but not changed.

## Dependencies

| Dependency | Type | Status |
|---|---|---|
| Phase 1 code-scan scripts | prerequisite | Completed, committed, verified (80 tests pass) |
| Phase 1 graph_schema.py | prerequisite for validation import | Completed, committed, verified |
| Phase 2 extract_imports.py | runtime input schema reference | Completed, merged |
| Python 3.10+ stdlib | runtime | Assumed present |
| pytest | test runtime | Assumed present |

## Test Obligations

### TDD protocol

Strict RED → GREEN → REFACTOR for each public function. Every public function must have at least one failing-first test.

### Test file mapping

| Source | Test | Coverage target |
|---|---|---|
| `load_batch_inputs` | `test_assemble_graph.py` | Valid JSON load; missing file raises; invalid JSON raises; empty list |
| `normalize_node_id` | `test_assemble_graph.py` | file:src/main.py; module:os; func:src/main.py:main; class:src/main.py:Config; unknown:raw; lowercase prefix; forward slash normalization |
| `build_nodes_from_scan` | `test_assemble_graph.py` | Scan with 3 files → 3 file nodes; correct node_ids, filePaths, labels, languages |
| `build_nodes_from_imports` | `test_assemble_graph.py` | Import map with 5 files importing 8 unique modules → 8 module nodes |
| `build_edges_from_imports` | `test_assemble_graph.py` | Import map → edges with file→module imports; correct edge_type "imports" |
| `deduplicate_nodes` | `test_assemble_graph.py` | Duplicate node_id: merge attributes, dedup lists, sorted result; count correct |
| `deduplicate_edges` | `test_assemble_graph.py` | Duplicate (source, target, edge_type): keep first; count correct |
| `merge_nodes` | `test_assemble_graph.py` | Two batches with overlapping file: merged node has combined attributes |
| `merge_edges` | `test_assemble_graph.py` | Two batches with overlapping imports: deduplicated |
| `assemble_graph` | `test_assemble_graph.py` | Full pipeline: scans + imports → graph; schema correct; summary counts match |
| `count_orphans` | `test_assemble_graph.py` | Nodes without edges counted correctly; referenced nodes not counted |
| `main()` | `test_assemble_graph.py` | CLI arg parsing; stdout output; file output; non-zero exit on invalid input |
| Graph schema validation | `test_assemble_graph.py` | Valid graph → zero issues; invalid graph → issues reported (but script doesn't fail) |

### Fixture requirements

`tests/code_scan/fixtures/graph-batch/` must contain:

| Fixture | Purpose |
|---|---|
| `batch1.json` | Scan output for 5 files with known languages — batch 1 |
| `batch2.json` | Scan output for 5 files, 2 overlapping with batch 1 — tests deduplication |
| `imports1.json` | Import map for batch 1 files — known imports |
| `imports2.json` | Import map for batch 2 files — some shared imports with batch 1 |
| `overlapping.json` | Intentional duplicate entries — tests merge correctness |
| `minimal.json` | Single file, no imports — tests minimal graph assembly |
| `bad_schema.json` | Invalid input — tests error handling in load_batch_inputs |

### RED/GREEN/FULL evidence required

- **RED:** Tests fail with `AttributeError` / `ModuleNotFoundError` / incorrect merge results
- **GREEN:** Each test passes with minimal implementation
- **FULL:** `pytest tests/code_scan/test_assemble_graph.py -v` — all pass, no warnings

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent
source venv/bin/activate

# Step 1: Unit tests
python -m pytest tests/code_scan/test_assemble_graph.py -v

# Step 2: End-to-end graph assembly
python scripts/code-scan/scan_project.py /home/jarrad/work/testbeds/ua-flywheel/cass_memory_system --output /tmp/scan-cass.json
python scripts/code-scan/extract_imports.py /tmp/scan-cass.json > /tmp/imports-cass.json
python scripts/code-scan/assemble_graph.py /tmp/scan-cass.json /tmp/imports-cass.json --output /tmp/graph-cass.json
python -c "
import json
g = json.load(open('/tmp/graph-cass.json'))
assert g['schema_version'] == '1.0.0'
assert 'nodes' in g
assert 'edges' in g
assert 'summary' in g
assert g['summary']['total_nodes'] > 0
assert g['summary']['total_edges'] >= 0
assert g['summary']['deduplicated_nodes'] >= 0
print(f'GRAPH ASSEMBLY PASS: {g[\"summary\"][\"total_nodes\"]} nodes, {g[\"summary\"][\"total_edges\"]} edges')
"

# Step 3: Node ID normalization check
python -c "
import json
g = json.load(open('/tmp/graph-cass.json'))
for node in g['nodes']:
    nid = node['node_id']
    assert ':' in nid, f'Node ID missing colon separator: {nid}'
    prefix = nid.split(':')[0]
    assert prefix in ('file', 'module', 'func', 'class', 'unknown'), f'Unknown prefix: {prefix}'
print('ID NORMALIZATION PASS')
"

# Step 4: Schema validation
python -c "
import sys, json
sys.path.insert(0, 'scripts/code-scan')
from graph_schema import validate_graph
g = json.load(open('/tmp/graph-cass.json'))
result = validate_graph(g)
if result['issues']:
    print(f'Validation issues (non-fatal):')
    for issue in result['issues']:
        print(f'  - {issue}')
else:
    print('GRAPH SCHEMA VALIDATION PASS')
"

# Step 5: Scope guardrail — no existing files modified
git diff -- scripts/code-scan/scan_project.py scripts/code-scan/extract_imports.py scripts/code-scan/language_registry.py scripts/code-scan/graph_schema.py scripts/code-scan/fingerprints.py .hermesignore | grep -q . && echo 'GUARDRAIL FAIL' && exit 1 || echo 'GUARDRAIL PASS'

# Step 6: Forbidden file check
git diff -- tools/skills_sync.py tests/tools/test_skills_sync.py | grep -q . && echo 'FORBIDDEN FAIL' || echo 'FORBIDDEN PASS'

# Step 7: No new runtime dependencies
python -c "
import ast
from pathlib import Path
stdlib = {'argparse','collections','dataclasses','datetime','enum','fnmatch','hashlib','json','os','pathlib','re','sys','typing','itertools','tempfile'}
tree = ast.parse(Path('scripts/code-scan/assemble_graph.py').read_text())
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.update(a.name.split('.')[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        imports.add(node.module.split('.')[0])
# graph_schema is a sibling import, not a pip dependency
nonstdlib = sorted(i for i in imports if i not in stdlib and i != 'graph_schema')
if nonstdlib:
    raise SystemExit(f'Non-stdlib imports: {nonstdlib}')
print('DEPENDENCY PASS')
"
```

### Expected pass criteria

1. `pytest tests/code_scan/test_assemble_graph.py -v` — 100% pass
2. Graph assembly: output JSON has correct schema, non-zero nodes
3. Node ID normalization: all node_ids have correct prefix format
4. Schema validation: graph passes graph_schema.py validation (or has non-fatal warnings)
5. Scope guardrail: no existing Phase 1/2/3 files modified
6. Forbidden files unchanged
7. No new non-stdlib imports (graph_schema import allowed as sibling)

## Approval Evidence

### Before commit — present this evidence bundle to JC

**1. Test output (verbatim):**
```
$ python -m pytest tests/code_scan/test_assemble_graph.py -v
```
Include the full output showing PASS/FAIL for every test.

**2. RED/GREEN/FULL evidence per function:**
- [ ] `load_batch_inputs`: RED → GREEN → FULL
- [ ] `normalize_node_id`: RED → GREEN → FULL
- [ ] `build_nodes_from_scan`: RED → GREEN → FULL
- [ ] `build_nodes_from_imports`: RED → GREEN → FULL
- [ ] `build_edges_from_imports`: RED → GREEN → FULL
- [ ] `deduplicate_nodes`: RED → GREEN → FULL
- [ ] `deduplicate_edges`: RED → GREEN → FULL
- [ ] `merge_nodes`: RED → GREEN → FULL
- [ ] `merge_edges`: RED → GREEN → FULL
- [ ] `assemble_graph`: RED → GREEN → FULL
- [ ] `count_orphans`: RED → GREEN → FULL
- [ ] `build_summary`: RED → GREEN → FULL
- [ ] `main()`: RED → GREEN → FULL
- [ ] Graph schema validation integration: RED → GREEN → FULL

**3. Deduplication evidence:**
```bash
# Create two batch inputs with known duplicates
# Run assemble_graph.py, verify output has unique nodes/edges only
# Verify summary deduplication counts match expected values
```

**4. Schema compliance:**
```python
g = json.load(open(graph_path))
assert g['schema_version'] == '1.0.0'
assert isinstance(g['nodes'], list)
assert isinstance(g['edges'], list)
assert isinstance(g['summary'], dict)
for key in ['total_nodes', 'total_edges', 'deduplicated_nodes', 'deduplicated_edges', 'orphan_nodes']:
    assert key in g['summary'], f'Missing summary key: {key}'
```

**5. Scope guardrail:**
```bash
git diff --name-only | grep -vE '^(scripts/code-scan/assemble_graph\.py|tests/code_scan/)' && echo 'SCOPE FAIL' || echo 'SCOPE PASS'
```

**6. Workspace cleanliness preserved:**
```
`git diff --name-only` shows only D3 allowed files
```

**7. Reviewer verdict:**
- [ ] Spec compliance (all functions present, ID normalization correct, deduplication works)
- [ ] Scope preservation (no Phase 1/2/other files touched)
- [ ] No new runtime dependencies (stdlib only, graph_schema import allowed)
- [ ] Context budget (zero — script only)
- [ ] Quality and security (no secrets, no path leaks, deterministic output)
- [ ] Workspace remains clean outside allowed files

**8. Commit gate:**
```
NO COMMIT, PUSH, OR MERGE until JC explicitly approves.
```

---

> **Bead execution readiness = this bead passes reviewer polish and JC approves execution.**
> **Bead completion = all verification commands exit 0 + reviewer PASS + JC commit approval.**
> Coder subagent has NO commit/push authority.
