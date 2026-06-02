---
id: phase2-d3-validation-gate-skill
title: Phase 2 D3 — validation-gate SKILL.md: two-phase reviewer skill
status: complete-and-committed
executor: delegate-coder
parallel_safe: false
base_branch: docs/ua-flywheel-phase1-phase2-plan
allowed_files:
  - skills/code-analysis/validation-gate/SKILL.md
forbidden_files:
  - tools/skills_sync.py
  - tests/tools/test_skills_sync.py
  - scripts/code-scan/scan_project.py
  - scripts/code-scan/language_registry.py
  - scripts/code-scan/graph_schema.py
  - scripts/code-scan/extract_imports.py
  - .hermesignore
  - skills/code-analysis/code-scan/
depends_on:
  - phase1-code-scan-completion-fix
  - phase2-d1-extract-imports
verification:
  - wc -l skills/code-analysis/validation-gate/SKILL.md
  - python scripts/code-scan/graph_schema.py --help 2>/dev/null || python -c "import importlib.util; spec=importlib.util.spec_from_file_location('graph_schema','scripts/code-scan/graph_schema.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('IMPORT PASS')"
  - python -c "
    import json
    from pathlib import Path
    spec_path = 'scripts/code-scan/graph_schema.py'
    # Inline contract test: create minimal graph, validate
    exec(compile(Path(spec_path).read_text(), spec_path, 'exec'))
    result = validate_graph({'nodes': [], 'edges': []})
    assert isinstance(result, dict), 'validate_graph must return dict'
    assert 'issues' in result
    assert 'warnings' in result
    print('CONTRACT PASS')
    "
risk: low
---

# Phase 2 D3 — validation-gate SKILL.md: Two-Phase Reviewer Skill

## Context & Intent

**Why this bead exists.** After the code-scan skill (D2) produces scan output and import maps, agents need a deterministic way to verify the integrity of any graph-like or analysis artifacts before proceeding. The validation-gate skill implements a two-phase pattern: Phase 1 runs a Python validation script (deterministic); Phase 2 reads the results and renders APPROVED/WARNING/REJECTED (LLM rendering of deterministic output). This maps to the existing **Revision gate** in the gates taxonomy.

**Authority docs.** `.plans/phase-2-flywheel-ua-integration.md` (§D3: `skills/code-analysis/validation-gate/SKILL.md`) defines scope: ≤80 lines, uses `graph_schema.py` for schema validation, two-phase pattern (script execution + LLM rendering), maps to Revision gate.

**Intent.** Create a single `SKILL.md` file at `skills/code-analysis/validation-gate/SKILL.md` that defines: (1) how to run `graph_schema.py` validation against an artifact, (2) how to interpret results, (3) how to render APPROVED/WARNING/REJECTED verdicts with structured notes. The skill is JIT-loaded only when a bead/stage requires verification of graph-like output.

**Non-goals.** No auto-injection. No dashboard. No React UI. No CLI command. No SQLite store. No tree-sitter/WASM. No new scripts — this is a prompt-only skill that references existing `graph_schema.py`. No modifications to Phase 1 or Phase 2 D1 files.

## Implementation Details

### Target file

| File | Purpose | Max LOC |
|---|---|---|
| `skills/code-analysis/validation-gate/SKILL.md` | JIT skill: two-phase validation orchestration | ≤80 |

### Required frontmatter

```yaml
---
name: validation-gate
hermes.tags: [on-demand, code-analysis, quality-gate]
---
```

The `on-demand` tag is mandatory. The `quality-gate` tag aids discoverability.

### Skill behavior (exact steps)

**Phase 1 — Deterministic validation:**
1. Accept a target artifact: path to graph JSON, scan output, or analysis result file.
2. Write or reference a Python validation script that imports `graph_schema.py` and runs `validate_graph()` (or `validate_node()`, `validate_edge()` as appropriate).
3. Execute the script → capture JSON results.
4. Parse results: `{"issues": [...], "warnings": [...]}`.

**Phase 2 — LLM rendering:**
5. Interpret results and render verdict:
   - **APPROVED:** `issues` is empty. Warnings may exist but are rendered as notes.
   - **WARNING:** `issues` is empty but `warnings` is non-empty. Render warnings as structured notes. Proceed with caution.
   - **REJECTED:** `issues` is non-empty. Render issues as structured blockers. Trigger revision gate (request changes before proceeding).
6. Present structured report to user following the output format below.

### Validation checks the skill must reference

- Node types are valid (via `NodeType` enum or `NODE_TYPE_ALIASES` map in `graph_schema.py`)
- Edge types are valid (via `EdgeType` enum or `EDGE_TYPE_ALIASES` map in `graph_schema.py`)
- No orphan nodes (all edges reference existing node IDs)
- No self-referencing edges (unless type allows)
- `node_id` present and string for each node
- `filePath` present and relative for each node
- `edge_type` present and resolves for each edge
- `source` and `target` reference existing node IDs for each edge

### Output format (prescribed)

```markdown
## Validation Gate: [APPROVED | WARNING | REJECTED]

### Summary
- **Issues:** <count> critical
- **Warnings:** <count> non-blocking

### Issues (if any)
- <issue 1>
- <issue 2>

### Warnings (if any)
- <warning 1>
- <warning 2>

### Recommendation
<Proceed with review / Fix critical issues before proceeding / Request changes>
```

### Line budget check

The SKILL.md MUST NOT exceed 80 lines. This includes frontmatter, all headers, body text, code blocks, and the output format example.

### No LLM intuition constraint

The skill must explicitly state: "Do NOT validate using LLM intuition — only report what the validation script returns." This prevents the agent from "feeling" that something is wrong based on content rather than schema contracts.

## Complexity Tier

**T1** — Single-file authoring. Prompt-only (no code logic). References existing `graph_schema.py` which is already implemented and tested in Phase 1. Estimated 2–3 subagent iterations.

## Execution Engine

**Executor:** `delegate-coder` — Hermes dispatches a coder subagent with the exact SKILL.md content spec.

**Process:**
1. Coder subagent writes `skills/code-analysis/validation-gate/SKILL.md`.
2. Hermes verifies: `wc -l` ≤ 80, frontmatter present, `on-demand` tag present, all validation checks referenced, output format included.
3. Hermes verifies `graph_schema.py` contract: `validate_graph()` returns `{"issues": [], "warnings": []}` with empty input.
4. Reviewer subagent validates: line budget, spec compliance, scope guardrails, no forbidden-file touches.
5. Hermes presents evidence to JC.

**Subagent reliability preflight:**
- Task shape: single markdown file authoring with reference to existing module
- Expected artifacts: 1 file, ≤80 lines
- `max_iterations`: 5 per subagent dispatch
- File-write: YES. Run-test: YES. Commit: NO.

## Required Inline Context

### Project context

- **Repo:** `/home/jarrad/.hermes/hermes-agent`
- **Current branch:** `docs/ua-flywheel-phase1-phase2-plan`

### graph_schema.py contract (from Phase 1)

```python
# From scripts/code-scan/graph_schema.py
class NodeType(str, Enum): ...
class EdgeType(str, Enum): ...
NODE_TYPE_ALIASES: dict[str, NodeType] = {...}
EDGE_TYPE_ALIASES: dict[str, EdgeType] = {...}

def validate_node(node: dict) -> list[str]: ...
def validate_edge(edge: dict, known_node_ids: set[str]) -> list[str]: ...
def validate_graph(graph: dict) -> dict[str, list[str]]:
    # Returns {"issues": [...], "warnings": [...]}
```

The skill references these functions by name but does NOT include their source code.

### Existing dirty files — DO NOT TOUCH

```
tools/skills_sync.py                 # dirty
tests/tools/test_skills_sync.py      # dirty
```

### Contract tests (line budget + graph_schema contract)

```bash
# Line budget
wc -l skills/code-analysis/validation-gate/SKILL.md
# Must output ≤ 80

# Frontmatter
grep "^name: validation-gate" skills/code-analysis/validation-gate/SKILL.md
grep "on-demand" skills/code-analysis/validation-gate/SKILL.md

# graph_schema.py contract test
python -c "
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('graph_schema', 'scripts/code-scan/graph_schema.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
result = mod.validate_graph({'nodes': [], 'edges': []})
assert isinstance(result, dict)
assert 'issues' in result
assert 'warnings' in result
assert result['issues'] == []
assert result['warnings'] == []
print('GRAPH_SCHEMA CONTRACT PASS')
"
```

## Dependencies

| Dependency | Type | Status |
|---|---|---|
| Phase 1 graph_schema.py | prerequisite + runtime reference | Completed, committed, verified |
| Phase 1 code-scan scripts | prerequisite | Completed |
| D1: extract_imports.py | prerequisite (for future graph assembly) | Must be implemented before full pipeline |

## Test Obligations

### Contract tests

| Check | Command | Pass criteria |
|---|---|---|
| File exists | `test -f skills/code-analysis/validation-gate/SKILL.md` | Exit 0 |
| Line budget | `wc -l skills/code-analysis/validation-gate/SKILL.md` | ≤80 |
| Frontmatter name | `grep -q "^name: validation-gate" skills/code-analysis/validation-gate/SKILL.md` | Match found |
| On-demand tag | `grep -q "on-demand" skills/code-analysis/validation-gate/SKILL.md` | Match found |
| References graph_schema.py | `grep -q "graph_schema.py" skills/code-analysis/validation-gate/SKILL.md` | Match found |
| References validate_graph | `grep -q "validate_graph" skills/code-analysis/validation-gate/SKILL.md` | Match found |
| References APPROVED/WARNING/REJECTED | `grep -qE "APPROVED|WARNING|REJECTED" skills/code-analysis/validation-gate/SKILL.md` | All three found |
| No LLM intuition clause | `grep -qi "LLM intuition" skills/code-analysis/validation-gate/SKILL.md` | Match found |
| graph_schema.py contract | Inline Python test (see above) | Returns `{"issues": [], "warnings": []}` for empty graph |

### RED/GREEN/FULL evidence required

- **RED:** File does not exist or has >80 lines or missing required content
- **GREEN:** File exists, ≤80 lines, all required content present
- **FULL:** All contract tests pass, graph_schema.py contract verified, reviewer approves

## Verification Command

```bash
cd /home/jarrad/.hermes/hermes-agent

# Step 1: File exists
test -f skills/code-analysis/validation-gate/SKILL.md && echo "FILE PASS" || echo "FILE FAIL"

# Step 2: Line budget ≤80
LINES=$(wc -l < skills/code-analysis/validation-gate/SKILL.md)
[ "$LINES" -le 80 ] && echo "BUDGET PASS ($LINES lines)" || echo "BUDGET FAIL ($LINES lines)"

# Step 3: Required content
grep -q "^name: validation-gate" skills/code-analysis/validation-gate/SKILL.md && echo "FRONTMATTER PASS" || echo "FRONTMATTER FAIL"
grep -q "on-demand" skills/code-analysis/validation-gate/SKILL.md && echo "TAG PASS" || echo "TAG FAIL"
grep -q "graph_schema.py" skills/code-analysis/validation-gate/SKILL.md && echo "GRAPH_SCHEMA PASS" || echo "GRAPH_SCHEMA FAIL"
grep -q "validate_graph" skills/code-analysis/validation-gate/SKILL.md && echo "VALIDATE_GRAPH PASS" || echo "VALIDATE_GRAPH FAIL"
grep -q "APPROVED" skills/code-analysis/validation-gate/SKILL.md && echo "APPROVED PASS" || echo "APPROVED FAIL"
grep -q "WARNING" skills/code-analysis/validation-gate/SKILL.md && echo "WARNING PASS" || echo "WARNING FAIL"
grep -q "REJECTED" skills/code-analysis/validation-gate/SKILL.md && echo "REJECTED PASS" || echo "REJECTED FAIL"
grep -qi "LLM intuition" skills/code-analysis/validation-gate/SKILL.md && echo "NO-LLM-INTUITION PASS" || echo "NO-LLM-INTUITION FAIL"

# Step 4: graph_schema.py contract test
python -c "
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('graph_schema', 'scripts/code-scan/graph_schema.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
result = mod.validate_graph({'nodes': [], 'edges': []})
assert isinstance(result, dict)
assert 'issues' in result
assert 'warnings' in result
print('CONTRACT PASS')
"

# Step 5: Scope guardrail
git diff --name-only | grep -vE '^(skills/code-analysis/validation-gate/SKILL\.md|\.beads/|\.plans/)' && echo 'SCOPE FAIL' || echo 'SCOPE PASS'

# Step 6: Forbidden files
git diff -- tools/skills_sync.py tests/tools/test_skills_sync.py 2>/dev/null | wc -l && echo "FORBIDDEN PASS" || echo "FORBIDDEN FAIL"
```

### Expected pass criteria

1. File exists at exact path
2. Line count ≤80
3. All required content present (frontmatter, tags, validation steps, verdict types, no-LLM clause)
4. graph_schema.py contract verified (validate_graph returns correct structure)
5. Only allowed file modified
6. Forbidden files untouched

## Approval Evidence

### Before commit — present this evidence bundle to JC

**1. Line count:**
```bash
wc -l skills/code-analysis/validation-gate/SKILL.md
```
Expected: ≤80.

**2. Content verification:**
```
FRONTMATTER PASS
TAG PASS
GRAPH_SCHEMA PASS
VALIDATE_GRAPH PASS
APPROVED PASS
WARNING PASS
REJECTED PASS
NO-LLM-INTUITION PASS
CONTRACT PASS
```

**3. File content preview:**
```bash
cat skills/code-analysis/validation-gate/SKILL.md
```

**4. Scope guardrail:**
```bash
git diff --name-only
# Only: .beads/phase2-d3-validation-gate-skill.md, skills/code-analysis/validation-gate/SKILL.md, plus planning files
```

**5. Reviewer verdict:**
- [ ] Spec compliance (two-phase pattern, all verdicts, ≤80 lines, references graph_schema.py)
- [ ] Scope preservation (no Phase 1 files, no forbidden patterns)
- [ ] Context budget (SKILL.md ≤80 lines, on-demand tag)
- [ ] No LLM intuition enforcement present
- [ ] Existing dirty files not modified

**6. Commit gate:**
```
NO COMMIT, PUSH, OR MERGE until JC explicitly approves.
```

---

> **Bead execution readiness = this bead passes reviewer polish and JC approves execution.**
> **Bead completion = all verification commands exit 0 + reviewer PASS + JC commit approval.**
> Coder subagent has NO commit/push authority.
