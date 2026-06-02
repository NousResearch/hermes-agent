---
bead_name: phase2-d1-d3-evaluation-plan-corrections
type: correction
parent_beads:
  - phase2-d1-extract-imports
  - phase2-d2-code-scan-skill
  - phase2-d3-validation-gate-skill
parent_plan: .plans/phase-2-d1-d3-evaluation-plan.md
eval_run_ref: "2026-05-30 evaluation run — plan-as-written FAIL (TC-2, TC-6)"
priority: medium
scope: documentation-only
status: prepared-not-executed
created: "2026-05-30"
constraints:
  - "No D4 work — D4 deferred; zero references to D4 implementation"
  - "No implementation changes to .py, .js, .rs, .go, or SKILL.md files"
  - "No push or merge without explicit JC approval"
  - "Documentation-only changes to evaluation plan and runner script"
  - "Do not modify tools/skills_sync.py or tests/tools/test_skills_sync.py"
---

# Bead: Phase 2 D1-D3 Evaluation Plan Corrections

## Exec Summary

The evaluation run on 2026-05-30 produced **FAIL** verdicts — but investigation confirmed the failures are caused by **bugs in the evaluation plan itself**, not by implementation defects in D1, D2, or D3.

| Failed TC | Root cause | Correction |
|----------|-----------|------------|
| TC-2 (TS fixtures) | Expected imports list omitted `lodash` even though `tests/code_scan/fixtures/imports/ts_sample.ts` contains `const lodash = require("lodash");` | Add `"lodash"` to the ts_sample expected imports list |
| TC-6.2 (valid graph) | Test used a self-edge `source: 'a', target: 'a'` but `graph_schema.py` line 142–144 explicitly forbids self-referencing edges | Replace with two nodes and a non-self edge |
| Runner denomination | TC-9 and TC-10 grouped as `tc9_10` → 10 runner records vs 11 planned test cases | Split into separate runner records so denominator = 11 |

This bead contains the **exact, atomic corrections** to apply to the evaluation plan and runner script, plus the verification steps to confirm the corrected plan produces PASS.

---

## Allowed Files

| File | Change type | Description |
|------|------------|-------------|
| `.plans/phase-2-d1-d3-evaluation-plan.md` | EDIT | TC-2 expected imports, TC-6.2 valid graph case |
| `.plans/_eval-guardrails.sh` | NO CHANGE | Do not modify — reference only |
| `.plans/phase-2-d1-d3-evaluation-plan.md` (Section 10 runner) | EDIT | Split TC-9/TC-10 into separate records |

## Forbidden Files

The following files must **NOT** be touched under any circumstances:

- `tools/skills_sync.py`
- `tests/tools/test_skills_sync.py`
- All files under `skills/` (SKILL.md files included)
- All files under `scripts/code-scan/*.py` (implementation scripts)
- All files under `tests/code_scan/` (test fixtures, test code)
- Any `.py`, `.md` outside `.plans/` and `.beads/`
- Any file not listed in "Allowed Files" above

---

## Correction Instructions

### Correction C1: TC-2 — Add `lodash` to TS expected imports

**File:** `.plans/phase-2-d1-d3-evaluation-plan.md`
**Location:** Section 5 (Test Cases) → TC-2 → Expected fixture outputs table

**Current row:**
```
| `ts_sample.ts` | `["react", "react-dom", "./App", "./utils"]` |
```

**New row:**
```
| `ts_sample.ts` | `["react", "react-dom", "./App", "./utils", "lodash"]` |
```

**Also update** the inline fixtures dict in the TC-2 script block:

**Current:**
```python
'ts': ('tests/code_scan/fixtures/imports/ts_sample.ts',
       extract_js_ts_imports, ['react', 'react-dom', './App', './utils']),
```

**New:**
```python
'ts': ('tests/code_scan/fixtures/imports/ts_sample.ts',
       extract_js_ts_imports, ['react', 'react-dom', './App', './utils', 'lodash']),
```

**Reason:** `ts_sample.ts` contains `const lodash = require("lodash");` on line 5. The extractor correctly finds this import; the plan's expected list was incomplete, causing a false-positive FAIL.

---

### Correction C2: TC-6.2 — Use two nodes, non-self edge

**File:** `.plans/phase-2-d1-d3-evaluation-plan.md`
**Location:** Section 5 → TC-6 → Test 2 (valid graph)

**Current test:**
```python
# Test 2: Valid nodes + edges → APPROVED
r = validate_graph({
    'nodes': [{'node_id': 'a', 'filePath': 'src/main.py', 'node_type': 'module'}],
    'edges': [{'source': 'a', 'target': 'a', 'edge_type': 'imports'}]
})
```

**New test:**
```python
# Test 2: Valid nodes + edges → APPROVED
r = validate_graph({
    'nodes': [
        {'node_id': 'a', 'filePath': 'src/main.py', 'node_type': 'module'},
        {'node_id': 'b', 'filePath': 'src/utils.py', 'node_type': 'module'}
    ],
    'edges': [{'source': 'a', 'target': 'b', 'edge_type': 'imports'}]
})
```

**Reason:** `validate_edge()` at line 143 checks `source == target` and flags it as a self-referencing edge issue. The original test used `source: 'a', target: 'a'`, which the schema correctly rejects. The corrected version uses two distinct nodes (`'a'` and `'b'`) with a valid `a → b` edge, which `validate_graph()` correctly returns with zero issues.

---

### Correction C3: Split TC-9/TC-10 runner record

**File:** `.plans/phase-2-d1-d3-evaluation-plan.md`
**Location:** Section 10 — Single-Command Runner, the combined `tc9_10` line

**Current:**
```bash
# TC-9 + TC-10 (guardrails)
run_test "tc9_10" bash ~/.hermes/hermes-agent/.plans/_eval-guardrails.sh
```

**New — two separate records:**
```bash
# TC-9: Scope guardrail
run_test "tc9" bash -c 'cd /home/jarrad/.hermes/hermes-agent && bash .plans/_eval-guardrails.sh 2>&1 | grep -q "TC-9 PASS"'

# TC-10: D4 absent
run_test "tc10" bash -c 'cd /home/jarrad/.hermes/hermes-agent && bash .plans/_eval-guardrails.sh 2>&1 | grep -q "TC-10 PASS: D4 correctly absent"'
```

**Reason:** The runner currently groups TC-9 and TC-10 into one record (`tc9_10`), producing 10 pass/fail rows. The plan defines 11 test cases (TC-1 through TC-11). This split ensures the runner denominator matches the planned test count.

**Also update** the Pass/Fail Summary table in Section 6 if the runner summary references row counts: verify the summary lists TC-9 and TC-10 on separate rows (they are already separate in Section 6 — this is purely a runner fix).

---

## Execution Steps

### Step 0: Pre-flight checks

```bash
cd /home/jarrad/.hermes/hermes-agent
git status --short        # Must be empty except branch-ahead metadata
git branch --show-current # Must be: docs/ua-flywheel-phase1-phase2-plan or a JC-approved correction branch
git log --oneline -5      # Must include 3c63af2e0 (docs: close Phase 2 and plan D1-D3 evaluation)
```

### Step 1: Create a working branch, if JC wants isolation

```bash
git checkout -b eval-plan-corrections docs/ua-flywheel-phase1-phase2-plan
```

If JC directs execution on the existing phase branch, skip branch creation and apply corrections directly there.

### Step 2: Apply corrections C1, C2, C3

Apply each correction from the instructions above using `patch`. No implementation files are touched.

### Step 3: Verify corrections are documentation-only

```bash
git diff --name-only
# Expected output — should contain ONLY:
# .plans/phase-2-d1-d3-evaluation-plan.md
# If any other files appear, STOP and review.
```

### Step 4: Run TC-2 (corrected)

```bash
cd /home/jarrad/.hermes/hermes-agent
python -c "
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts/code-scan')
from extract_imports import extract_python_imports, extract_js_ts_imports, \
    extract_rust_imports, extract_go_imports, extract_shell_imports

fixtures = {
    'python': ('tests/code_scan/fixtures/imports/python_sample.py',
               extract_python_imports, ['os', 'sys', 'json', 'pathlib']),
    'ts': ('tests/code_scan/fixtures/imports/ts_sample.ts',
           extract_js_ts_imports, ['react', 'react-dom', './App', './utils', 'lodash']),
    'rust': ('tests/code_scan/fixtures/imports/rust_sample.rs',
             extract_rust_imports, ['std', 'serde', 'tokio']),
    'go': ('tests/code_scan/fixtures/imports/go_sample.go',
           extract_go_imports, ['fmt', 'net/http', 'github.com/gin-gonic/gin']),
    'shell': ('tests/code_scan/fixtures/imports/shell_sample.sh',
              extract_shell_imports, ['env.sh', '~/.bashrc']),
}

all_pass = True
for lang, (fixture_path, extractor, expected) in fixtures.items():
    content = Path(fixture_path).read_text()
    actual = extractor(content)
    expected_set = set(expected)
    actual_set = set(actual)
    tp = len(expected_set & actual_set)
    fp = len(actual_set - expected_set)
    fn = len(expected_set - actual_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    status = 'PASS' if precision >= 0.90 and recall >= 0.85 else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    print(f'{lang}: TP={tp} FP={fp} FN={fn} precision={precision:.2f} recall={recall:.2f} [{status}]')
    if fp > 0:
        print(f'  False positives: {actual_set - expected_set}')
    if fn > 0:
        print(f'  False negatives: {expected_set - actual_set}')

if all_pass:
    print('\nTC-2 PASS: All languages meet precision/recall thresholds')
else:
    print('\nTC-2 FAIL: Some languages below thresholds')
    sys.exit(1)
"
```

### Step 5: Run TC-6 (corrected)

```bash
cd /home/jarrad/.hermes/hermes-agent
python -c "
import sys
from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location('graph_schema', 'scripts/code-scan/graph_schema.py')
mod = module_from_spec(spec)
spec.loader.exec_module(mod)
validate_graph = mod.validate_graph

passed = True

# Test 1: Valid empty graph → APPROVED (no issues)
r = validate_graph({'nodes': [], 'edges': []})
if r.get('issues') == [] or r.get('issues') is None:
    print('TC-6.1: Empty graph → APPROVED PASS [issues=%d]' % len(r.get('issues',[])))
else:
    print('TC-6.1: FAIL — empty graph has issues: %s' % r.get('issues'))
    passed = False

# Test 2: Valid nodes + edges (two nodes, non-self edge) → APPROVED
r = validate_graph({
    'nodes': [
        {'node_id': 'a', 'filePath': 'src/main.py', 'node_type': 'module'},
        {'node_id': 'b', 'filePath': 'src/utils.py', 'node_type': 'module'}
    ],
    'edges': [{'source': 'a', 'target': 'b', 'edge_type': 'imports'}]
})
if len(r.get('issues', [])) == 0:
    print('TC-6.2: Valid graph (a→b) → APPROVED PASS')
else:
    print('TC-6.2: FAIL — valid graph has issues: %s' % r.get('issues'))
    passed = False

# Test 3: Missing node_id → REJECTED
r = validate_graph({
    'nodes': [{'filePath': 'src/main.py'}],
    'edges': []
})
if len(r.get('issues', [])) > 0:
    print('TC-6.3: Missing node_id → REJECTED PASS [issues=%d]' % len(r.get('issues')))
else:
    print('TC-6.3: FAIL — missing node_id not caught')
    passed = False

# Test 4: Orphan edge → REJECTED
r = validate_graph({
    'nodes': [{'node_id': 'x', 'filePath': 'a.py', 'node_type': 'module'}],
    'edges': [{'source': 'x', 'target': 'nonexistent', 'edge_type': 'imports'}]
})
if len(r.get('issues', [])) > 0:
    print('TC-6.4: Orphan edge → REJECTED PASS [issues=%d]' % len(r.get('issues')))
else:
    print('TC-6.4: FAIL — orphan edge not caught')
    passed = False

print()
if passed:
    print('TC-6 PASS: Verdict accuracy confirmed on test corpus')
else:
    print('TC-6 FAIL: Some verdict tests failed')
    sys.exit(1)
"
```

### Step 6: Verify all other TCs are unaffected

```bash
cd /home/jarrad/.hermes/hermes-agent
# TC-1
python -m pytest tests/code_scan/test_extract_imports.py -v --tb=short

# TC-11
python -m pytest tests/code_scan/ -q --tb=line

# Guardrails (TC-9 + TC-10 combined for verification; the runner splits them)
bash .plans/_eval-guardrails.sh
```

### Step 7: Review

```bash
git diff
# Review all changes — must be ONLY TC-2, TC-6.2, and runner split
```

### Step 8: Commit (do NOT push yet)

```bash
git add .plans/phase-2-d1-d3-evaluation-plan.md
git commit -m "docs: fix evaluation plan TC-2 (ts lodash), TC-6.2 (self-edge), runner denom"
```

---

## Pass/Fail Criteria

| Criterion | PASS condition | FAIL condition |
|-----------|---------------|----------------|
| C1 applied | TS expected imports includes `"lodash"` | `"lodash"` missing from TS expected list |
| C2 applied | TC-6.2 uses two nodes (`'a'`, `'b'`) with non-self edge | TC-6.2 still uses `source: 'a', target: 'a'` |
| C3 applied | Runner has separate `tc9` and `tc10` records | Runner still groups as `tc9_10` |
| TC-2 verification | All 5 languages PASS (precision ≥0.90, recall ≥0.85) | Any language FAILs |
| TC-6 verification | All 4 sub-tests PASS | Any sub-test FAILs |
| No implementation drift | `git diff --name-only` shows ONLY `.plans/phase-2-d1-d3-evaluation-plan.md` | Any other file modified |
| No D4 leakage | No D4-related changes | Any D4 file touched or referenced as actionable |
| No forbidden files touched | `tools/skills_sync.py`, `tests/tools/test_skills_sync.py`, all of `skills/`, `scripts/code-scan/*.py`, `tests/code_scan/` untouched | Any forbidden file modified |
| Other TCs unaffected | TC-1 pass, TC-11 pass (111/111), guardrails pass | Any regression |

---

## Artifacts Produced

| Artifact | Path / Location | Content |
|----------|----------------|---------|
| Corrected evaluation plan | `.plans/phase-2-d1-d3-evaluation-plan.md` | Updated on branch |
| TC-2 verification output | Stdout of Step 4 | Per-language precision/recall; all must PASS |
| TC-6 verification output | Stdout of Step 5 | Sub-test verdicts; all must PASS |
| TC-11 verification | Stdout of Step 6 | 111/111 pass confirmation |
| Guardrails verification | Stdout of Step 6 | TC-9 + TC-10 pass confirmation |
| Git diff | `git diff` output | Three-document-only changes (C1, C2, C3) |
| Commit | On `eval-plan-corrections` branch | Single commit with full correction set |

---

## Rollback Plan

All corrections target a single file (`.plans/phase-2-d1-d3-evaluation-plan.md`) on a dedicated branch.

```bash
# If correction causes problems, discard branch:
git checkout docs/ua-flywheel-phase1-phase2-plan
git branch -D eval-plan-corrections

# Or reset the plan to pre-correction state:
git checkout 3c63af2e0 -- .plans/phase-2-d1-d3-evaluation-plan.md
```

No data loss risk. No infrastructure or state mutation. Zero side effects from rollback.

---

## Risks and Benefits

| Factor | Assessment |
|--------|-----------|
| **Risk: Low** | Single-file documentation changes on a feature branch. No prod impact. |
| **Benefit: High** | Corrects false FAILs in evaluation. After correction, the plan accurately reflects implementation behavior. Denominator mismatch between 10 runner records and 11 planned tests is eliminated. |
| **Risk of overreach** | Mitigated by forbidden-files list and pre-commit diff review. |
| **Risk of regression** | Mitigated by running TC-1, TC-11, and guardrails after corrections. |
| **Benefit of runner split** | Runner denominator (11) now matches planned test count (11). Prevents confusion in evaluation summaries. |

---

## Guardrails Summary

1. **No D4:** No file in `skills/software-development/requesting-code-review/`, no D4-related changes. D4 bead remains deferred.
2. **No implementation:** Zero changes to `.py`, `.js`, `.rs`, `.go` files. Only `.plans/phase-2-d1-d3-evaluation-plan.md` is modified during correction execution.
3. **Forbidden files:** `tools/skills_sync.py`, `tests/tools/test_skills_sync.py` are explicitly excluded.
4. **No push/merge:** Push and merge require separate explicit JC approval. A local correction commit is allowed only after JC approves execution and verification passes.
5. **Branch discipline:** Prefer an `eval-plan-corrections` branch from `docs/ua-flywheel-phase1-phase2-plan`; execute directly on the phase branch only if JC explicitly directs it.

---

## Roles and Responsibilities

| Role | Responsibility |
|------|---------------|
| **Hermes (coder)** | Apply corrections C1, C2, C3 exactly as specified after JC approval. No deviations. Verify via test commands. Commit locally only after verification passes. |
| **Hermes (reviewer)** | After correction execution, review `git diff` output. Confirm exactly three correction classes. Confirm no forbidden files touched. Confirm TC-2, TC-6, TC-11, and guardrails all PASS. |
| **JC** | Approve execution or request changes. Push/merge remain separate approval gates. |

---

> **STATUS: PREPARED-NOT-EXECUTED**
> This bead is ready for execution upon JC approval.
> Do not execute until explicitly directed.
> D4 remains deferred and out of scope.
