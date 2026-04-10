# Dead Code Audit Spec — hermes-agent

## Goal

One-time, maximum-impact dead code removal. Three tools (vulture, coverage.py, ast-grep) run independently, then their results are intersected to produce confidence-tiered findings. An Opus agent confirms ambiguous cases. Output: a Markdown report + per-tier git patches ready to apply.

---

## 1. Scope

### In scope

| Layer                      | Modules                                                                                                                                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Packages                   | `agent/`, `tools/`, `hermes_cli/`, `gateway/`, `cron/`                                                                                                                                                                                            |
| Top-level modules          | `run_agent.py`, `model_tools.py`, `toolsets.py`, `batch_runner.py`, `trajectory_compressor.py`, `toolset_distributions.py`, `cli.py`, `hermes_constants.py`, `hermes_state.py`, `hermes_time.py`, `hermes_logging.py`, `utils.py`, `mcp_serve.py` |
| Tests (coverage data only) | `tests/` — executes during coverage to generate line-hit data, but test imports do NOT count as reachability proof                                                                                                                                |

### Out of scope

| Excluded           | Reason                                   |
| ------------------ | ---------------------------------------- |
| `environments/`    | Experimental RL/benchmark code           |
| `mini-swe-agent/`  | Separate project                         |
| `skills/`          | Dynamically loaded user-facing skills    |
| `optional-skills/` | User-facing plugins, loaded by name      |
| `plugins/`         | Dynamically registered, exclude entirely |
| `acp_adapter/`     | Separate adapter, excluded per user      |
| `rl_cli.py`        | RL-specific, excluded per user           |
| `tinker-atropos/`  | Separate package (own egg-info)          |
| `website/`         | Documentation site, not Python runtime   |

### Entrypoints (roots for reachability analysis)

1. `hermes_cli.main:main` — `hermes` CLI
2. `run_agent:main` — `hermes-agent` CLI
3. `acp_adapter.entry:main` — `hermes-acp` CLI (out of scope but its imports into in-scope modules count as callers)

Additionally, discover whether `batch_runner.py`, `trajectory_compressor.py`, and `mcp_serve.py` have `if __name__ == "__main__"` blocks or are imported by in-scope production code. If they have main blocks, treat them as additional entrypoints.

### Reachability model

**Production entrypoints are the only roots.** A symbol is alive if and only if it is reachable from the production entrypoints listed above (directly or via dynamic dispatch maps). Tests are untrusted code that happens to generate coverage data as a side effect:

- **Test imports are not reachability proof.** `from agent.foo import bar` in a test file does NOT make `bar` alive. Tests may import dead code — that's expected and those test imports should also be cleaned up.
- **Coverage data from tests is trustworthy.** If a test exercises a code path, the coverage data reflects what actually executes, not what's imported. A test that imports `bar` but never calls it won't add coverage to `bar`'s lines. Coverage remains a reliable execution oracle.
- **Stale tests are a cleanup target.** If removing dead production code breaks test imports, those tests were testing dead code and should be removed too (see Phase 4 output).

---

## 2. Architecture

### Pipeline overview

```
Phase 1: Data Collection (parallel, agent-orchestrated)
├── Agent A: vulture scan → vulture_results.json
├── Agent B: coverage.py report → coverage_results.json
└── Agent C: dispatch map extraction → dispatch_roots.json

Phase 2: Intersection (deterministic script)
├── Parse vulture output → set of (file, line, symbol, type)
├── Parse coverage uncovered lines → set of (file, line_range)
├── Load dispatch roots → set of known-reachable symbols
├── Intersect → tiered findings

Phase 3: ast-grep Confirmation (agent-orchestrated)
├── For each finding: ast-grep import-aware search for callers (production only)
├── Opus agent reviews ambiguous cases
└── Initial classification (T1/T2/T3/T-cond)

Phase 3b: Deep Verification (Opus agent, full-repo)
├── For each T2 finding with ast_grep_confirmed=True:
│   ├── Full-repo search (including excluded dirs: plugins/, acp_adapter/, environments/)
│   ├── Check Fire CLI method exposure
│   ├── Check __init__.py re-exports
│   └── Check cross-scope production callers
├── Verified-dead T2 → promoted to T1
├── Found-alive T2 → demoted to T3
└── Updated classification

Phase 4: Output Generation (deterministic script)
├── Markdown report with tiered findings
├── Per-tier .patch files
└── Updated .dead-code-allowlist
```

### Confidence tiers

| Tier                            | Criteria                                                                                                                                                                                    | Action                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| **T1 — Auto-delete**            | All 3 tools agree, OR vulture + ast-grep agree and Opus deep verification confirms zero callers across the entire repo (including excluded dirs like plugins/, acp_adapter/, environments/) | Apply patch directly                     |
| **T2 — Review**                 | Any 2 of 3 tools agree but NOT yet verified by Opus deep pass                                                                                                                               | Human reviews before applying            |
| **T3 — Informational**          | Only 1 tool flags it                                                                                                                                                                        | Logged for awareness, no patch generated |
| **T-cond — Conditionally dead** | Code behind feature flags (`try: import X except ImportError`, `if HAS_*:`)                                                                                                                 | Flagged separately, never auto-deleted   |

---

## 3. Phase 1: Data Collection

### 3a. Vulture scan (Agent A)

**Tool:** `vulture`

**Command:**

```bash
vulture agent/ tools/ hermes_cli/ gateway/ cron/ \
  run_agent.py model_tools.py toolsets.py batch_runner.py \
  trajectory_compressor.py toolset_distributions.py cli.py \
  hermes_constants.py hermes_state.py hermes_time.py \
  hermes_logging.py utils.py mcp_serve.py \
  --min-confidence 60 \
  --sort-by-size \
  --whitelist .dead-code-allowlist
```

**Notes:**

- `tests/` is **NOT** included. Test imports must not count as callers — a test importing a dead function would suppress the finding. Vulture scans production code only.
- The `--min-confidence 60` threshold catches most dead code while reducing noise
- `--sort-by-size` prioritizes larger dead code blocks (higher impact deletions)
- The `.dead-code-allowlist` is passed directly to vulture via `--whitelist` — vulture parses its own whitelist format natively (Python files with dummy usages). We do NOT parse the allowlist ourselves.

**Output format:** Parse vulture's stdout into structured JSON:

```json
[
  {
    "file": "agent/foo.py",
    "line": 42,
    "symbol": "unused_function",
    "type": "function", // function | class | method | variable | attribute | import
    "confidence": 80,
    "message": "unused function 'unused_function' (80% confidence)"
  }
]
```

### 3b. Coverage report (Agent B)

**Tool:** `coverage.py`

**Prerequisites:**

1. Re-run coverage with integration tests included:

   ```bash
   python -m pytest --cov=agent --cov=tools --cov=hermes_cli \
     --cov=gateway --cov=cron \
     --cov-report=json:coverage_report.json \
     --cov-report=term-missing
   ```

   (User will provide API keys for integration test services)

2. If integration tests fail or aren't available, fall back to the existing `.coverage` file:
   ```bash
   coverage json -o coverage_report.json
   ```

**Output format:** coverage.py's JSON report natively provides:

```json
{
  "files": {
    "agent/foo.py": {
      "executed_lines": [1, 2, 5, 6, ...],
      "missing_lines": [42, 43, 44, 45],
      "excluded_lines": []
    }
  }
}
```

Transform to normalized format:

```json
[
  {
    "file": "agent/foo.py",
    "uncovered_ranges": [
      [42, 45],
      [80, 82]
    ],
    "coverage_pct": 72.5
  }
]
```

### 3c. Dispatch map extraction (Agent C)

**Tool:** Python runtime introspection

**Method:** Import `toolsets`, `model_tools`, and `toolset_distributions` in the repo's own venv and dump their dispatch maps.

```python
#!/usr/bin/env python3
"""Extract runtime dispatch maps to identify dynamically-reachable symbols."""
import json
import importlib
import sys

def extract_dispatch_maps():
    roots = set()

    for module_name in ["toolsets", "model_tools", "toolset_distributions"]:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue

        # Walk all module-level dicts looking for string→module/class mappings
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, dict):
                for key, value in attr.items():
                    if isinstance(value, str) and ("." in value or "/" in value):
                        roots.add(value)
                    elif isinstance(value, type):
                        roots.add(f"{value.__module__}.{value.__qualname__}")
                    elif callable(value):
                        roots.add(f"{value.__module__}.{value.__qualname__}")

    return sorted(roots)

if __name__ == "__main__":
    json.dump(extract_dispatch_maps(), sys.stdout, indent=2)
```

Also extract the gateway dispatcher routing to determine which adapter modules are reachable:

- Find the gateway dispatcher/router (likely in `gateway/__init__.py` or `gateway/runner.py`)
- Extract the adapter class/module mappings
- Add reachable adapter modules to the root set

**Output:** `dispatch_roots.json` — a list of dotted module/symbol paths that are dynamically reachable.

---

## 4. Phase 2: Intersection (Deterministic Script)

### `dead_code_intersect.py`

This is the core deterministic script that can be re-run for reproducibility.

**Input files:**

- `vulture_results.json` (from Phase 1a — allowlist already applied by vulture via `--whitelist`)
- `coverage_report.json` (from Phase 1b, coverage.py native JSON)
- `dispatch_roots.json` (from Phase 1c)

Note: the `.dead-code-allowlist` is consumed directly by vulture at scan time (Phase 1a). The intersection script does NOT parse it — vulture's own whitelist handling is correct and handles the Python file format natively.

**Algorithm:**

```python
def intersect(vulture_results, coverage_data, dispatch_roots, allowlist):
    findings = []

    for v in vulture_results:
        # Skip if in allowlist
        if is_allowlisted(v, allowlist):
            continue

        # Skip if in dispatch roots (dynamically reachable)
        if is_dispatch_reachable(v, dispatch_roots):
            continue

        # Skip findings within test files
        if v["file"].startswith("tests/"):
            continue

        # Check coverage
        coverage_agrees = is_uncovered(v["file"], v["line"], coverage_data)

        # Score
        v["vulture_flags"] = True
        v["coverage_uncovered"] = coverage_agrees
        v["ast_grep_confirmed"] = None  # Filled in Phase 3

        findings.append(v)

    # Dead file candidates: modules with 0% coverage.
    # IMPORTANT: 0% coverage alone is NOT enough for T1. A file could be imported
    # and used in production paths that tests don't exercise. Dead files MUST be
    # confirmed by ast-grep (zero importers in production code) before reaching T1.
    # At this stage we flag them as candidates; Phase 3 does the confirmation.
    for file_path, file_cov in coverage_data["files"].items():
        if file_cov["coverage_pct"] == 0:
            findings.append({
                "file": file_path,
                "line": 0,
                "symbol": "<entire file>",
                "type": "module",
                "confidence": 60,  # Low until ast-grep confirms
                "vulture_flags": True,
                "coverage_uncovered": True,
                "ast_grep_confirmed": None  # MUST be True for T1
            })

    return findings
```

**Output:** `intersection_results.json` — findings annotated with which tools flagged them.

---

## 5. Phase 3: ast-grep Confirmation (Agent-Orchestrated)

### 5a. Import-aware symbol search

For each finding from Phase 2, run ast-grep to check whether the symbol has callers in **production code only**.

**Critical: ignore test matches.** Hits in `tests/` do NOT count as callers. A stale test importing dead code shouldn't save it — those tests are themselves dead and will be cleaned up.

**Strategy: Import-aware search (production code only)**

For a finding like `agent/foo.py:42 unused_function`:

1. **Direct call search:** Find all calls to `unused_function` in production code

   ```bash
   sg --pattern 'unused_function($$$)' --lang python | grep -v '^tests/'
   ```

2. **Import search:** Find all imports of the symbol in production code

   ```bash
   sg --pattern 'from agent.foo import $$$unused_function$$$' --lang python | grep -v '^tests/'
   sg --pattern 'import agent.foo' --lang python | grep -v '^tests/'
   ```

3. **String reference search:** Check if the symbol name appears as a string (dynamic dispatch)

   ```bash
   sg --pattern '"unused_function"' --lang python | grep -v '^tests/'
   sg --pattern "'unused_function'" --lang python | grep -v '^tests/'
   ```

4. **Attribute access search:** For methods, check if accessed on any object
   ```bash
   sg --pattern '$OBJ.unused_function' --lang python | grep -v '^tests/'
   ```

If ANY of these find a match in production code outside the defining file, the finding is downgraded (not confirmed as dead). Matches in `tests/` are recorded separately for the dead test code report (see Phase 4d).

**For dead file candidates** (type: `module`), the ast-grep check is especially critical:

- Search for `import <module>` and `from <module> import` across all production code
- A file with 0% coverage but production importers is NOT dead — it's just untested
- A file with 0% coverage AND zero production importers → confirmed dead (T1 eligible)

### 5b. Opus confirmation agent

For findings where ast-grep results are ambiguous (e.g., name collision — `send()` appears in 50 places), an Opus agent reviews the context:

**Agent prompt template:**

```
You are reviewing a dead code finding. Determine if this symbol is actually dead
from the perspective of PRODUCTION code paths.

Symbol: {symbol} ({type})
File: {file}:{line}
Vulture confidence: {confidence}%
Coverage: {"never executed" | "partially executed"}
ast-grep matches (production only): {list of locations in non-test code}
ast-grep matches (tests only): {list of locations in tests/ — these do NOT prove liveness}

Context (surrounding code):
{20 lines around the symbol definition}

IMPORTANT: Test imports do NOT make a symbol alive. Only production entrypoints
(hermes_cli.main:main, run_agent:main, acp_adapter.entry:main) and dynamic
dispatch from production code count as reachability proof.

Consider:
1. Is any PRODUCTION ast-grep match actually calling THIS symbol from THIS module, or is it a name collision?
2. Could this be called via getattr, __getattr__, or dynamic dispatch in production code?
3. Is this a dunder method, ABC abstract method, or protocol method that's called implicitly?
4. Is this behind a feature flag or optional dependency guard?
5. Is this a public API that external consumers might use (even if nothing in-repo calls it)?
6. If this is a dead file (type: module), does ANY production code import it?

Respond with:
- DEAD: Confirmed dead code, safe to remove
- ALIVE: Has production callers or is needed for other reasons
- CONDITIONAL: Behind a feature flag, alive in some configurations
- UNCERTAIN: Can't determine with confidence

If DEAD, also list any test files that import this symbol — those tests are
stale and should be cleaned up.
```

**Model:** Opus 4.6 (per user preference for thoroughness)

### 5c. Feature flag detection

Before classification, check if the symbol is guarded by:

- `try: import X except ImportError` blocks
- `if HAS_*:` / `if ENABLE_*:` conditionals
- `@requires(...)` decorators

Flagged symbols → T-cond tier, never auto-deleted.

ast-grep patterns for detection:

```bash
# try/except ImportError guard
sg --pattern 'try: $$$ import $$$ $$$ except ImportError: $$$' --lang python

# Feature flag conditionals
sg --pattern 'if HAS_$NAME: $$$' --lang python
sg --pattern 'if ENABLE_$NAME: $$$' --lang python
```

---

## 6. Phase 4: Output Generation

### 6a. Report (`dead_code_report.md`)

```markdown
# Dead Code Audit Report

Generated: {timestamp}
Scope: {list of packages/modules}

## Summary

- Total findings: N
- T1 (auto-delete): N files, N symbols, N lines removable
- T2 (review): N files, N symbols
- T3 (informational): N symbols
- T-cond (conditional): N symbols

## T1 — Auto-Delete (high confidence)

### Dead Files

| File               | Lines | Last modified | Reason                      |
| ------------------ | ----- | ------------- | --------------------------- |
| agent/old_thing.py | 150   | 2024-03-01    | Zero importers, 0% coverage |

### Dead Symbols

| File:Line       | Symbol      | Type     | Size (lines) |
| --------------- | ----------- | -------- | ------------ |
| agent/foo.py:42 | unused_func | function | 15           |

## T2 — Needs Review

{same format, with additional "Why review needed" column}

## T3 — Informational

{compact list}

## T-cond — Conditionally Dead

| File:Line         | Symbol           | Guard                  | Feature     |
| ----------------- | ---------------- | ---------------------- | ----------- |
| tools/voice.py:10 | setup_elevenlabs | try/except ImportError | tts-premium |
```

### 6b. Patch files

- `dead_code_t1.patch` — All T1 removals. Apply with `git apply dead_code_t1.patch`
- `dead_code_t2.patch` — All T2 removals. Review first, then apply.
- No patch for T3 or T-cond.

Patches are generated by:

1. For dead files: `git rm <file>`
2. For dead symbols: Remove the function/class/variable definition
3. For dead imports: Remove the import line
4. **Orphan import cleanup (critical):** When a symbol is removed from `foo.py`, any file that has `from foo import that_symbol` now has a broken import. The Phase 3 agent tracks these in the `orphan_imports` field. The patch MUST include removal of these orphaned import lines — otherwise applying the patch produces immediate ImportErrors.
5. **Dead test cleanup:** When dead production code is removed, test files that import the deleted symbols also break. These are tracked in the `test_importers` field. The T1 patch includes:
   - Removal of import lines in test files that reference deleted symbols
   - If removing the import makes the entire test file dead (no remaining test functions reference live code), the test file is deleted entirely

The patch generation agent must verify the patch is self-consistent: apply it to a worktree, run the test suite, and confirm no ImportErrors.

### 6c. Dead test code report

When production code is flagged as dead, the Phase 3 agent also collects test files that import those dead symbols. This produces a separate section in the report:

```markdown
## Dead Test Code

Tests that import dead production symbols. These tests were testing dead code
and should be removed alongside the production code they test.

### Tests broken by T1 removals (included in T1 patch)

| Test file                     | Imports deleted symbol               | Action                           |
| ----------------------------- | ------------------------------------ | -------------------------------- |
| tests/agent/test_old_thing.py | from agent.old_thing import OldClass | Delete entire file               |
| tests/tools/test_foo.py:5     | from tools.foo import unused_func    | Remove import + test_unused_func |

### Tests broken by T2 removals (included in T2 patch)

{same format}
```

This is a feature, not a bug — these tests were testing dead code and their breakage confirms the production code is truly dead.

### 6d. Allowlist update

After the audit, any false positives identified during review should be added to `.dead-code-allowlist` in vulture's native whitelist format:

```python
# .dead-code-allowlist
# Vulture whitelist — symbols that appear dead but are alive.
# Format: dummy usage statements that tell vulture "this is used."

from agent.models import SomeClass  # used by external consumers
SomeClass.some_method  # called via protocol

from tools.voice_mode import setup_voice  # called dynamically from config
```

---

## 7. Agent Orchestration

### Coordinator flow

```
Coordinator (main conversation)
│
├─ spawn Agent A (sonnet): Run vulture, parse output → vulture_results.json
├─ spawn Agent B (sonnet): Run coverage, parse output → coverage_results.json
├─ spawn Agent C (sonnet): Extract dispatch maps → dispatch_roots.json
│  (all three run in parallel)
│
├─ Wait for all three
│
├─ Run dead_code_intersect.py locally (deterministic)
│  → intersection_results.json
│
├─ For each batch of findings:
│  └─ spawn Agent D (opus): Run ast-grep checks + contextual review
│     → confirmed_results.json (initial T1/T2/T3 classification)
│
├─ spawn Agent E (opus): Deep verification of T2 findings
│  ├─ Full-repo search for cross-scope callers (plugins/, acp_adapter/, etc.)
│  ├─ Fire CLI exposure check, __init__.py re-exports, string dispatch
│  ├─ Verified-dead T2 → promoted to T1
│  └─ Found-alive T2 → demoted to T3
│     → final_results.json
│
├─ Run output generation locally (deterministic)
│  → dead_code_report.md
│  → dead_code_t1.patch (includes orphan import + dead test cleanup)
│  → dead_code_t2.patch (includes orphan import + dead test cleanup)
│  → .dead-code-allowlist (if new false positives found)
│
├─ Validate: apply T1 patch to worktree, run tests, confirm no ImportErrors
│
└─ Present report to user
```

### Agent specifications

| Agent             | Model      | Task                                                                                                                                      | Tools needed            |
| ----------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| A — Vulture       | Sonnet 4.6 | Run vulture, parse output, handle config issues                                                                                           | Bash, Write             |
| B — Coverage      | Sonnet 4.6 | Run/parse coverage, normalize to JSON                                                                                                     | Bash, Write, Read       |
| C — Dispatch      | Sonnet 4.6 | Extract dispatch maps at runtime, find gateway router                                                                                     | Bash, Write, Read, Grep |
| D — Confirmer     | Opus 4.6   | ast-grep searches, contextual dead code review (production dirs only)                                                                     | Bash, Read, Grep, Write |
| E — Deep Verifier | Opus 4.6   | Full-repo verification of T2 findings: cross-scope callers, Fire CLI, re-exports. Promotes verified-dead T2→T1, demotes found-alive T2→T3 | Bash, Read, Grep, Write |

### Error handling in agent orchestration

- If vulture or coverage isn't installed or fails: the agent should install it (`pip install vulture` / `pip install coverage`) and retry
- If dispatch map extraction fails (import error): fall back to static AST parsing of the dict literals in toolsets.py/model_tools.py
- If ast-grep isn't available: fall back to ripgrep-based symbol search (less precise but functional)
- Each agent writes its output to a well-known path; the coordinator reads it

---

## 8. Gotchas & Special Cases

### Dynamic dispatch patterns to watch for

1. **`getattr` / `importlib`** — Scan for `getattr(obj, "symbol_name")` and `importlib.import_module("module.path")`. Any symbol referenced this way is alive.

2. **`__init__.py` re-exports** — A symbol defined in `agent/foo.py` and re-exported in `agent/__init__.py` (`from .foo import bar`) looks dead in foo.py to vulture if nothing imports from foo directly. The re-export makes it alive.

3. **String-based class instantiation** — Common in config-driven code:

   ```python
   cls = globals()[class_name]  # or locals()
   obj = cls()
   ```

   Scan for `globals()[`, `locals()[`, and `getattr(sys.modules[`.

4. **Pydantic model fields** — Fields on Pydantic models are accessed via attribute access at runtime. Methods like `model_validate`, `model_dump` call validators/serializers implicitly. Don't flag Pydantic validator methods (`@field_validator`, `@model_validator`).

5. **CLI subcommand registration** — `hermes_cli/` likely uses `fire` (per pyproject.toml dependency). Fire discovers methods on a class or functions in a module by name. All public methods on a Fire-exposed class are reachable.

6. **Test fixtures** — Not applicable. Tests are excluded from the vulture scan entirely. Test code is only cleaned up as a consequence of removing dead production code it imported.

7. **Dunder methods** — `__repr__`, `__str__`, `__eq__`, `__hash__`, `__enter__`, `__exit__`, etc. are called implicitly. Never flag these.

8. **Abstract methods / Protocol methods** — Methods defined in ABCs or Protocols are implemented by subclasses. The base definition looks dead but isn't.

9. **Decorator-registered handlers** — Watch for patterns like `@app.route`, `@register`, `@handler` that register functions in a global registry without explicit import.

---

## 9. Deterministic Script Skeleton

The following script is the reproducible core. Agents handle the messy parts (running tools, handling errors), but this script does the deterministic intersection.

```python
#!/usr/bin/env python3
"""
dead_code_intersect.py — Intersect vulture + coverage + ast-grep results.

Usage:
    python dead_code_intersect.py \
        --vulture vulture_results.json \
        --coverage coverage_report.json \
        --dispatch dispatch_roots.json \
        --output intersection_results.json
"""
import argparse
import json
import sys


def load_vulture(path: str) -> list[dict]:
    """Load vulture results: list of {file, line, symbol, type, confidence}.

    Allowlist is already applied by vulture at scan time (--whitelist flag).
    We do NOT parse the allowlist here — vulture handles its own Python-file
    whitelist format natively and correctly.
    """
    with open(path) as f:
        return json.load(f)


def load_coverage(path: str) -> dict:
    """Load coverage.py JSON report → {file: {missing_lines: set}}."""
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for fpath, fdata in raw.get("files", {}).items():
        result[fpath] = {
            "missing": set(fdata.get("missing_lines", [])),
            "executed": set(fdata.get("executed_lines", [])),
        }
    return result


def load_dispatch_roots(path: str) -> set[str]:
    """Load dispatch roots: set of dotted module.symbol paths."""
    with open(path) as f:
        return set(json.load(f))


def is_uncovered(file: str, line: int, coverage: dict) -> bool:
    """Check if a specific line is in coverage's missing set."""
    for cov_file, cov_data in coverage.items():
        if cov_file.endswith(file) or file.endswith(cov_file):
            return line in cov_data["missing"]
    return False  # File not in coverage data → can't confirm


def intersect(vulture: list[dict], coverage: dict, dispatch_roots: set[str]) -> list[dict]:
    findings = []
    for v in vulture:
        # Vulture scans production code only (tests/ excluded from scan).
        # No need to filter test files here — they never appear in results.

        # Skip dispatch-reachable symbols
        if any(root.endswith(v["symbol"]) for root in dispatch_roots):
            continue

        coverage_agrees = is_uncovered(v["file"], v["line"], coverage)

        v["coverage_uncovered"] = coverage_agrees
        v["ast_grep_confirmed"] = None  # Phase 3 fills this
        v["test_importers"] = []        # Phase 3 fills: test files that import this symbol
        v["orphan_imports"] = []        # Phase 3 fills: production imports that become orphaned
        v["tier"] = None                # Assigned after Phase 3

        findings.append(v)

    return findings


def classify(findings: list[dict]) -> list[dict]:
    """Assign tiers based on tool agreement after ast-grep pass.

    For dead files (type: module), ast-grep confirmation is REQUIRED for T1.
    A file with 0% coverage might just be untested but used in production.
    """
    for f in findings:
        votes = sum([
            True,  # vulture always flags (that's how it got here)
            f["coverage_uncovered"],
            f.get("ast_grep_confirmed", False),
        ])

        if f.get("feature_guarded"):
            f["tier"] = "T-cond"
        elif f["type"] == "module" and not f.get("ast_grep_confirmed"):
            # Dead files MUST have ast-grep zero-importer confirmation.
            # 0% coverage alone is not enough — could be used but untested.
            f["tier"] = "T2"  # Force review even if coverage agrees
        elif votes == 3:
            f["tier"] = "T1"
        elif votes == 2:
            f["tier"] = "T2"
        else:
            f["tier"] = "T3"

    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vulture", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--dispatch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    vulture = load_vulture(args.vulture)
    coverage = load_coverage(args.coverage)
    dispatch_roots = load_dispatch_roots(args.dispatch)

    findings = intersect(vulture, coverage, dispatch_roots)
    # Note: ast_grep_confirmed, test_importers, and orphan_imports are filled
    # by the Phase 3 agent, then re-run classify() and output generation.

    with open(args.output, "w") as f:
        json.dump(findings, f, indent=2, default=str)

    print(f"Wrote {len(findings)} findings to {args.output}")
    print(f"  - coverage agrees: {sum(1 for f in findings if f['coverage_uncovered'])}")
    print(f"  - needs ast-grep: {len(findings)}")


if __name__ == "__main__":
    main()
```

---

## 10. Execution Plan

### Step 1: Setup

- Verify vulture, coverage.py, ast-grep (sg) are installed
- Verify repo venv has all deps (`pip install -e '.[all,dev]'`)

### Step 2: Data collection (parallel agents)

- Agent A: vulture scan → `vulture_results.json`
- Agent B: coverage run (with integration tests) → `coverage_report.json`
- Agent C: dispatch map extraction → `dispatch_roots.json`

### Step 3: Intersection

- Run `dead_code_intersect.py` → `intersection_results.json`

### Step 4: ast-grep confirmation (Opus agent D)

- For each finding, run import-aware ast-grep searches (production dirs only)
- Opus agent reviews ambiguous cases
- Update `intersection_results.json` with `ast_grep_confirmed` and `feature_guarded` fields
- Initial tier classification (T1/T2/T3/T-cond)

### Step 4b: Deep verification (Opus agent E)

- For each T2 finding with `ast_grep_confirmed=True` and `type != "module"`:
  - Full-repo search including excluded dirs (plugins/, acp_adapter/, environments/)
  - Check Fire CLI method exposure on classes passed to `fire.Fire()`
  - Check `__init__.py` re-exports
  - Check cross-scope production callers
- Verified-dead → promoted to T1 (`verified_dead: true`)
- Found-alive → demoted to T3 with note explaining what caller was found
- T2 modules (alive-but-untested files) remain T2

### Step 5: Classification

- Final tier counts after deep verification
- Generate report + patches

### Step 6: Review

- User reviews T1 patch (should be safe to apply)
- User reviews T2 findings with agent assistance
- T-cond findings documented for future cleanup

---

## 11. Success Criteria

- T1 patch applies cleanly and all tests pass after application (no ImportErrors, no test failures)
- Zero false positives in T1 tier (validated by test suite running in a worktree)
- Report covers both dead files and dead symbols
- Orphan imports cleaned up in every patch (no broken `from X import deleted_symbol` left behind)
- Dead test code removed alongside the production code it tested
- Feature-guarded code is never in T1
- Dispatch-reachable code is never flagged
- `__init__.py` re-exports are never flagged
- Dunder methods and Fire CLI methods are never flagged
- Dead files require ast-grep zero-importer confirmation before T1 (0% coverage alone is insufficient)
- Test imports never count as reachability proof — only production entrypoint reachability matters
