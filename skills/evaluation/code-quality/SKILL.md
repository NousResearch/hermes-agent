---
name: code-quality
description: >-
  Comprehensive code quality analysis using real linters, AST-based complexity
  scoring, dependency auditing, and structured report generation. Use when
  evaluating code health, onboarding to a new codebase, or before refactoring.
version: 2.0.0
author: vominh1919
license: MIT
metadata:
  hermes:
    tags: [code-quality, analysis, linting, complexity, security, testing, refactoring]
    related_skills:
      - requesting-code-review
      - github-code-review
      - test-driven-development
      - systematic-debugging
---

# Code Quality Analysis

Evaluate codebases using real tools (ruff, mypy, radon, bandit, pytest, vulture)
and structured agent analysis. Produces a quality report with actionable findings.

**When to use:**
- User says "check code quality", "analyze this codebase", "how healthy is this project"
- Before starting a refactoring session on an unfamiliar codebase
- After onboarding to a new project — get a baseline health check
- When reviewing a PR and wanting deeper metrics than a quick diff scan

**This skill vs requesting-code-review:** `requesting-code-review` verifies YOUR
changes before committing. This skill analyzes existing code health holistically.

**This skill vs github-code-review:** `github-code-review` reviews OTHER people's
PRs with inline comments. This skill produces a full codebase quality report.

---

## Step 1 — Detect project language and tooling

```bash
# Language detection
ls *.py setup.py pyproject.toml 2>/dev/null && LANG=python
ls *.js *.ts package.json 2>/dev/null && LANG=javascript
ls *.go go.mod 2>/dev/null && LANG=golang
ls *.rs Cargo.toml 2>/dev/null && LANG=rust

# Count files and lines
find . -name "*.py" -not -path "*/.git/*" -not -path "*/node_modules/*" -not -path "*/__pycache__/*" | wc -l
find . -name "*.py" -not -path "*/.git/*" -not -path "*/node_modules/*" -not -path "*/__pycache__/*" -exec cat {} \; | wc -l

# Detect installed tools
which ruff && echo "ruff: available" || echo "ruff: not installed"
which mypy && echo "mypy: available" || echo "mypy: not installed"
which radon && echo "radon: available" || echo "radon: not installed"
which bandit && echo "bandit: available" || echo "bandit: not installed"
which pytest && echo "pytest: available" || echo "pytest: not installed"
which vulture && echo "vulture: available" || echo "vulture: not installed"
```

If a tool is not installed, skip that check silently — do NOT attempt to install.
Missing tools reduce report granularity but should not block analysis.

---

## Step 2 — Run automated analysis

Run each available tool and capture output. Execute in the project root.

### Python Projects

```bash
# Linting (ruff)
ruff check . --output-format=json 2>/dev/null > /tmp/cq-ruff.json || true
ruff check . 2>/dev/null | tail -30 || true

# Type checking (mypy)
mypy . --ignore-missing-imports --no-error-summary 2>/dev/null > /tmp/cq-mypy.txt || true

# Complexity (radon) — cyclomatic complexity per function/class
radon cc . -s -a -nc 2>/dev/null > /tmp/cq-radon-cc.txt || true
radon mi . -s -nc 2>/dev/null > /tmp/cq-radon-mi.txt || true

# Security (bandit)
bandit -r . -f json -q 2>/dev/null > /tmp/cq-bandit.json || true
bandit -r . -q 2>/dev/null | tail -20 || true

# Dead code (vulture)
vulture . --min-confidence 80 2>/dev/null > /tmp/cq-vulture.txt || true

# Test coverage
pytest --co -q 2>/dev/null | tail -5 || true
pytest --tb=no -q 2>/dev/null | tail -10 || true
```

### JavaScript/TypeScript Projects

```bash
npx eslint . 2>/dev/null | tail -30 || true
npx tsc --noEmit 2>/dev/null | tail -20 || true
```

### Rust Projects

```bash
cargo clippy -- -D warnings 2>&1 | tail -30 || true
```

### Go Projects

```bash
go vet ./... 2>&1 | tail -20 || true
staticcheck ./... 2>&1 | tail -20 || true
```

---

## Step 3 — Structural analysis with AST

Use `read_file` and `search_files` to analyze code structure. Focus on:

### 3a. Architecture health
```bash
# Module structure — flat vs deeply nested
find . -name "*.py" -not -path "*/.git/*" | head -50
find . -name "*.py" -not -path "*/.git/*" | awk -F/ '{print NF-1, $0}' | sort -rn | head -10

# File sizes — largest files often need refactoring
find . -name "*.py" -not -path "*/.git/*" -exec wc -l {} \; | sort -rn | head -10

# Import coupling — files with most imports (high coupling)
grep -rn "^import\|^from" --include="*.py" . | awk -F: '{print $1}' | sort | uniq -c | sort -rn | head -10
```

### 3b. Docstring coverage
For Python, check how many public functions/classes have docstrings:
```bash
# Count functions with vs without docstrings
python3 -c "
import ast, os, sys

with_doc = 0
without_doc = 0
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'node_modules', '.venv')]
    for f in files:
        if f.endswith('.py'):
            try:
                tree = ast.parse(open(os.path.join(root, f)).read())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if ast.get_docstring(node):
                            with_doc += 1
                        else:
                            without_doc += 1
            except: pass
total = with_doc + without_doc
print(f'Documented: {with_doc}/{total} ({100*with_doc/max(total,1):.0f}%)')
"
```

### 3c. Test coverage ratio
```bash
# Ratio of test files to source files
SRC=$(find . -name "*.py" -not -path "*/test*" -not -path "*/.git/*" -not -path "*__pycache__*" | wc -l)
TEST=$(find . -path "*/test*" -name "*.py" -not -path "*/.git/*" | wc -l)
echo "Source files: $SRC, Test files: $TEST (ratio: 1:$(( SRC > 0 ? TEST * 100 / SRC : 0 ))%)"
```

### 3d. TODO/FIXME/HACK debt
```bash
grep -rn "TODO\|FIXME\|HACK\|XXX\|WORKAROUND" --include="*.py" --include="*.js" --include="*.ts" --include="*.go" --include="*.rs" . | grep -v ".git/" | wc -l
```

---

## Step 4 — Dependency health check

```bash
# Python — check for outdated or vulnerable packages
pip list --outdated 2>/dev/null | head -10 || true

# If pip-audit is available
pip-audit 2>/dev/null | head -20 || true

# Node — check for vulnerabilities
npm audit 2>/dev/null | tail -10 || true
```

Check for `requirements.txt`, `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`:
- Are dependencies pinned to specific versions?
- Is there a lock file (poetry.lock, package-lock.json, Cargo.lock)?

---

## Step 5 — Generate quality report

Combine all findings into a structured report. Present to the user in this format:

```
## Code Quality Report: <project-name>

### Overview
- Language: Python
- Files: 42 files, 3,847 lines
- Tests: 18 test files (1:2.3 ratio)
- Docstring coverage: 67%
- TODO debt: 12 items

### Scores (0-100)
| Metric          | Score | Grade |
|-----------------|-------|-------|
| Linting         |  85   |  B    |
| Type Safety     |  72   |  C+   |
| Complexity      |  78   |  B-   |
| Security        |  95   |  A    |
| Documentation   |  67   |  D+   |
| Test Coverage   |  45   |  F    |
| **Overall**     | **74**| **C** |

### 🔴 Critical (fix immediately)
- src/auth.py:45 — eval() with user input (security)
- No test files found

### ⚠️ Warnings
- 5 functions with cyclomatic complexity > 15
- 3 files > 500 lines (consider splitting)
- 12 TODO/FIXME items

### 💡 Suggestions
- Add type hints to 23 functions
- Increase docstring coverage from 67% to 80%+
- Pin dependency versions in requirements.txt

### ✅ What looks good
- Clean import structure
- No hardcoded secrets detected
- Good naming conventions
```

---

## Step 6 — Suggest improvements

Based on the report, suggest specific actionable improvements prioritized by impact:

1. **Security issues first** — any bandit/eval/exec findings
2. **High complexity** — functions with CC > 10 that are hard to maintain
3. **Missing tests** — untested critical paths
4. **Documentation** — public APIs without docstrings
5. **Style** — lint issues (these are lowest priority)

For each suggestion, estimate effort (quick fix / moderate / significant).

---

## Pitfalls

- **No project context** — analyze the codebase from the root, not individual files
- **Tool not installed** — skip silently, note in report as "not available"
- **Large codebase** — for repos with 1000+ files, sample representative modules instead of scanning everything
- **False positives** — bandit and vulture often flag test code or intentional patterns; note this in the report
- **Mixed languages** — if project has Python + JS + Go, run tools for each language separately
- **Test fixtures may import dangerous patterns** — security scans should exclude test directories (add `-x ./tests` to bandit)

---

## Scoring Guide

| Score | Grade | Meaning |
|-------|-------|---------|
| 90-100 | A | Excellent — production-ready |
| 80-89  | B | Good — minor improvements needed |
| 70-79  | C | Fair — some issues to address |
| 60-69  | D | Poor — significant problems |
| < 60   | F | Failing — major overhaul needed |

**Per-metric scoring:**
- **Linting:** 100 - (issues × 2), min 0
- **Type Safety:** (typed functions / total functions) × 100
- **Complexity:** 100 - (functions with CC > 10 × 10), min 0
- **Security:** 100 - (findings × 25), min 0
- **Documentation:** (documented / total) × 100
- **Test Coverage:** (test files / source files) × 50 + pass_rate × 50
