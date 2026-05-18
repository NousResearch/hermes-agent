---
name: karpeslop
description: Detect AI-generated code patterns in TypeScript/JavaScript.
version: 1.0.0
author: Daniel King (CodeDeficient)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-quality, linter, static-analysis, code-review]
    related_skills: [requesting-code-review, test-driven-development]
---

# KarpeSlop Skill

Runs KarpeSlop against TypeScript and JavaScript projects to detect patterns
specific to AI-generated code. Catches hallucinated imports, `any` type abuse,
overconfident comments, and vibe-coded anti-patterns that eslint and tsc miss.

This skill does not replace `requesting-code-review`. It targets AI-specific
code quality issues.

## When to Use

- User says "check for slop", "scan for AI code", "detect slop", or "/slop"
- After AI-assisted code generation, before committing
- After completing a task with TS/JS file edits

Skip for: non-TS/JS projects, documentation-only changes, or when user says
"skip slop check".

## Prerequisites

- Node.js 20+ and npm/npx on PATH. If missing, install via the Hermes
  installer (`--ensure node`) or your package manager.

No install needed for KarpeSlop itself. The package downloads on first
`npx` invocation.

## How to Run

```bash
npx karpeslop@latest --quiet
```

For CI-grade blocking on critical issues:

```bash
npx karpeslop@latest --quiet --strict
```

## Quick Reference

| Flag | Purpose |
|------|---------|
| `--quiet, -q` | Scan core app dirs only |
| `--strict, -s` | Exit code 2 on critical issues |
| `--version, -v` | Show version |

| Exit code | Meaning |
|-----------|---------|
| 0 | No issues |
| 1 | Issues found (warnings) |
| 2 | Critical issues with `--strict` |

Output is written to `./ai-slop-report.json`.

## Procedure

### 1. Check project language

Use `search_files` to verify `.ts` or `.js` files exist. Skip if not.

### 2. Run the scan

Run as described in `## How to Run`. Use `--strict` when the user wants CI-grade
blocking on critical issues.

### 3. Parse the report

Read `./ai-slop-report.json` with `read_file`. The structure:

```json
{
  "summary": {
    "totalFilesScanned": 15,
    "totalIssues": 8,
    "slopIndex": { "total": 42 },
    "byAxis": {
      "informationUtility": 2,
      "informationQuality": 1,
      "style": 5
    },
    "bySeverity": { "critical": 1, "high": 3, "medium": 2, "low": 2 }
  },
  "issues": [{
    "type": "hallucinated_react_import",
    "file": "src/pages/Home.tsx",
    "line": 5,
    "message": "Hallucinated React import",
    "severity": "critical",
    "fix": "Import from correct package: next/router",
    "learnMore": "https://nextjs.org/docs/api-reference/next/router"
  }],
  "byCategory": {
    "informationUtility": [],
    "informationQuality": [],
    "style": []
  }
}
```

### 4. Summarize for the user

Group by axis, then severity. Report format:

```
KarpeSlop scan: N files, N issues.

Noise: N — debug logs, redundant comments
Lies: N — hallucinated imports, assumptions
Soul: N — overconfident comments, hedging, vibe coding

Slop Index: N/100
```

### 5. Handle critical findings

If `--strict` returned exit code 2, report each critical issue with its `fix`
and `learnMore` fields. Do not proceed with commit.

### 6. Offer fixes for high/medium findings

Present the `fix` suggestions. Do not apply automatically.

## Pitfalls

1. **TS/JS only.** Running on Python, Rust, or Go produces an empty scan.
   Skip gracefully.

2. **First run downloads the tool.** `npx karpeslop@latest` fetches the
   package on first invocation. Takes a few seconds.

3. **Large repos may time out.** `--quiet` limits scanning to core app
   directories. Full scans on repos with thousands of files can take 30+
   seconds.

4. **Axis 3 patterns are low-confidence.** Overconfident comments and
   hedging language rely on regex matching. Medium/low findings on this
   axis should not block commits.

5. **Report is overwritten on each run.** `ai-slop-report.json` is
   ephemeral. Copy it to preserve results.

## Verification

Confirm the tool is reachable:

```bash
npx karpeslop@latest --version
```

Run a quick scan:

```bash
npx karpeslop@latest --quiet --strict
cat ai-slop-report.json | python -c "import json,sys; r=json.load(sys.stdin); print(f'{r[\"summary\"][\"totalIssues\"]} issues')"
```
