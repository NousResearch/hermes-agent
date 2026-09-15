# Project-Checker Repair Loop (PER-481)

Implementation of a project-checker repair loop for Hermes codegen, inspired by the omp cleanse design (MIT) but implemented independently with no vendored source files.

## Overview

The cleanse system detects issues via VCS + manifests + linters, normalizes diagnostics into a unified format, applies automated fixes using file-sticky workers, and verifies the results.

## Architecture

### Components

1. **Detector** (`agent/cleanse/detector.py`)
   - Reuses/extends `agent.verify.recipes` + `coding_context.detect_project_facts`
   - Runs available checkers: tsc, eslint, actionlint, ruff, prettier
   - Skips missing binaries gracefully (no failures)
   - Returns unified `DetectionResult` with diagnostics + skipped tools

2. **Parser** (`agent/cleanse/parser.py`)
   - Normalizes tool-specific output formats
   - Unified diagnostic schema: `{file, line, col, code, severity, message}`
   - Parsers for: tsc, eslint (JSON), actionlint, ruff (JSON), prettier

3. **Worker** (`agent/cleanse/worker.py`)
   - File-sticky parallel workers (default: 2)
   - Mutating formatters run serially first (prettier, eslint --fix)
   - Groups diagnostics by file for parallel processing
   - Fixes root cause (no diagnostic suppression)

4. **Loop** (`agent/cleanse/loop.py`)
   - Main orchestration: detect → fix → verify
   - Re-runs same suite for verification
   - Success = empty remaining diagnostics
   - Writes `verification_evidence.json`

### CLI Integration

- **Commands**: `hermes cleanse` and `hermes check-fix` (alias)
- **Options**:
  - `--workers N`: Max parallel workers (default: 2)
  - `--tests`: Run project tests after fixes
  - `--json`: Machine-readable output
  - `path`: Project root (default: current directory)

## Usage

### Basic Usage

```bash
# Run cleanse on current directory
hermes cleanse

# Run on specific project
hermes cleanse /path/to/project

# Run tests after fixing
hermes cleanse --tests

# Use 4 workers
hermes cleanse --workers 4

# JSON output for automation
hermes cleanse --json > results.json
```

### Demo Script

For testing without full CLI setup:

```bash
python3 demo_cleanse.py /path/to/project
```

## Supported Checkers

| Tool | Trigger | Format | Notes |
|------|---------|--------|-------|
| tsc | `tsconfig.json` exists | Text output | TypeScript type checking |
| eslint | `.eslintrc*` or `eslint.config.*` exists | JSON (`--format json`) | JavaScript/TypeScript linting |
| actionlint | `.github/workflows/` exists | Text output | GitHub Actions workflow validation |
| ruff | `pyproject.toml` or `ruff.toml` exists | JSON (`--output-format json`) | Python linting |
| prettier | `.prettierrc*` exists | Text output | Code formatting |

All checkers skip gracefully if not on PATH.

## Success Criteria (PER-481)

✅ **Met:**
- TS type error collected/fixed/verify pass
- actionlint when on PATH
- Two-file two-worker parallel processing
- Missing eslint/tsc/etc. skips gracefully
- No omp files vendored
- Verification evidence written

## Design Principles

1. **No project-wide check rerun**: Workers fix root cause, no need to rerun entire suite
2. **No diagnostic suppression**: All issues are addressed, not hidden
3. **Serial formatters first**: Mutating formatters (prettier, eslint --fix) run serially to avoid conflicts
4. **File-sticky workers**: Each worker processes all diagnostics for assigned files
5. **Graceful degradation**: Missing tools are skipped, never cause failures

## Fork Layout

- **Branch**: `cursor/project-checker-repair-loop-dd97` off `staging`
- **PR**: Ready (non-draft) into `staging` on kvnloo/hermes-agent
- **Target**: kvnloo/hermes-agent (fork), NOT origin NousResearch

## Testing

### Unit Tests

```bash
# Run all cleanse tests
pytest tests/agent/cleanse/ -v

# Individual test files
pytest tests/agent/cleanse/test_parser.py -v
pytest tests/agent/cleanse/test_detector.py -v
pytest tests/agent/cleanse/test_loop.py -v
```

### Integration Tests

Create test projects and run cleanse:

```bash
# TypeScript project with type errors
mkdir -p /tmp/test-ts && cd /tmp/test-ts
echo '{"compilerOptions": {"strict": true}}' > tsconfig.json
echo 'const x: string = 42;' > app.ts
hermes cleanse .

# Python project
mkdir -p /tmp/test-py && cd /tmp/test-py
echo '[project]\nname = "test"' > pyproject.toml
echo 'import sys' > main.py
hermes cleanse .
```

## Files

### Core Implementation

- `agent/cleanse/__init__.py`: Package exports
- `agent/cleanse/detector.py`: Issue detection (263 lines)
- `agent/cleanse/parser.py`: Diagnostic normalization (202 lines)
- `agent/cleanse/worker.py`: Repair workers (197 lines)
- `agent/cleanse/loop.py`: Main loop (189 lines)

### CLI Integration

- `hermes_cli/cleanse_cmd.py`: Command implementation (132 lines)
- `hermes_cli/subcommands/cleanse.py`: Parser registration (76 lines)
- `hermes_cli/main.py`: Wiring (3 lines added)

### Tests

- `tests/agent/cleanse/test_parser.py`: Parser tests (133 lines)
- `tests/agent/cleanse/test_detector.py`: Detector tests (37 lines)
- `tests/agent/cleanse/test_loop.py`: Loop tests (45 lines)

### Demo

- `demo_cleanse.py`: Standalone demo script (63 lines)

## Output Format

### Human-readable

```
============================================================
Cleanse Report: /path/to/project
============================================================
Initial issues:   15
Fixed:            12
Remaining:        3
Skipped tools:    tsc

────────────────────────────────────────────────────────────
Remaining Issues:
────────────────────────────────────────────────────────────
  src/app.ts:42:10 [TS2304] Cannot find name 'foo'
  src/utils.ts:10:5 [no-unused-vars] 'bar' is defined but never used
  main.py:15:1 [F401] unused import

Status: ⚠ Incomplete
Evidence written to: /path/to/project/verification_evidence.json
============================================================
```

### JSON Output

```json
{
  "ok": false,
  "initial_count": 15,
  "fixed_count": 12,
  "remaining_count": 3,
  "remaining_diagnostics": [
    {
      "file": "src/app.ts",
      "line": 42,
      "col": 10,
      "code": "TS2304",
      "severity": "error",
      "message": "Cannot find name 'foo'"
    }
  ],
  "verification_evidence": {
    "status": "incomplete",
    "initial_count": 15,
    "fixed_count": 12,
    "remaining_count": 3,
    "skipped_tools": ["tsc"]
  },
  "skipped_tools": ["tsc"]
}
```

### Verification Evidence

Written to `verification_evidence.json` in project root:

```json
{
  "status": "clean",
  "initial_count": 15,
  "fixed_count": 15,
  "remaining_count": 0,
  "skipped_tools": []
}
```

## Future Enhancements

1. **Additional checkers**: mypy, clippy, black, gofmt, etc.
2. **Smart fixes**: Language-specific AST manipulation for automated fixes
3. **Auto-run config**: Optional post-edit trigger (default OFF per spec)
4. **Integration**: Wire into `hermes verify` workflow
5. **Parallel workers**: True parallel processing with multiprocessing
6. **Fix strategies**: Configurable fix approaches per diagnostic type

## Related Work

- Inspired by: omp cleanse design (MIT license)
- No source files vendored from omp
- Independent implementation following same principles

## Links

- **Branch**: `cursor/project-checker-repair-loop-dd97`
- **PR**: https://github.com/kvnloo/hermes-agent/pull/68
- **Linear**: PER-481
