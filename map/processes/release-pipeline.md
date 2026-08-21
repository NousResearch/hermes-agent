---
id: release-pipeline
kind: process
universe: runtime
name: Release Pipeline
summary: >
  Validate, test, and release Hermes Agent: deterministic local test runner
  feeds CI lanes, which gate merges on `all-checks-pass`.
aliases: []
tags: [ci, tests, release]
shape: process
steps:
  - id: step.1
    summary: >
      Run `scripts/run_tests.sh` with hermetic env, per-file pytest isolation,
      and deterministic time/hash/baseline.
  - id: step.2
    summary: >
      Classify changed areas in CI with `detect-changes` and set boolean lane flags.
  - id: step.3
    summary: >
      Run gated sub-workflows for Python tests, OS tests, lint, JS tests,
      installer tests, docs, supply-chain, and review labels.
  - id: step.4
    summary: >
      Aggregate all required jobs in `all-checks-pass`, treating `skipped` as success
      and failing only on actual `failure` outcomes.
entrypoints: [step.1]
produces: [repo:.github/workflows/ci.yaml]
consumes: [repo:scripts/run_tests.sh, repo:.github/workflows/tests.yml, repo:.github/workflows/lint.yml]
---

# Release Pipeline

1. **Local test execution**: `scripts/run_tests.sh` locates a pytest-bearing venv, pre-compiles bytecode cache via `python -m compileall -q -j 0 -- $(git ls-files '*.py')`, then execs `scripts/run_tests_parallel.py` in a hermetic `env -i` with `TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0` (`scripts/run_tests.sh:152-183`).
2. **Change detection**: CI runs `.github/actions/detect-changes` once, producing lane flags: `python`, `python_prod`, `frontend`, `site`, `scan`, `deps`, `uv_lock`, `npm_lock`, `installer`, `docker_meta`, `mcp_catalog`, `ci_review` (`.github/workflows/ci.yaml:38-63`).
3. **Python test lane**: gated on `python == true`; calls `tests.yml` with `slice_count: 12` (`.github/workflows/ci.yaml:69-75`).
4. **OS-specific test lane**: gated on `python == true`; calls `tests-os.yml` for macOS/Windows-marked tests (`.github/workflows/ci.yaml:81-85`).
5. **Lint lane**: gated on `python == true`; calls `lint.yml` with event-name gating (`.github/workflows/ci.yaml:87-93`).
6. **Frontend lane**: gated on `frontend == true`; calls `js-tests.yml` (`.github/workflows/ci.yaml:95-99`).
7. **Installer lane**: Windows-only; gated on `installer == true`; calls `installer-tests.yml` (`.github/workflows/ci.yaml:104-106`).
8. **E2E desktop lane**: temporarily disabled via `false &&`; gated on `python_prod` or `frontend`; calls `e2e-desktop.yml` (`.github/workflows/ci.yaml:108-124`).
9. **Docs, history, contributor, lockfile, docker, supply-chain, review-label, OSV lanes**: each gated by its detect output; history-check and contributor-check only run on `pull_request`; review-labels run on PRs needing CI review, MCP catalog, or supply-chain critical findings (`.github/workflows/ci.yaml:126-196`).
10. **Gate**: `all-checks-pass` aggregates all required jobs with `if: always()`, treating `skipped` as success and only failing on `failure` (`.github/workflows/ci.yaml:207-220`).

## Human check

Confirm `run_tests.sh` still uses `env -i` with the documented allowlist, and that `ci.yaml` still uses `workflow_call` reusable sub-workflows rather than inline job definitions.

## Deterministic validation

```bash
grep -n "workflow_call" .github/workflows/tests.yml .github/workflows/lint.yml .github/workflows/tests-os.yml
grep -n "slice_count" .github/workflows/tests.yml
grep -n "env -i" scripts/run_tests.sh
grep -n "all-checks-pass:" .github/workflows/ci.yaml
```

Expected: `workflow_call` present in each sub-workflow, `slice_count: 12` in tests.yml, `env -i` in run_tests.sh, and `all-checks-pass` job in ci.yaml.
