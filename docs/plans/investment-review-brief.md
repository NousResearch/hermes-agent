# Investment Review Brief Helper Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a local-first helper script that turns Joe's personal investment position JSON/YAML/CSV files into a concise review brief with risk, stale-review, and missing-thesis flags.

**Architecture:** Create one standalone script under `scripts/` with pure parsing/scoring helpers plus a tiny CLI. Keep data local, deterministic, and reversible. Add focused tests under `tests/scripts/` using temp fixture files.

**Tech Stack:** Python standard library + optional PyYAML when YAML inputs are used; pytest via `scripts/run_tests.sh` or documented direct-pytest fallback if wrapper dependencies are incomplete.

---

## Task 1: Write failing tests for portfolio loading and review brief behavior

**Objective:** Specify the desired script behavior before implementation.

**Files:**
- Create: `tests/scripts/test_investment_review_brief.py`
- Create later: `scripts/investment_review_brief.py`

**Step 1: Write failing tests**

Cover:
- JSON input produces a Traditional Chinese brief with summary, risks, missing thesis, and overdue review sections.
- `--silent-if-clear` prints exact `[SILENT]` when no positions need attention.
- CSV input works without third-party dependencies.
- Invalid / empty input returns a useful non-zero CLI error.

**Step 2: Run test to verify failure**

Run: `python -m pytest tests/scripts/test_investment_review_brief.py -q -o 'addopts='`
Expected: FAIL because `scripts/investment_review_brief.py` does not exist.

## Task 2: Implement minimal parser, analysis, renderer, and CLI

**Objective:** Make the tests pass with the smallest useful helper.

**Files:**
- Create: `scripts/investment_review_brief.py`

**Step 1: Implement pure helpers**

- `load_positions(path)` for JSON/YAML/CSV.
- `analyze_positions(positions, as_of, review_after_days, concentration_threshold)`.
- `render_brief(analysis, silent_if_clear)`.

**Step 2: Implement CLI**

Arguments:
- positional `path`
- `--as-of YYYY-MM-DD`
- `--review-after-days N`
- `--concentration-threshold FLOAT`
- `--silent-if-clear`

**Step 3: Run focused tests**

Run: `python -m pytest tests/scripts/test_investment_review_brief.py -q -o 'addopts='`
Expected: PASS.

## Task 3: Smoke test CLI and finalize

**Objective:** Verify user-facing CLI behavior and prepare PR.

**Files:**
- Modify: none beyond script/tests.

**Step 1: Run CLI smoke**

Run a temp JSON fixture through `python scripts/investment_review_brief.py ...` and inspect output.

**Step 2: Run git diff/status**

Ensure only intended files changed.

**Step 3: Commit and open PR**

Commit message: `feat: add investment review brief helper`.
