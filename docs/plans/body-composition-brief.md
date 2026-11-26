# Body Composition Brief Helper Implementation Plan

> **For Hermes:** Implement directly with strict TDD in this worktree.

**Goal:** Add a local-first script that turns Joe's body-composition logs into a concise Traditional Chinese morning brief or exact `[SILENT]` when there is nothing actionable.

**Architecture:** Create a standalone Python script under `scripts/` with pure parsing/scoring/rendering functions and a small CLI wrapper. Keep inputs local JSON/YAML/CSV, avoid network or private-data expansion, and make output deterministic for cron usage.

**Tech Stack:** Python standard library, optional PyYAML when installed, pytest tests under `tests/scripts/`.

---

### Task 1: Define expected parsing and silence behavior

**Objective:** Tests lock in JSON/YAML/CSV loading and `[SILENT]` output for empty inputs.

**Files:**
- Create: `tests/scripts/test_body_composition_brief.py`
- Create later: `scripts/body_composition_brief.py`

**Steps:**
1. Write failing tests for loading JSON records and empty exact `[SILENT]` rendering.
2. Run focused tests and verify failure because the script does not exist.
3. Implement minimal parser and renderer to pass.

### Task 2: Add trend/status behavior

**Objective:** Tests verify latest weight/body-fat summary and simple direction calculations.

**Files:**
- Modify: `tests/scripts/test_body_composition_brief.py`
- Modify: `scripts/body_composition_brief.py`

**Steps:**
1. Add failing tests for latest metric selection, body-fat target distance, and trend labels.
2. Run focused tests and verify failures.
3. Implement minimal summarization logic.

### Task 3: Add Joe-style Traditional Chinese report and CLI

**Objective:** CLI prints TL;DR, facts, hypotheses, and measurable next actions without sending anything.

**Files:**
- Modify: `tests/scripts/test_body_composition_brief.py`
- Modify: `scripts/body_composition_brief.py`

**Steps:**
1. Add failing tests for rendered headings and CLI smoke behavior.
2. Run focused tests and verify failures.
3. Implement CLI args: `--input`, `--days`, `--silent-if-empty`.
4. Verify focused tests and script smoke checks.
