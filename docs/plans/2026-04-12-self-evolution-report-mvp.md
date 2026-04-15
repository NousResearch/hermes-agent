# Self-Evolution Report MVP Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a first self-evolution CLI report that mines recent sessions and optional trajectory JSONL files for repeated failure/correction patterns, then emits concrete recommendations and candidate prompt deltas.

**Architecture:** Create a small analysis module (`agent/evolution.py`) that ingests normalized session/trajectory records, extracts repeated patterns with simple heuristics, and renders a Markdown report. Wire it into the argparse CLI as `hermes evolve report` so it can analyze recent SQLite-backed sessions via `SessionDB` and optional external trajectory files.

**Tech Stack:** Python 3.11, argparse, dataclasses, pathlib/json, existing `SessionDB`, pytest.

---

### Task 1: Add failing analyzer tests

**Objective:** Lock in the core behavior before implementation.

**Files:**
- Create: `tests/agent/test_evolution.py`

**Step 1: Write failing tests**
- Test that repeated tool failures across sessions are counted and surfaced.
- Test that repeated user corrections are counted and surfaced.
- Test that Markdown rendering includes recommendations and candidate prompt deltas.
- Test that trajectory JSONL loading tolerates blank lines and malformed JSON rows.

**Step 2: Run test to verify failure**

Run: `pytest tests/agent/test_evolution.py -v`
Expected: FAIL because `agent.evolution` does not exist yet.

### Task 2: Implement the analyzer module

**Objective:** Add the minimal production code needed to satisfy the tests.

**Files:**
- Create: `agent/evolution.py`

**Step 1: Implement normalized loaders**
- Add helpers to load trajectory JSONL entries.
- Add helpers to normalize session/message records.

**Step 2: Implement heuristic pattern extraction**
- Detect repeated tool failures from tool messages and failure-like content.
- Detect repeated user corrections from user messages.
- Detect repeated assistant blockage/apology patterns.

**Step 3: Implement report rendering**
- Produce deterministic Markdown with sections for findings, recommendations, and candidate prompt deltas.

**Step 4: Run tests**

Run: `pytest tests/agent/test_evolution.py -v`
Expected: PASS.

### Task 3: Add failing CLI tests

**Objective:** Define the public interface for the new command before wiring it in.

**Files:**
- Create: `tests/hermes_cli/test_evolve_command.py`

**Step 1: Write failing tests**
- Test that `hermes evolve report --help` exposes the new command and key flags.
- Test that `cmd_evolve()` writes the rendered report to stdout by default.
- Test that `cmd_evolve()` writes to `--output` when provided.

**Step 2: Run test to verify failure**

Run: `pytest tests/hermes_cli/test_evolve_command.py -v`
Expected: FAIL because the command does not exist yet.

### Task 4: Implement CLI wiring

**Objective:** Expose the analyzer through the main Hermes CLI.

**Files:**
- Modify: `hermes_cli/main.py`

**Step 1: Add `evolve` parser**
- Add top-level `evolve` command with `report` subcommand.
- Support flags: `--source`, `--limit`, `--trajectory`, `--min-count`, `--output`.

**Step 2: Add command handler**
- Load recent sessions with `SessionDB.list_sessions_rich()` + `get_messages()`.
- Optionally load trajectory files.
- Render Markdown and print or write it.
- Exit cleanly on errors with a concise message.

**Step 3: Run CLI tests**

Run: `pytest tests/hermes_cli/test_evolve_command.py -v`
Expected: PASS.

### Task 5: Regression verification

**Objective:** Verify the new feature and nearby behavior together.

**Files:**
- No code changes unless tests expose regressions.

**Step 1: Run targeted suite**

Run: `pytest tests/agent/test_evolution.py tests/hermes_cli/test_evolve_command.py -v`
Expected: PASS.

**Step 2: Smoke test the CLI manually**

Run: `python -m hermes_cli.main evolve report --help`
Expected: help text shows the new subcommand and flags.

**Step 3: Optional real-data smoke test**

Run: `python -m hermes_cli.main evolve report --limit 5`
Expected: emits a Markdown report or a concise ‘no findings yet’ report.
