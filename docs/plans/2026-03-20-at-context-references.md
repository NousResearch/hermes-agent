# @ Context References Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `@file:`, `@folder:`, `@diff`, `@staged`, `@git:N`, and `@url:` message references that expand into scoped context before the LLM sees the message, with token budgeting, CLI completion, gateway support, and tests.

**Architecture:** Implement a reusable preprocessing module that parses and expands `@` references against the current working directory, estimates injected token cost, and returns a rewritten user message plus structured metadata. Wire that module into the CLI input path and gateway message path, and extend the existing prompt_toolkit completer to support `@file:` and `@folder:` path completion without regressing slash-command completion.

**Tech Stack:** Python, prompt_toolkit, pytest, git subprocesses, Hermes auxiliary web extraction, existing model metadata/token estimation helpers.

---

### Task 1: Add failing parser and expander tests

**Files:**
- Create: `tests/test_context_references.py`

**Step 1: Write the failing test**

Add focused tests for:
- parsing multiple references in one message
- ignoring emails and bare `@username`
- expanding `@file:path` and `@file:path:start-end`
- expanding `@folder:path/`
- expanding `@diff`, `@staged`, and `@git:N`
- rejecting binary or missing files with inline warnings
- enforcing soft/hard token budgets

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_context_references.py -q`

Expected: FAIL because the module does not exist yet.

**Step 3: Write minimal implementation**

Create a new reusable module, likely `agent/context_references.py`, with:
- a parser for typed references only
- expansion helpers for file, folder, git, and url references
- rough token accounting using `agent.model_metadata.estimate_tokens_rough`
- a result object containing rewritten message, token totals, warnings, and a display summary

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_context_references.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_context_references.py agent/context_references.py
git commit -m "feat: add @ context reference parser and expander"
```

### Task 2: Add failing CLI completion tests

**Files:**
- Modify: `tests/hermes_cli/test_commands.py`
- Modify: `hermes_cli/commands.py`

**Step 1: Write the failing test**

Add tests covering:
- `@file:` path completion
- `@folder:` directory-only completion
- keeping slash-command completion unchanged
- continuing existing path completion for plain path-like words

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/hermes_cli/test_commands.py -q`

Expected: FAIL on the new `@file:` / `@folder:` completion assertions.

**Step 3: Write minimal implementation**

Extend `SlashCommandCompleter` in `hermes_cli/commands.py` to:
- detect `@file:` and `@folder:` under the cursor
- reuse a path-completion helper with optional directory-only filtering
- preserve slash-command behavior and existing non-slash path completion
- use `rg --files`-style semantics where practical, but avoid depending on shell calls in the completer

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/hermes_cli/test_commands.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/hermes_cli/test_commands.py hermes_cli/commands.py
git commit -m "feat: add @file and @folder completion"
```

### Task 3: Add failing CLI preprocessing tests

**Files:**
- Create: `tests/test_cli_context_references.py`
- Modify: `cli.py`

**Step 1: Write the failing test**

Add tests that verify the CLI:
- expands `@` references before `self.conversation_history.append(...)`
- preserves the user-visible original prompt logging
- surfaces warnings for over-budget or invalid references
- passes the expanded text, not the raw marker string, into `run_conversation`

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_cli_context_references.py -q`

Expected: FAIL because CLI preprocessing is not wired yet.

**Step 3: Write minimal implementation**

In `cli.py`:
- call the shared preprocessor after paste expansion and before `self.chat(...)`
- or, if cleaner, call it inside `HermesCLI.chat()` before appending to history
- print concise context-injection status and warnings
- keep voice-prefix behavior operating on the expanded message

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/test_cli_context_references.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_cli_context_references.py cli.py
git commit -m "feat: expand @ references in cli messages"
```

### Task 4: Add failing gateway preprocessing tests

**Files:**
- Create: `tests/gateway/test_context_references.py`
- Modify: `gateway/run.py`

**Step 1: Write the failing test**

Add tests that verify gateway text messages:
- expand `@file:` and git references before `_run_agent`
- preserve normal command routing
- include warnings for invalid references
- skip expansion safely when not in a git repo or when url extraction fails

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_context_references.py -q`

Expected: FAIL because the gateway path is still raw.

**Step 3: Write minimal implementation**

In `gateway/run.py`:
- preprocess text after existing transcription/document-note enrichment and before `_run_agent`
- resolve paths from the messaging CWD / terminal CWD
- thread any warning/status text into the outgoing conversation in a way that informs the model and the user without mutating prior history

**Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/gateway/test_context_references.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/gateway/test_context_references.py gateway/run.py
git commit -m "feat: expand @ references in gateway messages"
```

### Task 5: Add docs and final integration verification

**Files:**
- Modify: `website/docs/user-guide/cli.md`
- Modify: `website/docs/user-guide/features/skins.md` (only if command docs need cross-link cleanup; otherwise skip)
- Create or Modify: `website/docs/user-guide/features/context-references.md`
- Modify: `website/sidebars.ts`

**Step 1: Write the failing test**

If docs tests exist for sidebar/doc links, add or update them. Otherwise skip the explicit failing-doc-test step and verify with targeted checks.

**Step 2: Run targeted verification**

Run:
- `source .venv/bin/activate && python -m pytest tests/test_context_references.py tests/test_cli_context_references.py tests/hermes_cli/test_commands.py tests/gateway/test_context_references.py -q`
- `source .venv/bin/activate && python -m pytest tests/test_cli_init.py tests/gateway/ -q`

Expected: PASS

**Step 3: Write minimal documentation**

Document:
- supported reference forms
- budget behavior
- completion support in the CLI
- gateway limitations vs CLI

**Step 4: Run broader verification**

Run: `source .venv/bin/activate && python -m pytest tests/ -q`

Expected: PASS

**Step 5: Commit**

```bash
git add website/docs/user-guide/cli.md website/docs/user-guide/features/context-references.md website/sidebars.ts
git commit -m "docs: document @ context references"
```
