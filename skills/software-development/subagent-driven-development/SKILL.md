---
name: subagent-driven-development
description: "Execute plans via delegate_task subagents (2-stage review)."
version: 1.1.0
author: Hermes Agent (adapted from obra/superpowers)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [delegation, subagent, implementation, workflow, parallel]
    related_skills: [writing-plans, requesting-code-review, test-driven-development]
---

# Subagent-Driven Development

## Overview

Execute implementation plans by dispatching fresh subagents per task with systematic two-stage review.

**Core principle:** Fresh subagent per task + two-stage review (spec then quality) = high quality, fast iteration.

## When to Use

Use this skill when:
- You have an implementation plan (from writing-plans skill or user requirements)
- Tasks are mostly independent
- Quality and spec compliance are important
- You want automated review between tasks

**vs. manual execution:**
- Fresh context per task (no confusion from accumulated state)
- Automated review process catches issues early
- Consistent quality checks across all tasks
- Subagents can ask questions before starting work

## The Process

### 1. Read and Parse Plan

Read the plan file. Extract ALL tasks with their full text and context upfront. Create a todo list:

```python
# Read the plan
read_file("docs/plans/feature-plan.md")

# Create todo list with all tasks
todo([
    {"id": "task-1", "content": "Create User model with email field", "status": "pending"},
    {"id": "task-2", "content": "Add password hashing utility", "status": "pending"},
    {"id": "task-3", "content": "Create login endpoint", "status": "pending"},
])
```

**Key:** Read the plan ONCE. Extract everything. Don't make subagents read the plan file — provide the full task text directly in context.

### 2. Per-Task Workflow

For EACH task in the plan:

#### Step 1: Dispatch Implementer Subagent

Use `delegate_task` with complete context:

```python
delegate_task(
    goal="Implement Task 1: Create User model with email and password_hash fields",
    context="""
    TASK FROM PLAN:
    - Create: src/models/user.py
    - Add User class with email (str) and password_hash (str) fields
    - Use bcrypt for password hashing
    - Include __repr__ for debugging

    FOLLOW TDD:
    1. Write failing test in tests/models/test_user.py
    2. Run: pytest tests/models/test_user.py -v (verify FAIL)
    3. Write minimal implementation
    4. Run: pytest tests/models/test_user.py -v (verify PASS)
    5. Run: pytest tests/ -q (verify no regressions)
    6. Commit: git add -A && git commit -m "feat: add User model with password hashing"

    PROJECT CONTEXT:
    - Python 3.11, Flask app in src/app.py
    - Existing models in src/models/
    - Tests use pytest, run from project root
    - bcrypt already in requirements.txt
    """,
    toolsets=['terminal', 'file']
)
```

#### Step 2: Dispatch Spec Compliance Reviewer

After the implementer completes, verify against the original spec:

```python
delegate_task(
    goal="Review if implementation matches the spec from the plan",
    context="""
    ORIGINAL TASK SPEC:
    - Create src/models/user.py with User class
    - Fields: email (str), password_hash (str)
    - Use bcrypt for password hashing
    - Include __repr__

    CHECK:
    - [ ] All requirements from spec implemented?
    - [ ] File paths match spec?
    - [ ] Function signatures match spec?
    - [ ] Behavior matches expected?
    - [ ] Nothing extra added (no scope creep)?

    OUTPUT: PASS or list of specific spec gaps to fix.
    """,
    toolsets=['file']
)
```

**If spec issues found:** Fix gaps, then re-run spec review. Continue only when spec-compliant.

#### Step 3: Dispatch Code Quality Reviewer

After spec compliance passes:

```python
delegate_task(
    goal="Review code quality for Task 1 implementation",
    context="""
    FILES TO REVIEW:
    - src/models/user.py
    - tests/models/test_user.py

    CHECK:
    - [ ] Follows project conventions and style?
    - [ ] Proper error handling?
    - [ ] Clear variable/function names?
    - [ ] Adequate test coverage?
    - [ ] No obvious bugs or missed edge cases?
    - [ ] No security issues?

    OUTPUT FORMAT:
    - Critical Issues: [must fix before proceeding]
    - Important Issues: [should fix]
    - Minor Issues: [optional]
    - Verdict: APPROVED or REQUEST_CHANGES
    """,
    toolsets=['file']
)
```

**If quality issues found:** Fix issues, re-review. Continue only when approved.

#### Step 4: Mark Complete

```python
todo([{"id": "task-1", "content": "Create User model with email field", "status": "completed"}], merge=True)
```

### 3. Final Review

After ALL tasks are complete, dispatch a final integration reviewer:

```python
delegate_task(
    goal="Review the entire implementation for consistency and integration issues",
    context="""
    All tasks from the plan are complete. Review the full implementation:
    - Do all components work together?
    - Any inconsistencies between tasks?
    - All tests passing?
    - Ready for merge?
    """,
    toolsets=['terminal', 'file']
)
```

### 4. Verify and Commit

```bash
# Run full test suite
pytest tests/ -q

# Review all changes
git diff --stat

# Final commit if needed
git add -A && git commit -m "feat: complete [feature name] implementation"
```

## Task Granularity

**Each task = 2-5 minutes of focused work.**

**Too big:**
- "Implement user authentication system"

**Right size:**
- "Create User model with email and password fields"
- "Add password hashing function"
- "Create login endpoint"
- "Add JWT token generation"
- "Create registration endpoint"

## Pitfall: Subagent Timeout on Interactive CLI Commands

Subagents (via `delegate_task`) running interactive CLI tools like `npx shadcn@latest init`, `npm create`, or `npx prisma init` frequently **time out** because:
1. The tool prompts for user input (style choice, color, path, etc.)
2. The subagent cannot use `clarify` to ask the user
3. Even with `--yes` or `--defaults` flags, some tools still prompt
4. The subagent spins for 6+ minutes then gets interrupted

**Signs this is happening:** `delegate_task` returns with `status: "interrupted"`, empty summary, duration 300+ seconds, and the tool_trace shows many terminal calls but no file writes.

**Workaround:** For tasks requiring interactive CLI setup:
1. **Do the CLI init yourself** in the orchestrator session (you can handle prompts)
2. **Delegate the file editing/wiring** to the subagent after init is done
3. Or: run the CLI in background with `terminal(background=true)` and use `process(action="submit")` to answer prompts

**Better pattern for shadcn specifically:**
```bash
# In orchestrator session — do init directly
npx shadcn@latest init --yes
npx shadcn@latest add button card dialog --yes

# Then delegate the integration work
delegate_task(goal="Wire shadcn components into App.tsx...")
```

## Pitfall: Subagent Timeout with Partial Progress

When a subagent times out (600s limit), it often completed 80%+ of the work before being killed. **Don't assume zero progress.** Check what was done before re-delegating or redoing.

**Recovery pattern:**
1. Check filesystem for files the subagent was supposed to create
2. Verify content completeness (line count, grep for key patterns)
3. Run tsc/build to see if what exists is valid
4. Finish the remaining 20% directly or with a targeted follow-up delegation

```bash
# After subagent timeout — check what was done
test -f src/components/VideoPlayer.tsx && echo "EXISTS ($(wc -l < src/components/VideoPlayer.tsx) lines)"
grep -c "VideoPlayer" src/components/LearnSection.tsx  # wiring done?
npx tsc --noEmit 2>&1 | head -5  # valid so far?
```

**Key insight:** A timed-out subagent that installed packages and created 3 of 4 files is vastly better than starting from scratch. Respect partial progress.

## Red Flags — Never Do These

- Start implementation without a plan
- Skip reviews (spec compliance OR code quality)
- Proceed with unfixed critical/important issues
- Dispatch multiple implementation subagents for tasks that touch the same files
- Make subagent read the plan file (provide full text in context instead)
- Skip scene-setting context (subagent needs to understand where the task fits)
- Ignore subagent questions (answer before letting them proceed)
- Accept "close enough" on spec compliance
- Skip review loops (reviewer found issues → implementer fixes → review again)
- Let implementer self-review replace actual review (both are needed)
- **Start code quality review before spec compliance is PASS** (wrong order)
- Move to next task while either review has open issues

## Handling Issues

### If Subagent Asks Questions

- Answer clearly and completely
- Provide additional context if needed
- Don't rush them into implementation

### If Reviewer Finds Issues

- Implementer subagent (or a new one) fixes them
- Reviewer reviews again
- Repeat until approved
- Don't skip the re-review

### If Subagent Fails a Task

- Dispatch a new fix subagent with specific instructions about what went wrong
- Don't try to fix manually in the controller session (context pollution)

### Sibling Subagents Modify the Same File

When two subagents write to the same file (even if working on different tasks), the orchestrator's view of that file may be stale:

**Symptom:** Your `patch` call fails with "Found N matches for old_string" or "Could not find a match for old_string" — even though you can see the text on screen.

**Root cause:** A sibling subagent modified the file after your last `read_file`, but before your `patch`. The file on disk is different from what you read.

**Fix:** Always `read_file` again before `patch` when you know a sibling subagent has touched the file. Use more surrounding context in `old_string` to make the match unique:
```python
# BAD: single line, easy to match multiple places
patch("...", old_string="def foo():", new_string="def bar():")

# GOOD: multi-line unique context
patch("...", old_string="class MyClass:\n    def __init__(self):\n        self.value = 0\n    def foo(self):", new_string="...")
```

**Prevention:** Assign non-overlapping file ownership per subagent when possible. If files must overlap, coordinate in the orchestrator session (do the overlapping edits yourself, or dispatch them sequentially).

### Subagent Creates Files the Orchestrator Didn't Plan

When a subagent creates a file (e.g., `processors.py`) that another subagent depends on, subsequent subagents may fail with import errors. Track file creation as a dependency:

## Efficiency Notes

**Why fresh subagent per task:**
- Prevents context pollution from accumulated state
- Each subagent gets clean, focused context
- No confusion from prior tasks' code or reasoning

**Why two-stage review:**
- Spec review catches under/over-building early
- Quality review ensures the implementation is well-built
- Catches issues before they compound across tasks

**Cost trade-off:**
- More subagent invocations (implementer + 2 reviewers per task)
- But catches issues early (cheaper than debugging compounded problems later)

## Integration with Other Skills

### With writing-plans

This skill EXECUTES plans created by the writing-plans skill:
1. User requirements → writing-plans → implementation plan
2. Implementation plan → subagent-driven-development → working code

### With test-driven-development

Implementer subagents should follow TDD:
1. Write failing test first
2. Implement minimal code
3. Verify test passes
4. Commit

Include TDD instructions in every implementer context.

### With requesting-code-review

The two-stage review process IS the code review. For final integration review, use the requesting-code-review skill's review dimensions.

### With systematic-debugging

If a subagent encounters bugs during implementation:
1. Follow systematic-debugging process
2. Find root cause before fixing
3. Write regression test
4. Resume implementation

## Example Workflow

```
[Read plan: docs/plans/auth-feature.md]
[Create todo list with 5 tasks]

--- Task 1: Create User model ---
[Dispatch implementer subagent]
  Implementer: "Should email be unique?"
  You: "Yes, email must be unique"
  Implementer: Implemented, 3/3 tests passing, committed.

[Dispatch spec reviewer]
  Spec reviewer: ✅ PASS — all requirements met

[Dispatch quality reviewer]
  Quality reviewer: ✅ APPROVED — clean code, good tests

[Mark Task 1 complete]

--- Task 2: Password hashing ---
[Dispatch implementer subagent]
  Implementer: No questions, implemented, 5/5 tests passing.

[Dispatch spec reviewer]
  Spec reviewer: ❌ Missing: password strength validation (spec says "min 8 chars")

[Implementer fixes]
  Implementer: Added validation, 7/7 tests passing.

[Dispatch spec reviewer again]
  Spec reviewer: ✅ PASS

[Dispatch quality reviewer]
  Quality reviewer: Important: Magic number 8, extract to constant
  Implementer: Extracted MIN_PASSWORD_LENGTH constant
  Quality reviewer: ✅ APPROVED

[Mark Task 2 complete]

... (continue for all tasks)

[After all tasks: dispatch final integration reviewer]
[Run full test suite: all passing]
[Done!]
```

## Handling Large Content Creation Tasks

When delegating content-heavy tasks (documentation, code generation >500 lines, educational material), output length is a real risk. Subagents can time out mid-generation. Mitigate this:

**1. Estimate output size before delegating.**
A rough rule: each ~100 tokens of subagent context budget yields ~80-100 words of output. A task requiring 3000+ words of output needs either:
- A model with a large output window, OR
- Chunked delegation (break the task into parts)

**2. Use file output as the contract, not the return value.**
Always tell the subagent to `write_file` the output directly to disk:
```
Write the complete content to /path/to/output.md using write_file.
Do NOT return the content in your response — write it directly to disk.
Your response should only contain: (1) what you did, (2) file location, (3) line/word count.
```
This way, even if the subagent times out mid-generation, partial progress is saved to disk.

**3. Handle partial/timed-out output.**
If a subagent times out:
- Read the file it was writing to check how far it got.
- If incomplete, either: (a) have a new subagent complete from where it left off, or (b) write the remaining section yourself and merge.
- Check the file's line count — if it's suspiciously short (e.g., should be 2000 lines but is 400), it's incomplete.

**4. For multi-part documents, parallelize at the file level.**
Instead of one subagent for the whole document:
- Dispatch one subagent per Part/thread (Part 1, Part 2, Part 3.1, Part 3.2, etc.)
- Each writes to its own file
- Orchestrator reads outputs and combines
- Each subagent's risk is bounded to its own section

**5. Set a "completeness contract" in the subagent goal.**
Tell the subagent: "At the end of your response, state the exact line count of the file you wrote. If you cannot complete, write what you have to disk anyway and report the partial line count."

### Example: Large Document Creation

```
User request: Create 30-page educational guide across 3 parts

Plan:
- Subagent A → writes Part 1 to /tmp/part1.md (file output contract)
- Subagent B → writes Part 2 to /tmp/part2.md (file output contract)
- Subagent C → writes Part 3 to /tmp/part3.md (file output contract)
- Orchestrator reads all three files, combines into final document

Each subagent goal includes:
"Write the complete Part 1 content to /tmp/part1.md using write_file.
Do NOT return the content in your response. Report only: file written, line count.
If you cannot complete, write what you have and report partial line count."
```

## Document Splitting Pitfalls (Orchestrator-Side)

When the orchestrator reads a document created by subagents and splits it into parts for further processing (PDF generation, section review, etc.), common gotchas:

### Anchor String Collision (TOC vs Content)

**Problem:** Header strings like `# PART 1:` appear in BOTH the Table of Contents and the content body. A naive `content.find('# PART 1:')` returns the TOC entry first, giving you the TOC instead of the actual content.

**Symptoms:** Part 1 PDF is empty or contains only the TOC. The content section starts at the second occurrence.

**Fix:** Use `re.MULTILINE` to anchor at line starts, or find the Nth occurrence:

```python
import re

# WRONG: finds first occurrence (TOC entry)
matches = list(re.finditer(r'# PART \d+:', content))
# matches[0] → TOC entry, matches[1] → actual content

# RIGHT: anchor at line start (^ = beginning of line in MULTILINE mode)
matches = list(re.finditer(r'^# PART \d+:', content, re.MULTILINE))
# matches[0] → first content line (line 67), matches[1] → second (line 1864), etc.

# Alternative: skip past TOC by starting search AFTER the TOC
toc_end = content.find('---', content.find('## Table of Contents')) + 5
part1_start = content.find('\n# PART 1:', toc_end)
```

**Rule:** When splitting a structured document, always verify which occurrence of your anchor string you found. If the extracted section has 0 code blocks but the document has 45, you got the wrong occurrence.

### Verify Extraction Before Processing

After splitting, always verify you got the right section:

```python
part1 = content[matches[0].start():matches[1].start()]
print(f"Part 1: {len(part1)} chars, code blocks: {part1.count('```python')}")
# If code blocks == 0 but document has 45 → wrong occurrence, re-extract
```

## Remember

```
Fresh subagent per task
Two-stage review every time
Spec compliance FIRST
Code quality SECOND
Never skip reviews
Catch issues early
File output for large content — never return in response
Parallelize at file level for multi-part documents
Completeness contract: report line count at end
```

**Quality is not an accident. It's the result of systematic process.**

## Further reading (load when relevant)

When the orchestration involves significant context usage, long review loops, or complex validation checkpoints, load these references for the specific discipline:

- **`references/context-budget-discipline.md`** — Four-tier context degradation model (PEAK / GOOD / DEGRADING / POOR), read-depth rules that scale with context window size, and early warning signs of silent degradation. Load when a run will clearly consume significant context (multi-phase plans, many subagents, large artifacts).
- **`references/gates-taxonomy.md`** — The four canonical gate types (Pre-flight, Revision, Escalation, Abort) with behavior, recovery, and examples. Load when designing or reviewing any workflow that has validation checkpoints — use the vocabulary explicitly so each gate has defined entry, failure behavior, and resumption rules.

Both references adapted from gsd-build/get-shit-done (MIT © 2025 Lex Christopherson).

- **`references/multi-phase-orchestrated-upgrade.md`** — Sequential upgrade workflow with per-phase verification gates, direct-vs-delegated execution strategy, and checkpoint stops. Use when the user provides a numbered phase plan.
- **`references/vite7-jsx-node-modules-pitfall.md`** — Vite 7 build failure when npm packages ship JSX in `.js` production bundles (e.g., vidstack). Detection, workarounds, and pre-adoption check. Load when a build fails with `Expression expected` pointing to node_modules.

- **`references/crlf-file-writing-in-wsl.md`** — CRLF corruption pitfalls when writing files from WSL to Windows-mounted paths. Covers safe methods (Python binary mode, PowerShell, printf+sed), recovery, and TypeScript-specific issues (`_` prefix limitation, `ignoreDeprecations` version mismatch). **Load whenever working on a Windows/CRLF project from WSL.**

- **`references/markdown-to-pdf-weasyprint.md`** — Pipeline for converting markdown to PDF using WeasyPrint. Covers markdown→HTML→PDF conversion, WeasyPrint setup (pip bootstrap, font requirements), CSS patterns for code blocks and syntax highlighting, and the anchor collision pitfall when splitting large documents into parts.
