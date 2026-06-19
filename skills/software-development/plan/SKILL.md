---
name: plan
description: "Plan mode: write an actionable markdown plan to .hermes/plans/, no execution. Bite-sized tasks, exact paths, complete code."
version: 2.0.0
author: Hermes Agent (writing-craft adapted from obra/superpowers)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, plan-mode, implementation, workflow, design, documentation]
    related_skills: [subagent-driven-development, test-driven-development, requesting-code-review]
---

# Plan Mode

Use this skill when the user wants a plan instead of execution.

## Core behavior

For this turn, you are planning only.

- Do not implement code.
- Do not edit project files except the plan markdown file.
- Do not run mutating terminal commands, commit, push, or perform external actions.
- You may inspect the repo or other context with read-only commands/tools when needed.
- Your deliverable is either a markdown plan saved inside the active workspace under `.hermes/plans/`, or a questions-only intake artifact when clarity is still below the planning threshold.

## Planning Confidence Gate

For any non-trivial plan or plan-like artifact, do not draft the plan until you have **greater than 96% confidence** that you understand what you are planning for.

Before planning, ask follow-up questions until all of the following are clear:
- Objective
- Scope boundaries
- Success criteria
- Constraints and preferences
- Dependencies, risks, and unknowns

If those are not clear enough, do **not** draft a partial plan. Produce a **questions-only intake artifact** instead.

Apply this rule both when generating a plan and when reviewing a drafted plan.

Skip the formal gate only when the task is trivial and no real plan artifact is needed.

## Output requirements

Write a markdown plan that is concrete, actionable, and reviewable.

Include, when relevant:
- Goal
- Current context / assumptions
- Proposed approach
- Step-by-step plan
- Files likely to change
- Tests / validation
- Risks, tradeoffs, and open questions

Every non-trivial plan MUST also include:
- Confidence
- Known Unknowns
- Blocking Questions Resolved

If the task is code-related, include exact file paths, likely test targets, and verification steps.

## Save location

Save the plan with `write_file` under:
- `.hermes/plans/YYYY-MM-DD_HHMMSS-<slug>.md`

Treat that as relative to the active working directory / backend workspace. Hermes file tools are backend-aware, so using this relative path keeps the plan with the workspace on local, docker, ssh, modal, and daytona backends.

If the runtime provides a specific target path, use that exact path.
If not, create a sensible timestamped filename yourself under `.hermes/plans/`.

## Interaction style

- If the task is trivial and no real plan artifact is needed, skip the formal confidence gate.
- Otherwise, do not write the plan directly just because the request sounds familiar; first confirm you have >96% confidence.
- If no explicit instruction accompanies `/plan`, infer the task from the current conversation context.
- Ask follow-up questions until the objective, scope, success criteria, constraints/preferences, and unknowns are sufficiently clear.
- If clarity is still insufficient after initial clarification, produce a questions-only intake artifact instead of a plan.
- After saving the plan, reply briefly with what you planned, the saved path, and the confidence level.

---

# Writing the Plan Well

The rest of this skill is the craft of authoring a *good* plan artifact — the content that goes inside the markdown file above. Most examples below are implementation-oriented, but the same discipline applies to research plans, debugging plans, architecture proposals, migration checklists, and review task lists.

## Overview

Write comprehensive implementation plans assuming the implementer has zero context for the codebase and questionable taste. Document everything they need: which files to touch, complete code, testing commands, docs to check, how to verify. Give them bite-sized tasks. DRY. YAGNI. TDD. Frequent commits.

Assume the implementer is a skilled developer but knows almost nothing about the toolset or problem domain. Assume they don't know good test design very well.

**Core principle:** A good plan makes execution or review obvious. If someone has to guess, the plan is incomplete.

## When a Full Plan Artifact Helps

**Always use before:**
- Implementing multi-step features
- Breaking down complex requirements
- Delegating to subagents via subagent-driven-development
- Producing research plans, debugging plans, architecture proposals, migration checklists, or review task lists that others will execute or review

**Don't skip when:**
- Feature seems simple (assumptions cause bugs)
- You plan to implement it yourself (future you needs guidance)
- Working alone (documentation matters)

## Bite-Sized Task Granularity

**Each task = 2-5 minutes of focused work.**

Every step is one action:
- "Write the failing test" — step
- "Run it to make sure it fails" — step
- "Implement the minimal code to make the test pass" — step
- "Run the tests and make sure they pass" — step
- "Commit" — step

**Too big:**
```markdown
### Task 1: Build authentication system
[50 lines of code across 5 files]
```

**Right size:**
```markdown
### Task 1: Create User model with email field
[10 lines, 1 file]

### Task 2: Add password hash field to User
[8 lines, 1 file]

### Task 3: Create password hashing utility
[15 lines, 1 file]
```

## Plan Document Structure

### Header (Required)

Every non-trivial plan MUST start with:

```markdown
# [Feature Name] Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

**Confidence:** [e.g. 97% — why the objective/scope/success criteria are clear enough to plan safely]

**Known Unknowns:** [Unknown but acceptable items, or "None currently"]

**Blocking Questions Resolved:** [Questions that had to be answered before planning, or "None"]

---
```

### Task Structure

Each task follows this format:

````markdown
### Task N: [Descriptive Name]

**Objective:** What this task accomplishes (one sentence)

**Files:**
- Create: `exact/path/to/new_file.py`
- Modify: `exact/path/to/existing.py:45-67` (line numbers if known)
- Test: `tests/path/to/test_file.py`

**Step 1: Write failing test**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify failure**

Run: `pytest tests/path/test.py::test_specific_behavior -v`
Expected: FAIL — "function not defined"

**Step 3: Write minimal implementation**

```python
def function(input):
    return expected
```

**Step 4: Run test to verify pass**

Run: `pytest tests/path/test.py::test_specific_behavior -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
````

## Writing Process

### Step 1: Understand Requirements

Read and understand:
- Feature requirements
- Design documents or user description
- Acceptance criteria
- Constraints

### Step 2: Explore the Codebase

Use Hermes tools to understand the project:

```python
# Understand project structure
search_files("*.py", target="files", path="src/")

# Look at similar features
search_files("similar_pattern", path="src/", file_glob="*.py")

# Check existing tests
search_files("*.py", target="files", path="tests/")

# Read key files
read_file("src/app.py")
```

### Step 3: Design Approach

Decide:
- Architecture pattern
- File organization
- Dependencies needed
- Testing strategy

### Step 4: Write Tasks

Create tasks in order:
1. Setup/infrastructure
2. Core functionality (TDD for each)
3. Edge cases
4. Integration
5. Cleanup/documentation

### Step 5: Add Complete Details

For each task, include:
- **Exact file paths** (not "the config file" but `src/config/settings.py`)
- **Complete code examples** (not "add validation" but the actual code)
- **Exact commands** with expected output
- **Verification steps** that prove the task works

### Step 6: Review the Plan

Check:
- [ ] Confidence is >96% for this non-trivial plan artifact
- [ ] Objective is explicit
- [ ] Scope boundaries are explicit
- [ ] Success criteria are explicit
- [ ] Constraints and preferences are explicit
- [ ] Dependencies, risks, and unknowns are identified
- [ ] Unknowns are either resolved or explicitly accepted
- [ ] Tasks are sequential and logical
- [ ] Each task is bite-sized (2-5 min)
- [ ] File paths are exact
- [ ] Code examples are complete (copy-pasteable)
- [ ] Commands are exact with expected output
- [ ] The plan does not hide assumptions in place of clarification
- [ ] Confidence, Known Unknowns, and Blocking Questions Resolved are present
- [ ] DRY, YAGNI, TDD principles applied

### When Confidence Is Below Threshold

If you cannot honestly claim >96% confidence, do **not** finalize the plan. Write a questions-only intake artifact instead:

```markdown
# Questions-Only Intake

## Why no plan yet
I do not yet have >96% confidence that I understand what to plan for correctly.

## Missing clarity
- Objective:
- Scope boundaries:
- Success criteria:
- Constraints/preferences:
- Dependencies/risks/unknowns:

## Follow-up questions
1.
2.
3.

## What is blocked until clarified
-
-
-

## Confidence
Current confidence: __%
Threshold required to plan: >96%
```

## Principles

### DRY (Don't Repeat Yourself)

**Bad:** Copy-paste validation in 3 places
**Good:** Extract validation function, use everywhere

### YAGNI (You Aren't Gonna Need It)

**Bad:** Add "flexibility" for future requirements
**Good:** Implement only what's needed now

```python
# Bad — YAGNI violation
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.preferences = {}  # Not needed yet!
        self.metadata = {}     # Not needed yet!

# Good — YAGNI
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
```

### TDD (Test-Driven Development)

Every task that produces code should include the full TDD cycle:
1. Write failing test
2. Run to verify failure
3. Write minimal code
4. Run to verify pass

See `test-driven-development` skill for details.

### Frequent Commits

Commit after every task:
```bash
git add [files]
git commit -m "type: description"
```

## Common Mistakes

### Vague Tasks

**Bad:** "Add authentication"
**Good:** "Create User model with email and password_hash fields"

### Incomplete Code

**Bad:** "Step 1: Add validation function"
**Good:** "Step 1: Add validation function" followed by the complete function code

### Missing Verification

**Bad:** "Step 3: Test it works"
**Good:** "Step 3: Run `pytest tests/test_auth.py -v`, expected: 3 passed"

### Missing File Paths

**Bad:** "Create the model file"
**Good:** "Create: `src/models/user.py`"

## Execution Handoff

After saving the plan, offer the execution approach:

**"Plan complete and saved. Ready to execute using subagent-driven-development — I'll dispatch a fresh subagent per task with two-stage review (spec compliance then code quality). Shall I proceed?"**

When executing, use the `subagent-driven-development` skill:
- Fresh `delegate_task` per task with full context
- Spec compliance review after each task
- Code quality review after spec passes
- Proceed only when both reviews approve

## Final Planning Check

Append this review block to the end of plan-related prompts and ask it explicitly before finalizing:

```text
Final planning check:
Do not finalize this plan unless you have greater than 96% confidence that the objective, scope, success criteria, constraints, and unknowns are clear enough for safe execution. If confidence is lower, stop and ask follow-up questions instead.
```

## Remember

```
Bite-sized tasks (2-5 min each)
Exact file paths
Complete code (copy-pasteable)
Exact commands with expected output
Verification steps
DRY, YAGNI, TDD
Frequent commits
```

**A good plan makes implementation obvious.**
