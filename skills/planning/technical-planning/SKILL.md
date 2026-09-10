---
name: technical-planning
description: >
  Plans technical work by drafting structured implementation plans or refining existing ones.
  Covers architecture, module breakdowns, design decisions, tradeoffs, and step-by-step execution.
  Use when the user asks to plan, design, architect, or think through how to implement a feature or task.
  Combines architectural planning (eqlion/plan), step-by-step implementation planning (jtsang4/writing-plans),
  and batch execution with checkpoints (jtsang4/executing-plans).
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, architecture, implementation, design-decisions, tradeoffs]
    sources:
      - eqlion/skills-and-agents/skills/plan
      - jtsang4/efficient-coding/skills/writing-plans
      - jtsang4/efficient-coding/skills/executing-plans
---

# Technical Planning

This skill covers three interconnected activities: **architectural planning** (what and why), **writing implementation plans** (step-by-step tasks), and **executing plans** (batch execution with checkpoints).

---

## 1. Architectural Planning (Technical Spec)

`plan.md` is the **technical spec** for a task. It answers HOW we're going to achieve the goal — at the architectural level: which modules and layers are involved, where new responsibility lives, what flows through where, what alternatives were considered, and why this shape was chosen.

A good plan is one a senior engineer could read and then go implement the feature themselves without needing to ask which files to open — because the architecture and boundaries are clear enough to make those decisions locally.

### When to draft a plan

- The user wants to draft the initial technical plan for a task whose requirements are roughly settled
- The user wants to refine an existing plan after an architecture conversation
- The user wants to talk through design decisions, tradeoffs, or module boundaries
- The user wants to stress-test a plan

### What a plan is NOT

- **Not a file-by-file edit list.** Do not enumerate the specific files to touch.
- **Not a code-snippet appendix.** No code blocks. Restate the intent in prose; the implementer will write the code.
- **Not a phased implementation checklist.** Implementation order is decided during implementation. If sequencing matters *architecturally*, record it as a dependency under **Risks & open questions**, not as a phase.

### Plan template

```markdown
# <Ticket>: <Title>

## Context
One paragraph: pointer to requirements.md and a one-sentence framing of the goal.

## Prior art (optional)
1-3 bullets summarising lessons from POCs, abandoned attempts, ADRs, or related tickets.

## Architecture overview
The shape of the change, end-to-end. A diagram or sequence description covering:
- Which modules / layers are involved
- How data flows between them for the primary scenario
- Where the new responsibility lives, and where existing responsibilities are reused vs extended

## Module / layer breakdown
For each affected boundary: what responsibility this layer holds for this feature, what it depends on, what it exposes. Name modules/protocols by existing names; do NOT name new files — call the new responsibility by its role.

## Design decisions
For each non-obvious decision:
- **What**: the decision in one sentence.
- **Why**: the constraints / requirements that drove it.
- **Alternatives considered**: 1-3 bullets, each with the reason it was rejected. If there were no real alternatives, say so.

## Risks & open questions
- **Risks**: things that could go wrong at runtime or during rollout, and how the design bounds them.
- **Open questions**: things the implementer resolves at implementation time.

## Out of scope
Bulleted list of things this task explicitly does not address, one line each.
```

### Drafting rules

- **No file paths in the body** except when referencing existing, unchanged anchors.
- **No code snippets.** Prose only.
- **Scale to complexity.** A two-file fix needs Context plus a short Architecture paragraph. A new module needs the full template.
- **Surface ambiguities.** Put unresolved decisions in **Risks & open questions** rather than papering over them.

---

## 2. Writing Implementation Plans

Write comprehensive implementation plans before touching code. Document everything an implementer needs: which files to touch, exact code, testing instructions, how to verify.

**Announce at start:** "I'm using the technical-planning skill to create the implementation plan."

### Task granularity

Each step is one action (2-5 minutes):
- "Write the failing test" — step
- "Run it to make sure it fails" — step
- "Implement the minimal code" — step
- "Run the tests" — step
- "Commit" — step

### Plan document header

```markdown
# [Feature Name] Implementation Plan

**Goal:** [One sentence describing what this builds]
**Architecture:** [2-3 sentences about approach]
**Tech Stack:** [Key technologies/libraries]

---
```

### Task structure

```markdown
### Task N: [Component Name]

**Files:**
- Create: `exact/path/to/file.py`
- Modify: `exact/path/to/existing.py:123-145`
- Test: `tests/exact/path/to/test.py`

**Step 1: Write the failing test**

[test code]

**Step 2: Run test to verify it fails**
Run: `pytest tests/path/test.py::test_name -v`
Expected: FAIL

**Step 3: Write minimal implementation**
[exact code]

**Step 4: Run test to verify it passes**
Run: `pytest tests/path/test.py::test_name -v`
Expected: PASS

**Step 5: Commit**

git add tests/ src/
git commit -m "feat: add specific feature"
```

### Key principles
- Exact file paths always
- Complete code in plan (not "add validation")
- Exact commands with expected output
- DRY, YAGNI, TDD, frequent commits

---

## 3. Executing Plans

Load a written plan, review critically, execute tasks in batches, report for review between batches.

**Announce at start:** "I'm using the technical-planning skill to execute the plan."

### Process

**Step 1: Load and Review Plan**
1. Read plan file
2. Review critically — identify any questions or concerns
3. If concerns: raise them before starting
4. If no concerns: create task list and proceed

**Step 2: Execute Batch (default: first 3 tasks)**
For each task:
1. Mark as in_progress
2. Follow each step exactly
3. Run verifications as specified
4. Mark as completed

**Step 3: Report**
- Show what was implemented
- Show verification output
- Say: "Ready for feedback."

**Step 4: Continue**
- Apply changes based on feedback
- Execute next batch
- Repeat until complete

**Step 5: Complete Development**
- Verify all tests pass
- Present options for finalizing (PR, merge, etc.)

### When to Stop and Ask for Help
- Hit a blocker mid-batch (missing dependency, test fails, instruction unclear)
- Plan has critical gaps preventing starting
- Verification fails repeatedly
- **Ask for clarification rather than guessing.**

---

## Integration notes

- Start with **requirements** before planning (WHAT before HOW).
- Use the architectural plan to write the implementation plan.
- Execute the implementation plan in batches with review checkpoints.
- For code review of the implemented plan, use the `code-review` skill.
