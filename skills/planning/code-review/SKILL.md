---
name: code-review
description: >
  Multi-specialist review for plans and code changes. Runs structured review passes
  with area-specific checklists, tracks findings across rounds, and aggregates into an index.
  Covers plan review (architecture, boundaries, contracts, data-flow, alignment, tradeoffs, testability)
  and concise MR/PR description drafting. Use when the user asks to review a plan, review code changes,
  or write a merge request description.
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-review, plan-review, merge-request, description, review, architecture]
    sources:
      - eqlion/skills-and-agents/skills/plan-review
      - eqlion/skills-and-agents/skills/mr-description
---

# Code Review

This skill provides two review capabilities: **plan review** (reviewing a technical spec before implementation) and **MR/PR description drafting** (writing concise change descriptions).

---

## 1. Plan Review

Multi-specialist review of a task's `plan.md`. Orchestrate domain specialists to review architecture, boundaries, contracts, data-flow, requirements alignment, tradeoffs, and testability.

### When to invoke

- The user wants to review a plan before implementation
- The user wants to stress-test or challenge an architectural plan
- The user asks for a review of design decisions

### Inputs (required)

- **task_context_path** — absolute path to the task context directory (the folder with `plan.md`, `requirements.md`, etc.).

### Procedure

#### Step 1 — Validate inputs

1. Confirm `task_context_path` exists and is a directory.
2. Confirm `(task_context_path)/plan.md` exists and is non-empty. If missing, note "no plan to review" and exit.
3. Note whether `requirements.md` exists and is non-empty.

#### Step 2 — Prepare output paths

Create `(task_context_path)/plan-review/` if it does not exist.

Each specialist writes findings to its own file:

| Specialist | Area tag | Output file |
|---|---|---|
| boundaries | `boundaries` | `plan-review/boundaries.md` |
| contracts | `contracts` | `plan-review/contracts.md` |
| data-flow | `data-flow` | `plan-review/data-flow.md` |
| requirements-alignment | `requirements-alignment` | `plan-review/requirements-alignment.md` |
| tradeoffs | `tradeoffs` | `plan-review/tradeoffs.md` |
| testability | `testability` | `plan-review/testability.md` |

#### Step 3 — Review each area

For each specialist area, review the plan from that perspective:

- **Boundaries**: Are module/layer boundaries clean? Are responsibilities in the right place? Any coupling that shouldn't exist?
- **Contracts**: Are API/interface contracts clear? Are inputs/outputs well-defined? Are error cases covered?
- **Data-flow**: Does data flow correctly through the architecture? Are there bottlenecks, missing transformations, or race conditions?
- **Requirements-alignment**: Does the plan actually address what the requirements ask for? Are there gaps?
- **Tradeoffs**: Are the design decisions well-reasoned? Were viable alternatives considered and dismissed with good reasons?
- **Testability**: Can the proposed architecture be tested? Are there hard-to-test components?

For each finding, record:
- **id**: stable identifier (e.g., `boundaries-001`)
- **status**: `open` | `fixed` | `wontfix` | `disputed` | `resolved` | `withdrawn`
- **severity**: `blocker` | `major` | `minor` | `nit`
- **location**: section or area of the plan
- **title**: short description
- **description**: full explanation

#### Step 4 — Write the aggregated index

Write `(task_context_path)/plan-review/index.md`:

```markdown
# Plan Review Index

- **task:** <task-id-or-folder-name>
- **timestamp:** <YYYY-MM-DD HH:MM>
- **round:** <N>

## Convergence

- **converged:** <true | false> (true iff every finding is `resolved` or `withdrawn`)

## Status totals

| Status | Count |
|---|---|
| open | <N> |
| fixed | <N> |
| wontfix | <N> |
| disputed | <N> |
| resolved | <N> |
| withdrawn | <N> |

## Active worklist

Sorted by: status priority (disputed > open > wontfix > fixed), then severity (blocker > major > minor > nit).

Each row: `<status> · <id> · <severity> · <area> · <short title> · <location>`
```

#### Step 5 — Report to the user

Print a short summary (≤ 12 lines):
- Round number
- Converged: yes/no
- Status totals
- One-line note per area with non-terminal findings
- Pointer to the index file

### Multi-round review

After a review round:
1. The implementer reads the index, picks up each item from the Active worklist
2. For every `open` or `disputed` finding, the implementer edits the relevant per-area file
   - Set status to `fixed`, `wontfix`, or `disputed`
   - Add `planner-response` with a summary or justification
3. Re-run plan-review — reviewers evaluate planner responses and either resolve findings or hold
4. Repeat until **converged: yes**

---

## 2. MR/PR Description Drafting

Long MR descriptions don't get read. Reviewers skim. Keep the body on one screen and answer three questions: what changed, why, and how to verify.

### When to invoke

- The user asks to create/draft an MR or PR description
- The user asks to write up changes for review
- The user says "create a merge request" or "write a PR description"

### Procedure

**Step 1: Gather context**

1. Run `git log <base>..HEAD --oneline` and `git diff <base>...HEAD --stat` to understand the full set of changes.
2. Identify any ticket reference from branch name or commit subjects.
3. If the repo has a PR/MR template, read it.

**Step 2: Draft the title**

Format: `[<ticket>] <type>(<scope>): <description>`

Example: `[PROJ-1234] feat(auth): add biometric unlock flow`

**Step 3: Draft the body**

Target ≤ 15 lines for the body (excluding title and ticket link).

Required structure:
```markdown
# Links
- TICKET-XXX

# What's the goal or problem this MR addresses?
<1-2 sentences. State the outcome, not the steps.>

# How was it implemented?
- <main change 1, one line>
- <main change 2, one line>
- <main change 3, one line>
- <Optional: 1 line for any non-obvious trade-off>

/assign me
```

**Step 4: Verify length**

1. Count lines of the body.
2. If > 20 lines, warn and offer to regenerate with the strict template.

### Rules

#### Hard limits
- **Body**: ≤ 15 lines (excluding title and link)
- **No file-by-file changelog** — group by area or intent
- **No "background" sections longer than 2 sentences** — link to the ticket
- **No tables of changes** unless >5 distinct logical areas

#### Anti-patterns to avoid
- Restating every section of the plan/requirements doc
- Listing every modified file with a description
- Multiple subsection levels — flatten to one bullet list
- Embedding rollback strategy unless non-trivial
- "Stats" sections with token counts

#### When it can be longer

Exceed the 15-line cap only for:
- Architectural migrations affecting >20 files across modules
- Breaking API changes with explicit consumer migration steps
- Security-sensitive changes needing a threat-model rationale

In all other cases: **15 lines or less**.
