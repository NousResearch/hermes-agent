---
name: design-solution
description: Sketch and specify the update for a scoped improvement
borrows_from: Phase 4 (UX design)
---

# Design Solution

**Goal:** Design the solution for a scoped improvement — from quick sketch to development-ready specification.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Load Scenario

Read the scenario from [S] Scope Improvement:

- Target description
- Current vs desired state
- User journey
- Pages and components affected

### Step 2: Choose Design Approach

Based on the change scope, pick an approach:

- **Quick fix** — Small visual/copy change → Skip to Step 4 (specify directly)
- **Sketch first** — Layout or flow change → Sketch the before/after, then specify
- **Generate design** — Significant visual change → Use Phase 6 asset generation tools

### Step 3: Design the Change

For sketch or generate approaches:

1. **Before snapshot** — Capture or describe the current view
2. **After concept** — Sketch, generate, or describe the desired view
3. **Diff view** — Explicitly mark what changes: layout, components, content, behavior
4. **Edge cases** — What happens on mobile? With long text? With empty state?

Present design to user for feedback. Iterate until approved.

### Step 4: Write Specification

Create a mini page-spec at `{output_folder}/evolution/specs/`:

```markdown
# [Page/View Name] — Update Specification

## Change Summary
[One paragraph describing the change]

## Before
[Current state description or reference]

## After
[Detailed specification of the new state]

## Components
[List each component with its new properties/behavior]

## Responsive Behavior
[How the change adapts across breakpoints]

## Acceptance Criteria
[Testable criteria from the scenario]
```

### Step 5: Approve Specification

Present the specification for user sign-off:

- Does it match the scenario intent?
- Are acceptance criteria testable?
- Is scope still manageable?

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action
3. Return to activity menu
