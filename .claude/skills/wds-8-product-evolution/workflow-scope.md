---
name: scope-improvement
description: Create a focused scenario for a specific product improvement
borrows_from: Phase 3 (scenarios)
---

# Scope Improvement

**Goal:** Turn an improvement target into a concrete scenario — one focused change with clear before/after.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Load Analysis

Read the analysis from [A] Analyze Product:

- Product snapshot
- Selected improvement target
- Context and rationale

### Step 2: Define the Change

Scope the improvement as a mini-scenario:

1. **Which view/page** needs changing? (Be specific — one page, one flow section)
2. **Current state** — What does the user see today? What's wrong?
3. **Desired state** — What should the user experience after the change?
4. **Success criteria** — How do we know it worked? (measurable if possible)

### Step 3: Map the User Journey

For the selected view, map the micro-journey:

1. **Entry point** — How does the user arrive at this view?
2. **Current flow** — What happens step by step today?
3. **Pain points** — Where exactly does the experience break down?
4. **Proposed flow** — What should happen step by step after the change?

### Step 4: Estimate Scope

Assess the change:

- **Pages affected**: List specific pages/views
- **Components touched**: Which UI elements change?
- **Data changes**: Any API or data model changes?
- **Risk level**: Low (visual only) / Medium (behavior change) / High (structural change)

### Step 5: Write Scenario

Create a scenario document at `{output_folder}/evolution/scenarios/`:

```markdown
# [Scenario Name]

## Target
[What we're improving and why]

## Current State
[What users experience today]

## Desired State
[What users should experience after]

## User Journey
[Step-by-step flow]

## Success Criteria
[How we measure success]

## Scope
[Pages, components, risk level]
```

Present for user approval.

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action
3. Return to activity menu
