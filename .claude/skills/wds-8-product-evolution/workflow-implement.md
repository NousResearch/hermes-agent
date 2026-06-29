---
name: implement
description: Code the designed improvement in a new branch
borrows_from: Phase 5 (development)
---

# Implement

**Goal:** Implement the approved design in code, working in a dedicated branch like a developer on the team.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Load Specification

Read the specification from [D] Design Solution:

- Change summary
- Component specifications
- Acceptance criteria
- Pages affected

### Step 2: Create Branch

Create a feature branch for this improvement:

```
git checkout -b evolution/[scenario-name]
```

Naming convention: `evolution/` prefix + kebab-case scenario name.

### Step 3: Understand Current Code

Before writing code, understand what exists:

1. Locate the files for affected pages/views
2. Read current component implementations
3. Identify the tech stack patterns (framework, styling approach, state management)
4. Note any existing design tokens or theme configuration

Present a brief implementation plan:
- Which files will change
- What new files are needed (if any)
- Estimated complexity

### Step 4: Implement Changes

Write the code changes following the specification:

1. **Follow existing patterns** — Match the codebase's conventions, don't introduce new ones
2. **Minimal changes** — Only change what the specification calls for
3. **Commit incrementally** — One logical commit per change unit
4. **Test as you go** — Verify each change works before moving on

For each file changed, explain what was modified and why.

### Step 5: Self-Review

Before handing off:

1. Diff all changes: `git diff evolution/[scenario-name]..main`
2. Check against specification: every acceptance criterion addressed?
3. Check for unintended side effects: other pages/components still work?
4. Clean up: no debug code, no commented-out blocks, no unrelated changes

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action
3. Return to activity menu
