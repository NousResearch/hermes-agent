---
name: analyze-product
description: Understand current product state and find improvement targets
borrows_from: Phase 3 (scenarios)
---

# Analyze Product

**Goal:** Understand the existing product, identify what needs improving, and prioritize targets.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Load Product Context

Gather existing product information:

1. Read project configuration and any existing WDS artifacts
2. If the product has a live URL → analyze current state (structure, pages, flows)
3. If codebase available → scan for tech stack, component patterns, design tokens
4. Document what exists: pages, navigation, key user flows

Present a **Product Snapshot** — current state summary.

### Step 2: Identify Improvement Targets

With the user, identify what needs work:

1. **User feedback** — What are users struggling with?
2. **Business goals** — What metrics need improvement?
3. **Technical debt** — What's fragile or outdated?
4. **Visual gaps** — What looks inconsistent or dated?
5. **Competitor gaps** — What are competitors doing better?

Create a prioritized list of improvement targets.

### Step 3: Select Target

From the prioritized list, pick ONE target for this improvement cycle:

```
Improvement targets (prioritized):
1. [Target] — [Impact] — [Effort]
2. [Target] — [Impact] — [Effort]
...

Which target should we tackle first?
```

### Step 4: Document Analysis

Save analysis to `{output_folder}/evolution/analysis/`:

- Product snapshot
- Improvement targets with priorities
- Selected target with rationale

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action
3. Return to activity menu
