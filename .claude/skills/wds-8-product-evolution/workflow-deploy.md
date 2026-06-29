---
name: deploy
description: Create PR and deliver the improvement to the team
borrows_from: Phase 4 [H] (design delivery)
---

# Deploy

**Goal:** Package the tested improvement as a PR and deliver it to the development team.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.


## Steps

### Step 1: Pre-Deploy Checklist

Verify everything is ready:

- [ ] All acceptance criteria pass (from [T] test report)
- [ ] Branch is clean (no uncommitted changes)
- [ ] Commits are logical and well-described
- [ ] No unrelated changes included
- [ ] Documentation updated (if applicable)

### Step 2: Create Pull Request

Create a PR from the evolution branch:

```
gh pr create --title "[Improvement]: [Brief description]" --body "..."
```

PR body includes:
- **What changed** — Summary of the improvement
- **Why** — Link to scenario and analysis
- **How to test** — Steps from the test report
- **Screenshots** — Before/after if visual change
- **Acceptance criteria** — Checklist from spec

### Step 3: Package Delivery Context

Create a delivery summary at `{output_folder}/evolution/deliveries/`:

```markdown
# Delivery: [Scenario Name]

## PR
[Link to PR]

## Artifacts
- Analysis: [link]
- Scenario: [link]
- Specification: [link]
- Test Report: [link]

## Change Summary
[What was changed and why]

## Impact
[Expected improvement based on success criteria]

## Monitoring
[What to watch after deployment — metrics, error rates, user feedback]
```

### Step 4: Notify Team

If the project uses design log tracking or team notifications:

1. Create completion notification
2. Reference all artifacts (analysis → scenario → spec → test → PR)
3. Include any monitoring instructions

### Step 5: Plan Next Cycle

After deployment:

1. Archive this improvement cycle
2. Review remaining improvement targets from [A] analysis
3. Suggest next target or new analysis round

---

## AFTER COMPLETION

1. Update design log with completed improvement
2. Return to Phase 8 Activity Menu for next cycle
