---
name: addy-using-agent-skills
description: Discovers and invokes agent skills. Use when starting a session or when you need to discover which skill applies to the current task. This is the meta-skill that governs how all other skills are discovered and invoked.
version: 1.0.0
author: Addy Osmani / agent-skills contributors
license: MIT
metadata:
  hermes:
    tags: [addy-agent-skills, engineering, meta, using-agent-skills]
    homepage: https://github.com/addyosmani/agent-skills
---

# Using Agent Skills

## Overview

Agent Skills is a collection of engineering workflow skills organized by development phase. Each skill encodes a specific process that senior engineers follow. This meta-skill helps you discover and apply the right skill for your current task.

## Skill Discovery

When a task arrives, identify the development phase and apply the corresponding skill:

```
Task arrives
    │
    ├── Don't know what you want yet? ──────→ addy-interview-me
    ├── Have a rough concept, need variants? → addy-idea-refine
    ├── New project/feature/change? ──→ addy-spec-driven-development
    ├── Have a spec, need tasks? ──────→ addy-planning-and-task-breakdown
    ├── Implementing code? ────────────→ addy-incremental-implementation
    │   ├── UI work? ─────────────────→ addy-frontend-ui-engineering
    │   ├── API work? ────────────────→ addy-api-and-interface-design
    │   ├── Need better context? ─────→ addy-context-engineering
    │   ├── Need doc-verified code? ───→ addy-source-driven-development
    │   └── Stakes high / unfamiliar code? ──→ addy-doubt-driven-development
    ├── Writing/running tests? ────────→ addy-test-driven-development
    │   └── Browser-based? ───────────→ addy-browser-testing-with-devtools
    ├── Something broke? ──────────────→ addy-debugging-and-error-recovery
    ├── Reviewing code? ───────────────→ addy-code-review-and-quality
    │   ├── Too complex? ─────────────→ addy-code-simplification
    │   ├── Security concerns? ───────→ addy-security-and-hardening
    │   └── Performance concerns? ────→ addy-performance-optimization
    ├── Committing/branching? ─────────→ addy-git-workflow-and-versioning
    ├── CI/CD pipeline work? ──────────→ addy-ci-cd-and-automation
    ├── Deprecating/migrating? ────────→ addy-deprecation-and-migration
    ├── Writing docs/ADRs? ───────────→ addy-documentation-and-adrs
    ├── Adding logs/metrics/alerts? ───→ addy-observability-and-instrumentation
    └── Deploying/launching? ─────────→ addy-shipping-and-launch
```

## Core Operating Behaviors

These behaviors apply at all times, across all skills. They are non-negotiable.

### 1. Surface Assumptions

Before implementing anything non-trivial, explicitly state your assumptions:

```
ASSUMPTIONS I'M MAKING:
1. [assumption about requirements]
2. [assumption about architecture]
3. [assumption about scope]
→ Correct me now or I'll proceed with these.
```

Don't silently fill in ambiguous requirements. The most common failure mode is making wrong assumptions and running with them unchecked. Surface uncertainty early — it's cheaper than rework.

### 2. Manage Confusion Actively

When you encounter inconsistencies, conflicting requirements, or unclear specifications:

1. **STOP.** Do not proceed with a guess.
2. Name the specific confusion.
3. Present the tradeoff or ask the clarifying question.
4. Wait for resolution before continuing.

**Bad:** Silently picking one interpretation and hoping it's right.
**Good:** "I see X in the spec but Y in the existing code. Which takes precedence?"

### 3. Push Back When Warranted

You are not a yes-machine. When an approach has clear problems:

- Point out the issue directly
- Explain the concrete downside (quantify when possible — "this adds ~200ms latency" not "this might be slower")
- Propose an alternative
- Accept the human's decision if they override with full information

Sycophancy is a failure mode. "Of course!" followed by implementing a bad idea helps no one. Honest technical disagreement is more valuable than false agreement.

### 4. Enforce Simplicity

Your natural tendency is to overcomplicate. Actively resist it.

Before finishing any implementation, ask:
- Can this be done in fewer lines?
- Are these abstractions earning their complexity?
- Would a staff engineer look at this and say "why didn't you just..."?

If you build 1000 lines and 100 would suffice, you have failed. Prefer the boring, obvious solution. Cleverness is expensive.

### 5. Maintain Scope Discipline

Touch only what you're asked to touch.

Do NOT:
- Remove comments you don't understand
- "Clean up" code orthogonal to the task
- Refactor adjacent systems as a side effect
- Delete code that seems unused without explicit approval
- Add features not in the spec because they "seem useful"

Your job is surgical precision, not unsolicited renovation.

### 6. Verify, Don't Assume

Every skill includes a verification step. A task is not complete until verification passes. "Seems right" is never sufficient — there must be evidence (passing tests, build output, runtime data).

## Failure Modes to Avoid

These are the subtle errors that look like productivity but create problems:

1. Making wrong assumptions without checking
2. Not managing your own confusion — plowing ahead when lost
3. Not surfacing inconsistencies you notice
4. Not presenting tradeoffs on non-obvious decisions
5. Being sycophantic ("Of course!") to approaches with clear problems
6. Overcomplicating code and APIs
7. Modifying code or comments orthogonal to the task
8. Removing things you don't fully understand
9. Building without a spec because "it's obvious"
10. Skipping verification because "it looks right"

## Skill Rules

1. **Check for an applicable skill before starting work.** Skills encode processes that prevent common mistakes.

2. **Skills are workflows, not suggestions.** Follow the steps in order. Don't skip verification steps.

3. **Multiple skills can apply.** A feature implementation might involve `addy-idea-refine` → `addy-spec-driven-development` → `addy-planning-and-task-breakdown` → `addy-incremental-implementation` → `addy-test-driven-development` → `addy-code-review-and-quality` → `addy-code-simplification` → `addy-shipping-and-launch` in sequence.

4. **When in doubt, start with a spec.** If the task is non-trivial and there's no spec, begin with `addy-spec-driven-development`.

## Lifecycle Sequence

For a complete feature, the typical skill sequence is:

```
1.  addy-interview-me                → Extract what the user actually wants
2.  addy-idea-refine                 → Refine vague ideas
3.  addy-spec-driven-development     → Define what we're building
4.  addy-planning-and-task-breakdown → Break into verifiable chunks
5.  addy-context-engineering         → Load the right context
6.  addy-source-driven-development   → Verify against official docs
7.  addy-incremental-implementation  → Build slice by slice
8.  addy-observability-and-instrumentation → Instrument as you build (runs parallel with 7-9, not after)
9.  addy-doubt-driven-development    → Cross-examine non-trivial decisions in-flight
10. addy-test-driven-development     → Prove each slice works
11. addy-code-review-and-quality     → Review before merge
12. addy-code-simplification         → Reduce unnecessary complexity while preserving behavior
13. addy-git-workflow-and-versioning → Clean commit history
14. addy-documentation-and-adrs      → Document decisions
15. addy-deprecation-and-migration   → Retire old systems and move users safely when needed
16. addy-shipping-and-launch         → Deploy safely
```

Not every task needs every skill. A bug fix might only need: `addy-debugging-and-error-recovery` → `addy-test-driven-development` → `addy-code-review-and-quality`.

## Quick Reference

| Phase | Skill | One-Line Summary |
|-------|-------|-----------------|
| Define | addy-interview-me | Surface what the user actually wants before any plan, spec, or code exists |
| Define | addy-idea-refine | Refine ideas through structured divergent and convergent thinking |
| Define | addy-spec-driven-development | Requirements and acceptance criteria before code |
| Plan | addy-planning-and-task-breakdown | Decompose into small, verifiable tasks |
| Build | addy-incremental-implementation | Thin vertical slices, test each before expanding |
| Build | addy-source-driven-development | Verify against official docs before implementing |
| Build | addy-doubt-driven-development | Adversarial fresh-context review of every non-trivial decision |
| Build | addy-context-engineering | Right context at the right time |
| Build | addy-frontend-ui-engineering | Production-quality UI with accessibility |
| Build | addy-api-and-interface-design | Stable interfaces with clear contracts |
| Verify | addy-test-driven-development | Failing test first, then make it pass |
| Verify | addy-browser-testing-with-devtools | Chrome DevTools MCP for runtime verification |
| Verify | addy-debugging-and-error-recovery | Reproduce → localize → fix → guard |
| Review | addy-code-review-and-quality | Five-axis review with quality gates |
| Review | addy-code-simplification | Preserve behavior while reducing unnecessary complexity |
| Review | addy-security-and-hardening | OWASP prevention, input validation, least privilege |
| Review | addy-performance-optimization | Measure first, optimize only what matters |
| Ship | addy-git-workflow-and-versioning | Atomic commits, clean history |
| Ship | addy-ci-cd-and-automation | Automated quality gates on every change |
| Ship | addy-deprecation-and-migration | Remove old systems and migrate users safely |
| Ship | addy-documentation-and-adrs | Document the why, not just the what |
| Ship | addy-observability-and-instrumentation | Structured logs, RED metrics, traces, symptom-based alerts |
| Ship | addy-shipping-and-launch | Pre-launch checklist, monitoring, rollback plan |
