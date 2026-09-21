---
name: execute-discipline
description: "Implement EXACTLY what was planned. Plan verification, deviation handling, mid-implementation check-in, specialist delegation. No creative decisions during execution."
version: 1.0.0
author: KENSEI (extracted from withkynam/vibecode-pro-max-kit)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [execution, implementation, plan-fidelity, deviation-handling, self-review]
    related_skills: [writing-plans, systematic-debugging, github-code-review, subagent-driven-development]
---

# Execute Discipline

## Purpose

Implement EXACTLY what was specified in the approved plan. Write production-grade changes, not prototypes. Handle failures explicitly, validate at system boundaries, and do not leave correctness-blocking TODOs behind.

## Entry Requirement

ONLY enter after an explicit plan exists and has been approved.

This is a critical safety checkpoint. Never auto-enter execution without a plan.

## Plan Verification

At session start, before any implementation:

1. Require one explicit selected plan file or spec
2. Read that plan and confirm the phase/task to implement
3. If no exact plan was provided → **STOP**. Tell the user:
   "No explicit plan was provided. Please create or select an approved plan before proceeding."
4. Inspect the plan's touchpoints, public contracts, blast radius, verification evidence, and handoff sections before touching code
5. If a needed section is missing, stop and return to planning rather than guessing

**Exception**: Trivial fixes (single-file, under 15 lines, no schema/auth changes) may proceed without a plan file.

## Permitted Activities

- Implementing planned features
- Modifying source code files
- Creating new files per plan
- Running build/test commands
- Deleting files if specified in plan
- All development activities explicitly specified in plan
- Running the exact verification commands needed to prove the implementation works

## Strictly Forbidden

- Any deviation from approved plan
- Adding "improvements" not in plan
- Refactoring not specified
- Changing approach mid-implementation
- Making creative decisions not in plan

## Deviation Handling

If ANY issue requires deviating from plan:

1. **IMMEDIATELY STOP** implementation
2. Explain the issue clearly
3. Explain why deviation is needed
4. State: "This requires updating the plan. Returning to planning phase."
5. Wait for user to approve plan update
6. Resume execution with updated plan after approval

**Never silently deviate**. Always stop and get approval first.

## Mid-Implementation Check-In

At approximately 50% completion:

1. Provide status update
2. List completed checklist items
3. List remaining items
4. Ask: "Continue with current approach or pause and return to planning?"
5. If user indicates hesitation, immediately pause and reassess

## Specialist Delegation

During implementation, you may delegate to specialist agents for quality and verification:

- **After completing implementation sub-steps**: Invoke testing for diff-aware verification
- **When encountering a bug during implementation**: Invoke systematic debugging for root cause analysis
- **Before marking complete**: Invoke code review for production-readiness review
- **For UI/UX tasks**: Invoke design review

Delegation is optional but recommended for non-trivial work. Keep final execution ownership — helpers stay bounded.

## Self-Review After Execution

After completing implementation, perform line-by-line verification against approved plan:

1. **Re-read the approved plan**
2. **Check each checklist item** — was it implemented exactly as specified?
3. **Flag any deviations**, no matter how minor:
   - File path: [exact path]
   - Deviation: [what differs from plan]
   - Rationale: [why it was necessary]
4. **Summarize**:
   - ✅ Implementation matches plan — No deviations found
   - ❌ Deviations detected — List all deviations with rationale

If material deviations exist, STOP and suggest returning to planning to reconcile.

## Implementation Discipline

- Follow plan with 100% fidelity
- Don't stop until task is fully completed
- Check off items from plan checklist as you complete them
- Test critical functionality after implementation
- Validate input and output boundaries where the plan touches external data, APIs, or user input
- Add tests for new logic when the plan calls for testable behavior
- Resolve correctness issues before calling the work complete

Before marking execution complete, verify each item:

- [ ] Error handling added where required by the plan and existing code patterns
- [ ] External input boundaries validated where applicable
- [ ] No correctness-blocking TODO/FIXME left behind
- [ ] Interfaces and public behavior match the approved plan exactly
- [ ] New logic has matching tests when the plan requires them
- [ ] Typecheck/build/test verification completed where relevant

## High-Risk Verification

For high-risk work classes, require a durable evidence contract before treating the work as fully proven:

- Auth or identity
- Billing or credits
- Schema/data migration or destructive mutation
- Public API contract changes
- Deploy/runtime/container/proxy/gateway changes
- Permission, secret, or trust-boundary logic

Expected evidence pack:
- `risk-gate.json` — risk class, stop requirement, rationale
- `context-snippets.json` — log/query/diff evidence that proves the change
- `verification.json` — test results, manual verification notes
- `review-decision.json` — reviewer verdict and conditions
- `adversarial-validation.json` when the path is high-risk or attack-sensitive

If the risk gate says `mustStopBeforeFinalize: true`, or the required evidence pack is missing for applicable high-risk work, STOP and classify the work as `needs-reconciliation` instead of implying it is complete.

## Completion

After implementation and self-review:

1. Present results and self-review summary
2. Explicitly classify the closeout state:
   - `ready-for-archive` — no deviations, verification sufficient
   - `keep-active` — implementation complete but testing/verification still pending
   - `needs-reconciliation` — deviations exist, plan needs update
3. Include a short closeout packet:
   - selected plan reference
   - what was finished
   - what was verified versus still unverified
   - what cleanup/context capture remains
   - the single best next valid state

Never auto-transition to the next phase. Wait for user command.
