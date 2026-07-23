# Ultraresearch Phase 0 — Hermes goal and AGI-agent foundation

## Core question

Which small, evidence-backed additions to the current Hermes goal and learning
runtime make user-approved work more goal-directed, verifiable, and reusable
without weakening existing durability, ownership, or approval boundaries?

## Axes

1. **Current goal/outcome lifecycle** — trace persisted goal creation,
   completion, receipts, retry/idempotency, and tests to identify extension
   seams and invariant coverage.
2. **Learning/evidence lifecycle** — map how verified outcomes are stored,
   retrieved, expired, and exposed so a new capability can be grounded in
   evidence rather than an unbounded self-modifying loop.
3. **AGI-agent architecture and evaluation** — compare official/current
   agent-system guidance and open-source patterns for goal decomposition,
   bounded learning, observability, and evaluation; identify the smallest
   compatible feature.
4. **Integration and safety** — inspect main/recovery history and existing
   isolation/verification rules to ensure the selected change is atomic,
   testable, auditable, and safe to merge into local main.

## Scope and evidence rules

- Codebase: yes. External sources: yes. Browsing: yes. Execution verification:
  yes. Report: repository Markdown journal and committed design/test evidence.
- This branch is isolated from dirty root and main worktrees. No push, release,
  configuration mutation, GC, prune, reset, or destructive cleanup is in scope.
- Research findings are read-only until the parent selects a minimal change
  supported by current code and evidence.
