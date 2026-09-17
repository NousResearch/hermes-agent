# [PROJECT_NAME] Constitution

<!-- Seeded by spec-driven-dev's constitution-defaults.md. Refine every
     [BRACKETED] placeholder via /speckit-constitution before treating this
     as ratified -- these are starting defaults, not a finished document. -->

## Core Principles

### I. Test-First, Verification-Heavy (NON-NEGOTIABLE)
TDD mandatory: tests written before implementation, tests fail first, then
implement. Behavior contracts over snapshots -- a test asserts how two
pieces of data relate, never freezes a current value. No completion claims
before tests pass. Adversarial review for anything correctness-sensitive.

### II. Smallest Footprint First
Before adding new surface (a new module, tool, service, or dependency),
prefer extending what already exists. Apply a footprint ladder: extend
existing code > CLI command/script > service-gated capability > plugin >
new core surface (last resort). Grep for existing plumbing before writing
new code.

### III. Isolated, Reversible Work
Use isolated worktrees/branches for implementation work, not the main
checkout. Preserve blockers explicitly rather than guessing past them.
Safe canaries before broad rollout. Every irreversible or high-stakes
change gets explicit authority/owner gating before it lands.

### IV. Evidence Over Assumption
No guessed architecture. Cite repository evidence (`git log -p -S`,
existing code, prior spec/plan decisions) before asserting intent. When a
value judgment is genuinely required and doesn't block the current turn,
route it through Decision HUD rather than assuming an answer silently.

### V. [PRINCIPLE_5_NAME]
<!-- Project-specific: fill in during /speckit-constitution. -->
[PRINCIPLE_5_DESCRIPTION]

## Additional Constraints

<!-- Technology stack requirements, compliance standards, deployment
     policies specific to this project. Fill in during /speckit-constitution. -->

[SECTION_2_CONTENT]

## Development Workflow

Requirements are written as EARS-syntax lines (see
`docs/spec-driven/ears-syntax.md`) inside spec.md's Functional
Requirements section, not loose prose. Open questions surfaced during
`/speckit-clarify`, `/speckit-plan`, or `/speckit-tasks` are routed through
`spec_decision_gate.py` (see the spec-driven-dev skill) before being
answered silently or escalated -- never free-reasoned past.

## Governance

This constitution supersedes ad hoc practice. Amendments require
documentation of what changed and why, plus a migration note for any
work already in flight under the old version.

**Version**: 0.1.0 | **Ratified**: [RATIFICATION_DATE] | **Last Amended**: [LAST_AMENDED_DATE]
