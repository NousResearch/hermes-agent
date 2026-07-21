# Ultraresearch Phase 0 — goal-learning freshness and proposal integrity

## Core question

Which minimal, evidence-backed change lets Hermes reuse explicitly approved
goal-learning proposals without allowing stale proposals to overwrite current
memory or skill state?

## Axes

1. **Memory freshness and atomicity** — trace stage and apply paths, their
   existing locks, and the smallest snapshot contract that prevents a
   check-then-use race.
2. **Skill proposal integrity** — establish whether one skill mutation can be
   safely covered now, including reviewed-artifact and stale-state semantics.
3. **Approval queue operation** — trace claim, retry, cancellation and
   user-facing outcomes so stale records never silently requeue or apply.
4. **AI-agent learning boundary** — validate against primary documentation
   that approved learning stays explicit, observable and outside prompt-memory
   auto-injection.

## Scope and success criteria

- Reuse existing verified-goal, outcome-receipt and pending-approval systems;
  do not create a competing task or generic agent database.
- Research is read-only until one bounded implementation is supported by the
  source and direct tests.
- A chosen feature must fail closed for legacy or malformed proposals, prevent
  stale live-state mutation, record an auditable outcome, and have focused
  executed tests before local-main integration.
