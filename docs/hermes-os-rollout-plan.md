# Hermes OS Runtime Delegation Rollout Plan

## Stage 0 - Disabled

Hermes OS runs normally. Official Hermes Agent can be installed and tested through `hermes-agent`, but Hermes OS does not delegate work to it.

## Stage 1 - Dry Run

Hermes OS builds delegation requests and validates responses, but the official runtime is not invoked.

Success criteria:

- Request validation succeeds.
- Agent selection is correct.
- Results are stored in Hermes OS dry-run output records.

## Stage 2 - One Project, One Agent

Enable runtime delegation for one low-risk project and one agent kind, preferably documentation or research.

Success criteria:

- No command conflicts.
- Runtime failures degrade safely.
- Hermes OS stores final output.

## Stage 3 - Selected Agents

Enable research, coding, testing, review, documentation, and deployment agents selectively.

Success criteria:

- Runtime health is visible.
- Failures have clear retry/escalation behavior.
- Tool permissions are enforced by Hermes OS.

## Stage 4 - Long-Running Workflows

Enable checkpointed workflows such as research to validation to review to report.

Success criteria:

- Workflow resumes from checkpoints after interruption.
- Partial outputs are persisted in Hermes OS.
- Runtime memory is not treated as source of truth.

## Rollback

Disable runtime delegation and return to direct Hermes OS execution. Existing project state remains intact because Hermes OS never relinquishes source-of-truth ownership.
