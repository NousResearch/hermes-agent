# Documentation Index

## Delivery Management Stack (new)

1. **Playbook**
   - `docs/playbooks/subagent-driven-delivery-system.md`
   - End-to-end operating model for manager-style subagent development.

2. **Policy**
   - `docs/policies/branch-commit-pr-hygiene.md`
   - Branch, commit, PR, and cleanliness requirements.

3. **Templates**
   - `docs/templates/subagent-task-packet.md`
   - Standard task handoff packet for implementer subagents.
   - `docs/templates/manager-review-gates.md`
   - Mandatory review gates and final merge decision rubric.

4. **Operations Runbook**
   - `docs/runbooks/subagent-manager-operations.md`
   - Practical kickoff command + expected execution behavior.

## Existing Operational Docs
- `docs/runbooks/runtime-overrides-rbac-ops.md`
- `docs/plans/2026-03-12-gateway-final-cohesive-audit.md`

## Quick Start
Tell Hermes:

```text
Start subagent-driven development for <initiative>.
Follow docs/playbooks/subagent-driven-delivery-system.md and enforce docs/policies/branch-commit-pr-hygiene.md.
Create/update plan, execute via subagents, run review gates, and deliver merge-ready summary.
```
