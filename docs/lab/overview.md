# IT Automation Lab Overview

## Purpose

This checkout can be used as an IT automation laboratory for designing, documenting, testing, and safely running operational workflows with Hermes Agent.

The lab is intended for controlled experiments such as:

- system inventory collection
- service health checks
- log triage
- backup verification
- scheduled maintenance reports
- gateway/bot-driven operations
- GitHub issue, PR, and CI automation
- script prototyping before production rollout

## Non-goals

This lab is not a license to run destructive commands without review. It should not be used to modify production hosts, delete data, rotate secrets, restart critical services, or change network/firewall state without explicit human approval and a documented rollback path.

## Operating Model

1. Describe the operational procedure as a runbook in `docs/runbooks/`.
2. Add or update scripts only after the runbook is clear.
3. Prefer read-only discovery first.
4. Add `--dry-run` for actions that change state.
5. Verify results with commands, tests, or captured output.
6. Summarize the change and wait for human approval before committing or pushing unless approval has already been granted.

## Repository Relationship

The repository remains the Hermes Agent codebase. The lab documentation and scripts are an additional operating layer that helps agents and humans run IT automation work consistently without disrupting core development workflows.
