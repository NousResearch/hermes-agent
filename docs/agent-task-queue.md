# Agent task queue workflow

This repository can use GitHub Issues as a lightweight task queue for agent-assisted work.

## Default routing

1. **Librarian researches**
   - Gather source links, logs, related issues, prior decisions, and constraints.
   - Leave a concise evidence brief in the issue.
   - Do not implement code from the research lane.

2. **Hero8 architects / plans**
   - Convert research into a concrete implementation issue.
   - Add acceptance criteria, risk level, approval gates, likely files, and stop/ask conditions.
   - Mark the issue ready for a coding agent only when the task is independently actionable.

3. **Hero7 or Copilot implements**
   - Create a feature branch.
   - Make the smallest vertical-slice change that satisfies the issue.
   - Run targeted checks.
   - Open a PR and link it to the issue.
   - Comment with changed files, tests run, failures, and remaining risks.

4. **Hero8 reviews**
   - Review the diff against the acceptance criteria.
   - Check tests and risk/approval gates.
   - Request changes or mark ready for human approval.
   - Do not merge, release, deploy, or change repo settings without explicit approval.

5. **GuardianCOO audits when needed**
   - Use for security, privacy, production-impacting, or safety-sensitive changes.
   - Provide evidence and a recommendation, not unchecked implementation changes.

## Label taxonomy

Recommended labels for the agent queue:

### Lane

- `lane/librarian`
- `lane/architect`
- `lane/coder`
- `lane/reviewer`
- `lane/guardian`

### Status

- `status/researching`
- `status/needs-plan`
- `status/ready-for-coder`
- `status/in-progress`
- `status/pr-open`
- `status/reviewing`
- `status/changes-requested`
- `status/ready-for-approval`
- `status/done`

### Agent

- `agent/librarian`
- `agent/hero8`
- `agent/hero7`
- `agent/guardiancoo`
- `agent/copilot`

### Risk and approval

- `risk/low`
- `risk/medium`
- `risk/high`
- `approval/required`
- `production-impacting`
- `safety-sensitive`

### Type

Use the existing `type/*` labels where possible. Add `type/research` for evidence-only work.

## Issue quality bar

An agent-ready issue should include:

- Objective
- Context and evidence
- Acceptance criteria
- Likely files or areas, if known
- Out of scope
- Risk level
- Approval gates
- Handoff instructions with required proof

If those fields are missing, keep the issue in `status/needs-plan` instead of handing it to a coding agent.

## Work log standard

Every agent comment should include:

- Agent/lane
- Action taken
- Evidence or links
- Tests/checks run
- Remaining risk
- Next requested action

## Merge and release gates

Explicit human approval is required before:

- Merging to the default or protected branch
- Publishing releases, packages, images, or docs deployments
- Dispatching production-impacting workflows
- Changing secrets, permissions, branch protection, webhooks, or repo settings
- Taking action that affects safety-sensitive, clinical, education, privacy, or vulnerable-user workflows
