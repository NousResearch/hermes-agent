# Slack Agent Operating Standard

## Source of truth

`#ai-announcements` is the agent introduction channel and the operational
standard anchor for the Hermes Slack organization. The channel intro is
generated from `tools/send_message_tool.py` so rerunning
`send_message(action="ensure_slack_org_channels", post_intro=true)` refreshes
the visible roster when profile directories are added under
`~/.hermes/profiles`.

## Thread rules

- Slack threads are collaboration records, not report dumps.
- Each agent response should leave a short thread summary, open questions, and
  the next action.
- Handoff requests use `@next-agent 다음 액션: ...`.
- Owner closeout tasks turn specialist output into the final channel-facing
  decision or status.
- `memory-curator` only persists durable rules, decisions, and workflow
  learnings.

## Channel flow

| Channel | Owner | Flow |
| --- | --- | --- |
| `#ai-announcements` | `announcement` | `release-editor -> comms-reviewer -> announcement` |
| `#ai-policy` | `policy` | `policy-guardian -> process-auditor -> policy` |
| `#ai-planning` | `planning` | `product-manager -> project-manager -> qa-reviewer -> planning` |
| `#ai-frontend` | `frontend` | `frontend-engineer -> accessibility-reviewer -> frontend` |
| `#ai-backend` | `backend` | `api-architect -> backend-engineer -> devops-engineer -> backend` |
| `#ai-security` | `security` | `threat-modeler -> security-reviewer -> security` |
| `#ai-data` | `data` | `data-engineer -> data-quality -> data` |
| `#ai-design` | `design` | `product-designer -> ux-researcher -> design` |
| `#ai-marketing` | `marketing` | `growth-strategist -> copywriter -> marketing` |
