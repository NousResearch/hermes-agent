---
name: manage-communications
description: Use for safe, account-scoped communication workflows.
category: manage-communications
prerequisites:
  env_vars: []
  credential_files: []
  commands: []
---

# Manage Communications

Use `hermes communication` to inspect contacts, sync explicit accounts, explain
routes, analyze relationship evidence, and prepare drafts. Never query the
Communication Core SQLite database directly and never call a legacy sender.

## Safety contract

- Begin with read-only inspection. Do not initialize storage unless the user
  asks to set up Communication Core.
- Require exact `connected_account_id`, `person_id`, and endpoint IDs. Never
  infer a default account or choose among accounts by name, activity, or health.
- Treat Telegram Communication and Telegram News as separate domains.
- A channel change requires a directed account allowlist plus a per-person
  route. Run `routes dry-run` before `routes set` and report its explanation.
- Outbound work stops at draft and exact approval. This CLI has no production
  send command. Never bypass approval or activate an outbox worker.
- Do not expose credential references, browser profiles, raw private messages
  in logs, or private correspondence to a public-news workflow.
- Dating-site work requires a user-confirmed named pilot and test account.

## Workflow

1. Inspect accounts with `hermes communication accounts list`, then use
   `accounts show`, `status`, or `capabilities` for the exact selected ID.
2. Run `sync status` before `sync run`. Use `sync retry` only for that same
   account; surface partial runs and redacted issues.
3. Resolve a canonical person with `people search` and `people show`. Treat
   duplicate suggestions as candidates; `merge` requires explicit human
   evidence and remains reversible with `unmerge`.
4. For cross-channel work, inspect and dry-run the exact route. Apply it only
   after explicit user choice of the target account and endpoint.
5. Use timeline, analysis, brief, group, segment, or greeting reads as needed.
   Explain evidence and provenance; do not present tone markers as psychology.
6. Create a draft only after showing its person, source account/endpoint,
   target account/endpoint, recipient preview, and exact text. Approval is
   bound to that snapshot and TTL; any mutation requires a new approval.

## References

- Accounts, sync, and routes: [accounts-routing.md](references/accounts-routing.md)
- Identity and channel continuity: [identity-journey.md](references/identity-journey.md)
- Analysis and CRM: [analysis-crm.md](references/analysis-crm.md)
- Groups and greetings: [groups-greetings.md](references/groups-greetings.md)
- Draft and approval safety: [approvals-safety.md](references/approvals-safety.md)
- Adapter boundaries: [adapters.md](references/adapters.md)
