---
name: dialogue_campaigns
description: Retired legacy dialogue-campaign compatibility shim.
category: crm
prerequisites:
  env_vars: []
  credential_files: []
  commands: []
---

# Legacy dialogue campaigns — retired

This workflow is retained as a non-destructive compatibility marker. Do not
run the former deep-scrape, campaign-orchestrator, inbox-sweep, queue, or sender
scripts. They bypass the canonical account/identity/route boundary and include
manipulative pacing, sensitive-trait inference, or direct delivery patterns
that are not permitted by Communication Core.

Use `$manage-communications` instead:

```text
hermes communication people search <query>
hermes communication timeline show <person-id>
hermes communication analyze conversation <conversation-id>
hermes communication drafts create <person-id> <source-endpoint-id> --text <draft>
hermes communication drafts show <draft-id>
hermes communication drafts cancel <draft-id>
```

The replacement supports explainable commitments, unanswered questions,
non-diagnostic tone evidence, relationship briefs, and user-reviewed drafts.
It does not implement covert persuasion, romantic targeting, autonomous
follow-up, mass outreach, online-status pressure, or production sending.

Historical campaign records are preserved inertly by the Facebook migration
bridge as `legacy_records`; legacy approvals and outbox rows never become live
Core approvals or outbox items. Rollback and record-location details are in
`docs/communication-core/facebook-migration.md`.
