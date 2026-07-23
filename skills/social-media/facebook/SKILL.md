---
name: facebook
description: Facebook read-only Communication Core workflows.
category: social-media
prerequisites:
  env_vars: []
  credential_files: []
  commands: []
---

# Facebook communication shim

This skill is a platform-specific entry point. Use `$manage-communications`
for the shared account, identity, route, timeline, analysis, draft, approval,
and greeting workflow. Do not query SQLite or invoke legacy browser/sender
scripts from this skill.

Facebook reads are exposed through the account-explicit
`FacebookCommunicationAdapter`, which wraps the existing verified Facebook CRM
repository. Browser/E2EE synchronization remains owned by the existing
Facebook application service and persistent `facebook` browser profile; there
is no second Facebook stack.

Use only documented commands such as:

```text
hermes communication accounts status <connected-account-id>
hermes communication accounts capabilities <connected-account-id>
hermes communication sync run <connected-account-id> --mode incremental
hermes communication people search <query>
hermes communication people show <person-id>
hermes communication timeline show <person-id>
hermes communication analyze conversation <conversation-id>
```

Safety invariants:

- Require an exact connected account; never infer a default Facebook account.
- Keep `facebook_settings.write_actions_enabled=0`.
- Treat a missing database, disabled account, unsupported capability, or
  re-auth state as a fail-closed result; never fall back to a generic browser
  task.
- External sending is outside this skill. A requested reply becomes a Core
  draft and exact approval only; production outbox workers remain disabled.
- Redact credential refs, browser profile refs, message bodies, and other PII
  unless the selected read contract explicitly allows them.

Legacy `facebook_api.py` and sibling diagnostic scripts remain present only for
rollback/forensics. New workflows use `hermes communication`; see
`docs/communication-core/facebook-migration.md` for the inventory and cutover.
