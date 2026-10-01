# Connections

Use this guide when connecting, using, repairing, changing accounts for, or
disconnecting a service. Connections include CLIs, APIs, databases, MCP servers
and browser dashboards.
For normal work with a connected service, read its
`$HERMES_HOME/connections/<service>/manual.md` first.

## Choose the setup instructions

- Google Workspace: `google-workspace/guide.md` — Google authorization and the
  bundled API helpers.
- Email: `email/guide.md` — Himalaya mailbox setup and IMAP/SMTP operation.
- GitHub: `github/guide.md` — Git/gh authentication and remote operations.
- MCP servers: `../employee/references/native-mcp.md` — native Hermes MCP setup.
- Other services: use their official setup instructions and native CLI, API or
  browser authentication. Native credential administration is documented in
  `../employee/references/cli-reference.md`.

These are setup references, not a list of connected accounts. Check existing
manuals and configured access before creating another login. Use the account
and permissions needed for the requested work. Let the user complete sign-in
or provide secrets through native credential administration; do not put reusable secret
values in chat, guides, manuals or scripts.

Verify access with a harmless read and confirm the account and resource it
reaches. Record unfinished setup as pending; a saved configuration is not proof
that access works.

## When no bundled guide covers the service

A bundled guide is optional. Start with existing configured access and the
service's official documentation. Choose the supported access method that fits
the requested work: an existing or official CLI/client, direct API/database
access, native MCP, or browser login. Do not invent an authentication flow.

An API key or database URL alone may not identify the endpoint, account,
project, database or allowed work. Infer these from existing configuration and
official docs where possible; ask only for missing information needed to
connect. Have the user provision reusable secrets through native administration
or their secret manager, and refer to them by source name.

Complete setup using the official instructions, verify with a harmless read,
then document the working access patterns below. You do not need to create a
new setup guide. If access is blocked, record the missing step and continue
when it is available; do not claim the service is connected.

## Leave operating knowledge, not another setup guide

After access works, create or update:

```text
$HERMES_HOME/connections/<service>/
  manual.md
  references/   # longer operating recipes, when needed
  scripts/      # reusable service helpers, when needed
```

Copy and adapt the relevant **access and usage patterns** from the setup guide
into the manual: working commands or API calls, account/project selection,
resource identifiers, tool/server names, helper paths and credential-source
names without values. Include the verification performed and any known scope
limits. Keep enough to perform normal work without repeating setup discovery.

Do not copy installation steps, OAuth walkthroughs, token-creation instructions
or connection-request procedures. Link to the setup guide for reconnecting.
An access pattern is “use this configured account and helper to read the
calendar,” not “create an OAuth client and authorize it.”

Link reusable helpers and longer recipes from the manual. Reuse shipped helpers
when suitable; copy or create a local helper when it needs service-specific
adaptation. Never embed secrets. Avoid disconnected files or duplicate manuals;
distinguish multiple accounts within the same service folder.

Update operating patterns when real usage reveals missing steps or pitfalls.
Responsibilities link to these manuals and retain duties, authority and work
state; service manuals describe how to operate the service.

After access changes, verify and update the manual. On disconnect, revoke access
through the service's native mechanism and mark the manual disconnected.
Deleting a manual does not revoke access.
