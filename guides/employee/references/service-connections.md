# Service connections and operating knowledge

Use this reference whenever connecting, using, repairing, changing accounts for,
or disconnecting a service. A connection can be a CLI login, direct API, MCP
server, browser dashboard, or several of these for the same service.

## Read before operating

Check `$HERMES_HOME/connections/<service>/manual.md` before operating the
service. The prompt's `Service manuals` listing is a frozen discovery aid, not
proof of access or a complete live directory listing. Check the folder even
when the service is missing from that listing. An existing login may work even
when no manual exists; verify it and document it.

For setup, use the service's official instructions and native authentication.
MCP setup is covered in `native-mcp.md`.
Native credential and secret-store commands are in
`cli-reference.md`. Match the account and
permissions to the user's requested work. Let the user complete authorization
or enter reusable secrets through native administration or their secret manager.
Never put reusable secrets in chat or service manuals. Use native configuration and secret-management commands.

## Document every connection

Every service connection gets an employee-maintained folder:

```text
$HERMES_HOME/connections/<service>/
  manual.md
  references/       # supporting instructions when needed
  scripts/          # reusable operational helpers when needed
```

Use one folder per service. Distinguish accounts, projects and access methods
inside it. Update an existing manual instead of creating a competing guide or
skill. Create supporting files only when they help; a simple connection needs
only `manual.md`.

Before finishing connection setup, record:

- The account, whose it is, project/instance, and what work it serves.
- How access works: CLI commands, API endpoint, native MCP server name, or
  dashboard URL and browser profile behavior.
- Where credentials are managed and the required environment-variable names;
  never token values, passwords, session cookies or copied browser profiles.
- What permissions were granted and what remains unavailable.
- The harmless read-only check performed and its result. An attempted or pending
  login is not a verified connection; label unfinished setup and its next step.
- Any steps, commands or pitfalls the next conversation would otherwise have
  to rediscover.

## Keep reusable work with the connection

Save useful API/CLI helpers in `scripts/` and link them from `manual.md` with
usage, inputs and prerequisites. Read credentials through their native storage
or named environment variables; do not embed them in scripts. Put longer
procedures and provider-specific notes in `references/` and link those too.
This folder holds service operating knowledge; responsibility packages hold
owned duties, scheduling and work state.

After changing or repairing access, update the manual and verify again. Record
new operating knowledge as you use a service, including a previously undocumented
connection. When disconnecting, use the provider's revoke/logout procedure and
native connection management, then mark the manual disconnected. Deleting a
folder does not revoke access. Product guides under `../../` remain
product instructions; these per-service manuals and helpers hold your operating knowledge.
