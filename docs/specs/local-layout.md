# Local layout

Use native Hermes profile, configuration, memory, logs, sessions, cache, scratch
and working-directory behavior. There is no prescribed documents/repos tree,
custom temporary directory, hidden config home or two-folder sandbox.
Configuration and credentials follow native access and administration rules.

**Open:** final locations for responsibility packages and connection manuals.
For now existing code still uses `$HERMES_HOME/responsibilities/` and
`$HERMES_HOME/connections/`; this is a provisional location, not a new decision.
Do not move or delete existing user data before that decision.

Per-person memory remains tool-managed under `$HERMES_HOME/memory/people/`,
with its identity registry alongside. Shared memory uses the native
`memories/MEMORY.md` store. Product guides ship under `guides/` in the installed
repository; read them through native file tools. No special read budgets or
write prohibition are added for guides.

Native gateway/service lifecycle, attachment paths and session boundaries remain.
The Railway working directory is deployment configuration, not a global
filesystem convention. See the [scope ledger](../downstream/scope.md).
