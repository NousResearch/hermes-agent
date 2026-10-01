# Local layout

The active native Hermes home owns the employee's durable files:

```text
$HERMES_HOME/
  documents/          lasting documents, data, deliverables and cloud-doc stubs
  repos/              Git checkouts, one directory per repository
  responsibilities/   owned work, instructions and handoff state
  connections/        service manuals, references and operational helpers
```

These locations are settled. Native home initialization creates `documents/`
and `repos/` alongside its existing directories. The model sees resolved paths
and a pointer to the shipped file-keeping guide in its frozen prompt. It manages
these files through native tools; responsibility packages link to bulk material
in documents. Existing files are not migrated or removed.

Preserve native configuration, credentials, logs, sessions, caches, scratch,
attachment handling, working-directory behavior and managed-home permissions.
There is no new sandbox, temporary directory or attachment-expiry promise.
The Railway working directory remains deployment configuration; the filing
roots do not move when a terminal changes directories.

Per-person memory remains tool-managed under `$HERMES_HOME/memory/people/`,
with its identity registry alongside. Shared memory uses native
`memories/MEMORY.md`. Product guides ship under `guides/` in the installed
repository and use ordinary file reads with native pagination and size limits.

See the [divergence ledger](../downstream/scope.md).
