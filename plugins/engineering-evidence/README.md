# engineering-evidence

Bounded Hermes tooling for exact-HEAD engineering evidence.

The plugin provides four diagnostic-only surfaces:

- `snapshot` records tracked files and Python imports without editing a repository.
- `test-receipt` records test discovery and preserves unavailable/degraded results.
- `experience` stores a sanitized candidate in the configured learning Vault.
- `experience-promote` writes a separate validator record; it never mutates the candidate.
- `experience-search` retrieves only explicitly validated candidates at the exact repository HEAD.
- `contract` exposes typed receipt ownership and event names for Hermes routing.
- `kanban-attach` and `kanban-receipts` publish/read receipts through Hermes's existing Kanban attachment layer.
- `kanban-workflow` explicitly creates four bounded child lanes with existing Hermes assignees; it is never automatic fan-out.
- `validate` checks schema, exact commit identity, and authority fields.

Receipts do not grant merge, release, credential, trading, or deployment authority. They
must be independently reviewed before promotion into shared learning or any completion gate.
