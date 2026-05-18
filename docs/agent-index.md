# Agent Index

Index of active plans and agent-facing documentation.

## Active Plans

Plans in `docs/plans-active/` are works in progress — specifications, architecture
designs, and implementation plans that have been started but not yet completed.

| Plan | Description |
|------|-------------|
| [telegram-dm-user-managed-multisession-topics](plans-active/2026-05-02-telegram-dm-user-managed-multisession-topics.md) | Telegram DM multi-session mode via user-created topics |

## Plan Lifecycle

See [Session Workflow](session-workflow.md) for the full plan lifecycle and how
specifiers, architects, and implementors interact with plans.

**Quick reference:**

1. **Specifiers** write new plans into `docs/plans-active/` and add an entry to the
   Active Plans table above.
2. **Implementors** move completed plans to `docs/plans-complete/`, prefixed with a
   sortable completion date (`YYYY-MM-DD-`), and remove the entry from the Active
   Plans table.
3. **Implementors** annotate changes to plans using `[CHANGED]` and `[DOWNSTREAM]`
   markers rather than overwriting original content.
