# Recursive scoped issues in Hermes Kanban

Status: proposed architecture with the current implementation described below

## Decision

Hermes Kanban remains the only execution-lifecycle ledger. The existing `tasks` row is the universal issue record, with compatibility preserving “task” in existing APIs. Issues use the closed `kind` vocabulary `individual`, `group`, `company`, `portfolio`, `product`, `project`, `feature`, `task`, or `defect`; migrated rows default to `task`.

Containment and execution dependency are deliberately separate. `tasks.parent_id` is the single recursive containment parent used for breadcrumbs and filtering. `task_links` remains the many-to-many dependency graph that gates dispatcher readiness. `product_id` is structured portable scope identity; it is never inferred from a title, tenant, or board activity.

Each physical board remains the hard dispatch, workspace, attachment, and worker-visibility boundary. No hierarchy operation crosses that boundary and no issue is copied between boards.

## Current implementation

The additive SQLite migration adds `kind`, `parent_id`, and `product_id` plus exact-filter indexes. Existing rows read as root `task` issues. Creation validates the closed kind vocabulary, structured product IDs, parent existence, and the shared 32-edge maximum containment depth. Reparenting is atomic and rejects self-parenting, orphans, ancestry cycles, and moves that would push any descendant over that boundary. Breadcrumb traversal uses the same boundary and fails closed on corrupt cycles, orphans, or over-depth legacy rows rather than presenting truncated ancestry. Deletion refuses to orphan containment children. Archive and review-reopen paths leave containment and dependencies unchanged.

Portable scope IDs deliberately allow `:` as an internal namespace separator (for example, `company:zer0:product:hermes-agent`). This is unambiguous with qualified issue references because scope IDs are opaque task fields, while `<board>:<issue-id>` parsing applies only to issue references and generated issue IDs never contain `:`.

Existing model tools, CLI commands, and dashboard API fields are extended rather than duplicated. `kanban_create`, `kanban_show`, and `kanban_list` create, return, and filter hierarchy fields. CLI create/list/show JSON and dashboard create/list/update expose the same fields. The official Kanban plugin renders kind/product badges, breadcrumbs, containment-child counts separately from dependency counts, exact filters, creation controls, and reparenting.

Qualified `<board>:<issue-id>` references have one concrete read-only DB resolver. It requires explicit qualification, opens only the named board for lookup, and does not expose any cross-board mutation. A cross-board aggregate UI/API remains deferred.

## Compatibility and rollback

The migration is additive and idempotent. Older binaries ignore the new columns and indexes. Rollback is code rollback only; dropping columns would destroy hierarchy identity. Existing callers need not provide new fields and continue to create `task` roots. Existing `task_links` rows are never converted.

## Deferred

Board-level default product identity, cross-board aggregate projections, registry synchronization, arbitrary relation catalogs, multiple containment parents, and board auto-creation are not implemented by this slice. Product IDs must be explicit on issue rows when attribution is required.
