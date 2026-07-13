# Note Capture Runtime Proposed Patches

This document contains the proposed runtime text changes that should be applied
after approval of the projection contract.

It is intentionally concrete so approval can be based on exact wording.

For the corrected runtime event shape and field semantics, see
[`docs/note-capture-runtime-event-schema.md`](./note-capture-runtime-event-schema.md).

## Runtime Agent Prompt

Use this prompt block when you want Hermes-core runtime surfaces to understand
and consistently apply the capability:

```md
Use the canonical note-capture projection flow for generic captured notes and
files.

Resolve each capture to one canonical target identity before choosing a live
path. `1.Inbox` is a valid explicit canonical target when the routing decision
is to keep the note as raw inbox material. `Resources/_Inbox Review` remains
the low-confidence fallback when the route is uncertain.

Write a canonical capture event first, preserve routing metadata, and stage
one projected artifact per enabled downstream store. Do not treat the live
Obsidian vault, `/opt/second_brain/`, or other user-space folders as the
primary write surface for generic note capture. External sync or projection
workers are responsible for materializing staged artifacts into live targets.

Preserve and propagate routing metadata when available:
- `target_id`
- `target_class`
- `logical_path`
- `display_path`
- `resolved_target_path`
- `target_status`
- `trust_boundary`

Treat daily-note creation as a special-case workflow, not the generic note
capture model.
```

Add this schema rule block where runtime prompts need a compact implementation
guardrail:

```md
For generic note capture, preserve four separate concepts:
- canonical identity (`target_id`)
- resolved live target path (`resolved_target_path`)
- staged artifact path (`staged_relative_path`)
- projection state (`pending` / `projected` / `failed`)

Do not use a folder path as `target_id`.
Do not use a staging path as `resolved_target_path`.
Do not use `staged` as `target_status`.
Use `target_status` only for target lifecycle:
`active`, `deferred`, `pending_migration`, `unavailable`.
```

## 1. `~/HermesData/runtime/hermes-core/skills/note-taking/vault-routing/SKILL.md`

### Proposed replacement header

```md
---
name: vault-routing
description: Vault routing table for the Obsidian second brain — maps content types to canonical destinations and downstream projected paths. Load this skill before saving captured content.
platforms: [linux]
---

# Vault Routing — Content Destination Rules

Load this skill whenever you need to save content related to Neil's Obsidian
second brain.

Hermes does not treat `/opt/second_brain/` as the primary write surface for
generic captured content. The vault is one downstream projected target class.

**CRITICAL:** `/opt/second_brain/` is mounted read-only from this container
over VirtioFS. For generic note capture, Hermes should:

1. decide one canonical routing target
2. render canonical markdown content
3. write a canonical capture event
4. stage one projected artifact per enabled downstream store and target class

External sync outside the Hermes trust boundary then projects those staged
artifacts into the live vault, iCloud-backed Obsidian targets, broader
user-space folders, or targets that are currently deferred in the registry.

Current daily-note workflows remain a special runtime exception and may still
use the dedicated daily-note plugin path until they are migrated explicitly.
```

### Proposed replacement for the final mistake rule

```md
4. ❌ **Don't treat the mounted vault as the generic write surface** — it is
   read-only and downstream of canonical capture. For generic captured notes,
   write canonical capture data and staged projections first.
```

## 2. `~/HermesData/runtime/hermes-core/memories/MEMORY.md`

### Proposed replacement for the vault-routing memory line

Replace:

```md
Vault routing: load vault-routing skill before saving content. YouTube→Content Library, meetings→Interactions, raw→Inbox. Stage at /opt/data/staging/.
```

With:

```md
Vault routing: load vault-routing skill before saving captured content. Hermes
maps content to one canonical target identity, writes canonical capture events,
and stages downstream projections per store. Vault, projection trees, and
non-vault user-space targets are downstream materializations, not the primary
write surface. Check structured projection status rather than assuming a direct
live-path write. `1.Inbox` is a valid explicit target; `Resources/_Inbox Review`
is the low-confidence fallback.
```

## 3. `~/HermesData/runtime/hermes-core/profiles/orchestrator/memories/MEMORY.md`

### Proposed replacement for the write-boundary memory line

Replace:

```md
HERMES_WRITE_SAFE_ROOT=/opt/data blocks Hermes write_file/patch from /opt/second_brain/. Plugin bypasses via Python I/O — workaround: write to /opt/data/ and report.
```

With:

```md
HERMES_WRITE_SAFE_ROOT=/opt/data blocks Hermes write_file/patch from
/opt/second_brain/. Treat `/opt/second_brain/` as one downstream projected
target class for generic note capture. Hermes should write canonical capture
data and staged projections under trusted runtime storage, with external sync
responsible for projecting into the live vault and other approved user-space
targets. The daily-note plugin remains a special-case runtime path.
```

## 4. `~/HermesData/runtime/hermes-core/second-brain-infra/hermes/cron-jobs.md`

### Proposed addition near the top

```md
## Note on Trust Boundary

These cron jobs include a daily-note workflow that still uses a dedicated
plugin-backed runtime write path. That is a special case.

For generic note capture and vault routing, Hermes should use the canonical
note-capture flow: canonical event log first, staged projections second, live
vault, iCloud-backed Obsidian targets, and non-vault user-space targets outside
the Hermes trust boundary.
```

## 5. `~/HermesData/runtime/hermes-core/cron/jobs.json`

### Proposed wording change for `Daily Morning Briefing`

Add after the vault path introduction:

```text
The live vault is one projected surface for most captured notes. Do not treat
generic captured content as direct live-path writes. Daily-note creation
remains a special plugin-backed workflow.
```

### Proposed wording change for `Daily Memo Preparer`

Add after the task description:

```text
This job is a special-case daily-note workflow. It should not be generalized
into the generic note-capture routing path, which uses canonical capture events
plus staged downstream projections into approved target classes.
```

## Approval Questions

Applying the runtime edits above means approving these points:

1. Generic captured notes use canonical event logging plus staged projections.
2. Projected relative paths are derived as
   `<path_prefix>/<resolved-target-path>/<filename>`.
3. The mounted vault, projection trees, and non-vault user-space targets are
   described to Hermes as downstream target classes with distinct trust
   boundaries.
4. Daily-note creation remains an explicit runtime exception for now.
5. Future targets are represented through target registry status values
   (`deferred`, `unavailable`, `pending_migration`) rather than a separate
   target class.
