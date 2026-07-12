# Note Capture Runtime Proposed Patches

This document contains the proposed runtime text changes that should be applied
after approval of the projection contract.

It is intentionally concrete so approval can be based on exact wording.

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
live-path write.
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
