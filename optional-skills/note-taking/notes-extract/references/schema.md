# notes-extract data contract

## Structured entry (model → `upsert_entry.py`)

The model reads scanned notes and emits a JSON array of entries for **one**
source note. Scripts never parse semantics — the model produces the `claim`;
the script handles identity, dedup, and merging deterministically.

```json
{
  "entity": { "kind": "person|project", "name": "Jane Doe", "aliases": ["Jane"] },
  "section": "facts",
  "claim":   { "subject": "jane", "predicate": "employer", "object": "acme" },
  "text":    "VP Eng at Acme.",
  "op":      "assert"
}
```

- **section** — person: `facts`, `commitments`, `topics`; project: `ideas`,
  `decisions`, `blockers`, `todos`.
- **claim** — a normalized `{subject, predicate, object}` triple. This is the
  dedup key, *not* the prose. Rewording `text` keeps the same claim → same
  entry, so re-runs don't duplicate. A different source asserting the same claim
  gets its own entry, preserving provenance.
- **op** — `assert` (default) adds/updates the bullet; `retract` moves it to a
  `<section>-archive` region.

## Entry id

`nx-` + `sha256(entity_id, section, claim.subject, claim.predicate, claim.object,
source_id)[:8]`, written as an Obsidian block id at the end of the bullet:

```
- VP Eng at Acme. [[2026-05-20 Standup]] (2026-05-26) ^nx-7f3a9c01
```

## Entity identity

- `entity_id` is derived from the canonical (first-seen) name and stored as
  `id:` in frontmatter.
- The **slug** (filename) is NFKD-transliterated ASCII; pure non-ASCII names
  fall back to a short hash. Names are NFC-normalized first.
- An **alias index** in state maps normalized name/alias → `entity_id`, so
  "Jane", "Jane Doe" route to the same file.
- A slug collision with a *different* entity gets a `-2` suffix and a
  `needs_confirm` flag in the run report — two people are never silently merged.

## Managed regions

Generated bullets live between fences; everything outside is human-owned and
never modified:

```
## Facts
<!-- notes-extract:begin facts -->
- ... ^nx-...
<!-- notes-extract:end facts -->
```

## State (cache, not source of truth)

Stored under `HERMES_HOME/notes-extract/<vault-hash>.json` (never in the vault).
Records per-source sha + emitted entry ids (for change detection and per-source
reconciliation) and the entity/alias index. If deleted, a rescan re-derives the
same content from the `^nx-` ids already present in the files.
