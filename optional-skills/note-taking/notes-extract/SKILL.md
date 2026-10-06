---
name: notes-extract
description: Extract people and project facts from local files.
version: 1.0.0
author: David Murray (thedavidmurray)
license: MIT
metadata:
  hermes:
    tags: [Productivity, Notes, Knowledge]
    related_skills: [obsidian]
---

# Notes Extract Skill

Turn a folder of local markdown notes into structured **People** and
**Idea/Project** files inside a vault: scan notes, extract facts/commitments/
ideas, and write them into per-entity files keyed by a stable claim id so
re-runs never duplicate. It does **not** do general Obsidian CRUD (use the
`obsidian` skill for that), makes **no** network/API calls, and **never**
modifies text you wrote outside its managed fences.

## When to Use

Load this when the user wants to build or refresh a "People DB" / "Project DB"
from their own notes — e.g. "pull commitments and project ideas out of my notes
into per-person and per-project files," or to re-sync those files after adding
notes. For reading, searching, or editing individual notes, use the `obsidian`
skill instead.

## Prerequisites

- A vault directory of `.md` / `.txt` notes. Vault resolution follows the
  `obsidian` skill (see `skills/note-taking/obsidian/SKILL.md`): use
  `OBSIDIAN_VAULT_PATH`, falling back to `~/Documents/Obsidian Vault`. File
  tools do not expand shell variables — resolve the vault to a concrete absolute
  path first, and remember vault paths may contain spaces.
- Optional `NOTES_EXTRACT_SOURCES`: comma-separated **extra** plaintext source
  directories to scan in addition to the vault.
- No API keys, secrets, or external packages — scripts are Python stdlib only,
  and the extraction itself is done by you (the model), not a third-party API.

## How to Run

Invoke the two scripts through the `terminal` tool. `SKILL_DIR` is the directory
containing this SKILL.md.

```bash
# 1. Which notes are new or changed?
python3 SKILL_DIR/scripts/notes_scan.py --vault "/abs/vault path"

# 2. After extracting entries from ONE note, merge them in:
python3 SKILL_DIR/scripts/upsert_entry.py --vault "/abs/vault path" \
    --source-path "/abs/vault path/note.md" --source-link "note" \
    --source-sha "<sha from scan>" --entries-json '[ ... ]'
```

## Quick Reference

| Script | Purpose | Key args |
|---|---|---|
| `notes_scan.py` | List new/changed notes as JSON (read-only) | `--vault`, `--sources`, `--all` |
| `upsert_entry.py` | Merge one note's entries into entity files | `--vault`, `--source-path`, `--source-link`, `--source-sha`, `--entries-json` |

Entry/claim shape, ids, and managed-region format are documented in
`references/schema.md`. Generated files match `templates/person.md` and
`templates/project.md`.

## Procedure

1. Resolve the vault to an absolute path (per Prerequisites).
2. Run `notes_scan.py`. It emits a JSON array of `{path, link, source_id, sha,
   status, text}` for each new/changed note.
3. For **each** scanned note, read its `text` and extract a JSON array of
   entries — one per fact/commitment/topic (person) or idea/decision/blocker/
   todo (project). Each entry carries an `entity`, a `section`, a normalized
   `claim` `{subject, predicate, object}`, the human-readable `text`, and `op`
   (`assert` default, or `retract`). See `references/schema.md`.
4. Run `upsert_entry.py` once per note, passing that note's entries plus its
   `source-path`, `source-link`, and `source-sha`. The script resolves the
   entity, writes bullets inside managed fences, dedupes by claim id, and
   reconciles against the note's previous run (stale entries are removed).
5. Report the per-file actions and any `needs_confirm` slug collisions from the
   script's JSON output. The model emits the `claim`; the script owns all
   identity, dedup, and merge logic — never hand-write merge logic.

To verify or repair a person/project file, read it with `read_file`; to make a
manual correction, edit **outside** the fenced regions with `patch` — the skill
will preserve it. Use `search_files` to locate generated files under `People/`
and `Ideas-Projects/`.

## Pitfalls

- **Dedup is claim-keyed, best-effort.** Two entries with the same
  `{subject, predicate, object}` collapse to one id; if the model emits
  inconsistent claims for the same fact, near-duplicates can still appear. Keep
  `claim` fields normalized and lowercase.
- **No automatic contradiction handling.** A later note that contradicts an
  earlier fact does not auto-retract it. Emit an explicit `"op": "retract"`
  entry to move the old bullet to a `<section>-archive` region.
- **Same-name people are not merged blindly.** A slug collision with a different
  entity creates `<slug>-2.md` and reports `needs_confirm`; confirm with the
  user before treating them as one person.
- **Only fenced content is managed.** Anything outside the
  `notes-extract:begin/end` fences is yours; the skill never edits or reorders
  it. State lives under `HERMES_HOME`, never in the vault.
- **Out of scope:** Apple Notes, always-on recording devices, and database
  backends are not handled here.

## Verification

Scanning twice with no new notes, and re-running an upsert with unchanged
entries, must both be no-ops:

```bash
python3 SKILL_DIR/scripts/notes_scan.py --vault "/abs/vault path"
# → after a full run, a second scan returns [] and a repeated upsert reports
#   every file "unchanged" (no diff).
```
