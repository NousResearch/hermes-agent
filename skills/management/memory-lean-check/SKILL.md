---
name: memory-lean-check
description: Review MEMORY.md and ensure entries use wiki pointers instead of inline details.
---

# Memory Lean Check

Review `$HERMES_HOME/memories/MEMORY.md` and keep it compact.

## Steps

1. Read `$HERMES_HOME/memories/MEMORY.md`
2. Count current entries: split on `§` and count non-empty segments. **Remember this count.**
3. Determine the wiki path: read `wiki.path` from `${HERMES_HOME}/config.yaml` (e.g. `grep wiki.path ${HERMES_HOME}/config.yaml`). Fall back to `${HERMES_HOME}/wiki` if not configured. Never use `$HOME/wiki` — that's shared across profiles and breaks isolation. Read `$WIKI/index.md` to know what wiki pages exist
4. **Validate existing pointers first.** For each entry that contains `see wiki/`:
   - Extract the wiki path after `see wiki/` (up to next space, `.md`, or `(`)
   - Check if the file exists at `$WIKI/<path>`
   - If the file is **missing**: flag it in the report but do NOT delete the entry — the wiki page may have been moved or renamed
   - If the file **exists**: leave the pointer entry alone entirely — do not touch it
5. For each remaining entry (non-pointer), assess:
   - **Does it contain detailed information** (service names, ports, IPs, multi-sentence descriptions, config details, procedural steps) that would be better in a wiki page? → Condense it to a pointer.
   - **Is it a short fact, preference, or correction?** → Leave it alone.
6. When condensing:
   - If a relevant wiki page exists, rewrite as: `Topic: see wiki/path/page.md (brief description).`
   - If no wiki page exists but the detail is valuable, **load the `llm-wiki` skill first**, then create a wiki page following its conventions (frontmatter, cross-references, index.md update, log.md entry, schema-compliant tags). Place it under `$WIKI`.
   - If the detail is trivial/temporary (task progress, TODOs, session outcomes), just remove it entirely.
7. Do NOT condense:
   - Entries that are already pointers (contain `see wiki/`)
   - User preferences and corrections (these belong in memory)
   - Environment facts that are genuinely short (e.g. "Default workspace: /home/josh/workspace/")
   - The convention note itself
8. Save changes only if something was actually condensed or removed.
9. **Post-write integrity check (CRITICAL):**
   - Re-read the file from disk
   - Split on `§` — confirm entry count matches expected count (start count minus any removals)
   - If entry count is wrong, the file is corrupted — rewrite it immediately using the in-memory entries list
   - Report: final entry count, char usage, any broken pointers found

## Rules
- Never delete user preferences, corrections, or stable personal facts
- Never add new entries — this skill only trims/rewrites existing ones
- **§ is the entry delimiter, NOT decoration.** Entries are joined with `\n§\n` between them. Removing a § merges two entries into one and corrupts memory. The § character must appear exactly: once between each pair of entries, never at the start or end.
- After ANY write, verify entry count matches expected. If it doesn't, rewrite from in-memory state.
- Keep memory usage under 30% of the 2,200 char limit
- Be surgical — don't rewrite entries that are fine as-is
- Pointer entries (containing `see wiki/`) are already optimal — never modify them unless the target wiki page no longer exists
