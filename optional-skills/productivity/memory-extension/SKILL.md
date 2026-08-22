---
name: memory-extension
description: "Extend Hermes memory: index + on-demand detail files."
version: 1.2.0
author: Franck Iribaren (misstyka), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [memory, organization, persistence, markdown, index]
    related_skills: [obsidian]
    category: productivity
---

# Memory Extension Skill

Extends Hermes memory (`MEMORY.md` / `USER.md`) into an **index + detail files** architecture: the injected memory files stay tiny (just titles), while full details live in `memories/extended/*.md`, read on demand. Solves the "memory char limit" problem without losing information. It does not replace the `memory` tool — it organizes what the tool stores so the injected index stays under quota.

## When to Use

- **Memory is near its char limit** (`memory_char_limit` / `user_char_limit`) and you need more room.
- **A memory entry is long** (environment facts, workflows, model setups, project details) and you want it fully preserved but not injected into every message.
- **You keep losing details** because memory entries get truncated or compressed.
- **You want to organize memory by topic** (identity, projects, tools, preferences) instead of one flat list.
- Works with any Hermes profile.

**Don't use it yet when:** the index is under ~80% of its quota AND no entry exceeds ~500 chars. The structural cost (README, files, read-on-demand discipline) only pays off past those thresholds.

## Prerequisites

- No external dependencies. The consistency script needs `bash` (git-bash/MSYS on Windows, default on macOS/Linux).
- `$HERMES_HOME` resolves to the active profile's home (default `~/.hermes`). The script falls back to `$HOME/.hermes` when the variable is unset.

## How to Run

```bash
# 1. Create the extended dir + seed the guide
mkdir -p "$HERMES_HOME/memories/extended"
cp SKILL.md "$HERMES_HOME/memories/extended/README.md"
# Note: the copied README's relative references (references/…, scripts/…) are
# skill-dir-relative and do not resolve from memories/extended/ — treat them
# as pointers back to the skill directory.

# 2. Install the consistency script (recommended)
mkdir -p "$HERMES_HOME/scripts"
cp scripts/check-memory-coherence.sh "$HERMES_HOME/scripts/"

# 3. Rewrite MEMORY.md with title-only entries → see extended/<file>.md
# 4. Verify index ↔ files coherence
bash "$HERMES_HOME/scripts/check-memory-coherence.sh" "$HERMES_HOME"
```

## Quick Reference

| Concept | Location |
|---|---|
| Index (injected every message) | `$HERMES_HOME/memories/MEMORY.md`, `USER.md` |
| Detail (read on demand) | `$HERMES_HOME/memories/extended/*.md` |
| System guide (copy of this file) | `memories/extended/README.md` |
| Coherence check | `bash $HERMES_HOME/scripts/check-memory-coherence.sh "$HERMES_HOME"` |

Index line format: `Topic — one-line hint → see extended/<file>.md`

## Procedure

1. **Read before answering.** When a `MEMORY.md` / `USER.md` entry has `→ see extended/<file>.md` and the topic is relevant to the current conversation, `read_file` the detail file BEFORE replying. Never answer from the title alone.
2. **Route new details to the file.** When you learn a new detail about an externalized topic, patch the `extended/` file — not the index. The index keeps only title + pointer.
3. **Create new large entries as files.** Write `memories/extended/<name>.md`, then add the one-line index pointer via the `memory` tool (respects locks); fall back to `write_file` only if the tool fails.
4. **Never delete a detail file** without removing/updating its index line.
5. **Share files across memories when useful** (e.g. one `google-cloud.md` used by both `MEMORY.md` and `USER.md`).
6. **Use accent-free, space-free filenames** (`ecriture.md`, not `écriture.md`) — accents break under MSYS/Windows.
7. **When the index saturates**, consolidate: merge same-theme short entries into one `extended/<theme>.md` (or `divers.md`), replace N lines with 1, and audit weekly with the coherence script.

## Pitfalls

- **Index saturation is the #1 failure.** Externalization moves the bottleneck: short entries still accumulate in the index. Route every new detail to `extended/` and consolidate when the index approaches ~100% of quota (this happened in real testing: `MEMORY.md` hit 100% after 4 days).
- **Answering from the title alone** defeats the system. The title is in context; the detail is behind a file. If you don't read it, the system adds friction without value.
- **Local models (< 10B) hallucinate file state.** They invent files that don't exist, mix index lines into detail files, and claim reads that never happened. Read `references/local-model-pitfalls.md` before any modification with a weak model.
- **Index ↔ file drift is silent.** Without the coherence script, a renamed file with an un-updated index (or vice versa) breaks silently. Run the script periodically.
- **Filename accents break paths.** Always use ASCII filenames in `extended/`.
- **The script needs bash.** On Windows it runs under git-bash/MSYS (Hermes' default terminal shell); it is not a PowerShell script.

## Verification

```bash
# Coherence: 0 orphans, 0 dangling references, all files referenced
bash "$HERMES_HOME/scripts/check-memory-coherence.sh" "$HERMES_HOME"
# Expected: "OK: N extended/ file(s), all referenced"
```

Manual checklist:
- [ ] Every `extended/*.md` (except `README.md`) is referenced in `MEMORY.md` or `USER.md`
- [ ] Every `→ see extended/<file>.md` line points to an existing file
- [ ] No long detail inline in the index
- [ ] New topics routed to `extended/` (rule 7)

## References

- `references/local-model-pitfalls.md` — known bugs of local models (< 10B) with this system; safety rules to avoid hallucinated file state.
- `scripts/check-memory-coherence.sh` — automatic orphan + dangling-reference detection.
