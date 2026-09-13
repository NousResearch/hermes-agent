---
name: llm-wiki-native
description: "Scan, query, and record changesets via the native wiki API."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [wiki, knowledge-base, research, notes, markdown, native-api, changesets]
    category: research
    related_skills: [llm-wiki, obsidian, arxiv]
---

# LLM Wiki (Native API)

Build and maintain [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
— a persistent, compounding knowledge base of interlinked markdown files — using
Hermes' **native wiki API** instead of raw filesystem walks.

This is the API-native companion to the [`llm-wiki`](../llm-wiki/SKILL.md) skill.
The conventions (three layers, frontmatter, taxonomy, page thresholds, update
policy) are identical — read that skill for the full curation philosophy. **This
skill changes only the mechanics: how you orient, search, and — critically — how
you record what changed.**

## Why use this instead of plain filesystem ops

The Hermes gateway exposes the wiki through `tui_gateway/wiki_api.py`, and the
desktop/native app renders it: a graph view (`wiki.scan`), page detail
(`wiki.page`), taxonomy filtering, and a **Timeline tab** (`wiki.changesets`).

Those views only reflect your work if changes go through the native code path.
In particular, **the Timeline is empty unless you capture a changeset after every
write.** Editing files directly with `write_file` keeps the markdown correct but
leaves the app's timeline blank and its graph stale until the next manual rescan.

This skill drives the same code path the app reads:
- **Read/orient** through `wiki scan` / `wiki page` / `wiki changesets` — one call
  returns the structured graph or timeline, no manual link-walking.
- **Record** every create/update/archive/delete through `wiki capture`, which
  appends to `<wiki>/changesets/index.json` (and commits to git when available) —
  so `wiki.changesets` and the app's Timeline populate in real time.

## When to Use

Prefer this skill over `llm-wiki` when **the user runs the Hermes desktop/native
app** (or any client that renders `wiki.scan` / `wiki.changesets`) and wants the
graph and timeline to stay live. Otherwise the two are interchangeable. Activates when the user:

- Asks to ingest a source, file a query, or lint a wiki **and** uses the native app
- Asks "what changed in the wiki?", "show the timeline", or about recent wiki activity
- References their wiki/graph/timeline in the desktop app

## Prerequisites

- **Wiki location** — resolved exactly like the gateway: a name via
  `~/.hermes/wikis.yaml`, else `$WIKI_PATH`, else `~/wiki`. Pass `--wiki NAME` to
  target a specific registered wiki; omit it for the default.
- **Helper modules** — the CLI imports `tui_gateway/wiki_api.py` and
  `scripts/wiki_changeset.py`. It finds them automatically in an installed Hermes
  (`~/.hermes/scripts`) or a repo checkout. No third-party packages.
- **Git (optional)** — if the wiki dir is a git repo, `capture` records a commit
  hash per change. Without git it still records changesets (empty `git_commit`).

## How to Run

All operations go through the bundled CLI via the `terminal` tool. From the skill
directory (`~/.hermes/skills/research/llm-wiki-native/`):

```bash
python3 scripts/wiki.py <command> [args] [--wiki NAME]
```

## Quick Reference

| Goal | Command |
|------|---------|
| Graph (pages + links) | `python3 scripts/wiki.py scan` |
| Read one page | `python3 scripts/wiki.py page entities/llama-cpp.md` |
| Valid taxonomy paths | `python3 scripts/wiki.py taxonomy` |
| Expand a page's integration links | `python3 scripts/wiki.py expand llama-cpp` |
| Recent timeline (newest first) | `python3 scripts/wiki.py changesets --limit 20` |
| One page's history | `python3 scripts/wiki.py changesets --page entities/llama-cpp.md` |
| Creates this week | `python3 scripts/wiki.py changesets --action create --since 2026-06-22T00:00:00Z` |
| Event log (what caused updates) | `python3 scripts/wiki.py events --limit 20` |
| Sources that produced nothing yet | `python3 scripts/wiki.py events --json` → events with empty `changesets` |
| **Record a change** | `python3 scripts/wiki.py capture entities/llama-cpp.md update "Added speculative decoding benchmarks" --trigger ingest --source-event raw/articles/src.md` |

`--json` on `scan` / `page` / `changesets` / `events` returns raw JSON for
programmatic use. `capture` actions: `create` · `update` · `archive` · `delete`.
Triggers are conventionally `ingest` · `query` · `lint` · `process-inbox` ·
`manual`, but the set is **open**: what an event kind is gets declared by a
`type: event-type` wiki page, so a new ingestion source is a page you write,
not a code change.

## Provenance — always declare what caused a change

`capture` takes `--source-event RAW_PATH`, repeatable, naming each event that
caused the change. **Declare it on every capture that has one.** A synthesis
drawing on three sources passes it three times.

Omitting it records provenance as `unknown`. That is the honest answer for a
hand edit with no ingestion behind it, and a real gap for anything else —
`unknown` is never inferred or backfilled, precisely so that its presence
means something. The value of the whole log depends on `unknown` staying rare:
it should only ever describe changes that predate this field.

Two things follow that are worth using:

- `python3 scripts/wiki.py events` lists sources with **no** resulting
  changesets — ingested material nobody has synthesized from yet. That's a
  work queue, not an error.
- A page's `sources:` frontmatter is page-level provenance and is now returned
  by `scan`. Keep it in sync with what you capture.

## Procedure

### Orient (every session — do this first)

The native scan replaces manual SCHEMA/index/log reading for structure, but you
still read `SCHEMA.md` for conventions:

```bash
python3 scripts/wiki.py scan                       # what pages/links exist
read_file "$WIKI/SCHEMA.md"                         # domain conventions + taxonomy
python3 scripts/wiki.py changesets --limit 30       # what changed recently
```

This prevents duplicate pages and missed cross-references — the same goal as the
`llm-wiki` orientation, fewer reads.

### Ingest a source

1. Capture the raw source to `raw/` (`web_extract` → `write_file`), with raw
   frontmatter (`source_url`, `ingested`, `sha256`) — see `llm-wiki` for the format.
2. `python3 scripts/wiki.py scan` and `search_files` to find existing pages for
   the entities/concepts mentioned.
3. Write/update pages with `write_file`, following SCHEMA.md (frontmatter,
   `[[wikilinks]]` ≥ 2, taxonomy tags, page thresholds, update policy).
4. **Record each change** — once per page touched:
   ```bash
   python3 scripts/wiki.py capture entities/llama-cpp.md update \
     "Added b4820 speculative-decoding benchmarks" --trigger ingest \
     --source-event raw/articles/llama-cpp-release.md
   ```
5. Update `index.md` and `log.md` as usual. Report every file created/updated.

> A single ingest commonly touches 5–15 pages. Capture a changeset for **each** —
> that's what fills the app's timeline and keeps the graph current.

### Query

1. `python3 scripts/wiki.py scan` (and `search_files` on large wikis) to find
   relevant pages; `python3 scripts/wiki.py page <path>` to read them.
2. Synthesize, citing `[[pages]]`.
3. If the answer is worth keeping, write it to `queries/` or `comparisons/` and
   `capture … create … --trigger query`.

### Lint

Run the `llm-wiki` lint checks (orphans, broken links, frontmatter, contradictions,
stale pages). Two are easier here:
- **Orphans / broken links:** `python3 scripts/wiki.py scan --json` gives the full
  page set and resolved links — diff link targets against page ids in one pass.
- **Recent activity / log rotation:** `python3 scripts/wiki.py changesets` is the
  authoritative history.

Record fixes you make with `capture … --trigger lint`.

## Pitfalls

- **Capture after writing, not before.** `capture` hashes the page's current
  on-disk state, so write the file first, then capture. (`delete` is the exception
  — capture after removing the file.)
- **One capture per page, per logical change.** Don't capture the same file twice
  for one edit; do capture each distinct page an ingest touches.
- **Don't invent provenance.** If you don't know which source caused a change,
  leave `--source-event` off and let it record `unknown`. Naming a plausible
  source you didn't actually read is worse than admitting the gap — it makes
  every other recorded provenance untrustworthy.
- **`capture` doesn't write content.** It records a changeset for a page you wrote
  with `write_file`. It is not a substitute for writing the markdown.
- **Never modify `raw/`.** Sources are immutable; corrections go in wiki pages.
- **`--wiki NAME` must match `wikis.yaml`** (or be a path). A name the gateway
  can't resolve falls back to `$WIKI_PATH`/`~/wiki` — verify with `scan` first.
- **Everything else is the `llm-wiki` skill.** Frontmatter, taxonomy discipline,
  page thresholds, cross-reference minimums, and the contradiction/update policy
  are unchanged — follow them.

## Verification

After an operation, confirm it landed in the native path:

```bash
python3 scripts/wiki.py changesets --limit 5     # your change appears, newest first
python3 scripts/wiki.py scan | head              # new pages/links present
```

In the desktop app, the **Wiki → Timeline** tab should now show the changes, and
the graph should include any new pages/links.
