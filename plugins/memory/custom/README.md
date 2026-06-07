# Custom Memory Provider

Connect Hermes to a knowledge store **you already maintain** — an LLM wiki,
"second brain", or Obsidian-style markdown vault — instead of a hosted memory
service. Hermes recalls relevant notes before each turn and, because it is
self-improving, writes new memories back so future sessions can use them.

This version ships the **`files`** backend. A pluggable seam is in place for a
future **`http`** backend (recall/write against your own API endpoint).

## Setup

```bash
hermes memory setup   # choose "custom", then point it at your vault
```

Or edit `~/.hermes/config.yaml` directly:

```yaml
memory:
  provider: custom
  custom:
    mode: files                      # "files" (this version); "http" planned
    dir: "/path/to/obsidian-vault"   # your LLM wiki / second brain root
    write_subdir: "hermes-memory"    # where Hermes writes new notes
    write_format: markdown           # "markdown" | "jsonl"
    max_results: 5                   # notes injected per recall
    # read_globs: ["*.md", "*.txt"]  # which files recall scans
```

`dir` may be a local path or an NFS/SMB-mounted share. It can be `~`-expanded.

## How it works

- **Recall** (`prefetch`, `memory_search` tool): a dependency-free keyword scan
  over `read_globs` files under `dir`, returning the most relevant note
  snippets (file name + matching lines). No absolute paths are ever exposed to
  the model.
- **Write** (`sync_turn`, `on_memory_write`, `memory_add` tool): new memories
  are appended under `write_subdir`, on a background thread so turns never block.

## `write_format`

| Format | Stored as | Recalled later? | Best for |
|--------|-----------|-----------------|----------|
| `markdown` (default) | `session-<id>.md`, `facts.md` with frontmatter | **Yes** — same scanner reads them back | Obsidian vaults, a true read+write memory loop |
| `jsonl` | `session-<id>.jsonl`, `facts.jsonl` | No | Programmatic re-ingest, or an append-only **audit trail** |

With `markdown`, what Hermes writes becomes part of your vault and is recalled
on later turns. With `jsonl`, each write is a timestamped JSON record appended
to the file — not recalled, but ideal as an **audit trail** of exactly what the
agent stored and when (e.g. for review, compliance, or replay into another
system).
