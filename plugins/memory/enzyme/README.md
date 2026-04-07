# Enzyme Memory Provider

Compile-time semantic memory for agent workspaces. Reads your markdown vault once — tags, links, folders, timestamps — embeds locally, pre-computes thematic questions (catalysts), and serves 8ms queries with zero runtime LLM calls.

## Requirements

- `pip install enzyme-python-package` (auto-downloads the enzyme binary and embedding model on first run)
- A markdown vault (Obsidian, Logseq, or any folder of `.md` files)

## Setup

```bash
hermes memory setup    # select "enzyme"
```

Or manually:
```bash
hermes config set memory.provider enzyme
```

No API keys required — everything runs locally. The setup wizard prompts for your vault path.

## How It Works

Unlike other memory providers that store conversation facts, enzyme indexes the user's **existing notes** and surfaces connections between ideas.

**Compile step** (~10-20s, runs once per vault change):
1. Reads vault structure: tags, links, folders, timestamps
2. Embeds documents with a local ONNX model
3. Generates catalysts — thematic questions that bridge semantic gaps
4. Pre-computes similarities so queries resolve without vector search

**Runtime** (8ms, no API calls):
- Vault landscape is injected into the system prompt from turn zero
- Each turn re-ranks by the user's message
- Session end refreshes so notes written during the session are indexed

## Tools

| Tool | Description |
|------|-------------|
| `enzyme_petri` | Vault overview — main topics, how recently active they are, and catalysts (thematic questions that become the vocabulary for deeper search) |
| `enzyme_catalyze` | Concept search using catalyst vocabulary — reaches content the user's raw words won't find. Three registers: `explore` (patterns and tensions), `continuity` (prior decisions and context), `reference` (what the user chose to capture) |
| `enzyme_refresh` | Re-index vault content. Fast: skips unchanged files. Use `full=true` to force |
| `enzyme_status` | Vault stats: doc count, entity count, catalyst count, embedding coverage |
| `enzyme_init` | Initialize enzyme on a vault with an optional guide (entity list) |
