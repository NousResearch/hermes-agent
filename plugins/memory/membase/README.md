# Membase Memory Provider

Persistent long-term memory for Hermes Agent using Membase's hybrid vector
search, knowledge graph, and wiki retrieval.

The implementation lives in the
[`hermes-membase`](https://pypi.org/project/hermes-membase/) package. This
directory is a thin Hermes shim that registers the provider and installs the
package through `hermes memory setup`.

## Requirements

- Python 3.11+
- `hermes-membase>=0.1.5`
- Membase OAuth login

## Setup

```bash
hermes memory setup    # select "membase"
hermes membase login   # OAuth PKCE login
```

Or configure manually:

```bash
uv pip install --python "$(which python)" "hermes-membase>=0.1.5"
hermes config set memory.provider membase
hermes membase login
```

## Config

Config file: `$HERMES_HOME/membase.json`

Credentials file: `$HERMES_HOME/credentials/membase.json`

| Key | Default | Description |
|-----|---------|-------------|
| `apiUrl` | `https://api.membase.so` | Membase API URL. Override for self-hosted deployments. |
| `tokenFile` | `$HERMES_HOME/credentials/membase.json` | OAuth token cache path. |
| `autoRecall` | `false` | Inject relevant memories before each response. |
| `autoWikiRecall` | `false` | Inject relevant wiki documents before each response. |
| `autoCapture` | `true` | Automatically store conversations in Membase. |
| `mirrorBuiltin` | `true` | Mirror Hermes built-in `MEMORY.md` writes into Membase. |
| `maxRecallChars` | `4000` | Max characters of recalled context per turn. |
| `debug` | `false` | Enable verbose debug logging. |

## CLI Commands

```bash
hermes membase login
hermes membase logout
hermes membase status
hermes membase resync
hermes membase resync --dry-run
```

## Tools

| Tool | Description |
|------|-------------|
| `membase_search` | Search memories by semantic similarity with optional date and source filters. |
| `membase_store` | Save important conversational context to long-term memory. |
| `membase_forget` | Delete a memory through a confirmation flow. |
| `membase_profile` | Retrieve user profile context and related memories. |
| `membase_search_wiki` | Search wiki documents and return full document content. |
| `membase_add_wiki` | Create a wiki document from markdown content. |
| `membase_update_wiki` | Update an existing wiki document. |
| `membase_delete_wiki` | Delete a wiki document through a confirmation flow. |

## Behavior

When enabled, Hermes can:

- prefetch relevant memory and wiki context before turns
- expose explicit tools for memory and wiki search/write/delete operations
- capture user conversation context asynchronously
- mirror built-in `MEMORY.md` writes to Membase in the background

## Troubleshooting

If `hermes membase` is unavailable, make sure the provider is active:

```bash
hermes config set memory.provider membase
```

If Hermes says Membase is disconnected, run:

```bash
hermes membase login
```

If the provider loads but the implementation package is missing, install it:

```bash
uv pip install --python "$(which python)" "hermes-membase>=0.1.5"
```

Then restart Hermes so the real provider implementation is imported.

## Links

- [Membase](https://membase.so)
- [Hermes connector docs](https://docs.membase.so/connectors/hermes)
- [hermes-membase source](https://github.com/aristoapp/hermes-membase)
- [hermes-membase on PyPI](https://pypi.org/project/hermes-membase/)
