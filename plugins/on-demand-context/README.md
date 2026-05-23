# on-demand-context Plugin

A Hermes Plugin that defers context injection to save tokens.

## Why

When your agent has access to many context documents (team charters,
onboarding guides, API references, project wikis), injecting all of
them into every conversation turn wastes tokens — especially when the
agent only needs one or two of them.

This plugin solves that by:

1. **Injecting only a lightweight index (~400 bytes)** on the first
   turn, listing what's available.
2. **Providing a `load_context()` tool** the agent can call to fetch
   any document's full content only when needed.

## How it works

```
First turn:
  User: "Tell me about team governance"
  → Plugin injects index (~400B):
    📋 Available Knowledge:
      • team-governance — Defines roles, responsibilities...
      • onboarding-guide — How to set up your dev environment...
      • api-reference — Endpoints and authentication...
  → Agent sees the index and calls:
  
  Agent: load_context("team-governance")
  → Plugin returns full document content
  
  Agent: (now has the details it needed)
  → No tokens wasted on documents the agent didn't need.
```

## Installation

Place this directory in your Hermes plugins directory:

```bash
cp -r plugins/on-demand-context ~/.hermes/plugins/
```

Then enable it in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - on-demand-context
```

**Note**: To load plugins from `~/.hermes/plugins/`, Hermes will
automatically discover them. No additional configuration needed.

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HERMES_CONTEXT_DIR` | `~/.hermes/context-docs` | Directory to scan for context documents |
| `HERMES_CONTEXT_INDEX_TITLE` | `📋 Available Knowledge` | Title shown in the injected index |

### Plugin config (plugin.yaml)

```yaml
config:
  scan_dir: "~/.hermes/context-docs"
  index_title: "📋 Available Knowledge"
  file_pattern: "*.md"
```

## Usage

1. Place your context documents (`.md` files) in the configured
   `scan_dir`. Any markdown file works — the plugin reads the first
   heading as the title and the first few lines as the summary.

2. Start a conversation. On the first turn, you'll see the index
   injected automatically.

3. The agent can call `load_context(document_id)` to load any
   document's full content. The document ID is the filename without
   the `.md` extension.

### Example

```
📋 Available Knowledge:
  • team-governance  —  Defines roles, responsibilities, and...
  • onboarding-guide  —  How to set up your dev environment...
  • api-reference  —  Endpoints and authentication...
  
Use load_context() to fetch the full content of any document above.
```

## Development

### Source files

| File | Purpose |
|------|---------|
| `__init__.py` | Plugin implementation |
| `plugin.yaml` | Plugin metadata and configuration |
| `README.md` | This file |

### Hooks used

- `pre_llm_call` — Injects the context index on the first turn.

### Tools provided

- `load_context(id)` — Returns the full content of a document.

## License

Same as Hermes Agent.
