# Honcho Memory Provider

AI-native cross-session user modeling with dialectic Q&A, semantic search, peer cards, and persistent conclusions.

## Requirements

- `pip install honcho-ai`
- Honcho API key from [app.honcho.dev](https://app.honcho.dev)

## Setup

```bash
hermes memory setup    # select "honcho"
```

Or manually:
```bash
hermes config set memory.provider honcho
echo "HONCHO_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

Config file: `$HERMES_HOME/honcho.json` (or `~/.honcho/config.json` legacy)

Existing Honcho users: your config and data are preserved. Just set `memory.provider: honcho`.

## Tools

| Tool | Description |
|------|-------------|
| `honcho_profile` | User's peer card — key facts, no LLM |
| `honcho_search` | Semantic search over stored context |
| `honcho_context` | LLM-synthesized answer from memory |
| `honcho_conclude` | Write a fact about the user to memory |
