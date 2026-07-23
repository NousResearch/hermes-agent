---
name: uteke
description: "Local semantic memory with vector recall and graph."
version: 0.7.3
author: "Anaz S. Aji <ajianaz>"
license: Apache-2.0
platforms:
  - linux
  - macos
  - windows
metadata:
  hermes:
    tags:
      - memory
      - semantic-search
      - knowledge-graph
      - offline
    homepage: https://github.com/codecoradev/uteke
    related_skills:
      - hermes-agent
prerequisites:
  commands:
    - uteke
---

# Uteke Skill

[Uteke](https://github.com/codecoradev/uteke) is a local-first semantic memory engine.
Single Rust binary, ~30ms recall via EmbeddingGemma ONNX (768d). No API keys, no cloud.
Provides vector search, FTS5 keyword search, knowledge graph, rooms, and documents.

## When to Use

- Setting up offline semantic memory for Hermes agents
- Multi-agent namespace isolation (each agent gets its own memory silo)
- Knowledge graph queries — nodes, edges, shortest path
- Collaborative rooms — shared context across agents/namespaces
- Importing knowledge bases from Markdown, JSONL, or text files
- Auto-consolidation of near-duplicate memories

## Prerequisites

Install the `uteke` binary:

```bash
curl -fsSL https://raw.githubusercontent.com/codecoradev/uteke/main/install.sh | sh
uteke --version
```

Available for linux (x86_64, ARM), macOS (ARM), and Windows. Downloads a
pre-built binary — no Rust toolchain required.

## How to Run

### Mode A — Plugin (manual, requires uteke-serve)

```bash
uteke init --agent hermes
uteke-serve --port 8767
```

`uteke init --agent hermes` generates a plugin that auto-loads in new sessions,
providing an `uteke(action=...)` tool for manual operations.

### Mode C — pre_llm_call Shell Hook (automatic recall, no daemon)

Hermes injects `pre_llm_call` hook stdout into the prompt. The hook **must**
output `{"context": "..."}` on stdout — raw text is silently ignored
(`agent/shell_hooks.py:496-539`).

Create a handler script (e.g., `~/.hermes/hooks/uteke-recall.py`):

```python
#!/usr/bin/env python3
"""Recall relevant memories before each LLM call."""
import json, subprocess, sys

def main():
    payload = json.loads(sys.stdin.read())
    message = (payload.get("extra") or {}).get("user_message", "")
    if not message or len(message.strip()) < 5:
        sys.exit(0)

    result = subprocess.run(
        ["uteke", "recall", message.strip()[:500],
         "--limit", "5", "--context", "--json"],
        capture_output=True, text=True, timeout=15
    )
    if not result.stdout.strip():
        sys.exit(0)

    try:
        memories = json.loads(result.stdout)
        lines = []
        for item in memories[:5]:
            m = item.get("memory", item)
            score = item.get("score", 0)
            lines.append(f"[{score:.2f}] {m.get('content', '')[:200]}")
        if lines:
            json.dump(
                {"context": "Recalled memories:\n" + "\n".join(lines)},
                sys.stdout
            )
    except (json.JSONDecodeError, KeyError):
        sys.exit(0)

if __name__ == "__main__":
    main()
```

Register in `~/.hermes/profiles/<profile>/config.yaml`:

```yaml
hooks:
  pre_llm_call:
    - command: "python3 ~/.hermes/hooks/uteke-recall.py"
      timeout: 20
hooks_auto_accept: true
```

**Test locally:**

```bash
echo '{"user_message": "check CI status", "session_id": "test"}' | \
  timeout 20 python3 ~/.hermes/hooks/uteke-recall.py
```

### MCP Server (alternative)

```bash
# Stdio transport
hermes mcp add uteke --command uteke-mcp
# HTTP transport (requires uteke-serve)
hermes mcp add uteke --url http://127.0.0.1:8767/mcp
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `uteke remember "content" --tags t1,t2` | Store a memory |
| `uteke recall "query" --limit 5` | Semantic search (vector) |
| `uteke search "keywords"` | Keyword search (FTS5) |
| `uteke list --tag mytag --limit 20` | List memories by tag |
| `uteke forget <id>` | Delete a memory |
| `uteke stats` | Store statistics |
| `uteke doctor` | Health check |
| `uteke verify` | DB + index consistency |
| `uteke repair` | Rebuild index from SQLite |
| `uteke consolidate --threshold 0.85` | Merge near-duplicates |
| `uteke dream` | Full maintenance pipeline |
| `uteke pin <id>` | Prevent memory decay |
| `uteke graph nodes` | List graph nodes |
| `uteke graph neighbors "X"` | Find connected nodes (BFS) |
| `uteke graph path "A" "B"` | Shortest path |
| `uteke room create --id sprint --title "Sprint"` | Create room |
| `uteke room recall --room sprint "query"` | Recall from room |
| `uteke doc create --slug s --content "..."` | Create document |
| `uteke doc search "query"` | Search documents |

## Procedure

### Remember + Recall

```bash
uteke remember "User prefers Rust over Go" --tags preference --type preference
uteke recall "language preferences" --limit 3 --json
```

JSON output is nested under `"memory"` key:
`[{"memory": {"content": "..."}, "score": 0.72}]`

### Multi-Agent Namespace Isolation

```bash
uteke remember "CTO approved migration plan" --namespace cto --tags decision
uteke recall "migration plan" --namespace cto
uteke namespace list
```

Namespace is a strict silo — no cross-namespace search. Omitting `--namespace`
defaults to `default`, not `all`.

### Import Knowledge Base

```bash
uteke import knowledge.jsonl --namespace project --tags imported
uteke import --batch-dir ./docs/ --recursive --namespace docs
uteke import architecture.md --extract --namespace knowledge
```

### Knowledge Graph

```bash
uteke remember "PostgreSQL is used for user data" --entity PostgreSQL
uteke remember "Redis caches session tokens" --entity Redis
uteke graph path "PostgreSQL" "Redis"
uteke graph neighbors "PostgreSQL"
```

Entity is stored as metadata — use `--entity` consistently for graph
operations. Without it, `graph nodes` returns nothing.

### Warm Server (optional)

```bash
uteke-serve --port 8767 --auth-token <secret>
```

First CLI call loads the ONNX model (~3s). `uteke-serve` keeps it warm
in RAM (~208MB) for sub-50ms recall.

## Pitfalls

- **Cold start ~3s.** First CLI call loads the ONNX model (188MB).
  Use `uteke-serve` for persistent warm model.
- **EmbeddingGemma max ~8K chars.** Content longer than this is silently
  truncated for embedding. Full text is still stored and returned.
- **`recall --json` nested output.** Results under `"memory"` key —
  `[{memory: {content, ...}, score}]`. Do not parse as flat `{content, score}`.
- **`uteke-serve` is a separate binary.** Bundled in the same release
  tarball — both `uteke` and `uteke-serve` must be installed.
- **Import always re-embeds.** Export/re-import regenerates vectors.
  Portable format carries content only, not vectors.
- **Entity stored as metadata.** The `--entity` flag writes to the metadata
  JSON column, not a dedicated schema column. Use `--entity` consistently
  or graph operations (`graph nodes`, `neighbors`) return nothing.
- **`uteke init --agent hermes` generates a plugin file.** Place the output
  in your Hermes plugins directory. Requires `uteke-serve` running for
  the plugin to connect.
- **Shell hook must output JSON `{"context": "..."}`.** Raw text or CLI
  output without this wrapper is silently ignored by Hermes
  (`agent/shell_hooks.py`).

## Verification

```bash
uteke --version          # prints version
uteke doctor             # health check: DB, index, model, consistency

# Round-trip test
uteke remember "verification test" --tags test --namespace default
uteke recall "verification" --limit 1 --namespace default
uteke forget <id>        # clean up using ID from recall output

# Mode C hook test
echo '{"user_message": "test recall", "session_id": "verify"}' | \
  python3 ~/.hermes/hooks/uteke-recall.py
# Expected: {"context": "Recalled memories:\n..."}
```
