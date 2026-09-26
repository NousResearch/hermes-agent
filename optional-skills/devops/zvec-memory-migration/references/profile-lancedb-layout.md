# Profile LanceDB Layout (as of 2026-06-09)

## Plugin

- Path: `~/.hermes/plugins/memory-lancedb/` (Git-managed, independent repo)
- Plugin version: 1.0.0
- lancedb package: 0.33.0 (in `~/.hermes/venv/`)
- Embedding: Ollama bge-m3:latest → 1024-dim (runs on Windows, WSL via localhost:11434)

## Per-Profile Details

| Profile | HERMES_HOME | lance_dir (config) | Resolved Path | Data Files | HNSW+FTS | Config Key |
|---|---|---|---|---|---|---|
| default | `~/.hermes` | `$HERMES_HOME/记忆数据库/lance_memory` | `~/.hermes/记忆数据库/lance_memory` | 219 | ✅ | `plugins.memory-lancedb` |
| <profile> | `~/.hermes/profiles/<profile>` | `$HERMES_HOME/记忆数据库/lance_memory` | `~/.hermes/profiles/<profile>/记忆数据库/lance_memory` | 69 | ✅ | `plugins.memory-lancedb` |
| <profile> | `~/.hermes/profiles/<profile>` | absolute path (unicode-escaped) | `~/.hermes/profiles/<profile>/记忆数据库/lance_memory` | 1 | ✅ | `plugins.lancedb-embed` (legacy) |
| <profile> | `~/.hermes/profiles/<profile>` | `$HERMES_HOME/记忆数据库/lance_memory` | `~/.hermes/profiles/<profile>/记忆数据库/lance_memory` | 1 | ✅ | `plugins.memory-lancedb` |

## Directory Structure (per profile)

```
记忆数据库/lance_memory/
├── __manifest/
│   ├── _transactions/
│   └── _versions/
└── memories.lance/
    ├── _deletions/
    ├── _indices/        ← HNSW + FTS index files
    ├── _transactions/
    ├── _versions/
    └── data/            ← .lance data files
```

## Notes

- <profile> still uses legacy `plugins.lancedb-embed` key; backward compat in plugin code handles it
- default profile has 375 rows (heaviest usage); sub-profiles have sparse data
- All paths use Chinese directory name `记忆数据库` (configured in lance_dir)
- Ollama process: `ollama app.exe` on Windows (PID varies), WSL accesses via localhost forwarding
- nameserver in `/etc/resolv.conf`: 10.255.255.254
