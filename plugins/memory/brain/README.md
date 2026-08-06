# brain — Hermes MemoryProvider

Wraps the existing brain engine (`brain-mcp/dist/cli.js`, kuzu + sqlite, namespaced rerank recall) as a Hermes memory provider. Brain engine is **unchanged** — this is a subprocess adapter.

## Install (during P2)
```bash
cp -r migration-hermes/hermes-plugins/memory/brain "$HERMES_HOME/plugins/memory/brain"
export BRAIN_ROOT=/Users/octo/.openclaw/workspace/brain   # or set in brain.json
hermes memory setup            # pick "brain"
hermes memory status           # confirm active
```

## What it does
| Hermes lifecycle | Brain action |
|---|---|
| `prefetch(query)` before each turn | `recall <q> --namespace <ns>` → inject only relevant hits (NOT a static MEMORY.md dump — this is the cost win) |
| `brain_recall` / `brain_remember` tools | on-demand deep search / store |
| `on_memory_write` | mirror built-in MEMORY.md writes into brain |
| `on_pre_compress` / `on_session_end` | `checkpoint` before compaction / at end |

## Namespaces
`[global, apex, 5am, personal, brain]`. Resolved per-profile in `PROFILE_NAMESPACE` (`__init__.py`) so each per-group Hermes profile reads/writes its own space. **P2 TODO:** fill the map as group profiles are named.

## Status: SCAFFOLD (P1)
Not yet run against live Hermes (not installed). Syntax-validated. Wiring to be exercised in P2/P3 sandbox. Tune `sync_turn` salience + finalize namespace map in P2.
