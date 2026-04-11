# CerebroCortex Memory Provider

Brain-analogous AI memory for Hermes Agent with associative networks,
biologically-inspired decay, and dream consolidation.

## What it does

CerebroCortex replaces simple key-value memory with a system modeled on
how biological memory actually works:

- **6 memory types** — episodic, semantic, procedural, affective,
  prospective (TODOs), schematic (patterns)
- **Associative graph** — memories link together via 9 typed/weighted
  link types. Search spreads activation through the graph, surfacing
  related memories you didn't explicitly search for
- **Natural decay** — ACT-R power-law activation + FSRS spaced-repetition.
  Frequently-used memories stay sharp, noise fades automatically
- **Dream engine** — Offline LLM-powered consolidation: pattern extraction,
  schema formation, pruning, cross-domain discovery
- **Multi-agent messaging** — send_message/check_inbox for agent-to-agent
  communication through shared memory

All data is local (SQLite + ChromaDB + igraph). No cloud services required.
Runs on hardware as minimal as a Raspberry Pi 5 (4GB).

## Install

```bash
pip install cerebro-cortex
```

## Setup

```bash
hermes memory setup
# Select "cerebrocortex" from the list
```

Or manually in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: cerebrocortex
```

## Tools

| Tool | Description |
|------|-------------|
| `cc_remember` | Store a memory (auto-classifies, deduplicates, links) |
| `cc_recall` | Semantic search with graph activation and decay scoring |
| `cc_todo` | Store/list/resolve TODOs and reminders |
| `cc_message` | Cross-agent messaging (send/inbox) |
| `cc_health` | System health and statistics |

## Automatic features

- **Prefetch** — Before each turn, relevant memories are recalled and
  injected as context
- **Sync** — Significant user messages are auto-stored in background
- **Session end** — Session summary saved automatically
- **Memory mirror** — Built-in MEMORY.md/USER.md writes are mirrored
  to CerebroCortex for searchability
- **Delegation capture** — Subagent task/result pairs stored as memories

## Configuration

| Key | Description | Default |
|-----|-------------|---------|
| `agent_id` | Agent identifier for multi-agent setups | `HERMES` |
| `data_dir` | Data directory path | `~/.cerebro-cortex/` |

Set via `hermes memory setup` or in config.yaml:

```yaml
plugins:
  cerebrocortex:
    agent_id: HERMES
    data_dir: ~/.cerebro-cortex/
```

## Multi-agent

Multiple Hermes instances can share a CerebroCortex store:

```yaml
# Instance 1
plugins:
  cerebrocortex:
    agent_id: HERMES-PRIMARY

# Instance 2
plugins:
  cerebrocortex:
    agent_id: HERMES-RESEARCH
```

Agents communicate via `cc_message`:
- `cc_message(action="send", to="HERMES-RESEARCH", content="...")`
- `cc_message(action="inbox")`

## Links

- [CerebroCortex repo](https://github.com/buckster123/CerebroCortex)
- [PyPI](https://pypi.org/project/cerebro-cortex/)
- [Integration report](https://github.com/buckster123/CerebroCortex/blob/main/HERMES_INTEGRATION_REPORT.md)
