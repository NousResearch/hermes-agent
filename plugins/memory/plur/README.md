# PLUR Memory Provider

Persistent memory for Hermes Agent — local engram store with BM25 + embedding search, feedback-trained retrieval, episodic timeline, and cross-device sync. Zero cloud, works offline.

## Requirements

- `pip install plur-hermes>=0.18.1`
- Node.js 18+ (for the PLUR CLI — auto-resolved via `npx` on first use if not installed globally)

## Setup

```bash
pip install plur-hermes
hermes memory setup    # select "plur"
```

Or manually:

```bash
pip install plur-hermes
hermes config set memory.provider plur
```

The `plur-hermes` package is auto-discovered by Hermes as a plugin even without
the `memory.provider` key — installing it activates the standalone hook path
(auto-inject, auto-learn) automatically. Setting `memory.provider: plur` adds
the first-class MemoryProvider path on top, making PLUR visible in
`hermes plugins --memory` and selectable in the desktop Settings panel.

## What PLUR provides

Once active, PLUR runs in the background on every session:

- **Every turn**: relevant memories are injected into context (`prefetch`)
- **Every response**: corrections and insights are captured automatically (`sync_turn`)
- **Every session**: episodes are recorded to the timeline (`on_session_end`)

The agent also gains 22 explicit tools:

| Tool | What it does |
|------|-------------|
| `plur_learn` | Store a correction, preference, or pattern |
| `plur_recall` | Search memories by topic |
| `plur_inject` | Get relevant context for a task |
| `plur_list` | List all stored engrams |
| `plur_forget` | Retire outdated knowledge |
| `plur_feedback` | Rate a memory (trains what surfaces next time) |
| `plur_capture` | Record an episode |
| `plur_timeline` | Query past episodes |
| `plur_status` | Health check |
| `plur_sync` | Cross-device sync via git |
| `plur_packs_list` | List installed knowledge packs |
| `plur_packs_install` | Install a community knowledge pack |
| `plur_packs_export` | Export a pack to a shareable file |
| `plur_extract_meta` | Distill cross-domain principles from memories |
| `plur_meta_engrams` | List extracted meta-engrams |
| `plur_meta_submit_analysis` | Continue multi-turn extraction |
| `plur_validate_meta` | Test a principle against a new domain |
| `plur_ingest` | Extract engrams from text, logs, or conversations |
| `plur_promote` | Promote an engram to higher confidence |
| `plur_similarity_search` | Embedding-based similarity search |
| `plur_stores_add` | Register an additional engram store |
| `plur_stores_list` | List configured stores |

## How it works

Knowledge is stored as **engrams** — small assertions that strengthen with use and decay when irrelevant, modeled on human memory (ACT-R activation). Storage is plain YAML on disk at `~/.plur/`. Search is fully local (BM25 + BGE embeddings + Reciprocal Rank Fusion). The plugin calls the PLUR CLI via subprocess; if the CLI is not installed globally, it auto-resolves via `npx @plur-ai/cli` on first use.

## Configuration

The plugin works with zero configuration. Optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PLUR_PATH` | `~/.plur` | Storage directory |
| `PLUR_INJECT_MODE` | `fast` | `hybrid` for embedding-based injection (slower, more accurate) |
| `PLUR_INJECTION_FEEDBACK` | `true` | Set to `false` to disable automatic injection feedback |

## Links

- [PyPI: plur-hermes](https://pypi.org/project/plur-hermes/)
- [GitHub: plur-ai/plur](https://github.com/plur-ai/plur)
- [npm: @plur-ai/cli](https://www.npmjs.com/package/@plur-ai/cli)

## License

Apache-2.0
