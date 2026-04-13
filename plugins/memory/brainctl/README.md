# brainctl Memory Provider

SQLite-backed cognitive memory with FTS5 full-text search, optional vector recall, a knowledge graph, affect tracking, and session handoffs. One file, zero servers, zero API keys.

Upstream: [brainctl on GitHub](https://github.com/TSchonleber/brainctl) · [PyPI](https://pypi.org/project/brainctl/)

## Requirements

- `pip install 'brainctl>=1.3.0'` into Hermes's Python environment
- Optional vector recall: `pip install 'brainctl[vec]'` + [Ollama](https://ollama.com) running `nomic-embed-text`

## Setup

```bash
hermes memory setup    # select "brainctl"
```

Or manually:
```bash
hermes config set memory.provider brainctl
```

The brain lives at `$HERMES_HOME/brainctl/brain.db` by default — no external service required.

## Config

Config file: `$HERMES_HOME/brainctl/config.json` (profile-scoped). All fields are optional.

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/brainctl/brain.db` | SQLite brain path. Falls back to `$BRAIN_DB`. |
| `agent_id` | `hermes` | Recorded on every write for multi-agent scoping. |
| `memory_mode` | `hybrid` | `context` (auto-inject only), `tools` (tools only), or `hybrid` (both). |
| `recall_method` | `search` | `search` (FTS5), `vsearch` (vector), or `think` (spreading activation). |
| `recall_limit` | `8` | Max memories returned per auto-recall. |
| `auto_recall` | `true` | Auto-prefetch context before each turn. |
| `auto_retain` | `true` | Auto-retain completed turns. |
| `retain_category` | `conversation` | Category for auto-retained turns. |
| `retain_every_n_turns` | `1` | Batch retains every N turns. |
| `session_bookends` | `true` | Call `orient()` at start, `wrap_up()` at session end. |
| `mirror_memory_md` | `true` | Mirror built-in `MEMORY.md` / `USER.md` writes into `brain.db`. |

## Tools

| Tool | Description |
|------|-------------|
| `brainctl_remember` | Store a durable fact with category + tags |
| `brainctl_search` | FTS5 full-text search over stored memories |
| `brainctl_think` | Spreading-activation recall across the knowledge graph |
| `brainctl_log` | Append a structured event to the event stream |
| `brainctl_entity` | Create / observe / relate entities in the knowledge graph |
| `brainctl_decide` | Record a decision with its rationale |
| `brainctl_handoff` | Write a handoff packet for the next session |

## What it does behind the scenes

- **Auto-recall** — before each turn, the provider prefetches the top-K most relevant memories for the user message and injects them into the system prompt.
- **Auto-retain** — completed user/assistant turns are written back as `conversation` memories, so the brain grows as the agent works.
- **Session bookends** — `orient()` pulls the latest handoff + recent events + high-signal memories at session start; `wrap_up()` writes a handoff packet at session end so the next run resumes cleanly.
- **Pre-compress hook** — before Hermes compresses conversation history, the provider can summarize dropped turns into durable memories instead of losing them.
- **MEMORY.md / USER.md mirroring** — writes through Hermes's built-in memory surface are mirrored into `brain.db` so you get one unified store.

## Categories

`brainctl_remember` accepts a `category` field. Recommended taxonomy:
`convention`, `decision`, `environment`, `identity`, `integration`, `lesson`, `preference`, `project`, `user`, `general`.

## Troubleshooting

- **"brainctl is not installed"** — the package isn't in Hermes's venv. Activate the venv Hermes runs under, then `pip install 'brainctl>=1.3.0'`.
- **Vector recall returns nothing** — `recall_method: vsearch` requires `brainctl[vec]` and Ollama serving `nomic-embed-text`. Fall back to `search` if you don't want the vector dependency.
- **Stale handoffs** — `brain.db` is a plain SQLite file; you can inspect it with `sqlite3` or delete it to reset.

## License

Plugin source: MIT. brainctl itself: MIT.
