# Memanto Memory Agent Provider

[Memanto](https://memanto.ai) is a **memory agent** — typed long-term memory with
confidence and provenance, semantic recall, and RAG-style answers, backed by the
[Moorcheh](https://moorcheh.ai) vector platform. Each Memanto namespace is a
first-class *agent* (`memanto agent create/activate`), so this provider maps one
Hermes identity to one Memanto agent.

## Requirements

- `pip install memanto`
- Moorcheh API key from [console.moorcheh.ai/api-keys](https://console.moorcheh.ai/api-keys)

## Setup

```bash
hermes memory setup    # select "memanto"
```

Or manually:

```bash
hermes config set memory.provider memanto
echo 'MOORCHEH_API_KEY=***' >> ~/.hermes/.env
```

## Config

Config file: `$HERMES_HOME/memanto.json`

| Key | Default | Description |
|-----|---------|-------------|
| `agent_id` | `hermes-{identity}` | The Memanto agent id (memory namespace). Supports the `{identity}` template for per-profile scoping (e.g. `hermes-coder`). |
| `pattern` | `tool` | Memanto agent pattern used when auto-creating: `support`, `project`, or `tool`. |
| `auto_recall` | `true` | Inject relevant memories before each turn. |
| `auto_capture` | `true` | Store cleaned user/assistant turns as `event` memories. |
| `auto_create` | `true` | Create the agent on first use if it does not exist. |
| `mirror_memory_writes` | `true` | Echo Hermes built-in `memory` writes into Memanto. |
| `max_recall_results` | `10` | Max memories formatted into prefetch context (1–100). |
| `min_confidence` | `null` | Drop recalled memories below this confidence (0.0–1.0). |
| `session_duration_hours` | `null` | Override Memanto session lifetime (defaults to Memanto's config). |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MOORCHEH_API_KEY` | Moorcheh API key (required). |
| `MEMANTO_AGENT_ID` | Override the agent id (takes priority over the config file). |

## Tools

| Tool | Description |
|------|-------------|
| `memanto_remember` | Store a durable fact, preference, decision, goal, or instruction. |
| `memanto_recall` | Search memories by semantic similarity. |
| `memanto_answer` | Answer a question grounded only in stored memories (RAG). |

## Behavior

When enabled, Hermes can:

- prefetch relevant memories before each turn (`auto_recall`)
- store cleaned conversation turns as `event` memories after each response (`auto_capture`)
- mirror Hermes built-in `memory`/`user` writes into Memanto (`mirror_memory_writes`)
- expose explicit `remember` / `recall` / `answer` tools

Memory types follow Memanto's taxonomy: `fact`, `preference`, `goal`, `decision`,
`artifact`, `learning`, `event`, `instruction`, `relationship`, `context`,
`observation`, `commitment`, `error`.

## Profile-Scoped Agents

Use `{identity}` in `agent_id` to scope memories per Hermes profile:

```json
{
  "agent_id": "hermes-{identity}"
}
```

A profile named `coder` resolves to `hermes-coder`; the default profile resolves
to `hermes-default`. Without `{identity}`, all profiles share one Memanto agent.

## Notes

- The session is warmed up (agent ensured + activated) in the background at
  startup, so the first turn's recall does not also pay activation latency.
- Writes are skipped in `cron`, `flush`, and `subagent` contexts to avoid
  corrupting the user's memory representation.
- Only one external memory provider runs at a time, selected via
  `memory.provider` in `config.yaml`.

## Support

- [memanto.ai](https://memanto.ai)
- [moorcheh.ai](https://moorcheh.ai)
