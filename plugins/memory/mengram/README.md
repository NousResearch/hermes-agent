# Mengram memory provider

[Mengram](https://mengram.io) gives Hermes three memory types with server-side
extraction — no local LLM calls, no extra processes:

- **Semantic** — facts, preferences, entities, knowledge-graph relations
- **Episodic** — events, decisions, and their outcomes
- **Procedural** — workflows with version history that **evolve from
  failures**: each failed run records the violated assumption and derives a
  precondition that rides along with recall (`mengram_procedures` returns
  steps + "verify first" checks)

## Setup

```
hermes memory setup        # choose mengram, paste your API key
```

Free key at [mengram.io](https://mengram.io) (no card). For self-hosted
deployments set `base_url` to your instance (Apache 2.0 core).

## What it does

- **Prefetch**: background recall across all three memory types before each
  turn (hybrid vector + BM25 + RRF retrieval, 23-language multilingual)
- **Capture**: turns are buffered and flushed in the background every N turns
  (`sync_every_turns`, default 3 — one plan-quota `add` per flush, keeping
  free-tier usage sane), plus on session end and **before context
  compression** — memory keeps what the compaction summary loses
- **Tools**: `mengram_search` (everything), `mengram_remember` (explicit
  store), `mengram_procedures` (learned workflows with preconditions)
- **Cognitive profile** in the system prompt: a distilled who-is-this-user
  block generated server-side from all memory

## Cross-tool memory

By default the provider uses the same memory scope as Mengram's Claude Code
plugin, Cursor/MCP integrations, and CLI — so context built in any tool is
available in Hermes and vice versa. Gateway sessions (Telegram, Discord)
keep per-user isolation via gateway-native ids unless you configure a
canonical `user_id`.

## Notes

- stdlib-only (urllib) — no pip dependencies
- Circuit breaker: after 5 consecutive API failures, calls pause for 120s
- Writes are disabled automatically in cron/subagent contexts
- Secrets (API key) live in `$HERMES_HOME/.env`; non-secret settings in
  `$HERMES_HOME/mengram.json`
