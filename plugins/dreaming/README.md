# Dreaming — automatic memory consolidation

Reference implementation from [Willow 2.0](https://github.com/rudi193-cmd/willow-2.0), adapted to the Hermes plugin interface. Addresses [issue #25309](https://github.com/NousResearch/hermes-agent/issues/25309).

## What it does

Three-phase pipeline that runs automatically during idle periods:

| Phase | What happens |
|---|---|
| **Light Sleep** | Scans recent session transcripts, deduplicates by content hash, scores candidates using weighted criteria |
| **REM** | Calls a local LLM (Ollama) to extract themes and write a narrative diary entry to `DREAMS.md` |
| **Deep Sleep** | Promotes high-signal entries to `MEMORY.md`; routes meta-entries (about memory management itself) to `SKILL.md` instead |

## Enablement

Dreaming is opt-in at two levels:

1. **Plugin discovery** — enable the bundled plugin:

```bash
hermes plugins enable dreaming
```

2. **Behavioral config** — set `enabled: true` in the plugin-owned config file (see below). The plugin seeds defaults on first load.

## Configuration

Behavioral settings live in **`$HERMES_HOME/dreaming/config.yaml`** (plugin-owned, seeded from `plugins/dreaming/config.yaml` on first enable). You may also override the same keys under a top-level `dreaming:` section in `~/.hermes/config.yaml`.

```yaml
enabled: false

schedule:
  min_hours: 24
  min_sessions: 5
  poll_seconds: 300

rem:
  model: mistral:7b
  base_url: http://localhost:11434

scoring:
  promote_threshold: 0.55
```

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | Master switch for hooks and the background scheduler |
| `schedule.min_hours` | `24` | Minimum hours between automatic cycles |
| `schedule.min_sessions` | `5` | Minimum sessions queued before a cycle runs |
| `schedule.poll_seconds` | `300` | Background thread poll interval |
| `rem.model` | `mistral:7b` | Ollama model for REM narrative |
| `rem.base_url` | `http://localhost:11434` | Ollama base URL |
| `scoring.promote_threshold` | `0.55` | Score floor for MEMORY.md promotion |

Per Hermes policy, non-secret behavioral settings belong in `config.yaml` — not new `HERMES_DREAM_*` environment variables.

## Scoring

Candidates are scored before promotion using these weights (from the issue spec):

| Dimension | Weight |
|---|---|
| Relevance | 30% |
| Frequency | 24% |
| Query diversity | 15% |
| Recency | 15% |
| Consolidation (novelty vs existing) | 10% |
| Conceptual richness | 6% |

Promotion uses `scoring.promote_threshold` (default **0.55**). Candidates below threshold appear in the dream diary but are not written to `MEMORY.md`.

## Meta-entry filter

Entries about memory management itself (e.g. "memory is full", "update SKILL.md", "memory capacity") are detected and routed to `SKILL.md` rather than `MEMORY.md`. This addresses the most common source of memory rot described in [#25309](https://github.com/NousResearch/hermes-agent/issues/25309#issuecomment-4638538182).

## Slash commands

```
/dream            show status + last diary entry
/dream run        force a consolidation cycle immediately
/dream status     show hours since last cycle, sessions queued, ready state
/dream diary      show the last dream diary entry
```

## File layout

```
{HERMES_HOME}/
  dreaming/
    config.yaml           ← plugin-owned behavioral settings
  MEMORY.md               ← promoted entries appended here
  SKILL.md                ← meta-entries routed here
  DREAMS.md               ← REM narrative diary
  dreams/
    staging.jsonl         ← candidates queued from session_end hooks
    state.json            ← last_dream_at, sessions_since_dream
    lock                  ← present while a cycle is running
```

## Without Ollama

If Ollama is unavailable the REM phase falls back to a structured bullet-point summary. All other phases (Light Sleep, Deep Sleep, diary write) work without it.

## Relationship to Willow 2.0

In Willow this runs as `dream_run` (SAP MCP tool) + `dream_check` (gate) + `scripts/sleep_consolidation.py` (nightly batch) + `tension_scan` for semantic deduplication. The Hermes plugin is a standalone port — no Willow dependency.
