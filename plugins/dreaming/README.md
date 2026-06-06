# Dreaming — automatic memory consolidation

Reference implementation from [Willow 2.0](https://github.com/rudi193-cmd/willow-2.0), adapted to the Hermes plugin interface. Addresses [issue #25309](https://github.com/NousResearch/hermes-agent/issues/25309).

## What it does

Three-phase pipeline that runs automatically during idle periods:

| Phase | What happens |
|---|---|
| **Light Sleep** | Scans recent session transcripts, deduplicates by content hash, scores candidates using weighted criteria |
| **REM** | Calls a local LLM (Ollama) to extract themes and write a narrative diary entry to `DREAMS.md` |
| **Deep Sleep** | Promotes high-signal entries to `MEMORY.md`; routes meta-entries (about memory management itself) to `SKILL.md` instead |

## Opt-in

Disabled by default. Enable with:

```bash
export HERMES_DREAMING=1
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HERMES_DREAMING` | _(unset)_ | Set to `1` to enable |
| `HERMES_DREAM_MODEL` | `mistral:7b` | Ollama model for REM narrative |
| `HERMES_DREAM_MIN_HOURS` | `24` | Minimum hours between cycles |
| `HERMES_DREAM_MIN_SESSIONS` | `5` | Minimum sessions queued before cycle runs |
| `HERMES_DREAM_POLL_SECONDS` | `300` | Background thread poll interval (seconds) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |

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

Promotion threshold: **0.55**. Candidates below threshold appear in the dream diary but are not written to `MEMORY.md`.

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
