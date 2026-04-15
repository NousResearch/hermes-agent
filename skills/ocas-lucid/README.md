# 🌙 Lucid

Nightly journal curator. Batch-processes OCAS skill journals into MemPalace's verbatim store each night at 3am — classifying each entry for filing as a drawer, knowledge graph triple, Elephas Signal, or skip.

Skill packages follow the [agentskills.io](https://agentskills.io/specification) open standard and are compatible with OpenClaw, Hermes Agent, and any agentskills.io-compliant client.

---

## Overview

Lucid sits between the journal layer and the memory layer. Every night it reads new journals from all installed OCAS skills, scores each for relevance, and routes them: verbatim narratives go to MemPalace drawers, entity facts go to the MemPalace knowledge graph, structured entities worth Chronicle promotion go to Elephas as Signals, and routine telemetry is skipped or queued for re-evaluation.

The dream cycle is resilient: an incremental cursor advances after each successful filing, so interrupted runs resume cleanly. Re-emergence detection auto-promotes topics that keep appearing after initial skip. Two-pass stale handling prevents hasty invalidations. Change magnitude gates catch anomalous runs before they overwrite the store.

## Commands

| Command | Description |
|---|---|
| `lucid.dream` | Run the full dream cycle immediately |
| `lucid.status` | Last run timestamp, pending journals, filing stats, streak count |
| `lucid.init` | Create storage directories, initialize config, register cron jobs |
| `lucid.update` | Pull latest from GitHub source (preserves journals and data) |

## Setup

`lucid.init` runs automatically on first invocation. Requires the **MemPalace MCP server** to be installed and configured — Lucid cannot file to MemPalace without it.

Operators with many active skills may benefit from raising `max_turns` to 150 in their agent config for the `lucid:dream` cron session.

## Dependencies

**OCAS Skills**
- [Elephas](https://github.com/indigokarasu/elephas) — optional Signal pre-check via `elephas.query`; Lucid emits Signals via journal payload for Elephas to promote

**External**
- MemPalace MCP server (required) — verbatim drawer storage and knowledge graph

## Scheduled Tasks

| Job | Mechanism | Schedule | Command |
|---|---|---|---|
| `lucid:dream` | cron | `0 3 * * *` (3am local) | Nightly journal curation |
| `lucid:update` | cron | `0 0 * * *` (midnight daily) | Self-update from GitHub |

## Changelog

### v2.0.0 — April 13, 2026
- Complete rewrite: four-phase dream cycle (Orient, Gather, Classify, File), incremental cursor, re-emergence detection, two-pass stale handling, change magnitude gates, hibernation protection
- Standard `{agent_root}` paths, journal payload Signal emission, removed Python scripts
