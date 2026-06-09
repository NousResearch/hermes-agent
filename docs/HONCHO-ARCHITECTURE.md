# Honcho Unified Memory Architecture

## Overview

Honcho is a persistent memory layer that continuously reasons about conversations
to build rich peer representations. This doc describes how it is wired across all
tools used on this machine so that context learned in any one tool is available in
every other.

## Stack

```
Windows 11 / WSL2
├── Honcho Docker Stack          C:\dev\ai\honcho\
│   ├── api         port 8000   <- all tools write here (REST + MCP)
│   ├── deriver     (background LLM reasoning worker - triggers at ~1 000 tokens)
│   ├── database    port 5432   pgvector (postgres:15) - internal only
│   └── redis       port 6379   caching - internal only
│
├── Hermes         ~/.hermes/honcho.json           -> localhost:8000
├── Claude Code    .honcho/config.json             -> localhost:8000
├── OpenCode       .honcho/config.json (same file) -> localhost:8000
├── Antigravity    .honcho/config.json (same file) -> localhost:8000
└── Email cron     scripts/honcho-email-ingest.py  -> localhost:8000 (SDK direct)

Tailscale: all ports reachable at 100.108.75.69
```

## Unified Peer

Every tool uses the exact same two identifiers:

| Setting | Value |
|---------|-------|
| workspace | `calum-selfhost` |
| peerName | `calumai` |

A fact stored by Hermes is readable by Claude Code and vice versa, because both
write messages to the `calumai` peer inside `calum-selfhost`. The deriver worker
then reasons across ALL of those messages to build a single unified representation.

## Key Config Files

| Tool | Config file |
|------|-------------|
| Hermes | `~/.hermes/honcho.json` |
| Claude Code | `<project>/.honcho/config.json` |
| OpenCode (Codex) | `<project>/.honcho/config.json` (same file) |
| Antigravity CLI | `<project>/.honcho/config.json` (same file) |
| Email cron | `scripts/honcho-email-ingest.py` (constants at top of file) |
| Honcho server | `C:\dev\ai\honcho\.env` |

## Hermes Tools Exposed via Honcho

| Tool | What it does |
|------|-------------|
| `honcho_profile` | Fast peer card - key facts, no LLM call |
| `honcho_search` | Semantic search over stored memory |
| `honcho_context` | Dialectic Q&A - synthesised answer from deriver |
| `honcho_conclude` | Write a durable fact back into Honcho |

## LLM Routing

All Honcho LLM calls are routed through OpenRouter:

| Component | Model | Notes |
|-----------|-------|-------|
| Deriver | `google/gemini-2.5-flash` | Fast, cheap, good for extraction |
| Dialectic (all levels) | `anthropic/claude-haiku-4-5` | Good reasoning at low cost |
| Summary | `google/gemini-2.5-flash` | Summarises long sessions |

## Email Ingestion

Daily cron at 06:00 via Windows Task Scheduler:

```
python scripts/honcho-email-ingest.py C:\Users\calumai\exported-emails.json
```

Input: JSON array `[{"subject","from","body","timestamp" (ISO 8601 UTC),"thread_id"}]`

Each run creates/reuses a session named `email-import-YYYY-MM-DD` and batches
messages in groups of 100.

## Docker Compose Commands

```powershell
cd C:\dev\ai\honcho

# Start (first run builds from source - takes a few minutes)
docker compose up -d --build

# Health check
Invoke-WebRequest http://localhost:8000/health | Select-Object StatusCode, Content

# Tail deriver logs
docker compose logs deriver -f --tail 20

# Stop
docker compose down
```

## Workspace Bootstrap

After first start, create the workspace:

```powershell
Invoke-WebRequest -Method POST http://localhost:8000/v3/workspaces `
  -ContentType "application/json" `
  -Body '{"name":"calum-selfhost"}' | Select-Object Content
```

## Verification Checklist

| Check | Command | Expected |
|-------|---------|----------|
| API health | `Invoke-WebRequest http://localhost:8000/health` | `{"status":"ok"}` |
| Deriver running | `docker compose logs deriver --tail 5` | `polling` or `processing` |
| Hermes connected | `hermes memory status` | `Provider: honcho, connected` |
| Claude Code MCP | `/mcp` in Claude Code | `honcho: connected` |
| Cross-tool recall | Ask Claude Code about a Hermes-stated preference | Returns the preference |
