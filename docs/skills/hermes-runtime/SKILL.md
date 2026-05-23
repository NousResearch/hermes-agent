# Hermes Runtime

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18

## Purpose

Encode the Hermes/OpenClaw agent runtime mechanics -- how sessions start, how tools execute, how cron jobs fire, and how the infrastructure is configured. The screener-ops skill covers *what* the fleet does; the openclaw-agent-optimize skill covers *how to tune* it; this skill covers the runtime machinery itself.

---

## Repo Context

**Repo:** `Warrenpoobear/hermes-agent`
**Version:** v0.13.0 (latest release)
**Key files:**
- `cli.py` (568KB) -- Main CLI entry point
- `AGENTS.md` (46KB) -- Agent fleet documentation
- `CONTRIBUTING.md` (28KB) -- Contributor guide
- `SECURITY.md` (7KB) -- Security advisory handling
- `hermes_constants.py` (13KB) -- Runtime constants
- `batch_runner.py` (55KB) -- Batch execution engine

---

## Session Lifecycle

### Session Start

1. **Config load:** Read `cli-config.yaml` (or equivalent) for model routing, API keys, tool permissions
2. **Skill loading:** Load SKILL.md files from configured skill directories. Skills are Markdown files created after 5+ tool calls in prior sessions.
3. **Memory load:** Load `.learnings/memory.md` (HOT tier, <=100 lines) first, then namespace-specific files on demand
4. **Agent bootstrap:** Load per-agent `SOUL.md` and `AGENTS.md` configuration
5. **Tool registration:** Register available tools based on agent authority level
6. **Session ready:** Agent begins processing

### Session End

1. **Skill creation check:** If 5+ tool calls were made, evaluate whether a new skill document should be created
2. **Memory update:** Write any corrections or learnings to `.learnings/` files
3. **Artifact output:** Write results to designated output paths
4. **Heartbeat update:** Update `HEARTBEAT.md` with session completion timestamp

---

## Model Routing

### API Configuration (as of May 2026)

| Model Pattern | API Gateway | Notes |
| --- | --- | --- |
| `llama*` | Together AI (OpenAI-compatible) | Primary for all agents |
| `claude*` | Anthropic SDK | Fallback for Claude-specific models |
| Previous | OpenRouter | Out of credits as of 2026-05-13 |

### Primary Model

**Llama 3.3 70B Instruct Turbo** (Together AI) -- all agents default to this.

### Inference Parameters (Llama-optimized)

| Parameter | Value | Rationale |
| --- | --- | --- |
| Temperature | 0.2 | Stronger governance determinism |
| Frequency penalty | 0.1 | Reduce repetition loops |
| Top_p | 0.95 | Tighter nucleus sampling |
| Repetition penalty | 1.2 | Anti-loop guard |
| API timeout | 2400s | Together can spike 8-12s cold start |
| Retry strategy | Exponential backoff | 500ms-8000ms delays |
| Compression threshold | 0.5 | Less aggressive for 131K context window |

### Gateway Monitoring

- `~/.hermes/monitor_together_latency.py` tracks latency trends
- Alerts on success rate < 80% or avg latency > 5s
- Logs to `together_latency.log`

---

## Cron Job Management

### Production Cron Schedule

| Job | Time (ET) | Frequency | Notes |
| --- | --- | --- | --- |
| Daily production pipeline | 5:30 PM | Weekdays | 13-step orchestrator |
| @reboot catch-up | On boot | -- | Catches missed runs after sleep/restart |
| Universe maintenance | 10:00 AM | Weekdays | Fixed race condition (was running before rankings.csv existed) |

### Cron Infrastructure

- **Environment:** WSL2 on Windows host
- **Sleep-cliff risk:** Windows host suspend kills crons silently
- **Stopgap:** `powercfg /change standby-timeout-ac 0` (disable sleep)
- **Missed cron signature:** 24-48h gap in `data/snapshots/`
- **Planned migration:** $15/mo Linux VPS (DigitalOcean / Hetzner). No timeline set.

### Critical Constraint

No cron job may depend on a gateway token (from operational routing policy, Lane A).

---

## Tool Execution Pipeline

### Authority Levels

| Level | What It Can Do | Who Has It |
| --- | --- | --- |
| observe_only | Read files, check status | Most monitoring agents |
| observe_and_propose | Read + suggest changes | Analysis agents |
| write_artifacts | Write to artifacts/ | Report generators |
| mutate_data | Write to data/ directories | Only `crt_resolution_watcher` |
| mutate_config | Modify configuration | No agent (operator only) |

### Exec Allowlist

The tool execution pipeline has an exec allowlist that controls which shell commands agents can run. Known bypass vectors (from Texas A&M taxonomy, 470 advisories):
- Line continuation bypass
- Busybox multiplexing
- GNU long-option abbreviation
- These compose into a complete unauthenticated RCE path from LLM tool call to host process

### Execution Lanes (Operational Routing Policy)

| Lane | Description | LLM Usage | Cron Allowed |
| --- | --- | --- | --- |
| A (Deterministic Production) | Scripts, cron, tests only | None | Yes |
| B (Cheap Monitoring) | File/JSON checks first, LLM on anomaly only | Anomaly-triggered | Yes (via `run_agent_direct.py`) |
| C (High-Token Manual) | Synthesis, audits, refactoring | Full | No (manual sessions only) |

---

## Docker Deployment

### Files

- `Dockerfile` (4.3KB) -- Multi-stage build
- `docker-compose.yml` (3.1KB) -- Service composition
- `docker/` -- Additional Docker configuration

### Current Status

Docker deployment is available but the production environment runs on WSL2, not Docker. Docker is primarily used for reproducible development environments.

---

## Agent Fleet Configuration

### Agent Count (Authoritative: agent_governance.md)

30 total agents:
- 27 active
- 1 suppressed (bioshort_watch)
- 1 retired (company_news_ingest)
- 1 shadow (shadow_watch)

### Per-Agent Configuration

Each agent has:
- `SOUL.md` -- Operating manual defining boundaries, tools, heartbeat checks
- `AGENTS.md` entry -- Fleet-wide documentation
- Authority level from registry
- Llama-specific prompting (IF/THEN chains, step numbering, schema-first output)

### Uncertainty Handling (Per-Agent Rules)

| Agent | Missing Data Response | Confidence Rule |
| --- | --- | --- |
| ops_supervisor | RED (not GUESS) | < 0.7 -> escalate |
| sentinel | FAIL | Boundary cases -> WARN |
| data_auditor | FAIL | Specific counts, not "some" |
| ic_health_monitor | UNKNOWN | Threshold boundaries -> ALERT (conservative) |
| fleet_steward | MEDIUM | Missing last_run -> anomalous (not healthy) |

---

## Monitoring Stack

| Layer | Tool | Purpose |
| --- | --- | --- |
| Heartbeat | `tools/agent_heartbeat_checks.py` | Per-agent health (HEARTBEAT.md) |
| Supervisor | `agents/ops_supervisor/supervisor.py` | Fleet-wide anomaly classification |
| Post-snapshot | `tools/run_post_snapshot_supervisor.py` | Post-pipeline task orchestration |
| Sentinel | `tools/agent_supervisor_sentinel.py` | Final watchdog |

### Anomaly Classification

| Classification | Severity | Meaning |
| --- | --- | --- |
| new | ORANGE | First occurrence |
| carried | YELLOW | Same anomaly seen yesterday (exact text match) |
| resolved | GREEN | Previously seen, now gone |

Terminal agents (e.g., ops_supervisor) are intentionally unsupervised and do not carry HEARTBEAT.md.

---

## Town-Hermes Bridge (Runtime Side)

From the Hermes side, the bridge works via `common/operator_delivery.py`:

```
Hermes job completes
  -> write ledger artifact (repo)
  -> send_operator_event(channel="town", ...)
    -> structured email to djschulz@gmail.com
    -> Town routine triggers on [Hermes] subject prefix
```

**Phase A** (dry-run mode, `OPERATOR_DELIVERY_DRY_RUN=1`): Complete.
**Phase B** (live delivery): Not yet started.

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | First Check |
| --- | --- | --- |
| Agent STALE (no heartbeat > 48h) | Cron missed or agent crashed | `crontab -l`, check `together_latency.log` |
| Pipeline timeout | AACT Monday batch or API latency | Check pipeline step that timed out |
| Herald DARK | Classification pipeline broken or dedupe failed | Check for `deduped_{date}.jsonl` file |
| CI RED | Test failure or dependency issue | Check GitHub Actions, PR #285 status |
| Together API errors | Rate limit or service outage | Check `monitor_together_latency.py` output |
| Sleep-cliff miss | Windows host suspended | Check `data/snapshots/` for gap |