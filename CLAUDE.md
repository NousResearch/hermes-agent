# Hermes Agent - Claude Code Guide

This file is the Claude Code entry point for this repository. The canonical,
more complete agent/developer guide is `AGENTS.md`; use it as the source of
truth when details differ.

## MANDATORY: Fetch Hermes Context First

**Before doing ANY work in this repo, load fleet context via Hermes MCP.** Do not
rely on cached or stale context from previous sessions.

**Canonical reference:** [Cursor & Hermes](website/docs/user-guide/features/cursor-hermes.md)
(source-of-truth hierarchy, skills-only vs gateway mode, `HERMES_AGENTS_DIR`).

**Cursor rule:** `.cursor/rules/hermes-fleet.mdc` mirrors this checklist.

### Session Start Checklist

Prefer `fleet_context_snapshot()` when available; otherwise run:

1. **`skills_list()`** -- discover all custom agents and their SOUL.md files.
   This tells you what agents exist, which have skills documents, and where
   they live on disk.

2. **`agents_list(include_heartbeat=true)`** -- load the full agent registry
   with lane assignments, authority levels, status, and live heartbeat data.
   This is how you know which agents are healthy vs. stale.

3. **`learnings_read()`** -- read the HOT-tier persistent memory
   (`.learnings/memory.md`). This contains corrections, patterns, and
   operational knowledge the fleet has accumulated. Respect the 100-line cap.

4. **`knowledge_read(artifact="latest_state")`** -- load the current knowledge
   layer state. This is the fleet's shared understanding of production status,
   anomalies, and operational context.

5. **`knowledge_read(artifact="held_spec_ledger")`** -- check for any held
   specifications that constrain what changes are allowed.

### Before Modifying a Specific Agent

When the task involves a specific agent (e.g. "fix herald", "update bellringer"):

1. **`skills_read(name="<agent_name>")`** -- read the agent's SOUL.md. This
   defines the agent's identity, purpose, constraints, and behavioral rules.
   You must understand the SOUL.md before changing anything about the agent.

2. **`agents_get(name="<agent_name>")`** -- get the full agent detail including
   registry entry, heartbeat status, and file listing.

3. **`knowledge_read(artifact="contradiction_ledger")`** -- check for known
   contradictions or discrepancies that may affect this agent.

### Before Modifying Pipeline or Infrastructure

When the task involves pipeline code, cron, CI, or infrastructure:

1. **`artifacts_list()`** -- browse the artifacts directory to understand what
   operational outputs exist and their structure.

2. **`knowledge_read(artifact="operator_brief")`** -- read the latest daily
   operator brief for current production status and known issues.

3. **`learnings_read(file="projects/")`** -- list project-specific memory files
   for relevant namespace context.

### Why This Matters

The Hermes fleet has 30 agents with custom SOUL.md files, a 5-tier authority
model, 3 execution lanes, and a 4-layer monitoring stack. Changes that ignore
this context risk:

- Breaking agent authority boundaries
- Violating held specifications
- Contradicting fleet learnings
- Introducing regressions into a production pipeline that runs daily

**Read first. Then code.**

---

## Start Here

- Work from the current git branch unless the user asks you to switch.
- Prefer the repo's existing patterns and helper APIs over new abstractions.
- Do not revert unrelated user changes in the working tree.
- Keep edits scoped to the request and the affected subsystem.

## Environment

```bash
source .venv/bin/activate  # or: source venv/bin/activate
```

`scripts/run_tests.sh` is the required test wrapper. It probes `.venv`, `venv`,
and the shared Hermes checkout venv, then runs pytest with CI-like environment
settings.

## Test Commands

```bash
scripts/run_tests.sh
scripts/run_tests.sh tests/gateway/
scripts/run_tests.sh tests/tools/test_delegate.py::TestBlockedTools
.venv/bin/ruff check .
```

Do not call `pytest` directly unless there is no alternative; the wrapper
normalizes credentials, HOME, timezone, locale, and worker count.

## Important Project Invariants

- Profile-aware state paths must use `get_hermes_home()` from
  `hermes_constants`; user-facing path text should use `display_hermes_home()`.
- Tests must not write to a real `~/.hermes/`; use the existing fixtures and set
  `HERMES_HOME` when mocking home directories.
- Prompt caching must not be broken mid-conversation. Slash commands that alter
  tools, skills, memory, or system prompt state should defer invalidation unless
  an explicit `--now` flow exists.
- Built-in tools require both registration in `tools/*.py` and exposure through
  `toolsets.py`.
- Plugin capabilities should use generic plugin hooks/surfaces; do not hardcode
  plugin-specific logic into core files.

## High-Value Files

- `run_agent.py` - `AIAgent`, conversation loop, interrupts, compression.
- `model_tools.py` - tool discovery, schema filtering, function dispatch.
- `toolsets.py` - toolset definitions and platform bundles.
- `cli.py` - classic CLI and slash-command dispatch.
- `gateway/run.py` - messaging gateway runner.
- `hermes_cli/config.py` - default config and config migration.
- `tools/` - built-in tool implementations.
- `plugins/` - plugin systems and bundled plugins.
- `tests/` - pytest suite.

## MCP Server & Skills Integration

The Hermes MCP server (`mcp_serve.py`) runs as a stdio MCP server that Cursor
and Claude Code connect to automatically via `.cursor/mcp.json`. It provides
two tool surfaces:

### Messaging Tools (10 tools)

Conversations, messages, events, approvals across connected platforms:
`conversations_list`, `conversation_get`, `messages_read`,
`attachments_fetch`, `events_poll`, `events_wait`, `messages_send`,
`channels_list`, `permissions_list_open`, `permissions_respond`

### Skills & Knowledge Tools (hermes_skills_mcp.py, 7 tools)

Read-only access to the custom Hermes agent fleet, skills, knowledge layer,
and persistent memory.

| Tool | Purpose |
|------|---------|
| `skills_list` | List all agent SOUL.md files and repo skills |
| `skills_read` | Read a specific SOUL.md or skill document |
| `agents_list` | List agents with registry data and optional heartbeat |
| `agents_get` | Full agent detail: registry, SOUL.md, heartbeat, files |
| `knowledge_read` | Read knowledge layer artifacts (latest_state, ledgers) |
| `learnings_read` | Read .learnings/ memory files (HOT/WARM/COLD tiers) |
| `artifacts_list` | Browse the artifacts/ directory tree |

**Key paths** (resolved via HERMES_HOME and HERMES_REPO):
- `agents/` - Custom agent directories, each with SOUL.md, HEARTBEAT.md
- `agents/AGENT_REGISTRY.json` - Authoritative agent fleet manifest
- `artifacts/ops/knowledge_layer/` - Knowledge layer state files
- `artifacts/ops/held_spec_ledger/` - Held specification tracking
- `.learnings/memory.md` - HOT-tier persistent memory (100-line cap)
- `skills/` - Upstream OpenClaw skill categories

### Architecture Notes

- `hermes_skills_mcp.py` is a standalone module imported by `mcp_serve.py`
- All tools are **read-only** -- no mutation of skills, registry, or artifacts
- Gracefully degrades: if `hermes_skills_mcp` import fails, the messaging
  tools still work (logged at DEBUG level)
- Path resolution uses HERMES_HOME/HERMES_REPO env vars, same as the rest
  of the codebase

## Governance Constraints

**These constraints are active and must not be violated:**

1. **Read-only MCP access.** The skills MCP tools expose Hermes data for
   reading only. Do not attempt to write to `.learnings/`, `artifacts/`,
   `agents/AGENT_REGISTRY.json`, or any knowledge layer file through MCP
   or by circumventing the read-only surface.

2. **No Town-to-Hermes feedback automation.** The Town-Hermes Feedback
   Protocol is FROZEN until after h20d (May 26, 2026). Do not implement
   automated memory sync, contradiction-ledger routing, or `.learnings/`
   write paths. This is a governance decision, not a technical limitation.

3. **Held specifications.** Check the held_spec_ledger before making changes.
   If a specification is held, do not modify the constrained area without
   explicit operator approval.

4. **Authority model.** Respect the 5-tier authority model when modifying
   agent configurations. Most agents are `observe_only` or
   `observe_and_propose`. Only `crt_resolution_watcher` has `mutate_data`.
   No agent has `mutate_config` -- that is operator-only.

5. **Lane constraints.** Lane A agents (deterministic) must not depend on
   LLM gateway tokens. Lane B agents use LLM on anomaly only. Lane C
   agents are manual-only, no cron.

## Recent CI/PR Notes

This branch contains audit fixes around:

- subagent blocked-tool enforcement,
- `AIAgent.close()` cleanup of shared terminal/background resources,
- Google Chat plugin platform registration and Pub/Sub handoff,
- setup-provider config resync,
- gateway runtime env reload authority,
- concurrent interrupt test scaffolding.

When touching these areas, rerun the focused tests listed in the PR body before
committing.

---

## Hermes Skills Reference (Town-Sourced)

The following sections are exported from the Town AI skill library for Cursor/Claude Code context.
They encode operational knowledge about the Hermes runtime, LLM configuration, pipeline diagnostics,
daily production operations, and the Town-Hermes feedback protocol.

**These are reference material.** Do not modify production systems based on this content without
explicit operator approval. Respect the governance constraints in the main CLAUDE.md above.

---

## Skill: hermes-runtime

### Repo Context

**Repo:** `Warrenpoobear/hermes-agent`
**Version:** v0.13.0 (latest release)
**Key files:**
- `cli.py` (568KB) -- Main CLI entry point
- `AGENTS.md` (46KB) -- Agent fleet documentation
- `CONTRIBUTING.md` (28KB) -- Contributor guide
- `SECURITY.md` (7KB) -- Security advisory handling
- `hermes_constants.py` (13KB) -- Runtime constants
- `batch_runner.py` (55KB) -- Batch execution engine

### Session Lifecycle

1. **Config load:** `cli-config.yaml` for model routing, API keys, tool permissions
2. **Skill loading:** SKILL.md files from configured skill directories (created after 5+ tool calls)
3. **Memory load:** `.learnings/memory.md` (HOT tier, <=100 lines), then namespace-specific on demand
4. **Agent bootstrap:** Per-agent `SOUL.md` and `AGENTS.md` configuration
5. **Tool registration:** Based on agent authority level
6. **Session ready**

### Session End

1. Skill creation check (if 5+ tool calls, evaluate new skill doc)
2. Memory update to `.learnings/` files
3. Artifact output to designated paths
4. Heartbeat update (`HEARTBEAT.md`)

### Model Routing (May 2026)

| Model Pattern | API Gateway | Notes |
| --- | --- | --- |
| `llama*` | Together AI (OpenAI-compatible) | Primary for all agents |
| `claude*` | Anthropic SDK | Fallback for Claude-specific |
| Previous | OpenRouter | Out of credits 2026-05-13 |

Primary: **Llama 3.3 70B Instruct Turbo** (Together AI)

### Inference Parameters (Llama-optimized)

| Parameter | Value | Rationale |
| --- | --- | --- |
| Temperature | 0.2 | Governance determinism |
| Frequency penalty | 0.1 | Reduce repetition |
| Top_p | 0.95 | Tighter nucleus |
| Repetition penalty | 1.2 | Anti-loop |
| API timeout | 2400s | Together cold start spikes (8-12s) |
| Retry | Exponential backoff | 500ms-8000ms |
| Compression threshold | 0.5 | 131K context window |

### Gateway Monitoring

- `~/.hermes/monitor_together_latency.py` tracks latency trends
- Alerts on success rate < 80% or avg latency > 5s
- Logs to `together_latency.log`

### Cron Schedule

| Job | Time (ET) | Frequency | Notes |
| --- | --- | --- | --- |
| Daily production pipeline | 5:30 PM | Weekdays | 13-step orchestrator |
| @reboot catch-up | On boot | -- | Catches missed runs after sleep/restart |
| Universe maintenance | 10:00 AM | Weekdays | Fixed race condition |

### Cron Infrastructure

- **Environment:** WSL2 on Windows host
- **Sleep-cliff risk:** Windows host suspend kills crons silently
- **Stopgap:** `powercfg /change standby-timeout-ac 0` (disable sleep)
- **Missed cron signature:** 24-48h gap in `data/snapshots/`
- **Planned migration:** $15/mo Linux VPS (DigitalOcean / Hetzner). No timeline set.

**Critical:** No cron job may depend on a gateway token (Lane A constraint).

### Authority Levels

| Level | Capability | Holders |
| --- | --- | --- |
| observe_only | Read files, check status | Most monitoring agents |
| observe_and_propose | Read + suggest changes | Analysis agents |
| write_artifacts | Write to artifacts/ | Report generators |
| mutate_data | Write to data/ | Only `crt_resolution_watcher` |
| mutate_config | Modify configuration | No agent (operator only) |

### Exec Allowlist Security

The tool execution pipeline has an exec allowlist. Known bypass vectors (Texas A&M taxonomy, 470 advisories):
- Line continuation bypass
- Busybox multiplexing
- GNU long-option abbreviation
- These compose into a complete unauthenticated RCE path from LLM tool call to host process

### Execution Lanes

| Lane | Description | LLM Usage | Cron |
| --- | --- | --- | --- |
| A (Deterministic) | Scripts, cron, tests only | None | Yes |
| B (Cheap Monitoring) | File/JSON checks first, LLM on anomaly | Anomaly-triggered | Yes (via `run_agent_direct.py`) |
| C (High-Token Manual) | Synthesis, audits, refactoring | Full | No (manual only) |

### Agent Fleet (30 total)

- 27 active, 1 suppressed (bioshort_watch), 1 retired (company_news_ingest), 1 shadow (shadow_watch)
- Each agent has: `SOUL.md`, `AGENTS.md` entry, authority level, Llama-specific prompting (IF/THEN chains, step numbering, schema-first output)

### Uncertainty Handling (Per-Agent Rules)

| Agent | Missing Data Response | Confidence Rule |
| --- | --- | --- |
| ops_supervisor | RED (not GUESS) | < 0.7 -> escalate |
| sentinel | FAIL | Boundary cases -> WARN |
| data_auditor | FAIL | Specific counts, not "some" |
| ic_health_monitor | UNKNOWN | Threshold boundaries -> ALERT (conservative) |
| fleet_steward | MEDIUM | Missing last_run -> anomalous (not healthy) |

### Monitoring Stack

| Layer | Tool | Purpose |
| --- | --- | --- |
| Heartbeat | `tools/agent_heartbeat_checks.py` | Per-agent health |
| Supervisor | `agents/ops_supervisor/supervisor.py` | Fleet-wide anomaly |
| Post-snapshot | `tools/run_post_snapshot_supervisor.py` | Post-pipeline orchestration |
| Sentinel | `tools/agent_supervisor_sentinel.py` | Final watchdog |

### Anomaly Classification

| Classification | Severity | Meaning |
| --- | --- | --- |
| new | ORANGE | First occurrence |
| carried | YELLOW | Same anomaly seen yesterday (exact text match) |
| resolved | GREEN | Previously seen, now gone |

Terminal agents (e.g., ops_supervisor) are intentionally unsupervised.

### Docker Deployment

- `Dockerfile` (4.3KB), `docker-compose.yml` (3.1KB), `docker/` directory
- Available but production runs on WSL2, not Docker
- Docker primarily for reproducible dev environments

### Town-Hermes Bridge (Runtime Side)

Module: `common/operator_delivery.py`

```
Hermes job completes
  -> write ledger artifact (repo)
  -> send_operator_event(channel="town", ...)
    -> structured email to djschulz@gmail.com
    -> Town routine triggers on [Hermes] subject prefix
```

Phase A (dry-run, `OPERATOR_DELIVERY_DRY_RUN=1`): Complete.
Phase B (live delivery): Not started.

### Troubleshooting

| Symptom | Likely Cause | First Check |
| --- | --- | --- |
| Agent STALE (no heartbeat > 48h) | Cron missed or crash | `crontab -l`, `together_latency.log` |
| Pipeline timeout | AACT Monday batch or API latency | Check timed-out step |
| Herald DARK | Classification broken or dedupe failed | Check `deduped_{date}.jsonl` |
| CI RED | Test failure or dependency | GitHub Actions, PR status |
| Together API errors | Rate limit or outage | `monitor_together_latency.py` |
| Sleep-cliff miss | Windows host suspended | `data/snapshots/` gap |

---

## Skill: hermes-llm-config

### Key Facts

1. OpenAI is NOT a first-class Hermes provider (no `OPENAI_API_KEY` env var)
2. `OPENAI_BASE_URL` and `LLM_MODEL` env vars removed -- `config.yaml` is single source of truth
3. Secrets in `.env`, settings in `config.yaml`
4. Precedence: CLI args > `~/.hermes/config.yaml` > `~/.hermes/.env` > defaults

### Provider Access Paths

| Path | How |
| --- | --- |
| OpenRouter (recommended) | `OPENROUTER_API_KEY` in `.env` -- 200+ models |
| OpenAI Codex | `hermes model` -- OAuth device code flow |
| Custom Endpoint | Point at `https://api.openai.com/v1` as compatible endpoint |
| GitHub Copilot | OAuth -- accesses GPT-5.x through Copilot API |

### Recommended config.yaml

```yaml
model:
  provider: openrouter
  default: anthropic/claude-sonnet-4-6

custom_providers:
  - name: openai-direct
    base_url: https://api.openai.com/v1
    key_env: OPENAI_API_KEY

fallback_providers:
  - provider: openrouter
    model: openai/gpt-5.3-codex
```

### .env (target state)

```bash
# ~/.hermes/.env (chmod 0600)
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-proj-...
```

### Setup Commands

```bash
# Interactive wizard (run from terminal, outside session)
hermes model

# Or configure directly
hermes config set OPENROUTER_API_KEY sk-or-...
hermes config set model anthropic/claude-sonnet-4-6

# Verify
hermes config check
```

### Mid-Session Switching

```
/model custom:openai-direct:gpt-5.3-codex        # Direct OpenAI billing
/model openrouter:anthropic/claude-sonnet-4-6     # Back to OpenRouter default
/model openrouter:anthropic/claude-opus-4-7       # Upgrade for complex reasoning
```

### Model Recommendations (May 2026)

**Daily coding:** Claude Sonnet 4.6 ($3/$15 per 1M), Qwen 3.6 Plus (~$0.56/hr), GPT-5.3-Codex
**Complex reasoning:** Claude Opus 4.7 ($5/$25), GPT-5.5 (1M context), Gemini 3.1 Pro ($2/$12, 2M context)
**Budget/volume:** GPT-5 Nano ($0.05/M input), Haiku 4.5 ($0.80/M), DeepSeek V3.2 (~$0.09/hr)

### CCFT-Aware Routing

| Code Surface | Minimum Model |
| --- | --- |
| CCFT enforcement, selector, scoring, catalyst | Sonnet 4.6 or Opus 4.7 |
| Walk-forward harness, SHA256 hash flows | Sonnet 4.6 |
| Tests, utility scripts, data ingestion | GPT-5.3-Codex or Sonnet 4.6 |
| Documentation, non-scoring agent code | GPT-5.3-Codex |
| Log summarization, terminal output | Deterministic scripts (no LLM) |

### Directory Structure

```
~/.hermes/
  config.yaml     # Settings (model, provider, terminal, compression)
  .env            # API keys and secrets (chmod 0600)
  auth.json       # OAuth credentials (Nous Portal, Copilot, etc.)
  SOUL.md         # Agent identity/persona
  memories/       # Persistent memory (MEMORY.md, USER.md)
  skills/         # Agent-created skills
  cron/           # Scheduled jobs
  sessions/       # Gateway sessions
  logs/           # Logs (secrets auto-redacted)
```

### Local LLM via Ollama (16GB RAM Constraint)

On 16GB RAM Surface hardware, local models are Tier 6 (deferred).

If experimenting locally:
- Stick to 7-14B models: Qwen3 9B, Phi-4 14B, Llama 3.3 8B
- Set `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_CONTEXT_LENGTH=4096`
- Use Q4_K_M or Q5_K_M quantization
- Revisit at 32GB+ RAM or 24GB+ VRAM

### Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| "No provider configured" | Missing provider in config.yaml | Run `hermes model` wizard |
| "Invalid API key" on OpenRouter | Stale/missing key in `.env` | Regenerate at openrouter.ai/keys |
| Can't `/model` switch | Provider not in `custom_providers` | Add named provider entry |
| "OPENAI_BASE_URL not found" | Using removed legacy env var | Migrate to config.yaml `model:` block |
| Slow/OOM on local model | Model too large for RAM | Drop to 8-9B, reduce context to 4K |

### Claude Code Inspection Task

Paste into a fresh Hermes session to audit and configure providers:

```
Inspect the Hermes provider configuration and current setup.

Goal:
Configure Hermes to use Claude Sonnet 4.6 via OpenRouter as primary model,
with OpenAI GPT-5.3-Codex (direct API from my Platform business org) as named
custom provider for on-demand switching.

Rules:
- Primary: OpenRouter with OPENROUTER_API_KEY
- Alternate: OpenAI direct via custom_providers with key_env: OPENAI_API_KEY
- Do not overwrite existing config; merge changes
- Add fallback_providers for resilience
- Show exact files changed and commands to validate
- Do not print API keys

Check:
- ~/.hermes/config.yaml (current model, provider, custom_providers)
- ~/.hermes/.env (existing keys)
- hermes config check (missing options)

Then propose the smallest config change.
```

---

## Skill: pipeline-diagnostics

### Diagnostic Approach

1. Confirm symptoms from email signals
2. Check GitHub CI status
3. Cross-reference stale agent checklist
4. Build priority-ordered fix list

### Agent Inventory (30 total, 27 active)

**Lane A: Deterministic (5 agents)**
- `aact_trial_ingest` -- CT.gov clinical trial data
- `ctgov_poller` -- CT.gov polling
- `earnings_calendar_sync` -- earnings date sync
- `herald` -- daily press release digest (HIGHEST PRIORITY when dark)
- `universe_maintenance` -- ticker universe upkeep

**Lane B: Monitoring + Escalation (18 agents)**
- `bellringer` -- biotech earnings preview + results
- `biotech_news_digest`, `calibration`, `catalyst_delta`, `crt_resolution_watcher`
- `data_auditor`, `event_analyst`, `grok_biotech_watch`, `ic_health_monitor`
- `intraday_mover_watch`, `options_watch`, `policy_shadow_watch`, `postmortem`
- `price_action_watch`, `shadow_monitor`, and others

**Lane C: Manual Engineering (7 agents)**
- `fleet_steward` -- fleet health monitoring
- `ops`, `ops_supervisor`, `production_qa`, `qa`, `review_queue_steward`, `sentinel`

**Deprecated/Suppressed (3)**
- `company_news_ingest` (retired, replaced by herald)
- `bioshort_watch` (suppressed)
- `shadow_watch` (placeholder)

### Triage Priority Order

When multiple agents are down, fix in this order:

1. **hermes-mail bridge** -- if down, no agent can deliver email. Fix SMTP first.
2. **Herald Digest** -- highest-value daily signal. Dark Herald = blind on press releases.
3. **Bellringer** -- earnings preview/results. Check preview vs. results separately (different cron jobs).
4. **fleet_steward** -- restores health monitoring for everything else.
5. **Intraday Mover Watch** -- real-time price alerts during market hours.
6. **grok_biotech_watch** -- depends on XAI API key, often longest outage.
7. **Evening forward-shadow** -- watchdog pattern, lower priority.
8. **postmortem memory-write** -- cosmetic mtime bug, lowest priority.

### Herald Dark Diagnosis

```bash
crontab -l | grep herald
ls -la ~/.hermes/logs/ | grep herald
tail -50 logs/daily_production_*.log | grep -i herald
python3 run_agent_direct.py --agent herald
python3 run_agent_direct.py --agent herald --skip-preflight
```

Root causes:
- Timeout budget exceeded (sequential IR fetch over 341 tickers, 23+ min in sleeps)
- Fix: PR #266 parallelized fetcher (ThreadPoolExecutor, 12 workers, ~3-5 min)
- hermes-mail bridge down (Herald runs but email never arrives)
- Cron entry missing (WSL2 restart can lose entries)
- Preflight gate blocking (dirty git state, missing snapshot, governance hold)

### Bellringer Results Dark

Previews and results are different cron jobs. Previews working + results dark = results job failing silently.

```bash
crontab -l | grep -i bellringer
cat logs/bellringer_results_*.log | tail -50
python3 run_agent_direct.py --agent bellringer_results --date 2026-05-14
```

### hermes-mail Bridge

```bash
python3 hermes_mail_smoke_test.py
cat ~/.hermes/.env | grep SMTP
```

Fix: regenerate Gmail app password at https://myaccount.google.com/apppasswords
Success criteria: smoke test email arrives in djschulz@gmail.com.

### grok_biotech_watch Dark (XAI API Key)

```bash
grep XAI_API_KEY ~/.hermes/.env
# If missing/expired: regenerate at console.x.ai
# Add to .env: XAI_API_KEY=xai-xxxxx
python3 run_agent_direct.py --agent grok_biotech_watch
# 401/403 = key invalid, regenerate
# Rate limit = check xAI account billing
```

### WSL2 Cron Issues

```bash
sudo service cron status
# If stopped: sudo service cron start
crontab -l
# Should include: herald, fleet_steward, hermes_mail, bellringer, intraday_mover, evening forward-shadow
# PR #269 (open) adds WSL2-sleep-resilient catchup for evidence builds
```

### CI Failure Patterns

| Failure | Root Cause | Fix |
| --- | --- | --- |
| pytest CVE | Security advisory | Upgrade in requirements.txt |
| Agent registry enum | Invalid status values (`suppressed`, `retired`) | Change to valid enum (`deprecated`) |
| Ruleset drift test | New source not in allowed list | Add to `test_decision_ruleset.py` |
| Critical F821/F811 | Undefined variables, unused imports | Fix in source, run flake8 |
| Universe loading: 1 ticker | Stale `ipo_dates.json` | Update all `last_price_date` |
| Intraday mover NO_DATA | Poll before snapshot | Shift first poll to after ~10:30 ET |
| dep-audit failures | Outdated deps with known CVEs | Merge dependabot PRs |
| type-check failures | mypy version drift | Bump mypy, fix new type errors |

### Town AI Can vs. Terminal Required

**Town AI can:**
- Search emails to confirm pipeline delivery status
- Check GitHub CI status, recent commits, open PRs
- Read commit diffs
- Create GitHub issues tracking fixes
- Send runbook emails to work address

**Terminal required (Town AI cannot):**
- Run `crontab -l` or modify cron entries
- Run `python3 run_agent_direct.py`
- Check `~/.hermes/.env` for API keys
- Read local log files
- Run hermes-mail smoke tests
- Merge PRs with failing CI

When fixes require terminal, build a priority-ordered runbook and send to dschulz@wakerobin.co.

### Email Signal Verification

```
subject:"Herald" after:2026/05/12
from:djschulz@gmail.com subject:"Bellringer" "biotech earnings" after:2026/05/12
from:djschulz@gmail.com subject:"Bellringer" "results" after:2026/05/12
subject:"Intraday Mover" OR subject:"HIGH alert" after:2026/05/12
subject:"Morning Briefing" after:2026/05/12
subject:"Catalyst Update" after:2026/05/12
```

---

## Skill: screener-ops

### Daily Production Pipeline

**Runner:** `tools/run_daily_production.py` (13-step orchestrator)
**Cron:** 5:30 PM ET weekdays + `@reboot` catch-up
**Timeout:** 6000s (100 min). Previous 4500s was killing mid-AACT on Mondays.

#### Pipeline Steps (in order)

1. Price refresh
2. Cache warm (including FDA)
3. Screen (with `--inputs-manifest write`)
4. Audit
5. Gates
6. Manifest + promotion
7. Drift report
8. Action packet
9. Shadow portfolio
10. Trade plan
11. Portfolio report
12. Readiness scorecard
13. Ops digest + PIT backfill (optional)

**Rule:** Always warm 8-K cache BEFORE running screen.

### Knowledge Layer (Spec 089)

**Generator:** `tools/build_hermes_knowledge_layer.py`

Repo-native "ops brain" that answers:
1. What is the current operational state?
2. What changed since the last good state?
3. What is held, blocked, or awaiting first-fire validation?
4. What contradictions exist across specs, audit memos, cron, and registry?
5. What is the next allowed operator action?
6. What is explicitly not allowed?

#### Four Layers

| Layer | Purpose | Output |
| --- | --- | --- |
| Capture | Read-only from specs, artifacts, registry, git, cron | Raw state |
| Normalize | Structured ledgers | `artifacts/ops/knowledge_layer/` |
| Reason | Drift, contradiction, missed-run detection | Alerts |
| Deliver | Operator briefs | Daily/weekly summaries |

#### Output Artifacts

- `artifacts/ops/knowledge_layer/latest_state.{json,md}`
- `artifacts/ops/held_spec_ledger/latest.{json,md}`
- `artifacts/ops/first_fire_ledger/latest.{json,md}`
- `artifacts/ops/contradiction_ledger/latest.md`
- `artifacts/ops/operator_brief/daily/YYYY-MM-DD.md`

### Herald Pipeline

Done predicate requires BOTH deduped AND classified JSONL:
- `data/press_releases/deduped/deduped_{date}.jsonl`
- `data/press_releases/classified/classified_{date}.jsonl`

If classification failed but dedupe exists, next supervisor run retries classification.

### Agent Fleet (30 total)

- 27 active, 1 suppressed (bioshort_watch), 1 retired (company_news_ingest), 1 shadow (shadow_watch)
- Authority source: `agents/AGENT_REGISTRY.json` (schema v1.0, as-of 2026-05-17 per agent_governance.md)
- Only `crt_resolution_watcher` holds `mutate_data` (writes to catalyst resolution tables under orchestrator supervision)

### SOUL.md / Ruleset System

**SOUL.md:** Per-agent operating manual defining boundaries, tools, and heartbeat checks. Located at `agents/{name}/SOUL.md`.

**Ruleset Health Monitor:** `tools/ruleset_health_monitor.py`
- JSONL history grows with each evaluation date (idempotent on same-day reruns)
- Tracks consecutive WARN days by active ruleset ID
- Recommends rollback after sustained degradation

### Active Ruleset

- ID: `8887576e` (v1.14.0)
- File: `production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json`
- Prior: `2a3e79eb` (v1.13.0) -- RETIRED 2026-05-04
- Pinned in: `run_screen.py` AND `run_phase2_snapshot_delta.py` (must stay in sync)
- Manifest: 36+ entries, no duplicate IDs
- Architecture freeze: until post-h20d (~2026-05-26)

### Governance Artifacts (PR #286, merged May 16, 2026)

**governance/AGENT_ROUTING_POLICY.md** -- Tier 0-4 routing policy classifying codebase by governance sensitivity. Defines allowed tools, review requirements, merge rules per tier. The policy itself is Tier 4 (changes require a memo).

**governance/STATUS.md** -- AGENT_ROUTING_POLICY.md is live. Enforcement layers pending: agent_registry.yml (PR 2), AGENT_DIRECTORY_MAP.md, CI registry validation, import-graph validation.

**governance/HASH_ROTATIONS.md** -- Required landing zone for Tier 3 production-hash rotations. Each entry: old hash, new hash, effective date, affected surface, reason, downstream impact, reviewer.

**Operational Routing** (`docs/ops/hermes_openclaw_routing_policy.md`, v1.0, effective 2026-05-15):
- Lane A (Deterministic): No LLM. Scripts, cron, tests only.
- Lane B (Cheap Monitoring): File/JSON first. LLM on anomaly only via `run_agent_direct.py`.
- Lane C (High-Token Manual): Manual sessions. No cron.
- Critical: no cron job may depend on a gateway token.

### Expectation Layer Coverage Gate (Spec 105)

**QA file:** `production_qa_check.py`
**Status:** CODE-CLOSED (commit 0ddbb509). Pending live QA.

| Field | Required Coverage | Source |
| --- | --- | --- |
| `short_interest_pct` | 0.90 | Market data provider |
| `close_price` | 0.99 | Market data provider |
| `market_cap_mm` | 0.95 | Market data provider |
| `priced_move_pct` | 0.80 | Derived (catalyst pricing model) |
| `insider_net_buy_value_90d` | 0.30 | Form 4 (diagnostic only) |

Gate behavior:
- Hard-fails pipeline at Step 5 (Gates) if any field below threshold
- Error message includes: field name, actual coverage, required threshold
- Coverage stats logged every run regardless of pass/fail
- Source of truth: `FEATURE_COVERAGE_REQUIREMENTS` (not hardcoded)

### Export Contract Registry (Spec 101)

**Status:** CLOSED (commits eaa4ea87 + cba4ee0f). `ev_severity_score` now exported.

All exported (post-Spec 101):
- `runway_severity_score`, `ev_severity_score`, `runway_buffer_months`, `financing_truth_gate`
- `dilution_haircut`, `size_multiplier`, `severity_bucket`, `severity_notes`
- `check_severity_formulas()` QA runs on every snapshot

Derived field contracts (must hold for all non-null rows):
```
dilution_haircut == 0.35 * ev_severity_score       (tolerance 1e-6)
size_multiplier == max(0.40, 1 - 0.60 * ev_severity_score)  (tolerance 1e-6)
```

Pre-v1.1 snapshot readers default `ev_severity_score` to NaN (not fail).

### Diagnostic Fields Registry (Spec 104)

| Field | Status | Meaning of Null | Meaning of 0.0 |
| --- | --- | --- | --- |
| `insider_net_buy_value_90d` | DIAGNOSTIC ONLY | Not fetched / no Form 4 coverage | Fetched, no insider buy in 90d |

### Insider Model Isolation Guard (CRITICAL)

`insider_net_buy_value_90d` must NOT enter the expectation model's `market_features` input. The model has an `insider_net_buy_z` weight that activates silently if the field flows upstream.

Guard options (at least one required):
1. **Input exclusion (preferred):** Runtime assert field NOT in `market_features` DataFrame
2. **Weight zeroing:** `insider_net_buy_z` weight = 0.0 with test
3. **Drop guard:** Pre-inference drop with logged warning

Rules:
- Never collapse blank (NaN) and zero (0.0) -- different semantics
- Never impute zero for missing or blank for zero
- CI check: flag suspicious if column ALL zero or ALL null
- Field must remain in `DIAGNOSTIC_FIELDS`, NOT `ALPHA_FEATURE_REGISTRY`
- Promotion requires: 20+ stable snapshots, >= 60% coverage, IC > 0 at p < 0.05, Checklist v2 pass, explicit written approval

### Backfill Tooling (Spec 102)

Target fields: `short_interest_pct`, `close_price`, `market_cap_mm`, `priced_move_pct` (required); `insider_net_buy_value_90d` (optional)

Key rules:
- Default: additive-only (`recompute=False`). Original ranks/actions preserved.
- Every backfill emits structured manifest (snapshot_date, fields_added, coverage before/after, recompute flag, timestamp, version)
- `_backfill_version` metadata column added to all backfilled snapshots (null for originals)
- Research scripts must filter on `_backfill_version` to avoid silent pre/post mixing
- Default scope: 30 trading days, configurable

### Spec Lifecycle States

| State | Meaning |
| --- | --- |
| DRAFT | Under development |
| IN PROGRESS | Active work, phased |
| HELD | Blocked on dependency |
| RESOLVED | All acceptance criteria met |
| SUPERSEDED / MITIGATED | Failure modes neutralized via different route |
| CLOSED | Formally closed |

### Source Files

| Component | File |
| --- | --- |
| Daily Production Runner | `tools/run_daily_production.py` |
| Knowledge Layer Builder | `tools/build_hermes_knowledge_layer.py` |
| Operator Delivery | `common/operator_delivery.py` |
| Agent Heartbeat Checks | `tools/agent_heartbeat_checks.py` |
| Ops Supervisor | `agents/ops_supervisor/supervisor.py` |
| Post-Snapshot Supervisor | `tools/run_post_snapshot_supervisor.py` |
| Ruleset Health Monitor | `tools/ruleset_health_monitor.py` |
| Ops Digest Builder | `tools/build_ops_digest.py` |
| Readiness Scorecard | `tools/weekly_readiness_scorecard.py` |
| Cron Wrapper | `tools/cron_daily_production.sh` |

### Operational State (Snapshot -- verify before citing)

**Active ruleset:** `8887576e` (v1.14.0). Architecture freeze until post-h20d (~2026-05-26).

**Infrastructure:** WSL2 on Windows host. Llama 3.3 70B via Together AI (switched 2026-05-13). Daily cron 4:30-7:30 PM ET weekdays. Universe maintenance 10:00 AM ET.

**BioShort Research (Spec 092):** All phases A-D complete. DEFER verdict: 129 samples, 60.5% accuracy at T+5. Median T+5 +0.63%, T+20 +2.49%. Pseudo-PIT caveat applies.

**Town-Hermes Bridge:** Phase A complete (dry-run). Phase B not started.

---

## Skill: town-hermes-feedback

**Status:** DRAFT / NOT ACTIVE / FROZEN until after h20d (May 26, 2026)

### Current Communication Path

```
Hermes -> email -> Town (Spec 090, operational)
  - Hermes jobs send structured emails to djschulz@gmail.com
  - Town routines trigger on [Hermes] subject prefix
  - Town creates tasks / DMs operator

Town -> Hermes (NO FORMAL PATH)
  - Town memories accumulate (20+ global)
  - Town doc review findings accumulate
  - Operator feedback via chat accumulates
  - None systematically reaches Hermes agent prompts or .learnings/ files
```

### What Town Is NOT

- NOT a scheduler or cron controller for Hermes
- NOT a repo mutator or spec approver
- NOT allowed to reactivate suppressed agents (bioshort_watch)
- NOT the authoritative source for any production state
- NOT an automatic feedback channel (all feedback is operator-mediated)

### Proposed Channels (Design Only - NOT ACTIVE)

**Channel 1: Memory Sync** (Town -> Hermes `.learnings/`)

| Town Memory Type | Hermes Target | Sync Frequency |
| --- | --- | --- |
| Global corrections (style, format) | `.learnings/memory.md` | Weekly |
| Routine-specific findings | `.learnings/projects/{routine}.md` | Weekly |
| Watchlist corrections (ticker changes) | `production_data/` config | On detection (manual) |
| 13F filing alerts (do-not-re-alert) | Hermes dedup state | On detection (manual) |

Constraint: `.learnings/memory.md` has 100-line cap. Sync must be selective.

**Channel 2: Audit Finding Routing** (Town -> Hermes Knowledge Layer)

| Finding Type | Hermes Target | Example |
| --- | --- | --- |
| Skill text error | `artifacts/ops/contradiction_ledger/latest.md` | inst_delta_z scope error |
| Infrastructure alert | `artifacts/ops/knowledge_layer/latest_state.json` | 6 stale agents |
| CI red state | Ops supervisor anomaly input | CI red 10+ days |

Constraint: Town does NOT write to Hermes repo. Routing via email or manual action.

**Channel 3: Operator Decision Feedback** (Town -> Hermes Governance)

| Decision Type | Current Path | Proposed Path |
| --- | --- | --- |
| Spec approval | Chat -> email -> manual commit | Same + structured decision record |
| Threshold calibration | Town memory update | Same + governance log entry |
| Audit finding resolution | Doc review log update | Same + resolution to affected skill |

### Self-Improving Skill Integration

Dual storage model:
- **Town:** `add_memory()` -- operator-facing store (what human sees and edits)
- **Hermes:** `.learnings/` files -- agent-facing store (what agents load at session start)

Reconciliation: manual, operator-initiated, infrequent (monthly or at milestones).
The operator decides which Town learnings are worth propagating to Hermes.

### Governance Decision (2026-05-22)

FROZEN until after h20d. No Channel 1 memory sync, no `.learnings/` write path,
no contradiction-ledger routing, no automated Town routine export, no cron/agent
activation before h20d.

Allowed before h20d: draft manual templates only.

Post-h20d sequence:
1. Pilot governance decision logging
2. Pilot manual memory sync weekly
3. Only after two clean manual cycles consider Town routine export with operator approval

### Operational Role in DEM Stack

```
Hermes (WSL2/Docker)          Town (Cloud SaaS)
  30-agent fleet         <-->   20+ routines
  .learnings/ memory     <-->   Persistent memories (20+ global)
  Skills (repo-governed) <-->   Skills (39, doc-governed)
  Knowledge layer        <-->   Content Library (3 main collections)
  Cron scheduling        <-->   Trigger-based routines
  Production pipeline    <-->   Operator interface + ad-hoc research
```

Communication: Hermes -> structured email -> Town incoming_email trigger -> routine -> operator review.
Feedback gap: Town -> Hermes has NO formal automated path. All operator-mediated.

### Town Platform Capabilities (for Hermes agent context)

**What Town CAN do:**
1. Store/retrieve structured memories (global + routine-specific, full CRUD)
2. Search and read emails (parse Hermes output for structured data)
3. Create/edit documents (Town Docs, Google Docs, Sheets)
4. Execute code (sandboxed Python, optional internet)
5. Run trigger-based routines (incoming email triggers on Hermes output)
6. Maintain Content Library (permanent, organized, shareable)
7. Manage skills (39 named skills encoding DEM methodology)
8. Federated search (email, Drive, Content Library, integrations)
9. Delegate to sub-agents (research-person, general-purpose)
10. Generate audio (text-to-speech for briefings)

**What Town CANNOT do:**
1. Write to Hermes repo (no git push, no file mutation)
2. Schedule or control Hermes cron jobs (no WSL2/systemd/Docker access)
3. Approve or reject Hermes specs (operator-mediated only)
4. Reactivate suppressed agents
5. Access Hermes runtime state (no API to fleet status/sessions/knowledge layer)
6. Initiate Hermes sessions (email-only or operator-mediated)
7. Modify `.learnings/` files
8. Access Hermes production data (`production_data/`, CIK lookups, pipeline state)
