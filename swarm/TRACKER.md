# Swarm Implementation Tracker

## P1 Bug Fixes

| # | Task | Status |
|---|------|--------|
| B1 | Lazy import firecrawl | DONE |
| B2 | SQLite session double-write dedup | DONE |
| B3 | Background job cleanup safety | DONE |
| B4 | Trajectory nudge pollution | DONE |
| B5 | Delegate stdout/stderr isolation | DONE |

## Swarm Core

| # | Module | Status | File |
|---|--------|--------|------|
| S1 | Core types + exceptions | DONE | types.py, exceptions.py |
| S2 | DAG task scheduler | DONE | scheduler.py |
| S3 | Worker pool manager | DONE | worker.py |
| S4 | Artifact bus | DONE | artifacts.py |
| S5 | Model router (multi-provider) | DONE | router.py |
| S6 | Agent messaging bus | DONE | messaging.py |
| S7 | Git worktree orchestration | DONE | worktree.py |
| S8 | Telemetry + monitor | DONE | monitor.py |
| S9 | Agent roles | DONE | roles.py |
| S10 | Semantic conflict resolver | DONE | merge.py |
| S11 | Orchestrator (main loop) | DONE | orchestrator.py |
| S12 | Task decomposition planner | DONE | planner.py |
| S13 | Verifier framework | DONE | verifier.py |
| I1 | CLI /swarm command | DONE | cli.py |
| I2 | Modal worker backend | DONE | backends/modal_backend.py |
| I3 | Human approval gates | DONE | approval.py |

## Provider Support (in router.py)

| Provider | Key Source | Models |
|----------|-----------|--------|
| Anthropic | `ANTHROPIC_API_KEY` | claude-opus-4, claude-sonnet-4, claude-haiku-4 |
| OpenAI | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, o3, o3-mini |
| OpenRouter | `OPENROUTER_API_KEY` | gemini-2.5-flash, gemini-2.5-pro, + any |
| Local | auto | local/qwen3.5-9b (multi-instance on Apple Silicon) |
| Remote GPU | Lambda/AWS | qwen3.5-9b-remote |

Model allowlists: `SWARM_ANTHROPIC_MODELS`, `SWARM_OPENAI_MODELS`, `SWARM_OPENROUTER_MODELS`
Local scaling: `SWARM_LOCAL_MAX_CONCURRENT` (default: 2)

## Next Wave

| # | Task | Status |
|---|------|--------|
| N1 | Wire /swarm into cli.py COMMANDS dict | DONE |
| N2 | Real AIAgent execution in orchestrator._execute_task | DONE |
| N3 | Lambda Labs / AWS remote GPU backend | DEFERRED |
| N4 | Retrieval index over code + artifacts | DEFERRED |
| N5 | RL data pipeline from swarm traces | DEFERRED |
| N6 | Checkpoint/resume for subtree state | DEFERRED |
| N7 | Distributed browser pool | DEFERRED |
| N8 | Watch agents (logs, CI, deploys) | DEFERRED |

## Session 2 Features

| # | Task | Status |
|---|------|--------|
| F1 | /context command (ASCII context gauge) | DONE |
| F2 | /swarm wired into CLI with task planner | DONE |
| F3 | Real AIAgent in LocalWorkerBackend | DONE |
| F4 | Premature quit detection + nudge | DONE |
| F5 | Live swarm dashboard (Rich Live) | DONE |
| F6 | Ctrl+B background jobs + /jobs + /fg | DONE |
| F7 | 86 swarm module tests | DONE |
| F8 | 32+ headless CLI tests | DONE |
| F9 | BudgetExceededError kwargs fix | DONE |
| F10 | SOUL.md test isolation fix | DONE |
