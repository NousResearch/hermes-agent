# Hermes System Audit

Date: 2026-05-20
Branch: `hermes-control-plane-20260520-182036`
Scope: discovery and control-plane bootstrap for `/Users/agent1/Code/hermes-agent`,
`/Users/agent1/.hermes`, and `/Users/agent1/Operator/scripts`.

## Executive Diagnosis

Hermes is already a broad local-first agent platform with a CLI, messaging
gateway, API server, tool registry, plugin system, memory providers, cron, TUI,
web/dashboard surface, and Operator launchd wrappers. The current optimization
need is not a one-shot rewrite. It is a staged control plane that ties together
architecture, tools, memory, security, reliability, operations, validation, and
judge cycles.

The strongest existing primitives are:

- Central tool registry in `tools/registry.py`.
- Toolset definitions in `toolsets.py`.
- Gateway runtime in `gateway/run.py`.
- API server surface in `gateway/platforms/api_server.py`.
- CLI/status/doctor/plugin/config surfaces under `hermes_cli/`.
- Built-in memory plus external memory-provider interface.
- Launchd-backed gateway wrapper under `Operator/scripts`.
- Large pytest suite and CI workflows.

The highest-risk gaps are:

- No unified control-plane registry spanning tools, plugins, MCP, quick
  commands, cron, launchd, credentials, and health.
- Duplicate launchd labels exist; `ai.hermes.gateway` is active, while
  `com.agent1.hermes.gateway` remains present and can confuse recovery.
- Some historical runtime artifacts under `~/.hermes` are more readable than
  they should be.
- Built-in curated memory is near capacity, and holographic facts can drift
  when markdown memory entries are replaced or removed.
- Fallback routing is not configured even though at least one provider route is
  available by presence check.
- Browser and computer-use toolsets are configured/registered but have runtime
  dependency gates.

## Repo Safety State

- Repo root: `/Users/agent1/Code/hermes-agent`.
- Initial branch: `main`, behind `origin/main`.
- Working tree was already dirty before this pass. Unrelated modified files
  were not reverted.
- Safety branch created: `hermes-control-plane-20260520-182036`.
- Non-empty staged-doc rollback patch created under `.codex-backups/`.

Pre-existing tracked changes included code/tests around image generation,
gateway/API, CLI status, and related tests. Phase 0 only changed `AGENTS.md`
and new `docs/HERMES_*` documents. Phase 2 adds the staged read-only inventory
surface in `hermes_cli/control.py`, CLI registration in `hermes_cli/main.py`,
and focused tests in `tests/hermes_cli/test_control_inventory.py`.

## Language And Framework

- Primary language: Python >=3.11. `hermes doctor` reported Python 3.11.15;
  the local test runner may use a newer interpreter from the active venv.
- Packaging: `pyproject.toml`, `setup.py`, console scripts `hermes`,
  `hermes-agent`, `hermes-acp`.
- CLI libraries include Rich and prompt_toolkit.
- Messaging/API extras include aiohttp, python-telegram-bot, discord.py, Slack,
  Home Assistant, SMS, etc.
- Node surfaces exist for TUI/web/browser tooling; `package.json`, `ui-tui/`,
  `web/`, and `website/` are present.
- Dependencies are exact-pinned in `pyproject.toml`, with lazy extras for
  provider/tool-specific capabilities.

## Existing Folder Structure

Important load-bearing paths:

- `run_agent.py`: `AIAgent` runtime and conversation loop.
- `model_tools.py`: tool schema resolution and function dispatch.
- `toolsets.py`: canonical tool groups.
- `tools/`: built-in tool implementations and registry integration.
- `gateway/`: messaging gateway, session management, status, platforms,
  API server.
- `hermes_cli/`: CLI commands, config, plugins, status, doctor, logs, setup,
  dashboard.
- `agent/`: provider adapters, prompt/memory/context/redaction/guardrail
  internals.
- `plugins/`: memory, model provider, web/search, image/video, platform,
  observability, and standalone plugin roots.
- `cron/`: scheduler and job model.
- `skills/` and `optional-skills/`: bundled skills.
- `tests/`: broad pytest suite.
- `ui-tui/`, `web/`, `website/`: frontend and documentation surfaces.

## Existing Agent Architecture

`AIAgent` is the core execution object. It accepts provider/model/config/tool
settings, builds prompts, calls providers, handles tool calls, manages memory
provider lifecycle, records sessions, and enforces iteration budgets.

The gateway is the live operator nucleus. `GatewayRunner` owns platform
adapters, active sessions, approvals, delivery, cron integration, hooks, and
runtime state. The CLI and API server are separate surfaces but converge on the
same core agent/tool architecture.

Target implication: optimization orchestration should not be embedded directly
into `gateway/run.py`. It should be implemented as a plugin-backed
stage/policy/ledger layer using existing rails.

## Existing Tool And Plugin System

Tool architecture:

- `tools/registry.py` is the canonical registry.
- Tool modules self-register schema, handler, toolset, availability check, and
  metadata.
- `toolsets.py` groups tools for CLI, gateway, cron, and specialized profiles.
- `model_tools.py` resolves schemas and dispatches calls.

Plugin architecture:

- `hermes_cli/plugins.py` discovers bundled, user, project, and pip plugins.
- Standalone plugins are opt-in; backend/platform/exclusive categories have
  specialized loading rules.
- Hooks include pre/post tool call, transform tool result, transform LLM output,
  pre/post LLM call, gateway pre-dispatch, approval lifecycle, and session
  lifecycle.

Observed tool status:

- `hermes tools list` showed many core toolsets enabled.
- Disabled or unavailable lanes include video generation, X search, Home
  Assistant, Spotify, Yuanbao, computer use, and some credential-gated platform
  adapters.
- MCP server `playwright` is configured.

## Existing Memory System

Memory layers:

- Built-in curated memory: `~/.hermes/memories/MEMORY.md`.
- User profile memory: `~/.hermes/memories/USER.md`.
- Session/transcript recall: `~/.hermes/state.db` and `~/.hermes/sessions/`.
- Structured external provider: active `holographic` provider using
  `~/.hermes/memory_store.db`.
- Response store: `~/.hermes/response_store.db`.

Current issue:

- `MEMORY.md` and `USER.md` are near configured character limits.
- Holographic memory mirrors built-in additions, but replace/remove consistency
  needs explicit reconciliation.
- Auto-prune is disabled, which supports recall but increases privacy and disk
  retention obligations.

## Existing Model And Provider Configuration

Provider/model resolution is centralized through `hermes_cli/runtime_provider.py`
and provider helpers under `agent/` and `providers/`.

Observed safely:

- OpenAI Codex auth is logged in.
- OpenRouter API connectivity is OK.
- No fallback providers are configured.
- Several OAuth/provider lanes are not logged in or missing credentials.
- Auxiliary model routes exist in config structure for vision, extraction,
  compression, session search, approval, MCP, title generation, curator, etc.

No secrets were printed or copied.

## Environment And Config Files

Relevant runtime files:

- `~/.hermes/config.yaml`
- `~/.hermes/.env`
- `~/.hermes/auth.json`
- `~/.hermes/gateway_state.json`
- `~/.hermes/logs/*`
- `~/Library/LaunchAgents/ai.hermes.gateway.plist`
- `~/Library/LaunchAgents/com.agent1.hermes.gateway.plist`

File modes observed for core config/auth files are owner-only. However, some
runtime artifacts and historical dumps need permission hardening.

## Existing Tests And CI

Test runner:

- `scripts/run_tests.sh` activates `.venv`/`venv`, unsets credential-shaped
  environment variables, pins deterministic env, and runs pytest with xdist.

CI:

- `.github/workflows/tests.yml` runs non-integration tests and e2e tests.
- Additional workflows cover lint, docs, OSV, supply-chain audit, Nix, skills
  index, Docker publish, PyPI upload, and lockfile checks.

High-value existing coverage includes gateway API auth/health, bind guards,
platform reconnects, startup failure handling, redaction, logging, shutdown
forensics, memory provider behavior, CLI commands, tool registry, command
guards, and terminal/file tools.

## Startup And Runtime Method

Active runtime:

- Launchd label: `ai.hermes.gateway`.
- Program: `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- Gateway logs: `~/.hermes/logs/gateway.log` and
  `~/.hermes/logs/gateway.error.log`.
- Health guardian: `ai.hermes.health-guardian`, currently pulse-first.

The wrapper sets `HOME`, `HERMES_HOME`, cache dirs, sources
`Operator/scripts/hermes-env.sh`, changes into the repo, and executes:

```bash
hermes gateway run --accept-hooks --replace
```

Do not replace this with a bare launchd command unless the same Keychain-loaded
environment contract is preserved.

## Logs And Observability

Primary log/status surfaces:

- `~/.hermes/logs/agent.log`
- `~/.hermes/logs/errors.log`
- `~/.hermes/logs/gateway.log`
- `~/.hermes/logs/gateway.error.log`
- `~/.hermes/logs/gateway-exit-diag.log`
- `~/.hermes/logs/gateway-shutdown-diag.log`
- `~/.hermes/gateway_state.json`
- `/Users/agent1/Operator/health-loop/status.json`
- `/Users/agent1/Operator/health-loop/status.md`

Redaction is enabled by default in `agent/redact.py`.

## Existing Integrations

Repo/runtime integrations include:

- Telegram and API server currently connected through the live gateway.
- Other platform adapters: Discord, Slack, WhatsApp, Signal, Matrix,
  Mattermost, email, SMS, Home Assistant, Google Chat, Teams, Feishu, WeCom,
  Weixin, QQBot, BlueBubbles, Dingtalk, IRC/SimpleX plugins, etc.
- MCP server config for Playwright.
- Cron scheduler.
- Kanban multi-agent coordination.
- Mission Control and Operator command-center scripts.
- Toolbelt scripts for security/deps/CI/SBOM/SAST.
- Browser, image generation, TTS/STT, video, X search, web search, and
  provider plugin ecosystems.

## Existing Failure Points

1. Duplicate gateway label confusion between `ai.hermes.gateway` and
   `com.agent1.hermes.gateway`.
2. Wrapper bypass can break Keychain-backed auth.
3. Fallback provider chain is empty.
4. Browser enabled but `agent-browser` path/probe can fail.
5. Computer-use registered but missing external dependency gate.
6. Curated memory is nearly full.
7. Holographic memory can retain stale facts after markdown replace/remove.
8. Some runtime artifacts are world-readable.
9. Broad Docker env forwarding and host mounts need stricter per-job policy.
10. Smart approvals and quick commands need clearer risk-tier gates.
11. Disk free is low enough to deserve ongoing status reporting.

## Security Risks

High:

- Historical sessions/request dumps can contain private prompts or tool output
  and need strict permissions plus backfill.

Medium:

- Broad token forwarding into Docker should be opt-in per job.
- Smart approval default is not strong enough for a high-autonomy operator.
- Destructive slash confirmation is not currently the safest default.
- Quick commands are powerful exec shortcuts and need registry/risk metadata.

Strong existing controls:

- Keychain-backed wrapper for secrets.
- Owner-only core `.env`, `auth.json`, and `config.yaml`.
- Redaction defaults on.
- Hardline command blocklist.
- ACP edit approval.
- API server auth for detailed health/models when key exists.

## Missing High-Leverage Capabilities

- Read-only generated control-plane registry for tools/plugins/MCP/cron/quick
  commands/launchd/operator scripts/credentials/health.
- Stage-based optimization ledger and policy router.
- Audit JSONL for gated actions.
- Strict session/dump permissions.
- Memory compaction and deletion checklist.
- Fallback provider configuration.
- Canonical ops command/documentation replacing scattered wrapper knowledge.
- Judge cycle documented and repeatable.

## Safe Automation Opportunities

Hermes can support high-value automation when it remains local-first,
receipt-backed, and approval-gated:

- Research workflows: no-spend public research scouts, repository capability
  scouting, source-deduped briefs, local scoring, and Hermes review packets.
- Local file processing: repo audits, evidence bundles, CSV/receipt cleanup,
  local report compilation, and document indexing.
- Business intelligence/reporting: daily ops briefs, project health rollups,
  stale-job reports, cost/noise summaries, and lead/opportunity ledgers.
- Content workflow scaffolds: draft packets, approval queues, templates, and
  scheduling plans with no posting, liking, following, replying, or DMing
  unless explicitly approved.
- Coding assistant workflows: bounded kanban/subagent lanes, review passes,
  test runs, dependency/security reports, and PR-ready summaries.

Hard exclusions:

- Spam, scams, fake engagement, review manipulation, mass unsolicited outreach,
  abusive scraping, credential extraction, deceptive automation, unsafe
  financial actions, trades, wagers, transfers, live-bot actions, unauthorized
  account changes, and public publishing without explicit approval.
