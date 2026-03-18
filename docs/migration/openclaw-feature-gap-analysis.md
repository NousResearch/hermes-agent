# OpenClaw → Hermes Feature Gap Analysis
**Generated:** 2026-03-17
**Auditor:** Deep architectural audit of /mnt/projects/openclaw (TypeScript) vs /mnt/projects/hermes (Python)

---

## Executive Summary

Hermes already has versions of most major systems, but OpenClaw's implementations are significantly
more sophisticated in every area. The biggest gaps are in hook depth/coverage, browser session
persistence, cron robustness, OAuth infrastructure, and the plugin-as-provider architecture.
There is **no Composio integration in either codebase** — that is a net-new feature.

---

## 1. Hook System

### OpenClaw (TypeScript)
**Files:**
- `src/hooks/internal-hooks.ts` — core registry (globalThis singleton Map, 421 lines)
- `src/hooks/hooks.ts` — public re-export layer
- `src/hooks/plugin-hooks.ts` — plugin-scoped hook dir resolver
- `src/plugins/hooks.ts` — `createHookRunner()` (961 lines) — the plugin hook runner
- `src/plugins/hook-runner-global.ts` — singleton hook runner initialized at gateway startup
- `src/plugins/types.ts` lines 1343-1418 — 24 named `PluginHookName` types

**How it works (code level):**
Two distinct but complementary systems:

**System A — Internal hooks** (`internal-hooks.ts`): A global event bus keyed by
`"type:action"` strings (e.g. `"command:new"`, `"message:received"`). The registry is stored
on `globalThis.__openclaw_internal_hook_handlers__` to survive bundle splitting. Handlers are
async, errors are caught per-handler. Used by built-in code, not plugins.

Event types: `command | session | agent | gateway | message`
Actions per type: e.g. `message:received`, `message:sent`, `message:transcribed`,
`message:preprocessed`, `agent:bootstrap`, `gateway:startup`

**System B — Plugin hooks** (`plugins/hooks.ts`): Strongly-typed, per-plugin hook runner with
three execution modes:
- `runVoidHook()` — fire-and-forget, parallel execution (e.g. `llm_input`, `agent_end`)
- `runModifyingHook()` — sequential by priority, results merged (e.g. `before_prompt_build`,
  `before_model_resolve`)
- `runClaimingHook()` — first `{ handled: true }` response wins (e.g. `inbound_claim`,
  `before_tool_call`)

The 24 hook names in `PluginHookName`:
```
before_model_resolve, before_prompt_build, before_agent_start,
llm_input, llm_output, agent_end,
before_compaction, after_compaction, before_reset,
inbound_claim, message_received, message_sending, message_sent,
before_tool_call, after_tool_call, tool_result_persist, before_message_write,
session_start, session_end,
subagent_spawning, subagent_delivery_target, subagent_spawned, subagent_ended,
gateway_start, gateway_stop
```

Key capability: `before_prompt_build` can **inject/override system prompt and prepend context**
from plugins. `before_model_resolve` can **override which model/provider is used**. These are
the power hooks.

### Hermes (Python)
**Files:**
- `gateway/hooks.py` — `HookRegistry` class (153 lines)
- `hermes_cli/plugins.py` — `VALID_HOOKS` set + `PluginContext.register_hook()`

**What exists:**
Hermes has a basic working hook system. `HookRegistry` discovers `~/.hermes/hooks/` directories,
loads `handler.py` with a `handle(event_type, context)` function. Supports async and sync handlers,
wildcard matching (`command:*`).

Current event coverage: `gateway:startup`, `session:start`, `session:end`, `session:reset`,
`agent:start`, `agent:step`, `agent:end`, `command:*`

The plugin system (`hermes_cli/plugins.py`) has 6 hooks:
`pre_tool_call, post_tool_call, pre_llm_call, post_llm_call, on_session_start, on_session_end`

**GAPS vs OpenClaw:**
- No `before_prompt_build` (cannot inject system prompt / context from hooks)
- No `before_model_resolve` (cannot override model per-request from hooks)
- No `message_received` / `message_sending` / `message_sent` hooks
- No `before_tool_call` return value (Hermes's `pre_tool_call` fires but can't block/modify)
- No `inbound_claim` (cannot route/claim inbound messages from plugins)
- No subagent lifecycle hooks (spawning/spawned/ended)
- No `tool_result_persist` hook
- No `before_message_write` hook
- No merge/priority system for modifying hooks (all are fire-and-forget)
- No `before_compaction` / `after_compaction` hooks

**Port difficulty: MEDIUM**
Hermes already has the scaffold. The work is:
1. Add ~15 new event type strings and wire them in `run_agent.py` / `gateway/`
2. Add a `runModifyingHook` pattern (accumulate return values, merge dicts)
3. Add a `runClaimingHook` pattern (first truthy return wins)
4. Wire `before_prompt_build` into `agent/prompt_builder.py`
5. Wire `before_model_resolve` into provider resolution path

---

## 2. Cron / Scheduler System

### OpenClaw (TypeScript)
**Files (src/cron/ — ~60 files):**
- `src/cron/service.ts` — `CronService` class (the core scheduler)
- `src/cron/service/jobs.ts`, `store.ts`, `state.ts`, `ops.ts`, `timer.ts`, `locked.ts`
- `src/cron/schedule.ts` — `computeNextRunAtMs()`, uses `croner` library with LRU cache
- `src/cron/isolated-agent/run.ts` — isolated agent runner for each job
- `src/cron/isolated-agent/delivery-dispatch.ts` — post-run delivery routing
- `src/cron/store.ts` — JSON store with atomic writes
- `src/cron/run-log.ts` — JSONL per-job run log with size/line limits
- `src/cron/session-reaper.ts` — cleans up old cron sessions
- `src/cron/delivery.ts` — delivery targets and failure alerts
- `src/config/types.cron.ts` — `CronConfig` type with retry/failure config

**Schedule kinds:** `cron` (expression + tz), `every` (interval with anchor), `at` (one-shot timestamp)

**How it works:**
`CronService` runs as a real in-process service (not a polling loop):
- Timer-based: calculates `nextRunAt`, arms a `setTimeout` precisely
- Mutex locking per job using `service/locked.ts` to prevent concurrent runs
- `runIsolatedAgentJob()` callback spawns a fresh agent with its own session key
- Results streamed back via `subagent-followup.ts` or delivery dispatch
- Stagger logic (`stagger.ts`) prevents all hourly jobs from firing simultaneously
- Failure alerts with cooldown and configurable failure destinations
- Job retry config: `maxAttempts`, `backoffMs[]`, `retryOn` error type list
- Session reaper: prunes sessions older than `sessionRetention` config
- Run log: JSONL per job with `maxBytes`/`keepLines` limits

### Hermes (Python)
**Files:**
- `cron/scheduler.py` — `tick()` function + `run_job()` (534 lines)
- `cron/jobs.py` — job CRUD + `get_due_jobs()` (621 lines)
- `tools/cronjob_tools.py` — agent-facing cron tools
- `hermes_cli/cron.py` — CLI cron commands

**What exists:**
A file-locked polling scheduler. `tick()` is called every 60s from the gateway. Uses `fcntl`
file lock to prevent overlap. `get_due_jobs()` reads `~/.hermes/cron/jobs.json` and checks
timestamps. `run_job()` creates a fresh `AIAgent`, runs it, and delivers the result. Supports
`deliver=origin|local|platform:chat_id`. Has `[SILENT]` suppression marker. Multi-skill loading.

**GAPS vs OpenClaw:**
- **No precise timer arming** — polls every 60s; jobs can be up to 60s late
- **No per-job concurrency locking** — two ticks can run the same job simultaneously
- **No retry config** — if a job fails, it's just logged
- **No stagger logic** — all jobs due at :00 fire at once
- **No failure alerts** — no configurable notification on repeated failures
- **No session reaper** — cron sessions accumulate indefinitely
- **No run log JSONL** — only markdown output files
- **No `every` interval kind with anchor** — only cron expressions and one-shot
- **No `at` one-shot kind** — only recurring schedules
- **No isolated-agent session key** — uses ad-hoc `cron_{id}_{timestamp}` string

**Port difficulty: MEDIUM**
The polling architecture is fine for Hermes's use case. Key improvements to port:
1. `CronRetryConfig` — add `max_attempts` and `backoff_ms` to job schema and scheduler
2. Job-level file locking (per-job `.lock` file, not just global tick lock)
3. `at` one-shot schedule kind (run once at specific timestamp, then disable)
4. Stagger: add `math.fmod(anchor, 60*60*1000)` jitter for top-of-hour jobs
5. Failure alert: track consecutive failures, emit to delivery target after threshold

---

## 3. Composio Integration

### OpenClaw (TypeScript)
**Files:** None. No Composio integration found anywhere in the codebase.

### Hermes (Python)
**Files:** None. No Composio integration found anywhere in the codebase.

**Status: NET-NEW feature for both codebases.**

Composio (https://composio.dev) provides OAuth-managed tool connections to 250+ apps
(Gmail, GitHub, Slack, Notion, etc.) via a unified API.

**What would need to be built in Hermes:**
- `tools/composio_tool.py` — tool wrapping the Composio Python SDK
- OAuth flow: `hermes setup composio` → redirect → token stored in config
- Tool discovery: list connected apps, dynamically generate tool schemas
- Config: `composio_api_key` in `~/.hermes/config.yaml`

**Port difficulty: MEDIUM** (it's a pure Python SDK integration, no TypeScript to port)

---

## 4. Persistent Browser Sessions

### OpenClaw (TypeScript)
**Files (src/browser/ — ~130 files):**
- `src/browser/profiles.ts` — profile name/port/color allocation
- `src/browser/profiles-service.ts` — profile lifecycle service
- `src/browser/server-context.ts` — per-profile browser state management
- `src/browser/server-context.types.ts` — `ProfileContext`, `ProfileRuntimeState`, `BrowserTab`
- `src/browser/pw-session.ts` — `PlaywrightSession` (845 lines) — core Playwright wrapper
- `src/browser/chrome.ts` — Chrome process launch + user-data-dir management
- `src/browser/config.ts` — `ResolvedBrowserProfile` type with cdpPort, cdpUrl, cookies
- `src/browser/control-auth.ts` — browser server auth (loopback tokens)
- `src/browser/bridge-server.ts` — HTTP bridge between gateway and browser server
- `src/browser/routes/` — REST API routes for agent interactions
- `src/browser/server-context.tab-ops.ts` — tab open/close/select operations
- `src/browser/server-context.availability.ts` — Chrome reachability + CDP wait
- `src/browser/pw-tools-core.ts` — snapshot, click, type, scroll, download actions
- `src/browser/session-tab-registry.ts` — per-session tab tracking

**How it works (code level):**
OpenClaw runs a **persistent browser server** as a sidecar process. Each "profile" is a named
Chrome instance with its own:
- Dedicated Chrome user-data-dir (`~/.openclaw/browser/profiles/<name>/`)
- Dedicated CDP port (range 18800-18899, one per profile)
- Persistent login sessions (cookies survive between agent runs)
- Color-coded identity for visual distinction

The agent communicates via HTTP to `localhost:18791` (bridge) → browser server routes.
`PlaywrightSession` (`pw-session.ts`) connects via CDP to the running Chrome, manages a
`BrowserContext` + `Page` pool, tracks console/error/network events per page.

Key feature: **profiles persist cookies across sessions** — log into Gmail once, every future
cron job / agent session can reuse that authenticated state. This is the core value-add.

### Hermes (Python)
**Files:**
- `tools/browser_tool.py` — main browser tool (1804 lines)
- `tools/browser_providers/base.py` — `CloudBrowserProvider` ABC
- `tools/browser_providers/browserbase.py` — Browserbase cloud provider
- `tools/browser_providers/browser_use.py` — browser-use library integration

**What exists:**
Task-scoped (ephemeral) browser sessions. Each `task_id` gets its own browser session,
discarded on cleanup. Local mode uses `agent-browser` CLI subprocess wrapping headless Chromium.
Cloud mode uses Browserbase API. No persistent profiles, no saved cookies, no named profiles.

**GAPS vs OpenClaw:**
- **No persistent profiles** — cookies are lost when task ends; must re-login every time
- **No named profiles** — can't say "use my work-gmail profile"
- **No CDP-based session reuse** — no Playwright connection to a running Chrome
- **No browser server sidecar** — browser spawned fresh per task
- **No tab registry** — no per-session tab tracking
- **No bridge auth** — no loopback token security

**Port difficulty: HARD**
This is the largest gap to close. OpenClaw's browser system is ~130 files. A minimum viable
persistent-profile system for Hermes would need:
1. Profile directory structure: `~/.hermes/browser/profiles/<name>/`
2. Chrome launch with `--user-data-dir` pointing to profile dir
3. Playwright `chromium.connect_over_cdp()` to attach to running Chrome
4. Profile state saved to `~/.hermes/config.yaml` under `browser.profiles`
5. `browser_tool.py` extended with a `profile` parameter

A simpler intermediate step: use Playwright's `browser_context.storage_state()` to save/restore
cookies to a JSON file between sessions. This gives ~80% of the value with ~20% of the complexity.

---

## 5. Plugin System Architecture

### OpenClaw (TypeScript)
**Files:**
- `src/plugins/manifest.ts` — `openclaw.plugin.json` schema (307 lines)
- `src/plugins/loader.ts` — plugin discovery + loading
- `src/plugins/registry.ts` — `PluginRegistry` runtime state
- `src/plugins/types.ts` — 1951 lines of plugin type definitions
- `src/plugins/provider-runtime.ts` — plugin-as-provider architecture
- `src/plugins/runtime/` — 25 files for platform-specific runtimes (Discord, Telegram, WhatsApp, etc.)
- `src/plugins/slots.ts` — memory slot arbitration (only one memory plugin wins)
- `src/plugins/marketplace.ts` — plugin marketplace integration

**Plugin manifest (`openclaw.plugin.json`):**
```json
{
  "id": "my-plugin",
  "kind": "memory" | "context-engine",
  "channels": ["telegram", "discord"],
  "providers": ["openai", "anthropic"],
  "hooks": ["./hooks"],
  "skills": ["./skills"],
  "enabledByDefault": false,
  "configSchema": {...},
  "providerAuthChoices": [...]
}
```

**Plugin SDK (`src/plugin-sdk/`):**
Plugins export an `OpenClawPlugin` object via `index.ts`. The SDK gives plugins:
- `plugin.on(hookName, handler)` — register typed hook handler
- `plugin.tool(factory)` — register tool factories
- `plugin.channel(channelPlugin)` — register messaging platform
- `plugin.provider(providerPlugin)` — register LLM provider
- `plugin.speechProvider(speechPlugin)` — register TTS/STT provider
- `plugin.webSearchProvider(searchPlugin)` — register web search backend
- `plugin.imageGenProvider(imgPlugin)` — register image generation backend

The plugin can **completely replace** any subsystem via typed extension points.
`ProviderPlugin` type (lines 626-853) allows a plugin to define a full LLM provider with
model catalog, OAuth auth, dynamic model resolution, stream wrapping, and more.

**Memory slot system:** At most one `kind: "memory"` plugin is active at a time.
If multiple memory plugins are installed, config controls which wins (`plugins.slots.memory`).

### Hermes (Python)
**Files:**
- `hermes_cli/plugins.py` — `PluginManager`, `PluginContext`, `LoadedPlugin` (449 lines)

**Plugin manifest (`plugin.yaml`):**
```yaml
name: my-plugin
version: 1.0.0
description: ...
author: ...
requires_env: [MY_API_KEY]
provides_tools: [my_tool]
provides_hooks: [pre_tool_call]
```

**Plugin registration:**
```python
def register(ctx: PluginContext):
    ctx.register_tool(name, toolset, schema, handler)
    ctx.register_hook("pre_tool_call", my_callback)
```

Three plugin sources: user (`~/.hermes/plugins/`), project (`./.hermes/plugins/`), pip entry-points.

**GAPS vs OpenClaw:**
- **No `ProviderPlugin` type** — plugins cannot register new LLM providers
- **No channel plugins** — plugins cannot add messaging platforms
- **No speech provider plugins** — plugins cannot add TTS/STT backends
- **No web search provider plugins** — plugins cannot add search backends
- **No memory slot arbitration** — no `kind: memory` concept
- **No marketplace** — no plugin discovery/install from registry
- **No `on()` typed hook registration** — hooks are string-keyed with no type safety
- **No `configSchema`** — plugins can't declare their config structure
- **No `enabledByDefault`** — all discovered plugins are loaded

**Port difficulty: MEDIUM**
The directory-based plugin loading is already solid. The missing pieces:
1. Add `ProviderPlugin` concept to `runtime_provider.py` (let plugins register providers)
2. Add `kind: memory` to manifest + single-winner arbitration
3. Add `configSchema` to manifest + validation in `PluginContext`
4. Add `enabledByDefault` and per-plugin enable/disable config

The platform plugin architecture (channel plugins) would be HARD — Hermes platforms live in
`gateway/platforms/` and are not plugin-based.

---

## 6. Multi-Agent Routing

### OpenClaw (TypeScript)
**Files:**
- `src/routing/resolve-route.ts` — `resolveAgentRoute()` (804 lines)
- `src/routing/session-key.ts` — session key construction
- `src/routing/bindings.ts` — config-driven binding rules
- `src/routing/account-lookup.ts` — account ID resolution
- `src/agents/agent-scope.ts` — multi-agent config parsing (342 lines)
- `src/agents/acp-spawn.ts` — subagent spawning via ACP (762 lines)
- `src/agents/lanes.ts` — execution lane (nested/subagent/cron)
- `src/acp/control-plane/manager.ts` — ACP session manager
- `src/acp/control-plane/spawn.ts` — ACP spawn orchestration

**How it works:**
OpenClaw supports **multiple named agents** in `config.yaml`. Inbound messages are routed to
specific agents based on `bindings` rules:

```yaml
agents:
  list:
    - id: main
      default: true
      model: ...
    - id: coding
      model: claude-opus-4.6
      skills: [coding]

bindings:
  - match:
      channel: discord
      peer: { kind: group, id: "123456" }
    agent: coding
  - match:
      channel: telegram
      accountId: work-account
    agent: main
```

Binding resolution (`resolve-route.ts`) uses an indexed multi-level lookup:
`byPeer → byGuildWithRoles → byGuild → byTeam → byAccount → byChannel → default`

Each resolved route has a session key incorporating agent+channel+account+peer, enabling
completely isolated conversation histories per agent per context.

**ACP subagent spawning:** `acp-spawn.ts` spawns child agents via the ACP control plane.
Children run in isolated processes/threads with their own session, can stream output back to
parent via `acp-spawn-parent-stream.ts`. Parent blocks or continues in parallel based on mode.

**Execution lanes:** Jobs run in typed lanes (`cron`, `nested`, `subagent`) that control
resource scheduling and prevent cron jobs from blocking interactive sessions.

### Hermes (Python)
**Files:**
- `tools/delegate_tool.py` — subagent delegation (764 lines)
- `agent/smart_model_routing.py` — cheap/strong model routing (184 lines)
- `gateway/session.py` — per-session state

**What exists:**
- **Delegate tool** — spawns child `AIAgent` instances with restricted toolsets and isolated
  context. Parallel batch mode via `ThreadPoolExecutor`. Children return summary text to parent.
- **Smart model routing** — keyword-based routing between cheap and strong models for
  single-agent turns (not multi-agent)
- No named agents in config, no binding rules, no channel-specific agent routing

**GAPS vs OpenClaw:**
- **No named agent config** — only one agent identity per Hermes instance
- **No binding rules** — can't route "Discord #coding channel → coding agent"
- **No execution lanes** — all work competes for the same resources
- **No ACP-style isolated child processes** — subagents share parent's Python process
- **No agent → channel routing** — gateway doesn't route to different agent configs

**Port difficulty: HARD**
The multi-agent routing system is tightly coupled to session key construction, config loading,
and the gateway. A minimal port would be:
1. Add `agents.list` config support with named agent configs
2. Add `bindings` config matching channel/peer/account to agent
3. Route gateway inbound messages through the binding resolver
4. Give each agent its own session namespace

The ACP control plane (isolated process spawning) would be VERY HARD and likely not worth
porting — Hermes's delegate tool handles the key use case already.

---

## 7. Other Major Systems in OpenClaw Not in Hermes

### 7a. OAuth Provider Infrastructure
**Files:**
- `src/agents/auth-profiles/oauth.ts` — OAuth credential storage/refresh
- `src/plugins/provider-oauth-flow.ts` — `createVpsAwareOAuthHandlers()` — VPS-aware OAuth
- `src/commands/oauth-flow.ts` — browser-based OAuth flow for CLI
- `src/plugins/provider-api-key-auth.ts`, `provider-auth-choices.ts`

**What it does:** Plugins register OAuth flows. When a provider needs OAuth (e.g. GitHub Copilot,
Chutes, OpenAI Codex), the plugin defines the flow and OpenClaw handles token storage/refresh.
VPS-aware: if running on a remote server, shows URL for local browser; locally, opens browser.

**In Hermes:** `hermes_cli/auth.py` handles basic API key auth. No OAuth flow infrastructure.

**Port difficulty: MEDIUM** — Python has `authlib`, `httpx-oauth`, etc. The VPS-aware redirect
pattern (`createVpsAwareOAuthHandlers`) is straightforward to port.

### 7b. ACP (Agent Communication Protocol) Control Plane
**Files:** `src/acp/` (~30 files)
- `src/acp/control-plane/manager.ts` — manages running agent instances
- `src/acp/server.ts` — ACP WebSocket server
- `src/acp/session.ts` — ACP session lifecycle
- `src/acp/persistent-bindings.lifecycle.ts` — persistent session bindings across restarts

**What it does:** A WebSocket-based protocol for agents to talk to the gateway bidirectionally.
Supports spawning sub-agents, binding thread contexts, streaming results back to parents.
Persistent bindings survive gateway restarts.

**In Hermes:** No ACP. The `acp_adapter/` directory exists but is a different implementation
(incoming ACP compatibility, not the full control plane).

**Port difficulty: VERY HARD** — Deep architectural feature. Probably not worth porting; Hermes's
gateway WebSocket and delegate tool cover most use cases.

### 7c. Secrets / Credential Matrix
**Files:** `src/secrets/` (~30 files)
- `src/secrets/configure.ts` — interactive secret configuration
- `src/secrets/runtime.ts` — secret resolution at runtime (env var → file → exec ref → stored)
- `src/secrets/target-registry.ts` — registry of where each secret goes
- `src/secrets/ref-contract.ts` — `exec:`, `file:`, `env:` secret reference syntax

**What it does:** Secrets can be stored as env vars, file paths, `exec:` commands (e.g. keychain),
or encrypted refs. The runtime resolver handles all formats transparently.

**In Hermes:** `.env` file only. No `exec:` references, no file refs.

**Port difficulty: EASY-MEDIUM** — Add `exec:` and `file:` secret reference resolution to
`hermes_cli/config.py`.

### 7d. Skin / Persona Engine
**Files:**
- `src/agents/identity.ts` — per-channel identity config
- `src/agents/identity-file.ts` — identity file loading

**In Hermes:** `hermes_cli/skin_engine.py` — YAML-based skin/persona system already exists.
**GAP: MINIMAL** — Hermes skin engine is functionally equivalent.

### 7e. Canvas / A2UI Protocol
**Files:** `src/browser/routes/agent.act.ts`, Android/iOS apps
**What it does:** Visual canvas overlay for mobile apps to show agent activity.
**In Hermes:** Not present. Mobile-specific feature, unlikely to port.

---

## Priority Port Recommendations

| Feature | Difficulty | Value | Recommendation |
|---------|------------|-------|----------------|
| Hook: before_prompt_build | EASY | HIGH | Port first |
| Hook: before_model_resolve | EASY | HIGH | Port with above |
| Hook: message_received/sent | EASY | HIGH | Port with hooks |
| Hook: modifying (merge) mode | MEDIUM | HIGH | Port with hooks |
| Cron: retry config | EASY | MEDIUM | Port soon |
| Cron: at (one-shot) schedule | EASY | MEDIUM | Port soon |
| Cron: per-job locking | EASY | HIGH | Port soon |
| Browser: cookie persistence | MEDIUM | HIGH | Port via storage_state |
| Browser: named profiles | HARD | HIGH | Port after cookie persistence |
| Composio OAuth | MEDIUM | HIGH | Net-new, build fresh |
| Plugin: ProviderPlugin type | MEDIUM | MEDIUM | Port after hook gaps |
| Plugin: memory slot | EASY | LOW | Port with plugin work |
| Multi-agent routing | HARD | MEDIUM | Port bindings + agent config |
| OAuth provider infra | MEDIUM | MEDIUM | Port for Composio |
| Secret exec: references | EASY | LOW | Port anytime |
| ACP control plane | VERY HARD | LOW | Skip |

---

## Summary of Key File Pairs

| System | OpenClaw Source | Hermes Target |
|--------|----------------|---------------|
| Internal hooks | `src/hooks/internal-hooks.ts` | `gateway/hooks.py` (extend) |
| Plugin hooks | `src/plugins/hooks.ts` | `hermes_cli/plugins.py` (extend) |
| Hook events | `src/plugins/types.ts:1343` | `hermes_cli/plugins.py:VALID_HOOKS` |
| Cron service | `src/cron/service.ts` | `cron/scheduler.py` (extend) |
| Cron schedules | `src/cron/schedule.ts` | `cron/jobs.py` (extend) |
| Browser profiles | `src/browser/profiles.ts` | `tools/browser_tool.py` (new) |
| Browser sessions | `src/browser/pw-session.ts` | `tools/browser_providers/` (new) |
| Plugin manifest | `src/plugins/manifest.ts` | `hermes_cli/plugins.py:PluginManifest` |
| Plugin SDK | `src/plugin-sdk/` | `hermes_cli/plugins.py:PluginContext` |
| Agent routing | `src/routing/resolve-route.ts` | `gateway/session.py` (new) |
| OAuth flow | `src/plugins/provider-oauth-flow.ts` | `hermes_cli/auth.py` (extend) |
