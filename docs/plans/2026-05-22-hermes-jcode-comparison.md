# Hermes and jcode comparison

Date: 2026-05-22 PDT

## Executive summary

Hermes does not encapsulate jcode. They are adjacent agent harnesses with
different centers of gravity:

- Hermes is the broader distributed agent platform: messaging gateway,
  webhooks, plugins, memory-provider integrations, many web/browser backends,
  cron delivery, toolsets, and deployment-oriented terminal backends.
- jcode is the faster local harness: Rust runtime, persistent server/client
  model, highly optimized TUI, native swarm coordination, self-dev reload loop,
  browser tool, semantic memory architecture, and explicit performance budgets.

The best "mother repo" should not merge both codebases into one fork. The
lowest-regret design is a thin integration workspace that keeps both upstreams
intact and adds bridge contracts between them. Treat Hermes as the remote
gateway/orchestrator and jcode as a fast local execution/UI sidecar. Port
features only after a bridge proves they are valuable.

Implementation continuation: the first Hermes-side bridge scaffold now lives in
`plugins/jcode_bridge/`; see
`plugins/jcode_bridge/README.md` for the operator guide and
`docs/plans/2026-05-22-hermes-jcode-bridge-implementation.md` for the
implementation notes. The scaffold already treats jcode as the fast Rust
sidecar and Hermes as the gateway/policy owner: direct tool calls, debug-socket
dispatch, and webhook routes all pass through the Hermes plugin before jcode
receives work. Portable contract schemas now live in
`contracts/jcode_bridge/v1/`, and the concrete mother-repo architecture is
spelled out in
`docs/plans/2026-05-23-hermes-jcode-mother-repo-blueprint.md`.

## Local research setup

Repos analyzed:

| Repo | Local path | Commit |
| --- | --- | --- |
| Hermes | `/Users/aayu/Workspace/developer/hermes` | `729a778af0b3f984b4934361cad3050f6afb79ba` |
| jcode | `/Users/aayu/Workspace/developer/hermes/.codex-research/jcode` | `7951a2ddb91bad10155b911ccd0971de5baeafc8` |

Commands run:

```bash
git clone https://github.com/1jehuang/jcode.git .codex-research/jcode
graphify update .
cd .codex-research/jcode && graphify update .
```

Graph artifacts:

| Repo | Graph JSON | Graph report |
| --- | --- | --- |
| Hermes | `graphify-out/graph.json` | `graphify-out/GRAPH_REPORT.md` |
| jcode | `.codex-research/jcode/graphify-out/graph.json` | `.codex-research/jcode/graphify-out/GRAPH_REPORT.md` |

Graphify reported that both graphs were too large for HTML visualization. The
JSON and Markdown reports were generated successfully.

## Size and graph shape

| Metric | Hermes | jcode |
| --- | ---: | ---: |
| Git-tracked files from `rg --files`/repo scan | 3,580 | 1,105 |
| Total tracked lines from `wc -l` | 1,538,899 | 1,363,019 |
| Dominant implementation language | Python | Rust |
| Primary source files | 1,811 `.py` | 814 `.rs` |
| Test-like tracked files | 1,262 | 209 |
| Graphify files extracted | 2,212 | 872 |
| Graphify nodes | 67,683 | 18,201 |
| Graphify edges | 223,428 | 69,026 |
| Graphify communities | 452 in report summary; 570 in rebuild stdout | 81 in report summary; 155 in rebuild stdout |

Graphify "god nodes":

| Hermes | jcode |
| --- | --- |
| `PlatformConfig` | `get()` |
| `Platform` | `create_test_app()` |
| `MessageEvent` | `lock_test_env()` |
| `MessageType` | `set_var()` |
| `BasePlatformAdapter` | `info()` |
| `SendResult` | `take()` |
| `AIAgent` | `remove_var()` |
| `SessionSource` | `warn()` |
| `GatewayConfig` | `handle_remote_key_internal()` |
| `SessionDB` | `handle_server_event()` |

Interpretation: Hermes' graph is centered on cross-platform gateway/session
types and the Python agent loop. jcode's graph is more spread across Rust
helpers, TUI/server tests, state/event handling, and provider/runtime surfaces.
Its raw "god nodes" are less semantically helpful because common Rust method
names like `get`, `take`, and `set_var` dominate the graph.

## Core architectural difference

Hermes is a Python agent platform around `AIAgent`:

- `run_agent.py` owns the synchronous tool-calling loop.
- `model_tools.py` discovers tool modules and dispatches registered handlers.
- `toolsets.py` exposes composable tool groups, with `_HERMES_CORE_TOOLS` as
  the base set for CLI and messaging platforms.
- `gateway/run.py` and `gateway/platforms/*` normalize external platforms into
  `MessageEvent` sessions and route them to `AIAgent`.
- `hermes_state.py` stores sessions in SQLite with FTS5 search.

jcode is a Rust client/server harness:

- `src/main.rs` starts a Tokio runtime and configures allocator behavior.
- `src/server.rs` and `src/server/*` run the persistent shared server.
- `src/tui/*` owns the ratatui-based interactive client and custom rendering.
- `src/agent.rs` owns agent state, streaming, cache tracking, soft interrupts,
  compaction, memory toggles, and provider state.
- `src/tool/mod.rs` registers tools via a Rust `Tool` trait and caches base
  tools in a `OnceLock` for cheap per-session setup.

This matters for integration. Hermes features are easiest to add as Python
plugins/tools or gateway adapters. jcode features are easiest to add as Rust
tools/crates or server protocol events. A combined repo should use a protocol
boundary, not shared imports.

## Feature comparison

| Area | Hermes | jcode | Best combined behavior |
| --- | --- | --- | --- |
| UI | Classic CLI plus Ink TUI and browser dashboard embedding. Strong gateway status and tool progress, but not primarily a low-latency renderer. | Native Rust TUI with side panels, info widgets, inline mermaid, custom scrollback, hotkeys, and performance-focused rendering. README claims 14 ms first frame and 48.7 ms first input on one Linux benchmark. | Keep jcode as the fast local TUI. Let Hermes provide remote/web/mobile surfaces unless/until jcode UI can speak a Hermes-compatible protocol. Track local bridge overhead with `scripts/jcode_bridge_latency_probe.py` so the integration does not erase jcode's speed advantage. |
| Gateway/webhooks | First-class messaging gateway for Telegram, Discord, Slack, WhatsApp, Signal, SMS, Email, Home Assistant, Mattermost, Matrix, DingTalk, Feishu, WeCom, Weixin, BlueBubbles, QQ, Yuanbao, Teams, LINE, webhooks, and API server. Webhook adapter supports HMAC, route templates, direct delivery, agent runs, rate limits, body caps, idempotency, and cross-platform delivery. | Has a WebSocket gateway for remote clients/iOS/web that bridges to the existing newline JSON protocol. It is not a general external-service webhook router in the same sense. Safety docs mention notification webhooks for ambient approvals. | Use Hermes for inbound webhooks and external chat platforms. Add a route option that can dispatch a webhook-triggered task to jcode via a sidecar bridge when local speed/UI/browser state matters. |
| Browser automation | Multiple backends: Browserbase, Browser Use, Firecrawl, Camofox local, local Chromium CDP, local agent-browser. Includes URL safety, private URL local fallback, persistent Camofox profile options, screenshots, accessibility snapshots, console/CDP/dialog tools. | One first-class `browser` tool, currently Firefox Agent Bridge. The provider protocol doc is clean and aims at Firefox, Chrome, CDP, WebDriver/BiDi, Safari, and custom providers. Compact UI summaries hide sensitive typed text. | Port the jcode Firefox Agent Bridge as a Hermes browser provider, or expose jcode's browser tool through an MCP/sidecar bridge. Standardize on a common browser-provider contract. |
| Web research | Provider-rich `web_search`/`web_extract`: Firecrawl, Parallel, Tavily, Exa, SearXNG, Brave, DDGS, xAI/X search, managed Nous Tool Gateway. Extraction can use auxiliary LLM summarization. | `websearch` uses DuckDuckGo or Bing with optional Bing API, and `webfetch` fetches and converts pages to text/Markdown/HTML. Lean and fast, less provider breadth. | Prefer Hermes research backends for deep research. Keep jcode's lightweight path as a no-key, low-latency fallback. |
| Memory | Built-in bounded `MEMORY.md`/`USER.md` injected at session start, `session_search` over SQLite FTS5, plus external memory providers including Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, and Supermemory. | Semantic memory graph architecture: local embeddings, non-blocking turn N to N+1 retrieval, memory sidecar validation, graph/cascade retrieval, ambient consolidation, explicit memory/search/list/forget/link/tag tools. | Bridge memory rather than replace it. Hermes should use its provider plugin surface to sync with jcode's memory graph, once jcode exposes a stable memory API/CLI. |
| Skills | Bundled and optional skills, skill hub, slash commands, skill CRUD, plugin-aware docs. | Skills are not all loaded at startup. README says conversation embeddings can auto-inject skills on semantic hit, with manual skill tool and slash activation. | Combine Hermes' catalog and packaging with jcode's semantic activation. Add a skill-selection sidecar before prompt assembly. |
| Multi-agent | `delegate_task`, `execute_code`, kanban plugin, gateway-dispatched workers, cron, background tasks. | Native swarm: server tracks sessions in the same repo, notifies agents when files they read are changed by another agent, supports DM/broadcast/repo-scoped messaging, autonomous worker spawning. | Use jcode for local swarm conflict awareness. Use Hermes for cross-platform orchestration and notifications. Add a bridge so Hermes kanban tasks can spawn or monitor jcode swarm sessions. |
| Providers/auth | Broad Python provider registry and model-provider plugins. Strong exact dependency pinning. | Many OAuth/subscription-backed flows: Claude, OpenAI/Codex, Gemini, Copilot, Azure, Alibaba, Fireworks, MiniMax, local and OpenAI-compatible endpoints, multi-account switching. | Keep both provider stacks. Bridge at the task/protocol layer so each side can use its strongest authenticated route. |
| Safety | Strong shell approval system, hardline command blocklist, gateway allowlists/pairing/admin tiers, webhook HMAC and insecure-auth guardrails. `send_message` can deliver cross-platform messages, with target resolution and mirroring. | Safety system is mostly documented for ambient mode; core principle says direct communication with another human requires permission. Gmail send requires API tier and an explicit `confirmed: true`. | The combined tool should require explicit confirmation for outbound human communication and sensitive personal-data discovery. This is especially important for browser-driven DMs and contact lookup workflows. |
| Updates | Python package with plugins, exact pins, generated docs, large pytest suite. | Rust workspace with many crates, hot reload, self-dev workflow, performance scripts, release install scripts. | Use upstream submodules/subtrees and contract tests. Do not vendor-copy either codebase into a merged directory. |

## Webhooks and messaging

Hermes is materially stronger for external event ingress. Its webhook adapter is
already a productized HTTP service:

- `gateway/platforms/webhook.py` validates HMAC secrets per route.
- Routes can filter events, template prompts, attach skills, deliver to GitHub
  comments or messaging platforms, or run `deliver_only` for sub-second push
  notifications without an LLM run.
- The adapter has rate limiting, idempotency, body-size caps, and refuses
  unauthenticated public binds when `INSECURE_NO_AUTH` is used.

jcode's `src/gateway.rs` is a different thing: a WebSocket gateway for remote
clients such as iOS/web. It upgrades `/ws`, validates paired-device tokens, and
bridges WebSocket frames to the same server protocol used by Unix socket
clients. That is excellent for remote control of a local jcode server, but it is
not equivalent to Hermes' generic webhook routing layer.

Recommendation: do not port Hermes webhooks into jcode first. Instead, let
Hermes receive webhooks and add a `dispatch: jcode` route mode that sends the
rendered prompt to a local jcode server/session over a stable bridge.

## Browser and LinkedIn-style workflows

Neither repo has a dedicated LinkedIn API integration in this snapshot. The
"send a LinkedIn DM from my account" behavior is almost certainly browser
automation against a logged-in profile.

jcode's advantage is UX and simplicity: a single `browser` tool backed by
Firefox Agent Bridge, with compact sensitive-input summaries. Hermes' advantage
is backend breadth: cloud browsers, local Chromium, Camofox, CDP attachment,
private URL routing, screenshots, console/CDP, and persistent Camofox profile
options.

For a combined tool:

1. Add a common browser-provider contract.
2. Implement a Hermes provider for Firefox Agent Bridge, or expose jcode's
   `browser` tool through a local MCP server.
3. Add an explicit "outbound communication" confirmation layer above browser
   actions that submit forms, send DMs, post comments, or email people.
4. Keep login state scoped by profile, not by repo, so updates do not wipe
   browser sessions.

## Memory and personal preference recall

The user story where jcode learned a favorite workspace folder maps directly to
jcode's semantic memory design. jcode's README and memory architecture describe
automatic embedding of turns, sideagent verification, graph/cascade retrieval,
explicit memory tools, session search, and ambient consolidation.

Hermes can remember this kind of fact too, but its built-in memory is bounded
and curated: `MEMORY.md` and `USER.md` are injected at session start with small
character budgets. Hermes compensates with FTS5 `session_search` and optional
external memory providers such as Honcho and Mem0.

Best combined behavior:

- Keep Hermes' bounded memory for critical high-signal facts.
- Use jcode-style semantic retrieval for "the agent should just know this from
  past behavior" facts.
- Expose jcode memory as a Hermes memory provider only after jcode's memory
  graph has a stable programmatic interface.
- Sync durable facts in one direction first, with conflict logs, before trying
  full bidirectional reconciliation.

## Performance

jcode is explicitly performance-led:

- README publishes RAM and time-to-first-frame/input comparisons.
- `src/main.rs` configures allocator behavior and Rustls provider at startup.
- `src/tool/mod.rs` caches base tool instances in `OnceLock`, cloning Arcs per
  session.
- Scripts include startup, visible-ready, memory, compile, tool, swarm, and
  self-dev benchmarks.
- Server architecture keeps a detached daemon and lets clients reconnect,
  avoiding repeated cold starts.

Hermes has performance work too, but the architecture naturally pays Python
startup/import cost and gateway/tool discovery overhead. It optimizes through
prompt caching, tool-definition caching, lazy SDK imports, async loop reuse,
context compression, and long-running gateway processes.

I did not build or benchmark jcode locally in this pass. The performance
numbers above are documented claims and benchmark scripts from the jcode repo,
not independent measurements from this machine.

## What to combine

High-value ports/bridges:

1. Hermes webhook route to jcode sidecar.
2. jcode browser provider exposed to Hermes.
3. jcode semantic skill activation concept in Hermes prompt assembly.
4. jcode swarm/file-activity awareness exposed as Hermes kanban/delegation
   signals.
5. Hermes web research providers exposed to jcode through MCP.
6. Hermes messaging gateway delivery exposed to jcode as a `communicate` backend.
7. Shared outbound-human-communication confirmation contract.

Low-value or risky ports:

1. Rewriting Hermes in Rust.
2. Rewriting jcode's TUI in Python/Ink.
3. Copying all Hermes gateway adapters into jcode.
4. Copying jcode memory internals into Hermes without a stable API.
5. One monorepo that edits both upstream codebases directly.

## Mother repo design

Recommended layout:

```text
mother-agent/
|-- upstreams/
|   |-- hermes/       # git submodule or subtree: NousResearch/hermes-agent
|   `-- jcode/        # git submodule or subtree: 1jehuang/jcode
|-- bridges/
|   |-- hermes-jcode-mcp/
|   |-- hermes-plugin-jcode/
|   |-- jcode-tool-hermes/
|   `-- browser-provider-contract/
|-- contracts/
|   |-- task-runner.schema.json
|   |-- browser-provider.schema.json
|   |-- memory-event.schema.json
|   `-- outbound-communication.schema.json
|-- tests/
|   |-- contract/
|   |-- smoke/
|   `-- fixtures/
`-- docs/
    |-- architecture.md
    |-- upstream-sync.md
    `-- routing-policy.md
```

Use submodules if you want exact upstream commits and minimal merge noise. Use
subtrees if contributors dislike submodule ergonomics. Either way, keep bridge
code outside upstream directories.

## Upstream update strategy

1. Pin each upstream to a commit SHA in the mother repo.
2. Maintain small patch queues only when absolutely necessary.
3. Put all normal integration code in `bridges/`, not inside upstream source.
4. Define versioned contract schemas for task runs, browser actions, memory
   events, and outbound communication.
5. Build contract tests that run against both upstream heads.
6. Add an `upstream-sync` CI job:
   - fetch Hermes main
   - fetch jcode master
   - run graphify update on both
   - run `scripts/jcode_bridge_compat.py`
   - run `scripts/jcode_bridge_smoke.py`
   - run `scripts/jcode_bridge_upstream_report.py --smoke`
   - run contract tests
   - run a smoke matrix: Hermes webhook to jcode, jcode tool to Hermes web
     search, browser action, memory recall, outbound confirmation
7. Generate a short compatibility report on each upstream bump.

This makes future updates portable because the mother repo owns the integration
boundary, not the internals.

The first concrete version of that boundary is now the `jcode-bridge.v1`
contract in the Hermes plugin scaffold, with portable JSON Schemas in
`contracts/jcode_bridge/v1/`. Treat those schemas and fixtures as the minimum
compatibility gate before bumping the pinned jcode commit in a future mother
repo.

The first generated sync snapshot is
`docs/plans/2026-05-23-hermes-jcode-upstream-sync-report.md`. It records the
pinned SHAs, Graphify summaries, dirty worktree notes, and bridge-contract
status for this integration checkpoint.

The mother-repo blueprint is
`docs/plans/2026-05-23-hermes-jcode-mother-repo-blueprint.md`. It turns this
comparison into a concrete layout, routing policy, update gate, and phased path
for preserving jcode's Rust hot path while using Hermes as the autonomous
orchestration shell. `scripts/hermes_jcode_mother_repo.py` now turns that
blueprint into a standalone scaffold with copied bridge code, schemas, fixtures,
plan docs, and a manifest of the pinned Hermes/jcode state. The first reverse
bridge contract, `hermes-service.v1`, now gives jcode a stable newline-JSON
path to call allowlisted Hermes services such as web search/extract without
importing Hermes internals. `bridges/jcode-tool-hermes/` is the first
dependency-free Rust caller for that contract.

## Proposed phased implementation

### Phase 1: Sidecar bridge

Build a local bridge that can:

- start or discover `jcode serve`
- submit `jcode run` style prompts
- attach to or create a named jcode session
- return final response, logs, and session ID
- expose a health/status endpoint

Hermes side: a plugin tool named `jcode_run` and optional webhook route
delivery mode.

jcode side: no upstream change if CLI/server protocol is enough.

Current Hermes-side scaffold status: `plugins/jcode_bridge/` provides
`jcode_run`, `jcode_status`, `jcode_contract_check`, direct debug-socket
probing, `execution_mode=auto`, `ensure_server: true` startup of the persistent
Rust sidecar, and an opt-in `dispatch: jcode` webhook route hook. It also
includes the first Hermes-owned safety gate for unattended
human-contact/private-person-data prompts, plus contract fixtures for jcode
wrapper JSON, NDJSON, and debug-socket envelopes. Webhook routes can opt into
`preflight_contract`/`preflight_live` so Hermes verifies the bridge before
sending real work to the jcode sidecar. The same plugin now includes
`plugins/jcode_bridge/hermes_service.py`, a reverse service scaffold that can
run selected Hermes tools behind a `hermes-service.v1` request/response
contract, plus `bridges/jcode-tool-hermes/`, a Rust client that can call that
service from a future jcode tool wrapper. The same reverse service is now also
available through `bridges/hermes-mcp-server/`, a dependency-free stdio MCP
server that jcode can load without an upstream patch. Its jcode-facing
transport fixtures and schemas live in `contracts/hermes_mcp/v1/`, so future
jcode MCP changes have their own update alarm.
`scripts/jcode_bridge_latency_probe.py` measures the persistent local MCP bridge
path without model or network calls, giving the mother repo a repeatable speed
regression signal before either upstream is bumped.

### Phase 2: Browser bridge

Expose jcode's Firefox Agent Bridge to Hermes via one of:

- Hermes browser provider plugin
- MCP server wrapping jcode's `browser` tool
- standalone provider process implementing the common browser-provider contract

This targets the user-loved "logged-in browser just works" behavior.

### Phase 3: Research routing

Expose Hermes `web_search` and `web_extract` to jcode through the
`hermes-service.v1` stdio bridge or the MCP wrapper around it. Keep jcode's
DDG/Bing path as fallback. Add a routing policy:

- low-latency query: jcode websearch
- deep research or provider-specific extraction: Hermes web tools
- authenticated site interaction: browser bridge

### Phase 4: Memory bridge

Implement one-way export first:

- jcode semantic memory events -> Hermes external memory provider mirror
- Hermes curated memory writes -> jcode memory ingestion queue

Then add conflict detection and user-visible memory review.

### Phase 5: Swarm/kanban bridge

Let Hermes kanban spawn jcode swarm workers for same-repo local work. Let jcode
server events report back into Hermes gateway notifications.

### Phase 6: Safety and policy hardening

Before shipping broad web/browser communication:

- require confirmation for any human-directed outbound message
- require confirmation before submitting browser forms to social, email, job,
  financial, health, or government sites
- log recipient, destination, and exact body
- separate "find public business contact info" from private-person doxxing
  workflows

This preserves the useful "act through my account" capability while avoiding a
tool that silently contacts people or digs for sensitive personal data.

The bridge plugin now implements the first lexical gate for those categories.
The remaining work is to connect that gate to a richer approval UI, browser
form-submit policy, and contract fixtures that keep future jcode/Hermes updates
compatible.

## Decision

Do not stop because "Hermes is just better." It is not a superset of jcode.

Do not fork-merge them either. Build a bridge-first mother repo:

- Hermes handles webhooks, messaging, deep research backends, plugins, cron,
  and remote delivery.
- jcode handles fast local UI, local browser/session feel, swarms, self-dev,
  and performance-sensitive interactive work.
- Contract tests and JSON Schemas protect the seams so upstream updates can be
  pulled without repeatedly solving the same integration problem.
