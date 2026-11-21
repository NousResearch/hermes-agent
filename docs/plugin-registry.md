# Hermes Plugin Registry

Complete catalog of Hermes Agent plugins — official bundled plugins and community plugins.

## Quick Start

```bash
hermes plugins list                  # list all plugins
hermes plugins list --plain --no-bundled  # compact view, user plugins only
hermes plugins enable <name>         # enable a plugin
hermes plugins disable <name>        # disable a plugin
hermes plugins                       # interactive toggle UI
```

Plugins are **opt-in by default** — only `enabled` plugins load at startup.

---

## Official Bundled Plugins

### Platform / Gateway Adapters

Connect Hermes to messaging platforms. Each platform adapter handles inbound messages, outbound delivery, platform-specific features (threads, inline keyboards, adaptive cards), and user/chat allowlists.

| Plugin | Platform | Key Features |
|--------|----------|-------------|
| `telegram-platform` | Telegram | Threads/topics, streaming edits, native media, inline keyboards, slash commands, fallback network, mention gating |
| `slack-platform` | Slack | Socket Mode, slash commands, threads, mrkdwn, approval blocks, free-response channels, channel skill bindings |
| `discord-platform` | Discord | Gateway intents, slash commands, threads, embeds, roles, voice channel join |
| `whatsapp-platform` | WhatsApp | Local Node.js bridge (WA Web), DM/group policies, mention gating, free-response |
| `signal-platform` | Signal | signal-cli bridge, end-to-end encrypted |
| `sms-platform` | SMS (Twilio) | REST API + inbound webhook, plain text delivery |
| `teams-platform` | Microsoft Teams | Bot Framework, Adaptive Card approval prompts, personal/group/channel chats |
| `wecom-platform` | WeCom / 企业微信 | Smart Robot (WebSocket) + self-built callback (HTTP + AES), dual mode |
| `simplex-platform` | SimpleX Chat | Decentralized, no persistent user IDs, via local simplex-chat daemon |
| `matrix-platform` | Matrix | Federation, E2E encryption, room management |
| `imessage-platform` | iMessage | Native macOS bridge via AppleScript/Messages app |
| `photon-platform` | Photon Spectrum | iMessage via gRPC sidecar, managed Spectrum platform |
| `raft-platform` | Raft | Workspace agent via wake-channel bridge |

### Web Search Providers

| Plugin | Backend | API Key Required | Free Tier | Notes |
|--------|---------|:---:|:---:|-------|
| `web-ddgs` | DuckDuckGo | No | ✅ Unlimited | Privacy-respecting, no key |
| `web-searxng` | SearXNG | No (self-hosted) | ✅ Self-hosted | Requires `SEARXNG_URL` |
| `web-brave-free` | Brave Search | Yes | ✅ 2k/mo | `BRAVE_SEARCH_API_KEY` |
| `web-firecrawl` | Firecrawl | Yes* | — | Direct API or Nous gateway |
| `web-tavily` | Tavily | Yes | — | Search + extract + crawl |
| `web-exa` | Exa | Yes | — | Semantic search + content |
| `web-parallel` | Parallel.ai | Yes | — | Objective-tuned results |
| `web-xai` | xAI Grok | Yes* | — | OAuth or `XAI_API_KEY` |

### Browser Automation

| Plugin | Backend | Key Required | Notes |
|--------|---------|:---:|-------|
| `browser-browser-use` | Browser Use Cloud | Yes | `BROWSER_USE_API_KEY` or Nous subscription |
| `browser-browserbase` | Browserbase | Yes | `BROWSERBASE_API_KEY` + project, stealth/proxy |
| `browser-firecrawl` | Firecrawl Browser | Yes | `FIRECRAWL_API_KEY`, distinct from web plugin |

### Image Generation

| Plugin | Backend | Models |
|--------|---------|--------|
| `image_gen/fal` | FAL.ai | flux-2-klein, flux-2-pro, nano-banana, gpt-image-1.5 |

### Video Generation

| Plugin | Backend | Models |
|--------|---------|--------|
| `video_gen/fal` | FAL.ai | Veo 3.1, Kling, Pixverse (text-to-video, image-to-video) |
| `video_gen/xai` | xAI Grok | text-to-video, image-to-video, reference-to-video, video editing, video extension |

### Dashboard Authentication

| Plugin | Auth Method | Best For |
|--------|-------------|----------|
| `dashboard_auth/basic` | Username/password | Self-hosted, simple `scrypt` hashing, stateless HMAC tokens |
| `dashboard_auth/drain` | Shared bearer secret | Gateway drain-control endpoint, constant-time compare |
| `dashboard_auth/nous` | OAuth 2.0 + PKCE (Nous Portal) | Hosted agents, Nous subscription |
| `dashboard_auth/self-hosted` | OpenID Connect (generic) | Authentik, Keycloak, Zitadel, Authelia, Auth0, Okta, Google |

### Cron Providers

| Plugin | Backend | Use Case |
|--------|---------|----------|
| `cron_providers/chronos` | Nous NAS | Scale-to-zero hosted agents — wakes agent at fire time |

### Music & Audio

| Plugin | Tools | Auth |
|--------|-------|------|
| `spotify` | 7 tools: playback, devices, queue, search, playlists, albums, library | OAuth PKCE via `hermes auth spotify` |

### Security

| Plugin | Function |
|--------|----------|
| `security-guidance` | Appends warnings to file-write results for dangerous patterns (pickle, eval, os.system…). 25 rules forked from Anthropic's claude-plugins-off. Non-blocking — file is written, warning rides back to model. |

### Memory

| Plugin | Function |
|--------|----------|
| `memory/*` | Memory layer plugins — identity, workflow, project context persistence |

### Observability

| Plugin | Function |
|--------|----------|
| `observability/*` | Telemetry, tracing, and monitoring plugins |

### Utilities

| Plugin | Function |
|--------|----------|
| `disk-cleanup` | Auto-track and clean ephemeral files (test scripts, temp outputs, cron logs). Runs via plugin hooks — no agent action required. |
| `google_meet` | Join Google Meet, transcribe live captions, realtime audio (OpenAI Realtime + BlackHole). v3: remote node host mode. |
| `teams_pipeline` | Teams meeting pipeline — Graph-backed transcript-first meeting summaries |
| `context_engine/*` | Context management — compaction, summarization, and token budget control |
| `kanban` | Project kanban board integration |
| `hermes-achievements` | Gamification / achievement tracking |
| `model-providers/*` | Additional model provider adapters |

---

## Community Plugins

Community plugins are installed to `~/.hermes/plugins/`:

```bash
# Install a community plugin
hermes plugins install <name>     # install from registry
hermes plugins install <repo-url> # install from git
```

### Installing from npm

Some community skills ship as npm packages:

```bash
npx skillful install <skill-name>    # install a skill package
npx -y @scope/skill-name             # run directly
```

### Writing a Plugin

Plugins live in `~/.hermes/plugins/<name>/` and require:

```
my-plugin/
├── plugin.yaml    # metadata: name, version, description, plugin_type, requires
├── __init__.py    # Plugin class implementing the appropriate ABC
└── provider.py    # (optional) tool/service provider
```

#### `plugin.yaml` structure

```yaml
name: my-plugin
version: 1.0.0
description: What this plugin does
plugin_type: tool_provider       # tool_provider | platform_adapter | auth_provider | hook
requires:
  python: ">=3.10"
  env:
    - MY_API_KEY
  packages:
    - requests>=2.28
tools:                           # (tool_provider only)
  - name: my_tool
    description: Tool description for the LLM
  - name: another_tool
    description: Another tool
```

#### Plugin Types

| Type | Interface | When to Use |
|------|-----------|-------------|
| `tool_provider` | Register tools into the agent's toolset | Adding new capabilities (search, APIs, integrations) |
| `platform_adapter` | Inbound/outbound message relay | Connecting a messaging platform |
| `auth_provider` | Authentication flow for dashboard | Custom login methods |
| `hook` | Lifecycle hooks (startup, session_init, pre_tool, post_tool, shutdown) | Cross-cutting concerns (logging, security, cleanup) |

#### Hook Lifecycle

Plugins hook into Hermes lifecycle events:

| Hook | Fires | Use Case |
|------|-------|----------|
| `on_startup` | Gateway/TUI starts | Init connections, load config |
| `on_session_init` | New conversation starts | Inject context, set up per-session state |
| `pre_tool` | Before tool execution | Validation, rate limiting, logging |
| `post_tool` | After tool execution | Post-processing, security checks, cleanup |
| `on_shutdown` | Gateway/TUI stops | Graceful disconnection, flush state |

---

## Enabling/Disabling

```bash
hermes plugins enable telegram-platform    # enable
hermes plugins disable web-brave-free      # disable
hermes plugins                             # interactive TUI
```

Configuration in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - telegram-platform
    - discord-platform
    - web-ddgs
  disabled:
    - web-firecrawl
```

---

## Plugin Permissions

Plugins run in-process with agent-level access. Enable only plugins you trust. The `security-guidance` plugin provides an additional safety net — it scans file writes for dangerous patterns and warns the model.

For platform adapters, configure per-user and per-chat allowlists to restrict who can interact with your agent:

```yaml
platforms:
  telegram:
    allow_users: ["123456789"]    # only these Telegram user IDs
    allow_chats: ["-1001234567"]  # only these groups/channels
```

---

## See Also

- [MCP Reference](mcp-reference.md) — connecting Hermes to MCP servers for additional tools
- [IDE Integration Guide](ide-integration.md) — connecting Hermes to IDEs
- [Skills Hub](https://github.com/nousresearch/hermes-agent/tree/main/hermes_cli/skills_hub.py) — community skills registry
