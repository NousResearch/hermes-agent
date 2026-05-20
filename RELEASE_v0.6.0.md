# Hermes Agent v0.6.0 (v2026.3.30)

**Release Date:** March 30, 2026

> The multi-instance release — Profiles for running isolated agent instances, MCP server mode, Docker container, fallback provider chains, two new messaging platforms (Feishu/Lark and WeCom), Telegram webhook mode, Slack multi-workspace OAuth, 95 PRs and 16 resolved issues in 2 days.

---

## ✨ Highlights

- **Profiles — Multi-Instance Hermes** — Run multiple isolated Hermes instances from the same installation. Each profile gets its own config, memory, sessions, skills, and gateway service. Create with `hermes profile create`, switch with `hermes -p <name>`, export/import for sharing. Full token-lock isolation prevents two profiles from using the same bot credential. ([#3681](https://github.com/NousResearch/hermes-agent/pull/3681))

- **MCP Server Mode** — Expose Hermes conversations and sessions to any MCP-compatible client (Claude Desktop, Cursor, VS Code, etc.) via `hermes mcp serve`. Browse conversations, read messages, search across sessions, and manage attachments — all through the Model Context Protocol. Supports both stdio and Streamable HTTP transports. ([#3795](https://github.com/NousResearch/hermes-agent/pull/3795))

- **Docker Container** — Official Dockerfile for running Hermes Agent in a container. Supports both CLI and gateway modes with volume-mounted config. ([#3668](https://github.com/NousResearch/hermes-agent/pull/3668), closes [#850](https://github.com/NousResearch/hermes-agent/issues/850))

- **Ordered Fallback Provider Chain** — Configure multiple inference providers with automatic failover. When your primary provider returns errors or is unreachable, Hermes automatically tries the next provider in the chain. Configure via `fallback_providers` in config.yaml. ([#3813](https://github.com/NousResearch/hermes-agent/pull/3813), closes [#1734](https://github.com/NousResearch/hermes-agent/issues/1734))

- **Feishu/Lark Platform Support** — Full gateway adapter for Feishu (飞书) and Lark with event subscriptions, message cards, group chat, image/file attachments, and interactive card callbacks. ([#3799](https://github.com/NousResearch/hermes-agent/pull/3799), [#3817](https://github.com/NousResearch/hermes-agent/pull/3817), closes [#1788](https://github.com/NousResearch/hermes-agent/issues/1788))

- **WeCom (Enterprise WeChat) Platform Support** — New gateway adapter for WeCom (企业微信) with text/image/voice messages, group chats, and callback verification. ([#3847](https://github.com/NousResearch/hermes-agent/pull/3847))

- **Slack Multi-Workspace OAuth** — Connect a single Hermes gateway to multiple Slack workspaces via OAuth token file. Each workspace gets its own bot token, resolved dynamically per incoming event. ([#3903](https://github.com/NousResearch/hermes-agent/pull/3903))

- **Telegram Webhook Mode & Group Controls** — Run the Telegram adapter in webhook mode as an alternative to polling — faster response times and better for production deployments behind a reverse proxy. New group mention gating controls when the bot responds: always, only when @mentioned, or via regex triggers. ([#3880](https://github.com/NousResearch/hermes-agent/pull/3880), [#3870](https://github.com/NousResearch/hermes-agent/pull/3870))

- **Exa Search Backend** — Add Exa as an alternative web search and content extraction backend alongside Firecrawl and DuckDuckGo. Set `EXA_API_KEY` and configure as preferred backend. ([#3648](https://github.com/NousResearch/hermes-agent/pull/3648))

- **Skills & Credentials on Remote Backends** — Mount skill directories and credential files into Modal and Docker containers, so remote terminal sessions have access to the same skills and secrets as local execution. ([#3890](https://github.com/NousResearch/hermes-agent/pull/3890), [#3671](https://github.com/NousResearch/hermes-agent/pull/3671), closes [#3665](https://github.com/NousResearch/hermes-agent/issues/3665), [#3433](https://github.com/NousResearch/hermes-agent/issues/3433))

---

## 🏗️ Core Agent & Architecture

### Provider & Model Support
- **Ordered fallback provider chain** — automatic failover across multiple configured providers ([#3813](https://github.com/NousResearch/hermes-agent/pull/3813))
- **Fix api_mode on provider switch** — switching providers via `hermes model` now correctly clears stale `api_mode` instead of hardcoding `chat_completions`, fixing 404s for providers with Anthropic-compatible endpoints ([#3726](https://github.com/NousResearch/hermes-agent/pull/3726), [#3857](https://github.com/NousResearch/hermes-agent/pull/3857), closes [#3685](https://github.com/NousResearch/hermes-agent/issues/3685))
- **Stop silent OpenRouter fallback** — when no provider is configured, Hermes now raises a clear error instead of silently routing to OpenRouter ([#3807](https://github.com/NousResearch/hermes-agent/pull/3807), [#3862](https://github.com/NousResearch/hermes-agent/pull/3862))
- **Gemini 3.1 preview models** — added to OpenRouter and Nous Portal catalogs ([#3803](https://github.com/NousResearch/hermes-agent/pull/3803), closes [#3753](https://github.com/NousResearch/hermes-agent/issues/3753))
- **Gemini direct API context length** — full context length resolution for direct Google AI endpoints ([#3876](https://github.com/NousResearch/hermes-agent/pull/3876))
- **gpt-5.4-mini** added to Codex fallback catalog ([#3855](https://github.com/NousResearch/hermes-agent/pull/3855))
- **Curated model lists preferred** over live API probe when the probe returns fewer models ([#3856](https://github.com/NousResearch/hermes-agent/pull/3856), [#3867](https://github.com/NousResearch/hermes-agent/pull/3867))
- **User-friendly 429 rate limit messages** with Retry-After countdown ([#3809](https://github.com/NousResearch/hermes-agent/pull/3809))
- **Auxiliary client placeholder key** for local servers without auth requirements ([#3842](https://github.com/NousResearch/hermes-agent/pull/3842))
- **INFO-level logging** for auxiliary provider resolution ([#3866](https://github.com/NousResearch/hermes-agent/pull/3866))

### Agent Loop & Conversation
- **Subagent status reporting** — reports `completed` status when summary exists instead of generic failure ([#3829](https://github.com/NousResearch/hermes-agent/pull/3829))
- **Session log file updated during compression** — prevents stale file references after context compression ([#3835](https://github.com/NousResearch/hermes-agent/pull/3835))
- **Omit empty tools param** — sends no `tools` parameter when empty instead of `None`, fixing compatibility with strict providers ([#3820](https://github.com/NousResearch/hermes-agent/pull/3820))

### Profiles & Multi-Instance
- **Profiles system** — `hermes profile create/list/switch/delete/export/import/rename`. Each profile gets isolated HERMES_HOME, gateway service, CLI wrapper. Token locks prevent credential collisions. Tab completion for profile names. ([#3681](https://github.com/NousResearch/hermes-agent/pull/3681))
- **Profile-aware display paths** — all user-facing `~/.hermes` paths replaced with `display_hermes_home()` to show the correct profile directory ([#3623](https://github.com/NousResearch/hermes-agent/pull/3623))
- **Lazy display_hermes_home imports** — prevents `ImportError` during `hermes update` when modules cache stale bytecode ([#3776](https://github.com/NousResearch/hermes-agent/pull/3776))
- **HERMES_HOME for protected paths** — `.env` write-deny path now respects HERMES_HOME instead of hardcoded `~/.hermes` ([#3840](https://github.com/NousResearch/hermes-agent/pull/3840))

---

## 📱 Messaging Platforms (Gateway)

### New Platforms
- **Feishu/Lark** — Full adapter with event subscriptions, message cards, group chat, image/file attachments, interactive card callbacks ([#3799](https://github.com/NousResearch/hermes-agent/pull/3799), [#3817](https://github.com/NousResearch/hermes-agent/pull/3817))
- **WeCom (Enterprise WeChat)** — Text/image/voice messages, group chats, callback verification ([#3847](https://github.com/NousResearch/hermes-agent/pull/3847))

### Telegram
- **Webhook mode** — run as webhook endpoint instead of polling for production deployments ([#3880](https://github.com/NousResearch/hermes-agent/pull/3880))
- **Group mention gating & regex triggers** — configurable bot response behavior in groups: always, @mention-only, or regex-matched ([#3870](https://github.com/NousResearch/hermes-agent/pull/3870))
- **Gracefully handle deleted reply targets** — no more crashes when the message being replied to was deleted ([#3858](https://github.com/NousResearch/hermes-agent/pull/3858), closes [#3229](https://github.com/NousResearch/hermes-agent/issues/3229))

### Discord
- **Message processing reactions** — adds a reaction emoji while processing and removes it when done, giving visual feedback in channels ([#3871](https://github.com/NousResearch/hermes-agent/pull/3871))
- **DISCORD_IGNORE_NO_MENTION** — skip messages that @mention other users/bots but not Hermes ([#3640](https://github.com/NousResearch/hermes-agent/pull/3640))
- **Clean up deferred "thinking..."** — properly removes the "thinking..." indicator after slash commands complete ([#3674](https://github.com/NousResearch/hermes-agent/pull/3674), closes [#3595](https://github.com/NousResearch/hermes-agent/issues/3595))

### Slack
- **Multi-workspace OAuth** — connect to multiple Slack workspaces from a single gateway via OAuth token file ([#3903](https://github.com/NousResearch/hermes-agent/pull/3903))

### WhatsApp
- **Persistent aiohttp session** — reuse HTTP sessions across requests instead of creating new ones per message ([#3818](https://github.com/NousResearch/hermes-agent/pull/3818))
- **LID↔phone alias resolution** — correctly match Linked ID and phone number formats in allowlists ([#3830](https://github.com/NousResearch/hermes-agent/pull/3830))
- **Skip reply prefix in bot mode** — cleaner message formatting when running as a WhatsApp bot ([#3931](https://github.com/NousResearch/hermes-agent/pull/3931))

### Matrix
- **Native voice messages via MSC3245** — send voice messages as proper Matrix voice events instead of file attachments ([#3877](https://github.com/NousResearch/hermes-agent/pull/3877))

### Mattermost
- **Configurable mention behavior** — respond to messages without requiring @mention ([#3664](https://github.com/NousResearch/hermes-agent/pull/3664))

### Signal
- **URL-encode phone numbers** and correct attachment RPC parameter — fixes delivery failures with certain phone number formats ([#3670](https://github.com/NousResearch/hermes-agent/pull/3670)) — @kshitijk4poor

### Email
- **Close SMTP/IMAP connections on failure** — prevents connection leaks during error scenarios ([#3804](https://github.com/NousResearch/hermes-agent/pull/3804))

### Gateway Core
- **Atomic config writes** — use atomic file writes for config.yaml to prevent data loss during crashes ([#3800](https://github.com/NousResearch/hermes-agent/pull/3800))
- **Home channel env overrides** — apply environment variable overrides for home channels consistently ([#3796](https://github.com/NousResearch/hermes-agent/pull/3796), [#3808](https://github.com/NousResearch/hermes-agent/pull/3808))
- **Replace print() with logger** — BasePlatformAdapter now uses proper logging instead of print statements ([#3669](https://github.com/NousResearch/hermes-agent/pull/3669))
- **Cron delivery labels** — resolve human-friendly delivery labels via channel directory ([#3860](https://github.com/NousResearch/hermes-agent/pull/3860), closes [#1945](https://github.com/NousResearch/hermes-agent/issues/1945))
- **Cron [SILENT] tightening** — prevent agents from prefixing reports with [SILENT] to suppress delivery ([#3901](https://github.com/NousResearch/hermes-agent/pull/3901))
- **Background task media delivery** and vision download timeout fixes ([#3919](https://github.com/NousResearch/hermes-agent/pull/3919))
- **Boot-md hook** — example built-in hook to run a BOOT.md file on gateway startup ([#3733](https://github.com/NousResearch/hermes-agent/pull/3733))

---

## 🖥️ CLI & User Experience

### Interactive CLI
- **Configurable tool preview length** — show full file paths by default instead of truncating at 40 chars ([#3841](https://github.com/NousResearch/hermes-agent/pull/3841))
- **Tool token context display** — `hermes tools` checklist now shows estimated token cost per toolset ([#3805](https://github.com/NousResearch/hermes-agent/pull/3805))
- **/bg spinner TUI fix** — route background task spinner through the TUI widget to prevent status bar collision ([#3643](https://github.com/NousResearch/hermes-agent/pull/3643))
- **Prevent status bar wrapping** into duplicate rows ([#3883](https://github.com/NousResearch/hermes-agent/pull/3883)) — @kshitijk4poor
- **Handle closed stdout ValueError** in safe print paths — fixes crashes when stdout is closed during gateway thread shutdown ([#3843](https://github.com/NousResearch/hermes-agent/pull/3843), closes [#3534](https://github.com/NousResearch/hermes-agent/issues/3534))
- **Remove input() from /tools disable** — eliminates freeze in terminal when disabling tools ([#3918](https://github.com/NousResearch/hermes-agent/pull/3918))
- **TTY guard for interactive CLI commands** — prevent CPU spin when launched without a terminal ([#3933](https://github.com/NousResearch/hermes-agent/pull/3933))
- **Argparse entrypoint** — use argparse in the top-level launcher for cleaner error handling ([#3874](https://github.com/NousResearch/hermes-agent/pull/3874))
- **Lazy-initialized tools show yellow** in banner instead of red, reducing false alarm about "missing" tools ([#3822](https://github.com/NousResearch/hermes-agent/pull/3822))
- **Honcho tools shown in banner** when configured ([#3810](https://github.com/NousResearch/hermes-agent/pull/3810))

### Setup & Configuration
- **Auto-install matrix-nio** during `hermes setup` when Matrix is selected ([#3802](https://github.com/NousResearch/hermes-agent/pull/3802), [#3873](https://github.com/NousResearch/hermes-agent/pull/3873))
- **Session export stdout support** — export sessions to stdout with `-` for piping ([#3641](https://github.com/NousResearch/hermes-agent/pull/3641), closes [#3609](https://github.com/NousResearch/hermes-agent/issues/3609))
- **Configurable approval timeouts** — set how long dangerous command approval prompts wait before auto-denying ([#3886](https://github.com/NousResearch/hermes-agent/pull/3886), closes [#3765](https://github.com/NousResearch/hermes-agent/issues/3765))
- **Clear __pycache__ during update** — prevents stale bytecode ImportError after `hermes update` ([#3819](https://github.com/NousResearch/hermes-agent/pull/3819))

---

## 🔧 Tool System

### MCP
- **MCP Server Mode** — `hermes mcp serve` exposes conversations, sessions, and attachments to MCP clients via stdio or Streamable HTTP ([#3795](https://github.com/NousResearch/hermes-agent/pull/3795))
- **Dynamic tool discovery** — respond to `notifications/tools/list_changed` events to pick up new tools from MCP servers without reconnecting ([#3812](https://github.com/NousResearch/hermes-agent/pull/3812))
- **Non-deprecated HTTP transport** — switched from `sse_client` to `streamable_http_client` ([#3646](https://github.com/NousResearch/hermes-agent/pull/3646))

### Web Tools
- **Exa search backend** — alternative to Firecrawl and DuckDuckGo for web search and extraction ([#3648](https://github.com/NousResearch/hermes-agent/pull/3648))

### Browser
- **Guard against None LLM responses** in browser snapshot and vision tools ([#3642](https://github.com/NousResearch/hermes-agent/pull/3642))

### Terminal & Remote Backends
- **Mount skill directories** into Modal and Docker containers ([#3890](https://github.com/NousResearch/hermes-agent/pull/3890))
- **Mount credential files** into remote backends with mtime+size caching ([#3671](https://github.com/NousResearch/hermes-agent/pull/3671))
- **Preserve partial output** when commands time out instead of losing everything ([#3868](https://github.com/NousResearch/hermes-agent/pull/3868))
- **Stop marking persisted env vars as missing** on remote backends ([#3650](https://github.com/NousResearch/hermes-agent/pull/3650))

### Audio
- **.aac format support** in transcription tool ([#3865](https://github.com/NousResearch/hermes-agent/pull/3865), closes [#1963](https://github.com/NousResearch/hermes-agent/issues/1963))
- **Audio download retry** — retry logic for `cache_audio_from_url` matching the existing image download pattern ([#3401](https://github.com/NousResearch/hermes-agent/pull/3401)) — @binhnt92

### Vision
- **Reject non-image files** and enforce website-only policy for vision analysis ([#3845](https://github.com/NousResearch/hermes-agent/pull/3845))

### Tool Schema
- **Ensure name field** always present in tool definitions, fixing `KeyError: 'name'` crashes ([#3811](https://github.com/NousResearch/hermes-agent/pull/3811), closes [#3729](https://github.com/NousResearch/hermes-agent/issues/3729))

### ACP (Editor Integration)
- **Complete session management surface** for VS Code/Zed/JetBrains clients — proper task lifecycle, cancel support, session persistence ([#3675](https://github.com/NousResearch/hermes-agent/pull/3675))

---

## 🧩 Skills & Plugins

### Skills System
- **External skill directories** — configure additional skill directories via `skills.external_dirs` in config.yaml ([#3678](https://github.com/NousResearch/hermes-agent/pull/3678))
- **Category path traversal blocked** — prevents `../` attacks in skill category names ([#3844](https://github.com/NousResearch/hermes-agent/pull/3844))
- **parallel-cli moved to optional-skills** — reduces default skill footprint ([#3673](https://github.com/NousResearch/hermes-agent/pull/3673)) — @kshitijk4poor

### New Skills
- **memento-flashcards** — spaced repetition flashcard system ([#3827](https://github.com/NousResearch/hermes-agent/pull/3827))
- **songwriting-and-ai-music** — songwriting craft and AI music generation prompts ([#3834](https://github.com/NousResearch/hermes-agent/pull/3834))
- **SiYuan Note** — integration with SiYuan note-taking app ([#3742](https://github.com/NousResearch/hermes-agent/pull/3742))
- **Scrapling** — web scraping skill using Scrapling library ([#3742](https://github.com/NousResearch/hermes-agent/pull/3742))
- **one-three-one-rule** — communication framework skill ([#3797](https://github.com/NousResearch/hermes-agent/pull/3797))

### Plugin System
- **Plugin enable/disable commands** — `hermes plugins enable/disable <name>` for managing plugin state without removing them ([#3747](https://github.com/NousResearch/hermes-agent/pull/3747))
- **Plugin message injection** — plugins can now inject messages into the conversation stream on behalf of the user via `ctx.inject_message()` ([#3778](https://github.com/NousResearch/hermes-agent/pull/3778)) — @winglian
- **Honcho self-hosted support** — allow local Honcho instances without requiring an API key ([#3644](https://github.com/NousResearch/hermes-agent/pull/3644))

---

## 🔒 Security & Reliability

### Security Hardening
- **Hardened dangerous command detection** — expanded pattern matching for risky shell commands and added file tool path guards for sensitive locations (`/etc/`, `/boot/`, docker.sock) ([#3872](https://github.com/NousResearch/hermes-agent/pull/3872))
- **Sensitive path write checks** in approval system — catch writes to system config files through file tools, not just terminal ([#3859](https://github.com/NousResearch/hermes-agent/pull/3859))
- **Secret redaction expansion** — now covers ElevenLabs, Tavily, and Exa API keys ([#3920](https://github.com/NousResearch/hermes-agent/pull/3920))
- **Vision file rejection** — reject non-image files passed to vision analysis to prevent information disclosure ([#3845](https://github.com/NousResearch/hermes-agent/pull/3845))
- **Category path traversal blocking** — prevent directory traversal in skill category names ([#3844](https://github.com/NousResearch/hermes-agent/pull/3844))

### Reliability
- **Atomic config.yaml writes** — prevent data loss during gateway crashes ([#3800](https://github.com/NousResearch/hermes-agent/pull/3800))
- **Clear __pycache__ on update** — prevent stale bytecode from causing ImportError after updates ([#3819](https://github.com/NousResearch/hermes-agent/pull/3819))
- **Lazy imports for update safety** — prevent ImportError chains during `hermes update` when modules reference new functions ([#3776](https://github.com/NousResearch/hermes-agent/pull/3776))
- **Restore terminalbench2 from patch corruption** — recovered file damaged by patch tool's secret redaction ([#3801](https://github.com/NousResearch/hermes-agent/pull/3801))
- **Terminal timeout preserves partial output** — no more lost command output on timeout ([#3868](https://github.com/NousResearch/hermes-agent/pull/3868))

---

## 🐛 Notable Bug Fixes

- **OpenClaw migration model config overwrite** — migration no longer overwrites model config dict with a string ([#3924](https://github.com/NousResearch/hermes-agent/pull/3924)) — @0xbyt4
- **OpenClaw migration expanded** — covers full data footprint including sessions, cron, memory ([#3869](https://github.com/NousResearch/hermes-agent/pull/3869))
- **Telegram deleted reply targets** — gracefully handle replies to deleted messages instead of crashing ([#3858](https://github.com/NousResearch/hermes-agent/pull/3858))
- **Discord "thinking..." persistence** — properly cleans up deferred response indicators ([#3674](https://github.com/NousResearch/hermes-agent/pull/3674))
- **WhatsApp LID↔phone aliases** — fixes allowlist matching failures with Linked ID format ([#3830](https://github.com/NousResearch/hermes-agent/pull/3830))
- **Signal URL-encoded phone numbers** — fixes delivery failures with certain formats ([#3670](https://github.com/NousResearch/hermes-agent/pull/3670))
- **Email connection leaks** — properly close SMTP/IMAP connections on error ([#3804](https://github.com/NousResearch/hermes-agent/pull/3804))
- **_safe_print ValueError** — no more gateway thread crashes on closed stdout ([#3843](https://github.com/NousResearch/hermes-agent/pull/3843))
- **Tool schema KeyError 'name'** — ensure name field always present in tool definitions ([#3811](https://github.com/NousResearch/hermes-agent/pull/3811))
- **api_mode stale on provider switch** — correctly clear when switching providers via `hermes model` ([#3857](https://github.com/NousResearch/hermes-agent/pull/3857))

---

## 🧪 Testing

- Resolved 10+ CI failures across hooks, tiktoken, plugins, and skill tests ([#3848](https://github.com/NousResearch/hermes-agent/pull/3848), [#3721](https://github.com/NousResearch/hermes-agent/pull/3721), [#3936](https://github.com/NousResearch/hermes-agent/pull/3936))

---

## 📚 Documentation

- **Comprehensive OpenClaw migration guide** — step-by-step guide for migrating from OpenClaw/Claw3D to Hermes Agent ([#3864](https://github.com/NousResearch/hermes-agent/pull/3864), [#3900](https://github.com/NousResearch/hermes-agent/pull/3900))
- **Credential file passthrough docs** — document how to forward credential files and env vars to remote backends ([#3677](https://github.com/NousResearch/hermes-agent/pull/3677))
- **DuckDuckGo requirements clarified** — note runtime dependency on duckduckgo-search package ([#3680](https://github.com/NousResearch/hermes-agent/pull/3680))
- **Skills catalog updated** — added red-teaming category and optional skills listing ([#3745](https://github.com/NousResearch/hermes-agent/pull/3745))
- **Feishu docs MDX fix** — escape angle-bracket URLs that break Docusaurus build ([#3902](https://github.com/NousResearch/hermes-agent/pull/3902))

---

## 👥 Contributors

### Core
- **@teknium1** — 90 PRs across all subsystems

### Community Contributors
- **@kshitijk4poor** — 3 PRs: Signal phone number fix ([#3670](https://github.com/NousResearch/hermes-agent/pull/3670)), parallel-cli to optional-skills ([#3673](https://github.com/NousResearch/hermes-agent/pull/3673)), status bar wrapping fix ([#3883](https://github.com/NousResearch/hermes-agent/pull/3883))
- **@winglian** — 1 PR: Plugin message injection interface ([#3778](https://github.com/NousResearch/hermes-agent/pull/3778))
- **@binhnt92** — 1 PR: Audio download retry logic ([#3401](https://github.com/NousResearch/hermes-agent/pull/3401))
- **@0xbyt4** — 1 PR: OpenClaw migration model config fix ([#3924](https://github.com/NousResearch/hermes-agent/pull/3924))

### Issues Resolved from Community
@Material-Scientist ([#850](https://github.com/NousResearch/hermes-agent/issues/850)), @hanxu98121 ([#1734](https://github.com/NousResearch/hermes-agent/issues/1734)), @penwyp ([#1788](https://github.com/NousResearch/hermes-agent/issues/1788)), @dan-and ([#1945](https://github.com/NousResearch/hermes-agent/issues/1945)), @AdrianScott ([#1963](https://github.com/NousResearch/hermes-agent/issues/1963)), @clawdbot47 ([#3229](https://github.com/NousResearch/hermes-agent/issues/3229)), @alanfwilliams ([#3404](https://github.com/NousResearch/hermes-agent/issues/3404)), @kentimsit ([#3433](https://github.com/NousResearch/hermes-agent/issues/3433)), @hayka-pacha ([#3534](https://github.com/NousResearch/hermes-agent/issues/3534)), @primmer ([#3595](https://github.com/NousResearch/hermes-agent/issues/3595)), @dagelf ([#3609](https://github.com/NousResearch/hermes-agent/issues/3609)), @HenkDz ([#3685](https://github.com/NousResearch/hermes-agent/issues/3685)), @tmdgusya ([#3729](https://github.com/NousResearch/hermes-agent/issues/3729)), @TypQxQ ([#3753](https://github.com/NousResearch/hermes-agent/issues/3753)), @acsezen ([#3765](https://github.com/NousResearch/hermes-agent/issues/3765))

---

**Full Changelog**: [v2026.3.28...v2026.3.30](https://github.com/NousResearch/hermes-agent/compare/v2026.3.28...v2026.3.30)
