# Hermes Agent v0.3.0 (v2026.3.17)

**Release Date:** March 17, 2026

> The streaming, plugins, and provider release — unified real-time token delivery, first-class plugin architecture, rebuilt provider system with Vercel AI Gateway, native Anthropic provider, smart approvals, live Chrome CDP browser connect, ACP IDE integration, Honcho memory, voice mode, persistent shell, and 50+ bug fixes across every platform.

---

## ✨ Highlights

- **Unified Streaming Infrastructure** — Real-time token-by-token delivery in CLI and all gateway platforms. Responses stream as they're generated instead of arriving as a block. ([#1538](https://github.com/NousResearch/hermes-agent/pull/1538))

- **First-Class Plugin Architecture** — Drop Python files into `~/.hermes/plugins/` to extend Hermes with custom tools, commands, and hooks. No forking required. ([#1544](https://github.com/NousResearch/hermes-agent/pull/1544), [#1555](https://github.com/NousResearch/hermes-agent/pull/1555))

- **Native Anthropic Provider** — Direct Anthropic API calls with Claude Code credential auto-discovery, OAuth PKCE flows, and native prompt caching. No OpenRouter middleman needed. ([#1097](https://github.com/NousResearch/hermes-agent/pull/1097))

- **Smart Approvals + /stop Command** — Codex-inspired approval system that learns which commands are safe and remembers your preferences. `/stop` kills the current agent run immediately. ([#1543](https://github.com/NousResearch/hermes-agent/pull/1543))

- **Honcho Memory Integration** — Async memory writes, configurable recall modes, session title integration, and multi-user isolation in gateway mode. By @erosika. ([#736](https://github.com/NousResearch/hermes-agent/pull/736))

- **Voice Mode** — Push-to-talk in CLI, voice notes in Telegram/Discord, Discord voice channel support, and local Whisper transcription via faster-whisper. ([#1299](https://github.com/NousResearch/hermes-agent/pull/1299), [#1185](https://github.com/NousResearch/hermes-agent/pull/1185), [#1429](https://github.com/NousResearch/hermes-agent/pull/1429))

- **Concurrent Tool Execution** — Multiple independent tool calls now run in parallel via ThreadPoolExecutor, significantly reducing latency for multi-tool turns. ([#1152](https://github.com/NousResearch/hermes-agent/pull/1152))

- **PII Redaction** — When `privacy.redact_pii` is enabled, personally identifiable information is automatically scrubbed before sending context to LLM providers. ([#1542](https://github.com/NousResearch/hermes-agent/pull/1542))

- **`/browser connect` via CDP** — Attach browser tools to a live Chrome instance through Chrome DevTools Protocol. Debug, inspect, and interact with pages you already have open. ([#1549](https://github.com/NousResearch/hermes-agent/pull/1549))

- **Vercel AI Gateway Provider** — Route Hermes through Vercel's AI Gateway for access to their model catalog and infrastructure. ([#1628](https://github.com/NousResearch/hermes-agent/pull/1628))

- **Centralized Provider Router** — Rebuilt provider system with `call_llm` API, unified `/model` command, auto-detect provider on model switch, and direct endpoint overrides for auxiliary/delegation clients. ([#1003](https://github.com/NousResearch/hermes-agent/pull/1003), [#1506](https://github.com/NousResearch/hermes-agent/pull/1506), [#1375](https://github.com/NousResearch/hermes-agent/pull/1375))

- **ACP Server (IDE Integration)** — VS Code, Zed, and JetBrains can now connect to Hermes as an agent backend, with full slash command support. ([#1254](https://github.com/NousResearch/hermes-agent/pull/1254), [#1532](https://github.com/NousResearch/hermes-agent/pull/1532))

- **Persistent Shell Mode** — Local and SSH terminal backends can maintain shell state across tool calls — cd, env vars, and aliases persist. By @alt-glitch. ([#1067](https://github.com/NousResearch/hermes-agent/pull/1067), [#1483](https://github.com/NousResearch/hermes-agent/pull/1483))

- **Agentic On-Policy Distillation (OPD)** — New RL training environment for distilling agent policies, expanding the Atropos training ecosystem. ([#1149](https://github.com/NousResearch/hermes-agent/pull/1149))

---

## 🏗️ Core Agent & Architecture

### Provider & Model Support
- **Centralized provider router** with `call_llm` API and unified `/model` command — switch models and providers seamlessly ([#1003](https://github.com/NousResearch/hermes-agent/pull/1003))
- **Vercel AI Gateway** provider support ([#1628](https://github.com/NousResearch/hermes-agent/pull/1628))
- **Auto-detect provider** when switching models via `/model` ([#1506](https://github.com/NousResearch/hermes-agent/pull/1506))
- **Direct endpoint overrides** for auxiliary and delegation clients — point vision/subagent calls at specific endpoints ([#1375](https://github.com/NousResearch/hermes-agent/pull/1375))
- **Native Anthropic auxiliary vision** — use Claude's native vision API instead of routing through OpenAI-compatible endpoints ([#1377](https://github.com/NousResearch/hermes-agent/pull/1377))
- Anthropic OAuth flow improvements — auto-run `claude setup-token`, reauthentication, PKCE state persistence, identity fingerprinting ([#1132](https://github.com/NousResearch/hermes-agent/pull/1132), [#1360](https://github.com/NousResearch/hermes-agent/pull/1360), [#1396](https://github.com/NousResearch/hermes-agent/pull/1396), [#1597](https://github.com/NousResearch/hermes-agent/pull/1597))
- Fix adaptive thinking without `budget_tokens` for Claude 4.6 models — by @ASRagab ([#1128](https://github.com/NousResearch/hermes-agent/pull/1128))
- Fix Anthropic cache markers through adapter — by @brandtcormorant ([#1216](https://github.com/NousResearch/hermes-agent/pull/1216))
- Retry Anthropic 429/529 errors and surface details to users — by @0xbyt4 ([#1585](https://github.com/NousResearch/hermes-agent/pull/1585))
- Fix Anthropic adapter max_tokens, fallback crash, proxy base_url — by @0xbyt4 ([#1121](https://github.com/NousResearch/hermes-agent/pull/1121))
- Fix DeepSeek V3 parser dropping multiple parallel tool calls — by @mr-emmett-one ([#1365](https://github.com/NousResearch/hermes-agent/pull/1365), [#1300](https://github.com/NousResearch/hermes-agent/pull/1300))
- Accept unlisted models with warning instead of rejecting ([#1047](https://github.com/NousResearch/hermes-agent/pull/1047), [#1102](https://github.com/NousResearch/hermes-agent/pull/1102))
- Skip reasoning params for unsupported OpenRouter models ([#1485](https://github.com/NousResearch/hermes-agent/pull/1485))
- MiniMax Anthropic API compatibility fix ([#1623](https://github.com/NousResearch/hermes-agent/pull/1623))
- Custom endpoint `/models` verification and `/v1` base URL suggestion ([#1480](https://github.com/NousResearch/hermes-agent/pull/1480))
- Resolve delegation providers from `custom_providers` config ([#1328](https://github.com/NousResearch/hermes-agent/pull/1328))
- Kimi model additions and User-Agent fix ([#1039](https://github.com/NousResearch/hermes-agent/pull/1039))
- Strip `call_id`/`response_item_id` for Mistral compatibility ([#1058](https://github.com/NousResearch/hermes-agent/pull/1058))

### Agent Loop & Conversation
- **Anthropic Context Editing API** support ([#1147](https://github.com/NousResearch/hermes-agent/pull/1147))
- Improved context compaction handoff summaries — compressor now preserves more actionable state ([#1273](https://github.com/NousResearch/hermes-agent/pull/1273))
- Sync session_id after mid-run context compression ([#1160](https://github.com/NousResearch/hermes-agent/pull/1160))
- Session hygiene threshold tuned to 50% for more proactive compression ([#1096](https://github.com/NousResearch/hermes-agent/pull/1096), [#1161](https://github.com/NousResearch/hermes-agent/pull/1161))
- Include session ID in system prompt via `--pass-session-id` flag ([#1040](https://github.com/NousResearch/hermes-agent/pull/1040))
- Prevent closed OpenAI client reuse across retries ([#1391](https://github.com/NousResearch/hermes-agent/pull/1391))
- Sanitize chat payloads and provider precedence ([#1253](https://github.com/NousResearch/hermes-agent/pull/1253))
- Handle dict tool call arguments from Codex and local backends ([#1393](https://github.com/NousResearch/hermes-agent/pull/1393), [#1440](https://github.com/NousResearch/hermes-agent/pull/1440))

### Memory & Sessions
- **Improve memory prioritization** — user preferences and corrections weighted above procedural knowledge ([#1548](https://github.com/NousResearch/hermes-agent/pull/1548))
- Tighter memory and session recall guidance in system prompts ([#1329](https://github.com/NousResearch/hermes-agent/pull/1329))
- Persist CLI token counts to session DB for `/insights` ([#1498](https://github.com/NousResearch/hermes-agent/pull/1498))
- Keep Honcho recall out of the cached system prefix ([#1201](https://github.com/NousResearch/hermes-agent/pull/1201))
- Correct `seed_ai_identity` to use `session.add_messages()` ([#1475](https://github.com/NousResearch/hermes-agent/pull/1475))
- Isolate Honcho session routing for multi-user gateway ([#1500](https://github.com/NousResearch/hermes-agent/pull/1500))

---

## 📱 Messaging Platforms (Gateway)

### Gateway Core
- **System gateway service mode** — run as a system-level systemd service, not just user-level ([#1371](https://github.com/NousResearch/hermes-agent/pull/1371))
- **Gateway install scope prompts** — choose user vs system scope during setup ([#1374](https://github.com/NousResearch/hermes-agent/pull/1374))
- **Reasoning hot reload** — change reasoning settings without restarting the gateway ([#1275](https://github.com/NousResearch/hermes-agent/pull/1275))
- Default group sessions to per-user isolation — no more shared state across users in group chats ([#1495](https://github.com/NousResearch/hermes-agent/pull/1495), [#1417](https://github.com/NousResearch/hermes-agent/pull/1417))
- Harden gateway restart recovery ([#1310](https://github.com/NousResearch/hermes-agent/pull/1310))
- Cancel active runs during shutdown ([#1427](https://github.com/NousResearch/hermes-agent/pull/1427))
- SSL certificate auto-detection for NixOS and non-standard systems ([#1494](https://github.com/NousResearch/hermes-agent/pull/1494))
- Auto-detect D-Bus session bus for `systemctl --user` on headless servers ([#1601](https://github.com/NousResearch/hermes-agent/pull/1601))
- Auto-enable systemd linger during gateway install on headless servers ([#1334](https://github.com/NousResearch/hermes-agent/pull/1334))
- Fall back to module entrypoint when `hermes` is not on PATH ([#1355](https://github.com/NousResearch/hermes-agent/pull/1355))
- Fix dual gateways on macOS launchd after `hermes update` ([#1567](https://github.com/NousResearch/hermes-agent/pull/1567))
- Remove recursive ExecStop from systemd units ([#1530](https://github.com/NousResearch/hermes-agent/pull/1530))
- Prevent logging handler accumulation in gateway mode ([#1251](https://github.com/NousResearch/hermes-agent/pull/1251))
- Restart on retryable startup failures — by @jplew ([#1517](https://github.com/NousResearch/hermes-agent/pull/1517))
- Backfill model on gateway sessions after agent runs ([#1306](https://github.com/NousResearch/hermes-agent/pull/1306))
- PID-based gateway kill and deferred config write ([#1499](https://github.com/NousResearch/hermes-agent/pull/1499))

### Telegram
- Buffer media groups to prevent self-interruption from photo bursts ([#1341](https://github.com/NousResearch/hermes-agent/pull/1341), [#1422](https://github.com/NousResearch/hermes-agent/pull/1422))
- Retry on transient TLS failures during connect and send ([#1535](https://github.com/NousResearch/hermes-agent/pull/1535))
- Harden polling conflict handling ([#1339](https://github.com/NousResearch/hermes-agent/pull/1339))
- Escape chunk indicators and inline code in MarkdownV2 ([#1478](https://github.com/NousResearch/hermes-agent/pull/1478), [#1626](https://github.com/NousResearch/hermes-agent/pull/1626))
- Check updater/app state before disconnect ([#1389](https://github.com/NousResearch/hermes-agent/pull/1389))

### Discord
- `/thread` command with `auto_thread` config and media metadata fixes ([#1178](https://github.com/NousResearch/hermes-agent/pull/1178))
- Auto-thread on @mention, skip mention text in bot threads ([#1438](https://github.com/NousResearch/hermes-agent/pull/1438))
- Retry without reply reference for system messages ([#1385](https://github.com/NousResearch/hermes-agent/pull/1385))
- Preserve native document and video attachment support ([#1392](https://github.com/NousResearch/hermes-agent/pull/1392))
- Defer discord adapter annotations to avoid optional import crashes ([#1314](https://github.com/NousResearch/hermes-agent/pull/1314))

### Slack
- Thread handling overhaul — progress messages, responses, and session isolation all respect threads ([#1103](https://github.com/NousResearch/hermes-agent/pull/1103))
- Formatting, reactions, user resolution, and command improvements ([#1106](https://github.com/NousResearch/hermes-agent/pull/1106))
- Fix MAX_MESSAGE_LENGTH 3900 → 39000 ([#1117](https://github.com/NousResearch/hermes-agent/pull/1117))
- File upload fallback preserves thread context — by @0xbyt4 ([#1122](https://github.com/NousResearch/hermes-agent/pull/1122))
- Improve setup guidance ([#1387](https://github.com/NousResearch/hermes-agent/pull/1387))

### Email
- Fix IMAP UID tracking and SMTP TLS verification ([#1305](https://github.com/NousResearch/hermes-agent/pull/1305))
- Add `skip_attachments` option via config.yaml ([#1536](https://github.com/NousResearch/hermes-agent/pull/1536))

### Home Assistant
- Event filtering closed by default ([#1169](https://github.com/NousResearch/hermes-agent/pull/1169))

---

## 🖥️ CLI & User Experience

### Interactive CLI
- **Persistent CLI status bar** — always-visible model, provider, and token counts ([#1522](https://github.com/NousResearch/hermes-agent/pull/1522))
- **File path autocomplete** in the input prompt ([#1545](https://github.com/NousResearch/hermes-agent/pull/1545))
- **`/plan` command** — generate implementation plans from specs ([#1372](https://github.com/NousResearch/hermes-agent/pull/1372), [#1381](https://github.com/NousResearch/hermes-agent/pull/1381))
- **Major `/rollback` improvements** — richer checkpoint history, clearer UX ([#1505](https://github.com/NousResearch/hermes-agent/pull/1505))
- **Preload CLI skills on launch** — skills are ready before the first prompt ([#1359](https://github.com/NousResearch/hermes-agent/pull/1359))
- **Centralized slash command registry** — all commands defined once, consumed everywhere ([#1603](https://github.com/NousResearch/hermes-agent/pull/1603))
- `/bg` alias for `/background` ([#1590](https://github.com/NousResearch/hermes-agent/pull/1590))
- Prefix matching for slash commands — `/mod` resolves to `/model` ([#1320](https://github.com/NousResearch/hermes-agent/pull/1320))
- `/new`, `/reset`, `/clear` now start genuinely fresh sessions ([#1237](https://github.com/NousResearch/hermes-agent/pull/1237))
- Accept session ID prefixes for session actions ([#1425](https://github.com/NousResearch/hermes-agent/pull/1425))
- TUI prompt and accent output now respect active skin ([#1282](https://github.com/NousResearch/hermes-agent/pull/1282))
- Centralize tool emoji metadata in registry + skin integration ([#1484](https://github.com/NousResearch/hermes-agent/pull/1484))
- "View full command" option added to dangerous command approval — by @teknium1 based on design by community ([#887](https://github.com/NousResearch/hermes-agent/pull/887))
- Non-blocking startup update check and banner deduplication ([#1386](https://github.com/NousResearch/hermes-agent/pull/1386))
- `/reasoning` command output ordering and inline think extraction fixes ([#1031](https://github.com/NousResearch/hermes-agent/pull/1031))
- Verbose mode shows full untruncated output ([#1472](https://github.com/NousResearch/hermes-agent/pull/1472))
- Fix `/status` to report live state and tokens ([#1476](https://github.com/NousResearch/hermes-agent/pull/1476))
- Seed a default global SOUL.md ([#1311](https://github.com/NousResearch/hermes-agent/pull/1311))

### Setup & Configuration
- **OpenClaw migration** during first-time setup — by @kshitijk4poor ([#981](https://github.com/NousResearch/hermes-agent/pull/981))
- `hermes claw migrate` command + migration docs ([#1059](https://github.com/NousResearch/hermes-agent/pull/1059))
- Smart vision setup that respects the user's chosen provider ([#1323](https://github.com/NousResearch/hermes-agent/pull/1323))
- Handle headless setup flows end-to-end ([#1274](https://github.com/NousResearch/hermes-agent/pull/1274))
- Prefer curses over `simple_term_menu` in setup.py ([#1487](https://github.com/NousResearch/hermes-agent/pull/1487))
- Show effective model and provider in `/status` ([#1284](https://github.com/NousResearch/hermes-agent/pull/1284))
- Config set examples use placeholder syntax ([#1322](https://github.com/NousResearch/hermes-agent/pull/1322))
- Reload .env over stale shell overrides ([#1434](https://github.com/NousResearch/hermes-agent/pull/1434))
- Fix is_coding_plan NameError crash — by @0xbyt4 ([#1123](https://github.com/NousResearch/hermes-agent/pull/1123))
- Add missing packages to setuptools config — by @alt-glitch ([#912](https://github.com/NousResearch/hermes-agent/pull/912))
- Installer: clarify why sudo is needed at every prompt ([#1602](https://github.com/NousResearch/hermes-agent/pull/1602))

---

## 🔧 Tool System

### Terminal & Execution
- **Persistent shell mode** for local and SSH backends — maintain shell state across tool calls — by @alt-glitch ([#1067](https://github.com/NousResearch/hermes-agent/pull/1067), [#1483](https://github.com/NousResearch/hermes-agent/pull/1483))
- **Tirith pre-exec command scanning** — security layer that analyzes commands before execution ([#1256](https://github.com/NousResearch/hermes-agent/pull/1256))
- Strip Hermes provider env vars from all subprocess environments ([#1157](https://github.com/NousResearch/hermes-agent/pull/1157), [#1172](https://github.com/NousResearch/hermes-agent/pull/1172), [#1399](https://github.com/NousResearch/hermes-agent/pull/1399), [#1419](https://github.com/NousResearch/hermes-agent/pull/1419)) — initial fix by @eren-karakus0
- SSH preflight check ([#1486](https://github.com/NousResearch/hermes-agent/pull/1486))
- Docker backend: make cwd workspace mount explicit opt-in ([#1534](https://github.com/NousResearch/hermes-agent/pull/1534))
- Add project root to PYTHONPATH in execute_code sandbox ([#1383](https://github.com/NousResearch/hermes-agent/pull/1383))
- Eliminate execute_code progress spam on gateway platforms ([#1098](https://github.com/NousResearch/hermes-agent/pull/1098))
- Clearer docker backend preflight errors ([#1276](https://github.com/NousResearch/hermes-agent/pull/1276))

### Browser
- **`/browser connect`** — attach browser tools to a live Chrome instance via CDP ([#1549](https://github.com/NousResearch/hermes-agent/pull/1549))
- Improve browser cleanup, local browser PATH setup, and screenshot recovery ([#1333](https://github.com/NousResearch/hermes-agent/pull/1333))

### MCP
- **Selective tool loading** with utility policies — filter which MCP tools are available ([#1302](https://github.com/NousResearch/hermes-agent/pull/1302))
- Auto-reload MCP tools when `mcp_servers` config changes without restart ([#1474](https://github.com/NousResearch/hermes-agent/pull/1474))
- Resolve npx stdio connection failures ([#1291](https://github.com/NousResearch/hermes-agent/pull/1291))
- Preserve MCP toolsets when saving platform tool config ([#1421](https://github.com/NousResearch/hermes-agent/pull/1421))

### Vision
- Unify vision backend gating ([#1367](https://github.com/NousResearch/hermes-agent/pull/1367))
- Surface actual error reason instead of generic message ([#1338](https://github.com/NousResearch/hermes-agent/pull/1338))
- Make Claude image handling work end-to-end ([#1408](https://github.com/NousResearch/hermes-agent/pull/1408))

### Cron
- **Compress cron management into one tool** — single `cronjob` tool replaces multiple commands ([#1343](https://github.com/NousResearch/hermes-agent/pull/1343))
- Suppress duplicate cron sends to auto-delivery targets ([#1357](https://github.com/NousResearch/hermes-agent/pull/1357))
- Persist cron sessions to SQLite ([#1255](https://github.com/NousResearch/hermes-agent/pull/1255))
- Per-job runtime overrides (provider, model, base_url) ([#1398](https://github.com/NousResearch/hermes-agent/pull/1398))
- Atomic write in `save_job_output` to prevent data loss on crash ([#1173](https://github.com/NousResearch/hermes-agent/pull/1173))
- Preserve thread context for `deliver=origin` ([#1437](https://github.com/NousResearch/hermes-agent/pull/1437))

### Patch Tool
- Avoid corrupting pipe chars in V4A patch apply ([#1286](https://github.com/NousResearch/hermes-agent/pull/1286))
- Permissive `block_anchor` thresholds and unicode normalization ([#1539](https://github.com/NousResearch/hermes-agent/pull/1539))

### Delegation
- Add observability metadata to subagent results (model, tokens, duration, tool trace) ([#1175](https://github.com/NousResearch/hermes-agent/pull/1175))

---

## 🧩 Skills Ecosystem

### Skills System
- **Integrate skills.sh** as a hub source alongside ClawHub ([#1303](https://github.com/NousResearch/hermes-agent/pull/1303))
- Secure skill env setup on load ([#1153](https://github.com/NousResearch/hermes-agent/pull/1153))
- Honor policy table for dangerous verdicts ([#1330](https://github.com/NousResearch/hermes-agent/pull/1330))
- Harden ClawHub skill search exact matches ([#1400](https://github.com/NousResearch/hermes-agent/pull/1400))
- Fix ClawHub skill install — use `/download` ZIP endpoint ([#1060](https://github.com/NousResearch/hermes-agent/pull/1060))
- Avoid mislabeling local skills as builtin — by @arceus77-7 ([#862](https://github.com/NousResearch/hermes-agent/pull/862))

### New Skills
- **Linear** project management ([#1230](https://github.com/NousResearch/hermes-agent/pull/1230))
- **X/Twitter** via x-cli ([#1285](https://github.com/NousResearch/hermes-agent/pull/1285))
- **Telephony** — Twilio, SMS, and AI calls ([#1289](https://github.com/NousResearch/hermes-agent/pull/1289))
- **1Password** — by @arceus77-7 ([#883](https://github.com/NousResearch/hermes-agent/pull/883), [#1179](https://github.com/NousResearch/hermes-agent/pull/1179))
- **NeuroSkill BCI** integration ([#1135](https://github.com/NousResearch/hermes-agent/pull/1135))
- **Blender MCP** for 3D modeling ([#1531](https://github.com/NousResearch/hermes-agent/pull/1531))
- **OSS Security Forensics** ([#1482](https://github.com/NousResearch/hermes-agent/pull/1482))
- **Parallel CLI** research skill ([#1301](https://github.com/NousResearch/hermes-agent/pull/1301))
- **OpenCode** CLI skill ([#1174](https://github.com/NousResearch/hermes-agent/pull/1174))
- **ASCII Video** skill refactored — by @SHL0MS ([#1213](https://github.com/NousResearch/hermes-agent/pull/1213), [#1598](https://github.com/NousResearch/hermes-agent/pull/1598))

---

## 🎙️ Voice Mode

- Voice mode foundation — push-to-talk CLI, Telegram/Discord voice notes ([#1299](https://github.com/NousResearch/hermes-agent/pull/1299))
- Free local Whisper transcription via faster-whisper ([#1185](https://github.com/NousResearch/hermes-agent/pull/1185))
- Discord voice channel reliability fixes ([#1429](https://github.com/NousResearch/hermes-agent/pull/1429))
- Restore local STT fallback for gateway voice notes ([#1490](https://github.com/NousResearch/hermes-agent/pull/1490))
- Honor `stt.enabled: false` across gateway transcription ([#1394](https://github.com/NousResearch/hermes-agent/pull/1394))
- Fix bogus incapability message on Telegram voice notes (Issue [#1033](https://github.com/NousResearch/hermes-agent/issues/1033))

---

## 🔌 ACP (IDE Integration)

- Restore ACP server implementation ([#1254](https://github.com/NousResearch/hermes-agent/pull/1254))
- Support slash commands in ACP adapter ([#1532](https://github.com/NousResearch/hermes-agent/pull/1532))

---

## 🧪 RL Training

- **Agentic On-Policy Distillation (OPD)** environment — new RL training environment for agent policy distillation ([#1149](https://github.com/NousResearch/hermes-agent/pull/1149))
- Make tinker-atropos RL training fully optional ([#1062](https://github.com/NousResearch/hermes-agent/pull/1062))

---

## 🔒 Security & Reliability

### Security Hardening
- **Tirith pre-exec command scanning** — static analysis of terminal commands before execution ([#1256](https://github.com/NousResearch/hermes-agent/pull/1256))
- **PII redaction** when `privacy.redact_pii` is enabled ([#1542](https://github.com/NousResearch/hermes-agent/pull/1542))
- Strip Hermes provider/gateway/tool env vars from all subprocess environments ([#1157](https://github.com/NousResearch/hermes-agent/pull/1157), [#1172](https://github.com/NousResearch/hermes-agent/pull/1172), [#1399](https://github.com/NousResearch/hermes-agent/pull/1399), [#1419](https://github.com/NousResearch/hermes-agent/pull/1419))
- Docker cwd workspace mount now explicit opt-in — never auto-mount host directories ([#1534](https://github.com/NousResearch/hermes-agent/pull/1534))
- Escape parens and braces in fork bomb regex pattern ([#1397](https://github.com/NousResearch/hermes-agent/pull/1397))
- Harden `.worktreeinclude` path containment ([#1388](https://github.com/NousResearch/hermes-agent/pull/1388))
- Use description as `pattern_key` to prevent approval collisions ([#1395](https://github.com/NousResearch/hermes-agent/pull/1395))

### Reliability
- Guard init-time stdio writes ([#1271](https://github.com/NousResearch/hermes-agent/pull/1271))
- Session log writes reuse shared atomic JSON helper ([#1280](https://github.com/NousResearch/hermes-agent/pull/1280))
- Atomic temp cleanup protected on interrupts ([#1401](https://github.com/NousResearch/hermes-agent/pull/1401))

---

## 🐛 Notable Bug Fixes

- **`/status` always showing 0 tokens** — now reports live state (Issue [#1465](https://github.com/NousResearch/hermes-agent/issues/1465), [#1476](https://github.com/NousResearch/hermes-agent/pull/1476))
- **Custom model endpoints not working** — restored config-saved endpoint resolution (Issue [#1460](https://github.com/NousResearch/hermes-agent/issues/1460), [#1373](https://github.com/NousResearch/hermes-agent/pull/1373))
- **MCP tools not visible until restart** — auto-reload on config change (Issue [#1036](https://github.com/NousResearch/hermes-agent/issues/1036), [#1474](https://github.com/NousResearch/hermes-agent/pull/1474))
- **`hermes tools` removing MCP tools** — preserve MCP toolsets when saving (Issue [#1247](https://github.com/NousResearch/hermes-agent/issues/1247), [#1421](https://github.com/NousResearch/hermes-agent/pull/1421))
- **Terminal subprocesses inheriting `OPENAI_BASE_URL`** breaking external tools (Issue [#1002](https://github.com/NousResearch/hermes-agent/issues/1002), [#1399](https://github.com/NousResearch/hermes-agent/pull/1399))
- **Background process lost on gateway restart** — improved recovery (Issue [#1144](https://github.com/NousResearch/hermes-agent/issues/1144))
- **Cron jobs not persisting state** — now stored in SQLite (Issue [#1416](https://github.com/NousResearch/hermes-agent/issues/1416), [#1255](https://github.com/NousResearch/hermes-agent/pull/1255))
- **Cronjob `deliver: origin` not preserving thread context** (Issue [#1219](https://github.com/NousResearch/hermes-agent/issues/1219), [#1437](https://github.com/NousResearch/hermes-agent/pull/1437))
- **Gateway systemd service failing to auto-restart** when browser processes orphaned (Issue [#1617](https://github.com/NousResearch/hermes-agent/issues/1617))
- **`/background` completion report cut off in Telegram** (Issue [#1443](https://github.com/NousResearch/hermes-agent/issues/1443))
- **Model switching not taking effect** (Issue [#1244](https://github.com/NousResearch/hermes-agent/issues/1244), [#1183](https://github.com/NousResearch/hermes-agent/pull/1183))
- **`hermes doctor` reporting cronjob as unavailable** (Issue [#878](https://github.com/NousResearch/hermes-agent/issues/878), [#1180](https://github.com/NousResearch/hermes-agent/pull/1180))
- **WhatsApp bridge messages not received** from mobile (Issue [#1142](https://github.com/NousResearch/hermes-agent/issues/1142))
- **Setup wizard hanging on headless SSH** (Issue [#905](https://github.com/NousResearch/hermes-agent/issues/905), [#1274](https://github.com/NousResearch/hermes-agent/pull/1274))
- **Log handler accumulation** degrading gateway performance (Issue [#990](https://github.com/NousResearch/hermes-agent/issues/990), [#1251](https://github.com/NousResearch/hermes-agent/pull/1251))
- **Gateway NULL model in DB** (Issue [#987](https://github.com/NousResearch/hermes-agent/issues/987), [#1306](https://github.com/NousResearch/hermes-agent/pull/1306))
- **Strict endpoints rejecting replayed tool_calls** (Issue [#893](https://github.com/NousResearch/hermes-agent/issues/893))
- **Remaining hardcoded `~/.hermes` paths** — all now respect `HERMES_HOME` (Issue [#892](https://github.com/NousResearch/hermes-agent/issues/892), [#1233](https://github.com/NousResearch/hermes-agent/pull/1233))
- **Delegate tool not working with custom inference providers** (Issue [#1011](https://github.com/NousResearch/hermes-agent/issues/1011), [#1328](https://github.com/NousResearch/hermes-agent/pull/1328))
- **Skills Guard blocking official skills** (Issue [#1006](https://github.com/NousResearch/hermes-agent/issues/1006), [#1330](https://github.com/NousResearch/hermes-agent/pull/1330))
- **Setup writing provider before model selection** (Issue [#1182](https://github.com/NousResearch/hermes-agent/issues/1182))
- **`GatewayConfig.get()` AttributeError** crashing all message handling (Issue [#1158](https://github.com/NousResearch/hermes-agent/issues/1158), [#1287](https://github.com/NousResearch/hermes-agent/pull/1287))
- **`/update` hard-failing with "command not found"** (Issue [#1049](https://github.com/NousResearch/hermes-agent/issues/1049))
- **Image analysis failing silently** (Issue [#1034](https://github.com/NousResearch/hermes-agent/issues/1034), [#1338](https://github.com/NousResearch/hermes-agent/pull/1338))
- **API `BadRequestError` from `'dict'` object has no attribute `'strip'`** (Issue [#1071](https://github.com/NousResearch/hermes-agent/issues/1071))
- **Slash commands requiring exact full name** — now uses prefix matching (Issue [#928](https://github.com/NousResearch/hermes-agent/issues/928), [#1320](https://github.com/NousResearch/hermes-agent/pull/1320))
- **Gateway stops responding when terminal is closed on headless** (Issue [#1005](https://github.com/NousResearch/hermes-agent/issues/1005))

---

## 🧪 Testing

- Cover empty cached Anthropic tool-call turns ([#1222](https://github.com/NousResearch/hermes-agent/pull/1222))
- Fix stale CI assumptions in parser and quick-command coverage ([#1236](https://github.com/NousResearch/hermes-agent/pull/1236))
- Fix gateway async tests without implicit event loop ([#1278](https://github.com/NousResearch/hermes-agent/pull/1278))
- Make gateway async tests xdist-safe ([#1281](https://github.com/NousResearch/hermes-agent/pull/1281))
- Cross-timezone naive timestamp regression for cron ([#1319](https://github.com/NousResearch/hermes-agent/pull/1319))
- Isolate codex provider tests from local env ([#1335](https://github.com/NousResearch/hermes-agent/pull/1335))
- Lock retry replacement semantics ([#1379](https://github.com/NousResearch/hermes-agent/pull/1379))
- Improve error logging in session search tool — by @aydnOktay ([#1533](https://github.com/NousResearch/hermes-agent/pull/1533))

---

## 📚 Documentation

- Comprehensive SOUL.md guide ([#1315](https://github.com/NousResearch/hermes-agent/pull/1315))
- Voice mode documentation ([#1316](https://github.com/NousResearch/hermes-agent/pull/1316), [#1362](https://github.com/NousResearch/hermes-agent/pull/1362))
- Provider contribution guide ([#1361](https://github.com/NousResearch/hermes-agent/pull/1361))
- ACP and internal systems implementation guides ([#1259](https://github.com/NousResearch/hermes-agent/pull/1259))
- Expand Docusaurus coverage across CLI, tools, skills, and skins ([#1232](https://github.com/NousResearch/hermes-agent/pull/1232))
- Terminal backend and Windows troubleshooting ([#1297](https://github.com/NousResearch/hermes-agent/pull/1297))
- Skills hub reference section ([#1317](https://github.com/NousResearch/hermes-agent/pull/1317))
- Checkpoint, /rollback, and git worktrees guide ([#1493](https://github.com/NousResearch/hermes-agent/pull/1493), [#1524](https://github.com/NousResearch/hermes-agent/pull/1524))
- CLI status bar and /usage reference ([#1523](https://github.com/NousResearch/hermes-agent/pull/1523))
- Fallback providers + /background command docs ([#1430](https://github.com/NousResearch/hermes-agent/pull/1430))
- Gateway service scopes docs ([#1378](https://github.com/NousResearch/hermes-agent/pull/1378))
- Slack thread reply behavior docs ([#1407](https://github.com/NousResearch/hermes-agent/pull/1407))
- Redesigned landing page with Nous blue palette — by @austinpickett ([#974](https://github.com/NousResearch/hermes-agent/pull/974))
- Fix several documentation typos — by @JackTheGit ([#953](https://github.com/NousResearch/hermes-agent/pull/953))
- Stabilize website diagrams ([#1405](https://github.com/NousResearch/hermes-agent/pull/1405))
- CLI vs messaging quick reference in README ([#1491](https://github.com/NousResearch/hermes-agent/pull/1491))
- Add search to Docusaurus ([#1053](https://github.com/NousResearch/hermes-agent/pull/1053))
- Home Assistant integration docs ([#1170](https://github.com/NousResearch/hermes-agent/pull/1170))

---

## 👥 Contributors

### Core
- **@teknium1** — 220+ PRs spanning every area of the codebase

### Top Community Contributors

- **@0xbyt4** (4 PRs) — Anthropic adapter fixes (max_tokens, fallback crash, 429/529 retry), Slack file upload thread context, setup NameError fix
- **@erosika** (1 PR) — Honcho memory integration: async writes, memory modes, session title integration
- **@SHL0MS** (2 PRs) — ASCII video skill design patterns and refactoring
- **@alt-glitch** (2 PRs) — Persistent shell mode for local/SSH backends, setuptools packaging fix
- **@arceus77-7** (2 PRs) — 1Password skill, fix skills list mislabeling
- **@kshitijk4poor** (1 PR) — OpenClaw migration during setup wizard
- **@ASRagab** (1 PR) — Fix adaptive thinking for Claude 4.6 models
- **@eren-karakus0** (1 PR) — Strip Hermes provider env vars from subprocess environment
- **@mr-emmett-one** (1 PR) — Fix DeepSeek V3 parser multi-tool call support
- **@jplew** (1 PR) — Gateway restart on retryable startup failures
- **@brandtcormorant** (1 PR) — Fix Anthropic cache control for empty text blocks
- **@aydnOktay** (1 PR) — Improve error logging in session search tool
- **@austinpickett** (1 PR) — Landing page redesign with Nous blue palette
- **@JackTheGit** (1 PR) — Documentation typo fixes

### All Contributors

@0xbyt4, @alt-glitch, @arceus77-7, @ASRagab, @austinpickett, @aydnOktay, @brandtcormorant, @eren-karakus0, @erosika, @JackTheGit, @jplew, @kshitijk4poor, @mr-emmett-one, @SHL0MS, @teknium1

---

**Full Changelog**: [v2026.3.12...v2026.3.17](https://github.com/NousResearch/hermes-agent/compare/v2026.3.12...v2026.3.17)
