# Hermes Agent v0.13.0 (v2026.5.7)

**Release Date:** May 7, 2026
**Since v0.12.0:** 864 commits · 588 merged PRs · 829 files changed · 128,366 insertions · 282 issues closed (13 P0, 36 P1) · 295 community contributors (including co-authors)

> The Tenacity Release — Hermes Agent now finishes what it starts. Kanban ships as a durable multi-agent board (heartbeat, reclaim, zombie detection, auto-block on incomplete exit, per-task retries, hallucination recovery). `/goal` keeps the agent locked on a target across turns (Ralph loop). Checkpoints v2 rewrites state persistence with real pruning. Gateway auto-resumes interrupted sessions after restart. Cron grows a `no_agent` watchdog mode. A security wave closes 8 P0s — redaction is now ON by default, Discord role-allowlists are guild-scoped, WhatsApp rejects strangers by default, and TOCTOU windows close across auth.json and MCP OAuth. Google Chat becomes the 20th platform. Providers become a pluggable surface. Seven i18n locales ship.

---

## ✨ Highlights

- **Multi-agent Kanban — delegate to an AI team that actually finishes** — Spin up a durable board, drop tasks on it, and let multiple Hermes workers pick them up, hand off, and close them out. Heartbeats, reclaim, zombie detection, retry budgets, and a hallucination gate keep the team honest. One install, many kanbans. ([#17805](https://github.com/NousResearch/hermes-agent/pull/17805), [#19653](https://github.com/NousResearch/hermes-agent/pull/19653), [#20232](https://github.com/NousResearch/hermes-agent/pull/20232), [#20332](https://github.com/NousResearch/hermes-agent/pull/20332), [#21330](https://github.com/NousResearch/hermes-agent/pull/21330), [#21183](https://github.com/NousResearch/hermes-agent/pull/21183), [#21214](https://github.com/NousResearch/hermes-agent/pull/21214))

- **`/goal` — the agent doesn't forget what you asked it to do** — Lock the agent onto a target and it stays on task across turns. The Ralph loop as a first-class primitive. ([#18262](https://github.com/NousResearch/hermes-agent/pull/18262), [#18275](https://github.com/NousResearch/hermes-agent/pull/18275), [#21287](https://github.com/NousResearch/hermes-agent/pull/21287))

- **Show it a video** — new `video_analyze` tool for native video understanding on Gemini and compatible multimodal models. (@alt-glitch) ([#19301](https://github.com/NousResearch/hermes-agent/pull/19301))

- **Clone a voice** — xAI Custom Voices lands as a TTS provider with voice cloning support. (@alt-glitch) ([#18776](https://github.com/NousResearch/hermes-agent/pull/18776))

- **Hermes speaks your language** — static gateway + CLI messages translate to 7 locales: Chinese, Japanese, German, Spanish, French, Ukrainian, and Turkish. Docs site gains a Chinese (zh-Hans) locale. ([#20231](https://github.com/NousResearch/hermes-agent/pull/20231), [#20329](https://github.com/NousResearch/hermes-agent/pull/20329), [#20467](https://github.com/NousResearch/hermes-agent/pull/20467), [#20474](https://github.com/NousResearch/hermes-agent/pull/20474), [#20430](https://github.com/NousResearch/hermes-agent/pull/20430), [#20431](https://github.com/NousResearch/hermes-agent/pull/20431))

- **Google Chat — the 20th messaging platform** — plus a generic platform-plugin hooks surface so third-party adapters drop in without touching core (IRC and Teams migrated). ([#21306](https://github.com/NousResearch/hermes-agent/pull/21306), [#21331](https://github.com/NousResearch/hermes-agent/pull/21331))

- **Sessions survive restarts** — gateway bounces mid-agent, `/update` restarts, source-file reloads — conversations auto-resume when the gateway comes back. ([#21192](https://github.com/NousResearch/hermes-agent/pull/21192))

- **Security wave — 8 P0 closures** — redaction ON by default, Discord role-allowlists guild-scoped (CVSS 8.1 cross-guild DM bypass closed), WhatsApp rejects strangers by default, TOCTOU windows closed across `auth.json` and MCP OAuth, browser enforces cloud-metadata SSRF floor, cron prompt-injection scans assembled skill content, `hermes debug share` redacts at upload. ([#21193](https://github.com/NousResearch/hermes-agent/pull/21193), [#21241](https://github.com/NousResearch/hermes-agent/pull/21241), [#21291](https://github.com/NousResearch/hermes-agent/pull/21291), [#21176](https://github.com/NousResearch/hermes-agent/pull/21176), [#21194](https://github.com/NousResearch/hermes-agent/pull/21194), [#21228](https://github.com/NousResearch/hermes-agent/pull/21228), [#21350](https://github.com/NousResearch/hermes-agent/pull/21350), [#19318](https://github.com/NousResearch/hermes-agent/pull/19318))

- **Checkpoints v2** — state persistence rewritten. Real pruning, disk guardrails, no more orphan shadow repos. ([#20709](https://github.com/NousResearch/hermes-agent/pull/20709))

- **The agent lints its own writes** — post-write delta lint on `write_file` + `patch`. Python, JSON, YAML, TOML. Syntax errors surface immediately instead of shipping downstream. ([#20191](https://github.com/NousResearch/hermes-agent/pull/20191))

- **`no_agent` cron mode — script-only watchdog** — cron jobs can now skip the agent entirely and just run a script. Empty stdout is silent, non-empty gets delivered verbatim. ([#19709](https://github.com/NousResearch/hermes-agent/pull/19709))

- **Platform allowlists everywhere** — `allowed_channels` / `allowed_chats` / `allowed_rooms` config across Slack, Telegram, Mattermost, Matrix, and DingTalk. ([#21251](https://github.com/NousResearch/hermes-agent/pull/21251))

- **Providers are now plugins** — `ProviderProfile` ABC + `plugins/model-providers/`. Drop in third-party providers without touching core. ([#20324](https://github.com/NousResearch/hermes-agent/pull/20324))

- **API server — long-term memory per session** — `X-Hermes-Session-Key` header gives memory providers a stable session identifier. ([#20199](https://github.com/NousResearch/hermes-agent/pull/20199))

- **MCP levels up** — SSE transport with OAuth forwarding, stale-pipe retries, image results surface as MEDIA tags instead of getting dropped, keepalive on long-lived lifecycle waits. ([#21227](https://github.com/NousResearch/hermes-agent/pull/21227), [#21323](https://github.com/NousResearch/hermes-agent/pull/21323), [#21289](https://github.com/NousResearch/hermes-agent/pull/21289), [#21328](https://github.com/NousResearch/hermes-agent/pull/21328), [#20209](https://github.com/NousResearch/hermes-agent/pull/20209))

- **Curator grows subcommands** — `hermes curator archive`, `prune`, `list-archived`. Manual `hermes curator run` is synchronous now — you see results without polling. ([#20200](https://github.com/NousResearch/hermes-agent/pull/20200), [#21236](https://github.com/NousResearch/hermes-agent/pull/21236), [#21216](https://github.com/NousResearch/hermes-agent/pull/21216))

- **ACP — `/steer` and `/queue`** — direct the in-flight agent or queue follow-ups from Zed, VS Code, or JetBrains. Plus atomic session persistence and reasoning-metadata preservation across restarts. (@HenkDz) ([#18114](https://github.com/NousResearch/hermes-agent/pull/18114), [#20279](https://github.com/NousResearch/hermes-agent/pull/20279), [#20296](https://github.com/NousResearch/hermes-agent/pull/20296), [#20433](https://github.com/NousResearch/hermes-agent/pull/20433))

- **TUI glow-up** — `/model` picker matches `hermes model` with inline auth (@austinpickett), collapsible startup banner sections (@kshitijk4poor), context-compression counter in the status bar. ([#18117](https://github.com/NousResearch/hermes-agent/pull/18117), [#20625](https://github.com/NousResearch/hermes-agent/pull/20625), [#21218](https://github.com/NousResearch/hermes-agent/pull/21218))

- **Dashboard grows up** — Plugins page (manage, enable/disable, auth status) (@austinpickett), Profiles management page (@vincez-hms-coder), sortable analytics tables, reverse-proxy support via `X-Forwarded-Prefix`, new `default-large` 18px theme. ([#18095](https://github.com/NousResearch/hermes-agent/pull/18095), [#16419](https://github.com/NousResearch/hermes-agent/pull/16419), [#18192](https://github.com/NousResearch/hermes-agent/pull/18192), [#21296](https://github.com/NousResearch/hermes-agent/pull/21296), [#20820](https://github.com/NousResearch/hermes-agent/pull/20820))

- **SearXNG + split web tools** — SearXNG ships as a native search-only backend; web tools now let you pick different backends per capability (search vs extract vs browse). (@kshitijk4poor) ([#20823](https://github.com/NousResearch/hermes-agent/pull/20823), [#20061](https://github.com/NousResearch/hermes-agent/pull/20061), [#20841](https://github.com/NousResearch/hermes-agent/pull/20841))

- **OpenRouter response caching** — explicit cache control for models that expose it. (@kshitijk4poor) ([#19132](https://github.com/NousResearch/hermes-agent/pull/19132))

- **`[[as_document]]` — skill media-routing directive** — skills can force the gateway to deliver output as a document on platforms that support it. ([#21210](https://github.com/NousResearch/hermes-agent/pull/21210))

- **`transform_llm_output` plugin hook** — new lifecycle hook that lets plugins reshape or filter LLM output before it hits the conversation. Useful for context-window reducers and content filters. ([#21235](https://github.com/NousResearch/hermes-agent/pull/21235))

- **Nous OAuth persists across profiles** — shared token store: sign in once, every profile inherits the session. ([#19712](https://github.com/NousResearch/hermes-agent/pull/19712))

- **QQBot — native approval keyboards** — feature parity with Telegram / Discord approval UX. Chunked upload, quoted attachments. ([#21342](https://github.com/NousResearch/hermes-agent/pull/21342), [#21353](https://github.com/NousResearch/hermes-agent/pull/21353))

- **6 new optional skills** — Shopify (Admin + Storefront GraphQL), here.now, shop-app personal shopping assistant, Anthropic financial-services bundle, kanban-video-orchestrator (@SHL0MS), searxng-search (@kshitijk4poor). ([#18116](https://github.com/NousResearch/hermes-agent/pull/18116), [#18170](https://github.com/NousResearch/hermes-agent/pull/18170), [#20702](https://github.com/NousResearch/hermes-agent/pull/20702), [#21180](https://github.com/NousResearch/hermes-agent/pull/21180), [#19281](https://github.com/NousResearch/hermes-agent/pull/19281), [#20841](https://github.com/NousResearch/hermes-agent/pull/20841))

- **New models** — `deepseek/deepseek-v4-pro`, `x-ai/grok-4.3`, `openrouter/owl-alpha` (free), `tencent/hy3-preview` (@Contentment003111), Arcee Trinity Large Thinking temperature + compression overrides. ([#20495](https://github.com/NousResearch/hermes-agent/pull/20495), [#20497](https://github.com/NousResearch/hermes-agent/pull/20497), [#18071](https://github.com/NousResearch/hermes-agent/pull/18071), [#21077](https://github.com/NousResearch/hermes-agent/pull/21077), [#20473](https://github.com/NousResearch/hermes-agent/pull/20473))

- **100 fresh CLI startup tips** — the random tip banner gets 100 new entries covering cron, kanban, curator, plugins, and lesser-known flags. ([#20168](https://github.com/NousResearch/hermes-agent/pull/20168))

---

## 🧩 Multi-Agent Kanban (Durable)

### New — durable multi-profile collaboration board
- **`feat(kanban): durable multi-profile collaboration board`** — post-revert reimplementation, multi-profile by design ([#17805](https://github.com/NousResearch/hermes-agent/pull/17805))
- **Multi-project boards** — one install, many kanbans ([#19653](https://github.com/NousResearch/hermes-agent/pull/19653), [#19679](https://github.com/NousResearch/hermes-agent/pull/19679))
- **Share board, workspaces, and worker logs across profiles** ([#19378](https://github.com/NousResearch/hermes-agent/pull/19378))
- **Hallucination gate + recovery UX for worker-created-card claims** (closes #20017) ([#20232](https://github.com/NousResearch/hermes-agent/pull/20232))
- **Generic diagnostics engine for task distress signals** ([#20332](https://github.com/NousResearch/hermes-agent/pull/20332))
- **Per-task `max_retries` override** (supersedes #20972) ([#21330](https://github.com/NousResearch/hermes-agent/pull/21330))
- **Multiline textarea for inline-create title** (salvage of #20970) ([#21243](https://github.com/NousResearch/hermes-agent/pull/21243))

### Kanban Dashboard
- **Workspace kind + path inputs in inline create form** ([#19679](https://github.com/NousResearch/hermes-agent/pull/19679))
- **Per-platform home-channel notification toggles** ([#19864](https://github.com/NousResearch/hermes-agent/pull/19864))
- **Sharper home-channel toggle contrast + drop → running action** ([#19916](https://github.com/NousResearch/hermes-agent/pull/19916))
- Fix: reject direct status transition to 'running' via dashboard API (salvage of #19554) ([#19705](https://github.com/NousResearch/hermes-agent/pull/19705))
- Fix: dashboard board pin authoritative over server current file (#20879) ([#21230](https://github.com/NousResearch/hermes-agent/pull/21230))
- Fix: treat dashboard event-stream cancellation as normal shutdown (#20790) ([#21222](https://github.com/NousResearch/hermes-agent/pull/21222))
- Fix: filter dashboard board by selected tenant (#19817) ([#21349](https://github.com/NousResearch/hermes-agent/pull/21349))
- Fix: code/pre styling theme-immune across all themes (#21086) ([#21247](https://github.com/NousResearch/hermes-agent/pull/21247))
- Fix: reset `<code>` background inside dashboard board ([#20687](https://github.com/NousResearch/hermes-agent/pull/20687))
- Fix: preserve dashboard completion summaries + add kanban edit (salvages #20016) ([#20195](https://github.com/NousResearch/hermes-agent/pull/20195))
- Fix: avoid fragile failure-column renames (salvage #20848) (@kshitijk4poor) ([#20855](https://github.com/NousResearch/hermes-agent/pull/20855))

### Worker lifecycle + reliability
- **Heartbeat + reclaim + zombie + retry-cap fixes** (#21147, #21141, #21169, #20881) ([#21183](https://github.com/NousResearch/hermes-agent/pull/21183))
- **Auto-block workers that exit without completing + shutdown race** (#20894) ([#21214](https://github.com/NousResearch/hermes-agent/pull/21214))
- **Detect darwin zombie workers** (salvages #20023) ([#20188](https://github.com/NousResearch/hermes-agent/pull/20188))
- **Unify failure counter across spawn/timeout/crash outcomes** ([#20410](https://github.com/NousResearch/hermes-agent/pull/20410))
- **Enforce worker task-ownership on destructive tool calls** ([#19713](https://github.com/NousResearch/hermes-agent/pull/19713))
- **Drop worker identity claim from KANBAN_GUIDANCE** ([#19427](https://github.com/NousResearch/hermes-agent/pull/19427))
- Fix: skip dispatch for tasks assigned to non-profile lanes (salvages #20105, #20134) ([#20165](https://github.com/NousResearch/hermes-agent/pull/20165))
- Fix: include default profile in on-disk assignee enumeration (salvages #20123) ([#20170](https://github.com/NousResearch/hermes-agent/pull/20170))
- Fix: ignore stale current board pointers (salvages #20063) ([#20183](https://github.com/NousResearch/hermes-agent/pull/20183))
- Fix: profile discovery ignores HERMES_HOME in custom-root deployments (@jackey8616) ([#19020](https://github.com/NousResearch/hermes-agent/pull/19020))
- Fix: allow orchestrator profiles to see kanban tools via toolsets config ([#19606](https://github.com/NousResearch/hermes-agent/pull/19606))

### Batch salvages
- Tier-1 batch — metadata test, max_spawn config, run-id lifecycle guard (salvages #19522 #19556 #19829) ([#20440](https://github.com/NousResearch/hermes-agent/pull/20440))
- Tier-2 batch — doctor, started_at, parent-guard, latest_summary, selects, linked-children ([#20448](https://github.com/NousResearch/hermes-agent/pull/20448))

### Documentation
- Backfill multi-board refs in reference docs ([#19704](https://github.com/NousResearch/hermes-agent/pull/19704))
- Document `/kanban` slash command ([#19584](https://github.com/NousResearch/hermes-agent/pull/19584))
- Document recommended handoff evidence metadata (salvage #19512) ([#20415](https://github.com/NousResearch/hermes-agent/pull/20415))
- Fix orchestrator + worker skill setup instructions (@helix4u) ([#20958](https://github.com/NousResearch/hermes-agent/pull/20958), [#20960](https://github.com/NousResearch/hermes-agent/pull/20960))

---

## 🎯 Persistent Goals, Checkpoints & Session Durability

### `/goal` — persistent cross-turn goals (Ralph loop)
- **`feat: /goal — persistent cross-turn goals`** ([#18262](https://github.com/NousResearch/hermes-agent/pull/18262))
- **Docs page — Persistent Goals (/goal)** ([#18275](https://github.com/NousResearch/hermes-agent/pull/18275))
- Fix: honor configured goal turn budget (salvage #19423) ([#21287](https://github.com/NousResearch/hermes-agent/pull/21287))

### Checkpoints v2
- **Single-store rewrite with real pruning + disk guardrails** ([#20709](https://github.com/NousResearch/hermes-agent/pull/20709))

### Session durability
- **Auto-resume interrupted sessions after gateway restart** (salvage #20888) ([#21192](https://github.com/NousResearch/hermes-agent/pull/21192))
- **Preserve pending update prompts across restarts** ([#20160](https://github.com/NousResearch/hermes-agent/pull/20160))
- **Preserve home-channel thread targets across restart notifications** (salvage #18440) ([#19271](https://github.com/NousResearch/hermes-agent/pull/19271))
- **Preserve thread routing from cached live session sources** ([#21206](https://github.com/NousResearch/hermes-agent/pull/21206))
- **Preserve assistant metadata when branching sessions** ([#18222](https://github.com/NousResearch/hermes-agent/pull/18222))
- **Preserve thread routing for /update progress and prompts** ([#18193](https://github.com/NousResearch/hermes-agent/pull/18193))
- **Preserve document type when merging queued events** ([#18215](https://github.com/NousResearch/hermes-agent/pull/18215))

---

## 🛡️ Security & Reliability

### Security hardening (8 P0 closures)
- **Enable secret redaction by default** (#17691, #20785) ([#21193](https://github.com/NousResearch/hermes-agent/pull/21193))
- **Discord — scope `DISCORD_ALLOWED_ROLES` to originating guild** (#12136, CVSS 8.1) ([#21241](https://github.com/NousResearch/hermes-agent/pull/21241))
- **WhatsApp — reject strangers by default, never respond in self-chat** (#8389) ([#21291](https://github.com/NousResearch/hermes-agent/pull/21291))
- **MCP OAuth — close TOCTOU window when saving credentials** ([#21176](https://github.com/NousResearch/hermes-agent/pull/21176))
- **`hermes_cli/auth.py` — close TOCTOU window in credential writers** ([#21194](https://github.com/NousResearch/hermes-agent/pull/21194))
- **Browser — enforce cloud-metadata SSRF floor in hybrid routing** (#16234) ([#21228](https://github.com/NousResearch/hermes-agent/pull/21228))
- **`hermes debug share` — redact log content at upload time** (@GodsBoy) ([#19318](https://github.com/NousResearch/hermes-agent/pull/19318))
- **Cron — scan assembled prompt including skill content for prompt injection** (#3968) ([#21350](https://github.com/NousResearch/hermes-agent/pull/21350))
- **Restore .env/auth.json/state.db with 0600 perms** ([#19699](https://github.com/NousResearch/hermes-agent/pull/19699))
- **SRI integrity for dashboard plugin scripts** (salvage #19389) ([#21277](https://github.com/NousResearch/hermes-agent/pull/21277))
- **Bind Meet node server to localhost, restrict token file to owner read** ([#19597](https://github.com/NousResearch/hermes-agent/pull/19597))
- **Extend sensitive-write target to cover shell RC and credential files** ([#19282](https://github.com/NousResearch/hermes-agent/pull/19282))
- **Harden YOLO mode env parsing against quoted-bool strings** ([#18214](https://github.com/NousResearch/hermes-agent/pull/18214))
- **OSV-Scanner CI + Dependabot for github-actions only** ([#20037](https://github.com/NousResearch/hermes-agent/pull/20037))

### Reliability — critical bug closures
- **CLI crash on startup — `Invalid key 'c-S-c'`** (P0, prompt_toolkit doesn't support Shift modifier) ([#19895](https://github.com/NousResearch/hermes-agent/pull/19895), [#19919](https://github.com/NousResearch/hermes-agent/pull/19919))
- **CLOSE_WAIT fd leak audit** — httpx keepalive + WhatsApp aiohttp leak + Feishu hygiene (#18451) ([#18766](https://github.com/NousResearch/hermes-agent/pull/18766))
- **Gateway creates AIAgent with empty OpenRouter API key when OPENROUTER_API_KEY is missing** (#20982) — fallback providers correctly honored
- **Background review + curator protected from overwriting bundled/hub skills** (#20273) ([#20194](https://github.com/NousResearch/hermes-agent/pull/20194))
- **TUI compression continuation — ghost sessions with incomplete metadata** (#20001)
- **`hermes mcp add` silently launches chat instead of registering MCP server** (#19785) ([#21204](https://github.com/NousResearch/hermes-agent/pull/21204))
- **Background review agent runtime propagation** — provider/model/credentials now actually inherit from parent
- **Inbound document host paths translated to container paths for Docker backend** (salvage #19048) ([#21184](https://github.com/NousResearch/hermes-agent/pull/21184))
- **Matrix gateway race between auto-redaction and message delivery with high-speed models** (#19075)
- **`/new` during active agent session never sends response on Telegram** (#18912)

---

## 📱 Messaging Platforms (Gateway)

### New platform
- **Google Chat — 20th platform** + generic `env_enablement_fn` / `cron_deliver_env_var` platform-plugin hooks (IRC + Teams migrated) ([#21306](https://github.com/NousResearch/hermes-agent/pull/21306), [#21331](https://github.com/NousResearch/hermes-agent/pull/21331))

### Cross-platform
- **`allowed_{channels,chats,rooms}` whitelist** — Slack (salvage #7401), Telegram, Mattermost, Matrix, DingTalk ([#21251](https://github.com/NousResearch/hermes-agent/pull/21251))
- **Per-platform `gateway_restart_notification` flag** ([#20892](https://github.com/NousResearch/hermes-agent/pull/20892))
- **`busy_ack_enabled` config — suppress ack messages** ([#18194](https://github.com/NousResearch/hermes-agent/pull/18194))
- **Auto-delete slash-command system notices after TTL** ([#18266](https://github.com/NousResearch/hermes-agent/pull/18266))
- **Opt-in cleanup of temporary progress bubbles** ([#21186](https://github.com/NousResearch/hermes-agent/pull/21186))
- **`[[as_document]]` directive — skill media routing** (salvage #19069) ([#21210](https://github.com/NousResearch/hermes-agent/pull/21210))
- **`hermes gateway list` — cross-profile status** (salvage #19129) ([#21225](https://github.com/NousResearch/hermes-agent/pull/21225))
- **Auto-resume interrupted sessions after restart** (salvage #20888) ([#21192](https://github.com/NousResearch/hermes-agent/pull/21192))
- **Atomic restart markers + Windows runtime-lock offset** (#17842) ([#18179](https://github.com/NousResearch/hermes-agent/pull/18179))
- Fix: `config.yaml` wins over `.env` for agent/display/timezone settings ([#18764](https://github.com/NousResearch/hermes-agent/pull/18764))
- Fix: auto-restart when source files change out from under us (#17648) ([#18409](https://github.com/NousResearch/hermes-agent/pull/18409))
- Fix: use git HEAD SHA for stale-code check, not file mtimes ([#19740](https://github.com/NousResearch/hermes-agent/pull/19740))
- Fix: shutdown + restart hygiene — drain timeout, false-fatal, success log ([#18761](https://github.com/NousResearch/hermes-agent/pull/18761))
- Fix: preserve max_turns after env reload (salvage #19183) ([#21240](https://github.com/NousResearch/hermes-agent/pull/21240))
- Fix: exclude ancestor PIDs from gateway process scan ([#19586](https://github.com/NousResearch/hermes-agent/pull/19586))
- Fix: move quick-command alias dispatch before built-ins ([#19588](https://github.com/NousResearch/hermes-agent/pull/19588))
- Fix: show other profiles in 'gateway status' to prevent confusion ([#19582](https://github.com/NousResearch/hermes-agent/pull/19582))
- Fix: include external_dirs skills in Telegram/Discord slash commands (salvage #8790) ([#18741](https://github.com/NousResearch/hermes-agent/pull/18741))
- Fix: match disabled/optional skills by frontmatter slug, not dir name ([#18753](https://github.com/NousResearch/hermes-agent/pull/18753))
- Fix: read /status token totals from SessionDB (#17158) ([#18206](https://github.com/NousResearch/hermes-agent/pull/18206))
- Fix: snapshot callback generation after agent binds it, not before ([#18219](https://github.com/NousResearch/hermes-agent/pull/18219))
- Fix: re-inject topic-bound skill after /new or /reset ([#18205](https://github.com/NousResearch/hermes-agent/pull/18205))
- Fix: isolate pending native image paths by session ([#18202](https://github.com/NousResearch/hermes-agent/pull/18202))
- Fix: clear queued reload skills notes on new/resume/branch ([#19431](https://github.com/NousResearch/hermes-agent/pull/19431))
- Fix: hide required-arg commands from Telegram menu ([#19400](https://github.com/NousResearch/hermes-agent/pull/19400))
- Fix: bridge top-level `require_mention` to Telegram config ([#19429](https://github.com/NousResearch/hermes-agent/pull/19429))
- Fix: suppress duplicate voice transcripts ([#19428](https://github.com/NousResearch/hermes-agent/pull/19428))
- Fix: show friendly error when service is not installed ([#19707](https://github.com/NousResearch/hermes-agent/pull/19707))
- Fix: read context_length from custom_providers in session info header ([#19708](https://github.com/NousResearch/hermes-agent/pull/19708))
- Fix: preserve WSL interop PATH in systemd units ([#19867](https://github.com/NousResearch/hermes-agent/pull/19867))
- Fix: handle planned service stops (salvage #19876) ([#19936](https://github.com/NousResearch/hermes-agent/pull/19936))
- Fix: keep DoH-confirmed Telegram IPs that match system DNS (salvage #17043) ([#20175](https://github.com/NousResearch/hermes-agent/pull/20175))
- Fix: load `reply_to_mode` from config.yaml for Discord + Telegram (salvage #17117) ([#20171](https://github.com/NousResearch/hermes-agent/pull/20171))
- Fix: tolerate malformed HERMES_HUMAN_DELAY_* env vars (salvage #16933) ([#20217](https://github.com/NousResearch/hermes-agent/pull/20217))
- Fix: deterministic thread eviction preserves newest entries (salvage #13639) ([#20285](https://github.com/NousResearch/hermes-agent/pull/20285))
- Fix: don't dead-end setup wizard when only system-scope unit is installed ([#20905](https://github.com/NousResearch/hermes-agent/pull/20905))
- Fix: wait for systemd restart readiness + harden Discord slash-command sync ([#20949](https://github.com/NousResearch/hermes-agent/pull/20949))
- Fix: avoid duplicated Responses history (salvage #18995) ([#21185](https://github.com/NousResearch/hermes-agent/pull/21185))
- Fix: surface bootstrap failures to stderr (salvage #21157) ([#21278](https://github.com/NousResearch/hermes-agent/pull/21278))
- Fix: log agent task failures instead of silently losing usage data (salvage #21159) ([#21274](https://github.com/NousResearch/hermes-agent/pull/21274))
- Fix: log runtime-status write failures with rate-limiting (salvage #21158) ([#21285](https://github.com/NousResearch/hermes-agent/pull/21285))
- Fix: reset-failed before every fallback restart so the gateway can't get stranded ([#21371](https://github.com/NousResearch/hermes-agent/pull/21371))
- Fix: Telegram — preserve `thread_id=1` for forum General typing indicator ([#21390](https://github.com/NousResearch/hermes-agent/pull/21390))
- Fix: batch critical fixes — session resume, /new race, HA WebSocket scheme (@kshitijk4poor) ([#19182](https://github.com/NousResearch/hermes-agent/pull/19182))

### Telegram
- **DM user-managed multi-session topics** (salvage of #19185) ([#19206](https://github.com/NousResearch/hermes-agent/pull/19206))

### Discord
- **Message deletion action** (salvage #19052) ([#21197](https://github.com/NousResearch/hermes-agent/pull/21197))
- Fix: allow `free_response_channels` to override `DISCORD_IGNORE_NO_MENTION` ([#19629](https://github.com/NousResearch/hermes-agent/pull/19629))

### Slack
- Fix: ephemeral slash-command ack, private notice delivery, format_message fixes (@kshitijk4poor) ([#18198](https://github.com/NousResearch/hermes-agent/pull/18198))

### WhatsApp
- Fix: load WhatsApp home channel from env overrides ([#18190](https://github.com/NousResearch/hermes-agent/pull/18190))

### Feishu
- **Operator-configurable bot admission and mention policy** ([#18208](https://github.com/NousResearch/hermes-agent/pull/18208))
- Fix: force text mode for markdown tables (salvage of #13723 by @WuTianyi123) ([#20275](https://github.com/NousResearch/hermes-agent/pull/20275))

### Matrix + Email
- Fix: `/sethome` on Matrix and Email now persists across restarts ([#18272](https://github.com/NousResearch/hermes-agent/pull/18272))

### Teams
- **Docs + feat: sidebar + threading with group-chat fallback** ([#20042](https://github.com/NousResearch/hermes-agent/pull/20042))

### Weixin
- Fix: deduplicate Weixin messages by content fingerprint ([#19742](https://github.com/NousResearch/hermes-agent/pull/19742))

### QQBot
- **Port SDK improvements in-tree — chunked upload, approval keyboards, quoted attachments** ([#21342](https://github.com/NousResearch/hermes-agent/pull/21342))
- **Wire native tool-approval UX via inline keyboards** ([#21353](https://github.com/NousResearch/hermes-agent/pull/21353))

---

## 🏗️ Core Agent & Architecture

### Provider & Model Support

#### Pluggable providers
- **ProviderProfile ABC + `plugins/model-providers/`** — inference providers are now a pluggable surface (salvage of #14424) ([#20324](https://github.com/NousResearch/hermes-agent/pull/20324))
- **`list_picker_providers`** — credential-filtered picker (salvage #13561) ([#20298](https://github.com/NousResearch/hermes-agent/pull/20298))
- **Remove `/provider` alias for `/model`** ([#20358](https://github.com/NousResearch/hermes-agent/pull/20358))
- **Shared Hermes dotenv loader across CLI + plugins** (salvage #13660) ([#20281](https://github.com/NousResearch/hermes-agent/pull/20281))
- **Nous OAuth persisted across profiles via shared token store** ([#19712](https://github.com/NousResearch/hermes-agent/pull/19712))

#### New models
- `deepseek/deepseek-v4-pro` added to OpenRouter + Nous Portal ([#20495](https://github.com/NousResearch/hermes-agent/pull/20495))
- `x-ai/grok-4.3` added to OpenRouter + Nous Portal ([#20497](https://github.com/NousResearch/hermes-agent/pull/20497))
- `openrouter/owl-alpha` (free tier) added to curated OpenRouter list ([#18071](https://github.com/NousResearch/hermes-agent/pull/18071))
- `tencent/hy3-preview` paid route on OpenRouter (@Contentment003111) ([#21077](https://github.com/NousResearch/hermes-agent/pull/21077))
- Arcee Trinity Large Thinking — temperature + compression overrides ([#20473](https://github.com/NousResearch/hermes-agent/pull/20473))
- Rename `x-ai/grok-4.20-beta` to `x-ai/grok-4.20` ([#19640](https://github.com/NousResearch/hermes-agent/pull/19640))
- Demote Vercel AI Gateway to bottom of provider picker ([#18112](https://github.com/NousResearch/hermes-agent/pull/18112))

#### Provider configuration
- **OpenRouter — response caching support** (@kshitijk4poor) ([#19132](https://github.com/NousResearch/hermes-agent/pull/19132))
- **`image_gen.model` from config.yaml honored** (salvage #19376) ([#21273](https://github.com/NousResearch/hermes-agent/pull/21273))
- Fix: honor runtime default model during delegate provider resolution (@johnncenae) ([#17587](https://github.com/NousResearch/hermes-agent/pull/17587))
- Fix: avoid Bedrock credential probe in provider picker (@helix4u) ([#18998](https://github.com/NousResearch/hermes-agent/pull/18998))
- Fix: drop stale env-var override of persisted provider for cron ([#19627](https://github.com/NousResearch/hermes-agent/pull/19627))
- Fix: auxiliary curator api_key/base_url into runtime resolution ([#19421](https://github.com/NousResearch/hermes-agent/pull/19421))

### Agent Loop & Conversation
- **`video_analyze` — native video understanding tool** (@alt-glitch) ([#19301](https://github.com/NousResearch/hermes-agent/pull/19301))
- **Show context compression count in status bar** (CLI + TUI) ([#21218](https://github.com/NousResearch/hermes-agent/pull/21218))
- **Isolate `get_tool_definitions` quiet_mode cache + dedup LCM injection** (#17335) ([#17889](https://github.com/NousResearch/hermes-agent/pull/17889))
- Fix: warning-first tool-call loop guardrails ([#18227](https://github.com/NousResearch/hermes-agent/pull/18227))
- Fix: break permanent empty-response loop from orphan tool-tail ([#21385](https://github.com/NousResearch/hermes-agent/pull/21385))
- Fix: propagate ContextVars to concurrent tool worker threads (salvage #16660) ([#18123](https://github.com/NousResearch/hermes-agent/pull/18123))
- Fix: surface self-improvement review summaries across CLI, TUI, and gateway ([#18073](https://github.com/NousResearch/hermes-agent/pull/18073))
- Fix: serialize concurrent `hermes_tools` RPC calls from `execute_code` ([#17894](https://github.com/NousResearch/hermes-agent/pull/17894), [#17902](https://github.com/NousResearch/hermes-agent/pull/17902))
- Fix: include system prompt + tool schemas in token estimates for compression ([#18265](https://github.com/NousResearch/hermes-agent/pull/18265))

### Compression
- Fix: skip non-string tool content in dedup pass to prevent AttributeError ([#19398](https://github.com/NousResearch/hermes-agent/pull/19398))
- Fix: reset `_summary_failure_cooldown_until` on session reset ([#19622](https://github.com/NousResearch/hermes-agent/pull/19622))
- Fix: trigger fallback on timeout errors alongside model-unavailable errors ([#19665](https://github.com/NousResearch/hermes-agent/pull/19665))
- Fix: `_prune_old_tool_results` boundary direction ([#19725](https://github.com/NousResearch/hermes-agent/pull/19725))
- Fix: soften summary prompt for content filters (salvage #19456) ([#21302](https://github.com/NousResearch/hermes-agent/pull/21302))

### Delegate
- Fix: inherit parent fallback_chain in `_build_child_agent` ([#19601](https://github.com/NousResearch/hermes-agent/pull/19601))
- Fix: guard `_load_config()` against `delegation: null` in config.yaml ([#19662](https://github.com/NousResearch/hermes-agent/pull/19662))
- Fix: inherit parent api_key when `delegation.base_url` set without `delegation.api_key` ([#19741](https://github.com/NousResearch/hermes-agent/pull/19741))
- Fix: expand composite toolsets before intersection (salvage #19455) ([#21300](https://github.com/NousResearch/hermes-agent/pull/21300))
- Fix: correct ACP docs — Claude Code CLI has no --acp flag (salvage #19058) ([#21201](https://github.com/NousResearch/hermes-agent/pull/21201))

### Session & Memory
- **Hindsight — probe API for `update_mode='append'` to dedupe across processes** (@nicoloboschi) ([#20222](https://github.com/NousResearch/hermes-agent/pull/20222))

### Curator
- **`hermes curator archive` and `prune` subcommands** ([#20200](https://github.com/NousResearch/hermes-agent/pull/20200))
- **`hermes curator list-archived`** (#20651) ([#21236](https://github.com/NousResearch/hermes-agent/pull/21236))
- **Synchronous manual `hermes curator run`** (#20555) ([#21216](https://github.com/NousResearch/hermes-agent/pull/21216))
- Fix: preserve `last_report_path` in state ([#18169](https://github.com/NousResearch/hermes-agent/pull/18169))
- Fix: rewrite cron job skill refs after consolidation ([#18253](https://github.com/NousResearch/hermes-agent/pull/18253))
- Fix: defer first run + `--dry-run` preview (#18373) ([#18389](https://github.com/NousResearch/hermes-agent/pull/18389))
- Fix: authoritative `absorbed_into` on delete + restore cron skill links on rollback (#18671) ([#18731](https://github.com/NousResearch/hermes-agent/pull/18731))
- Fix: prevent false-positive consolidation from substring matching ([#19573](https://github.com/NousResearch/hermes-agent/pull/19573))
- Fix: only mark agent-created for background-review sediment ([#19621](https://github.com/NousResearch/hermes-agent/pull/19621))
- Fix: protect hub skills by frontmatter name ([#20194](https://github.com/NousResearch/hermes-agent/pull/20194))

---

## 🔧 Tool System

### File tools
- **Post-write delta lint on `write_file` + `patch`** — in-proc linters for Python, JSON, YAML, TOML ([#20191](https://github.com/NousResearch/hermes-agent/pull/20191))

### Cron
- **`no_agent` mode — script-only cron jobs (watchdog pattern)** ([#19709](https://github.com/NousResearch/hermes-agent/pull/19709))
- **`context_from` chaining docs** (salvage #15724) ([#20394](https://github.com/NousResearch/hermes-agent/pull/20394))
- Fix: treat non-dict origin as missing instead of crashing tick ([#19283](https://github.com/NousResearch/hermes-agent/pull/19283))
- Fix: bump skill usage when cron jobs load skills ([#19433](https://github.com/NousResearch/hermes-agent/pull/19433))
- Fix: recover null `next_run_at` jobs ([#19576](https://github.com/NousResearch/hermes-agent/pull/19576))
- Fix: skip AI call when prerun script produces no output ([#19628](https://github.com/NousResearch/hermes-agent/pull/19628))
- Fix: expand config.yaml refs during job execution ([#19872](https://github.com/NousResearch/hermes-agent/pull/19872))
- Fix: serialize `get_due_jobs` writes to prevent parallel state corruption ([#19874](https://github.com/NousResearch/hermes-agent/pull/19874))
- Fix: initialize MCP servers before constructing the cron AIAgent ([#21354](https://github.com/NousResearch/hermes-agent/pull/21354))

### MCP
- **SSE transport support** (salvage #19135) ([#21227](https://github.com/NousResearch/hermes-agent/pull/21227))
- **Forward OAuth auth + bump `sse_read_timeout` on SSE transport** ([#21323](https://github.com/NousResearch/hermes-agent/pull/21323))
- **Retry stale pipe transport failures as session-expired** ([#21289](https://github.com/NousResearch/hermes-agent/pull/21289))
- **Surface image tool results as MEDIA tags instead of dropping them** ([#21328](https://github.com/NousResearch/hermes-agent/pull/21328))
- **Periodic keepalive to `_wait_for_lifecycle_event`** (salvage #17016) ([#20209](https://github.com/NousResearch/hermes-agent/pull/20209))
- Fix: reconnect on terminated sessions ([#19380](https://github.com/NousResearch/hermes-agent/pull/19380))
- Fix: decouple AnyUrl import from mcp dependency ([#19695](https://github.com/NousResearch/hermes-agent/pull/19695))
- Fix: `mcp add --command` gets distinct argparse dest ([#21204](https://github.com/NousResearch/hermes-agent/pull/21204))
- Fix: clear stale thread interrupt before MCP discovery ([#21276](https://github.com/NousResearch/hermes-agent/pull/21276))
- Fix: report configured timeout in MCP call errors ([#21281](https://github.com/NousResearch/hermes-agent/pull/21281))
- Fix: include exception type in error messages when str(exc) is empty (salvage #19425) ([#21292](https://github.com/NousResearch/hermes-agent/pull/21292))
- Fix: re-raise CancelledError explicitly in `MCPServerTask.run` ([#21318](https://github.com/NousResearch/hermes-agent/pull/21318))
- Fix: coerce numeric tool args defensively in `mcp_serve` ([#21329](https://github.com/NousResearch/hermes-agent/pull/21329))
- Fix: gate utility stubs on server-advertised capabilities ([#21347](https://github.com/NousResearch/hermes-agent/pull/21347))

### Browser
- Fix: allow explicit CDP override without local agent-browser ([#19670](https://github.com/NousResearch/hermes-agent/pull/19670))
- Fix: inject `--no-sandbox` for root + AppArmor userns restrictions ([#19747](https://github.com/NousResearch/hermes-agent/pull/19747))
- Fix: tighten Lightpanda fallback edge cases (@kshitijk4poor) ([#20672](https://github.com/NousResearch/hermes-agent/pull/20672))

### Web tools
- **Per-capability backend selection — search/extract split** (@kshitijk4poor) ([#20061](https://github.com/NousResearch/hermes-agent/pull/20061))
- **SearXNG native search-only backend** (@kshitijk4poor) ([#20823](https://github.com/NousResearch/hermes-agent/pull/20823))

### Approval / Tool gating
- Fix: wake blocked gateway approvals on session cleanup ([#18171](https://github.com/NousResearch/hermes-agent/pull/18171))
- Fix: harden YOLO mode env parsing against quoted-bool strings ([#18214](https://github.com/NousResearch/hermes-agent/pull/18214))
- Fix: extend sensitive write target to cover shell RC and credential files ([#19282](https://github.com/NousResearch/hermes-agent/pull/19282))

---

## 🔌 Plugin System

- **`transform_llm_output` plugin hook** (salvage of #20813) ([#21235](https://github.com/NousResearch/hermes-agent/pull/21235))
- **Document `env_enablement_fn` + `cron_deliver_env_var` platform-plugin hooks** ([#21331](https://github.com/NousResearch/hermes-agent/pull/21331))
- **Pluggable surfaces coverage — model-provider guide, full plugin map, opt-in fix** ([#20749](https://github.com/NousResearch/hermes-agent/pull/20749))
- **Plugin-authoring gaps — image-gen provider guide + publishing a skill tap** ([#20800](https://github.com/NousResearch/hermes-agent/pull/20800))

---

## 🧩 Skills Ecosystem

### New optional skills
- **Shopify** — Admin + Storefront GraphQL optional skill ([#18116](https://github.com/NousResearch/hermes-agent/pull/18116))
- **here.now** — optional skill ([#18170](https://github.com/NousResearch/hermes-agent/pull/18170))
- **shop-app** — personal shopping assistant (optional) ([#20702](https://github.com/NousResearch/hermes-agent/pull/20702))
- **Anthropic financial-services bundle** — ported as optional finance skills ([#21180](https://github.com/NousResearch/hermes-agent/pull/21180))
- **kanban-video-orchestrator** — creative optional skill (@SHL0MS) ([#19281](https://github.com/NousResearch/hermes-agent/pull/19281))
- **searxng-search** — optional skill + Web Search + Extract docs page (@kshitijk4poor) ([#20841](https://github.com/NousResearch/hermes-agent/pull/20841), [#20844](https://github.com/NousResearch/hermes-agent/pull/20844))

### Skill UX
- **Linear skill — add Documents support + Python helper script** ([#20752](https://github.com/NousResearch/hermes-agent/pull/20752))
- **Modernize Obsidian skill to use file tools** (salvage #19332) ([#20413](https://github.com/NousResearch/hermes-agent/pull/20413))
- **Default custom tool creation to plugins** (@kshitijk4poor) ([#19755](https://github.com/NousResearch/hermes-agent/pull/19755))
- **skill_commands cache — rescan on platform scope changes** (salvage #14570 by @LeonSGP43) ([#18739](https://github.com/NousResearch/hermes-agent/pull/18739))
- **Skills — additional rescan paths in skill_commands cache** (salvage #19042) ([#21181](https://github.com/NousResearch/hermes-agent/pull/21181))
- Fix: regression tests for non-dict metadata in `extract_skill_conditions` ([#18213](https://github.com/NousResearch/hermes-agent/pull/18213))
- Docs: explain restoring bundled skills (salvage #19254) ([#20404](https://github.com/NousResearch/hermes-agent/pull/20404))
- Docs: document `hermes skills reset` subcommand (salvage #11544) ([#20395](https://github.com/NousResearch/hermes-agent/pull/20395))
- Docs: himalaya v1.2.0 `folder.aliases` syntax ([#19882](https://github.com/NousResearch/hermes-agent/pull/19882))
- Point agent at `hermes-agent` skill + docs site sync ([#20390](https://github.com/NousResearch/hermes-agent/pull/20390))

---

## 🖥️ CLI & User Experience

### CLI
- **`/new` accepts optional session name argument** (salvage of #19555) ([#19637](https://github.com/NousResearch/hermes-agent/pull/19637))
- **100 new CLI startup tips** ([#20168](https://github.com/NousResearch/hermes-agent/pull/20168))
- **`display.language` — static message translation** (zh/ja/de/es) ([#20231](https://github.com/NousResearch/hermes-agent/pull/20231))
- **French (fr) locale** (@Foolafroos) ([#20329](https://github.com/NousResearch/hermes-agent/pull/20329))
- **Ukrainian (uk) locale** ([#20467](https://github.com/NousResearch/hermes-agent/pull/20467))
- **Turkish (tr) locale** ([#20474](https://github.com/NousResearch/hermes-agent/pull/20474))
- Fix: recover classic CLI output after resize (@helix4u) ([#20444](https://github.com/NousResearch/hermes-agent/pull/20444))
- Fix: complete absolute paths as paths (@helix4u) ([#19930](https://github.com/NousResearch/hermes-agent/pull/19930))
- Fix: resolve lazy session creation regressions (#18370 fallout) (@alt-glitch) ([#20363](https://github.com/NousResearch/hermes-agent/pull/20363))
- Fix: local backend CLI always uses launch directory (@alt-glitch) ([#19334](https://github.com/NousResearch/hermes-agent/pull/19334))
- Refactor: drop dead c-S-c key binding (follow-up to #19895) ([#19919](https://github.com/NousResearch/hermes-agent/pull/19919))

### TUI (Ink)
- **`/model` picker overhaul to match `hermes model` with inline auth** (@austinpickett) ([#18117](https://github.com/NousResearch/hermes-agent/pull/18117))
- **Collapsible sections in startup banner** — skills, system prompt, MCP (@kshitijk4poor) ([#20625](https://github.com/NousResearch/hermes-agent/pull/20625))
- **Show context compression count in status bar** ([#21218](https://github.com/NousResearch/hermes-agent/pull/21218))
- Perf: reduce overlay render churn with focused selectors (@OutThisLife) ([#20393](https://github.com/NousResearch/hermes-agent/pull/20393))
- Fix: restore voice push-to-talk parity (salvage of #16189 by @Montbra) (@OutThisLife) ([#20897](https://github.com/NousResearch/hermes-agent/pull/20897))
- Fix: kanban button (@austinpickett) ([#18358](https://github.com/NousResearch/hermes-agent/pull/18358))

### Dashboard
- **Plugins page — manage, enable/disable, auth status** (@austinpickett) ([#18095](https://github.com/NousResearch/hermes-agent/pull/18095))
- **Profiles management page** (@vincez-hms-coder) ([#16419](https://github.com/NousResearch/hermes-agent/pull/16419))
- **Interactive column sorting in analytics tables** ([#18192](https://github.com/NousResearch/hermes-agent/pull/18192))
- **`default-large` built-in theme with 18px base size** ([#20820](https://github.com/NousResearch/hermes-agent/pull/20820))
- **Support serving under URL prefix via `X-Forwarded-Prefix`** (salvage #19450) ([#21296](https://github.com/NousResearch/hermes-agent/pull/21296))
- **Launch dashboard as side-process via `HERMES_DASHBOARD=1` in Docker** (@benbarclay) ([#19540](https://github.com/NousResearch/hermes-agent/pull/19540))
- Fix: dashboard theme layout shift (@AllardQuek) ([#17232](https://github.com/NousResearch/hermes-agent/pull/17232))
- Fix: gateway model picker current context (@helix4u) ([#20513](https://github.com/NousResearch/hermes-agent/pull/20513))

### Update + setup
- **`hermes update --yes/-y` to skip interactive prompts** ([#18261](https://github.com/NousResearch/hermes-agent/pull/18261))
- **Restart manual profile gateways after update** ([#18178](https://github.com/NousResearch/hermes-agent/pull/18178))

### Profiles
- **`--no-skills` flag for empty profile creation** ([#20986](https://github.com/NousResearch/hermes-agent/pull/20986))

---

## 🎵 Voice, Image & Media

- **xAI Custom Voices — voice cloning** (@alt-glitch) ([#18776](https://github.com/NousResearch/hermes-agent/pull/18776))
- **Achievements — share card render on unlocked badges** ([#19657](https://github.com/NousResearch/hermes-agent/pull/19657))
- **Refresh systemd unit on gateway boot (not just start/restart)** (@alt-glitch) ([#19684](https://github.com/NousResearch/hermes-agent/pull/19684))

---

## 🔗 API Server & Remote Access

- **`X-Hermes-Session-Key` header for long-term memory scoping** (closes #20060) ([#20199](https://github.com/NousResearch/hermes-agent/pull/20199))

---

## 🧰 ACP Adapter (VS Code / Zed / JetBrains)

- **`/steer` and `/queue` slash commands** (@HenkDz) ([#18114](https://github.com/NousResearch/hermes-agent/pull/18114))
- Fix: translate Windows cwd for WSL sessions (salvage #18128) ([#18233](https://github.com/NousResearch/hermes-agent/pull/18233))
- Fix: run `/steer` as a regular prompt on idle sessions ([#18258](https://github.com/NousResearch/hermes-agent/pull/18258))
- Fix: route Zed thoughts to reasoning + polish tool/context rendering ([#19139](https://github.com/NousResearch/hermes-agent/pull/19139))
- Fix: atomic session persistence via `replace_messages` (salvage #13675) ([#20279](https://github.com/NousResearch/hermes-agent/pull/20279))
- Fix: preserve assistant reasoning metadata in session persistence (salvage #13575) ([#20296](https://github.com/NousResearch/hermes-agent/pull/20296))
- Docs: update VS Code setup for ACP Client extension (salvage #12495) ([#20433](https://github.com/NousResearch/hermes-agent/pull/20433))

---

## 🐳 Docker

- **Launch dashboard as side-process via `HERMES_DASHBOARD=1`** (@benbarclay) ([#19540](https://github.com/NousResearch/hermes-agent/pull/19540))
- **Refuse root gateway runs in official image** (salvage #19215) ([#21250](https://github.com/NousResearch/hermes-agent/pull/21250))
- **Chown runtime `node_modules` trees to hermes user** (salvage #19303) ([#21267](https://github.com/NousResearch/hermes-agent/pull/21267))
- Fix: exclude compose/profile runtime state from build context ([#19626](https://github.com/NousResearch/hermes-agent/pull/19626))
- CI: don't cancel overlapping builds, guard `:latest` (@ethernet8023) ([#20890](https://github.com/NousResearch/hermes-agent/pull/20890))
- Test: align Dockerfile contract tests with simplified TUI flow (salvage #19024) ([#21174](https://github.com/NousResearch/hermes-agent/pull/21174))
- Docs: connect to local inference servers (vLLM, Ollama) (salvage #12335) ([#20407](https://github.com/NousResearch/hermes-agent/pull/20407))
- Docs: document `API_SERVER_*` env vars (salvage #11758) ([#20409](https://github.com/NousResearch/hermes-agent/pull/20409))
- Docs: clarify Docker terminal backend is a single persistent container ([#20003](https://github.com/NousResearch/hermes-agent/pull/20003))

---

## 🐛 Notable Bug Fixes

### Agent
- Fix: recover lazy session creation regressions (#18370 fallout) (@alt-glitch) ([#20363](https://github.com/NousResearch/hermes-agent/pull/20363))
- Fix: propagate ContextVars to concurrent tool worker threads (salvage #16660) ([#18123](https://github.com/NousResearch/hermes-agent/pull/18123))
- Fix: warning-first tool-call loop guardrails ([#18227](https://github.com/NousResearch/hermes-agent/pull/18227))
- Fix: surface self-improvement review summaries across CLI, TUI, and gateway ([#18073](https://github.com/NousResearch/hermes-agent/pull/18073))

### Gateway streaming
- Fix: harden StreamingConfig bool and numeric coercion (@simbam99) ([#16463](https://github.com/NousResearch/hermes-agent/pull/16463))

### Model
- Fix: avoid Bedrock credential probe in provider picker (@helix4u) ([#18998](https://github.com/NousResearch/hermes-agent/pull/18998))

### Doctor
- Fix: check global agent-browser when local install not found ([#19671](https://github.com/NousResearch/hermes-agent/pull/19671))
- Test: kimi-coding-cn provider validation regression ([#19734](https://github.com/NousResearch/hermes-agent/pull/19734))

### Update
- Fix: patch `isatty` on real streams to fix xdist-flaky `--yes` tests (salvage #19026) ([#21175](https://github.com/NousResearch/hermes-agent/pull/21175))
- Fix: teach restart-mocks about the post-update survivor sweep (salvage #19031) ([#21177](https://github.com/NousResearch/hermes-agent/pull/21177))

### Auth
- Fix: acp preserve assistant reasoning metadata ([#20296](https://github.com/NousResearch/hermes-agent/pull/20296))

### Redact
- Fix: add `code_file` param to skip false-positive ENV/JSON patterns ([#19715](https://github.com/NousResearch/hermes-agent/pull/19715))

### Email
- Fix: quoted-relative file-drop paths + Date header on tool email path ([#19646](https://github.com/NousResearch/hermes-agent/pull/19646))

---

## 🧪 Testing

- **ACP — accept prompt persistence kwargs in MCP E2E mocks** (@stephenschoettler) ([#18047](https://github.com/NousResearch/hermes-agent/pull/18047))
- **Toolsets — include kanban in expected post-#17805 toolset assertions** (@briandevans) ([#18122](https://github.com/NousResearch/hermes-agent/pull/18122))
- **Agent — cover max-iterations summary message sanitization** ([#19580](https://github.com/NousResearch/hermes-agent/pull/19580))
- **run_agent — `-inf` and `nan` regression coverage for `_coerce_number`** ([#19703](https://github.com/NousResearch/hermes-agent/pull/19703))

---

## 📚 Documentation

### Major docs additions
- **`llms.txt` + `llms-full.txt` — agent-friendly ingestion** ([#18276](https://github.com/NousResearch/hermes-agent/pull/18276))
- **User Stories and Use Cases collage page** ([#18282](https://github.com/NousResearch/hermes-agent/pull/18282))
- **Persistent Goals (/goal) feature page** ([#18275](https://github.com/NousResearch/hermes-agent/pull/18275))
- **Windows (WSL2) guide expansion** — filesystem, networking, services, pitfalls ([#20748](https://github.com/NousResearch/hermes-agent/pull/20748))
- **Chinese (zh-CN) README translation** (salvage #13508) ([#20431](https://github.com/NousResearch/hermes-agent/pull/20431))
- **zh-Hans Docusaurus locale** + Tool Gateway / image-gen / WSL quickstart translations (salvage #11728) ([#20430](https://github.com/NousResearch/hermes-agent/pull/20430))
- **Tool Gateway docs restructure** — lead with what it does, config moved to bottom ([#20827](https://github.com/NousResearch/hermes-agent/pull/20827))
- **Quickstart — Onchain AI Garage Hermes tutorials playlist** ([#20192](https://github.com/NousResearch/hermes-agent/pull/20192))
- **Open WebUI bootstrap script** (salvage #9566) ([#20427](https://github.com/NousResearch/hermes-agent/pull/20427))
- **Local Ollama setup guide** (salvage #5842) ([#20426](https://github.com/NousResearch/hermes-agent/pull/20426))
- **Google Gemini guide** (salvage #17450) ([#20401](https://github.com/NousResearch/hermes-agent/pull/20401))
- **Custom model aliases for /model command** ([#20475](https://github.com/NousResearch/hermes-agent/pull/20475))
- **Together/Groq/Perplexity cookbook via `custom_providers`** (salvage #15214) ([#20400](https://github.com/NousResearch/hermes-agent/pull/20400))
- **Doubao speech integration examples** (TTS + STT) (salvage #18065) ([#20418](https://github.com/NousResearch/hermes-agent/pull/20418))
- **WSL-to-Windows Chrome MCP bridge** (salvage #8313) ([#20428](https://github.com/NousResearch/hermes-agent/pull/20428))
- **Hermes skills docs sync** — slash commands + durable-systems section ([#20390](https://github.com/NousResearch/hermes-agent/pull/20390))
- **AGENTS.md — curator/cron/delegation/toolsets + fix plugin tree** ([#20226](https://github.com/NousResearch/hermes-agent/pull/20226))
- **Bedrock quickstart entry + fallback comment + deployment link** (salvage #11093) ([#20397](https://github.com/NousResearch/hermes-agent/pull/20397))

### Docs polish
- Collapse exploding skills tree to a single Skills node ([#18259](https://github.com/NousResearch/hermes-agent/pull/18259))
- Clarify `session_search` auxiliary model docs ([#19593](https://github.com/NousResearch/hermes-agent/pull/19593))
- Open WebUI Quick Setup gap fill ([#19654](https://github.com/NousResearch/hermes-agent/pull/19654))
- Default custom tool creation to plugins (@kshitijk4poor) ([#19755](https://github.com/NousResearch/hermes-agent/pull/19755))
- Clarify Telegram group chat troubleshooting (salvage #18672) ([#20416](https://github.com/NousResearch/hermes-agent/pull/20416))
- Codex OAuth auth prerequisite clarification (salvage #18688) ([#20417](https://github.com/NousResearch/hermes-agent/pull/20417))
- Discord Server Members Intent + SSRC-mapping drift + /voice join slash Choice (salvage #11350) ([#20411](https://github.com/NousResearch/hermes-agent/pull/20411))
- Document `ctx.dispatch_tool()` (salvage #10955) ([#20391](https://github.com/NousResearch/hermes-agent/pull/20391))
- Document `hermes webhook subscribe --deliver-only` (salvage #12612) ([#20392](https://github.com/NousResearch/hermes-agent/pull/20392))
- Document `hermes import` reference (salvage #14711) ([#20396](https://github.com/NousResearch/hermes-agent/pull/20396))
- Document per-provider TTS `max_text_length` caps (salvage #13825) ([#20389](https://github.com/NousResearch/hermes-agent/pull/20389))
- Clarify supported prompt customization surfaces (salvage #19987) ([#20383](https://github.com/NousResearch/hermes-agent/pull/20383))
- Correct `web_extract` summarizer timeout comment (salvage #20051) ([#20381](https://github.com/NousResearch/hermes-agent/pull/20381))
- Fix fallback provider config paths (salvage #20033) ([#20382](https://github.com/NousResearch/hermes-agent/pull/20382))
- Fix misleading RL install-extras claim (salvage #19080) ([#21213](https://github.com/NousResearch/hermes-agent/pull/21213))
- Clarify API server tool execution locality (salvage #19117) ([#21223](https://github.com/NousResearch/hermes-agent/pull/21223))
- Prefer `.venv` to match AGENTS.md and scripts/run_tests.sh (@xxxigm) ([#21334](https://github.com/NousResearch/hermes-agent/pull/21334))
- Align tool discovery + test runner with AGENTS.md (@xxxigm) ([#20791](https://github.com/NousResearch/hermes-agent/pull/20791))
- Align terminal-backend count and naming across docs and code (salvage #19044) ([#20402](https://github.com/NousResearch/hermes-agent/pull/20402))
- Refresh stale platform counts (salvage #19053) ([#20403](https://github.com/NousResearch/hermes-agent/pull/20403))

---

## 👥 Contributors

### Core
- **@teknium1** — salvage, triage, review, feature work, and release management

### Top Community Contributors

- **@kshitijk4poor** (21 PRs) — SearXNG native search backend, per-capability backend selection, collapsible TUI startup banner, Slack ephemeral ack + format fixes, Lightpanda fallback hardening, searxng-search optional skill + Web Search + Extract docs, default custom tool creation to plugins, kanban failure-column fix
- **@alt-glitch** (13 PRs) — video_analyze tool, xAI Custom Voices (voice cloning), local-backend CLI launch-directory fix, lazy-session creation regression recovery, systemd unit refresh on gateway boot
- **@OutThisLife** (9 PRs) — TUI perf — overlay render churn reduction, voice push-to-talk parity restoration (salvaging @Montbra)
- **@helix4u** (6 PRs) — Classic CLI output recovery after resize, absolute-path TUI completion, gateway model picker current-context fix, Bedrock credential probe avoidance, kanban docs fixes
- **@ethernet8023** (3 PRs) — Docker CI — don't cancel overlapping builds, :latest guard
- **@benbarclay** (3 PRs) — Docker — launch dashboard as side-process via HERMES_DASHBOARD=1
- **@austinpickett** (3 PRs) — Dashboard Plugins page, TUI /model picker overhaul with inline auth, kanban button fix
- **@sprmn24** (2 PRs) — Contributor (2 PRs)
- **@asheriif** (2 PRs) — Contributor (2 PRs)
- **@xxxigm** (2 PRs) — Contributing docs — .venv preference and test runner alignment with AGENTS.md
- **@stephenschoettler** (1 PR) — ACP — MCP E2E mock kwargs
- **@vincez-hms-coder** (1 PR) — Dashboard — Profiles management page
- **@cdanis** (1 PR) — Contributor
- **@briandevans** (1 PR) — Toolsets test — kanban assertions post-#17805
- **@heyitsaamir** (1 PR) — Contributor

### All Contributors

Thanks to everyone who contributed to v0.13.0 — commits, co-authored work, and salvaged PRs. 295 contributors in one week.

@0oAstro, @0xDevNinja, @0xharryriddle, @0xKingBack, @0xsir0000, @0xyg3n, @0z1-ghb, @abhinav11082001-stack,
@acc001k, @acesjohnny, @adamludwin, @adybag14-cyber, @agentlinker, @agilejava, @ai-ag2026, @AJV20,
@alanxchen85, @albert748, @AllardQuek, @alt-glitch, @altmazza0-star, @ambition0802, @amitgaur, @amroessam,
@andrewhosf, @Asce66, @asheriif, @ashermorse, @asimons81, @Aslaaen, @Asunfly, @atongrun, @austinpickett,
@banditburai, @barteqpl, @Bartok9, @Beandon13, @beardthelion, @beibi9966, @benbarclay, @binhnt92, @bjianhang,
@BlackJulySnow, @bobashopcashier, @bogerman1, @Bongulielmi, @Brecht-H, @briandevans, @brooklynnicholson,
@c3115644151, @camaragon, @CashWilliams, @CCClelo, @cdanis, @CES4751, @cg2aigc, @changchun989, @ChanlerDev,
@CharlieKerfoot, @chengoak, @chenyunbo411, @chinadbo, @CIRWEL, @cixuuz, @cmcgrabby-hue, @colorcross,
@Contentment003111, @CoreyNoDream, @counterposition, @curiouscleo, @DaniuXie, @deep-name, @dengtaoyuan450-a11y,
@discodirector, @donramon77, @dpaluy, @ee-blog, @ehz0ah, @el-analista, @elmatadorgh, @EmelyanenkoK,
@Emidomenge, @emozilla, @Es1la, @EthanGuo-coder, @etherman-os, @ethernet8023, @EvilDrag0n, @exxmen, @Fearvox,
@Feranmi10, @firefly, @flobo3, @fmercurio, @Foolafroos, @formulahendry, @franksong2702, @ggnnggez, @GinWU05,
@giwaov, @glesperance, @gnanirahulnutakki, @GodsBoy, @Gosuj, @Grey0202, @guillaumemeyer, @Gutslabs, @h0tp-ftw,
@haidao1919, @halmisen, @happy5318, @hedirman, @helix4u, @hendrixfreire, @HenkDz, @hex-clawd, @heyitsaamir,
@hharry11, @Hinotoi-agent, @holynn-q, @hrkzogw, @Hypn0sis, @Hypnus-Yuan, @ideathinklab01-source, @IMHaoyan,
@Interstellar-code, @ishardo, @jacdevos, @jackey8616, @JanCong, @jasonoutland, @jatingodnani, @JayGwod,
@jethac, @JezzaHehn, @JiaDe-Wu, @jjjojoj, @jkausel-ai, @John-tip, @johnncenae, @jrusso1020, @jslizar,
@JTroyerOvermatch, @julysir, @Junass1, @JustinUssuri, @Kailigithub, @keepcalmqqf, @kiala9, @konsisumer,
@kowenhaoai, @Krionex, @kshitijk4poor, @kyan12, @leavrcn, @leon7609, @LeonSGP43, @leprincep35700, @lhysdl,
@likejudy, @lisanhu, @liu-collab, @liuguangyong93, @liuhao1024, @LucianoSP, @luoyuctl, @luyao618, @M3RCUR2Y,
@maciekczech, @Magicray1217, @magicray1217, @MaHaoHao-ch, @malaiwah, @manateelazycat, @masonjames, @megastary,
@memosr, @MichaelWDanko, @mikeyobrien, @millerc79, @Mind-Dragon, @mioimotoai-lgtm, @misery-hl, @molvikar,
@momowind, @Montbra, @MottledShadow, @mrbob-git, @mrcharlesiv, @mrcoferland, @ms-alan, @mwnickerson,
@nazirulhafiy, @nftpoetrist, @nicoloboschi, @nightq, @nikolay-bratanov, @NikolayGusev-astra, @nocturnum91,
@noOne-list, @nouseman666, @novax635, @npmisantosh, @nudiltoys-cmyk, @olisikh, @oluwadareab12, @Oxidane-bot,
@pama0227, @pander, @pasevin, @paul-tian, @pdonizete, @perlowja, @pingchesu, @PratikRai0101, @priveperfumes,
@probepark, @QifengKuang, @quocanh261997, @qWaitCrypto, @qxxaa, @r266-tech, @rames-jusso, @revaraver,
@Ricardo-M-L, @rob-maron, @Roy-oss1, @rxdxxxx, @SandroHub013, @Sanjays2402, @Sertug17, @shashwatgokhe,
@shellybotmoyer, @SHL0MS, @SimbaKingjoe, @simbam99, @simplenamebox-ops, @socrates1024, @sonic-netizen,
@sprmn24, @steezkelly, @stephen0110, @stephenschoettler, @stevenchanin, @stevenchouai, @stormhierta,
@subtract0, @suncokret12, @swithek, @taeng0204, @TakeshiSawaguchi, @tangyuanjc, @TheEpTic, @thelumiereguy,
@Tkander1715, @tmdgusya, @Tranquil-Flow, @TruaShamu, @UgwujaGeorge, @valda, @vincez-hms-coder, @VinVC,
@vominh1919, @wabrent, @WadydX, @wanazhar, @WanderWang, @warabe1122, @web-dev0521, @WideLee, @willy-scr,
@wmagev, @WuTianyi123, @wxst, @wysie, @Wysie, @xsfX20, @xxxigm, @xyiy001, @YanzhongSu, @ygd58, @Yoimex,
@yuehei, @Yukipukii1, @yuqianma, @YX234, @zeejaytan, @zhanggttry, @zhao0112, @zng8418, @zons-zhaozhy, @Zyproth

---

**Full Changelog**: [v2026.4.30...v2026.5.7](https://github.com/NousResearch/hermes-agent/compare/v2026.4.30...v2026.5.7)
