# Hermes Agent v0.14.0 (v2026.5.16)

**Release Date:** May 16, 2026
**Since v0.13.0:** 808 commits · 633 merged PRs · 1393 files changed · 165,061 insertions · 545 issues closed (12 P0, 50 P1) · 215 community contributors (including co-authors)

> The Foundation Release — Hermes Agent installs and runs anywhere now. Native Windows ships in early beta with a full PowerShell installer story, a `pip install hermes-agent` wheel lands on PyPI, lazy-deps reshape what `pip install hermes-agent` actually pulls down, the supply-chain checker scans every install/upgrade for unsafe versions, and a new OpenAI-compatible local proxy lets Codex / Aider / Cline talk to OAuth-only providers (Claude Pro, ChatGPT Pro, SuperGrok). The cold-start wave shaves ~19 seconds off `hermes` launch, browser-tool CDP calls run 180x faster, and `hermes tools` All-Platforms drops from 14s to under 1.5s. Two new messaging platforms (LINE and SimpleX Chat) and a Microsoft Graph foundation (Teams pipeline + webhook adapter) land alongside `/handoff` that finally transfers sessions live, `vision_analyze` passing pixels through to vision-capable models, `x_search` as a first-class tool, LSP semantic diagnostics on every `write_file` / `patch`, a unified pluggable `video_generate`, a `computer_use` cua-driver backend, cross-session 1-hour Claude prompt caching, a per-turn file-mutation verifier, plus 9 new optional skills. 50+ P1 closures, 12 P0 closures.

---

## ✨ Highlights

- **Native Windows support (early beta)** — full PowerShell installer, native subprocess/PTY paths, taskkill-based process management, MinGit auto-install, Microsoft Store python stub detection, foreground Ctrl+C preservation, taskkill+ps2 fallback, npm prefix handling, and ~40 follow-up Windows-only fixes across CLI / gateway / TUI / curator / tools. Hermes finally runs natively on `cmd.exe` and PowerShell, no WSL required. ([#21561](https://github.com/NousResearch/hermes-agent/pull/21561), [#22130](https://github.com/NousResearch/hermes-agent/pull/22130), [#22752](https://github.com/NousResearch/hermes-agent/pull/22752), [#26618](https://github.com/NousResearch/hermes-agent/pull/26618), and many more)

- **`pip install hermes-agent && hermes`** — Hermes Agent is now a real PyPI package. One command, no clone, no git, no shell installer. Wheel includes the Ink TUI bundle and shell launcher. (salvage of [#26350](https://github.com/NousResearch/hermes-agent/pull/26350)) ([#26593](https://github.com/NousResearch/hermes-agent/pull/26593))

- **Cold-start performance wave — ~19s off `hermes` launch** — skills cache, lazy Feishu import, no Nous HTTP at startup, plus PEP-562 lazy adapter imports (QQ, Yuanbao, Teams, Google Chat), deferred `fal_client` / `google-cloud` / `httpx` loads, models.dev disk-cache-first lookup, parallel doctor API checks, eager-skip plugin discovery on built-in subcommands, `hermes tools` All-Platforms drops from 14s to <1.5s, welcome banner skipped on `chat -q`. ([#22138](https://github.com/NousResearch/hermes-agent/pull/22138), [#22120](https://github.com/NousResearch/hermes-agent/pull/22120), [#22681](https://github.com/NousResearch/hermes-agent/pull/22681), [#22790](https://github.com/NousResearch/hermes-agent/pull/22790), [#22808](https://github.com/NousResearch/hermes-agent/pull/22808), [#22831](https://github.com/NousResearch/hermes-agent/pull/22831), [#22859](https://github.com/NousResearch/hermes-agent/pull/22859), [#22904](https://github.com/NousResearch/hermes-agent/pull/22904), [#22766](https://github.com/NousResearch/hermes-agent/pull/22766), [#25341](https://github.com/NousResearch/hermes-agent/pull/25341))

- **180x faster `browser_console` evaluations** — routed through the supervisor's persistent CDP WebSocket instead of spawning a fresh DevTools session per call. Real-world page interactions feel instant. ([#23226](https://github.com/NousResearch/hermes-agent/pull/23226))

- **Supply-chain advisory checker + lazy-deps framework + tiered install fallback** — every `pip install` / `hermes update` scans dependencies against an advisory list, lazy-deps replace heavy import-time loads with first-use installs, and the installer falls back through extras tiers when a wheel rejects on the target platform. ([#24220](https://github.com/NousResearch/hermes-agent/pull/24220))

- **OpenAI-compatible local proxy** — `hermes proxy` exposes any OAuth-authed provider (Claude Pro, ChatGPT Pro, SuperGrok) as an OpenAI-compatible endpoint that Codex / Aider / Cline / VS Code Continue can hit. Your subscription, your tools. ([#25969](https://github.com/NousResearch/hermes-agent/pull/25969))

- **Cross-session 1-hour Claude prompt cache** — Anthropic / OpenRouter / Nous Portal now share a 1h prefix cache across sessions for Claude models. Fast resume, fast `/new`, lower cost on repeat work. ([#23828](https://github.com/NousResearch/hermes-agent/pull/23828))

- **Two new messaging platforms — LINE + SimpleX Chat** — LINE Messaging API lands as a first-class platform, SimpleX Chat salvages #2558 onto the modern adapter spec. Hermes is now on 22 platforms. ([#23197](https://github.com/NousResearch/hermes-agent/pull/23197), [#26232](https://github.com/NousResearch/hermes-agent/pull/26232))

- **Microsoft Graph foundation — Teams pipeline + webhook adapter** — `msgraph` auth/client foundation, webhook listener platform, Teams pipeline plugin runtime, and Teams outbound delivery via the existing adapter — Hermes can now read and post to Teams. (salvages of #21408–#21411) ([#21922](https://github.com/NousResearch/hermes-agent/pull/21922), [#21969](https://github.com/NousResearch/hermes-agent/pull/21969), [#22007](https://github.com/NousResearch/hermes-agent/pull/22007), [#22024](https://github.com/NousResearch/hermes-agent/pull/22024))

- **`/handoff` actually transfers the session live** — the agent's active session moves to a different model / persona / profile mid-conversation, with messages, tool history, and context preserved. ([#23395](https://github.com/NousResearch/hermes-agent/pull/23395))

- **`x_search` — first-class X (Twitter) search tool** — gated tool with OAuth-or-API-key auth, no skill needed to query the timeline. ([#26763](https://github.com/NousResearch/hermes-agent/pull/26763))

- **`vision_analyze` returns pixels to vision-capable models** — when the active model can see, `vision_analyze` now hands the image straight through instead of falling back to a text description. ([#22955](https://github.com/NousResearch/hermes-agent/pull/22955))

- **LSP semantic diagnostics on every write** — `write_file` and `patch` now run real language-server diagnostics on the post-edit file (delta-only) and surface real errors before they ship downstream. ([#24168](https://github.com/NousResearch/hermes-agent/pull/24168), [#25978](https://github.com/NousResearch/hermes-agent/pull/25978))

- **Per-turn file-mutation verifier footer** — after every turn that wrote files, the agent gets a verifier footer summarizing what actually changed on disk — catches silent overwrites and "wrote it but it didn't land" bugs. ([#24498](https://github.com/NousResearch/hermes-agent/pull/24498))

- **Unified `video_generate` with pluggable provider backends** — single tool, any backend. Drop in a new video provider as a plugin, no core changes. ([#25126](https://github.com/NousResearch/hermes-agent/pull/25126))

- **`computer_use` cua-driver backend** — proper focus-safe ops, non-Anthropic provider support, refresh on `hermes update`. Computer-use is no longer locked to a single SDK. (re-salvage of #16936) ([#21967](https://github.com/NousResearch/hermes-agent/pull/21967), [#24063](https://github.com/NousResearch/hermes-agent/pull/24063))

- **xAI Grok OAuth provider — SuperGrok via subscription** — sign in with your xAI account, talk to Grok models from Hermes. ([#26534](https://github.com/NousResearch/hermes-agent/pull/26534))

- **Clarify with buttons — native inline keyboards on Telegram + Discord** — the `clarify` tool renders multi-choice prompts as platform-native buttons instead of typed responses. ([#24199](https://github.com/NousResearch/hermes-agent/pull/24199), [#25485](https://github.com/NousResearch/hermes-agent/pull/25485))

- **Discord channel history backfill (default on)** — Hermes reads recent channel history when joining a thread so it actually knows what's been said. ([#25984](https://github.com/NousResearch/hermes-agent/pull/25984))

- **Watchers skill — RSS / HTTP JSON / GitHub polling via cron `no_agent` mode** — skill recipes that wire change-detection sources directly into cron's script-only watchdog mode. ([#21881](https://github.com/NousResearch/hermes-agent/pull/21881))

- **Zed ACP Registry integration + uvx distribution** — Hermes is in the Zed registry, installable via `uvx` (no npm). Plus `hermes acp --setup-browser` bootstraps browser tools for registry installs. (salvage of [#25908](https://github.com/NousResearch/hermes-agent/pull/25908)) ([#26079](https://github.com/NousResearch/hermes-agent/pull/26079), [#26120](https://github.com/NousResearch/hermes-agent/pull/26120), [#26234](https://github.com/NousResearch/hermes-agent/pull/26234))

- **OpenRouter Pareto Code router** — wire a new OpenRouter router with `min_coding_score` knob. Pick the cheapest model that meets your quality bar. ([#22838](https://github.com/NousResearch/hermes-agent/pull/22838))

- **Optional codex app-server runtime for OpenAI/Codex models** — drives the OpenAI Codex CLI under the hood for OpenAI/Codex paths, with session reuse, wedge retirement, and OAuth refresh classification. ([#24182](https://github.com/NousResearch/hermes-agent/pull/24182), [#25769](https://github.com/NousResearch/hermes-agent/pull/25769))

- **`hermes-skills/huggingface` as a trusted default tap** — community skills index from huggingface.co/skills is available by default in the Skills Hub. ([#26219](https://github.com/NousResearch/hermes-agent/pull/26219))

- **9 new optional skills** — Hyperliquid (perp/spot trading via SDK + REST) (@kshitijk4poor & Hermes), Yahoo Finance market data, api-testing (REST/GraphQL debug), unified EVM multi-chain skill (folds #25291 + #2010 + base/), darwinian-evolver, osint-investigation (closes #355), pinggy-tunnel, watchers (RSS/HTTP/GitHub via cron), Notion overhaul for the Developer Platform (May 2026). ([#23582](https://github.com/NousResearch/hermes-agent/pull/23582), [#23583](https://github.com/NousResearch/hermes-agent/pull/23583), [#23590](https://github.com/NousResearch/hermes-agent/pull/23590), [#25299](https://github.com/NousResearch/hermes-agent/pull/25299), [#26760](https://github.com/NousResearch/hermes-agent/pull/26760), [#26729](https://github.com/NousResearch/hermes-agent/pull/26729), [#26765](https://github.com/NousResearch/hermes-agent/pull/26765), [#21881](https://github.com/NousResearch/hermes-agent/pull/21881), [#26612](https://github.com/NousResearch/hermes-agent/pull/26612))

- **API server exposes run approval events** — long-running runs surface approval requests over the API stream, no more silent stalls. (salvage of [#20311](https://github.com/NousResearch/hermes-agent/pull/20311)) ([#21899](https://github.com/NousResearch/hermes-agent/pull/21899))

- **`/subgoal` — user-added criteria appended to active `/goal`** — layer extra success criteria onto a running goal loop. The judge sees them in the prompt, no behavior change when subgoals are empty. ([#25449](https://github.com/NousResearch/hermes-agent/pull/25449))

- **Plugins can run any LLM call via `ctx.llm`** — plugins get a first-class hook to make their own LLM requests through the active provider/credentials, no manual wiring. Plus `tool_override` flag for replacing built-in tools. ([#23194](https://github.com/NousResearch/hermes-agent/pull/23194), [#26759](https://github.com/NousResearch/hermes-agent/pull/26759))

- **Brave Search (free tier) + DuckDuckGo (DDGS) as web-search providers** — two new free search backends alongside Tavily / SearXNG / Exa. ([#21337](https://github.com/NousResearch/hermes-agent/pull/21337))

- **Sudo brute-force block + sudo-stdin/askpass DANGEROUS classification** — closes the `sudo -S` brute-force avenue; approval gates classify stdin-fed and askpass-stripped sudo invocations as dangerous. (salvages of #22194 + #21128) ([#23736](https://github.com/NousResearch/hermes-agent/pull/23736))

- **Provider rename — Alibaba Cloud → Qwen Cloud, picker reorder** — matches what the world calls it. Existing config keys still work. ([#24835](https://github.com/NousResearch/hermes-agent/pull/24835))


---

## 🪟 Windows — Native Support (Early Beta)

### Bootstrap & installer
- **Native Windows support (early beta)** — first-class native Windows path across CLI / gateway / TUI / tools ([#21561](https://github.com/NousResearch/hermes-agent/pull/21561))
- **PyPI wheel packaging — `pip install hermes-agent && hermes`** (salvage of #26350) ([#26593](https://github.com/NousResearch/hermes-agent/pull/26593))
- **Recognise Shift+Enter as a newline key** + Windows docs (salvage #21545) ([#22130](https://github.com/NousResearch/hermes-agent/pull/22130))
- **Preserve Ctrl+C for Windows foreground runs** (@helix4u) ([#22752](https://github.com/NousResearch/hermes-agent/pull/22752))
- **Stop spamming cwd-missing + tirith-spawn warnings on every terminal call** ([#26618](https://github.com/NousResearch/hermes-agent/pull/26618))
- **Use `--extra all` not `--all-extras`; drop lazy-covered extras from `[all]`** ([#24515](https://github.com/NousResearch/hermes-agent/pull/24515))

### Windows-specific fixes (40+ across cli / tools / gateway / curator / TUI)
A long tail of native-Windows fixes shipped alongside the beta — taskkill-based subprocess management, MinGit auto-install, Microsoft Store python stub detection, npm prefix handling, native PTY paths, signal handling differences, foreground process management, ANSI sequence handling, path normalization, file-locking semantics, and many more. Full list in commit log under `fix(windows)` / `feat(windows)` / `windows`.

---

## 🚀 Performance Wave

### Cold start
- **Cut ~19s from `hermes` cold start** — skills cache + lazy Feishu + no Nous HTTP at startup ([#22138](https://github.com/NousResearch/hermes-agent/pull/22138))
- **Skip eager plugin discovery on known built-in subcommands** ([#22120](https://github.com/NousResearch/hermes-agent/pull/22120))
- **Cache Nous auth + .env loads** — `hermes tools` All Platforms from 14s to <1.5s ([#25341](https://github.com/NousResearch/hermes-agent/pull/25341))
- **Skip welcome banner on `chat -q` single-query mode** ([#22904](https://github.com/NousResearch/hermes-agent/pull/22904))
- **Defer heavy google-cloud imports in google_chat to first adapter use** ([#22681](https://github.com/NousResearch/hermes-agent/pull/22681))
- **Defer QQAdapter and YuanbaoAdapter imports via PEP 562** ([#22790](https://github.com/NousResearch/hermes-agent/pull/22790))
- **Defer httpx import in teams to first webhook call** ([#22831](https://github.com/NousResearch/hermes-agent/pull/22831))
- **Defer fal_client import to first generation request** ([#22859](https://github.com/NousResearch/hermes-agent/pull/22859))
- **models.dev cache-first lookup, skip network when disk cache is fresh** ([#22808](https://github.com/NousResearch/hermes-agent/pull/22808))
- **Parallelize API connectivity checks in `hermes doctor` and disable IMDS** ([#22766](https://github.com/NousResearch/hermes-agent/pull/22766))

### Runtime
- **180x faster `browser_console` evaluations** — route through supervisor's persistent CDP WebSocket ([#23226](https://github.com/NousResearch/hermes-agent/pull/23226))
- **Tune Telegram cadence + adaptive fast-path for short replies** (salvage of #10388) ([#23587](https://github.com/NousResearch/hermes-agent/pull/23587))
- **Accumulate length-continuation prefix via list+join** ([#26237](https://github.com/NousResearch/hermes-agent/pull/26237))

### Prompt caching
- **Cross-session 1h prefix cache for Claude on Anthropic / OpenRouter / Nous Portal** ([#23828](https://github.com/NousResearch/hermes-agent/pull/23828))
- **Hit prefix cache in background review fork** (salvage #17276 + #25427) ([#25434](https://github.com/NousResearch/hermes-agent/pull/25434))

---

## 📦 Installation & Distribution

### PyPI + supply-chain
- **PyPI wheel packaging — `pip install hermes-agent && hermes`** (salvage of #26350) ([#26593](https://github.com/NousResearch/hermes-agent/pull/26593))
- **Supply-chain advisory checker + lazy-install framework + tiered install fallback** ([#24220](https://github.com/NousResearch/hermes-agent/pull/24220))
- **Use `--extra all` not `--all-extras`; drop lazy-covered extras from `[all]`** ([#24515](https://github.com/NousResearch/hermes-agent/pull/24515))
- **Skip browser download when system chromium exists** (@helix4u) ([#25317](https://github.com/NousResearch/hermes-agent/pull/25317))

### Nix
- **`extraDependencyGroups` for sealed venv extras** (@alt-glitch) ([#21817](https://github.com/NousResearch/hermes-agent/pull/21817))
- **Refresh npm lockfile hashes** — keeps Nix flake builds reproducible

### Docker
- **Bootstrap auth.json from env on first boot** ([#21880](https://github.com/NousResearch/hermes-agent/pull/21880))
- **Drop manual @hermes/ink build, rely on esbuild bundle** — slimmer image

### ACP / Zed
- **Zed ACP Registry integration** (salvage of #25908) ([#26079](https://github.com/NousResearch/hermes-agent/pull/26079))
- **Switch to uvx distribution, drop npm launcher** ([#26120](https://github.com/NousResearch/hermes-agent/pull/26120))
- **`hermes acp --setup-browser` bootstraps browser tools for registry installs** ([#26234](https://github.com/NousResearch/hermes-agent/pull/26234))

---

## 🏗️ Core Agent & Architecture

### Sessions & handoff
- **`/handoff` actually transfers the session live** ([#23395](https://github.com/NousResearch/hermes-agent/pull/23395))
- **Expose `HERMES_SESSION_ID` env var to agent tools** (@alt-glitch) ([#23847](https://github.com/NousResearch/hermes-agent/pull/23847))

### Goals (Ralph loop)
- **`/subgoal` — user-added criteria appended to active `/goal`** ([#25449](https://github.com/NousResearch/hermes-agent/pull/25449))
- **`/goal` checklist + /subgoal user controls** ([#23456](https://github.com/NousResearch/hermes-agent/pull/23456)) — rolled back in window ([#23813](https://github.com/NousResearch/hermes-agent/pull/23813)); /subgoal returned in simpler form via #25449

### Compression
- **Make `protect_first_n` configurable** ([#25447](https://github.com/NousResearch/hermes-agent/pull/25447))

### Verification
- **Per-turn file-mutation verifier footer** ([#24498](https://github.com/NousResearch/hermes-agent/pull/24498))

### Stream retry
- **Log inner cause, upstream headers, bytes/elapsed on every drop** ([#23005](https://github.com/NousResearch/hermes-agent/pull/23005))

---

## 🤖 Models & Providers

### New providers
- **xAI Grok OAuth (SuperGrok Subscription) provider** ([#26534](https://github.com/NousResearch/hermes-agent/pull/26534))
- **NovitaAI provider** (salvage #7219) (@kshitijk4poor) ([#25507](https://github.com/NousResearch/hermes-agent/pull/25507))
- **NVIDIA NIM billing origin header** (salvage #25211) ([#26585](https://github.com/NousResearch/hermes-agent/pull/26585))

### Provider work
- **OpenRouter Pareto Code router with `min_coding_score` knob** ([#22838](https://github.com/NousResearch/hermes-agent/pull/22838))
- **Optional codex app-server runtime for OpenAI/Codex models** ([#24182](https://github.com/NousResearch/hermes-agent/pull/24182))
- **Codex-runtime: retire wedged sessions + post-tool watchdog + OAuth refresh classify** ([#25769](https://github.com/NousResearch/hermes-agent/pull/25769))
- **Codex-runtime: skip unavailable plugins during migration** ([#25437](https://github.com/NousResearch/hermes-agent/pull/25437))
- **Codex-runtime: de-dup `[plugins.X]` tables and stop leaking HERMES_HOME into config.toml** (#26250) (@kshitijk4poor) ([#26260](https://github.com/NousResearch/hermes-agent/pull/26260))
- **Pass `reasoning.effort` to xAI Responses API** ([#22807](https://github.com/NousResearch/hermes-agent/pull/22807))
- **Custom provider: prompt and persist explicit `api_mode`** ([#25068](https://github.com/NousResearch/hermes-agent/pull/25068))
- **Rename Alibaba Cloud → Qwen Cloud, reorder picker** ([#24835](https://github.com/NousResearch/hermes-agent/pull/24835))
- **Restore gpt-5.3-codex-spark for ChatGPT Pro** (salvage #18286 + #19530, fixes #16172) (@kshitijk4poor) ([#22991](https://github.com/NousResearch/hermes-agent/pull/22991))
- **Inject tool-use enforcement for GLM models** ([#24715](https://github.com/NousResearch/hermes-agent/pull/24715))
- **Use Nous Portal as model metadata authority** (@rob-maron) ([#24502](https://github.com/NousResearch/hermes-agent/pull/24502))
- **Unified `client=hermes-client-v<version>` tag on every Portal request** ([#24779](https://github.com/NousResearch/hermes-agent/pull/24779))
- **Prevent stale Ollama credentials after provider switch** (@kshitijk4poor) ([#21703](https://github.com/NousResearch/hermes-agent/pull/21703))
- **Auxiliary client: rotate pooled auth after quota failures** (salvage #22779) ([#22792](https://github.com/NousResearch/hermes-agent/pull/22792))
- **Auxiliary client: skip providers without credentials immediately** (#25395) ([#25487](https://github.com/NousResearch/hermes-agent/pull/25487))
- **Auth: send Nous refresh token via header** (@shannonsands) ([#21578](https://github.com/NousResearch/hermes-agent/pull/21578))
- **MiniMax: harden OAuth dashboard and runtime** ([#24165](https://github.com/NousResearch/hermes-agent/pull/24165))

### OpenAI-compatible proxy
- **Local OpenAI-compatible proxy for OAuth providers** — Codex / Aider / Cline can hit Claude Pro, ChatGPT Pro, SuperGrok ([#25969](https://github.com/NousResearch/hermes-agent/pull/25969))

---

## 📱 Messaging Platforms (Gateway)

### New platforms
- **LINE Messaging API platform plugin** ([#23197](https://github.com/NousResearch/hermes-agent/pull/23197))
- **SimpleX Chat platform plugin** (salvages #2558) ([#26232](https://github.com/NousResearch/hermes-agent/pull/26232))

### Microsoft Graph foundation
- **msgraph: add auth and client foundation** (salvage of #21408) ([#21922](https://github.com/NousResearch/hermes-agent/pull/21922))
- **msgraph: add webhook listener platform** (salvage of #21409) ([#21969](https://github.com/NousResearch/hermes-agent/pull/21969))
- **teams-pipeline: add plugin runtime and operator cli** (salvage of #21410) ([#22007](https://github.com/NousResearch/hermes-agent/pull/22007))
- **teams: add pipeline outbound delivery via existing adapter** (salvage of #21411) ([#22024](https://github.com/NousResearch/hermes-agent/pull/22024))

### Cross-platform
- **Per-platform admin/user split for slash commands** (salvage of #4443) ([#23373](https://github.com/NousResearch/hermes-agent/pull/23373))
- **Forensics on signal handling — non-blocking diag, per-phase timing, stale-unit warning** ([#23285](https://github.com/NousResearch/hermes-agent/pull/23285))
- **Keep gateway running when platforms fail; add per-platform circuit breaker + `/platform`** ([#26600](https://github.com/NousResearch/hermes-agent/pull/26600))
- **Wire `clarify` tool with inline keyboard buttons on Telegram** ([#24199](https://github.com/NousResearch/hermes-agent/pull/24199))
- **Add `chat_id` to `hook_ctx` for message source tracking** ([#24710](https://github.com/NousResearch/hermes-agent/pull/24710))

### Telegram
- **Native draft streaming via `sendMessageDraft` (Bot API 9.5+)** (salvage of #3412) ([#23512](https://github.com/NousResearch/hermes-agent/pull/23512))
- **Stream Telegram edits safely** — salvage of #22264 (@kshitijk4poor) ([#22518](https://github.com/NousResearch/hermes-agent/pull/22518))
- **Telegram notification mode** (salvage #22772) ([#22793](https://github.com/NousResearch/hermes-agent/pull/22793))
- **Telegram guest mention mode** (@kshitijk4poor) ([#22759](https://github.com/NousResearch/hermes-agent/pull/22759))
- **Split-and-deliver oversized edits instead of silent truncation** (salvage of #19537) ([#23576](https://github.com/NousResearch/hermes-agent/pull/23576))
- **Preserve DM topic routing via reply fallback** (salvage #22053) (@kshitijk4poor) ([#22410](https://github.com/NousResearch/hermes-agent/pull/22410))
- **Pass `source.thread_id` explicitly on auto-reset notice** (carve-out of #7404) ([#23440](https://github.com/NousResearch/hermes-agent/pull/23440))

### Discord
- **Render clarify choices as buttons** ([#25485](https://github.com/NousResearch/hermes-agent/pull/25485))
- **Channel history backfill — default on, broadened scope** ([#25984](https://github.com/NousResearch/hermes-agent/pull/25984))
- **`thread_require_mention` for multi-bot threads** (salvage #25313) ([#25445](https://github.com/NousResearch/hermes-agent/pull/25445))

### Slack
- **Support `!cmd` as alternate prefix for slash commands in threads** ([#25355](https://github.com/NousResearch/hermes-agent/pull/25355))

### WhatsApp
- **Surface quoted reply metadata from Baileys** (#25398) ([#25489](https://github.com/NousResearch/hermes-agent/pull/25489))

### Feishu / Google Chat / others
- **Feishu: native update prompt cards** (@kshitijk4poor) ([#22448](https://github.com/NousResearch/hermes-agent/pull/22448))
- **Google Chat: repair setup prompt imports** (@helix4u) ([#22038](https://github.com/NousResearch/hermes-agent/pull/22038))
- **Google Chat: honor relay-declared sender_type** (salvage of #22107) (@kshitijk4poor) ([#22432](https://github.com/NousResearch/hermes-agent/pull/22432))
- **LINE: use `build_source` instead of nonexistent `create_source`** ([#24717](https://github.com/NousResearch/hermes-agent/pull/24717))
- **Add `weixin, and more` to gateway docs** (salvage of #21063 by @wuwuzhijing)

---

## 🖥️ CLI & TUI

### CLI
- **Show YOLO mode warning in banner and status bar** ([#26238](https://github.com/NousResearch/hermes-agent/pull/26238))
- **Confirm prompt for destructive slash commands** (#4069) ([#22687](https://github.com/NousResearch/hermes-agent/pull/22687))
- **`docker_extra_args` + `display.timestamps`** ([#23599](https://github.com/NousResearch/hermes-agent/pull/23599))
- **Delegate tool: show user's actual concurrency / spawn-depth limits in description** ([#22694](https://github.com/NousResearch/hermes-agent/pull/22694))

### TUI
- **`/sessions` slash command for browsing and resuming previous sessions** (@austinpickett) ([#20805](https://github.com/NousResearch/hermes-agent/pull/20805))
- **Segment turns with rule above non-first user msgs; trim ticker dead space** (@OutThisLife) ([#21846](https://github.com/NousResearch/hermes-agent/pull/21846))
- **Support attaching to an existing gateway** (@OutThisLife) ([#21978](https://github.com/NousResearch/hermes-agent/pull/21978))
- **Resolve markdown links to readable page titles** (@OutThisLife) ([#24013](https://github.com/NousResearch/hermes-agent/pull/24013))
- **Width-aware markdown table rendering with vertical fallback** (@alt-glitch) ([#26195](https://github.com/NousResearch/hermes-agent/pull/26195))
- **Keep Ink displayCursor in sync with fast-echo writes so cursor stops drifting** (@OutThisLife) ([#26717](https://github.com/NousResearch/hermes-agent/pull/26717))
- **Allow transcript scroll + Esc during approval/clarify/confirm prompts** (@OutThisLife) ([#26414](https://github.com/NousResearch/hermes-agent/pull/26414))
- **Preserve session when switching personality** (@austinpickett) ([#20942](https://github.com/NousResearch/hermes-agent/pull/20942))
- **Skip native safety net on OSC52-capable terminals** (@benbarclay) ([#20954](https://github.com/NousResearch/hermes-agent/pull/20954))

### Dashboard / GUI
- **Route embedded TUI through dashboard gateway** (@OutThisLife) ([#21979](https://github.com/NousResearch/hermes-agent/pull/21979))
- **Hide token/cost analytics behind config flag (default off)** ([#25438](https://github.com/NousResearch/hermes-agent/pull/25438))
- **Fix Langfuse observability — trace I/O, tool outputs, placeholder credentials** (closes #22342, #22763) (@kshitijk4poor) ([#26320](https://github.com/NousResearch/hermes-agent/pull/26320))
- **MiniMax 'Login' button launched Claude OAuth** (salvage #22849) ([#24058](https://github.com/NousResearch/hermes-agent/pull/24058))
- **Update cron modals** (@austinpickett) ([#25985](https://github.com/NousResearch/hermes-agent/pull/25985))
- **Analytics: prevent silent token loss and add Claude 4.5–4.7 pricing** (@austinpickett) ([#21455](https://github.com/NousResearch/hermes-agent/pull/21455))

---

## 🔧 Tools & Capabilities

### Vision & video
- **`vision_analyze` returns pixels to vision-capable models** ([#22955](https://github.com/NousResearch/hermes-agent/pull/22955))
- **Unified `video_generate` with pluggable provider backends** ([#25126](https://github.com/NousResearch/hermes-agent/pull/25126))
- **`image_gen`: actionable setup message when no FAL backend is reachable** ([#26222](https://github.com/NousResearch/hermes-agent/pull/26222))

### Computer use
- **`computer_use` cua-driver backend + focus-safe ops + non-Anthropic provider fix** (re-salvage #16936) ([#21967](https://github.com/NousResearch/hermes-agent/pull/21967))
- **Refresh cua-driver on `hermes update` + add `install --upgrade`** ([#24063](https://github.com/NousResearch/hermes-agent/pull/24063))

### LSP & write-time diagnostics
- **Semantic diagnostics from real language servers in `write_file`/`patch`** ([#24168](https://github.com/NousResearch/hermes-agent/pull/24168))
- **Shift baseline diagnostics into post-edit coordinates** ([#25978](https://github.com/NousResearch/hermes-agent/pull/25978))

### Search & web
- **Brave Search (free tier) and DDGS search providers** ([#21337](https://github.com/NousResearch/hermes-agent/pull/21337))
- **Bearer auth header for Tavily `/crawl` endpoint** ([#24658](https://github.com/NousResearch/hermes-agent/pull/24658))

### X (Twitter)
- **Gated `x_search` tool with OAuth-or-API-key auth** ([#26763](https://github.com/NousResearch/hermes-agent/pull/26763))

### Browser
- **Route `browser_console` eval through supervisor's persistent CDP WS (180x faster)** ([#23226](https://github.com/NousResearch/hermes-agent/pull/23226))
- **Support externally managed Camofox sessions** ([#24499](https://github.com/NousResearch/hermes-agent/pull/24499))

### MCP
- **`supports_parallel_tool_calls` for MCP servers** (salvage of #9944) ([#26825](https://github.com/NousResearch/hermes-agent/pull/26825))
- **Codex preset for Codex CLI MCP server** (salvage #22663) ([#22679](https://github.com/NousResearch/hermes-agent/pull/22679))
- **Stop retrying initial MCP auth failures** (#25624) ([#25776](https://github.com/NousResearch/hermes-agent/pull/25776))

### Google Workspace
- **Drive write ops + Docs/Sheets create/append** ([#21895](https://github.com/NousResearch/hermes-agent/pull/21895))

### Per-turn verifier
- **Per-turn file-mutation verifier footer** ([#24498](https://github.com/NousResearch/hermes-agent/pull/24498))

---

## 🧩 Kanban (Multi-Agent)

- **`specify` — auxiliary LLM fleshes out triage tasks** ([#21435](https://github.com/NousResearch/hermes-agent/pull/21435))
- **Orchestrator board tools — `kanban_list` + `kanban_unblock`** (carve-out of #20568) ([#23012](https://github.com/NousResearch/hermes-agent/pull/23012))
- **`stranded_in_ready` diagnostic for unclaimed tasks** ([#23578](https://github.com/NousResearch/hermes-agent/pull/23578))
- **Dashboard batch QOL upgrade** (salvage of #23240) ([#23550](https://github.com/NousResearch/hermes-agent/pull/23550))
- **Tooltips and docs link across dashboard** ([#21541](https://github.com/NousResearch/hermes-agent/pull/21541))
- **Dedupe notifier delivery via atomic claim + rewind on failure** (salvage #22558) ([#23401](https://github.com/NousResearch/hermes-agent/pull/23401))
- **Keep notifier subscriptions alive across retry cycles** (salvage #21398) ([#23423](https://github.com/NousResearch/hermes-agent/pull/23423))
- **Drop caller-controlled author override in `kanban_comment`** (salvage of #22109) (@kshitijk4poor) ([#22435](https://github.com/NousResearch/hermes-agent/pull/22435))
- **Sanitize comment author rendering in `build_worker_context`** ([#22769](https://github.com/NousResearch/hermes-agent/pull/22769))

---

## 🧠 Plugins & Extension

### Plugin surface
- **Run any LLM call from inside a plugin via `ctx.llm`** ([#23194](https://github.com/NousResearch/hermes-agent/pull/23194))
- **`tool_override` flag for replacing built-in tools** (closes #11049) ([#26759](https://github.com/NousResearch/hermes-agent/pull/26759))
- **`standalone_sender_fn` for out-of-process cron delivery** (@kshitijk4poor) ([#22461](https://github.com/NousResearch/hermes-agent/pull/22461))
- **`HERMES_PLUGINS_DEBUG=1` surfaces plugin discovery logs** ([#22684](https://github.com/NousResearch/hermes-agent/pull/22684))
- **Hindsight-client as optional dependency** (@alt-glitch) ([#21818](https://github.com/NousResearch/hermes-agent/pull/21818))

### Profile & distribution
- **Shareable profile distributions via git** ([#20831](https://github.com/NousResearch/hermes-agent/pull/20831))

---

## ⏰ Cron

- **Routing intent — `deliver=all` fans out to every connected channel** ([#21495](https://github.com/NousResearch/hermes-agent/pull/21495))
- **Support name-based lookup for job operations** ([#26231](https://github.com/NousResearch/hermes-agent/pull/26231))
- **Blank Cron dashboard tab + partial-record crashes** (salvage #21042 + #22330) (@kshitijk4poor) ([#22389](https://github.com/NousResearch/hermes-agent/pull/22389))
- **Do not seed `HERMES_SESSION_*` contextvars from cron origin** (salvage of #22356) (@kshitijk4poor) ([#22382](https://github.com/NousResearch/hermes-agent/pull/22382))
- **Scan assembled prompt including skill content for prompt injection** (#3968)

---

## 🧩 Skills Ecosystem

### Skills Hub
- **`hermes-skills/huggingface` as a trusted default tap** (closes #2549) ([#26219](https://github.com/NousResearch/hermes-agent/pull/26219))
- **Show per-skill pages in the left sidebar** ([#26646](https://github.com/NousResearch/hermes-agent/pull/26646))
- **Richer info panels on the Skills Hub** ([#22905](https://github.com/NousResearch/hermes-agent/pull/22905))
- **Refuse `skill_view` name collisions instead of guessing** (closes #6136 @polkn)

### Curator
- **Show rename map in user-visible summary** ([#22910](https://github.com/NousResearch/hermes-agent/pull/22910))
- **Hint at `hermes curator pin` in the rename block** ([#23212](https://github.com/NousResearch/hermes-agent/pull/23212))

### New optional skills
- **Hyperliquid** — perp/spot trading via SDK + REST (salvage of #1952) ([#23583](https://github.com/NousResearch/hermes-agent/pull/23583))
- **Yahoo Finance** market data ([#23590](https://github.com/NousResearch/hermes-agent/pull/23590))
- **api-testing** (REST/GraphQL debug, salvages #1800) ([#23582](https://github.com/NousResearch/hermes-agent/pull/23582))
- **Unified EVM multi-chain skill** (salvages #25291 + #2010 + folds in base/) ([#25299](https://github.com/NousResearch/hermes-agent/pull/25299))
- **darwinian-evolver** ([#26760](https://github.com/NousResearch/hermes-agent/pull/26760))
- **osint-investigation** (closes #355) ([#26729](https://github.com/NousResearch/hermes-agent/pull/26729))
- **pinggy-tunnel** ([#26765](https://github.com/NousResearch/hermes-agent/pull/26765))
- **watchers** — RSS / HTTP JSON / GitHub polling via cron no-agent ([#21881](https://github.com/NousResearch/hermes-agent/pull/21881))
- **Notion overhaul for the Developer Platform** (May 2026) ([#26612](https://github.com/NousResearch/hermes-agent/pull/26612))

---

## 🔒 Security & Reliability

### Security hardening
- **Sudo brute-force block + sudo-stdin/askpass DANGEROUS** (salvage of #22194 + #21128) (@kshitijk4poor) ([#23736](https://github.com/NousResearch/hermes-agent/pull/23736))
- **Drop caller-controlled author override in `kanban_comment`** (salvage of #22109) (@kshitijk4poor) ([#22435](https://github.com/NousResearch/hermes-agent/pull/22435))
- **Cover remaining SSRF fetch paths in skills-hub** (salvage #22804) ([#22843](https://github.com/NousResearch/hermes-agent/pull/22843))
- **Use credential_pool for custom endpoint model listing probes** (salvage #22810) ([#22842](https://github.com/NousResearch/hermes-agent/pull/22842))
- **Require dashboard auth for plugin API routes** (salvage #19541) ([#23220](https://github.com/NousResearch/hermes-agent/pull/23220))
- **Sanitize env and redact output in quick commands + remove write-only `_pending_messages`** ([#23584](https://github.com/NousResearch/hermes-agent/pull/23584))
- **Reduce unnecessary `shell=True` in subprocess calls** ([#25149](https://github.com/NousResearch/hermes-agent/pull/25149))
- **Sanitize Google Chat sender_type from relay** (salvage of #22107) (@kshitijk4poor) ([#22432](https://github.com/NousResearch/hermes-agent/pull/22432))
- **Supply-chain advisory checker** ([#24220](https://github.com/NousResearch/hermes-agent/pull/24220))
- **Rewrite security policy around OS-level isolation as the boundary** (@jquesnelle) ([#20317](https://github.com/NousResearch/hermes-agent/pull/20317))
- **Remove public security advisory page** ([#24253](https://github.com/NousResearch/hermes-agent/pull/24253))

### Reliability — notable bug closures
- **SQLite: fall back to `journal_mode=DELETE` on NFS/SMB/FUSE** (fixes `/resume` on network mounts) (@kshitijk4poor) ([#22043](https://github.com/NousResearch/hermes-agent/pull/22043))
- **Codex-runtime: retire wedged sessions + post-tool watchdog + OAuth refresh classify** ([#25769](https://github.com/NousResearch/hermes-agent/pull/25769))
- **Codex-runtime: de-dup `[plugins.X]` tables and stop leaking HERMES_HOME** (#26250) (@kshitijk4poor) ([#26260](https://github.com/NousResearch/hermes-agent/pull/26260))
- **Daytona: migrate legacy-sandbox lookup to cursor-based `list()`** ([#24587](https://github.com/NousResearch/hermes-agent/pull/24587))
- **MCP: stop retrying initial MCP auth failures** (#25624) ([#25776](https://github.com/NousResearch/hermes-agent/pull/25776))
- **Gateway: enable text-intercept for multi-choice clarify fallback** (#25587) ([#25778](https://github.com/NousResearch/hermes-agent/pull/25778))
- **Gateway: keep running when platforms fail; per-platform circuit breaker + `/platform`** ([#26600](https://github.com/NousResearch/hermes-agent/pull/26600))
- **Delegate: salvage #21933 JSON-string batch + diagnostic logging** (@kshitijk4poor) ([#22436](https://github.com/NousResearch/hermes-agent/pull/22436))
- **Profiles+banner: exclude infrastructure from `--clone-all` + fix stale update-check repo resolution** (@kshitijk4poor) ([#22475](https://github.com/NousResearch/hermes-agent/pull/22475))
- **ACP: inline file attachment resources** (salvage #21400 + image support) ([#21407](https://github.com/NousResearch/hermes-agent/pull/21407))
- **CI: unblock shared PR checks** (@stephenschoettler) ([#21012](https://github.com/NousResearch/hermes-agent/pull/21012), [#25957](https://github.com/NousResearch/hermes-agent/pull/25957))

### Notable reverts in window
- **`/goal` checklist + /subgoal feature stack** — rolled back ([#23813](https://github.com/NousResearch/hermes-agent/pull/23813)); `/subgoal` returned in simpler form via [#25449](https://github.com/NousResearch/hermes-agent/pull/25449)
- **Scrollback box width clamp** (#25975) rolled back to restore full-width borders ([#26163](https://github.com/NousResearch/hermes-agent/pull/26163))
- **`fix(cli): tolerate unreadable dirs when building systemd PATH`** rolled back

---

## 🌍 i18n

- **Localize all gateway commands + web dashboard, add 8 new locales (16 total)** ([#22914](https://github.com/NousResearch/hermes-agent/pull/22914))

---

## 📚 Documentation

- **Repair Voice & TTS provider table** (@nightcityblade, fixes #24101) ([#24138](https://github.com/NousResearch/hermes-agent/pull/24138))
- **Show per-skill pages in the left sidebar** ([#26646](https://github.com/NousResearch/hermes-agent/pull/26646))
- **Mention Weixin in gateway help and docstrings** (salvage of #21063 by @wuwuzhijing)
- **Richer info panels on the Skills Hub** ([#22905](https://github.com/NousResearch/hermes-agent/pull/22905))
- Many more doc updates across providers, platforms, skills, Windows install paths, and dashboard.

---

## 🧪 Testing & CI

- **Unblock shared PR checks** (@stephenschoettler) ([#21012](https://github.com/NousResearch/hermes-agent/pull/21012))
- **Stabilize shared test state after 21012** (@stephenschoettler) ([#25957](https://github.com/NousResearch/hermes-agent/pull/25957))
- A long tail of test additions for platforms, providers, plugins, and edge cases — 8 explicit `test:` PRs plus ~250 fix PRs that also added regression coverage.

---

## 👥 Contributors

### Core
- @teknium1 — release lead, architecture, ~406 PRs merged in window

### Top community contributors
- **@kshitijk4poor** — 38 PRs · Telegram cadence/streaming/topic routing, security hardening (sudo, SSRF, kanban_comment, dashboard auth), codex-runtime hygiene, NovitaAI provider, profile/banner fixes, Feishu update cards, gateway QOL across the board
- **@alt-glitch** — 13 PRs · Markdown-table TUI rendering, `HERMES_SESSION_ID` env var, hindsight-client optional dep, Nix `extraDependencyGroups`
- **@OutThisLife** (Brooklyn Nicholson) — 12 PRs · TUI turn segmentation, attach-to-gateway, markdown link titles, embedded TUI via dashboard gateway, Ink cursor sync, scroll/Esc during prompts
- **@austinpickett** — 8 PRs · `/sessions` slash command, personality switching preserves session, cron modals, dashboard analytics
- **@helix4u** — 5 PRs · Google Chat setup, browser install skip on system chromium, Windows Ctrl+C preservation
- **@rob-maron** — 4 PRs · Nous Portal as model metadata authority, provider polish
- **@stephenschoettler** — 3 PRs · CI stabilization
- **@ethernet8023** — 3 PRs · platform/gateway work

### All contributors (alphabetical)

@02356abc, @0xbyt4, @0xharryriddle, @1000Delta, @1RB, @29206394, @A-kamal, @aashizpoudel, @Abd0r,
@adybag14-cyber, @AgentArcLab, @ahmedbadr3, @AhmetArif0, @alblez, @Alex-yang00, @ALIYILD, @AllynSheep,
@alt-glitch, @am423, @amathxbt, @amethystani, @ArecaNon, @Arkmusn, @askclaw-vesper, @AsoTora, @austinpickett,
@aydnOktay, @ayushere, @baocin, @Bartok9, @benbarclay, @BennetYrWang, @Bihruze, @binhnt92, @briandevans,
@brooklynnicholson, @btorresgil, @buntingszn, @CalmProton, @chrisworksai, @CoinTheHat, @dandacompany, @Dangooy,
@DanielLSM, @David-0x221Eight, @ddupont808, @dhruv-saxena, @diablozzc, @dlkakbs, @dmahan93, @dmnkhorvath,
@domtriola, @donrhmexe, @Dusk1e, @eloklam, @emozilla, @ephron-ren, @erenkarakus, @EthanGuo-coder,
@ethernet8023, @evgyur, @explainanalyze, @fahdad, @fr33d3m0n, @Freeman-Consulting, @freqyfreqy, @Frowtek,
@fu576, @github-actions[bot], @gnanirahulnutakki, @GodsBoy, @guglielmofonda, @Gutslabs, @hanzckernel,
@heathley, @hekaru-agent, @helix4u, @HenkDz, @HiddenPuppy, @hllqkb, @hrygo, @HuangYuChuh, @Hugo-SEQUIER, @HxT9,
@iacker, @InB4DevOps, @isaachuangGMICLOUD, @iuyup, @Jaaneek, @jackey8616, @jackjin1997, @Jaggia, @jak983464779,
@jelrod27, @jethac, @JithendraNara, @johnisag, @Julientalbot, @Jwd-gity, @kallidean, @keyuyuan, @kfa-ai,
@kidonng, @KiraKatana, @kjames2001, @konsisumer, @Korkyzer, @kshitijk4poor, @KvnGz, @lars-hagen, @leehack,
@leepoweii, @LeonSGP43, @li0near, @libo1106, @liquidchen, @littlewwwhite, @liuhao1024, @liyoungc, @luandiasrj,
@luoyuctl, @luyao618, @magic524, @mbac, @McClean, @memosr, @Mibayy, @ming1523, @mizgyo, @mrshu, @ms-alan,
@MustafaKara7, @nederev, @nicoechaniz, @nidhi-singh02, @nightcityblade, @nik1t7n, @Ninso112, @NivOO5,
@novax635, @nv-kasikritc, @oferlaor, @oswaldb22, @outdoorsea, @oxngon, @PaTTeeL, @pearjelly, @pefontana,
@perng, @PhilipAD, @phuongvm, @polkn, @Prasanna28Devadiga, @princepal9120, @pty819, @purzbeats, @Quarkex,
@quocanh261997, @qWaitCrypto, @Qwinty, @rahimsais, @raymaylee, @ReqX, @rewbs, @RhombusMaximus, @rob-maron,
@Ruzzgar, @ryptotalent, @Sanjays2402, @shannonsands, @shaun0927, @SiliconID, @silv-mt-holdings, @simpolism,
@smwbev, @soichiyo, @sprmn24, @steezkelly, @stephenschoettler, @Sylw3ster, @szymonclawd, @teyrebaz33,
@Tianyu199509, @Tranquil-Flow, @TreyDong, @TurgutKural, @tw2818, @tymrtn, @uzunkuyruk, @v1b3coder,
@vanthinh6886, @VinceZcrikl, @vKongv, @vominh1919, @voteblake, @VTRiot, @wali-reheman, @wesleysimplicio,
@wilsen0, @WorldWriter, @worlldz, @wuli666, @wuwuzhijing, @Wysie, @XiaoXiao0221, @xieNniu, @xxxigm, @yehuosi,
@ygd58, @yifengingit, @yuga-hashimoto, @zccyman, @ZeterMordio, @Zhekinmaksim, @zhengyn0001

Also: @Nagatha (Claude Opus 4.7).

---

**Full Changelog**: [v2026.5.7...v2026.5.16](https://github.com/NousResearch/hermes-agent/compare/v2026.5.7...v2026.5.16)
