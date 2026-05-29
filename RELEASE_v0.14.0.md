# Hermes Agent v0.14.0 (v2026.5.16)

**Release Date:** May 16, 2026
**Since v0.13.0:** 808 commits · 633 merged PRs · 1393 files changed · 165,061 insertions · 545 issues closed (12 P0, 50 P1) · 215 community contributors (including co-authors)

> The Foundation Release — Hermes installs and runs anywhere, ships with the things you actually want to use, and stops shipping the things you don't. xAI Grok lands as a SuperGrok OAuth provider with grok-4.3 bumped to a 1M context window. A new OpenAI-compatible local proxy turns any OAuth-authed Hermes provider — Claude Pro, ChatGPT Pro, SuperGrok — into an endpoint that Codex / Aider / Cline / Continue can hit. `x_search` lands as a first-class X (Twitter) search tool with OAuth-or-API-key auth. The Microsoft Teams stack is wired end-to-end (Graph auth + webhook listener + pipeline runtime + outbound delivery). A debloating wave makes installs dramatically lighter — heavyweight backends now lazy-install on first use, the `[all]` extras drop everything covered by lazy-deps, and a tiered install falls back when a wheel rejects on your platform. `pip install hermes-agent` works from PyPI. The cold-start wave shaves ~19 seconds off `hermes` launch. Browser CDP calls are 180x faster. Two new messaging platforms (LINE + SimpleX Chat) bring the total to 22. Cross-session 1-hour Claude prompt caching, `/handoff` that actually transfers sessions live, native button UI for `clarify` on Telegram and Discord, Discord channel history backfill, LSP semantic diagnostics on every write, a unified pluggable `video_generate`, a `computer_use` cua-driver backend that finally works with non-Anthropic providers, clickable URLs in any terminal, Zed ACP Registry integration via `uvx`, native Windows beta, 9 new optional skills, OpenRouter Pareto Code router, huggingface/skills as a trusted default tap. 12 P0 + 50 P1 closures.

---

## ✨ Highlights

- **xAI Grok via SuperGrok OAuth — and grok-4.3 jumps to a 1M context window** — If you pay for SuperGrok, you can now use Grok inside Hermes by signing in with your xAI account — no API key, no separate billing. The wire-through also bumps grok-4.3 to a 1M token context window, so you can drop whole codebases or research corpora into a single prompt. Includes proper handling for entitlement errors and an SSH-to-tunnel docs page for when you're SSH'd into a remote box and need to complete the OAuth flow. ([#26534](https://github.com/NousResearch/hermes-agent/pull/26534), [#26664](https://github.com/NousResearch/hermes-agent/pull/26664), [#26644](https://github.com/NousResearch/hermes-agent/pull/26644), [#26592](https://github.com/NousResearch/hermes-agent/pull/26592))

- **OpenAI-compatible local proxy for OAuth providers** — Run `hermes proxy` and you get a `http://localhost:port` endpoint that speaks the OpenAI API but is backed by whichever OAuth provider you're signed into — Claude Pro, ChatGPT Pro, SuperGrok. Now any tool that expects an OpenAI-compatible endpoint (Codex CLI, Aider, Cline, Continue, your custom scripts) just works with your existing subscription, no API key required. One subscription, every tool. ([#25969](https://github.com/NousResearch/hermes-agent/pull/25969))

- **`x_search` — first-class X (Twitter) search tool** — The agent can now search X directly without installing a skill or wiring up a custom integration. Search the timeline, find threads, surface specific posts — straight from the chat. Auth with either your X OAuth login or an API key, whichever you have. ([#26763](https://github.com/NousResearch/hermes-agent/pull/26763))

- **Microsoft Teams — end-to-end** — Hermes can now read messages from Teams and post back. The full Microsoft Graph stack lands together: auth + client foundation, a webhook listener that receives Teams events, a pipeline plugin runtime, and outbound delivery. Wire up the bot once, then chat to your agent from any Teams channel, DM, or group. (salvages of #21408–#21411) ([#21922](https://github.com/NousResearch/hermes-agent/pull/21922), [#21969](https://github.com/NousResearch/hermes-agent/pull/21969), [#22007](https://github.com/NousResearch/hermes-agent/pull/22007), [#22024](https://github.com/NousResearch/hermes-agent/pull/22024))

- **Debloating wave — lighter installs, less you don't use** — A clean `pip install hermes-agent` used to pull down everything: every messaging adapter SDK, every image-gen SDK, every voice/TTS provider, whether you used them or not. Now those heavy backends (Slack / Matrix / Feishu / DingTalk adapters, hindsight client, codex app-server, Pixverse / Camofox / image-gen SDKs, voice/TTS providers) install automatically the first time you actually use them. The `[all]` extras drop everything covered by lazy-deps, the installer falls back through tiers when a wheel doesn't fit your platform, and a supply-chain advisory checker scans every install for unsafe versions. Faster installs, smaller disk footprint, fewer transitive vulnerabilities. ([#24220](https://github.com/NousResearch/hermes-agent/pull/24220), [#24515](https://github.com/NousResearch/hermes-agent/pull/24515), [#25014](https://github.com/NousResearch/hermes-agent/pull/25014), [#25038](https://github.com/NousResearch/hermes-agent/pull/25038), [#25766](https://github.com/NousResearch/hermes-agent/pull/25766), [#21818](https://github.com/NousResearch/hermes-agent/pull/21818))

- **`pip install hermes-agent && hermes`** — Hermes Agent is now a real PyPI package. No more cloning the repo or running shell installers — one pip command and you're running. The wheel ships with the Ink TUI bundle and the shell launcher, so the full experience comes out of the box. (salvage of [#26350](https://github.com/NousResearch/hermes-agent/pull/26350)) ([#26593](https://github.com/NousResearch/hermes-agent/pull/26593), [#26148](https://github.com/NousResearch/hermes-agent/pull/26148))

- **Cross-session 1h Claude prompt cache** — When you use Claude through Anthropic, OpenRouter, or Nous Portal, the prompt prefix (system prompt, skills, memory) now caches for an hour across sessions. Start a `/new` session and the first response comes back faster and cheaper because the cache is still warm from your last session. Background memory review hits the cache too, so it's not paying full price every turn. ([#23828](https://github.com/NousResearch/hermes-agent/pull/23828), [#25434](https://github.com/NousResearch/hermes-agent/pull/25434), [#24778](https://github.com/NousResearch/hermes-agent/pull/24778))

- **180x faster `browser_console` evaluations** — When the agent uses the browser tool to inspect a page or run JavaScript, those calls now share one persistent connection to Chrome instead of spinning up a new DevTools session every time. The difference is huge: things that used to take a couple of seconds per call return in milliseconds. Real-world page interactions feel instant. ([#23226](https://github.com/NousResearch/hermes-agent/pull/23226))

- **Cold-start performance wave — ~19 seconds off `hermes` launch** — Running `hermes` used to make you wait through a chunk of import overhead and network calls before you saw a prompt. Now the launch path is mostly deferred: heavy adapters only load when you use them, model catalogs come from disk cache first, doctor checks run in parallel, and `chat -q` skips the welcome banner entirely. The `hermes tools` All-Platforms screen alone dropped from 14 seconds to under 1.5 seconds. ([#22138](https://github.com/NousResearch/hermes-agent/pull/22138), [#22120](https://github.com/NousResearch/hermes-agent/pull/22120), [#22681](https://github.com/NousResearch/hermes-agent/pull/22681), [#22790](https://github.com/NousResearch/hermes-agent/pull/22790), [#22808](https://github.com/NousResearch/hermes-agent/pull/22808), [#22831](https://github.com/NousResearch/hermes-agent/pull/22831), [#22859](https://github.com/NousResearch/hermes-agent/pull/22859), [#22904](https://github.com/NousResearch/hermes-agent/pull/22904), [#22766](https://github.com/NousResearch/hermes-agent/pull/22766), [#25341](https://github.com/NousResearch/hermes-agent/pull/25341))

- **Two new messaging platforms — LINE + SimpleX Chat** — LINE is huge in Japan, Korea, and Taiwan, and now Hermes runs natively on the LINE Messaging API. SimpleX Chat is the privacy-focused decentralized messenger with no user IDs — also wired up as a first-class platform. That brings Hermes to 22 messaging platforms total, so wherever you and your team chat, the agent can be there. ([#23197](https://github.com/NousResearch/hermes-agent/pull/23197), [#26232](https://github.com/NousResearch/hermes-agent/pull/26232))

- **`/handoff` actually transfers the session live** — Switching models or personalities mid-conversation used to mean losing context or starting over. Now `/handoff` moves your active session — every message, every tool call, every piece of context — to the target model, persona, or profile, live, without dropping anything. Mid-debugging hand off from a fast model to a deep-reasoning one, or pass a session between profiles for different parts of a task. ([#23395](https://github.com/NousResearch/hermes-agent/pull/23395))

- **Native button UI for `clarify` on Telegram and Discord** — When the agent uses the `clarify` tool to ask you a multiple-choice question, it now shows real platform-native buttons on Telegram and Discord instead of asking you to type back the option number. Tap the button, the agent gets your answer. Especially nice on mobile. ([#24199](https://github.com/NousResearch/hermes-agent/pull/24199), [#25485](https://github.com/NousResearch/hermes-agent/pull/25485))

- **Discord channel history backfill (default on)** — When Hermes joins a Discord channel or thread for the first time, it now reads the recent message history so it knows what's been said before it responds. No more "what are we talking about?" — the agent has the context that's already on screen for everyone else. ([#25984](https://github.com/NousResearch/hermes-agent/pull/25984))

- **`vision_analyze` returns pixels to vision-capable models** — When you point the agent at an image with `vision_analyze` and the active model can actually see (GPT-5, Claude, Gemini, Grok-vision), Hermes now passes the raw pixels straight to the model instead of converting them to a text description first. You get the model's actual visual reasoning instead of a degraded text-summary round-trip. ([#22955](https://github.com/NousResearch/hermes-agent/pull/22955))

- **Per-turn file-mutation verifier footer** — After every turn that wrote or edited files, the agent now gets a short footer summarizing exactly what changed on disk — the file paths, the line counts, the actual delta. That means the agent catches its own mistakes when a write didn't land or got silently overwritten, instead of confidently telling you "I added the function" when the file wasn't actually saved. ([#24498](https://github.com/NousResearch/hermes-agent/pull/24498))

- **LSP semantic diagnostics on every write** — When the agent uses `write_file` or `patch`, Hermes now runs a real language server against the edited file and surfaces any new errors back to the agent before the next turn. Type errors, undefined symbols, missing imports — caught immediately. Goes way beyond v0.13.0's basic Python/JSON/YAML/TOML linting because it's actual semantic analysis. ([#24168](https://github.com/NousResearch/hermes-agent/pull/24168), [#25978](https://github.com/NousResearch/hermes-agent/pull/25978))

- **Unified `video_generate` with pluggable provider backends** — One tool, any video model. Hermes ships with the obvious backends already, but you can drop in a new video provider as a plugin without touching core. So when a new video model lands next month, it can be a one-file plugin instead of a fork. ([#25126](https://github.com/NousResearch/hermes-agent/pull/25126))

- **`computer_use` cua-driver backend — works with non-Anthropic models now** — Computer-use (the agent controlling your mouse and keyboard to drive GUI apps) used to be locked to Anthropic's SDK. The new cua-driver backend works with non-Anthropic providers too, has proper focus-safe operations, and refreshes itself on `hermes update`. Now any vision-capable model can drive your desktop. (re-salvage of #16936) ([#21967](https://github.com/NousResearch/hermes-agent/pull/21967), [#24063](https://github.com/NousResearch/hermes-agent/pull/24063))

- **Clickable URLs in any terminal** — Links in agent output are now real OSC8 hyperlinks with hover-highlight in any terminal that supports them. Click to open in your browser — no more copy-paste-trim of long URLs from the transcript. Just works in iTerm2, Kitty, Ghostty, modern Windows Terminal, etc. (@OutThisLife) ([#25071](https://github.com/NousResearch/hermes-agent/pull/25071), [#24013](https://github.com/NousResearch/hermes-agent/pull/24013))

- **Zed ACP Registry — `uvx` install in one click** — Hermes is now listed in Zed's Agent Client Protocol registry, so Zed users can install it with one click. The install path uses `uvx` so there's no npm dependency. `hermes acp --setup-browser` bootstraps the browser tools for registry-driven installs. (salvage of [#25908](https://github.com/NousResearch/hermes-agent/pull/25908)) ([#26079](https://github.com/NousResearch/hermes-agent/pull/26079), [#26120](https://github.com/NousResearch/hermes-agent/pull/26120), [#26234](https://github.com/NousResearch/hermes-agent/pull/26234))

- **OpenRouter Pareto Code router with `min_coding_score` knob** — OpenRouter's "Pareto" router automatically picks the cheapest model that meets a minimum quality bar. The new `min_coding_score` config lets you set that bar for coding tasks specifically — Hermes routes to the most affordable model that's at least that good at code. Stop paying for top-tier models when a mid-tier one would do. ([#22838](https://github.com/NousResearch/hermes-agent/pull/22838))

- **NovitaAI as a new model provider** — NovitaAI joins the provider lineup, giving you another option for open-source model hosting (Llama, Qwen, DeepSeek, etc.) with their pricing and rate limits. (salvage #7219) (@kshitijk4poor) ([#25507](https://github.com/NousResearch/hermes-agent/pull/25507))

- **Codex app-server runtime for OpenAI/Codex models** — An optional runtime that drives OpenAI's Codex CLI under the hood when you're using OpenAI or Codex paths. You get session reuse, automatic retirement of wedged sessions, and proper OAuth refresh classification — the kind of plumbing that makes long agentic runs not fall over. ([#24182](https://github.com/NousResearch/hermes-agent/pull/24182), [#25769](https://github.com/NousResearch/hermes-agent/pull/25769))

- **`huggingface/skills` as a trusted default tap** — The community skills index hosted at huggingface.co/skills is now wired into the Skills Hub by default. So when somebody publishes a useful skill there, you can install it from your own `hermes skills` browser without any extra config. (closes #2549) ([#26219](https://github.com/NousResearch/hermes-agent/pull/26219))

- **9 new optional skills** — Hyperliquid (perp + spot trading via the SDK and REST API), Yahoo Finance (live market data, fundamentals, historicals), api-testing (REST + GraphQL debug recipes), unified EVM multi-chain (one skill covers Ethereum + L2s + Base), darwinian-evolver (evolutionary prompt/skill tuning), osint-investigation (OSINT recipes for people / domains / orgs), pinggy-tunnel (expose local services to the public internet), watchers (polls RSS / HTTP JSON / GitHub via cron `no_agent` mode for change detection), and a full Notion overhaul for the May 2026 Developer Platform. ([#23582](https://github.com/NousResearch/hermes-agent/pull/23582), [#23583](https://github.com/NousResearch/hermes-agent/pull/23583), [#23590](https://github.com/NousResearch/hermes-agent/pull/23590), [#25299](https://github.com/NousResearch/hermes-agent/pull/25299), [#26760](https://github.com/NousResearch/hermes-agent/pull/26760), [#26729](https://github.com/NousResearch/hermes-agent/pull/26729), [#26765](https://github.com/NousResearch/hermes-agent/pull/26765), [#21881](https://github.com/NousResearch/hermes-agent/pull/21881), [#26612](https://github.com/NousResearch/hermes-agent/pull/26612))

- **API server exposes run approval events** — If you're driving Hermes programmatically through the HTTP API, long-running runs no longer silently hang when the agent hits an approval-required command. The approval request now surfaces on the API stream so your client can prompt the user and reply — no more silent stalls. (salvage of [#20311](https://github.com/NousResearch/hermes-agent/pull/20311)) ([#21899](https://github.com/NousResearch/hermes-agent/pull/21899))

- **Plugins can run any LLM call via `ctx.llm` + replace built-in tools via `tool_override`** — If you're writing a Hermes plugin, you now get first-class access to make LLM calls through the active provider and credentials — no manual client wiring. The new `tool_override` flag lets a plugin swap out a built-in tool with its own implementation cleanly. Plugin authors get the same model-routing and auth plumbing the core agent uses. (closes #11049) ([#23194](https://github.com/NousResearch/hermes-agent/pull/23194), [#26759](https://github.com/NousResearch/hermes-agent/pull/26759))

- **Brave Search (free tier) + DuckDuckGo (DDGS) as web-search providers** — Two new free web-search backends join Tavily, SearXNG, and Exa. Brave Search has a generous free tier; DDGS is the DuckDuckGo scraper that needs no key at all. Pick whichever fits your budget and rate-limit needs. ([#21337](https://github.com/NousResearch/hermes-agent/pull/21337))

- **Sudo brute-force block + 3 dangerous-command bypasses closed + tool-error sanitization** — The approval gate now blocks `sudo -S` brute-force attempts and classifies stdin-fed or askpass-stripped sudo invocations as DANGEROUS. Three known bypasses of dangerous-command detection are closed (inspired by Claude Code's command-detection work). And tool error strings are now sanitized before being re-injected into the model context, so a malicious file or remote service can't pass instructions to your agent through error output. ([#23736](https://github.com/NousResearch/hermes-agent/pull/23736), [#26829](https://github.com/NousResearch/hermes-agent/pull/26829), [#26823](https://github.com/NousResearch/hermes-agent/pull/26823))

- **`/subgoal` — user-added criteria appended to an active `/goal`** — When you've got a `/goal` running (the persistent Ralph-loop goal where the agent keeps going until criteria are met), you can now use `/subgoal <text>` to layer extra success criteria onto it mid-run. The judge factors your new criteria into the done-or-keep-going decision without restarting the loop. ([#25449](https://github.com/NousResearch/hermes-agent/pull/25449))

- **Provider rename — Alibaba Cloud → Qwen Cloud** — The Alibaba Cloud provider is renamed to Qwen Cloud in the picker and config to match what the rest of the world calls it. Existing config keys still work — no breaking changes — but the UI matches the actual brand now. ([#24835](https://github.com/NousResearch/hermes-agent/pull/24835))

- **Native Windows support (early beta)** — Hermes now runs natively on `cmd.exe` and PowerShell without WSL. A full PowerShell installer handles MinGit auto-install, Microsoft Store python stub detection, and the foreground Ctrl+C dance. There's still rough edges (this is the "early beta" stamp) — ~40 follow-up Windows-only fixes already landed in the window — but the basic loop works end-to-end on a clean Windows box. ([#21561](https://github.com/NousResearch/hermes-agent/pull/21561))


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
