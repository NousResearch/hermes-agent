# Hermes Agent v0.4.0 (v2026.3.23)

**Release Date:** March 23, 2026

> The platform expansion release — OpenAI-compatible API server, 6 new messaging adapters, 4 new inference providers, MCP server management with OAuth 2.1, @ context references, gateway prompt caching, streaming enabled by default, and a sweeping reliability pass with 200+ bug fixes.

---

## ✨ Highlights

- **OpenAI-compatible API server** — Expose Hermes as an `/v1/chat/completions` endpoint with a new `/api/jobs` REST API for cron job management, hardened with input limits, field whitelists, SQLite-backed response persistence, and CORS origin protection ([#1756](https://github.com/NousResearch/hermes-agent/pull/1756), [#2450](https://github.com/NousResearch/hermes-agent/pull/2450), [#2456](https://github.com/NousResearch/hermes-agent/pull/2456), [#2451](https://github.com/NousResearch/hermes-agent/pull/2451), [#2472](https://github.com/NousResearch/hermes-agent/pull/2472))

- **6 new messaging platform adapters** — Signal, DingTalk, SMS (Twilio), Mattermost, Matrix, and Webhook adapters join Telegram, Discord, and WhatsApp. Gateway auto-reconnects failed platforms with exponential backoff ([#2206](https://github.com/NousResearch/hermes-agent/pull/2206), [#1685](https://github.com/NousResearch/hermes-agent/pull/1685), [#1688](https://github.com/NousResearch/hermes-agent/pull/1688), [#1683](https://github.com/NousResearch/hermes-agent/pull/1683), [#2166](https://github.com/NousResearch/hermes-agent/pull/2166), [#2584](https://github.com/NousResearch/hermes-agent/pull/2584))

- **@ context references** — Claude Code-style `@file` and `@url` context injection with tab completions in the CLI ([#2343](https://github.com/NousResearch/hermes-agent/pull/2343), [#2482](https://github.com/NousResearch/hermes-agent/pull/2482))

- **4 new inference providers** — GitHub Copilot (OAuth + token validation), Alibaba Cloud / DashScope, Kilo Code, and OpenCode Zen/Go ([#1924](https://github.com/NousResearch/hermes-agent/pull/1924), [#1879](https://github.com/NousResearch/hermes-agent/pull/1879) by @mchzimm, [#1673](https://github.com/NousResearch/hermes-agent/pull/1673), [#1666](https://github.com/NousResearch/hermes-agent/pull/1666), [#1650](https://github.com/NousResearch/hermes-agent/pull/1650))

- **MCP server management CLI** — `hermes mcp` commands for installing, configuring, and authenticating MCP servers with full OAuth 2.1 PKCE flow ([#2465](https://github.com/NousResearch/hermes-agent/pull/2465))

- **Gateway prompt caching** — Cache AIAgent instances per session, preserving Anthropic prompt cache across turns for dramatic cost reduction on long conversations ([#2282](https://github.com/NousResearch/hermes-agent/pull/2282), [#2284](https://github.com/NousResearch/hermes-agent/pull/2284), [#2361](https://github.com/NousResearch/hermes-agent/pull/2361))

- **Context compression overhaul** — Structured summaries with iterative updates, token-budget tail protection, configurable summary endpoint, and fallback model support ([#2323](https://github.com/NousResearch/hermes-agent/pull/2323), [#1727](https://github.com/NousResearch/hermes-agent/pull/1727), [#2224](https://github.com/NousResearch/hermes-agent/pull/2224))

- **Streaming enabled by default** — CLI streaming on by default with proper spinner/tool progress display during streaming mode, plus extensive linebreak and concatenation fixes ([#2340](https://github.com/NousResearch/hermes-agent/pull/2340), [#2161](https://github.com/NousResearch/hermes-agent/pull/2161), [#2258](https://github.com/NousResearch/hermes-agent/pull/2258))

---

## 🖥️ CLI & User Experience

### New Commands & Interactions
- **@ context completions** — Tab-completable `@file`/`@url` references that inject file content or web pages into the conversation ([#2482](https://github.com/NousResearch/hermes-agent/pull/2482), [#2343](https://github.com/NousResearch/hermes-agent/pull/2343))
- **`/statusbar`** — Toggle a persistent config bar showing model + provider info in the prompt ([#2240](https://github.com/NousResearch/hermes-agent/pull/2240), [#1917](https://github.com/NousResearch/hermes-agent/pull/1917))
- **`/queue`** — Queue prompts for the agent without interrupting the current run ([#2191](https://github.com/NousResearch/hermes-agent/pull/2191), [#2469](https://github.com/NousResearch/hermes-agent/pull/2469))
- **`/permission`** — Switch approval mode dynamically during a session ([#2207](https://github.com/NousResearch/hermes-agent/pull/2207))
- **`/browser`** — Interactive browser sessions from the CLI ([#2273](https://github.com/NousResearch/hermes-agent/pull/2273), [#1814](https://github.com/NousResearch/hermes-agent/pull/1814))
- **`/cost`** — Live pricing and usage tracking in gateway mode ([#2180](https://github.com/NousResearch/hermes-agent/pull/2180))
- **`/approve` and `/deny`** — Replaced bare text approval in gateway with explicit commands ([#2002](https://github.com/NousResearch/hermes-agent/pull/2002))

### Streaming & Display
- Streaming enabled by default in CLI ([#2340](https://github.com/NousResearch/hermes-agent/pull/2340))
- Show spinners and tool progress during streaming mode ([#2161](https://github.com/NousResearch/hermes-agent/pull/2161))
- Show reasoning/thinking blocks when `show_reasoning` enabled ([#2118](https://github.com/NousResearch/hermes-agent/pull/2118))
- Context pressure warnings for CLI and gateway ([#2159](https://github.com/NousResearch/hermes-agent/pull/2159))
- Fix: streaming chunks concatenated without whitespace ([#2258](https://github.com/NousResearch/hermes-agent/pull/2258))
- Fix: iteration boundary linebreak prevents stream concatenation ([#2413](https://github.com/NousResearch/hermes-agent/pull/2413))
- Fix: defer streaming linebreak to prevent blank line stacking ([#2473](https://github.com/NousResearch/hermes-agent/pull/2473))
- Fix: suppress spinner animation in non-TTY environments ([#2216](https://github.com/NousResearch/hermes-agent/pull/2216))
- Fix: display provider and endpoint in API error messages ([#2266](https://github.com/NousResearch/hermes-agent/pull/2266))
- Fix: resolve garbled ANSI escape codes in status printouts ([#2448](https://github.com/NousResearch/hermes-agent/pull/2448))
- Fix: update gold ANSI color to true-color format ([#2246](https://github.com/NousResearch/hermes-agent/pull/2246))
- Fix: normalize toolset labels and use skin colors in banner ([#1912](https://github.com/NousResearch/hermes-agent/pull/1912))

### CLI Polish
- Fix: prevent 'Press ENTER to continue...' on exit ([#2555](https://github.com/NousResearch/hermes-agent/pull/2555))
- Fix: flush stdout during agent loop to prevent macOS display freeze ([#1654](https://github.com/NousResearch/hermes-agent/pull/1654))
- Fix: show human-readable error when `hermes setup` hits permissions error ([#2196](https://github.com/NousResearch/hermes-agent/pull/2196))
- Fix: `/stop` command crash + UnboundLocalError in streaming media delivery ([#2463](https://github.com/NousResearch/hermes-agent/pull/2463))
- Fix: allow custom/local endpoints without API key ([#2556](https://github.com/NousResearch/hermes-agent/pull/2556))
- Fix: Kitty keyboard protocol Shift+Enter for Ghostty/WezTerm (attempted + reverted due to prompt_toolkit crash) ([#2345](https://github.com/NousResearch/hermes-agent/pull/2345), [#2349](https://github.com/NousResearch/hermes-agent/pull/2349))

### Configuration
- **`${ENV_VAR}` substitution** in config.yaml ([#2684](https://github.com/NousResearch/hermes-agent/pull/2684))
- **Real-time config reload** — config.yaml changes apply without restart ([#2210](https://github.com/NousResearch/hermes-agent/pull/2210))
- **`custom_models.yaml`** for user-managed model additions ([#2214](https://github.com/NousResearch/hermes-agent/pull/2214))
- **Priority-based context file selection** + CLAUDE.md support ([#2301](https://github.com/NousResearch/hermes-agent/pull/2301))
- **Merge nested YAML sections** instead of replacing on config update ([#2213](https://github.com/NousResearch/hermes-agent/pull/2213))
- Fix: config.yaml provider key overrides env var silently ([#2272](https://github.com/NousResearch/hermes-agent/pull/2272))
- Fix: log warning instead of silently swallowing config.yaml errors ([#2683](https://github.com/NousResearch/hermes-agent/pull/2683))
- Fix: disabled toolsets re-enable themselves after `hermes tools` ([#2268](https://github.com/NousResearch/hermes-agent/pull/2268))
- Fix: platform default toolsets silently override tool deselection ([#2624](https://github.com/NousResearch/hermes-agent/pull/2624))
- Fix: honor bare YAML `approvals.mode: off` ([#2620](https://github.com/NousResearch/hermes-agent/pull/2620))
- Fix: `hermes update` use `.[all]` extras with fallback ([#1728](https://github.com/NousResearch/hermes-agent/pull/1728))
- Fix: `hermes update` prompt before resetting working tree on stash conflicts ([#2390](https://github.com/NousResearch/hermes-agent/pull/2390))
- Fix: use git pull --rebase in update/install to avoid divergent branch error ([#2274](https://github.com/NousResearch/hermes-agent/pull/2274))
- Fix: add zprofile fallback and create zshrc on fresh macOS installs ([#2320](https://github.com/NousResearch/hermes-agent/pull/2320))
- Fix: remove `ANTHROPIC_BASE_URL` env var to avoid collisions ([#1675](https://github.com/NousResearch/hermes-agent/pull/1675))
- Fix: don't ask IMAP password if already in keyring or env ([#2212](https://github.com/NousResearch/hermes-agent/pull/2212))
- Fix: OpenCode Zen/Go show OpenRouter models instead of their own ([#2277](https://github.com/NousResearch/hermes-agent/pull/2277))

---

## 🏗️ Core Agent & Architecture

### New Providers
- **GitHub Copilot** — Full OAuth auth, API routing, token validation, and 400k context. ([#1924](https://github.com/NousResearch/hermes-agent/pull/1924), [#1896](https://github.com/NousResearch/hermes-agent/pull/1896), [#1879](https://github.com/NousResearch/hermes-agent/pull/1879) by @mchzimm, [#2507](https://github.com/NousResearch/hermes-agent/pull/2507))
- **Alibaba Cloud / DashScope** — Full integration with DashScope v1 runtime, model dot preservation, and 401 auth fixes ([#1673](https://github.com/NousResearch/hermes-agent/pull/1673), [#2332](https://github.com/NousResearch/hermes-agent/pull/2332), [#2459](https://github.com/NousResearch/hermes-agent/pull/2459))
- **Kilo Code** — First-class inference provider ([#1666](https://github.com/NousResearch/hermes-agent/pull/1666))
- **OpenCode Zen and OpenCode Go** — New provider backends ([#1650](https://github.com/NousResearch/hermes-agent/pull/1650), [#2393](https://github.com/NousResearch/hermes-agent/pull/2393) by @0xbyt4)
- **NeuTTS** — Local TTS provider backend with built-in setup flow, replacing the old optional skill ([#1657](https://github.com/NousResearch/hermes-agent/pull/1657), [#1664](https://github.com/NousResearch/hermes-agent/pull/1664))

### Provider Improvements
- **Eager fallback** to backup model on rate-limit errors ([#1730](https://github.com/NousResearch/hermes-agent/pull/1730))
- **Endpoint metadata** for custom model context and pricing; query local servers for actual context window size ([#1906](https://github.com/NousResearch/hermes-agent/pull/1906), [#2091](https://github.com/NousResearch/hermes-agent/pull/2091) by @dusterbloom)
- **Context length detection overhaul** — models.dev integration, provider-aware resolution, fuzzy matching for custom endpoints, `/v1/props` for llama.cpp ([#2158](https://github.com/NousResearch/hermes-agent/pull/2158), [#2051](https://github.com/NousResearch/hermes-agent/pull/2051), [#2403](https://github.com/NousResearch/hermes-agent/pull/2403))
- **Model catalog updates** — gpt-5.4-mini, gpt-5.4-nano, healer-alpha, haiku-4.5, minimax-m2.7, claude 4.6 at 1M context ([#1913](https://github.com/NousResearch/hermes-agent/pull/1913), [#1915](https://github.com/NousResearch/hermes-agent/pull/1915), [#1900](https://github.com/NousResearch/hermes-agent/pull/1900), [#2155](https://github.com/NousResearch/hermes-agent/pull/2155), [#2474](https://github.com/NousResearch/hermes-agent/pull/2474))
- **Custom endpoint improvements** — `model.base_url` in config.yaml, `api_mode` override for responses API, allow endpoints without API key, fail fast on missing keys ([#2330](https://github.com/NousResearch/hermes-agent/pull/2330), [#1651](https://github.com/NousResearch/hermes-agent/pull/1651), [#2556](https://github.com/NousResearch/hermes-agent/pull/2556), [#2445](https://github.com/NousResearch/hermes-agent/pull/2445), [#1994](https://github.com/NousResearch/hermes-agent/pull/1994), [#1998](https://github.com/NousResearch/hermes-agent/pull/1998))
- Inject model and provider into system prompt ([#1929](https://github.com/NousResearch/hermes-agent/pull/1929))
- Tie `api_mode` to provider config instead of env var ([#1656](https://github.com/NousResearch/hermes-agent/pull/1656))
- Fix: prevent Anthropic token leaking to third-party `anthropic_messages` providers ([#2389](https://github.com/NousResearch/hermes-agent/pull/2389))
- Fix: prevent Anthropic fallback from inheriting non-Anthropic `base_url` ([#2388](https://github.com/NousResearch/hermes-agent/pull/2388))
- Fix: `auxiliary_is_nous` flag never resets — leaked Nous tags to other providers ([#1713](https://github.com/NousResearch/hermes-agent/pull/1713))
- Fix: Anthropic `tool_choice 'none'` still allowed tool calls ([#1714](https://github.com/NousResearch/hermes-agent/pull/1714))
- Fix: Mistral parser nested JSON fallback extraction ([#2335](https://github.com/NousResearch/hermes-agent/pull/2335))
- Fix: MiniMax 401 auth resolved by defaulting to `anthropic_messages` ([#2103](https://github.com/NousResearch/hermes-agent/pull/2103))
- Fix: case-insensitive model family matching ([#2350](https://github.com/NousResearch/hermes-agent/pull/2350))
- Fix: ignore placeholder provider keys in activation checks ([#2358](https://github.com/NousResearch/hermes-agent/pull/2358))
- Fix: Preserve Ollama model:tag colons in context length detection ([#2149](https://github.com/NousResearch/hermes-agent/pull/2149))
- Fix: recognize Claude Code OAuth credentials in startup gate ([#1663](https://github.com/NousResearch/hermes-agent/pull/1663))
- Fix: detect Claude Code version dynamically for OAuth user-agent ([#1670](https://github.com/NousResearch/hermes-agent/pull/1670))
- Fix: OAuth flag stale after refresh/fallback ([#1890](https://github.com/NousResearch/hermes-agent/pull/1890))
- Fix: auxiliary client skips expired Codex JWT ([#2397](https://github.com/NousResearch/hermes-agent/pull/2397))

### Agent Loop
- **Gateway prompt caching** — Cache AIAgent per session, keep assistant turns, fix session restore ([#2282](https://github.com/NousResearch/hermes-agent/pull/2282), [#2284](https://github.com/NousResearch/hermes-agent/pull/2284), [#2361](https://github.com/NousResearch/hermes-agent/pull/2361))
- **Context compression overhaul** — Structured summaries, iterative updates, token-budget tail protection, configurable `summary_base_url` ([#2323](https://github.com/NousResearch/hermes-agent/pull/2323), [#1727](https://github.com/NousResearch/hermes-agent/pull/1727), [#2224](https://github.com/NousResearch/hermes-agent/pull/2224))
- **Pre-call sanitization and post-call tool guardrails** ([#1732](https://github.com/NousResearch/hermes-agent/pull/1732))
- **Auto-recover** from provider-rejected `tool_choice` by retrying without ([#2174](https://github.com/NousResearch/hermes-agent/pull/2174))
- **Background memory/skill review** replaces inline nudges ([#2235](https://github.com/NousResearch/hermes-agent/pull/2235))
- **SOUL.md as primary agent identity** instead of hardcoded default ([#1922](https://github.com/NousResearch/hermes-agent/pull/1922))
- Fix: prevent silent tool result loss during context compression ([#1993](https://github.com/NousResearch/hermes-agent/pull/1993))
- Fix: handle empty/null function arguments in tool call recovery ([#2163](https://github.com/NousResearch/hermes-agent/pull/2163))
- Fix: handle API refusal responses gracefully instead of crashing ([#2156](https://github.com/NousResearch/hermes-agent/pull/2156))
- Fix: prevent stuck agent loop on malformed tool calls ([#2114](https://github.com/NousResearch/hermes-agent/pull/2114))
- Fix: return JSON parse error to model instead of dispatching with empty args ([#2342](https://github.com/NousResearch/hermes-agent/pull/2342))
- Fix: consecutive assistant message merge drops content on mixed types ([#1703](https://github.com/NousResearch/hermes-agent/pull/1703))
- Fix: message role alternation violations in JSON recovery and error handler ([#1722](https://github.com/NousResearch/hermes-agent/pull/1722))
- Fix: `compression_attempts` resets each iteration — allowed unlimited compressions ([#1723](https://github.com/NousResearch/hermes-agent/pull/1723))
- Fix: `length_continue_retries` never resets — later truncations got fewer retries ([#1717](https://github.com/NousResearch/hermes-agent/pull/1717))
- Fix: compressor summary role violated consecutive-role constraint ([#1720](https://github.com/NousResearch/hermes-agent/pull/1720), [#1743](https://github.com/NousResearch/hermes-agent/pull/1743))
- Fix: remove hardcoded `gemini-3-flash-preview` as default summary model ([#2464](https://github.com/NousResearch/hermes-agent/pull/2464))
- Fix: correctly handle empty tool results ([#2201](https://github.com/NousResearch/hermes-agent/pull/2201))
- Fix: crash on None entry in `tool_calls` list ([#2209](https://github.com/NousResearch/hermes-agent/pull/2209) by @0xbyt4, [#2316](https://github.com/NousResearch/hermes-agent/pull/2316))
- Fix: per-thread persistent event loops in worker threads ([#2214](https://github.com/NousResearch/hermes-agent/pull/2214) by @jquesnelle)
- Fix: prevent 'event loop already running' when async tools run in parallel ([#2207](https://github.com/NousResearch/hermes-agent/pull/2207))
- Fix: strip ANSI at the source — clean terminal output before it reaches the model ([#2115](https://github.com/NousResearch/hermes-agent/pull/2115))
- Fix: skip top-level `cache_control` on role:tool for OpenRouter ([#2391](https://github.com/NousResearch/hermes-agent/pull/2391))
- Fix: delegate tool — save parent tool names before child construction mutates global ([#2083](https://github.com/NousResearch/hermes-agent/pull/2083) by @ygd58, [#1894](https://github.com/NousResearch/hermes-agent/pull/1894))
- Fix: only strip last assistant message if empty string ([#2326](https://github.com/NousResearch/hermes-agent/pull/2326))

### Session & Memory
- **Session search** and management slash commands ([#2198](https://github.com/NousResearch/hermes-agent/pull/2198))
- **Auto session titles** and `.hermes.md` project config ([#1712](https://github.com/NousResearch/hermes-agent/pull/1712))
- Fix: concurrent memory writes silently drop entries — added file locking ([#1726](https://github.com/NousResearch/hermes-agent/pull/1726))
- Fix: search all sources by default in `session_search` ([#1892](https://github.com/NousResearch/hermes-agent/pull/1892))
- Fix: handle hyphenated FTS5 queries and preserve quoted literals ([#1776](https://github.com/NousResearch/hermes-agent/pull/1776))
- Fix: skip corrupt lines in `load_transcript` instead of crashing ([#1744](https://github.com/NousResearch/hermes-agent/pull/1744))
- Fix: normalize session keys to prevent case-sensitive duplicates ([#2157](https://github.com/NousResearch/hermes-agent/pull/2157))
- Fix: prevent `session_search` crash when no sessions exist ([#2194](https://github.com/NousResearch/hermes-agent/pull/2194))
- Fix: reset token counters on new session for accurate usage display ([#2101](https://github.com/NousResearch/hermes-agent/pull/2101) by @InB4DevOps)
- Fix: prevent stale memory overwrites by flush agent ([#2687](https://github.com/NousResearch/hermes-agent/pull/2687))
- Fix: remove synthetic error message injection, fix session resume after repeated failures ([#2303](https://github.com/NousResearch/hermes-agent/pull/2303))
- Fix: quiet mode with `--resume` now passes conversation_history ([#2357](https://github.com/NousResearch/hermes-agent/pull/2357))
- Fix: unify resume logic in batch mode ([#2331](https://github.com/NousResearch/hermes-agent/pull/2331))

### Honcho Memory
- Honcho config fixes and @ context reference integration ([#2343](https://github.com/NousResearch/hermes-agent/pull/2343))
- Self-hosted / Docker configuration documentation ([#2475](https://github.com/NousResearch/hermes-agent/pull/2475))

---

## 📱 Messaging Platforms (Gateway)

### New Platform Adapters
- **Signal Messenger** — Full adapter with attachment handling, group message filtering, and Note to Self echo-back protection ([#2206](https://github.com/NousResearch/hermes-agent/pull/2206), [#2400](https://github.com/NousResearch/hermes-agent/pull/2400), [#2297](https://github.com/NousResearch/hermes-agent/pull/2297), [#2156](https://github.com/NousResearch/hermes-agent/pull/2156))
- **DingTalk** — Adapter with gateway wiring and setup docs ([#1685](https://github.com/NousResearch/hermes-agent/pull/1685), [#1690](https://github.com/NousResearch/hermes-agent/pull/1690), [#1692](https://github.com/NousResearch/hermes-agent/pull/1692))
- **SMS (Twilio)** ([#1688](https://github.com/NousResearch/hermes-agent/pull/1688))
- **Mattermost** — With @-mention-only channel filter ([#1683](https://github.com/NousResearch/hermes-agent/pull/1683), [#2443](https://github.com/NousResearch/hermes-agent/pull/2443))
- **Matrix** — With vision support and image caching ([#1683](https://github.com/NousResearch/hermes-agent/pull/1683), [#2520](https://github.com/NousResearch/hermes-agent/pull/2520))
- **Webhook** — Platform adapter for external event triggers ([#2166](https://github.com/NousResearch/hermes-agent/pull/2166))
- **OpenAI-compatible API server** — `/v1/chat/completions` endpoint with `/api/jobs` cron management ([#1756](https://github.com/NousResearch/hermes-agent/pull/1756), [#2450](https://github.com/NousResearch/hermes-agent/pull/2450), [#2456](https://github.com/NousResearch/hermes-agent/pull/2456))

### Telegram Improvements
- MarkdownV2 support — strikethrough, spoiler, blockquotes, escape parentheses/braces/backslashes/backticks ([#2199](https://github.com/NousResearch/hermes-agent/pull/2199), [#2200](https://github.com/NousResearch/hermes-agent/pull/2200) by @llbn, [#2386](https://github.com/NousResearch/hermes-agent/pull/2386))
- Auto-detect HTML tags and use `parse_mode=HTML` ([#1709](https://github.com/NousResearch/hermes-agent/pull/1709))
- Telegram group vision support + thread-based sessions ([#2153](https://github.com/NousResearch/hermes-agent/pull/2153))
- Auto-reconnect polling after network interruption ([#2517](https://github.com/NousResearch/hermes-agent/pull/2517))
- Aggregate split text messages before dispatching ([#1674](https://github.com/NousResearch/hermes-agent/pull/1674))
- Fix: streaming config bridge, not-modified, flood control ([#1782](https://github.com/NousResearch/hermes-agent/pull/1782), [#1783](https://github.com/NousResearch/hermes-agent/pull/1783))
- Fix: edited_message event crashes ([#2074](https://github.com/NousResearch/hermes-agent/pull/2074))
- Fix: retry 409 polling conflicts before giving up ([#2312](https://github.com/NousResearch/hermes-agent/pull/2312))
- Fix: topic delivery via `platform:chat_id:thread_id` format ([#2455](https://github.com/NousResearch/hermes-agent/pull/2455))

### Discord Improvements
- Document caching and text-file injection ([#2503](https://github.com/NousResearch/hermes-agent/pull/2503))
- Persistent typing indicator for DMs ([#2468](https://github.com/NousResearch/hermes-agent/pull/2468))
- Discord DM vision — inline images + attachment analysis ([#2186](https://github.com/NousResearch/hermes-agent/pull/2186))
- Persist thread participation across gateway restarts ([#1661](https://github.com/NousResearch/hermes-agent/pull/1661))
- Fix: gateway crash on non-ASCII guild names ([#2302](https://github.com/NousResearch/hermes-agent/pull/2302))
- Fix: thread permission errors ([#2073](https://github.com/NousResearch/hermes-agent/pull/2073))
- Fix: slash event routing in threads ([#2460](https://github.com/NousResearch/hermes-agent/pull/2460))
- Fix: remove bugged followup messages + `/ask` command ([#1836](https://github.com/NousResearch/hermes-agent/pull/1836))
- Fix: graceful WebSocket reconnection ([#2127](https://github.com/NousResearch/hermes-agent/pull/2127))
- Fix: voice channel TTS when streaming enabled ([#2322](https://github.com/NousResearch/hermes-agent/pull/2322))

### WhatsApp & Other Adapters
- WhatsApp: outbound `send_message` routing ([#1769](https://github.com/NousResearch/hermes-agent/pull/1769) by @sai-samarth), LID format self-chat ([#1667](https://github.com/NousResearch/hermes-agent/pull/1667)), `reply_prefix` config fix ([#1923](https://github.com/NousResearch/hermes-agent/pull/1923)), restart on bridge child exit ([#2334](https://github.com/NousResearch/hermes-agent/pull/2334)), image/bridge improvements ([#2181](https://github.com/NousResearch/hermes-agent/pull/2181))
- Matrix: correct `reply_to_message_id` parameter ([#1895](https://github.com/NousResearch/hermes-agent/pull/1895)), bare media types fix ([#1736](https://github.com/NousResearch/hermes-agent/pull/1736))
- Mattermost: MIME types for media attachments ([#2329](https://github.com/NousResearch/hermes-agent/pull/2329))

### Gateway Core
- **Auto-reconnect** failed platforms with exponential backoff ([#2584](https://github.com/NousResearch/hermes-agent/pull/2584))
- **Notify users when session auto-resets** ([#2519](https://github.com/NousResearch/hermes-agent/pull/2519))
- **Reply-to message context** for out-of-session replies ([#1662](https://github.com/NousResearch/hermes-agent/pull/1662))
- **Ignore unauthorized DMs** config option ([#1919](https://github.com/NousResearch/hermes-agent/pull/1919))
- Fix: `/reset` in thread-mode resets global session instead of thread ([#2254](https://github.com/NousResearch/hermes-agent/pull/2254))
- Fix: deliver MEDIA: files after streaming responses ([#2382](https://github.com/NousResearch/hermes-agent/pull/2382))
- Fix: cap interrupt recursion depth to prevent resource exhaustion ([#1659](https://github.com/NousResearch/hermes-agent/pull/1659))
- Fix: detect stopped processes and release stale locks on `--replace` ([#2406](https://github.com/NousResearch/hermes-agent/pull/2406), [#1908](https://github.com/NousResearch/hermes-agent/pull/1908))
- Fix: PID-based wait with force-kill for gateway restart ([#1902](https://github.com/NousResearch/hermes-agent/pull/1902))
- Fix: prevent `--replace` mode from killing the caller process ([#2185](https://github.com/NousResearch/hermes-agent/pull/2185))
- Fix: `/model` shows active fallback model instead of config default ([#1660](https://github.com/NousResearch/hermes-agent/pull/1660))
- Fix: `/title` command fails when session doesn't exist in SQLite yet ([#2379](https://github.com/NousResearch/hermes-agent/pull/2379) by @ten-jampa)
- Fix: process `/queue`'d messages after agent completion ([#2469](https://github.com/NousResearch/hermes-agent/pull/2469))
- Fix: strip orphaned `tool_results` + let `/reset` bypass running agent ([#2180](https://github.com/NousResearch/hermes-agent/pull/2180))
- Fix: prevent agents from starting gateway outside systemd management ([#2617](https://github.com/NousResearch/hermes-agent/pull/2617))
- Fix: prevent systemd restart storm on gateway connection failure ([#2327](https://github.com/NousResearch/hermes-agent/pull/2327))
- Fix: include resolved node path in systemd unit ([#1767](https://github.com/NousResearch/hermes-agent/pull/1767) by @sai-samarth)
- Fix: send error details to user in gateway outer exception handler ([#1966](https://github.com/NousResearch/hermes-agent/pull/1966))
- Fix: improve error handling for 429 usage limits and 500 context overflow ([#1839](https://github.com/NousResearch/hermes-agent/pull/1839))
- Fix: add all missing platform allowlist env vars to startup warning check ([#2628](https://github.com/NousResearch/hermes-agent/pull/2628))
- Fix: media delivery fails for file paths containing spaces ([#2621](https://github.com/NousResearch/hermes-agent/pull/2621))
- Fix: duplicate session-key collision in multi-platform gateway ([#2171](https://github.com/NousResearch/hermes-agent/pull/2171))
- Fix: Matrix and Mattermost never report as connected ([#1711](https://github.com/NousResearch/hermes-agent/pull/1711))
- Fix: PII redaction config never read — missing yaml import ([#1701](https://github.com/NousResearch/hermes-agent/pull/1701))
- Fix: NameError on skill slash commands ([#1697](https://github.com/NousResearch/hermes-agent/pull/1697))
- Fix: persist watcher metadata in checkpoint for crash recovery ([#1706](https://github.com/NousResearch/hermes-agent/pull/1706))
- Fix: pass `message_thread_id` in send_image_file, send_document, send_video ([#2339](https://github.com/NousResearch/hermes-agent/pull/2339))
- Fix: media-group aggregation on rapid successive photo messages ([#2160](https://github.com/NousResearch/hermes-agent/pull/2160))

---

## 🔧 Tool System

### MCP Enhancements
- **MCP server management CLI** + OAuth 2.1 PKCE auth ([#2465](https://github.com/NousResearch/hermes-agent/pull/2465))
- **Expose MCP servers as standalone toolsets** ([#1907](https://github.com/NousResearch/hermes-agent/pull/1907))
- **Interactive MCP tool configuration** in `hermes tools` ([#1694](https://github.com/NousResearch/hermes-agent/pull/1694))
- Fix: MCP-OAuth port mismatch, path traversal, and shared handler state ([#2552](https://github.com/NousResearch/hermes-agent/pull/2552))
- Fix: preserve MCP tool registrations across session resets ([#2124](https://github.com/NousResearch/hermes-agent/pull/2124))
- Fix: concurrent file access crash + duplicate MCP registration ([#2154](https://github.com/NousResearch/hermes-agent/pull/2154))
- Fix: normalise MCP schemas + expand session list columns ([#2102](https://github.com/NousResearch/hermes-agent/pull/2102))
- Fix: `tool_choice` `mcp_` prefix handling ([#1775](https://github.com/NousResearch/hermes-agent/pull/1775))

### Web Tool Backends
- **Tavily** as web search/extract/crawl backend ([#1731](https://github.com/NousResearch/hermes-agent/pull/1731))
- **Parallel** as alternative web search/extract backend ([#1696](https://github.com/NousResearch/hermes-agent/pull/1696))
- **Configurable web backend** — Firecrawl/BeautifulSoup/Playwright selection ([#2256](https://github.com/NousResearch/hermes-agent/pull/2256))
- Fix: whitespace-only env vars bypass web backend detection ([#2341](https://github.com/NousResearch/hermes-agent/pull/2341))

### New Tools
- **IMAP email** reading and sending ([#2173](https://github.com/NousResearch/hermes-agent/pull/2173))
- **STT (speech-to-text)** tool using Whisper API ([#2072](https://github.com/NousResearch/hermes-agent/pull/2072))
- **Route-aware pricing estimates** ([#1695](https://github.com/NousResearch/hermes-agent/pull/1695))

### Tool Improvements
- TTS: `base_url` support for OpenAI TTS provider ([#2064](https://github.com/NousResearch/hermes-agent/pull/2064) by @hanai)
- Vision: configurable timeout, tilde expansion in file paths, DM vision with multi-image and base64 fallback ([#2480](https://github.com/NousResearch/hermes-agent/pull/2480), [#2585](https://github.com/NousResearch/hermes-agent/pull/2585), [#2211](https://github.com/NousResearch/hermes-agent/pull/2211))
- Browser: race condition fix in session creation ([#1721](https://github.com/NousResearch/hermes-agent/pull/1721)), TypeError on unexpected LLM params ([#1735](https://github.com/NousResearch/hermes-agent/pull/1735))
- File tools: strip ANSI escape codes from write_file and patch content ([#2532](https://github.com/NousResearch/hermes-agent/pull/2532)), include pagination args in repeated search key ([#1824](https://github.com/NousResearch/hermes-agent/pull/1824) by @cutepawss), improve fuzzy matching accuracy + position calculation refactor ([#2096](https://github.com/NousResearch/hermes-agent/pull/2096), [#1681](https://github.com/NousResearch/hermes-agent/pull/1681))
- Code execution: resource leak and double socket close fix ([#2381](https://github.com/NousResearch/hermes-agent/pull/2381))
- Delegate: thread safety for concurrent subagent delegation ([#1672](https://github.com/NousResearch/hermes-agent/pull/1672)), preserve parent agent's tool list after delegation ([#1778](https://github.com/NousResearch/hermes-agent/pull/1778))
- Fix: make concurrent tool batching path-aware for file mutations ([#1914](https://github.com/NousResearch/hermes-agent/pull/1914))
- Fix: chunk long messages in `send_message_tool` before platform dispatch ([#1646](https://github.com/NousResearch/hermes-agent/pull/1646))
- Fix: add missing 'messaging' toolset ([#1718](https://github.com/NousResearch/hermes-agent/pull/1718))
- Fix: prevent unavailable tool names from leaking into model schemas ([#2072](https://github.com/NousResearch/hermes-agent/pull/2072))
- Fix: pass visited set by reference to prevent diamond dependency duplication ([#2311](https://github.com/NousResearch/hermes-agent/pull/2311))
- Fix: Daytona sandbox lookup migrated from `find_one` to `get/list` ([#2063](https://github.com/NousResearch/hermes-agent/pull/2063) by @rovle)

---

## 🧩 Skills Ecosystem

### Skills System Improvements
- **Agent-created skills** — Caution-level findings allowed, dangerous skills ask instead of block ([#1840](https://github.com/NousResearch/hermes-agent/pull/1840), [#2446](https://github.com/NousResearch/hermes-agent/pull/2446))
- **`--yes` flag** to bypass confirmation in `/skills install` and uninstall ([#1647](https://github.com/NousResearch/hermes-agent/pull/1647))
- **Disabled skills respected** across banner, system prompt, and slash commands ([#1897](https://github.com/NousResearch/hermes-agent/pull/1897))
- Fix: skills custom_tools import crash + sandbox file_tools integration ([#2239](https://github.com/NousResearch/hermes-agent/pull/2239))
- Fix: agent-created skills with pip requirements crash on install ([#2145](https://github.com/NousResearch/hermes-agent/pull/2145))
- Fix: race condition in `Skills.__init__` when `hub.yaml` missing ([#2242](https://github.com/NousResearch/hermes-agent/pull/2242))
- Fix: validate skill metadata before install and block duplicates ([#2241](https://github.com/NousResearch/hermes-agent/pull/2241))
- Fix: skills hub inspect/resolve — 4 bugs in inspect, redirects, discovery, tap list ([#2447](https://github.com/NousResearch/hermes-agent/pull/2447))
- Fix: agent-created skills keep working after session reset ([#2121](https://github.com/NousResearch/hermes-agent/pull/2121))

### New Skills
- **OCR-and-documents** — PDF/DOCX/XLS/PPTX/image OCR with optional GPU ([#2236](https://github.com/NousResearch/hermes-agent/pull/2236), [#2461](https://github.com/NousResearch/hermes-agent/pull/2461))
- **Huggingface-hub** bundled skill ([#1921](https://github.com/NousResearch/hermes-agent/pull/1921))
- **Sherlock OSINT** username search ([#1671](https://github.com/NousResearch/hermes-agent/pull/1671))
- **Meme-generation** — Image generator with Pillow ([#2344](https://github.com/NousResearch/hermes-agent/pull/2344))
- **Bioinformatics** gateway skill — index to 400+ bio skills ([#2387](https://github.com/NousResearch/hermes-agent/pull/2387))
- **Inference.sh** skill (terminal-based) ([#1686](https://github.com/NousResearch/hermes-agent/pull/1686))
- **Base blockchain** optional skill ([#1643](https://github.com/NousResearch/hermes-agent/pull/1643))
- **3D-model-viewer** optional skill ([#2226](https://github.com/NousResearch/hermes-agent/pull/2226))
- **FastMCP** optional skill ([#2113](https://github.com/NousResearch/hermes-agent/pull/2113))
- **Hermes-agent-setup** skill ([#1905](https://github.com/NousResearch/hermes-agent/pull/1905))

---

## 🔌 Plugin System Enhancements

- **TUI extension hooks** — Build custom CLIs on top of Hermes ([#2333](https://github.com/NousResearch/hermes-agent/pull/2333))
- **`hermes plugins install/remove/list`** commands ([#2337](https://github.com/NousResearch/hermes-agent/pull/2337))
- **Slash command registration** for plugins ([#2359](https://github.com/NousResearch/hermes-agent/pull/2359))
- **`session:end` lifecycle event** hook ([#1725](https://github.com/NousResearch/hermes-agent/pull/1725))
- Fix: require opt-in for project plugin discovery ([#2215](https://github.com/NousResearch/hermes-agent/pull/2215))

---

## 🔒 Security & Reliability

### Security
- **SSRF protection** for vision_tools and web_tools ([#2679](https://github.com/NousResearch/hermes-agent/pull/2679))
- **Shell injection prevention** in `_expand_path` via `~user` path suffix ([#2685](https://github.com/NousResearch/hermes-agent/pull/2685))
- **Block untrusted browser-origin** API server access ([#2451](https://github.com/NousResearch/hermes-agent/pull/2451))
- **Block sandbox backend creds** from subprocess env ([#1658](https://github.com/NousResearch/hermes-agent/pull/1658))
- **Block @ references** from reading secrets outside workspace ([#2601](https://github.com/NousResearch/hermes-agent/pull/2601) by @Gutslabs)
- **Malicious code pattern pre-exec scanner** for terminal_tool ([#2245](https://github.com/NousResearch/hermes-agent/pull/2245))
- **Harden terminal safety** and sandbox file writes ([#1653](https://github.com/NousResearch/hermes-agent/pull/1653))
- **PKCE verifier leak** fix + OAuth refresh Content-Type ([#1775](https://github.com/NousResearch/hermes-agent/pull/1775))
- **Eliminate SQL string formatting** in `execute()` calls ([#2061](https://github.com/NousResearch/hermes-agent/pull/2061) by @dusterbloom)
- **Harden jobs API** — input limits, field whitelist, startup check ([#2456](https://github.com/NousResearch/hermes-agent/pull/2456))

### Reliability
- Thread locks on 4 SessionDB methods ([#1704](https://github.com/NousResearch/hermes-agent/pull/1704))
- File locking for concurrent memory writes ([#1726](https://github.com/NousResearch/hermes-agent/pull/1726))
- Handle OpenRouter errors gracefully ([#2112](https://github.com/NousResearch/hermes-agent/pull/2112))
- Guard print() calls against OSError ([#1668](https://github.com/NousResearch/hermes-agent/pull/1668))
- Safely handle non-string inputs in redacting formatter ([#2392](https://github.com/NousResearch/hermes-agent/pull/2392), [#1700](https://github.com/NousResearch/hermes-agent/pull/1700))
- ACP: preserve session provider on model switch, persist sessions to disk ([#2380](https://github.com/NousResearch/hermes-agent/pull/2380), [#2071](https://github.com/NousResearch/hermes-agent/pull/2071))
- API server: persist ResponseStore to SQLite across restarts ([#2472](https://github.com/NousResearch/hermes-agent/pull/2472))
- Fix: `fetch_nous_models` always TypeError from positional args ([#1699](https://github.com/NousResearch/hermes-agent/pull/1699))
- Fix: resolve merge conflict markers in cli.py breaking startup ([#2347](https://github.com/NousResearch/hermes-agent/pull/2347))
- Fix: `minisweagent_path.py` missing from wheel ([#2098](https://github.com/NousResearch/hermes-agent/pull/2098) by @JiwaniZakir)

### Cron System
- **`[SILENT]` response** — cron agents can suppress delivery ([#1833](https://github.com/NousResearch/hermes-agent/pull/1833))
- **Scale missed-job grace window** with schedule frequency ([#2449](https://github.com/NousResearch/hermes-agent/pull/2449))
- **Recover recent one-shot jobs** ([#1918](https://github.com/NousResearch/hermes-agent/pull/1918))
- Fix: normalize `repeat<=0` to None — jobs deleted after first run when LLM passes -1 ([#2612](https://github.com/NousResearch/hermes-agent/pull/2612) by @Mibayy)
- Fix: Matrix added to scheduler delivery platform_map ([#2167](https://github.com/NousResearch/hermes-agent/pull/2167) by @buntingszn)
- Fix: naive ISO timestamps without timezone — jobs fire at wrong time ([#1729](https://github.com/NousResearch/hermes-agent/pull/1729))
- Fix: `get_due_jobs` reads `jobs.json` twice — race condition ([#1716](https://github.com/NousResearch/hermes-agent/pull/1716))
- Fix: silent jobs return empty response for delivery skip ([#2442](https://github.com/NousResearch/hermes-agent/pull/2442))
- Fix: stop injecting cron outputs into gateway session history ([#2313](https://github.com/NousResearch/hermes-agent/pull/2313))
- Fix: close abandoned coroutine when `asyncio.run()` raises RuntimeError ([#2317](https://github.com/NousResearch/hermes-agent/pull/2317))

---

## 🧪 Testing

- Resolve all consistently failing tests ([#2488](https://github.com/NousResearch/hermes-agent/pull/2488))
- Replace `FakePath` with `monkeypatch` for Python 3.12 compat ([#2444](https://github.com/NousResearch/hermes-agent/pull/2444))
- Align Hermes setup and full-suite expectations ([#1710](https://github.com/NousResearch/hermes-agent/pull/1710))

---

## 📚 Documentation

- Comprehensive docs update for recent features ([#1693](https://github.com/NousResearch/hermes-agent/pull/1693), [#2183](https://github.com/NousResearch/hermes-agent/pull/2183))
- Alibaba Cloud and DingTalk setup guides ([#1687](https://github.com/NousResearch/hermes-agent/pull/1687), [#1692](https://github.com/NousResearch/hermes-agent/pull/1692))
- Detailed skills documentation ([#2244](https://github.com/NousResearch/hermes-agent/pull/2244))
- Honcho self-hosted / Docker configuration ([#2475](https://github.com/NousResearch/hermes-agent/pull/2475))
- Context length detection FAQ and quickstart references ([#2179](https://github.com/NousResearch/hermes-agent/pull/2179))
- Fix docs inconsistencies across reference and user guides ([#1995](https://github.com/NousResearch/hermes-agent/pull/1995))
- Fix MCP install commands — use uv, not bare pip ([#1909](https://github.com/NousResearch/hermes-agent/pull/1909))
- Replace ASCII diagrams with Mermaid/lists ([#2402](https://github.com/NousResearch/hermes-agent/pull/2402))
- Gemini OAuth provider implementation plan ([#2467](https://github.com/NousResearch/hermes-agent/pull/2467))
- Discord Server Members Intent marked as required ([#2330](https://github.com/NousResearch/hermes-agent/pull/2330))
- Fix MDX build error in api-server.md ([#1787](https://github.com/NousResearch/hermes-agent/pull/1787))
- Align venv path to match installer ([#2114](https://github.com/NousResearch/hermes-agent/pull/2114))
- New skills added to hub index ([#2281](https://github.com/NousResearch/hermes-agent/pull/2281))

---

## 👥 Contributors

### Core
- **@teknium1** (Teknium) — 280 PRs

### Community Contributors
- **@mchzimm** (to_the_max) — GitHub Copilot provider integration ([#1879](https://github.com/NousResearch/hermes-agent/pull/1879))
- **@jquesnelle** (Jeffrey Quesnelle) — Per-thread persistent event loops fix ([#2214](https://github.com/NousResearch/hermes-agent/pull/2214))
- **@llbn** (lbn) — Telegram MarkdownV2 strikethrough, spoiler, blockquotes, and escape fixes ([#2199](https://github.com/NousResearch/hermes-agent/pull/2199), [#2200](https://github.com/NousResearch/hermes-agent/pull/2200))
- **@dusterbloom** — SQL injection prevention + local server context window querying ([#2061](https://github.com/NousResearch/hermes-agent/pull/2061), [#2091](https://github.com/NousResearch/hermes-agent/pull/2091))
- **@0xbyt4** — Anthropic tool_calls None guard + OpenCode-Go provider config fix ([#2209](https://github.com/NousResearch/hermes-agent/pull/2209), [#2393](https://github.com/NousResearch/hermes-agent/pull/2393))
- **@sai-samarth** (Saisamarth) — WhatsApp send_message routing + systemd node path ([#1769](https://github.com/NousResearch/hermes-agent/pull/1769), [#1767](https://github.com/NousResearch/hermes-agent/pull/1767))
- **@Gutslabs** (Guts) — Block @ references from reading secrets ([#2601](https://github.com/NousResearch/hermes-agent/pull/2601))
- **@Mibayy** (Mibay) — Cron job repeat normalization ([#2612](https://github.com/NousResearch/hermes-agent/pull/2612))
- **@ten-jampa** (Tenzin Jampa) — Gateway /title command fix ([#2379](https://github.com/NousResearch/hermes-agent/pull/2379))
- **@cutepawss** (lila) — File tools search pagination fix ([#1824](https://github.com/NousResearch/hermes-agent/pull/1824))
- **@hanai** (Hanai) — OpenAI TTS base_url support ([#2064](https://github.com/NousResearch/hermes-agent/pull/2064))
- **@rovle** (Lovre Pešut) — Daytona sandbox API migration ([#2063](https://github.com/NousResearch/hermes-agent/pull/2063))
- **@buntingszn** (bunting szn) — Matrix cron delivery support ([#2167](https://github.com/NousResearch/hermes-agent/pull/2167))
- **@InB4DevOps** — Token counter reset on new session ([#2101](https://github.com/NousResearch/hermes-agent/pull/2101))
- **@JiwaniZakir** (Zakir Jiwani) — Missing file in wheel fix ([#2098](https://github.com/NousResearch/hermes-agent/pull/2098))
- **@ygd58** (buray) — Delegate tool parent tool names fix ([#2083](https://github.com/NousResearch/hermes-agent/pull/2083))

---

**Full Changelog**: [v2026.3.17...v2026.3.23](https://github.com/NousResearch/hermes-agent/compare/v2026.3.17...v2026.3.23)
