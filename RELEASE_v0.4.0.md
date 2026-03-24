# Hermes Agent v0.4.0 (v2026.3.23)

**Release Date:** March 23, 2026

> The biggest release yet — 300 merged PRs in one week. Streaming output, native browser tools, Skills Hub, plugin system, 7 new messaging platforms, MCP server management, @ context references, prompt caching, API server, and a sweeping reliability overhaul across every subsystem.

---

## ✨ Highlights

- **Streaming CLI output** — Real-time token streaming enabled by default in CLI mode with proper tool progress spinners during streaming ([#2251](https://github.com/NousResearch/hermes-agent/pull/2251), [#2340](https://github.com/NousResearch/hermes-agent/pull/2340), [#2161](https://github.com/NousResearch/hermes-agent/pull/2161))
- **Native browser tools** — Full Browserbase-powered browser automation: navigate, click, type, screenshot, scrape — plus an interactive `/browser` CLI command ([#2270](https://github.com/NousResearch/hermes-agent/pull/2270), [#2273](https://github.com/NousResearch/hermes-agent/pull/2273))
- **Skills Hub** — Discover, install, and manage skills from curated community taps with `/skills` commands ([#2235](https://github.com/NousResearch/hermes-agent/pull/2235))
- **Plugin system** — TUI extension hooks for building custom CLIs on top of Hermes, plus `hermes plugins install/remove/list` commands and slash command registration for plugins ([#2333](https://github.com/NousResearch/hermes-agent/pull/2333), [#2337](https://github.com/NousResearch/hermes-agent/pull/2337), [#2359](https://github.com/NousResearch/hermes-agent/pull/2359))
- **7 new messaging platforms** — Signal, DingTalk, SMS (Twilio), Mattermost, Matrix, WhatsApp bridge, and Webhook adapters join Telegram and Discord ([#2206](https://github.com/NousResearch/hermes-agent/pull/2206), [#1685](https://github.com/NousResearch/hermes-agent/pull/1685), [#1688](https://github.com/NousResearch/hermes-agent/pull/1688), [#1683](https://github.com/NousResearch/hermes-agent/pull/1683), [#2168](https://github.com/NousResearch/hermes-agent/pull/2168), [#2166](https://github.com/NousResearch/hermes-agent/pull/2166))
- **@ context references** — Claude Code-style `@file` and `@url` context injection with tab completions ([#2343](https://github.com/NousResearch/hermes-agent/pull/2343), [#2482](https://github.com/NousResearch/hermes-agent/pull/2482))
- **OpenAI-compatible API server** — Expose Hermes as an API endpoint with `/api/jobs` for cron management ([#1756](https://github.com/NousResearch/hermes-agent/pull/1756), [#2450](https://github.com/NousResearch/hermes-agent/pull/2450))

---

## 🏗️ Core Agent & Architecture

### Provider & Model Support
- **GitHub Copilot provider** — Full OAuth auth, API routing, token validation, and documentation. Copilot context now correctly resolves to 400k ([#1924](https://github.com/NousResearch/hermes-agent/pull/1924), [#1896](https://github.com/NousResearch/hermes-agent/pull/1896), [#1879](https://github.com/NousResearch/hermes-agent/pull/1879) by @mchzimm, [#2507](https://github.com/NousResearch/hermes-agent/pull/2507))
- **Claude Code OAuth provider** — Anthropic-native API mode with dynamic version detection for OAuth user-agent ([#2199](https://github.com/NousResearch/hermes-agent/pull/2199), [#1663](https://github.com/NousResearch/hermes-agent/pull/1663), [#1670](https://github.com/NousResearch/hermes-agent/pull/1670))
- **Alibaba Cloud / DashScope provider** — Full integration with DashScope v1 runtime mode, model dot preservation, and 401 auth fixes ([#1673](https://github.com/NousResearch/hermes-agent/pull/1673), [#2332](https://github.com/NousResearch/hermes-agent/pull/2332), [#2459](https://github.com/NousResearch/hermes-agent/pull/2459))
- **Kilo Code provider** — Added as first-class inference provider ([#1666](https://github.com/NousResearch/hermes-agent/pull/1666))
- **OpenCode Zen and OpenCode Go providers** — New provider backends with custom endpoint support ([#1650](https://github.com/NousResearch/hermes-agent/pull/1650), [#2393](https://github.com/NousResearch/hermes-agent/pull/2393) by @0xbyt4)
- **Multi-provider architecture** — Automatic fallback, OpenRouter routing backend, Mistral native tool calling, Google Gemini integration ([#2090](https://github.com/NousResearch/hermes-agent/pull/2090), [#2100](https://github.com/NousResearch/hermes-agent/pull/2100), [#2098](https://github.com/NousResearch/hermes-agent/pull/2098), [#2094](https://github.com/NousResearch/hermes-agent/pull/2094), [#2092](https://github.com/NousResearch/hermes-agent/pull/2092))
- **Eager fallback to backup model** on rate-limit errors ([#1730](https://github.com/NousResearch/hermes-agent/pull/1730))
- **Endpoint metadata** for custom model context and pricing; query local servers for actual context window size ([#1906](https://github.com/NousResearch/hermes-agent/pull/1906), [#2091](https://github.com/NousResearch/hermes-agent/pull/2091) by @dusterbloom)
- **Context length detection overhaul** — models.dev integration, provider-aware resolution, fuzzy matching for custom endpoints, `/v1/props` for llama.cpp ([#2158](https://github.com/NousResearch/hermes-agent/pull/2158), [#2051](https://github.com/NousResearch/hermes-agent/pull/2051), [#2403](https://github.com/NousResearch/hermes-agent/pull/2403))
- **Model catalog updates** — gpt-5.4-mini, gpt-5.4-nano, healer-alpha, haiku-4.5, minimax-m2.7, claude 4.6 at 1M context ([#1913](https://github.com/NousResearch/hermes-agent/pull/1913), [#1915](https://github.com/NousResearch/hermes-agent/pull/1915), [#1900](https://github.com/NousResearch/hermes-agent/pull/1900), [#2155](https://github.com/NousResearch/hermes-agent/pull/2155), [#2474](https://github.com/NousResearch/hermes-agent/pull/2474))
- **Custom endpoint improvements** — config.yaml `model.base_url` support, custom endpoints use responses API via `api_mode` override, allow custom/local endpoints without API key, fail fast when explicit provider has no key ([#2330](https://github.com/NousResearch/hermes-agent/pull/2330), [#1651](https://github.com/NousResearch/hermes-agent/pull/1651), [#2556](https://github.com/NousResearch/hermes-agent/pull/2556), [#2445](https://github.com/NousResearch/hermes-agent/pull/2445), [#1994](https://github.com/NousResearch/hermes-agent/pull/1994), [#1998](https://github.com/NousResearch/hermes-agent/pull/1998))
- Inject model and provider into system prompt ([#1929](https://github.com/NousResearch/hermes-agent/pull/1929))
- Fix: prevent Anthropic token leaking to third-party `anthropic_messages` providers ([#2389](https://github.com/NousResearch/hermes-agent/pull/2389))
- Fix: prevent Anthropic fallback from inheriting non-Anthropic `base_url` ([#2388](https://github.com/NousResearch/hermes-agent/pull/2388))
- Fix: `auxiliary_is_nous` flag never resets — leaked Nous tags to other providers ([#1713](https://github.com/NousResearch/hermes-agent/pull/1713))
- Fix: Anthropic `tool_choice 'none'` still allowed tool calls ([#1714](https://github.com/NousResearch/hermes-agent/pull/1714))
- Fix: Mistral parser nested JSON fallback extraction ([#2335](https://github.com/NousResearch/hermes-agent/pull/2335))
- Fix: MiniMax 401 auth error resolved by defaulting to `anthropic_messages` ([#2103](https://github.com/NousResearch/hermes-agent/pull/2103))
- Fix: case-insensitive model family matching ([#2350](https://github.com/NousResearch/hermes-agent/pull/2350))
- Fix: ignore placeholder provider keys in activation checks ([#2358](https://github.com/NousResearch/hermes-agent/pull/2358))
- Fix: Copilot models response decoding and provider bootstrap error logging ([#2202](https://github.com/NousResearch/hermes-agent/pull/2202))
- Fix: Preserve Ollama model:tag colons in context length detection ([#2149](https://github.com/NousResearch/hermes-agent/pull/2149))

### Agent Loop & Conversation
- **Streaming output** — CLI streaming with proper linebreak handling, iteration boundary prevention, and blank line stacking fixes ([#2251](https://github.com/NousResearch/hermes-agent/pull/2251), [#2340](https://github.com/NousResearch/hermes-agent/pull/2340), [#2258](https://github.com/NousResearch/hermes-agent/pull/2258), [#2413](https://github.com/NousResearch/hermes-agent/pull/2413), [#2473](https://github.com/NousResearch/hermes-agent/pull/2473))
- **Context compression overhaul** — Structured summaries, iterative updates, token-budget tail protection, fallback model support ([#2323](https://github.com/NousResearch/hermes-agent/pull/2323), [#2128](https://github.com/NousResearch/hermes-agent/pull/2128), [#2224](https://github.com/NousResearch/hermes-agent/pull/2224), [#1727](https://github.com/NousResearch/hermes-agent/pull/1727))
- **Context pressure warnings** for CLI and gateway ([#2159](https://github.com/NousResearch/hermes-agent/pull/2159))
- **Prompt caching for gateway** — Cache AIAgent per session, keep assistant turns, fix session restore ([#2282](https://github.com/NousResearch/hermes-agent/pull/2282), [#2284](https://github.com/NousResearch/hermes-agent/pull/2284), [#2361](https://github.com/NousResearch/hermes-agent/pull/2361))
- **Show reasoning/thinking blocks** when `show_reasoning` is enabled ([#2118](https://github.com/NousResearch/hermes-agent/pull/2118))
- **Subagent delegation** for parallel task execution with thread safety ([#2119](https://github.com/NousResearch/hermes-agent/pull/2119), [#1672](https://github.com/NousResearch/hermes-agent/pull/1672), [#1778](https://github.com/NousResearch/hermes-agent/pull/1778))
- **Pre-call sanitization and post-call tool guardrails** ([#1732](https://github.com/NousResearch/hermes-agent/pull/1732))
- **Auto-recover** from provider-rejected `tool_choice` by retrying without ([#2174](https://github.com/NousResearch/hermes-agent/pull/2174))
- **Rate limit handling** with exponential backoff retry ([#2071](https://github.com/NousResearch/hermes-agent/pull/2071))
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
- Fix: correctly handle empty tool results ([#2201](https://github.com/NousResearch/hermes-agent/pull/2201))
- Fix: crash on None entry in `tool_calls` list during Anthropic conversion ([#2209](https://github.com/NousResearch/hermes-agent/pull/2209) by @0xbyt4, [#2316](https://github.com/NousResearch/hermes-agent/pull/2316))
- Fix: per-thread persistent event loops in worker threads ([#2214](https://github.com/NousResearch/hermes-agent/pull/2214) by @jquesnelle)
- Fix: prevent 'event loop already running' when async tools run in parallel ([#2207](https://github.com/NousResearch/hermes-agent/pull/2207))
- Fix: strip ANSI escape codes from terminal output before sending to model ([#2115](https://github.com/NousResearch/hermes-agent/pull/2115), [#2585](https://github.com/NousResearch/hermes-agent/pull/2585))
- Fix: skip top-level `cache_control` on role:tool for OpenRouter ([#2391](https://github.com/NousResearch/hermes-agent/pull/2391))
- Fix: delegate tool — save parent tool names before child construction mutates global ([#2083](https://github.com/NousResearch/hermes-agent/pull/2083) by @ygd58, [#1894](https://github.com/NousResearch/hermes-agent/pull/1894))
- Fix: only strip last assistant message if empty string ([#2326](https://github.com/NousResearch/hermes-agent/pull/2326))

### Session & Memory
- **Honcho long-term memory backend** integration ([#2276](https://github.com/NousResearch/hermes-agent/pull/2276))
- **Per-session SQLite persistence** for gateway ([#2134](https://github.com/NousResearch/hermes-agent/pull/2134))
- **`--resume` flag** for CLI session persistence across restarts + `/resume` and `/sessions` commands ([#2135](https://github.com/NousResearch/hermes-agent/pull/2135), [#2143](https://github.com/NousResearch/hermes-agent/pull/2143))
- **Session search** and management slash commands ([#2198](https://github.com/NousResearch/hermes-agent/pull/2198))
- **Auto session titles** and `.hermes.md` project config ([#1712](https://github.com/NousResearch/hermes-agent/pull/1712))
- **Background memory/skill review** replaces inline nudges ([#2235](https://github.com/NousResearch/hermes-agent/pull/2235))
- **SOUL.md** as primary agent identity instead of hardcoded default ([#1922](https://github.com/NousResearch/hermes-agent/pull/1922), [#1927](https://github.com/NousResearch/hermes-agent/pull/1927))
- **Priority-based context file selection** + CLAUDE.md support ([#2301](https://github.com/NousResearch/hermes-agent/pull/2301))
- Fix: concurrent memory writes silently drop entries — added file locking ([#1726](https://github.com/NousResearch/hermes-agent/pull/1726))
- Fix: search all sources by default in `session_search` ([#1892](https://github.com/NousResearch/hermes-agent/pull/1892))
- Fix: handle hyphenated FTS5 queries and preserve quoted literals ([#1776](https://github.com/NousResearch/hermes-agent/pull/1776))
- Fix: skip corrupt lines in `load_transcript` instead of crashing ([#1744](https://github.com/NousResearch/hermes-agent/pull/1744))
- Fix: normalize session keys to prevent case-sensitive duplicates ([#2157](https://github.com/NousResearch/hermes-agent/pull/2157))
- Fix: prevent `session_search` crash when no sessions exist ([#2194](https://github.com/NousResearch/hermes-agent/pull/2194))
- Fix: reset token counters on new session for accurate usage display ([#2101](https://github.com/NousResearch/hermes-agent/pull/2101) by @InB4DevOps)
- Fix: prevent stale memory overwrites by flush agent ([#2687](https://github.com/NousResearch/hermes-agent/pull/2687))
- Fix: remove synthetic error message injection, fix session resume after repeated failures ([#2303](https://github.com/NousResearch/hermes-agent/pull/2303))

---

## 📱 Messaging Platforms (Gateway)

### New Platform Adapters
- **Signal Messenger** adapter ([#2206](https://github.com/NousResearch/hermes-agent/pull/2206)) with attachment handling fix ([#2400](https://github.com/NousResearch/hermes-agent/pull/2400)), group message filtering ([#2297](https://github.com/NousResearch/hermes-agent/pull/2297)), and Note to Self echo-back protection ([#2156](https://github.com/NousResearch/hermes-agent/pull/2156))
- **DingTalk** adapter with gateway wiring and setup docs ([#1685](https://github.com/NousResearch/hermes-agent/pull/1685), [#1690](https://github.com/NousResearch/hermes-agent/pull/1690), [#1692](https://github.com/NousResearch/hermes-agent/pull/1692))
- **SMS (Twilio)** adapter ([#1688](https://github.com/NousResearch/hermes-agent/pull/1688))
- **Mattermost and Matrix** adapters with @-mention-only filter for Mattermost channels ([#1683](https://github.com/NousResearch/hermes-agent/pull/1683), [#2443](https://github.com/NousResearch/hermes-agent/pull/2443))
- **WhatsApp bridge** adapter ([#2168](https://github.com/NousResearch/hermes-agent/pull/2168))
- **Webhook** platform adapter for external event triggers ([#2166](https://github.com/NousResearch/hermes-agent/pull/2166))
- **OpenAI-compatible API server** platform adapter with `/api/jobs` cron management endpoints ([#1756](https://github.com/NousResearch/hermes-agent/pull/1756), [#2450](https://github.com/NousResearch/hermes-agent/pull/2450), [#2456](https://github.com/NousResearch/hermes-agent/pull/2456))

### Telegram
- Auto-detect HTML tags and use `parse_mode=HTML` in `send_message` ([#1709](https://github.com/NousResearch/hermes-agent/pull/1709))
- Telegram group vision support + thread-based sessions ([#2153](https://github.com/NousResearch/hermes-agent/pull/2153))
- MarkdownV2 support — strikethrough, spoiler, blockquotes, escape parentheses/braces/backslashes/backticks ([#2199](https://github.com/NousResearch/hermes-agent/pull/2199), [#2200](https://github.com/NousResearch/hermes-agent/pull/2200) by @llbn, [#2386](https://github.com/NousResearch/hermes-agent/pull/2386))
- Auto-reconnect polling after network interruption ([#2517](https://github.com/NousResearch/hermes-agent/pull/2517))
- Aggregate split text messages before dispatching ([#1674](https://github.com/NousResearch/hermes-agent/pull/1674))
- Fix: streaming config bridge, not-modified, flood control ([#1782](https://github.com/NousResearch/hermes-agent/pull/1782), [#1783](https://github.com/NousResearch/hermes-agent/pull/1783))
- Fix: edited_message event crashes ([#2074](https://github.com/NousResearch/hermes-agent/pull/2074))
- Fix: retry 409 polling conflicts before giving up ([#2312](https://github.com/NousResearch/hermes-agent/pull/2312))
- Fix: Telegram topic delivery via `platform:chat_id:thread_id` format ([#2455](https://github.com/NousResearch/hermes-agent/pull/2455))

### Discord
- Document caching and text-file injection ([#2503](https://github.com/NousResearch/hermes-agent/pull/2503))
- Persistent typing indicator for DMs ([#2468](https://github.com/NousResearch/hermes-agent/pull/2468))
- Discord DM vision support — inline images + attachment analysis ([#2186](https://github.com/NousResearch/hermes-agent/pull/2186))
- Persist thread participation across gateway restarts ([#1661](https://github.com/NousResearch/hermes-agent/pull/1661))
- Fix: prevent gateway crash on non-ASCII guild names ([#2302](https://github.com/NousResearch/hermes-agent/pull/2302))
- Fix: handle thread permission errors gracefully ([#2073](https://github.com/NousResearch/hermes-agent/pull/2073))
- Fix: properly route slash event handling in threads ([#2460](https://github.com/NousResearch/hermes-agent/pull/2460))
- Fix: remove bugged followup messages + remove `/ask` command ([#1836](https://github.com/NousResearch/hermes-agent/pull/1836))
- Fix: handle graceful reconnection on WebSocket errors ([#2127](https://github.com/NousResearch/hermes-agent/pull/2127))
- Fix: voice channel TTS not working when streaming enabled ([#2322](https://github.com/NousResearch/hermes-agent/pull/2322))

### Other Platforms
- WhatsApp: outbound `send_message` routing ([#1769](https://github.com/NousResearch/hermes-agent/pull/1769) by @sai-samarth), LID format self-chat support ([#1667](https://github.com/NousResearch/hermes-agent/pull/1667)), `reply_prefix` config bridging fix ([#1923](https://github.com/NousResearch/hermes-agent/pull/1923)), restart on bridge child exit ([#2334](https://github.com/NousResearch/hermes-agent/pull/2334)), image/bridge improvements ([#2181](https://github.com/NousResearch/hermes-agent/pull/2181))
- Matrix: duplicate messages and image caching for vision support ([#2520](https://github.com/NousResearch/hermes-agent/pull/2520)), correct `reply_to_message_id` parameter ([#1895](https://github.com/NousResearch/hermes-agent/pull/1895)), bare media types fix ([#1736](https://github.com/NousResearch/hermes-agent/pull/1736))
- Mattermost: MIME types for media attachments ([#2329](https://github.com/NousResearch/hermes-agent/pull/2329))

### Gateway Core
- **Multi-platform gateway** support (Discord + Telegram + all adapters) ([#2125](https://github.com/NousResearch/hermes-agent/pull/2125))
- **Auto-reconnect** failed platforms with exponential backoff ([#2584](https://github.com/NousResearch/hermes-agent/pull/2584))
- **Notify users when session auto-resets** ([#2519](https://github.com/NousResearch/hermes-agent/pull/2519))
- **`/queue` command** to queue prompts without interrupting ([#2191](https://github.com/NousResearch/hermes-agent/pull/2191))
- **Inject reply-to message context** for out-of-session replies ([#1662](https://github.com/NousResearch/hermes-agent/pull/1662))
- **Replace bare text approval** with `/approve` and `/deny` commands ([#2002](https://github.com/NousResearch/hermes-agent/pull/2002))
- **Support ignoring unauthorized gateway DMs** ([#1919](https://github.com/NousResearch/hermes-agent/pull/1919))
- **Configurable approvals** in gateway + `/cost` command with live pricing ([#2180](https://github.com/NousResearch/hermes-agent/pull/2180))
- Fix: prevent duplicate session-key collision in multi-platform gateway ([#2171](https://github.com/NousResearch/hermes-agent/pull/2171))
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
- Fix: improve gateway error handling for 429 usage limits and 500 context overflow ([#1839](https://github.com/NousResearch/hermes-agent/pull/1839))
- Fix: add all missing platform allowlist env vars to startup warning check ([#2628](https://github.com/NousResearch/hermes-agent/pull/2628))
- Fix: show startup banner with all env vars when `verbose_logging=true` ([#2298](https://github.com/NousResearch/hermes-agent/pull/2298))
- Fix: webhook platform config loading from config.yaml ([#2328](https://github.com/NousResearch/hermes-agent/pull/2328))
- Fix: media-group aggregation on rapid successive photo messages ([#2160](https://github.com/NousResearch/hermes-agent/pull/2160))
- Fix: media delivery fails for file paths containing spaces ([#2621](https://github.com/NousResearch/hermes-agent/pull/2621))
- Fix: Matrix and Mattermost never report as connected ([#1711](https://github.com/NousResearch/hermes-agent/pull/1711))
- Fix: PII redaction config never read — missing yaml import ([#1701](https://github.com/NousResearch/hermes-agent/pull/1701))
- Fix: NameError on skill slash commands ([#1697](https://github.com/NousResearch/hermes-agent/pull/1697))
- Fix: persist watcher metadata in checkpoint for crash recovery ([#1706](https://github.com/NousResearch/hermes-agent/pull/1706))
- Fix: pass `message_thread_id` in `send_image_file`, `send_document`, `send_video` ([#2339](https://github.com/NousResearch/hermes-agent/pull/2339))

---

## 🖥️ CLI & User Experience

### Interactive CLI
- **@ context completions** — Claude Code-style `@file`/`@url` references with tab completion ([#2482](https://github.com/NousResearch/hermes-agent/pull/2482), [#2343](https://github.com/NousResearch/hermes-agent/pull/2343))
- **Persistent config bar** in prompt with model + provider info + `/statusbar` toggle ([#2240](https://github.com/NousResearch/hermes-agent/pull/2240), [#1917](https://github.com/NousResearch/hermes-agent/pull/1917))
- **`/permission` command** for dynamic approval mode switching ([#2207](https://github.com/NousResearch/hermes-agent/pull/2207))
- **`/browser` command** for interactive browser sessions ([#2273](https://github.com/NousResearch/hermes-agent/pull/2273))
- **`/tools` disable/enable/list** slash commands with session reset ([#1652](https://github.com/NousResearch/hermes-agent/pull/1652))
- **`/model` command** for runtime model switching with live API probe for custom endpoints ([#2110](https://github.com/NousResearch/hermes-agent/pull/2110), [#1645](https://github.com/NousResearch/hermes-agent/pull/1645), [#2078](https://github.com/NousResearch/hermes-agent/pull/2078))
- **Real-time config reload** — config.yaml changes apply without restart ([#2210](https://github.com/NousResearch/hermes-agent/pull/2210))
- **Kitty keyboard protocol** Shift+Enter handling for Ghostty/WezTerm (reverted due to prompt_toolkit crash) ([#2345](https://github.com/NousResearch/hermes-agent/pull/2345), [#2349](https://github.com/NousResearch/hermes-agent/pull/2349))
- Fix: prevent 'Press ENTER to continue...' on exit ([#2555](https://github.com/NousResearch/hermes-agent/pull/2555))
- Fix: flush stdout during agent loop to prevent macOS display freeze ([#1654](https://github.com/NousResearch/hermes-agent/pull/1654))
- Fix: show human-readable error when `hermes setup` hits permissions error ([#2196](https://github.com/NousResearch/hermes-agent/pull/2196))
- Fix: `/stop` command crash + UnboundLocalError in streaming media delivery ([#2463](https://github.com/NousResearch/hermes-agent/pull/2463))
- Fix: resolve garbled ANSI escape codes in status printouts ([#2448](https://github.com/NousResearch/hermes-agent/pull/2448))
- Fix: normalize toolset labels and use skin colors in banner ([#1912](https://github.com/NousResearch/hermes-agent/pull/1912))
- Fix: update gold ANSI color to true-color format ([#2246](https://github.com/NousResearch/hermes-agent/pull/2246))
- Fix: suppress spinner animation in non-TTY environments ([#2216](https://github.com/NousResearch/hermes-agent/pull/2216))
- Fix: display provider and endpoint in API error messages ([#2266](https://github.com/NousResearch/hermes-agent/pull/2266))

### Setup & Configuration
- **YAML-based config** with backward-compatible env var fallback ([#2172](https://github.com/NousResearch/hermes-agent/pull/2172))
- **`${ENV_VAR}` substitution** in config.yaml ([#2684](https://github.com/NousResearch/hermes-agent/pull/2684))
- **`custom_models.yaml`** for user-managed model additions ([#2214](https://github.com/NousResearch/hermes-agent/pull/2214))
- **Merge nested YAML sections** instead of replacing ([#2213](https://github.com/NousResearch/hermes-agent/pull/2213))
- Fix: log warning instead of silently swallowing config.yaml errors ([#2683](https://github.com/NousResearch/hermes-agent/pull/2683))
- Fix: config.yaml provider key overrides env var silently ([#2272](https://github.com/NousResearch/hermes-agent/pull/2272))
- Fix: `hermes update` use `.[all]` extras with fallback ([#1728](https://github.com/NousResearch/hermes-agent/pull/1728))
- Fix: `hermes update` prompt before resetting working tree on stash conflicts ([#2390](https://github.com/NousResearch/hermes-agent/pull/2390))
- Fix: add zprofile fallback and create zshrc on fresh macOS installs ([#2320](https://github.com/NousResearch/hermes-agent/pull/2320))
- Fix: use git pull --rebase in update/install to avoid divergent branch error ([#2274](https://github.com/NousResearch/hermes-agent/pull/2274))
- Fix: disabled toolsets re-enable themselves after `hermes tools` ([#2268](https://github.com/NousResearch/hermes-agent/pull/2268))
- Fix: platform default toolsets silently override tool deselection ([#2624](https://github.com/NousResearch/hermes-agent/pull/2624))
- Fix: honor bare YAML `approvals.mode: off` ([#2620](https://github.com/NousResearch/hermes-agent/pull/2620))
- Fix: remove `ANTHROPIC_BASE_URL` env var to avoid collisions ([#1675](https://github.com/NousResearch/hermes-agent/pull/1675))
- Fix: don't ask IMAP password if already in keyring or env ([#2212](https://github.com/NousResearch/hermes-agent/pull/2212))
- Fix: prevent `/model` crash when provider list is empty ([#2209](https://github.com/NousResearch/hermes-agent/pull/2209))
- Fix: OpenCode Zen/Go show OpenRouter models instead of their own ([#2277](https://github.com/NousResearch/hermes-agent/pull/2277))

---

## 🔧 Tool System

### Browser Tools
- **Native Hermes browser tools** — navigate, click, type, screenshot, scrape via Browserbase ([#2270](https://github.com/NousResearch/hermes-agent/pull/2270))
- Fix: race condition in session creation orphans cloud sessions ([#1721](https://github.com/NousResearch/hermes-agent/pull/1721))
- Fix: browser handlers TypeError on unexpected LLM params ([#1735](https://github.com/NousResearch/hermes-agent/pull/1735))
- Fix: add `/browser` to COMMAND_REGISTRY for help + autocomplete ([#1814](https://github.com/NousResearch/hermes-agent/pull/1814))

### MCP (Model Context Protocol)
- **MCP server management CLI** + OAuth 2.1 PKCE auth ([#2465](https://github.com/NousResearch/hermes-agent/pull/2465))
- **Interactive MCP tool configuration** in `hermes tools` ([#1694](https://github.com/NousResearch/hermes-agent/pull/1694))
- **Expose MCP servers as standalone toolsets** ([#1907](https://github.com/NousResearch/hermes-agent/pull/1907))
- **Optional FastMCP skill** ([#2113](https://github.com/NousResearch/hermes-agent/pull/2113))
- Fix: MCP-OAuth port mismatch, path traversal, and shared handler state ([#2552](https://github.com/NousResearch/hermes-agent/pull/2552))
- Fix: preserve MCP tool registrations across session resets ([#2124](https://github.com/NousResearch/hermes-agent/pull/2124))
- Fix: concurrent file access crash + duplicate MCP registration ([#2154](https://github.com/NousResearch/hermes-agent/pull/2154))
- Fix: normalise MCP schemas + expand session list columns ([#2102](https://github.com/NousResearch/hermes-agent/pull/2102))
- Fix: `tool_choice` `mcp_` prefix handling ([#1775](https://github.com/NousResearch/hermes-agent/pull/1775))

### Web Tools
- **Configurable web backend** — Firecrawl/BeautifulSoup/Playwright ([#2256](https://github.com/NousResearch/hermes-agent/pull/2256))
- **Parallel** as alternative web search/extract backend ([#1696](https://github.com/NousResearch/hermes-agent/pull/1696))
- **Tavily** as web search/extract/crawl backend ([#1731](https://github.com/NousResearch/hermes-agent/pull/1731))
- Fix: whitespace-only env vars bypass web backend detection ([#2341](https://github.com/NousResearch/hermes-agent/pull/2341))

### Other Tools
- **Vision analysis tool** for image understanding with configurable timeout ([#2182](https://github.com/NousResearch/hermes-agent/pull/2182), [#2480](https://github.com/NousResearch/hermes-agent/pull/2480))
- **Code execution tool** for containerized Python/Node/Bash execution ([#2299](https://github.com/NousResearch/hermes-agent/pull/2299))
- **TTS tool** using OpenAI API with `base_url` support ([#2118](https://github.com/NousResearch/hermes-agent/pull/2118), [#2064](https://github.com/NousResearch/hermes-agent/pull/2064) by @hanai)
- **STT (speech-to-text) tool** using Whisper API ([#2072](https://github.com/NousResearch/hermes-agent/pull/2072))
- **IMAP email** reading and sending tools ([#2173](https://github.com/NousResearch/hermes-agent/pull/2173))
- **RL training data generation tool** ([#2225](https://github.com/NousResearch/hermes-agent/pull/2225))
- **Route-aware pricing estimates** ([#1695](https://github.com/NousResearch/hermes-agent/pull/1695))
- Fix: chunk long messages in `send_message_tool` before platform dispatch ([#1646](https://github.com/NousResearch/hermes-agent/pull/1646))
- Fix: make concurrent tool batching path-aware for file mutations ([#1914](https://github.com/NousResearch/hermes-agent/pull/1914))
- Fix: tool result truncation on large outputs ([#2088](https://github.com/NousResearch/hermes-agent/pull/2088))
- Fix: concurrent file writes safely with atomic operations ([#2086](https://github.com/NousResearch/hermes-agent/pull/2086))
- Fix: improve fuzzy matching accuracy for file search + position calculation refactor ([#2096](https://github.com/NousResearch/hermes-agent/pull/2096), [#1681](https://github.com/NousResearch/hermes-agent/pull/1681))
- Fix: `search_files` wrong line numbers for multi-line matches ([#2069](https://github.com/NousResearch/hermes-agent/pull/2069))
- Fix: include pagination args in repeated search key ([#1824](https://github.com/NousResearch/hermes-agent/pull/1824) by @cutepawss)
- Fix: strip ANSI escape codes from write_file and patch content ([#2532](https://github.com/NousResearch/hermes-agent/pull/2532))
- Fix: expand tilde (~) in vision_analyze local file paths ([#2585](https://github.com/NousResearch/hermes-agent/pull/2585))
- Fix: resource leak and double socket close in `code_execution_tool` ([#2381](https://github.com/NousResearch/hermes-agent/pull/2381))
- Fix: resolve vision analysis race condition and path handling ([#2191](https://github.com/NousResearch/hermes-agent/pull/2191))
- Fix: DM vision — handle multiple images and base64 fallback ([#2211](https://github.com/NousResearch/hermes-agent/pull/2211))
- Fix: `model_supports_images` for custom `base_url` providers returns wrong value ([#2278](https://github.com/NousResearch/hermes-agent/pull/2278))
- Fix: add missing 'messaging' toolset — couldn't enable/disable `send_message` ([#1718](https://github.com/NousResearch/hermes-agent/pull/1718))
- Fix: prevent unavailable tool names from leaking into model schemas ([#2072](https://github.com/NousResearch/hermes-agent/pull/2072))
- Fix: disabled toolsets re-enable themselves after `hermes tools` ([#2268](https://github.com/NousResearch/hermes-agent/pull/2268))
- Fix: pass visited set by reference to prevent diamond dependency duplication ([#2311](https://github.com/NousResearch/hermes-agent/pull/2311))
- Fix: Daytona sandbox lookup migrated from `find_one` to `get/list` ([#2063](https://github.com/NousResearch/hermes-agent/pull/2063) by @rovle)

---

## 🧩 Skills Ecosystem

### Skills System
- **Skills Hub** — discover, install, and manage skills from curated taps ([#2235](https://github.com/NousResearch/hermes-agent/pull/2235))
- **Agent-created persistent skills** with caution-level findings allowed, dangerous skills ask instead of block ([#2116](https://github.com/NousResearch/hermes-agent/pull/2116), [#1840](https://github.com/NousResearch/hermes-agent/pull/1840), [#2446](https://github.com/NousResearch/hermes-agent/pull/2446))
- **`--yes` flag** to bypass confirmation in `/skills install` and uninstall ([#1647](https://github.com/NousResearch/hermes-agent/pull/1647))
- **Disabled skills respected** across banner, system prompt, and slash commands ([#1897](https://github.com/NousResearch/hermes-agent/pull/1897))
- Fix: skills custom_tools import crash + sandbox file_tools integration ([#2239](https://github.com/NousResearch/hermes-agent/pull/2239))
- Fix: agent-created skills with pip requirements crash on install ([#2145](https://github.com/NousResearch/hermes-agent/pull/2145))
- Fix: race condition in `Skills.__init__` when `hub.yaml` missing ([#2242](https://github.com/NousResearch/hermes-agent/pull/2242))
- Fix: validate skill metadata before install and block duplicates ([#2241](https://github.com/NousResearch/hermes-agent/pull/2241))
- Fix: skills hub inspect/resolve — 4 bugs in inspect, redirects, discovery, tap list ([#2447](https://github.com/NousResearch/hermes-agent/pull/2447))
- Fix: agent-created skills keep working after session reset ([#2121](https://github.com/NousResearch/hermes-agent/pull/2121))

### New Skills
- **OCR-and-documents** — PDF/DOCX/XLS/PPTX/image OCR with optional GPU ([#2236](https://github.com/NousResearch/hermes-agent/pull/2236))
- **Huggingface-hub** bundled skill ([#1921](https://github.com/NousResearch/hermes-agent/pull/1921))
- **Sherlock OSINT** username search skill ([#1671](https://github.com/NousResearch/hermes-agent/pull/1671))
- **Inference.sh** skill (terminal-based) ([#1686](https://github.com/NousResearch/hermes-agent/pull/1686))
- **Meme-generation** — real image generator with Pillow ([#2344](https://github.com/NousResearch/hermes-agent/pull/2344))
- **Bioinformatics** gateway skill — index to 400+ bio skills ([#2387](https://github.com/NousResearch/hermes-agent/pull/2387))
- **Base blockchain** optional skill ([#1643](https://github.com/NousResearch/hermes-agent/pull/1643))
- **3D-model-viewer** optional skill ([#2226](https://github.com/NousResearch/hermes-agent/pull/2226))
- **Hermes-agent-setup** skill ([#1905](https://github.com/NousResearch/hermes-agent/pull/1905))

---

## 🔒 Security & Reliability

### Security Hardening
- **SSRF protection** for vision_tools and web_tools (hardened) ([#2679](https://github.com/NousResearch/hermes-agent/pull/2679))
- **Shell injection prevention** in `_expand_path` via `~user` path suffix ([#2685](https://github.com/NousResearch/hermes-agent/pull/2685))
- **Block untrusted browser-origin** API server access ([#2451](https://github.com/NousResearch/hermes-agent/pull/2451))
- **Block sandbox backend creds** from subprocess env ([#1658](https://github.com/NousResearch/hermes-agent/pull/1658))
- **Block @ references** from reading secrets outside workspace ([#2601](https://github.com/NousResearch/hermes-agent/pull/2601) by @Gutslabs)
- **Require opt-in** for project plugin discovery ([#2215](https://github.com/NousResearch/hermes-agent/pull/2215))
- **Malicious code pattern pre-exec scanner** for terminal_tool ([#2245](https://github.com/NousResearch/hermes-agent/pull/2245))
- **Harden terminal safety** and sandbox file writes ([#1653](https://github.com/NousResearch/hermes-agent/pull/1653))
- **PKCE verifier leak** fix, OAuth refresh Content-Type fix ([#1775](https://github.com/NousResearch/hermes-agent/pull/1775))
- **Eliminate SQL string formatting** in `execute()` calls ([#2061](https://github.com/NousResearch/hermes-agent/pull/2061) by @dusterbloom)
- **Harden jobs API** — input limits, field whitelist, startup check ([#2456](https://github.com/NousResearch/hermes-agent/pull/2456))
- Fix: OAuth flag stale after refresh/fallback ([#1890](https://github.com/NousResearch/hermes-agent/pull/1890))
- Fix: auxiliary client skips expired Codex JWT ([#2397](https://github.com/NousResearch/hermes-agent/pull/2397))

### Reliability
- **Concurrent tool safety** — path-aware file mutation batching, thread locks on SessionDB methods, file locking for memory writes ([#1914](https://github.com/NousResearch/hermes-agent/pull/1914), [#1704](https://github.com/NousResearch/hermes-agent/pull/1704), [#1726](https://github.com/NousResearch/hermes-agent/pull/1726))
- **Error recovery** — handle OpenRouter errors gracefully, guard print() calls against OSError ([#2112](https://github.com/NousResearch/hermes-agent/pull/2112), [#1668](https://github.com/NousResearch/hermes-agent/pull/1668))
- **Redacting formatter** — safely handle non-string inputs, NameError fix when verbose_logging=True ([#2392](https://github.com/NousResearch/hermes-agent/pull/2392), [#1700](https://github.com/NousResearch/hermes-agent/pull/1700))
- **ACP** — preserve session provider when switching models, persist sessions to disk, preserve leading whitespace in streaming chunks ([#2380](https://github.com/NousResearch/hermes-agent/pull/2380), [#2071](https://github.com/NousResearch/hermes-agent/pull/2071), [#2192](https://github.com/NousResearch/hermes-agent/pull/2192))
- **API server** — persist ResponseStore to SQLite across restarts ([#2472](https://github.com/NousResearch/hermes-agent/pull/2472))
- Fix: `fetch_nous_models` called with positional args — always TypeError ([#1699](https://github.com/NousResearch/hermes-agent/pull/1699))
- Fix: `make_is_write_denied` robust to Path objects ([#1678](https://github.com/NousResearch/hermes-agent/pull/1678))
- Fix: resolve merge conflict markers in cli.py breaking hermes startup ([#2347](https://github.com/NousResearch/hermes-agent/pull/2347))
- Fix: `minisweagent_path.py` missing from wheel ([#2098](https://github.com/NousResearch/hermes-agent/pull/2098) by @JiwaniZakir)

### Cron System
- **Cron job scheduling** for gateway ([#2140](https://github.com/NousResearch/hermes-agent/pull/2140))
- **`[SILENT]` response** — cron agents can suppress delivery ([#1833](https://github.com/NousResearch/hermes-agent/pull/1833))
- **Scale missed-job grace window** with schedule frequency ([#2449](https://github.com/NousResearch/hermes-agent/pull/2449))
- **Recover recent one-shot jobs** ([#1918](https://github.com/NousResearch/hermes-agent/pull/1918))
- Fix: normalize `repeat<=0` to None — cron jobs deleted after first run when LLM passes -1 ([#2612](https://github.com/NousResearch/hermes-agent/pull/2612) by @Mibayy)
- Fix: Matrix added to scheduler delivery platform_map ([#2167](https://github.com/NousResearch/hermes-agent/pull/2167) by @buntingszn)
- Fix: naive ISO timestamps stored without timezone — jobs fire at wrong time ([#1729](https://github.com/NousResearch/hermes-agent/pull/1729))
- Fix: `get_due_jobs` reads `jobs.json` twice — race condition ([#1716](https://github.com/NousResearch/hermes-agent/pull/1716))
- Fix: silent jobs return empty response for delivery skip ([#2442](https://github.com/NousResearch/hermes-agent/pull/2442))
- Fix: stop injecting cron outputs into gateway session history ([#2313](https://github.com/NousResearch/hermes-agent/pull/2313))
- Fix: close abandoned coroutine when `asyncio.run()` raises RuntimeError ([#2317](https://github.com/NousResearch/hermes-agent/pull/2317))

---

## 🐛 Notable Bug Fixes

- Fix: show full command in dangerous command approval prompt ([#1649](https://github.com/NousResearch/hermes-agent/pull/1649))
- Fix: Telegram streaming message length overflow ([#1783](https://github.com/NousResearch/hermes-agent/pull/1783))
- Fix: prevent `/model` crash when provider list is empty ([#2209](https://github.com/NousResearch/hermes-agent/pull/2209))
- Fix: batch of 5 small contributor fixes — PortAudio, SafeWriter, IMAP, thread lock, prefill ([#2466](https://github.com/NousResearch/hermes-agent/pull/2466))
- Fix: `dingtalk-stream` added to optional dependencies ([#2452](https://github.com/NousResearch/hermes-agent/pull/2452))
- Fix: remove hardcoded `gemini-3-flash-preview` as default summary model ([#2464](https://github.com/NousResearch/hermes-agent/pull/2464))
- Fix: remove post-compression file-read history injection ([#2226](https://github.com/NousResearch/hermes-agent/pull/2226))
- Fix: truncated `AUXILIARY_WEB_EXTRACT_API_KEY` env var name ([#2309](https://github.com/NousResearch/hermes-agent/pull/2309))
- Fix: update validator does not stop ([#2204](https://github.com/NousResearch/hermes-agent/pull/2204), [#2067](https://github.com/NousResearch/hermes-agent/pull/2067))
- Fix: log disk warning check failures at debug level ([#2394](https://github.com/NousResearch/hermes-agent/pull/2394))
- Fix: quiet mode with `--resume` now passes conversation_history ([#2357](https://github.com/NousResearch/hermes-agent/pull/2357))
- Fix: unify resume logic in batch mode for consistent `--resume` behavior ([#2331](https://github.com/NousResearch/hermes-agent/pull/2331))
- Fix: prevent unavailable tool names from leaking into model schemas ([#2072](https://github.com/NousResearch/hermes-agent/pull/2072))
- Fix: remove `_is_special_key` hack and fix `/skills` path completion ([#2271](https://github.com/NousResearch/hermes-agent/pull/2271))
- Fix: use home-relative state paths if XDG dirs don't exist ([#2325](https://github.com/NousResearch/hermes-agent/pull/2325))
- Fix: inject model identity for Alibaba Coding Plan ([#2314](https://github.com/NousResearch/hermes-agent/pull/2314))
- Fix: OpenClaw migration warns when API keys are skipped ([#1655](https://github.com/NousResearch/hermes-agent/pull/1655))
- Fix: email `send_typing` metadata + ☤ Hermes staff symbol ([#1665](https://github.com/NousResearch/hermes-agent/pull/1665))
- Fix: replace production `print()` calls with logger in rl_training_tool ([#2462](https://github.com/NousResearch/hermes-agent/pull/2462))
- Fix: restore opencode-go provider config corrupted by secret redaction ([#2393](https://github.com/NousResearch/hermes-agent/pull/2393) by @0xbyt4)

---

## 🧪 Testing

- Resolve all consistently failing tests ([#2488](https://github.com/NousResearch/hermes-agent/pull/2488))
- Replace `FakePath` with `monkeypatch` for Python 3.12 compat ([#2444](https://github.com/NousResearch/hermes-agent/pull/2444))
- Align Hermes setup and full-suite expectations ([#1710](https://github.com/NousResearch/hermes-agent/pull/1710))
- Add tests for API server jobs API hardening ([#2456](https://github.com/NousResearch/hermes-agent/pull/2456))

---

## 📚 Documentation

- Comprehensive documentation update for recent features ([#1693](https://github.com/NousResearch/hermes-agent/pull/1693), [#2183](https://github.com/NousResearch/hermes-agent/pull/2183))
- Alibaba Cloud and DingTalk setup guide ([#1687](https://github.com/NousResearch/hermes-agent/pull/1687), [#1692](https://github.com/NousResearch/hermes-agent/pull/1692))
- SOUL.md as primary agent identity documentation ([#1927](https://github.com/NousResearch/hermes-agent/pull/1927))
- Detailed skills documentation ([#2244](https://github.com/NousResearch/hermes-agent/pull/2244))
- Honcho self-hosted / Docker configuration section ([#2475](https://github.com/NousResearch/hermes-agent/pull/2475))
- Context length detection references in FAQ and quickstart ([#2179](https://github.com/NousResearch/hermes-agent/pull/2179))
- Fix documentation inconsistencies across reference and user guides ([#1995](https://github.com/NousResearch/hermes-agent/pull/1995))
- Fix MCP install commands — use uv, not bare pip ([#1909](https://github.com/NousResearch/hermes-agent/pull/1909))
- Fix MDX build error in api-server.md ([#1787](https://github.com/NousResearch/hermes-agent/pull/1787))
- Replace ASCII diagrams with Mermaid/lists ([#2402](https://github.com/NousResearch/hermes-agent/pull/2402))
- Add missing gateway commands and correct examples ([#2329](https://github.com/NousResearch/hermes-agent/pull/2329))
- Clarify self-hosted Firecrawl setup ([#1669](https://github.com/NousResearch/hermes-agent/pull/1669))
- NeuTTS provider documentation ([#1903](https://github.com/NousResearch/hermes-agent/pull/1903))
- Gemini OAuth provider implementation plan ([#2467](https://github.com/NousResearch/hermes-agent/pull/2467))
- Discord Server Members Intent marked as required ([#2330](https://github.com/NousResearch/hermes-agent/pull/2330))
- Align venv path to match installer (venv/ not .venv/) ([#2114](https://github.com/NousResearch/hermes-agent/pull/2114))
- New skills added to hub index ([#2281](https://github.com/NousResearch/hermes-agent/pull/2281))
- OCR-and-documents skill — split, merge, search examples ([#2461](https://github.com/NousResearch/hermes-agent/pull/2461))

---

## 👥 Contributors

### Core
- **@teknium1** (Teknium) — 280 PRs

### Community Contributors
- **@mchzimm** (to_the_max) — GitHub Copilot provider integration across Hermes ([#1879](https://github.com/NousResearch/hermes-agent/pull/1879))
- **@jquesnelle** (Jeffrey Quesnelle) — Per-thread persistent event loops in worker threads ([#2214](https://github.com/NousResearch/hermes-agent/pull/2214))
- **@llbn** (lbn) — Telegram MarkdownV2 support: strikethrough, spoiler, blockquotes, and escape fixes ([#2199](https://github.com/NousResearch/hermes-agent/pull/2199), [#2200](https://github.com/NousResearch/hermes-agent/pull/2200))
- **@dusterbloom** — SQL injection prevention + local server context window querying ([#2061](https://github.com/NousResearch/hermes-agent/pull/2061), [#2091](https://github.com/NousResearch/hermes-agent/pull/2091))
- **@0xbyt4** — Anthropic tool_calls None guard + OpenCode-Go provider config fix ([#2209](https://github.com/NousResearch/hermes-agent/pull/2209), [#2393](https://github.com/NousResearch/hermes-agent/pull/2393))
- **@sai-samarth** (Saisamarth) — WhatsApp send_message routing + systemd node path fix ([#1769](https://github.com/NousResearch/hermes-agent/pull/1769), [#1767](https://github.com/NousResearch/hermes-agent/pull/1767))
- **@Gutslabs** (Guts) — Block @ references from reading secrets outside workspace ([#2601](https://github.com/NousResearch/hermes-agent/pull/2601))
- **@Mibayy** (Mibay) — Cron job repeat normalization fix ([#2612](https://github.com/NousResearch/hermes-agent/pull/2612))
- **@ten-jampa** (Tenzin Jampa) — Gateway /title command session fix ([#2379](https://github.com/NousResearch/hermes-agent/pull/2379))
- **@cutepawss** (lila) — File tools search pagination fix ([#1824](https://github.com/NousResearch/hermes-agent/pull/1824))
- **@hanai** (Hanai) — OpenAI TTS base_url support ([#2064](https://github.com/NousResearch/hermes-agent/pull/2064))
- **@rovle** (Lovre Pešut) — Daytona sandbox API migration ([#2063](https://github.com/NousResearch/hermes-agent/pull/2063))
- **@buntingszn** (bunting szn) — Matrix cron delivery support ([#2167](https://github.com/NousResearch/hermes-agent/pull/2167))
- **@InB4DevOps** — Token counter reset on new session ([#2101](https://github.com/NousResearch/hermes-agent/pull/2101))
- **@JiwaniZakir** (Zakir Jiwani) — Missing file in wheel fix ([#2098](https://github.com/NousResearch/hermes-agent/pull/2098))
- **@ygd58** (buray) — Delegate tool parent tool names fix ([#2083](https://github.com/NousResearch/hermes-agent/pull/2083))

---

**Full Changelog**: [v2026.3.17...v2026.3.23](https://github.com/NousResearch/hermes-agent/compare/v2026.3.17...v2026.3.23)
