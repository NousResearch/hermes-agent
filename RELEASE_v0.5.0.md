# Hermes Agent v0.5.0 (v2026.3.28)

**Release Date:** March 28, 2026

> The hardening release — Hugging Face provider, /model command overhaul, Telegram Private Chat Topics, native Modal SDK, plugin lifecycle hooks, tool-use enforcement for GPT models, Nix flake, 50+ security and reliability fixes, and a comprehensive supply chain audit.

---

## ✨ Highlights

- **Nous Portal now supports 400+ models** — The Nous Research inference portal has expanded dramatically, giving Hermes Agent users access to over 400 models through a single provider endpoint

- **Hugging Face as a first-class inference provider** — Full integration with HF Inference API including curated agentic model picker that maps to OpenRouter analogues, live `/models` endpoint probe, and setup wizard flow ([#3419](https://github.com/NousResearch/hermes-agent/pull/3419), [#3440](https://github.com/NousResearch/hermes-agent/pull/3440))

- **Telegram Private Chat Topics** — Project-based conversations with functional skill binding per topic, enabling isolated workflows within a single Telegram chat ([#3163](https://github.com/NousResearch/hermes-agent/pull/3163))

- **Native Modal SDK backend** — Replaced swe-rex dependency with native Modal SDK (`Sandbox.create.aio` + `exec.aio`), eliminating tunnels and simplifying the Modal terminal backend ([#3538](https://github.com/NousResearch/hermes-agent/pull/3538))

- **Plugin lifecycle hooks activated** — `pre_llm_call`, `post_llm_call`, `on_session_start`, and `on_session_end` hooks now fire in the agent loop and CLI/gateway, completing the plugin hook system ([#3542](https://github.com/NousResearch/hermes-agent/pull/3542))

- **Improved OpenAI Model Reliability** — Added `GPT_TOOL_USE_GUIDANCE` to prevent GPT models from describing intended actions instead of making tool calls, plus automatic stripping of stale budget warnings from conversation history that caused models to avoid tools across turns ([#3528](https://github.com/NousResearch/hermes-agent/pull/3528))

- **Nix flake** — Full uv2nix build, NixOS module with persistent container mode, auto-generated config keys from Python source, and suffix PATHs for agent-friendliness ([#20](https://github.com/NousResearch/hermes-agent/pull/20), [#3274](https://github.com/NousResearch/hermes-agent/pull/3274), [#3061](https://github.com/NousResearch/hermes-agent/pull/3061)) by @alt-glitch

- **Supply chain hardening** — Removed compromised `litellm` dependency, pinned all dependency version ranges, regenerated `uv.lock` with hashes, added CI workflow scanning PRs for supply chain attack patterns, and bumped deps to fix CVEs ([#2796](https://github.com/NousResearch/hermes-agent/pull/2796), [#2810](https://github.com/NousResearch/hermes-agent/pull/2810), [#2812](https://github.com/NousResearch/hermes-agent/pull/2812), [#2816](https://github.com/NousResearch/hermes-agent/pull/2816), [#3073](https://github.com/NousResearch/hermes-agent/pull/3073))

- **Anthropic output limits fix** — Replaced hardcoded 16K `max_tokens` with per-model native output limits (128K for Opus 4.6, 64K for Sonnet 4.6), fixing "Response truncated" and thinking-budget exhaustion on direct Anthropic API ([#3426](https://github.com/NousResearch/hermes-agent/pull/3426), [#3444](https://github.com/NousResearch/hermes-agent/pull/3444))

---

## 🏗️ Core Agent & Architecture

### New Provider: Hugging Face
- First-class Hugging Face Inference API integration with auth, setup wizard, and model picker ([#3419](https://github.com/NousResearch/hermes-agent/pull/3419))
- Curated model list mapping OpenRouter agentic defaults to HF equivalents — providers with 8+ curated models skip live `/models` probe for speed ([#3440](https://github.com/NousResearch/hermes-agent/pull/3440))
- Added glm-5-turbo to Z.AI provider model list ([#3095](https://github.com/NousResearch/hermes-agent/pull/3095))

### Provider & Model Improvements
- `/model` command overhaul — extracted shared `switch_model()` pipeline for CLI and gateway, custom endpoint support, provider-aware routing ([#2795](https://github.com/NousResearch/hermes-agent/pull/2795), [#2799](https://github.com/NousResearch/hermes-agent/pull/2799))
- Removed `/model` slash command from CLI and gateway in favor of `hermes model` subcommand ([#3080](https://github.com/NousResearch/hermes-agent/pull/3080))
- Preserve `custom` provider instead of silently remapping to `openrouter` ([#2792](https://github.com/NousResearch/hermes-agent/pull/2792))
- Read root-level `provider` and `base_url` from config.yaml into model config ([#3112](https://github.com/NousResearch/hermes-agent/pull/3112))
- Align Nous Portal model slugs with OpenRouter naming ([#3253](https://github.com/NousResearch/hermes-agent/pull/3253))
- Fix Alibaba provider default endpoint and model list ([#3484](https://github.com/NousResearch/hermes-agent/pull/3484))
- Allow MiniMax users to override `/v1` → `/anthropic` auto-correction ([#3553](https://github.com/NousResearch/hermes-agent/pull/3553))
- Migrate OAuth token refresh to `platform.claude.com` with fallback ([#3246](https://github.com/NousResearch/hermes-agent/pull/3246))

### Agent Loop & Conversation
- **Improved OpenAI model reliability** — `GPT_TOOL_USE_GUIDANCE` prevents GPT models from describing actions instead of calling tools + automatic budget warning stripping from history ([#3528](https://github.com/NousResearch/hermes-agent/pull/3528))
- **Surface lifecycle events** — All retry, fallback, and compression events now surface to the user as formatted messages ([#3153](https://github.com/NousResearch/hermes-agent/pull/3153))
- **Anthropic output limits** — Per-model native output limits instead of hardcoded 16K `max_tokens` ([#3426](https://github.com/NousResearch/hermes-agent/pull/3426))
- **Thinking-budget exhaustion detection** — Skip useless continuation retries when model uses all output tokens on reasoning ([#3444](https://github.com/NousResearch/hermes-agent/pull/3444))
- Always prefer streaming for API calls to prevent hung subagents ([#3120](https://github.com/NousResearch/hermes-agent/pull/3120))
- Restore safe non-streaming fallback after stream failures ([#3020](https://github.com/NousResearch/hermes-agent/pull/3020))
- Give subagents independent iteration budgets ([#3004](https://github.com/NousResearch/hermes-agent/pull/3004))
- Update `api_key` in `_try_activate_fallback` for subagent auth ([#3103](https://github.com/NousResearch/hermes-agent/pull/3103))
- Graceful return on max retries instead of crashing thread ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Count compression restarts toward retry limit ([#3070](https://github.com/NousResearch/hermes-agent/pull/3070))
- Include tool tokens in preflight estimate, guard context probe persistence ([#3164](https://github.com/NousResearch/hermes-agent/pull/3164))
- Update context compressor limits after fallback activation ([#3305](https://github.com/NousResearch/hermes-agent/pull/3305))
- Validate empty user messages to prevent Anthropic API 400 errors ([#3322](https://github.com/NousResearch/hermes-agent/pull/3322))
- GLM reasoning-only and max-length handling ([#3010](https://github.com/NousResearch/hermes-agent/pull/3010))
- Increase API timeout default from 900s to 1800s for slow-thinking models ([#3431](https://github.com/NousResearch/hermes-agent/pull/3431))
- Send `max_tokens` for Claude/OpenRouter + retry SSE connection errors ([#3497](https://github.com/NousResearch/hermes-agent/pull/3497))
- Prevent AsyncOpenAI/httpx cross-loop deadlock in gateway mode ([#2701](https://github.com/NousResearch/hermes-agent/pull/2701)) by @ctlst

### Streaming & Reasoning
- **Persist reasoning across gateway session turns** with new schema v6 columns (`reasoning`, `reasoning_details`, `codex_reasoning_items`) ([#2974](https://github.com/NousResearch/hermes-agent/pull/2974))
- Detect and kill stale SSE connections ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Fix stale stream detector race causing spurious `RemoteProtocolError` ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Skip duplicate callback for `<think>`-extracted reasoning during streaming ([#3116](https://github.com/NousResearch/hermes-agent/pull/3116))
- Preserve reasoning fields in `rewrite_transcript` ([#3311](https://github.com/NousResearch/hermes-agent/pull/3311))
- Preserve Gemini thought signatures in streamed tool calls ([#2997](https://github.com/NousResearch/hermes-agent/pull/2997))
- Ensure first delta is fired during reasoning updates ([untagged commit](https://github.com/NousResearch/hermes-agent))

### Session & Memory
- **Session search recent sessions mode** — Omit query to browse recent sessions with titles, previews, and timestamps ([#2533](https://github.com/NousResearch/hermes-agent/pull/2533))
- **Session config surfacing** on `/new`, `/reset`, and auto-reset ([#3321](https://github.com/NousResearch/hermes-agent/pull/3321))
- **Third-party session isolation** — `--source` flag for isolating sessions by origin ([#3255](https://github.com/NousResearch/hermes-agent/pull/3255))
- Add `/resume` CLI handler, session log truncation guard, `reopen_session` API ([#3315](https://github.com/NousResearch/hermes-agent/pull/3315))
- Clear compressor summary and turn counter on `/clear` and `/new` ([#3102](https://github.com/NousResearch/hermes-agent/pull/3102))
- Surface silent SessionDB failures that cause session data loss ([#2999](https://github.com/NousResearch/hermes-agent/pull/2999))
- Session search fallback preview on summarization failure ([#3478](https://github.com/NousResearch/hermes-agent/pull/3478))
- Prevent stale memory overwrites by flush agent ([#2687](https://github.com/NousResearch/hermes-agent/pull/2687))

### Context Compression
- Replace dead `summary_target_tokens` with ratio-based scaling ([#2554](https://github.com/NousResearch/hermes-agent/pull/2554))
- Expose `compression.target_ratio`, `protect_last_n`, and `threshold` in `DEFAULT_CONFIG` ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Restore sane defaults and cap summary at 12K tokens ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Preserve transcript on `/compress` and hygiene compression ([#3556](https://github.com/NousResearch/hermes-agent/pull/3556))
- Update context pressure warnings and token estimates after compaction ([untagged commit](https://github.com/NousResearch/hermes-agent))

### Architecture & Dependencies
- **Remove mini-swe-agent dependency** — Inline Docker and Modal backends directly ([#2804](https://github.com/NousResearch/hermes-agent/pull/2804))
- **Replace swe-rex with native Modal SDK** for Modal backend ([#3538](https://github.com/NousResearch/hermes-agent/pull/3538))
- **Plugin lifecycle hooks** — `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end` now fire in the agent loop ([#3542](https://github.com/NousResearch/hermes-agent/pull/3542))
- Fix plugin toolsets invisible in `hermes tools` and standalone processes ([#3457](https://github.com/NousResearch/hermes-agent/pull/3457))
- Consolidate `get_hermes_home()` and `parse_reasoning_effort()` ([#3062](https://github.com/NousResearch/hermes-agent/pull/3062))
- Remove unused Hermes-native PKCE OAuth flow ([#3107](https://github.com/NousResearch/hermes-agent/pull/3107))
- Remove ~100 unused imports across 55 files ([#3016](https://github.com/NousResearch/hermes-agent/pull/3016))
- Fix 154 f-strings, simplify getattr/URL patterns, remove dead code ([#3119](https://github.com/NousResearch/hermes-agent/pull/3119))

---

## 📱 Messaging Platforms (Gateway)

### Telegram
- **Private Chat Topics** — Project-based conversations with functional skill binding per topic, enabling isolated workflows within a single Telegram chat ([#3163](https://github.com/NousResearch/hermes-agent/pull/3163))
- **Auto-discover fallback IPs via DNS-over-HTTPS** when `api.telegram.org` is unreachable ([#3376](https://github.com/NousResearch/hermes-agent/pull/3376))
- **Configurable reply threading mode** ([#2907](https://github.com/NousResearch/hermes-agent/pull/2907))
- Fall back to no `thread_id` on "Message thread not found" BadRequest ([#3390](https://github.com/NousResearch/hermes-agent/pull/3390))
- Self-reschedule reconnect when `start_polling` fails after 502 ([#3268](https://github.com/NousResearch/hermes-agent/pull/3268))

### Discord
- Stop phantom typing indicator after agent turn completes ([#3003](https://github.com/NousResearch/hermes-agent/pull/3003))

### Slack
- Send tool call progress messages to correct Slack thread ([#3063](https://github.com/NousResearch/hermes-agent/pull/3063))
- Scope progress thread fallback to Slack only ([#3488](https://github.com/NousResearch/hermes-agent/pull/3488))

### WhatsApp
- Download documents, audio, and video media from messages ([#2978](https://github.com/NousResearch/hermes-agent/pull/2978))

### Matrix
- Add missing Matrix entry in `PLATFORMS` dict ([#3473](https://github.com/NousResearch/hermes-agent/pull/3473))
- Harden e2ee access-token handling ([#3562](https://github.com/NousResearch/hermes-agent/pull/3562))
- Add backoff for `SyncError` in sync loop ([#3280](https://github.com/NousResearch/hermes-agent/pull/3280))

### Signal
- Track SSE keepalive comments as connection activity ([#3316](https://github.com/NousResearch/hermes-agent/pull/3316))

### Email
- Prevent unbounded growth of `_seen_uids` in EmailAdapter ([#3490](https://github.com/NousResearch/hermes-agent/pull/3490))

### Gateway Core
- **Config-gated `/verbose` command** for messaging platforms — toggle tool output verbosity from chat ([#3262](https://github.com/NousResearch/hermes-agent/pull/3262))
- **Background review notifications** delivered to user chat ([#3293](https://github.com/NousResearch/hermes-agent/pull/3293))
- **Retry transient send failures** and notify user on exhaustion ([#3288](https://github.com/NousResearch/hermes-agent/pull/3288))
- Recover from hung agents — `/stop` hard-kills session lock ([#3104](https://github.com/NousResearch/hermes-agent/pull/3104))
- Thread-safe `SessionStore` — protect `_entries` with `threading.Lock` ([#3052](https://github.com/NousResearch/hermes-agent/pull/3052))
- Fix gateway token double-counting with cached agents — use absolute set instead of increment ([#3306](https://github.com/NousResearch/hermes-agent/pull/3306), [#3317](https://github.com/NousResearch/hermes-agent/pull/3317))
- Fingerprint full auth token in agent cache signature ([#3247](https://github.com/NousResearch/hermes-agent/pull/3247))
- Silence background agent terminal output ([#3297](https://github.com/NousResearch/hermes-agent/pull/3297))
- Include per-platform `ALLOW_ALL` and `SIGNAL_GROUP` in startup allowlist check ([#3313](https://github.com/NousResearch/hermes-agent/pull/3313))
- Include user-local bin paths in systemd unit PATH ([#3527](https://github.com/NousResearch/hermes-agent/pull/3527))
- Track background task references in `GatewayRunner` ([#3254](https://github.com/NousResearch/hermes-agent/pull/3254))
- Add request timeouts to HA, Email, Mattermost, SMS adapters ([#3258](https://github.com/NousResearch/hermes-agent/pull/3258))
- Add media download retry to Mattermost, Slack, and base cache ([#3323](https://github.com/NousResearch/hermes-agent/pull/3323))
- Detect virtualenv path instead of hardcoding `venv/` ([#2797](https://github.com/NousResearch/hermes-agent/pull/2797))
- Use `TERMINAL_CWD` for context file discovery, not process cwd ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Stop loading hermes repo AGENTS.md into gateway sessions (~10k wasted tokens) ([#2891](https://github.com/NousResearch/hermes-agent/pull/2891))

---

## 🖥️ CLI & User Experience

### Interactive CLI
- **Configurable busy input mode** + fix `/queue` always working ([#3298](https://github.com/NousResearch/hermes-agent/pull/3298))
- **Preserve user input on multiline paste** ([#3065](https://github.com/NousResearch/hermes-agent/pull/3065))
- **Tool generation callback** — streaming "preparing terminal…" updates during tool argument generation ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Show tool progress for substantive tools, not just "preparing" ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Buffer reasoning preview chunks and fix duplicate display ([#3013](https://github.com/NousResearch/hermes-agent/pull/3013))
- Prevent reasoning box from rendering 3x during tool-calling loops ([#3405](https://github.com/NousResearch/hermes-agent/pull/3405))
- Eliminate "Event loop is closed" / "Press ENTER to continue" during idle — three-layer fix with `neuter_async_httpx_del()`, custom exception handler, and stale client cleanup ([#3398](https://github.com/NousResearch/hermes-agent/pull/3398))
- Fix status bar shows 26K instead of 260K for token counts with trailing zeros ([#3024](https://github.com/NousResearch/hermes-agent/pull/3024))
- Fix status bar duplicates and degrades during long sessions ([#3291](https://github.com/NousResearch/hermes-agent/pull/3291))
- Refresh TUI before background task output to prevent status bar overlap ([#3048](https://github.com/NousResearch/hermes-agent/pull/3048))
- Suppress KawaiiSpinner animation under `patch_stdout` ([#2994](https://github.com/NousResearch/hermes-agent/pull/2994))
- Skip KawaiiSpinner when TUI handles tool progress ([#2973](https://github.com/NousResearch/hermes-agent/pull/2973))
- Guard `isatty()` against closed streams via `_is_tty` property ([#3056](https://github.com/NousResearch/hermes-agent/pull/3056))
- Ensure single closure of streaming boxes during tool generation ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Cap context pressure percentage at 100% in display ([#3480](https://github.com/NousResearch/hermes-agent/pull/3480))
- Clean up HTML error messages in CLI display ([#3069](https://github.com/NousResearch/hermes-agent/pull/3069))
- Show HTTP status code and 400 body in API error output ([#3096](https://github.com/NousResearch/hermes-agent/pull/3096))
- Extract useful info from HTML error pages, dump debug on max retries ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Prevent TypeError on startup when `base_url` is None ([#3068](https://github.com/NousResearch/hermes-agent/pull/3068))
- Prevent update crash in non-TTY environments ([#3094](https://github.com/NousResearch/hermes-agent/pull/3094))
- Handle EOFError in sessions delete/prune confirmation prompts ([#3101](https://github.com/NousResearch/hermes-agent/pull/3101))
- Catch KeyboardInterrupt during `flush_memories` on exit and in exit cleanup handlers ([#3025](https://github.com/NousResearch/hermes-agent/pull/3025), [#3257](https://github.com/NousResearch/hermes-agent/pull/3257))
- Guard `.strip()` against None values from YAML config ([#3552](https://github.com/NousResearch/hermes-agent/pull/3552))
- Guard `config.get()` against YAML null values to prevent AttributeError ([#3377](https://github.com/NousResearch/hermes-agent/pull/3377))
- Store asyncio task references to prevent GC mid-execution ([#3267](https://github.com/NousResearch/hermes-agent/pull/3267))

### Setup & Configuration
- Use explicit key mapping for returning-user menu dispatch instead of positional index ([#3083](https://github.com/NousResearch/hermes-agent/pull/3083))
- Use `sys.executable` for pip in update commands to fix PEP 668 ([#3099](https://github.com/NousResearch/hermes-agent/pull/3099))
- Harden `hermes update` against diverged history, non-main branches, and gateway edge cases ([#3492](https://github.com/NousResearch/hermes-agent/pull/3492))
- OpenClaw migration overwrites defaults and setup wizard skips imported sections — fixed ([#3282](https://github.com/NousResearch/hermes-agent/pull/3282))
- Stop recursive AGENTS.md walk, load top-level only ([#3110](https://github.com/NousResearch/hermes-agent/pull/3110))
- Add macOS Homebrew paths to browser and terminal PATH resolution ([#2713](https://github.com/NousResearch/hermes-agent/pull/2713))
- YAML boolean handling for `tool_progress` config ([#3300](https://github.com/NousResearch/hermes-agent/pull/3300))
- Reset default SOUL.md to baseline identity text ([#3159](https://github.com/NousResearch/hermes-agent/pull/3159))
- Reject relative cwd paths for container terminal backends ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Add explicit `hermes-api-server` toolset for API server platform ([#3304](https://github.com/NousResearch/hermes-agent/pull/3304))
- Reorder setup wizard providers — OpenRouter first ([untagged commit](https://github.com/NousResearch/hermes-agent))

---

## 🔧 Tool System

### API Server
- **Idempotency-Key support**, body size limit, and OpenAI error envelope ([#2903](https://github.com/NousResearch/hermes-agent/pull/2903))
- Allow Idempotency-Key in CORS headers ([#3530](https://github.com/NousResearch/hermes-agent/pull/3530))
- Cancel orphaned agent + true interrupt on SSE disconnect ([#3427](https://github.com/NousResearch/hermes-agent/pull/3427))
- Fix streaming breaks when agent makes tool calls ([#2985](https://github.com/NousResearch/hermes-agent/pull/2985))

### Terminal & File Operations
- Handle addition-only hunks in V4A patch parser ([#3325](https://github.com/NousResearch/hermes-agent/pull/3325))
- Exponential backoff for persistent shell polling ([#2996](https://github.com/NousResearch/hermes-agent/pull/2996))
- Add timeout to subprocess calls in `context_references` ([#3469](https://github.com/NousResearch/hermes-agent/pull/3469))

### Browser & Vision
- Handle 402 insufficient credits error in vision tool ([#2802](https://github.com/NousResearch/hermes-agent/pull/2802))
- Fix `browser_vision` ignores `auxiliary.vision.timeout` config ([#2901](https://github.com/NousResearch/hermes-agent/pull/2901))
- Make browser command timeout configurable via config.yaml ([#2801](https://github.com/NousResearch/hermes-agent/pull/2801))

### MCP
- MCP toolset resolution for runtime and config ([#3252](https://github.com/NousResearch/hermes-agent/pull/3252))
- Add MCP tool name collision protection ([#3077](https://github.com/NousResearch/hermes-agent/pull/3077))

### Auxiliary LLM
- Guard aux LLM calls against None content + reasoning fallback + retry ([#3449](https://github.com/NousResearch/hermes-agent/pull/3449))
- Catch ImportError from `build_anthropic_client` in vision auto-detection ([#3312](https://github.com/NousResearch/hermes-agent/pull/3312))

### Other Tools
- Add request timeouts to `send_message_tool` HTTP calls ([#3162](https://github.com/NousResearch/hermes-agent/pull/3162)) by @memosr
- Auto-repair `jobs.json` with invalid control characters ([#3537](https://github.com/NousResearch/hermes-agent/pull/3537))
- Enable fine-grained tool streaming for Claude/OpenRouter ([#3497](https://github.com/NousResearch/hermes-agent/pull/3497))

---

## 🧩 Skills Ecosystem

### Skills System
- **Env var passthrough** for skills and user config — skills can declare environment variables to pass through ([#2807](https://github.com/NousResearch/hermes-agent/pull/2807))
- Cache skills prompt with shared `skill_utils` module for faster TTFT ([#3421](https://github.com/NousResearch/hermes-agent/pull/3421))
- Avoid redundant file re-read for skill conditions ([#2992](https://github.com/NousResearch/hermes-agent/pull/2992))
- Use Git Trees API to prevent silent subdirectory loss during install ([#2995](https://github.com/NousResearch/hermes-agent/pull/2995))
- Fix skills-sh install for deeply nested repo structures ([#2980](https://github.com/NousResearch/hermes-agent/pull/2980))
- Handle null metadata in skill frontmatter ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Preserve trust for skills-sh identifiers + reduce resolution churn ([#3251](https://github.com/NousResearch/hermes-agent/pull/3251))
- Agent-created skills were incorrectly treated as untrusted community content — fixed ([untagged commit](https://github.com/NousResearch/hermes-agent))

### New Skills
- **G0DM0D3 godmode jailbreaking skill** + docs ([#3157](https://github.com/NousResearch/hermes-agent/pull/3157))
- **Docker management skill** added to optional-skills ([#3060](https://github.com/NousResearch/hermes-agent/pull/3060))
- **OpenClaw migration v2** — 17 new modules, terminal recap for migrating from OpenClaw to Hermes ([#2906](https://github.com/NousResearch/hermes-agent/pull/2906))

---

## 🔒 Security & Reliability

### Security Hardening
- **SSRF protection** added to `browser_navigate` ([#3058](https://github.com/NousResearch/hermes-agent/pull/3058))
- **SSRF protection** added to `vision_tools` and `web_tools` (hardened) ([#2679](https://github.com/NousResearch/hermes-agent/pull/2679))
- **Restrict subagent toolsets** to parent's enabled set ([#3269](https://github.com/NousResearch/hermes-agent/pull/3269))
- **Prevent zip-slip path traversal** in self-update ([#3250](https://github.com/NousResearch/hermes-agent/pull/3250))
- **Prevent shell injection** in `_expand_path` via `~user` path suffix ([#2685](https://github.com/NousResearch/hermes-agent/pull/2685))
- **Normalize input** before dangerous command detection ([#3260](https://github.com/NousResearch/hermes-agent/pull/3260))
- Make tirith block verdicts approvable instead of hard-blocking ([#3428](https://github.com/NousResearch/hermes-agent/pull/3428))
- Remove compromised `litellm`/`typer`/`platformdirs` from deps ([#2796](https://github.com/NousResearch/hermes-agent/pull/2796))
- Pin all dependency version ranges ([#2810](https://github.com/NousResearch/hermes-agent/pull/2810))
- Regenerate `uv.lock` with hashes, use lockfile in setup ([#2812](https://github.com/NousResearch/hermes-agent/pull/2812))
- Bump dependencies to fix CVEs + regenerate `uv.lock` ([#3073](https://github.com/NousResearch/hermes-agent/pull/3073))
- Supply chain audit CI workflow for PR scanning ([#2816](https://github.com/NousResearch/hermes-agent/pull/2816))

### Reliability
- **SQLite WAL write-lock contention** causing 15-20s TUI freeze — fixed ([#3385](https://github.com/NousResearch/hermes-agent/pull/3385))
- **SQLite concurrency hardening** + session transcript integrity ([#3249](https://github.com/NousResearch/hermes-agent/pull/3249))
- Prevent recurring cron job re-fire on gateway crash/restart loop ([#3396](https://github.com/NousResearch/hermes-agent/pull/3396))
- Mark cron session as ended after job completes ([#2998](https://github.com/NousResearch/hermes-agent/pull/2998))

---

## ⚡ Performance

- **TTFT startup optimizations** — salvaged easy-win startup improvements ([#3395](https://github.com/NousResearch/hermes-agent/pull/3395))
- Cache skills prompt with shared `skill_utils` module ([#3421](https://github.com/NousResearch/hermes-agent/pull/3421))
- Avoid redundant file re-read for skill conditions in prompt builder ([#2992](https://github.com/NousResearch/hermes-agent/pull/2992))

---

## 🐛 Notable Bug Fixes

- Fix gateway token double-counting with cached agents ([#3306](https://github.com/NousResearch/hermes-agent/pull/3306), [#3317](https://github.com/NousResearch/hermes-agent/pull/3317))
- Fix "Event loop is closed" / "Press ENTER to continue" during idle sessions ([#3398](https://github.com/NousResearch/hermes-agent/pull/3398))
- Fix reasoning box rendering 3x during tool-calling loops ([#3405](https://github.com/NousResearch/hermes-agent/pull/3405))
- Fix status bar shows 26K instead of 260K for token counts ([#3024](https://github.com/NousResearch/hermes-agent/pull/3024))
- Fix `/queue` always working regardless of config ([#3298](https://github.com/NousResearch/hermes-agent/pull/3298))
- Fix phantom Discord typing indicator after agent turn ([#3003](https://github.com/NousResearch/hermes-agent/pull/3003))
- Fix Slack progress messages appearing in wrong thread ([#3063](https://github.com/NousResearch/hermes-agent/pull/3063))
- Fix WhatsApp media downloads (documents, audio, video) ([#2978](https://github.com/NousResearch/hermes-agent/pull/2978))
- Fix Telegram "Message thread not found" killing progress messages ([#3390](https://github.com/NousResearch/hermes-agent/pull/3390))
- Fix OpenClaw migration overwriting defaults ([#3282](https://github.com/NousResearch/hermes-agent/pull/3282))
- Fix returning-user setup menu dispatching wrong section ([#3083](https://github.com/NousResearch/hermes-agent/pull/3083))
- Fix `hermes update` PEP 668 "externally-managed-environment" error ([#3099](https://github.com/NousResearch/hermes-agent/pull/3099))
- Fix subagents hitting `max_iterations` prematurely via shared budget ([#3004](https://github.com/NousResearch/hermes-agent/pull/3004))
- Fix YAML boolean handling for `tool_progress` config ([#3300](https://github.com/NousResearch/hermes-agent/pull/3300))
- Fix `config.get()` crashes on YAML null values ([#3377](https://github.com/NousResearch/hermes-agent/pull/3377))
- Fix `.strip()` crash on None values from YAML config ([#3552](https://github.com/NousResearch/hermes-agent/pull/3552))
- Fix hung agents on gateway — `/stop` now hard-kills session lock ([#3104](https://github.com/NousResearch/hermes-agent/pull/3104))
- Fix `_custom` provider silently remapped to `openrouter` ([#2792](https://github.com/NousResearch/hermes-agent/pull/2792))
- Fix Matrix missing from `PLATFORMS` dict ([#3473](https://github.com/NousResearch/hermes-agent/pull/3473))
- Fix Email adapter unbounded `_seen_uids` growth ([#3490](https://github.com/NousResearch/hermes-agent/pull/3490))

---

## 🧪 Testing

- Pin `agent-client-protocol` < 0.9 to handle breaking upstream release ([#3320](https://github.com/NousResearch/hermes-agent/pull/3320))
- Catch anthropic ImportError in vision auto-detection tests ([#3312](https://github.com/NousResearch/hermes-agent/pull/3312))
- Update retry-exhaust test for new graceful return behavior ([#3320](https://github.com/NousResearch/hermes-agent/pull/3320))
- Add regression tests for null metadata frontmatter ([untagged commit](https://github.com/NousResearch/hermes-agent))

---

## 📚 Documentation

- Update all docs for `/model` command overhaul and custom provider support ([#2800](https://github.com/NousResearch/hermes-agent/pull/2800))
- Fix stale and incorrect documentation across 18 files ([#2805](https://github.com/NousResearch/hermes-agent/pull/2805))
- Document 9 previously undocumented features ([#2814](https://github.com/NousResearch/hermes-agent/pull/2814))
- Add missing skills, CLI commands, and messaging env vars to docs ([#2809](https://github.com/NousResearch/hermes-agent/pull/2809))
- Fix api-server response storage documentation — SQLite, not in-memory ([#2819](https://github.com/NousResearch/hermes-agent/pull/2819))
- Quote pip install extras to fix zsh glob errors ([#2815](https://github.com/NousResearch/hermes-agent/pull/2815))
- Unify hooks documentation — add plugin hooks to hooks page, add `session:end` event ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Clarify two-mode behavior in `session_search` schema description ([untagged commit](https://github.com/NousResearch/hermes-agent))
- Fix Discord Public Bot setting for Discord-provided invite link ([#3519](https://github.com/NousResearch/hermes-agent/pull/3519)) by @mehmoodosman
- Revise v0.4.0 changelog — fix feature attribution, reorder sections ([untagged commit](https://github.com/NousResearch/hermes-agent))

---

## 👥 Contributors

### Core
- **@teknium1** — 157 PRs covering the full scope of this release

### Community Contributors
- **@alt-glitch** (Siddharth Balyan) — 2 PRs: Nix flake with uv2nix build, NixOS module, and persistent container mode ([#20](https://github.com/NousResearch/hermes-agent/pull/20)); auto-generated config keys and suffix PATHs for Nix builds ([#3061](https://github.com/NousResearch/hermes-agent/pull/3061), [#3274](https://github.com/NousResearch/hermes-agent/pull/3274))
- **@ctlst** — 1 PR: Prevent AsyncOpenAI/httpx cross-loop deadlock in gateway mode ([#2701](https://github.com/NousResearch/hermes-agent/pull/2701))
- **@memosr** (memosr.eth) — 1 PR: Add request timeouts to `send_message_tool` HTTP calls ([#3162](https://github.com/NousResearch/hermes-agent/pull/3162))
- **@mehmoodosman** (Osman Mehmood) — 1 PR: Fix Discord docs for Public Bot setting ([#3519](https://github.com/NousResearch/hermes-agent/pull/3519))

### All Contributors
@alt-glitch, @ctlst, @mehmoodosman, @memosr, @teknium1

---

**Full Changelog**: [v2026.3.23...v2026.3.28](https://github.com/NousResearch/hermes-agent/compare/v2026.3.23...v2026.3.28)
