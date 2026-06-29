# Hermes Agent v0.2.0 (v2026.3.12)

**Release Date:** March 12, 2026

> First tagged release since v0.1.0 (the initial pre-public foundation). In just over two weeks, Hermes Agent went from a small internal project to a full-featured AI agent platform — thanks to an explosion of community contributions. This release covers **216 merged pull requests** from **63 contributors**, resolving **119 issues**.

---

## ✨ Highlights

- **Multi-Platform Messaging Gateway** — Telegram, Discord, Slack, WhatsApp, Signal, Email (IMAP/SMTP), and Home Assistant platforms with unified session management, media attachments, and per-platform tool configuration.

- **MCP (Model Context Protocol) Client** — Native MCP support with stdio and HTTP transports, reconnection, resource/prompt discovery, and sampling (server-initiated LLM requests). ([#291](https://github.com/NousResearch/hermes-agent/pull/291) — @0xbyt4, [#301](https://github.com/NousResearch/hermes-agent/pull/301), [#753](https://github.com/NousResearch/hermes-agent/pull/753))

- **Skills Ecosystem** — 70+ bundled and optional skills across 15+ categories with a Skills Hub for community discovery, per-platform enable/disable, conditional activation based on tool availability, and prerequisite validation. ([#743](https://github.com/NousResearch/hermes-agent/pull/743) — @teyrebaz33, [#785](https://github.com/NousResearch/hermes-agent/pull/785) — @teyrebaz33)

- **Centralized Provider Router** — Unified `call_llm()`/`async_call_llm()` API replaces scattered provider logic across vision, summarization, compression, and trajectory saving. All auxiliary consumers route through a single code path with automatic credential resolution. ([#1003](https://github.com/NousResearch/hermes-agent/pull/1003))

- **ACP Server** — VS Code, Zed, and JetBrains editor integration via the Agent Communication Protocol standard. ([#949](https://github.com/NousResearch/hermes-agent/pull/949))

- **CLI Skin/Theme Engine** — Data-driven visual customization: banners, spinners, colors, branding. 7 built-in skins + custom YAML skins.

- **Git Worktree Isolation** — `hermes -w` launches isolated agent sessions in git worktrees for safe parallel work on the same repo. ([#654](https://github.com/NousResearch/hermes-agent/pull/654))

- **Filesystem Checkpoints & Rollback** — Automatic snapshots before destructive operations with `/rollback` to restore. ([#824](https://github.com/NousResearch/hermes-agent/pull/824))

- **3,289 Tests** — From near-zero test coverage to a comprehensive test suite covering agent, gateway, tools, cron, and CLI.

---

## 🏗️ Core Agent & Architecture

### Provider & Model Support
- Centralized provider router with `resolve_provider_client()` + `call_llm()` API ([#1003](https://github.com/NousResearch/hermes-agent/pull/1003))
- Nous Portal as first-class provider in setup ([#644](https://github.com/NousResearch/hermes-agent/issues/644))
- OpenAI Codex (Responses API) with ChatGPT subscription support ([#43](https://github.com/NousResearch/hermes-agent/pull/43)) — @grp06
- Codex OAuth vision support + multimodal content adapter
- Validate `/model` against live API instead of hardcoded lists
- Self-hosted Firecrawl support ([#460](https://github.com/NousResearch/hermes-agent/pull/460)) — @caentzminger
- Kimi Code API support ([#635](https://github.com/NousResearch/hermes-agent/pull/635)) — @christomitov
- MiniMax model ID update ([#473](https://github.com/NousResearch/hermes-agent/pull/473)) — @tars90percent
- OpenRouter provider routing configuration (provider_preferences)
- Nous credential refresh on 401 errors ([#571](https://github.com/NousResearch/hermes-agent/pull/571), [#269](https://github.com/NousResearch/hermes-agent/pull/269)) — @rewbs
- z.ai/GLM, Kimi/Moonshot, MiniMax, Azure OpenAI as first-class providers
- Unified `/model` and `/provider` into single view

### Agent Loop & Conversation
- Simple fallback model for provider resilience ([#740](https://github.com/NousResearch/hermes-agent/pull/740))
- Shared iteration budget across parent + subagent delegation
- Iteration budget pressure via tool result injection
- Configurable subagent provider/model with full credential resolution
- Handle 413 payload-too-large via compression instead of aborting ([#153](https://github.com/NousResearch/hermes-agent/pull/153)) — @tekelala
- Retry with rebuilt payload after compression ([#616](https://github.com/NousResearch/hermes-agent/pull/616)) — @tripledoublev
- Auto-compress pathologically large gateway sessions ([#628](https://github.com/NousResearch/hermes-agent/issues/628))
- Tool call repair middleware — auto-lowercase and invalid tool handler
- Reasoning effort configuration and `/reasoning` command ([#921](https://github.com/NousResearch/hermes-agent/pull/921))
- Detect and block file re-read/search loops after context compression ([#705](https://github.com/NousResearch/hermes-agent/pull/705)) — @0xbyt4

### Session & Memory
- Session naming with unique titles, auto-lineage, rich listing, and resume by name ([#720](https://github.com/NousResearch/hermes-agent/pull/720))
- Interactive session browser with search filtering ([#733](https://github.com/NousResearch/hermes-agent/pull/733))
- Display previous messages when resuming a session ([#734](https://github.com/NousResearch/hermes-agent/pull/734))
- Honcho AI-native cross-session user modeling ([#38](https://github.com/NousResearch/hermes-agent/pull/38)) — @erosika
- Proactive async memory flush on session expiry
- Smart context length probing with persistent caching + banner display
- `/resume` command for switching to named sessions in gateway
- Session reset policy for messaging platforms

---

## 📱 Messaging Platforms (Gateway)

### Telegram
- Native file attachments: send_document + send_video
- Document file processing for PDF, text, and Office files — @tekelala
- Forum topic session isolation ([#766](https://github.com/NousResearch/hermes-agent/pull/766)) — @spanishflu-est1918
- Browser screenshot sharing via MEDIA: protocol ([#657](https://github.com/NousResearch/hermes-agent/pull/657))
- Location support for find-nearby skill
- TTS voice message accumulation fix ([#176](https://github.com/NousResearch/hermes-agent/pull/176)) — @Bartok9
- Improved error handling and logging ([#763](https://github.com/NousResearch/hermes-agent/pull/763)) — @aydnOktay
- Italic regex newline fix + 43 format tests ([#204](https://github.com/NousResearch/hermes-agent/pull/204)) — @0xbyt4

### Discord
- Channel topic included in session context ([#248](https://github.com/NousResearch/hermes-agent/pull/248)) — @Bartok9
- DISCORD_ALLOW_BOTS config for bot message filtering ([#758](https://github.com/NousResearch/hermes-agent/pull/758))
- Document and video support ([#784](https://github.com/NousResearch/hermes-agent/pull/784))
- Improved error handling and logging ([#761](https://github.com/NousResearch/hermes-agent/pull/761)) — @aydnOktay

### Slack
- App_mention 404 fix + document/video support ([#784](https://github.com/NousResearch/hermes-agent/pull/784))
- Structured logging replacing print statements — @aydnOktay

### WhatsApp
- Native media sending — images, videos, documents ([#292](https://github.com/NousResearch/hermes-agent/pull/292)) — @satelerd
- Multi-user session isolation ([#75](https://github.com/NousResearch/hermes-agent/pull/75)) — @satelerd
- Cross-platform port cleanup replacing Linux-only fuser ([#433](https://github.com/NousResearch/hermes-agent/pull/433)) — @Farukest
- DM interrupt key mismatch fix ([#350](https://github.com/NousResearch/hermes-agent/pull/350)) — @Farukest

### Signal
- Full Signal messenger gateway via signal-cli-rest-api ([#405](https://github.com/NousResearch/hermes-agent/issues/405))
- Media URL support in message events ([#871](https://github.com/NousResearch/hermes-agent/pull/871))

### Email (IMAP/SMTP)
- New email gateway platform — @0xbyt4

### Home Assistant
- REST tools + WebSocket gateway integration ([#184](https://github.com/NousResearch/hermes-agent/pull/184)) — @0xbyt4
- Service discovery and enhanced setup
- Toolset mapping fix ([#538](https://github.com/NousResearch/hermes-agent/pull/538)) — @Himess

### Gateway Core
- Expose subagent tool calls and thinking to users ([#186](https://github.com/NousResearch/hermes-agent/pull/186)) — @cutepawss
- Configurable background process watcher notifications ([#840](https://github.com/NousResearch/hermes-agent/pull/840))
- `edit_message()` for Telegram/Discord/Slack with fallback
- `/compress`, `/usage`, `/update` slash commands
- Eliminated 3x SQLite message duplication in gateway sessions ([#873](https://github.com/NousResearch/hermes-agent/pull/873))
- Stabilize system prompt across gateway turns for cache hits ([#754](https://github.com/NousResearch/hermes-agent/pull/754))
- MCP server shutdown on gateway exit ([#796](https://github.com/NousResearch/hermes-agent/pull/796)) — @0xbyt4
- Pass session_db to AIAgent, fixing session_search error ([#108](https://github.com/NousResearch/hermes-agent/pull/108)) — @Bartok9
- Persist transcript changes in /retry, /undo; fix /reset attribute ([#217](https://github.com/NousResearch/hermes-agent/pull/217)) — @Farukest
- UTF-8 encoding fix preventing Windows crashes ([#369](https://github.com/NousResearch/hermes-agent/pull/369)) — @ch3ronsa

---

## 🖥️ CLI & User Experience

### Interactive CLI
- Data-driven skin/theme engine — 7 built-in skins (default, ares, mono, slate, poseidon, sisyphus, charizard) + custom YAML skins
- `/personality` command with custom personality + disable support ([#773](https://github.com/NousResearch/hermes-agent/pull/773)) — @teyrebaz33
- User-defined quick commands that bypass the agent loop ([#746](https://github.com/NousResearch/hermes-agent/pull/746)) — @teyrebaz33
- `/reasoning` command for effort level and display toggle ([#921](https://github.com/NousResearch/hermes-agent/pull/921))
- `/verbose` slash command to toggle debug at runtime ([#94](https://github.com/NousResearch/hermes-agent/pull/94)) — @cesareth
- `/insights` command — usage analytics, cost estimation & activity patterns ([#552](https://github.com/NousResearch/hermes-agent/pull/552))
- `/background` command for managing background processes
- `/help` formatting with command categories
- Bell-on-complete — terminal bell when agent finishes ([#738](https://github.com/NousResearch/hermes-agent/pull/738))
- Up/down arrow history navigation
- Clipboard image paste (Alt+V / Ctrl+V)
- Loading indicators for slow slash commands ([#882](https://github.com/NousResearch/hermes-agent/pull/882))
- Spinner flickering fix under patch_stdout ([#91](https://github.com/NousResearch/hermes-agent/pull/91)) — @0xbyt4
- `--quiet/-Q` flag for programmatic single-query mode
- `--fuck-it-ship-it` flag to bypass all approval prompts ([#724](https://github.com/NousResearch/hermes-agent/pull/724)) — @dmahan93
- Tools summary flag ([#767](https://github.com/NousResearch/hermes-agent/pull/767)) — @luisv-1
- Terminal blinking fix on SSH ([#284](https://github.com/NousResearch/hermes-agent/pull/284)) — @ygd58
- Multi-line paste detection fix ([#84](https://github.com/NousResearch/hermes-agent/pull/84)) — @0xbyt4

### Setup & Configuration
- Modular setup wizard with section subcommands and tool-first UX
- Container resource configuration prompts
- Backend validation for required binaries
- Config migration system (currently v7)
- API keys properly routed to .env instead of config.yaml ([#469](https://github.com/NousResearch/hermes-agent/pull/469)) — @ygd58
- Atomic write for .env to prevent API key loss on crash ([#954](https://github.com/NousResearch/hermes-agent/pull/954))
- `hermes tools` — per-platform tool enable/disable with curses UI
- `hermes doctor` for health checks across all configured providers
- `hermes update` with auto-restart for gateway service
- Show update-available notice in CLI banner
- Multiple named custom providers
- Shell config detection improvement for PATH setup ([#317](https://github.com/NousResearch/hermes-agent/pull/317)) — @mehmetkr-31
- Consistent HERMES_HOME and .env path resolution ([#51](https://github.com/NousResearch/hermes-agent/pull/51), [#48](https://github.com/NousResearch/hermes-agent/pull/48)) — @deankerr
- Docker backend fix on macOS + subagent auth for Nous Portal ([#46](https://github.com/NousResearch/hermes-agent/pull/46)) — @rsavitt

---

## 🔧 Tool System

### MCP (Model Context Protocol)
- Native MCP client with stdio + HTTP transports ([#291](https://github.com/NousResearch/hermes-agent/pull/291) — @0xbyt4, [#301](https://github.com/NousResearch/hermes-agent/pull/301))
- Sampling support — server-initiated LLM requests ([#753](https://github.com/NousResearch/hermes-agent/pull/753))
- Resource and prompt discovery
- Automatic reconnection and security hardening
- Banner integration, `/reload-mcp` command
- `hermes tools` UI integration

### Browser
- Local browser backend — zero-cost headless Chromium (no Browserbase needed)
- Console/errors tool, annotated screenshots, auto-recording, dogfood QA skill ([#745](https://github.com/NousResearch/hermes-agent/pull/745))
- Screenshot sharing via MEDIA: on all messaging platforms ([#657](https://github.com/NousResearch/hermes-agent/pull/657))

### Terminal & Execution
- `execute_code` sandbox with json_parse, shell_quote, retry helpers
- Docker: custom volume mounts ([#158](https://github.com/NousResearch/hermes-agent/pull/158)) — @Indelwin
- Daytona cloud sandbox backend ([#451](https://github.com/NousResearch/hermes-agent/pull/451)) — @rovle
- SSH backend fix ([#59](https://github.com/NousResearch/hermes-agent/pull/59)) — @deankerr
- Shell noise filtering and login shell execution for environment consistency
- Head+tail truncation for execute_code stdout overflow
- Configurable background process notification modes

### File Operations
- Filesystem checkpoints and `/rollback` command ([#824](https://github.com/NousResearch/hermes-agent/pull/824))
- Structured tool result hints (next-action guidance) for patch and search_files ([#722](https://github.com/NousResearch/hermes-agent/issues/722))
- Docker volumes passed to sandbox container config ([#687](https://github.com/NousResearch/hermes-agent/pull/687)) — @manuelschipper

---

## 🧩 Skills Ecosystem

### Skills System
- Per-platform skill enable/disable ([#743](https://github.com/NousResearch/hermes-agent/pull/743)) — @teyrebaz33
- Conditional skill activation based on tool availability ([#785](https://github.com/NousResearch/hermes-agent/pull/785)) — @teyrebaz33
- Skill prerequisites — hide skills with unmet dependencies ([#659](https://github.com/NousResearch/hermes-agent/pull/659)) — @kshitijk4poor
- Optional skills — shipped but not activated by default
- `hermes skills browse` — paginated hub browsing
- Skills sub-category organization
- Platform-conditional skill loading
- Atomic skill file writes ([#551](https://github.com/NousResearch/hermes-agent/pull/551)) — @aydnOktay
- Skills sync data loss prevention ([#563](https://github.com/NousResearch/hermes-agent/pull/563)) — @0xbyt4
- Dynamic skill slash commands for CLI and gateway

### New Skills (selected)
- **ASCII Art** — pyfiglet (571 fonts), cowsay, image-to-ascii ([#209](https://github.com/NousResearch/hermes-agent/pull/209)) — @0xbyt4
- **ASCII Video** — Full production pipeline ([#854](https://github.com/NousResearch/hermes-agent/pull/854)) — @SHL0MS
- **DuckDuckGo Search** — Firecrawl fallback ([#267](https://github.com/NousResearch/hermes-agent/pull/267)) — @gamedevCloudy; DDGS API expansion ([#598](https://github.com/NousResearch/hermes-agent/pull/598)) — @areu01or00
- **Solana Blockchain** — Wallet balances, USD pricing, token names ([#212](https://github.com/NousResearch/hermes-agent/pull/212)) — @gizdusum
- **AgentMail** — Agent-owned email inboxes ([#330](https://github.com/NousResearch/hermes-agent/pull/330)) — @teyrebaz33
- **Polymarket** — Prediction market data (read-only) ([#629](https://github.com/NousResearch/hermes-agent/pull/629))
- **OpenClaw Migration** — Official migration tool ([#570](https://github.com/NousResearch/hermes-agent/pull/570)) — @unmodeled-tyler
- **Domain Intelligence** — Passive recon: subdomains, SSL, WHOIS, DNS ([#136](https://github.com/NousResearch/hermes-agent/pull/136)) — @FurkanL0
- **Superpowers** — Software development skills ([#137](https://github.com/NousResearch/hermes-agent/pull/137)) — @kaos35
- **Hermes-Atropos** — RL environment development skill ([#815](https://github.com/NousResearch/hermes-agent/pull/815))
- Plus: arXiv search, OCR/documents, Excalidraw diagrams, YouTube transcripts, GIF search, Pokémon player, Minecraft modpack server, OpenHue (Philips Hue), Google Workspace, Notion, PowerPoint, Obsidian, find-nearby, and 40+ MLOps skills

---

## 🔒 Security & Reliability

### Security Hardening
- Path traversal fix in skill_view — prevented reading arbitrary files ([#220](https://github.com/NousResearch/hermes-agent/issues/220)) — @Farukest
- Shell injection prevention in sudo password piping ([#65](https://github.com/NousResearch/hermes-agent/pull/65)) — @leonsgithub
- Dangerous command detection: multiline bypass fix ([#233](https://github.com/NousResearch/hermes-agent/pull/233)) — @Farukest; tee/process substitution patterns ([#280](https://github.com/NousResearch/hermes-agent/pull/280)) — @dogiladeveloper
- Symlink boundary check fix in skills_guard ([#386](https://github.com/NousResearch/hermes-agent/pull/386)) — @Farukest
- Symlink bypass fix in write deny list on macOS ([#61](https://github.com/NousResearch/hermes-agent/pull/61)) — @0xbyt4
- Multi-word prompt injection bypass prevention ([#192](https://github.com/NousResearch/hermes-agent/pull/192)) — @0xbyt4
- Cron prompt injection scanner bypass fix ([#63](https://github.com/NousResearch/hermes-agent/pull/63)) — @0xbyt4
- Enforce 0600/0700 file permissions on sensitive files ([#757](https://github.com/NousResearch/hermes-agent/pull/757))
- .env file permissions restricted to owner-only ([#529](https://github.com/NousResearch/hermes-agent/pull/529)) — @Himess
- `--force` flag properly blocked from overriding dangerous verdicts ([#388](https://github.com/NousResearch/hermes-agent/pull/388)) — @Farukest
- FTS5 query sanitization + DB connection leak fix ([#565](https://github.com/NousResearch/hermes-agent/pull/565)) — @0xbyt4
- Expand secret redaction patterns + config toggle to disable
- In-memory permanent allowlist to prevent data leak ([#600](https://github.com/NousResearch/hermes-agent/pull/600)) — @alireza78a

### Atomic Writes (data loss prevention)
- sessions.json ([#611](https://github.com/NousResearch/hermes-agent/pull/611)) — @alireza78a
- Cron jobs ([#146](https://github.com/NousResearch/hermes-agent/pull/146)) — @alireza78a
- .env config ([#954](https://github.com/NousResearch/hermes-agent/pull/954))
- Process checkpoints ([#298](https://github.com/NousResearch/hermes-agent/pull/298)) — @aydnOktay
- Batch runner ([#297](https://github.com/NousResearch/hermes-agent/pull/297)) — @aydnOktay
- Skill files ([#551](https://github.com/NousResearch/hermes-agent/pull/551)) — @aydnOktay

### Reliability
- Guard all print() against OSError for systemd/headless environments ([#963](https://github.com/NousResearch/hermes-agent/pull/963))
- Reset all retry counters at start of run_conversation ([#607](https://github.com/NousResearch/hermes-agent/pull/607)) — @0xbyt4
- Return deny on approval callback timeout instead of None ([#603](https://github.com/NousResearch/hermes-agent/pull/603)) — @0xbyt4
- Fix None message content crashes across codebase ([#277](https://github.com/NousResearch/hermes-agent/pull/277))
- Fix context overrun crash with local LLM backends ([#403](https://github.com/NousResearch/hermes-agent/pull/403)) — @ch3ronsa
- Prevent `_flush_sentinel` from leaking to external APIs ([#227](https://github.com/NousResearch/hermes-agent/pull/227)) — @Farukest
- Prevent conversation_history mutation in callers ([#229](https://github.com/NousResearch/hermes-agent/pull/229)) — @Farukest
- Fix systemd restart loop ([#614](https://github.com/NousResearch/hermes-agent/pull/614)) — @voidborne-d
- Close file handles and sockets to prevent fd leaks ([#568](https://github.com/NousResearch/hermes-agent/pull/568) — @alireza78a, [#296](https://github.com/NousResearch/hermes-agent/pull/296) — @alireza78a, [#709](https://github.com/NousResearch/hermes-agent/pull/709) — @memosr)
- Prevent data loss in clipboard PNG conversion ([#602](https://github.com/NousResearch/hermes-agent/pull/602)) — @0xbyt4
- Eliminate shell noise from terminal output ([#293](https://github.com/NousResearch/hermes-agent/pull/293)) — @0xbyt4
- Timezone-aware now() for prompt, cron, and execute_code ([#309](https://github.com/NousResearch/hermes-agent/pull/309)) — @areu01or00

### Windows Compatibility
- Guard POSIX-only process functions ([#219](https://github.com/NousResearch/hermes-agent/pull/219)) — @Farukest
- Windows native support via Git Bash + ZIP-based update fallback
- pywinpty for PTY support ([#457](https://github.com/NousResearch/hermes-agent/pull/457)) — @shitcoinsherpa
- Explicit UTF-8 encoding on all config/data file I/O ([#458](https://github.com/NousResearch/hermes-agent/pull/458)) — @shitcoinsherpa
- Windows-compatible path handling ([#354](https://github.com/NousResearch/hermes-agent/pull/354), [#390](https://github.com/NousResearch/hermes-agent/pull/390)) — @Farukest
- Regex-based search output parsing for drive-letter paths ([#533](https://github.com/NousResearch/hermes-agent/pull/533)) — @Himess
- Auth store file lock for Windows ([#455](https://github.com/NousResearch/hermes-agent/pull/455)) — @shitcoinsherpa

---

## 🐛 Notable Bug Fixes

- Fix DeepSeek V3 tool call parser silently dropping multi-line JSON arguments ([#444](https://github.com/NousResearch/hermes-agent/pull/444)) — @PercyDikec
- Fix gateway transcript losing 1 message per turn due to offset mismatch ([#395](https://github.com/NousResearch/hermes-agent/pull/395)) — @PercyDikec
- Fix /retry command silently discarding the agent's final response ([#441](https://github.com/NousResearch/hermes-agent/pull/441)) — @PercyDikec
- Fix max-iterations retry returning empty string after think-block stripping ([#438](https://github.com/NousResearch/hermes-agent/pull/438)) — @PercyDikec
- Fix max-iterations retry using hardcoded max_tokens ([#436](https://github.com/NousResearch/hermes-agent/pull/436)) — @Farukest
- Fix Codex status dict key mismatch ([#448](https://github.com/NousResearch/hermes-agent/pull/448)) and visibility filter ([#446](https://github.com/NousResearch/hermes-agent/pull/446)) — @PercyDikec
- Strip \<think\> blocks from final user-facing responses ([#174](https://github.com/NousResearch/hermes-agent/pull/174)) — @Bartok9
- Fix \<think\> block regex stripping visible content when model discusses tags literally ([#786](https://github.com/NousResearch/hermes-agent/issues/786))
- Fix Mistral 422 errors from leftover finish_reason in assistant messages ([#253](https://github.com/NousResearch/hermes-agent/pull/253)) — @Sertug17
- Fix OPENROUTER_API_KEY resolution order across all code paths ([#295](https://github.com/NousResearch/hermes-agent/pull/295)) — @0xbyt4
- Fix OPENAI_BASE_URL API key priority ([#420](https://github.com/NousResearch/hermes-agent/pull/420)) — @manuelschipper
- Fix Anthropic "prompt is too long" 400 error not detected as context length error ([#813](https://github.com/NousResearch/hermes-agent/issues/813))
- Fix SQLite session transcript accumulating duplicate messages — 3-4x token inflation ([#860](https://github.com/NousResearch/hermes-agent/issues/860))
- Fix setup wizard skipping API key prompts on first install ([#748](https://github.com/NousResearch/hermes-agent/pull/748))
- Fix setup wizard showing OpenRouter model list for Nous Portal ([#575](https://github.com/NousResearch/hermes-agent/pull/575)) — @PercyDikec
- Fix provider selection not persisting when switching via hermes model ([#881](https://github.com/NousResearch/hermes-agent/pull/881))
- Fix Docker backend failing when docker not in PATH on macOS ([#889](https://github.com/NousResearch/hermes-agent/pull/889))
- Fix ClawHub Skills Hub adapter for API endpoint changes ([#286](https://github.com/NousResearch/hermes-agent/pull/286)) — @BP602
- Fix Honcho auto-enable when API key is present ([#243](https://github.com/NousResearch/hermes-agent/pull/243)) — @Bartok9
- Fix duplicate 'skills' subparser crash on Python 3.11+ ([#898](https://github.com/NousResearch/hermes-agent/issues/898))
- Fix memory tool entry parsing when content contains section sign ([#162](https://github.com/NousResearch/hermes-agent/pull/162)) — @aydnOktay
- Fix piped install silently aborting when interactive prompts fail ([#72](https://github.com/NousResearch/hermes-agent/pull/72)) — @cutepawss
- Fix false positives in recursive delete detection ([#68](https://github.com/NousResearch/hermes-agent/pull/68)) — @cutepawss
- Fix Ruff lint warnings across codebase ([#608](https://github.com/NousResearch/hermes-agent/pull/608)) — @JackTheGit
- Fix Anthropic native base URL fail-fast ([#173](https://github.com/NousResearch/hermes-agent/pull/173)) — @adavyas
- Fix install.sh creating ~/.hermes before moving Node.js directory ([#53](https://github.com/NousResearch/hermes-agent/pull/53)) — @JoshuaMart
- Fix SystemExit traceback during atexit cleanup on Ctrl+C ([#55](https://github.com/NousResearch/hermes-agent/pull/55)) — @bierlingm
- Restore missing MIT license file ([#620](https://github.com/NousResearch/hermes-agent/pull/620)) — @stablegenius49

---

## 🧪 Testing

- **3,289 tests** across agent, gateway, tools, cron, and CLI
- Parallelized test suite with pytest-xdist ([#802](https://github.com/NousResearch/hermes-agent/pull/802)) — @OutThisLife
- Unit tests batch 1: 8 core modules ([#60](https://github.com/NousResearch/hermes-agent/pull/60)) — @0xbyt4
- Unit tests batch 2: 8 more modules ([#62](https://github.com/NousResearch/hermes-agent/pull/62)) — @0xbyt4
- Unit tests batch 3: 8 untested modules ([#191](https://github.com/NousResearch/hermes-agent/pull/191)) — @0xbyt4
- Unit tests batch 4: 5 security/logic-critical modules ([#193](https://github.com/NousResearch/hermes-agent/pull/193)) — @0xbyt4
- AIAgent (run_agent.py) unit tests ([#67](https://github.com/NousResearch/hermes-agent/pull/67)) — @0xbyt4
- Trajectory compressor tests ([#203](https://github.com/NousResearch/hermes-agent/pull/203)) — @0xbyt4
- Clarify tool tests ([#121](https://github.com/NousResearch/hermes-agent/pull/121)) — @Bartok9
- Telegram format tests — 43 tests for italic/bold/code rendering ([#204](https://github.com/NousResearch/hermes-agent/pull/204)) — @0xbyt4
- Vision tools type hints + 42 tests ([#792](https://github.com/NousResearch/hermes-agent/pull/792))
- Compressor tool-call boundary regression tests ([#648](https://github.com/NousResearch/hermes-agent/pull/648)) — @intertwine
- Test structure reorganization ([#34](https://github.com/NousResearch/hermes-agent/pull/34)) — @0xbyt4
- Shell noise elimination + fix 36 test failures ([#293](https://github.com/NousResearch/hermes-agent/pull/293)) — @0xbyt4

---

## 🔬 RL & Evaluation Environments

- WebResearchEnv — Multi-step web research RL environment ([#434](https://github.com/NousResearch/hermes-agent/pull/434)) — @jackx707
- Modal sandbox concurrency limits to avoid deadlocks ([#621](https://github.com/NousResearch/hermes-agent/pull/621)) — @voteblake
- Hermes-atropos-environments bundled skill ([#815](https://github.com/NousResearch/hermes-agent/pull/815))
- Local vLLM instance support for evaluation — @dmahan93
- YC-Bench long-horizon agent benchmark environment
- OpenThoughts-TBLite evaluation environment and scripts

---

## 📚 Documentation

- Full documentation website (Docusaurus) with 37+ pages
- Comprehensive platform setup guides for Telegram, Discord, Slack, WhatsApp, Signal, Email
- AGENTS.md — development guide for AI coding assistants
- CONTRIBUTING.md ([#117](https://github.com/NousResearch/hermes-agent/pull/117)) — @Bartok9
- Slash commands reference ([#142](https://github.com/NousResearch/hermes-agent/pull/142)) — @Bartok9
- Comprehensive AGENTS.md accuracy audit ([#732](https://github.com/NousResearch/hermes-agent/pull/732))
- Skin/theme system documentation
- MCP documentation and examples
- Docs accuracy audit — 35+ corrections
- Documentation typo fixes ([#825](https://github.com/NousResearch/hermes-agent/pull/825), [#439](https://github.com/NousResearch/hermes-agent/pull/439)) — @JackTheGit
- CLI config precedence and terminology standardization ([#166](https://github.com/NousResearch/hermes-agent/pull/166), [#167](https://github.com/NousResearch/hermes-agent/pull/167), [#168](https://github.com/NousResearch/hermes-agent/pull/168)) — @Jr-kenny
- Telegram token regex documentation ([#713](https://github.com/NousResearch/hermes-agent/pull/713)) — @VolodymyrBg

---

## 👥 Contributors

Thank you to the 63 contributors who made this release possible! In just over two weeks, the Hermes Agent community came together to ship an extraordinary amount of work.

### Core
- **@teknium1** — 43 PRs: Project lead, core architecture, provider router, sessions, skills, CLI, documentation

### Top Community Contributors
- **@0xbyt4** — 40 PRs: MCP client, Home Assistant, security fixes (symlink, prompt injection, cron), extensive test coverage (6 batches), ascii-art skill, shell noise elimination, skills sync, Telegram formatting, and dozens more
- **@Farukest** — 16 PRs: Security hardening (path traversal, dangerous command detection, symlink boundary), Windows compatibility (POSIX guards, path handling), WhatsApp fixes, max-iterations retry, gateway fixes
- **@aydnOktay** — 11 PRs: Atomic writes (process checkpoints, batch runner, skill files), error handling improvements across Telegram, Discord, code execution, transcription, TTS, and skills
- **@Bartok9** — 9 PRs: CONTRIBUTING.md, slash commands reference, Discord channel topics, think-block stripping, TTS fix, Honcho fix, session count fix, clarify tests
- **@PercyDikec** — 7 PRs: DeepSeek V3 parser fix, /retry response discard, gateway transcript offset, Codex status/visibility, max-iterations retry, setup wizard fix
- **@teyrebaz33** — 5 PRs: Skills enable/disable system, quick commands, personality customization, conditional skill activation
- **@alireza78a** — 5 PRs: Atomic writes (cron, sessions), fd leak prevention, security allowlist, code execution socket cleanup
- **@shitcoinsherpa** — 3 PRs: Windows support (pywinpty, UTF-8 encoding, auth store lock)
- **@Himess** — 3 PRs: Cron/HomeAssistant/Daytona fix, Windows drive-letter parsing, .env permissions
- **@satelerd** — 2 PRs: WhatsApp native media, multi-user session isolation
- **@rovle** — 1 PR: Daytona cloud sandbox backend (4 commits)
- **@erosika** — 1 PR: Honcho AI-native memory integration
- **@dmahan93** — 1 PR: --fuck-it-ship-it flag + RL environment work
- **@SHL0MS** — 1 PR: ASCII video skill

### All Contributors
@0xbyt4, @BP602, @Bartok9, @Farukest, @FurkanL0, @Himess, @Indelwin, @JackTheGit, @JoshuaMart, @Jr-kenny, @OutThisLife, @PercyDikec, @SHL0MS, @Sertug17, @VencentSoliman, @VolodymyrBg, @adavyas, @alireza78a, @areu01or00, @aydnOktay, @batuhankocyigit, @bierlingm, @caentzminger, @cesareth, @ch3ronsa, @christomitov, @cutepawss, @deankerr, @dmahan93, @dogiladeveloper, @dragonkhoi, @erosika, @gamedevCloudy, @gizdusum, @grp06, @intertwine, @jackx707, @jdblackstar, @johnh4098, @kaos35, @kshitijk4poor, @leonsgithub, @luisv-1, @manuelschipper, @mehmetkr-31, @memosr, @PeterFile, @rewbs, @rovle, @rsavitt, @satelerd, @spanishflu-est1918, @stablegenius49, @tars90percent, @tekelala, @teknium1, @teyrebaz33, @tripledoublev, @unmodeled-tyler, @voidborne-d, @voteblake, @ygd58

---

**Full Changelog**: [v0.1.0...v2026.3.12](https://github.com/NousResearch/hermes-agent/compare/v0.1.0...v2026.3.12)
