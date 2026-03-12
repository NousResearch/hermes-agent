# Hermes Agent v0.2.0 (v2026.3.12)

**Release Date:** March 12, 2026

> 🎉 **First official tagged release!** Hermes Agent has been in active development since July 2025, with 1,388 commits from 74+ contributors across 231 merged pull requests. This release marks the beginning of regular weekly releases. Everything below represents the full feature set shipping today.

---

## ✨ Highlights

- **Multi-Platform Messaging Gateway** — Run Hermes Agent on Telegram, Discord, Slack, WhatsApp, Signal, Email (IMAP/SMTP), and Home Assistant, all from a single codebase with unified session management, conversation persistence, and per-platform tool configuration.

- **MCP (Model Context Protocol) Client** — Full native MCP support with stdio and HTTP transports, server reconnection, resource/prompt discovery, sampling (server-initiated LLM requests), and `hermes tools` UI integration. ([#291](https://github.com/NousResearch/hermes-agent/pull/291), [#301](https://github.com/NousResearch/hermes-agent/pull/301), [#753](https://github.com/NousResearch/hermes-agent/pull/753)) — @0xbyt4

- **Skills Ecosystem** — 70+ bundled and optional skills across 15+ categories (research, creative, gaming, smart-home, productivity, MLOps, and more). Skills are data-driven Markdown files with YAML frontmatter — the agent loads them dynamically based on task context. Includes a Skills Hub for community discovery and per-platform skill enable/disable. ([#743](https://github.com/NousResearch/hermes-agent/pull/743)) — @teyrebaz33

- **Centralized Provider Router** — Unified `resolve_provider_client()` + `call_llm()`/`async_call_llm()` API replaces scattered provider logic. All auxiliary consumers (vision, summarization, context compression, trajectory saving) route through a single code path with automatic credential resolution. ([#1003](https://github.com/NousResearch/hermes-agent/pull/1003))

- **ACP (Agent Communication Protocol) Server** — VS Code, Zed, and JetBrains editor integration via the agent-protocol standard. ([#949](https://github.com/NousResearch/hermes-agent/pull/949))

- **Reinforcement Learning Environments** — Atropos-compatible RL training environments: TerminalBench2 (tool-calling), WebResearchEnv (multi-step web research), YC-Bench (long-horizon agent benchmark), and OpenThoughts-TBLite evaluation. ([#17](https://github.com/NousResearch/hermes-agent/pull/17), [#434](https://github.com/NousResearch/hermes-agent/pull/434)) — @dmahan93, @jackx707

- **Git Worktree Isolation** — `hermes -w` launches isolated agent sessions in git worktrees, enabling safe parallel work on the same repo without conflicts. ([#654](https://github.com/NousResearch/hermes-agent/pull/654))

---

## 🏗️ Core Agent & Architecture

### Agent Loop & Conversation
- Shared iteration budget across parent + subagent delegation to prevent runaway chains
- Iteration budget pressure via tool result injection — agent gets warned as it approaches limits
- Configurable subagent provider/model with full credential resolution ([#609](https://github.com/NousResearch/hermes-agent/issues/609))
- Fallback model for provider resilience — automatic retry on a different model when primary fails ([#740](https://github.com/NousResearch/hermes-agent/pull/740), [#454](https://github.com/NousResearch/hermes-agent/issues/454))
- Context compression improvements: retry with rebuilt payload after compression ([#616](https://github.com/NousResearch/hermes-agent/pull/616)) — @tripledoublev; regression tests for tool-call boundary handling ([#648](https://github.com/NousResearch/hermes-agent/pull/648)) — @intertwine
- Auto-compress pathologically large gateway sessions ([#628](https://github.com/NousResearch/hermes-agent/issues/628))
- Handle 413 payload-too-large via compression instead of aborting ([#153](https://github.com/NousResearch/hermes-agent/pull/153)) — @tekelala
- Tool call repair middleware — auto-lowercase and invalid tool handler ([#520](https://github.com/NousResearch/hermes-agent/issues/520))
- Reasoning effort configuration and `/reasoning` command for effort levels + display toggle ([#921](https://github.com/NousResearch/hermes-agent/pull/921))
- Default reasoning effort tuned from xhigh to medium

### Provider & Model Support
- **First-class providers:** OpenRouter, OpenAI, Anthropic, Nous Portal, Codex (OpenAI Responses API), Google Gemini, z.ai/GLM, Kimi/Moonshot, MiniMax, DeepSeek, Azure OpenAI, custom endpoints
- Nous Portal as first-class provider option in setup ([#644](https://github.com/NousResearch/hermes-agent/issues/644)) — @Indelwin
- OpenAI Codex (Responses API) with OAuth support, ChatGPT subscription Codex ([#43](https://github.com/NousResearch/hermes-agent/pull/43)) — @grp06
- Codex OAuth vision support + multimodal content adapter
- Validate `/model` against live API instead of hardcoded lists
- Support for self-hosted Firecrawl instances ([#460](https://github.com/NousResearch/hermes-agent/pull/460)) — @caentzminger
- OpenRouter provider routing configuration (provider_preferences)
- Nous credential refresh on 401 errors ([#571](https://github.com/NousResearch/hermes-agent/pull/571), [#269](https://github.com/NousResearch/hermes-agent/pull/269)) — @rewbs
- Dynamic max tokens handling for various providers
- Kimi Code API support ([#635](https://github.com/NousResearch/hermes-agent/pull/635)) — @christomitov

### Session & Memory
- Session naming with unique titles, auto-lineage, rich listing, and resume by name ([#720](https://github.com/NousResearch/hermes-agent/pull/720))
- Interactive session browser with search filtering ([#733](https://github.com/NousResearch/hermes-agent/pull/733))
- Display previous messages when resuming a session in CLI ([#734](https://github.com/NousResearch/hermes-agent/pull/734))
- Proactive async memory flush on session expiry
- Session reset policy for messaging platforms
- Honcho AI-native cross-session user modeling integration ([#38](https://github.com/NousResearch/hermes-agent/pull/38)) — @Erosika
- `/resume` command for switching to named sessions in gateway
- Smart context length probing with persistent caching + banner display

---

## 📱 Messaging Platforms (Gateway)

### Telegram
- Native file attachments: send_document + send_video ([#779](https://github.com/NousResearch/hermes-agent/pull/779))
- Document file processing for PDF, text, and Office files ([#153](https://github.com/NousResearch/hermes-agent/pull/153)) — @tekelala
- Forum topic session isolation ([#766](https://github.com/NousResearch/hermes-agent/pull/766)) — @spanishflu-est1918
- Browser screenshot sharing via MEDIA: protocol
- Location support for find-nearby skill
- TTS voice message fix — prevent accumulation across turns ([#176](https://github.com/NousResearch/hermes-agent/pull/176)) — @Bartok9
- Improved error handling and logging ([#763](https://github.com/NousResearch/hermes-agent/pull/763)) — @aydnOktay

### Discord
- Thread-aware free-response routing ([#insecurejezza](https://github.com/NousResearch/hermes-agent/pull/insecurejezza))
- Channel topic included in session context ([#248](https://github.com/NousResearch/hermes-agent/pull/248)) — @Bartok9
- DISCORD_ALLOW_BOTS config for bot message filtering ([#758](https://github.com/NousResearch/hermes-agent/pull/758))
- Improved error handling and logging ([#761](https://github.com/NousResearch/hermes-agent/pull/761)) — @aydnOktay
- Document and video support ([#784](https://github.com/NousResearch/hermes-agent/pull/784))

### Slack
- App_mention fix + document/video support ([#784](https://github.com/NousResearch/hermes-agent/pull/784))
- Structured logging replacing print statements — @aydnOktay

### WhatsApp
- Native media sending — images, videos, documents ([#292](https://github.com/NousResearch/hermes-agent/pull/292)) — @satelerd
- Consolidate tool progress into single editable message — @satelerd
- Multi-user session isolation and bridge message handling ([#75](https://github.com/NousResearch/hermes-agent/pull/75)) — @satelerd
- Cross-platform port cleanup replacing Linux-only fuser ([#433](https://github.com/NousResearch/hermes-agent/pull/433)) — @Farukest

### Signal
- Full Signal messenger gateway via signal-cli-rest-api ([#405](https://github.com/NousResearch/hermes-agent/issues/405))
- Media URL support in message events ([#871](https://github.com/NousResearch/hermes-agent/pull/871))

### Email (IMAP/SMTP)
- New email gateway platform ([#291 area](https://github.com/NousResearch/hermes-agent/pull/291)) — @0xbyt4

### Home Assistant
- REST tools + WebSocket gateway integration ([#184](https://github.com/NousResearch/hermes-agent/pull/184)) — @0xbyt4
- Service discovery and enhanced setup

### Gateway Core
- Configurable background process watcher notifications: all, result, error, off ([#840](https://github.com/NousResearch/hermes-agent/pull/840), [#592](https://github.com/NousResearch/hermes-agent/issues/592))
- Expose subagent tool calls and thinking to users ([#186](https://github.com/NousResearch/hermes-agent/pull/186)) — @cutepawss
- `/compress`, `/usage`, `/update` slash commands for conversation management
- `edit_message()` for Telegram/Discord/Slack with fallback
- Session transcript deduplication fix — eliminated 3x SQLite message inflation ([#873](https://github.com/NousResearch/hermes-agent/pull/873))
- MCP server shutdown on gateway exit ([#796](https://github.com/NousResearch/hermes-agent/pull/796)) — @0xbyt4
- Stable system prompt across gateway turns for cache hits ([#754](https://github.com/NousResearch/hermes-agent/pull/754))

---

## 🖥️ CLI & User Experience

### Interactive CLI
- Data-driven skin/theme engine for CLI customization — banners, spinners, colors, branding
- Built-in skins: default (gold/kawaii), ares (crimson war-god), mono (grayscale), slate (cool blue), poseidon, sisyphus, charizard, and custom YAML skins
- `/personality` command with custom personality support + ability to disable default personality ([#773](https://github.com/NousResearch/hermes-agent/pull/773)) — @teyrebaz33
- User-defined quick commands that bypass the agent loop ([#746](https://github.com/NousResearch/hermes-agent/pull/746)) — @teyrebaz33
- `/reasoning` command for effort level and display toggle ([#921](https://github.com/NousResearch/hermes-agent/pull/921))
- `/verbose` slash command to toggle debug output at runtime ([#94](https://github.com/NousResearch/hermes-agent/pull/94)) — @cesareth
- `/insights` command with usage analytics, cost estimation & activity patterns ([#552](https://github.com/NousResearch/hermes-agent/pull/552))
- `/background` command for managing background processes
- `/help` formatting with command categories ([#640](https://github.com/NousResearch/hermes-agent/issues/640))
- Bell-on-complete — terminal bell when agent finishes ([#738](https://github.com/NousResearch/hermes-agent/pull/738))
- Up/down arrow history navigation
- Clipboard image paste (Alt+V / Ctrl+V)
- Loading indicators for slow slash commands ([#882](https://github.com/NousResearch/hermes-agent/pull/882))
- Spinner flickering fix under patch_stdout ([#91](https://github.com/NousResearch/hermes-agent/pull/91)) — @0xbyt4
- `--quiet/-Q` flag for programmatic single-query mode
- `--fuck-it-ship-it` flag to bypass all approval prompts ([#724](https://github.com/NousResearch/hermes-agent/pull/724)) — @dmahan93
- Tools summary flag ([#767](https://github.com/NousResearch/hermes-agent/pull/767)) — @luisv-1

### Setup & Configuration
- Modular setup wizard with section subcommands and tool-first UX
- Interactive setup for messaging platforms in gateway CLI
- Container resource configuration prompts
- Backend validation for required binaries
- Config migration system with version tracking (currently v7)
- API keys properly routed to .env instead of config.yaml ([#469](https://github.com/NousResearch/hermes-agent/pull/469)) — @ygd58
- Atomic writes for .env to prevent API key loss on crash ([#954](https://github.com/NousResearch/hermes-agent/pull/954)) — @alireza78a
- `hermes tools` — per-platform tool enable/disable with curses UI
- `hermes skills` — per-platform skill enable/disable ([#743](https://github.com/NousResearch/hermes-agent/pull/743)) — @teyrebaz33
- Multiple named custom providers
- `hermes doctor` for health checks across all configured providers and tools
- `hermes update` with auto-restart for gateway service
- Show update-available notice in CLI banner

### Filesystem & Safety
- Filesystem checkpoints and `/rollback` command ([#824](https://github.com/NousResearch/hermes-agent/pull/824), [#452](https://github.com/NousResearch/hermes-agent/issues/452))
- Structured tool result hints for patch and search_files ([#722](https://github.com/NousResearch/hermes-agent/issues/722))
- High-value tool result CTAs — next-action guidance

---

## 🔧 Tool System

### Browser
- Local browser backend — zero-cost headless Chromium via agent-browser (no Browserbase needed)
- Console/errors tool, annotated screenshots, auto-recording ([#745](https://github.com/NousResearch/hermes-agent/pull/745))
- Browser screenshot sharing via MEDIA: on all messaging platforms ([#657](https://github.com/NousResearch/hermes-agent/pull/657))

### Terminal & Execution
- `execute_code` sandbox with json_parse, shell_quote, retry helpers
- Docker backend improvements: custom volume mounts ([#158](https://github.com/NousResearch/hermes-agent/pull/158)) — @Indelwin
- Daytona cloud sandbox backend ([#451](https://github.com/NousResearch/hermes-agent/pull/451)) — @rovle, with CLI setup, doctor, and status display
- SSH backend fixes ([#59](https://github.com/NousResearch/hermes-agent/pull/59)) — @deankerr
- Shell noise filtering and login shell execution for environment consistency
- Head+tail truncation for execute_code stdout overflow
- Background process management with configurable notification modes

### Delegation
- Subagent tool call and thinking exposure to users
- Additional parameters for child agent configuration
- Shared iteration budget across parent + subagents

### File Operations
- Fuzzy-matching patch with 9 strategies
- File search via ripgrep backend
- Atomic writes across all file operations

---

## 🧩 Skills Ecosystem

### System
- Skill slash commands — dynamic CLI and gateway integration
- Optional skills — official skills shipped but not activated by default
- Conditional skill activation based on tool availability ([#785](https://github.com/NousResearch/hermes-agent/pull/785)) — @teyrebaz33
- Platform-conditional skill loading
- Skill prerequisites — hide skills with unmet dependencies ([#659](https://github.com/NousResearch/hermes-agent/pull/659)) — @kshitijk4poor
- `hermes skills browse` — paginated browsing of all hub skills
- Skills sub-category organization
- Atomic skill file writes ([#551](https://github.com/NousResearch/hermes-agent/pull/551)) — @aydnOktay
- Skills sync data loss prevention ([#563](https://github.com/NousResearch/hermes-agent/pull/563)) — @0xbyt4

### Bundled Skills (selected highlights)
- **MLOps:** Axolotl, vLLM, TRL, Unsloth, PyTorch FSDP/Lightning, GGUF, PEFT, Flash Attention, Weights & Biases, Modal, Lambda Labs, and 25+ more
- **Research:** arXiv search, agentic research ideas, ML paper writing
- **Creative:** ASCII art (pyfiglet + cowsay + 571 fonts), ASCII video production, Excalidraw diagrams
- **Software Development:** Systematic debugging, TDD, subagent-driven development, writing plans, code review
- **Productivity:** Google Workspace, Notion, PowerPoint, Obsidian, nano-PDF
- **Gaming:** Minecraft modpack server, Pokémon player
- **Smart Home:** OpenHue (Philips Hue control)
- **Domain:** Passive reconnaissance (subdomains, SSL, WHOIS, DNS)
- **Media:** YouTube transcripts, GIF search, text-to-speech
- **Market Data:** Polymarket prediction markets
- **OCR:** PDF and scanned document extraction
- **Blockchain:** Solana skill with USD pricing ([#212](https://github.com/NousResearch/hermes-agent/pull/212)) — @gizdusum
- **Email:** AgentMail for agent-owned inboxes ([#330](https://github.com/NousResearch/hermes-agent/pull/330)) — @teyrebaz33
- **Feeds:** BlogWatcher for RSS/Atom monitoring
- **DuckDuckGo Search:** Firecrawl fallback ([#267](https://github.com/NousResearch/hermes-agent/pull/267)) — @gamedevCloudy; expanded with DDGS Python API ([#598](https://github.com/NousResearch/hermes-agent/pull/598)) — @areu01or00
- **OpenClaw Migration:** Official migration skill ([#570](https://github.com/NousResearch/hermes-agent/pull/570)) — @unmodeled-tyler
- **ASCII Video:** Full production pipeline ([#854](https://github.com/NousResearch/hermes-agent/pull/854)) — @SHL0MS

---

## 🔒 Security & Reliability

### Security Hardening
- Path traversal fix in skill_view — prevented reading arbitrary files including API keys ([#220](https://github.com/NousResearch/hermes-agent/issues/220)) — @Farukest
- Shell injection prevention in sudo password piping ([#65](https://github.com/NousResearch/hermes-agent/pull/65)) — @leonsgithub
- Dangerous command detection: multiline bypass fix ([#233](https://github.com/NousResearch/hermes-agent/pull/233)), tee/process substitution patterns ([#280](https://github.com/NousResearch/hermes-agent/pull/280)) — @Farukest, @dogiladeveloper
- Symlink boundary check fix in skills_guard ([#386](https://github.com/NousResearch/hermes-agent/pull/386)) — @Farukest
- Multi-word prompt injection bypass prevention in skills_guard ([#192](https://github.com/NousResearch/hermes-agent/pull/192)) — @0xbyt4
- Symlink bypass fix in write deny list on macOS ([#61](https://github.com/NousResearch/hermes-agent/pull/61)) — @0xbyt4
- Enforce 0600/0700 file permissions on sensitive files ([#757](https://github.com/NousResearch/hermes-agent/pull/757))
- .env file permissions restricted to owner-only ([#529](https://github.com/NousResearch/hermes-agent/pull/529)) — @Himess
- Expand secret redaction patterns + config toggle to disable
- FTS5 query sanitization ([#565](https://github.com/NousResearch/hermes-agent/pull/565)) — @0xbyt4
- `--force` flag properly blocked from overriding dangerous verdicts ([#388](https://github.com/NousResearch/hermes-agent/pull/388)) — @Farukest

### Reliability & Stability
- Atomic writes for: sessions.json ([#611](https://github.com/NousResearch/hermes-agent/pull/611)) — @alireza78a; cron jobs ([#146](https://github.com/NousResearch/hermes-agent/pull/146)) — @alireza78a; .env config ([#954](https://github.com/NousResearch/hermes-agent/pull/954)); process checkpoints ([#298](https://github.com/NousResearch/hermes-agent/pull/298)) — @aydnOktay; batch runner ([#297](https://github.com/NousResearch/hermes-agent/pull/297)) — @aydnOktay; skill files ([#551](https://github.com/NousResearch/hermes-agent/pull/551)) — @aydnOktay
- Guard all print() against OSError for systemd/headless environments ([#963](https://github.com/NousResearch/hermes-agent/pull/963))
- Detect, warn, and block file re-read/search loops after context compression ([#705](https://github.com/NousResearch/hermes-agent/pull/705)) — @0xbyt4
- Reset all retry counters at start of run_conversation ([#607](https://github.com/NousResearch/hermes-agent/pull/607)) — @0xbyt4
- Return deny on approval callback timeout instead of None ([#603](https://github.com/NousResearch/hermes-agent/pull/603)) — @0xbyt4
- Fix None message content crashes across codebase ([#277](https://github.com/NousResearch/hermes-agent/pull/277))
- Fix context overrun crash with local LLM backends ([#403](https://github.com/NousResearch/hermes-agent/pull/403)) — @ch3ronsa
- Fix `_flush_sentinel` leaking to external API providers ([#227](https://github.com/NousResearch/hermes-agent/pull/227)) — @Farukest
- Prevent conversation_history mutation in callers ([#229](https://github.com/NousResearch/hermes-agent/pull/229)) — @Farukest
- Fix systemd restart loop ([#614](https://github.com/NousResearch/hermes-agent/pull/614)) — @voidborne-d
- Close file handles and sockets properly to prevent fd leaks ([#568](https://github.com/NousResearch/hermes-agent/pull/568), [#296](https://github.com/NousResearch/hermes-agent/pull/296), [#709](https://github.com/NousResearch/hermes-agent/pull/709)) — @alireza78a, @memosr

### Windows Compatibility
- Guard POSIX-only process functions for Windows ([#219](https://github.com/NousResearch/hermes-agent/pull/219)) — @Farukest
- Windows native support via Git Bash, ZIP-based update fallback
- Install to %LOCALAPPDATA%\hermes on Windows
- pywinpty for PTY support on Windows ([#457](https://github.com/NousResearch/hermes-agent/pull/457)) — @shitcoinsherpa
- Explicit UTF-8 encoding on all config/data file I/O ([#458](https://github.com/NousResearch/hermes-agent/pull/458)) — @shitcoinsherpa
- Windows-compatible path handling in skill listing ([#354](https://github.com/NousResearch/hermes-agent/pull/354), [#390](https://github.com/NousResearch/hermes-agent/pull/390)) — @Farukest
- Regex-based search output parsing for Windows drive-letter paths ([#533](https://github.com/NousResearch/hermes-agent/pull/533)) — @Himess
- Auth store file lock for Windows ([#455](https://github.com/NousResearch/hermes-agent/pull/455)) — @shitcoinsherpa

---

## 🧪 Testing

- **3,289 tests** across agent, gateway, tools, cron, and CLI
- Parallelized test suite with pytest-xdist ([#802](https://github.com/NousResearch/hermes-agent/pull/802)) — @OutThisLife
- Comprehensive unit test batches covering core modules ([#34](https://github.com/NousResearch/hermes-agent/pull/34), [#60](https://github.com/NousResearch/hermes-agent/pull/60), [#62](https://github.com/NousResearch/hermes-agent/pull/62), [#67](https://github.com/NousResearch/hermes-agent/pull/67), [#191](https://github.com/NousResearch/hermes-agent/pull/191), [#193](https://github.com/NousResearch/hermes-agent/pull/193)) — @0xbyt4
- Telegram format tests (43 tests for italic/bold/code rendering) ([#204](https://github.com/NousResearch/hermes-agent/pull/204)) — @0xbyt4
- Clarify tool tests ([#121](https://github.com/NousResearch/hermes-agent/pull/121)) — @Bartok9
- Vision tools type hints and 42 tests ([#792](https://github.com/NousResearch/hermes-agent/pull/792))
- Context compressor boundary regression tests ([#648](https://github.com/NousResearch/hermes-agent/pull/648)) — @intertwine
- RL environment tests — vLLM integration, Atropos tool calling — @dmahan93

---

## 🔬 RL & Evaluation Environments

- **Atropos Integration** — Full agentic RL training pipeline with tool calling support ([#17](https://github.com/NousResearch/hermes-agent/pull/17))
- **TerminalBench2** — Terminal-based tool calling evaluation
- **WebResearchEnv** — Multi-step web research RL environment ([#434](https://github.com/NousResearch/hermes-agent/pull/434)) — @jackx707
- **YC-Bench** — Long-horizon agent benchmark environment
- **OpenThoughts-TBLite** — Evaluation environment and scripts
- **Modal sandbox** — Cloud evaluation with concurrency limits ([#621](https://github.com/NousResearch/hermes-agent/pull/621)) — @voteblake
- Local vLLM instance support for evaluation — @dmahan93
- Hermes-atropos-environments bundled skill ([#815](https://github.com/NousResearch/hermes-agent/pull/815))

---

## 📚 Documentation

- **Full documentation website** (Docusaurus) with 37+ pages covering setup, configuration, tools, skills, messaging platforms, and guides
- Comprehensive platform setup guides for Telegram, Discord, Slack, WhatsApp, Signal, and Email
- AGENTS.md — development guide for AI coding assistants
- CONTRIBUTING.md — contributor guidelines ([#117](https://github.com/NousResearch/hermes-agent/pull/117)) — @Bartok9
- Slash commands reference ([#142](https://github.com/NousResearch/hermes-agent/pull/142)) — @Bartok9
- Skin/theme system documentation
- MCP documentation and examples
- Auxiliary models documentation
- Comprehensive accuracy audit (35+ corrections)
- Documentation typo fixes ([#825](https://github.com/NousResearch/hermes-agent/pull/825), [#439](https://github.com/NousResearch/hermes-agent/pull/439)) — @JackTheGit
- Terminology and CLI formatting standardization ([#166](https://github.com/NousResearch/hermes-agent/pull/166), [#167](https://github.com/NousResearch/hermes-agent/pull/167), [#168](https://github.com/NousResearch/hermes-agent/pull/168)) — @Jr-kenny

---

## 🐛 Notable Bug Fixes

- Fix DeepSeek V3 tool call parser silently dropping multi-line JSON arguments ([#444](https://github.com/NousResearch/hermes-agent/pull/444)) — @PercyDikec
- Fix gateway transcript losing 1 message per turn due to offset mismatch ([#395](https://github.com/NousResearch/hermes-agent/pull/395)) — @PercyDikec
- Fix /retry command silently discarding the agent's final response ([#441](https://github.com/NousResearch/hermes-agent/pull/441)) — @PercyDikec
- Fix max-iterations retry returning empty string after think-block stripping ([#438](https://github.com/NousResearch/hermes-agent/pull/438)) — @PercyDikec
- Fix Codex status dict key mismatch ([#448](https://github.com/NousResearch/hermes-agent/pull/448)) and visibility filter ([#446](https://github.com/NousResearch/hermes-agent/pull/446)) — @PercyDikec
- Fix `_strip_think_blocks` regex stripping visible content when model discusses \<think\> tags literally ([#786](https://github.com/NousResearch/hermes-agent/issues/786))
- Strip \<think\> blocks from final user-facing responses ([#174](https://github.com/NousResearch/hermes-agent/pull/174)) — @Bartok9
- Fix Mistral 422 errors from leftover finish_reason in assistant messages ([#253](https://github.com/NousResearch/hermes-agent/pull/253)) — @Sertug17
- Fix OPENROUTER_API_KEY resolution order across all code paths ([#295](https://github.com/NousResearch/hermes-agent/pull/295)) — @0xbyt4
- Fix gateway session_search crash from missing session_db ([#108](https://github.com/NousResearch/hermes-agent/pull/108)) — @Bartok9
- Fix /retry, /undo having no effect and /reset silently losing memories in gateway ([#217](https://github.com/NousResearch/hermes-agent/pull/217)) — @Farukest
- Fix empty file content in ReadResult.to_dict() ([#225](https://github.com/NousResearch/hermes-agent/pull/225)) — @Farukest
- Fix retry exhaustion IndexError fallthrough ([#223](https://github.com/NousResearch/hermes-agent/pull/223)) — @Farukest
- Fix Anthropic native base URL detection failing fast ([#173](https://github.com/NousResearch/hermes-agent/pull/173)) — @adavyas
- Fix ClawHub Skills Hub adapter for API endpoint changes ([#286](https://github.com/NousResearch/hermes-agent/pull/286)) — @BP602
- Fix terminal blinking on SSH due to UI invalidate throttling ([#284](https://github.com/NousResearch/hermes-agent/pull/284)) — @ygd58
- Fix multi-line input paste detection destroying input ([#84](https://github.com/NousResearch/hermes-agent/pull/84)) — @0xbyt4
- Fix cron job timezone handling for naive timestamps ([#309](https://github.com/NousResearch/hermes-agent/pull/309)) — @areu01or00
- Fix memory tool entry parsing when content contains section sign ([#162](https://github.com/NousResearch/hermes-agent/pull/162)) — @aydnOktay
- Fix Docker backend on macOS and subagent auth for Nous Portal ([#46](https://github.com/NousResearch/hermes-agent/pull/46)) — @rsavitt
- Fix piped install silently aborting when interactive prompts fail ([#72](https://github.com/NousResearch/hermes-agent/pull/72)) — @cutepawss
- Fix false positives in recursive delete detection ([#68](https://github.com/NousResearch/hermes-agent/pull/68)) — @cutepawss
- Eliminate shell noise from terminal output + fix 36 test failures ([#293](https://github.com/NousResearch/hermes-agent/pull/293)) — @0xbyt4
- Fix Honcho auto-enable when API key is present ([#243](https://github.com/NousResearch/hermes-agent/pull/243)) — @Bartok9
- Fix duplicate 'skills' subparser crash on Python 3.11+ ([#898](https://github.com/NousResearch/hermes-agent/issues/898))
- Fix Telegram italic regex newline bug ([#204](https://github.com/NousResearch/hermes-agent/pull/204)) — @0xbyt4
- Fix Ruff lint warnings across codebase ([#608](https://github.com/NousResearch/hermes-agent/pull/608)) — @JackTheGit

---

## 👥 Contributors

Thank you to everyone who has contributed to Hermes Agent! This project is built by a growing community of developers, researchers, and AI enthusiasts.

### Core Team
- **@teknium1** — Project creator, lead developer (~1,100 commits)
- **@dmahan93** — RL environments, Atropos integration, evaluation infrastructure

### Top Community Contributors
- **@0xbyt4** — 35 PRs: MCP client, Home Assistant, security fixes, extensive test coverage, ascii-art skill, and dozens of bug fixes across the codebase
- **@Farukest** — 15 PRs: Security hardening (path traversal, shell injection, symlink bypass), Windows compatibility, WhatsApp fixes
- **@aydnOktay** — 8 PRs: Atomic writes, error handling improvements across Telegram, Discord, transcription, code execution, and skills
- **@teyrebaz33** — 4 PRs: Skills enable/disable system, quick commands, personality customization, conditional skill activation, embedding infrastructure
- **@Bartok9** — 8 PRs: CONTRIBUTING.md, slash commands reference, Discord channel topics, think-block stripping, TTS fix, session count fix, Honcho fix, clarify tool tests
- **@PercyDikec** — 7 PRs: DeepSeek V3 parser fix, /retry fix, gateway transcript fix, Codex fixes, max-iterations retry fixes
- **@rovle** — Daytona cloud sandbox backend (4 PRs)
- **@alireza78a** — Atomic writes for cron/sessions, fd leak prevention, security allowlist fix
- **@satelerd** — WhatsApp native media, multi-user session isolation, tool progress consolidation
- **@Erosika** — Honcho AI-native memory integration
- **@SHL0MS** — ASCII video skill
- **@shitcoinsherpa** — Windows support (pywinpty, UTF-8 encoding, auth store lock)

### All Contributors
@0xbyt4, @Aum08Desai, @BP602, @Bartok9, @Farukest, @FurkanL0, @Himess, @Indelwin, @JackTheGit, @JoshuaMart, @Jr-kenny, @OutThisLife, @PercyDikec, @SHL0MS, @Sertug17, @VencentSoliman, @VolodymyrBg, @adavyas, @alireza78a, @areu01or00, @aydnOktay, @batuhankocyigit, @bierlingm, @caentzminger, @cesareth, @ch3ronsa, @christomitov, @cutepawss, @deankerr, @dmahan93, @dogiladeveloper, @dragonkhoi, @erosika, @gamedevCloudy, @gizdusum, @grp06, @hjc-puro, @insecurejezza, @intertwine, @jackx707, @jdblackstar, @johnh4098, @kaos35, @kshitijk4poor, @leonsgithub, @luisv-1, @manuelschipper, @mehmetkr-31, @memosr, @mormio, @rsavitt, @rewbs, @rovle, @satelerd, @spanishflu-est1918, @stablegenius49, @tars90percent, @tekelala, @teknium1, @teyrebaz33, @tripledoublev, @unmodeled-tyler, @voidborne-d, @voteblake, @ygd58

---

## 📦 Installation

```bash
curl -fsSL https://hermes.nousresearch.com/install | bash
```

Or clone and install manually:

```bash
git clone https://github.com/NousResearch/hermes-agent.git ~/.hermes/hermes-agent
cd ~/.hermes/hermes-agent
./install.sh
hermes setup
```

---

**Full Changelog**: [v2026.3.12](https://github.com/NousResearch/hermes-agent/commits/v2026.3.12)
