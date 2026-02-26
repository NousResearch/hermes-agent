# Hermes Agent — Architectural Map

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                 │
│                                                                       │
│   cli.py (HermesCLI)          gateway/run.py (GatewayRunner)        │
│   Interactive terminal         Multi-platform messaging bot          │
│        ↓                              ↓                              │
│   ┌─────────┐    ┌──────────────────────────────────────────┐       │
│   │ User    │    │  Platform Adapters (gateway/platforms/)  │       │
│   │ REPL    │    │  Discord | Telegram | Slack | WhatsApp   │       │
│   └────┬────┘    └──────────────────┬───────────────────────┘       │
│        │                            │                                │
│        └──────────┬─────────────────┘                                │
│                   ↓                                                   │
│            ┌──────────────┐                                          │
│            │   AIAgent    │  (run_agent.py)                          │
│            │  Core Engine │                                          │
│            └──────┬───────┘                                          │
│                   │                                                   │
│     ┌─────────────┼──────────────────┐                               │
│     ↓             ↓                  ↓                               │
│  ┌────────┐  ┌──────────┐   ┌──────────────┐                       │
│  │ Tools  │  │ Context  │   │   Memory &   │                       │
│  │Registry│  │Compressor│   │   State      │                       │
│  └────────┘  └──────────┘   └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Entry Points

### CLI (`cli.py`)
- **`HermesCLI`** — Interactive REPL with prompt_toolkit
- Handles slash commands (`/help`, `/reset`, `/tools`, `/skills`, `/cron`, etc.)
- Creates `AIAgent` in `_init_agent()`, passing config, session DB, toolsets
- `chat()` sends user messages to `AIAgent.run_conversation()`
- Session resume: loads prior messages from `SessionDB.get_messages_as_conversation()`

### Gateway (`gateway/run.py`)
- **`GatewayRunner`** — Multi-platform bot controller
- Creates platform adapters, session store, delivery router, hook registry, pairing store
- `_handle_message()` is the central dispatch: receives `MessageEvent` → resolves session → runs agent
- `_run_agent()` spawns `AIAgent` per request, streams responses back to platform
- Manages concurrent agents per session with interrupt/queuing logic

---

## 2. Discord Setup (`gateway/platforms/discord.py`)

```
discord.py
  ├── DiscordAdapter(BasePlatformAdapter)
  │     ├── connect() → creates discord.py Bot with message_content intent
  │     ├── _handle_message() → filters by allowed_user_ids, builds MessageEvent
  │     ├── _resolve_allowed_usernames() → resolves username strings to Discord user IDs
  │     ├── _register_slash_commands() → registers /hermes slash command
  │     ├── send() → splits long messages (MAX_MESSAGE_LENGTH=2000), sends to channel
  │     ├── send_voice() → sends audio as Discord attachment
  │     ├── send_image() → sends image as embed or attachment
  │     └── send_exec_approval() → sends interactive button view for dangerous commands
  │
  └── ExecApprovalView(discord.ui.View)
        ├── allow_once / allow_always / deny buttons
        └── _check_auth() → only allowed users can approve
```

**Flow:** Discord message → `on_message` callback → `_handle_message()` → filters by user allowlist → creates `MessageEvent` with `SessionSource(platform=DISCORD)` → passed to `BasePlatformAdapter.handle_message()` → callback to `GatewayRunner._handle_message()`

**Config:** `PlatformConfig` with token from `DISCORD_TOKEN` env var. Allowed users set via `DISCORD_ALLOWED_USERS` (comma-separated IDs or usernames).

---

## 3. The Agent Core (`run_agent.py` — AIAgent)

```
AIAgent
  ├── __init__()
  │     ├── Creates OpenAI client (compatible with any OpenAI-format API)
  │     ├── Resolves tools via model_tools.get_tool_definitions()
  │     ├── Creates ContextCompressor, MemoryStore, TodoStore
  │     ├── Sets up session logging, trajectory saving
  │     └── Loads memory/skill config from hermes_cli.config
  │
  ├── run_conversation(user_message, system_message, history, task_id)
  │     ├── Builds system prompt (_build_system_prompt)
  │     ├── Applies prompt caching (apply_anthropic_cache_control)
  │     ├── Context compression check (should_compress → _compress_context)
  │     ├── API call loop:
  │     │     ├── _build_api_kwargs() → constructs request with tools
  │     │     ├── _interruptible_api_call() → handles interrupts mid-stream
  │     │     ├── Parse tool_calls from response
  │     │     ├── _execute_tool_calls() → dispatches via model_tools.handle_function_call()
  │     │     ├── Check for interrupt between turns
  │     │     └── Loop until: no tool calls, max_iterations, or finish_reason=stop
  │     ├── Persists session to SessionDB
  │     ├── Saves trajectory if configured
  │     └── Returns {response, messages, token counts}
  │
  ├── _build_system_prompt()
  │     ├── Base identity (DEFAULT_AGENT_IDENTITY)
  │     ├── Platform hints (Discord/Telegram/CLI-specific)
  │     ├── Memory block (MemoryStore.format_for_system_prompt)
  │     ├── User profile block
  │     ├── Skills index (build_skills_system_prompt)
  │     ├── Context files (build_context_files_prompt)
  │     ├── Session context (from gateway)
  │     └── Ephemeral system prompt override
  │
  └── flush_memories() → persists memory store to disk
```

---

## 4. Context Compaction (`agent/context_compressor.py`)

```
ContextCompressor
  ├── __init__(model, threshold_percent=75)
  │     ├── Fetches model context length via API metadata
  │     ├── Calculates threshold_tokens = context_length × threshold_percent
  │     └── Creates auxiliary LLM client for summarization
  │
  ├── should_compress(prompt_tokens) → bool
  │     └── True if prompt_tokens > threshold_tokens
  │
  ├── should_compress_preflight(messages) → bool
  │     └── Rough estimate via estimate_messages_tokens_rough()
  │
  ├── compress(messages, current_tokens) → compressed_messages
  │     ├── Protects first N messages (protect_first_n=2)
  │     ├── Protects last N messages (protect_last_n=4)
  │     ├── Middle messages → sent to _generate_summary()
  │     ├── Summary replaces middle section as a single system message
  │     └── Returns: [protected_first] + [summary] + [protected_last]
  │
  └── _generate_summary(turns_to_summarize) → str
        ├── Uses auxiliary LLM (separate from main model)
        ├── Prompt: "Summarize preserving tool results, file paths, decisions"
        └── Target: summary_target_tokens (default 800)
```

**Trigger path in `AIAgent.run_conversation()`:**
1. Before each API call, checks `compressor.should_compress_preflight(messages)`
2. If true, calls `_compress_context(messages, system_message, approx_tokens)`
3. `_compress_context` calls `compressor.compress()` which summarizes the middle of the conversation
4. The compressed messages replace the full history for subsequent API calls

---

## 5. Memory System

### Persistent Memory (`tools/memory_tool.py`)

```
MemoryStore
  ├── Two stores: "memory" (agent notes) and "user" (user profile)
  ├── File-backed: ~/.hermes/memories/MEMORY.md and USER.md
  ├── Bounded: configurable char limits (default from config)
  ├── Entry delimiter: "\n§\n"
  │
  ├── add(target, content) → appends new entry
  ├── replace(target, old_text, new_content) → finds by substring, replaces
  ├── remove(target, old_text) → finds by substring, deletes
  ├── format_for_system_prompt(target) → renders block for injection
  │
  └── Security: _scan_memory_content() checks for injection patterns
        (prompt injection, exfil via curl/wget, ssh backdoor, etc.)
```

**Memory in the system prompt:** `AIAgent._build_system_prompt()` calls `MemoryStore.format_for_system_prompt("memory")` and `format_for_system_prompt("user")`, injecting both blocks into every API call.

**Auto-flush:** `AIAgent.flush_memories()` is called on `/reset`, conversation end, and periodically based on `_memory_flush_min_turns`.

### Session State (`hermes_state.py`)

```
SessionDB (SQLite + FTS5)
  ├── Path: ~/.hermes/state.db
  ├── Tables: sessions, messages, messages_fts (full-text search)
  │
  ├── create_session() / end_session()
  ├── append_message() → stores each role/content/tool_calls
  ├── get_messages_as_conversation() → reconstructs OpenAI message format
  ├── search_messages() → FTS5 full-text search across all sessions
  └── search_sessions() → find sessions by source
```

### Session Search (`tools/session_search_tool.py`)
- Tool exposed to the agent for recalling past conversations
- Uses `SessionDB.search_messages()` with FTS5
- Can summarize found sessions using auxiliary LLM

---

## 6. Gateway Architecture

```
GatewayRunner
  ├── SessionStore (gateway/session.py)
  │     ├── Maps (platform, chat_id) → session_key → SessionEntry
  │     ├── Reset policies: daily, idle-timeout, manual
  │     ├── Tracks token usage per session
  │     └── Backed by sessions.json + SessionDB (SQLite)
  │
  ├── DeliveryRouter (gateway/delivery.py)
  │     ├── Routes cron job outputs to platforms
  │     ├── Resolves targets: "origin", "discord:channel_id", "local"
  │     └── Truncates long messages for platform limits
  │
  ├── HookRegistry (gateway/hooks.py)
  │     ├── Discovers hooks from ~/.hermes/hooks/
  │     ├── Each hook: manifest.yaml + handler.py
  │     └── Events: message_received, before_agent, after_agent, etc.
  │
  ├── PairingStore (gateway/pairing.py)
  │     ├── Code-based user approval for new platform users
  │     ├── Generate 8-char code → user enters in DM → approved
  │     ├── Rate limiting + lockout protection
  │     └── Per-platform approved user lists
  │
  └── Platform Adapters (gateway/platforms/)
        ├── BasePlatformAdapter (ABC)
        │     ├── MessageEvent normalization
        │     ├── Message queuing + interrupt handling
        │     ├── Typing indicators (_keep_typing)
        │     ├── Image/audio caching
        │     └── extract_media() for inline images/voice
        │
        ├── DiscordAdapter   → discord.py library
        ├── TelegramAdapter  → python-telegram-bot
        ├── SlackAdapter     → slack_bolt
        └── WhatsAppAdapter  → WhatsApp Cloud API (HTTP)
```

---

## 7. Tool System

```
ToolRegistry (tools/registry.py) — Singleton
  ├── register(name, toolset, schema, handler, check_fn)
  ├── get_definitions(tool_names) → OpenAI function schemas
  ├── dispatch(name, args) → calls handler, returns string result
  │
  └── Each tool file calls registry.register() at import time

model_tools.py — Orchestration layer
  ├── _discover_tools() → imports all tool modules to trigger registration
  ├── get_tool_definitions(enabled_toolsets) → filtered list
  └── handle_function_call(name, args) → dispatches to registry

Toolsets (toolsets.py)
  ├── Composable tool groups: "web", "terminal", "browser", "file", etc.
  ├── Per-platform defaults: hermes-cli, hermes-discord, hermes-telegram
  ├── All platforms share _HERMES_CORE_TOOLS (30+ tools)
  └── Toolsets can include other toolsets (recursive resolution)
```

### Key Tools

| Tool | Module | Purpose |
|------|--------|---------|
| `terminal` | `tools/terminal_tool.py` | Shell execution with Docker/VM/local backends |
| `memory` | `tools/memory_tool.py` | Persistent notes + user profile |
| `delegate_task` | `tools/delegate_tool.py` | Subagent spawning (max depth 2, max 3 concurrent) |
| `web_search/extract` | `tools/web_tools.py` | Firecrawl-based web search + extraction |
| `browser_*` | `tools/browser_tool.py` | Playwright browser automation |
| `skills_*` | `tools/skills_tool.py` | Skill document management |
| `send_message` | `tools/send_message_tool.py` | Cross-platform messaging |
| `session_search` | `tools/session_search_tool.py` | Past conversation recall |
| `todo` | `tools/todo_tool.py` | Task tracking |
| `vision_analyze` | `tools/vision_tools.py` | Image analysis via auxiliary model |
| `text_to_speech` | `tools/tts_tool.py` | TTS via Edge/ElevenLabs/OpenAI |
| `clarify` | `tools/clarify_tool.py` | Ask user clarifying questions |

---

## 8. Prompt Assembly (`agent/prompt_builder.py`)

```
System prompt structure:
  ┌─────────────────────────────────────────┐
  │ 1. Agent Identity (DEFAULT_AGENT_IDENTITY)│
  │ 2. Platform Hints (Discord/Telegram/CLI)  │
  │ 3. Ephemeral System Prompt (if set)       │
  │ 4. Memory Block ("§"-delimited entries)   │
  │ 5. User Profile Block                     │
  │ 6. Memory Guidance (proactive save hint)  │
  │ 7. Session Search Guidance                │
  │ 8. Skills Index (from ~/.hermes/skills/)  │
  │ 9. Skills Guidance (save-as-skill hint)   │
  │ 10. Context Files (.hermescontext, etc.)  │
  │ 11. Session Context (gateway source info) │
  │ 12. Tool Descriptions (formatted list)    │
  └─────────────────────────────────────────┘
```

**Prompt Caching** (`agent/prompt_caching.py`): Applies Anthropic's `cache_control` markers using a "system_and_3" strategy — caches the system message plus the 3 most recent conversation turns.

---

## 9. Environments / RL Training

```
environments/
  ├── agent_loop.py (HermesAgentLoop)
  │     └── Standalone agent loop for non-interactive execution
  │         Uses tool_call_parsers for non-OpenAI model formats
  │
  ├── hermes_base_env.py (HermesAgentBaseEnv)
  │     └── Base class for RL environments
  │         Handles trajectory collection + reward computation
  │
  ├── tool_call_parsers/  (11 parsers)
  │     └── Hermes, Llama, Mistral, Qwen, DeepSeek, GLM, Kimi, Longcat...
  │         Parse raw model output into standardized tool calls
  │
  ├── hermes_swe_env/    → SWE-bench evaluation
  ├── terminal_test_env/ → Terminal task evaluation
  └── benchmarks/terminalbench_2/ → TerminalBench v2 eval
```

---

## 10. Cron System

```
cron/
  ├── jobs.py → CRUD for scheduled jobs (JSON file at ~/.hermes/cron/)
  │     Supports: interval ("every 2h"), daily ("at 09:00"), cron expressions
  │
  └── scheduler.py → Background thread ticks every N seconds
        ├── tick() → checks due jobs, runs them
        ├── run_job() → spawns AIAgent with job prompt
        └── _deliver_result() → routes output via DeliveryRouter
```

---

## 11. Delegation / Subagents (`tools/delegate_tool.py`)

```
delegate_task()
  ├── Single mode: one goal → one child AIAgent
  ├── Batch mode: up to 3 tasks → concurrent children (ThreadPoolExecutor)
  │
  ├── Constraints:
  │     ├── Max depth: 2 (no recursive delegation)
  │     ├── Max concurrent: 3
  │     ├── Blocked tools: delegate_task, clarify, memory, send_message, execute_code
  │     └── Default toolsets: terminal, file, web
  │
  └── Each child gets:
        ├── Isolated AIAgent instance
        ├── Custom system prompt with goal + context
        ├── Own terminal session
        └── stdout/stderr captured (not shown to user)
```

---

## 12. Security

- **Memory injection scanning** (`_scan_memory_content`): Regex patterns for prompt injection, exfiltration, SSH backdoors
- **Context file scanning** (`_scan_context_content`): Same patterns applied to `.hermescontext` and similar files
- **Pairing system** (`gateway/pairing.py`): Code-based approval before unknown users can interact via messaging platforms
- **Exec approval** (`ExecApprovalView`): Discord interactive buttons for approving dangerous terminal commands
- **Skill guard** (`tools/skills_guard.py`): Security scanning of skill files before installation, including LLM audit

---

## Data Flow Summary

```
User Message (CLI or Platform)
       │
       ↓
  Session Resolution (SessionStore)
       │
       ↓
  System Prompt Assembly (prompt_builder)
  [identity + memory + skills + context files + session info]
       │
       ↓
  AIAgent.run_conversation()
       │
       ├── Context compression check
       │     └── If over threshold → summarize middle messages
       │
       ├── API call → LLM response
       │     └── Prompt caching for Anthropic models
       │
       ├── Tool execution loop
       │     ├── model_tools.handle_function_call()
       │     ├── registry.dispatch() → tool handler
       │     └── Result appended to messages → next API call
       │
       ├── Persist to SessionDB
       │
       └── Return response
              │
              ↓
       Platform delivery (or CLI print)
```

---

## File Layout Reference

```
hermes-agent/
├── cli.py                      # CLI entry point (HermesCLI)
├── run_agent.py                # AIAgent core engine
├── model_tools.py              # Tool orchestration layer
├── toolsets.py                 # Toolset definitions and resolution
├── hermes_state.py             # SessionDB (SQLite + FTS5)
├── hermes_constants.py         # Shared constants
├── batch_runner.py             # Batch evaluation runner
├── toolset_distributions.py    # Tool sampling for RL training
├── trajectory_compressor.py    # Trajectory compression for training data
│
├── agent/
│   ├── context_compressor.py   # Context window compression
│   ├── prompt_builder.py       # System prompt assembly
│   ├── prompt_caching.py       # Anthropic cache_control markers
│   ├── model_metadata.py       # Model context length + token estimation
│   ├── display.py              # Terminal UI (spinner, tool previews)
│   ├── auxiliary_client.py     # Auxiliary LLM client (for summarization)
│   └── trajectory.py           # Trajectory saving
│
├── gateway/
│   ├── run.py                  # GatewayRunner (main controller)
│   ├── config.py               # GatewayConfig, PlatformConfig, reset policies
│   ├── session.py              # SessionStore, SessionContext, SessionSource
│   ├── delivery.py             # DeliveryRouter (cron output routing)
│   ├── hooks.py                # HookRegistry (event hooks)
│   ├── mirror.py               # Cross-platform session mirroring
│   ├── pairing.py              # User approval via pairing codes
│   ├── channel_directory.py    # Cached channel/contact directory
│   ├── status.py               # PID file management
│   ├── sticker_cache.py        # Sticker caching
│   └── platforms/
│       ├── base.py             # BasePlatformAdapter (ABC)
│       ├── discord.py          # DiscordAdapter + ExecApprovalView
│       ├── telegram.py         # TelegramAdapter
│       ├── slack.py            # SlackAdapter
│       └── whatsapp.py         # WhatsAppAdapter
│
├── tools/
│   ├── registry.py             # ToolRegistry singleton
│   ├── terminal_tool.py        # Shell execution (Docker/VM/local)
│   ├── memory_tool.py          # MemoryStore + memory tool
│   ├── delegate_tool.py        # Subagent delegation
│   ├── browser_tool.py         # Playwright browser automation
│   ├── web_tools.py            # Web search + extraction (Firecrawl)
│   ├── vision_tools.py         # Image analysis
│   ├── skills_tool.py          # Skill listing/viewing
│   ├── skill_manager_tool.py   # Skill CRUD
│   ├── skills_hub.py           # Skill marketplace/sources
│   ├── skills_guard.py         # Skill security scanning
│   ├── skills_sync.py          # Bundled skill syncing
│   ├── session_search_tool.py  # Past conversation search
│   ├── send_message_tool.py    # Cross-platform messaging
│   ├── todo_tool.py            # Task tracking
│   ├── clarify_tool.py         # User clarification
│   ├── tts_tool.py             # Text-to-speech
│   ├── transcription_tools.py  # Audio transcription
│   └── rl_training_tool.py     # RL training management
│
├── environments/
│   ├── agent_loop.py           # Standalone agent loop (non-interactive)
│   ├── hermes_base_env.py      # RL environment base class
│   ├── tool_context.py         # Tool execution context
│   ├── patches.py              # Runtime patches
│   ├── tool_call_parsers/      # 11 model-specific parsers
│   ├── hermes_swe_env/         # SWE-bench environment
│   ├── terminal_test_env/      # Terminal task environment
│   └── benchmarks/             # Evaluation benchmarks
│
├── cron/
│   ├── jobs.py                 # Job CRUD (JSON-backed)
│   └── scheduler.py            # Background scheduler
│
├── hermes_cli/
│   ├── auth.py                 # Provider authentication
│   ├── config.py               # CLI configuration
│   ├── commands.py             # CLI command handlers
│   ├── skills_hub.py           # Skills hub CLI interface
│   ├── banner.py               # Welcome banner
│   ├── callbacks.py            # CLI callbacks
│   ├── colors.py               # Terminal colors
│   ├── cron.py                 # Cron CLI interface
│   ├── doctor.py               # System diagnostics
│   └── gateway.py              # Gateway CLI interface
│
└── docs/                       # Documentation
```
