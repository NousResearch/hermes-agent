"""Random tips shown at CLI session start to help users discover features."""

import random


def _detect_tip_language() -> str:
    """Detect the configured language for tips."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        lang = (config.get("approvals") or {}).get("language", "")
        if lang:
            return lang.lower()
        lang = (config.get("display") or {}).get("language", "")
        if lang:
            return lang.lower()
    except Exception:
        pass
    return "en"


# ---------------------------------------------------------------------------
# Tip corpus — one-liners covering slash commands, CLI flags, config,
# keybindings, tools, gateway, skills, profiles, and workflow tricks.
# ---------------------------------------------------------------------------

TIPS_EN = [
    # --- Slash Commands ---
    "/btw <question> asks a quick side question without tools or history — great for clarifications.",
    "/background <prompt> runs a task in a separate session while your current one stays free.",
    "/branch forks the current session so you can explore a different direction without losing progress.",
    "/compress manually compresses conversation context when things get long.",
    "/rollback lists filesystem checkpoints — restore files the agent modified to any prior state.",
    "/rollback diff 2 previews what changed since checkpoint 2 without restoring anything.",
    "/rollback 2 src/file.py restores a single file from a specific checkpoint.",
    "/title \"my project\" names your session — resume it later with /resume or hermes -c.",
    "/resume picks up where you left off in a previously named session.",
    "/queue <prompt> queues a message for the next turn without interrupting the current one.",
    "/undo removes the last user/assistant exchange from the conversation.",
    "/retry resends your last message — useful when the agent's response wasn't quite right.",
    "/verbose cycles tool progress display: off → new → all → verbose.",
    "/reasoning high increases the model's thinking depth. /reasoning show displays the reasoning.",
    "/fast toggles priority processing for faster API responses (provider-dependent).",
    "/yolo skips all dangerous command approval prompts for the rest of the session.",
    "/model lets you switch models mid-session — try /model sonnet or /model gpt-5.",
    "/model --global changes your default model permanently.",
    "/personality pirate sets a fun personality — 14 built-in options from kawaii to shakespeare.",
    "/skin changes the CLI theme — try ares, mono, slate, poseidon, or charizard.",
    "/statusbar toggles a persistent bar showing model, tokens, context fill %, cost, and duration.",
    "/tools disable browser temporarily removes browser tools for the current session.",
    "/browser connect attaches browser tools to your running Chrome instance via CDP.",
    "/plugins lists installed plugins and their status.",
    "/cron manages scheduled tasks — set up recurring prompts with delivery to any platform.",
    "/reload-mcp hot-reloads MCP server configuration without restarting.",
    "/usage shows token usage, cost breakdown, and session duration.",
    "/insights shows usage analytics for the last 30 days.",
    "/paste checks your clipboard for an image and attaches it to your next message.",
    "/profile shows which profile is active and its home directory.",
    "/config shows your current configuration at a glance.",
    "/stop kills all running background processes spawned by the agent.",

    # --- @ Context References ---
    "@file:path/to/file.py injects file contents directly into your message.",
    "@file:main.py:10-50 injects only lines 10-50 of a file.",
    "@folder:src/ injects a directory tree listing.",
    "@diff injects your unstaged git changes into the message.",
    "@staged injects your staged git changes (git diff --staged).",
    "@git:5 injects the last 5 commits with full patches.",
    "@url:https://example.com fetches and injects a web page's content.",
    "Typing @ triggers filesystem path completion — navigate to any file interactively.",
    "Combine multiple references: \"Review @file:main.py and @file:test.py for consistency.\"",

    # --- Keybindings ---
    "Alt+Enter (or Ctrl+J) inserts a newline for multi-line input.",
    "Ctrl+C interrupts the agent. Double-press within 2 seconds to force exit.",
    "Ctrl+Z suspends Hermes to the background — run fg in your shell to resume.",
    "Tab accepts auto-suggestion ghost text or autocompletes slash commands.",
    "Type a new message while the agent is working to interrupt and redirect it.",
    "Alt+V pastes an image from your clipboard into the conversation.",
    "Pasting 5+ lines auto-saves to a file and inserts a compact reference instead.",

    # --- CLI Flags ---
    "hermes -c resumes your most recent CLI session. hermes -c \"project name\" resumes by title.",
    "hermes -w creates an isolated git worktree — perfect for parallel agent workflows.",
    "hermes -w -q \"Fix issue #42\" combines worktree isolation with a one-shot query.",
    "hermes chat -t web,terminal enables only specific toolsets for a focused session.",
    "hermes chat -s github-pr-workflow preloads a skill at launch.",
    "hermes chat -q \"query\" runs a single non-interactive query and exits.",
    "hermes chat --max-turns 200 overrides the default 90-iteration limit per turn.",
    "hermes chat --checkpoints enables filesystem snapshots before every destructive file change.",
    "hermes --yolo bypasses all dangerous command approval prompts for the entire session.",
    "hermes chat --source telegram tags the session for filtering in hermes sessions list.",
    "hermes -p work chat runs under a specific profile without changing your default.",

    # --- CLI Subcommands ---
    "hermes doctor --fix diagnoses and auto-repairs config and dependency issues.",
    "hermes dump outputs a compact setup summary — great for bug reports.",
    "hermes config set KEY VALUE auto-routes secrets to .env and everything else to config.yaml.",
    "hermes config edit opens config.yaml in your default editor.",
    "hermes config check scans for missing or stale configuration options.",
    "hermes sessions browse opens an interactive session picker with search.",
    "hermes sessions stats shows session counts by platform and database size.",
    "hermes sessions prune --older-than 30 cleans up old sessions.",
    "hermes skills search react --source skills-sh searches the skills.sh public directory.",
    "hermes skills check scans installed hub skills for upstream updates.",
    "hermes skills tap add myorg/skills-repo adds a custom GitHub skill source.",
    "hermes skills snapshot export setup.json exports your skill configuration for backup or sharing.",
    "hermes mcp add github --command npx adds MCP servers from the command line.",
    "hermes mcp serve runs Hermes itself as an MCP server for other agents.",
    "hermes auth add lets you add multiple API keys for credential pool rotation.",
    "hermes completion bash >> ~/.bashrc enables tab completion for all commands and profiles.",
    "hermes logs -f follows agent.log in real time. --level WARNING --since 1h filters output.",
    "hermes backup creates a zip backup of your entire Hermes home directory.",
    "hermes profile create coder creates an isolated profile that becomes its own command.",
    "hermes profile create work --clone copies your current config and keys to a new profile.",
    "hermes update syncs new bundled skills to ALL profiles automatically.",
    "hermes gateway install sets up Hermes as a system service (systemd/launchd).",
    "hermes memory setup lets you configure an external memory provider (Honcho, Mem0, etc.).",
    "hermes webhook subscribe creates event-driven webhook routes with HMAC validation.",

    # --- Configuration ---
    "Set display.bell_on_complete: true in config.yaml to hear a bell when long tasks finish.",
    "Set display.streaming: true to see tokens appear in real time as the model generates.",
    "Set display.show_reasoning: true to watch the model's chain-of-thought reasoning.",
    "Set display.compact: true to reduce whitespace in output for denser information.",
    "Set display.busy_input_mode: queue to queue messages instead of interrupting the agent.",
    "Set display.resume_display: minimal to skip the full conversation recap on session resume.",
    "Set compression.threshold: 0.50 to control when auto-compression fires (default: 50% of context).",
    "Set agent.max_turns: 200 to let the agent take more tool-calling steps per turn.",
    "Set file_read_max_chars: 200000 to increase the max content per read_file call.",
    "Set approvals.mode: smart to let an LLM auto-approve safe commands and auto-deny dangerous ones.",
    "Set fallback_model in config.yaml to automatically fail over to a backup provider.",
    "Set privacy.redact_pii: true to hash user IDs and phone numbers before sending to the LLM.",
    "Set browser.record_sessions: true to auto-record browser sessions as WebM videos.",
    "Set worktree: true in config.yaml to always create a git worktree (same as hermes -w).",
    "Set security.website_blocklist.enabled: true to block specific domains from web tools.",
    "Set cron.wrap_response: false to deliver raw agent output without the cron header/footer.",
    "HERMES_TIMEZONE overrides the server timezone with any IANA timezone string.",
    "Environment variable substitution works in config.yaml: use ${VAR_NAME} syntax.",
    "Quick commands in config.yaml run shell commands instantly with zero token usage.",
    "Custom personalities can be defined in config.yaml under agent.personalities.",
    "provider_routing controls OpenRouter provider sorting, whitelisting, and blacklisting.",

    # --- Tools & Capabilities ---
    "execute_code runs Python scripts that call Hermes tools programmatically — results stay out of context.",
    "delegate_task spawns up to 3 concurrent sub-agents by default (configurable via delegation.max_concurrent_children) with isolated contexts for parallel work.",
    "web_extract works on PDF URLs — pass any PDF link and it converts to markdown.",
    "search_files is ripgrep-backed and faster than grep — use it instead of terminal grep.",
    "patch uses 9 fuzzy matching strategies so minor whitespace differences won't break edits.",
    "patch supports V4A format for bulk multi-file edits in a single call.",
    "read_file suggests similar filenames when a file isn't found.",
    "read_file auto-deduplicates — re-reading an unchanged file returns a lightweight stub.",
    "browser_vision takes a screenshot and analyzes it with AI — works for CAPTCHAs and visual content.",
    "browser_console can evaluate JavaScript expressions in the page context.",
    "image_generate creates images with FLUX 2 Pro and automatic 2x upscaling.",
    "text_to_speech converts text to audio — plays as voice bubbles on Telegram.",
    "send_message can reach any connected messaging platform from within a session.",
    "The todo tool helps the agent track complex multi-step tasks during a session.",
    "session_search performs full-text search across ALL past conversations.",
    "The agent automatically saves preferences, corrections, and environment facts to memory.",
    "mixture_of_agents routes hard problems through 4 frontier LLMs collaboratively.",
    "Terminal commands support background mode with notify_on_complete for long-running tasks.",
    "Terminal background processes support watch_patterns to alert on specific output lines.",
    "The terminal tool supports 6 backends: local, Docker, SSH, Modal, Daytona, and Singularity.",

    # --- Profiles ---
    "Each profile gets its own config, API keys, memory, sessions, skills, and cron jobs.",
    "Profile names become shell commands — 'hermes profile create coder' creates the 'coder' command.",
    "hermes profile export coder -o backup.tar.gz creates a portable profile archive.",
    "If two profiles accidentally share a bot token, the second gateway is blocked with a clear error.",

    # --- Sessions ---
    "Sessions auto-generate descriptive titles after the first exchange — no manual naming needed.",
    "Session titles support lineage: \"my project\" → \"my project #2\" → \"my project #3\".",
    "When exiting, Hermes prints a resume command with session ID and stats.",
    "hermes sessions export backup.jsonl exports all sessions for backup or analysis.",
    "hermes -r SESSION_ID resumes any specific past session by its ID.",

    # --- Memory ---
    "Memory is a frozen snapshot — changes appear in the system prompt only at next session start.",
    "Memory entries are automatically scanned for prompt injection and exfiltration patterns.",
    "The agent has two memory stores: personal notes (~2200 chars) and user profile (~1375 chars).",
    "Corrections you give the agent (\"no, do it this way\") are often auto-saved to memory.",

    # --- Skills ---
    "Over 80 bundled skills covering github, creative, mlops, productivity, research, and more.",
    "Every installed skill automatically becomes a slash command — type / to see them all.",
    "hermes skills install official/security/1password installs optional skills from the repo.",
    "Skills can restrict to specific OS platforms — some only load on macOS or Linux.",
    "skills.external_dirs in config.yaml lets you load skills from custom directories.",
    "The agent can create its own skills as procedural memory using skill_manage.",
    "The plan skill saves markdown plans under .hermes/plans/ in the active workspace.",

    # --- Cron & Scheduling ---
    "Cron jobs can attach skills: hermes cron add --skill blogwatcher \"Check for new posts\".",
    "Cron delivery targets include telegram, discord, slack, email, sms, and 12+ more platforms.",
    "If a cron response starts with [SILENT], delivery is suppressed — useful for monitoring-only jobs.",
    "Cron supports relative delays (30m), intervals (every 2h), cron expressions, and ISO timestamps.",
    "Cron jobs run in completely fresh agent sessions — prompts must be self-contained.",

    # --- Voice ---
    "Voice mode works with zero API keys if faster-whisper is installed (free local speech-to-text).",
    "Five TTS providers available: Edge TTS (free), ElevenLabs, OpenAI, NeuTTS (free local), MiniMax.",
    "/voice on enables voice mode in the CLI. Ctrl+B toggles push-to-talk recording.",
    "Streaming TTS plays sentences as they generate — you don't wait for the full response.",
    "Voice messages on Telegram, Discord, WhatsApp, and Slack are auto-transcribed.",

    # --- Gateway & Messaging ---
    "Hermes runs on 18 platforms: Telegram, Discord, Slack, WhatsApp, Signal, Matrix, email, and more.",
    "hermes gateway install sets it up as a system service that starts on boot.",
    "DingTalk uses Stream Mode — no webhooks or public URL needed.",
    "BlueBubbles brings iMessage to Hermes via a local macOS server.",
    "Webhook routes support HMAC validation, rate limiting, and event filtering.",
    "The API server exposes an OpenAI-compatible endpoint compatible with Open WebUI and LibreChat.",
    "Discord voice channel mode: the bot joins VC, transcribes speech, and talks back.",
    "group_sessions_per_user: true gives each person their own session in group chats.",
    "/sethome marks a chat as the home channel for cron job deliveries.",
    "The gateway supports inactivity-based timeouts — active agents can run indefinitely.",

    # --- Security ---
    "Dangerous command approval has 4 tiers: once, session, always (permanent allowlist), deny.",
    "Smart approval mode uses an LLM to auto-approve safe commands and flag dangerous ones.",
    "SSRF protection blocks private networks, loopback, link-local, and cloud metadata addresses.",
    "Tirith pre-exec scanning detects homograph URL spoofing and pipe-to-interpreter patterns.",
    "MCP subprocesses receive a filtered environment — only safe system vars pass through.",
    "Context files (.hermes.md, AGENTS.md) are security-scanned for prompt injection before loading.",
    "command_allowlist in config.yaml permanently approves specific shell command patterns.",

    # --- Context & Compression ---
    "Context auto-compresses when it reaches the threshold — memories are flushed and history summarized.",
    "The status bar turns yellow, then orange, then red as context fills up.",
    "SOUL.md at ~/.hermes/SOUL.md is the agent's primary identity — customize it to shape behavior.",
    "Hermes loads project context from .hermes.md, AGENTS.md, CLAUDE.md, or .cursorrules (first match).",
    "Subdirectory AGENTS.md files are discovered progressively as the agent navigates into folders.",
    "Context files are capped at 20,000 characters with smart head/tail truncation.",

    # --- Browser ---
    "Five browser providers: local Chromium, Browserbase, Browser Use, Camofox, and Firecrawl.",
    "Camofox is an anti-detection browser — Firefox fork with C++ fingerprint spoofing.",
    "browser_navigate returns a page snapshot automatically — no need to call browser_snapshot after.",
    "browser_vision with annotate=true overlays numbered labels on interactive elements.",

    # --- MCP ---
    "MCP servers are configured in config.yaml — both stdio and HTTP transports supported.",
    "Per-server tool filtering: tools.include whitelists and tools.exclude blacklists specific tools.",
    "MCP servers auto-generate toolsets at runtime — hermes tools can toggle them per platform.",
    "MCP OAuth support: auth: oauth enables browser-based authorization with PKCE.",

    # --- Checkpoints & Rollback ---
    "Checkpoints have zero overhead when no files are modified — enabled by default.",
    "A pre-rollback snapshot is saved automatically so you can undo the undo.",
    "/rollback also undoes the conversation turn, so the agent doesn't remember rolled-back changes.",
    "Checkpoints use shadow repos in ~/.hermes/checkpoints/ — your project's .git is never touched.",

    # --- Batch & Data ---
    "batch_runner.py processes hundreds of prompts in parallel for training data generation.",
    "hermes chat -Q enables quiet mode for programmatic use — suppresses banner and spinner.",
    "Trajectory saving (--save-trajectories) captures full tool-use traces for model training.",

    # --- Plugins ---
    "Three plugin types: general (tools/hooks), memory providers, and context engines.",
    "hermes plugins install owner/repo installs plugins directly from GitHub.",
    "8 external memory providers available: Honcho, OpenViking, Mem0, Hindsight, and more.",
    "Plugin hooks include pre/post_tool_call, pre/post_llm_call, and transform_terminal_output for output canonicalization.",

    # --- Miscellaneous ---
    "Prompt caching (Anthropic) reduces costs by reusing cached system prompt prefixes.",
    "The agent auto-generates session titles in a background thread — zero latency impact.",
    "Smart model routing can auto-route simple queries to a cheaper model.",
    "Slash commands support prefix matching: /h resolves to /help, /mod to /model.",
    "Dragging a file path into the terminal auto-attaches images or sends as context.",
    ".worktreeinclude in your repo root lists gitignored files to copy into worktrees.",
    "hermes acp runs Hermes as an ACP server for VS Code, Zed, and JetBrains integration.",
    "Custom providers: save named endpoints in config.yaml under custom_providers.",
    "HERMES_EPHEMERAL_SYSTEM_PROMPT injects a system prompt that's never persisted to history.",
    "credential_pool_strategies supports fill_first, round_robin, least_used, and random rotation.",
    "hermes login supports OAuth-based auth for Nous and OpenAI Codex providers.",
    "The API server supports both Chat Completions and Responses API with server-side state.",
    "tool_preview_length: 0 in config shows full file paths in the spinner's activity feed.",
    "hermes status --deep runs deeper diagnostic checks across all components.",

    # --- Hidden Gems & Power-User Tricks ---
    "BOOT.md at ~/.hermes/BOOT.md runs automatically on every gateway start — use it for startup checks.",
    "Cron jobs can attach a Python script (--script) whose stdout is injected into the prompt as context.",
    "Cron scripts live in ~/.hermes/scripts/ and run before the agent — perfect for data collection pipelines.",
    "prefill_messages_file in config.yaml injects few-shot examples into every API call, never saved to history.",
    "SOUL.md completely replaces the agent's default identity — rewrite it to make Hermes your own.",
    "SOUL.md is auto-seeded with a default personality on first run. Edit ~/.hermes/SOUL.md to customize.",
    "/compress <focus topic> allocates 60-70% of the summary budget to your topic and aggressively trims the rest.",
    "On second+ compression, the compressor updates the previous summary instead of starting from scratch.",
    "Before a gateway session reset, Hermes auto-flushes important facts to memory in the background.",
    "network.force_ipv4: true in config.yaml fixes hangs on servers with broken IPv6 — monkey-patches socket.",
    "The terminal tool annotates common exit codes: grep returning 1 = 'No matches found (not an error)'.",
    "Failed foreground terminal commands auto-retry up to 3 times with exponential backoff (2s, 4s, 8s).",
    "Bare sudo commands are auto-rewritten to pipe SUDO_PASSWORD from .env — no interactive prompt needed.",
    "execute_code has built-in helpers: json_parse() for tolerant parsing, shell_quote(), and retry() with backoff.",
    "execute_code's 7 sandbox tools (web_search, terminal, read/write/search/patch) use RPC — never enter context.",
    "Reading the same file region 3+ times triggers a warning. At 4+, it's hard-blocked to prevent loops.",
    "write_file and patch detect if a file was externally modified since the last read and warn about staleness.",
    "V4A patch format supports Add File, Delete File, and Move File directives — not just Update.",
    "MCP servers can request LLM completions back via sampling — the agent becomes a tool for the server.",
    "MCP servers send notifications/tools/list_changed to trigger automatic tool re-registration without restart.",
    "delegate_task with acp_command: 'claude' spawns Claude Code as a child agent from any platform.",
    "Delegation has a heartbeat thread — child activity propagates to the parent, preventing gateway timeouts.",
    "When a provider returns HTTP 402 (payment required), the auxiliary client auto-falls back to the next one.",
    "agent.tool_use_enforcement steers models that describe actions instead of calling tools — auto for GPT/Codex.",
    "agent.restart_drain_timeout (default 60s) lets running agents finish before a gateway restart takes effect.",
    "The gateway caches AIAgent instances per session — destroying this cache breaks Anthropic prompt caching.",
    "Any website can expose skills via /.well-known/skills/index.json — the skills hub discovers them automatically.",
    "The skills audit log at ~/.hermes/skills/.hub/audit.log tracks every install and removal operation.",
    "Stale git worktrees are auto-cleaned: 24-72h old with no unpushed commits get pruned on startup.",
    "Each profile gets its own subprocess HOME at HERMES_HOME/home/ — isolated git, ssh, npm, gh configs.",
    "HERMES_HOME_MODE env var (octal, e.g. 0701) sets custom directory permissions for web server traversal.",
    "Container mode: place .container-mode in HERMES_HOME and the host CLI auto-execs into the container.",
    "Ctrl+C has 5 priority tiers: cancel recording → cancel prompts → cancel picker → interrupt agent → exit.",
    "Every interrupt during an agent run is logged to ~/.hermes/interrupt_debug.log with timestamps.",
    "BROWSER_CDP_URL connects browser tools to any running Chrome — accepts WebSocket, HTTP, or host:port.",
    "BROWSERBASE_ADVANCED_STEALTH=true enables advanced anti-detection with custom Chromium (Scale Plan).",
    "The CLI auto-switches to compact mode in terminals narrower than 80 columns.",
    "Quick commands support two types: exec (run shell command directly) and alias (redirect to another command).",
    "Per-task delegation model: delegation.model and delegation.provider in config route subagents to cheaper models.",
    "delegation.reasoning_effort independently controls thinking depth for subagents.",
    "display.platforms in config.yaml allows per-platform display overrides: {telegram: {tool_progress: all}}.",
    "human_delay.mode in config simulates human typing speed — configurable min_ms/max_ms range.",
    "Config version migrations run automatically on load — new config keys appear without manual intervention.",
    "GPT and Codex models get special system prompt guidance for tool discipline and mandatory tool use.",
    "Gemini models get tailored directives for absolute paths, parallel tool calls, and non-interactive commands.",
    "context.engine in config.yaml can be set to a plugin name for alternative context management strategies.",
    "Browser pages over 8000 tokens are auto-summarized by the auxiliary LLM before returning to the agent.",
    "The compressor does a cheap pre-pass: tool outputs over 200 chars are replaced with placeholders before the LLM runs.",
    "When compression fails, further attempts are paused for 10 minutes to avoid API hammering.",
    "Long dangerous commands (>70 chars) get a 'view' option in the approval prompt to see the full text first.",
    "Audio level visualization shows ▁▂▃▄▅▆▇ bars during voice recording based on microphone RMS levels.",
    "Profile names cannot collide with existing PATH binaries — 'hermes profile create ls' would be rejected.",
    "hermes profile create backup --clone-all copies everything (config, keys, SOUL.md, memories, skills, sessions).",
    "The voice record key is configurable via voice.record_key in config.yaml — not just Ctrl+B.",
    ".cursorrules and .cursor/rules/*.mdc files are auto-detected and loaded as project context.",
    "Context files support 10+ prompt injection patterns — invisible Unicode, 'ignore instructions', exfil attempts.",
    "GPT-5 and Codex use 'developer' role instead of 'system' in the message format.",
    "Per-task auxiliary overrides: auxiliary.vision.provider, auxiliary.compression.model, etc. in config.yaml.",
    "The auxiliary client treats 'main' as a provider alias — resolves to your actual primary provider + model.",
    "hermes claw migrate --dry-run previews OpenClaw migration without writing anything.",
    "File paths pasted with quotes or escaped spaces are handled automatically — no manual cleanup needed.",
    "Slash commands never trigger the large-paste collapse — /command with big arguments works correctly.",
    "In interrupt mode, slash commands typed during agent execution bypass interrupt logic and run immediately.",
    "HERMES_DEV=1 bypasses container mode detection for local development.",
    "Each MCP server gets its own toolset (mcp-servername) that can be toggled independently via hermes tools.",
    "MCP ${ENV_VAR} placeholders in config are resolved at server spawn — including vars from ~/.hermes/.env.",
    "Skills from trusted repos (NousResearch) get a 'trusted' security level; community skills get extra scanning.",
    "The skills quarantine at ~/.hermes/skills/.hub/quarantine/ holds skills pending security review.",
]

# Chinese tips
TIPS_ZH = [
    # --- 斜杠命令 ---
    "/btw <问题> 可以快速提问而不使用工具或历史记录 — 适合澄清疑问。",
    "/background <提示> 在独立会话中运行任务，同时保持当前会话可用。",
    "/branch 分叉当前会话，让你探索不同方向而不丢失进度。",
    "/compress 手动压缩对话上下文。",
    "/rollback 列出文件系统检查点 — 将文件恢复到代理修改前的任何状态。",
    '/title "我的项目" 为会话命名 — 稍后使用 /resume 或 hermes -c 恢复。',
    "/resume 从之前命名的会话继续。",
    "/queue <提示> 将消息加入队列，不中断当前操作。",
    "/undo 删除对话中的最后一次用户/助手交互。",
    "/retry 重新发送你的最后一条消息。",
    "/verbose 循环切换工具进度显示：off → new → all → verbose。",
    "/reasoning high 增加模型的思考深度。/reasoning show 显示推理过程。",
    "/yolo 跳过本会话剩余时间的所有危险命令批准提示。",
    "/model 让你可以在会话中途切换模型 — 试试 /model sonnet 或 /model gpt-5。",
    "/personality pirate 设置有趣的性格 — 14 种内置选项从可爱到莎士比亚风格。",
    "/skin 更改 CLI 主题 — 试试 ares、mono、slate、poseidon 或 chinese。",
    "/statusbar 切换显示持久状态栏，显示模型、token、上下文填充%、成本和时长。",
    "/tools disable browser 临时移除当前会话的浏览器工具。",
    "/usage 显示 token 使用情况、成本分解和会话时长。",
    "/config 显示当前配置概览。",
    
    # --- @ 上下文引用 ---
    "@file:path/to/file.py 将文件内容直接注入到你的消息中。",
    "@file:main.py:10-50 只注入文件的第 10-50 行。",
    "@folder:src/ 注入目录树列表。",
    "@diff 注入你未暂存的 git 更改。",
    "@url:https://example.com 获取并注入网页内容。",
    "输入 @ 触发文件系统路径自动完成 — 交互导航到任何文件。",
    
    # --- 快捷键 ---
    "Alt+Enter（或 Ctrl+J）插入换行符用于多行输入。",
    "Ctrl+C 中断代理。2 秒内双击强制退出。",
    "Ctrl+Z 将 Hermes 挂起到后台 — 在 shell 中运行 fg 恢复。",
    "Tab 接受自动建议幽灵文本或自动完成斜杠命令。",
    "代理工作时输入新消息可中断并重新引导它。",
    "Alt+V 从剪贴板粘贴图像到对话中。",
    
    # --- CLI 标志 ---
    'hermes -c 恢复最近的 CLI 会话。hermes -c "项目名" 按标题恢复。',
    "hermes -w 创建隔离的 git 工作树 — 适合并行代理工作流。",
    "hermes chat -t web,terminal 仅启用特定工具集进行专注会话。",
    "hermes chat -s github-pr-workflow 启动时预加载技能。",
    "hermes --yolo 绕过整个会话的所有危险命令批准提示。",
    
    # --- CLI 子命令 ---
    "hermes doctor --fix 诊断并自动修复配置和依赖问题。",
    "hermes config edit 在默认编辑器中打开 config.yaml。",
    "hermes sessions browse 打开交互式会话选择器并支持搜索。",
    "hermes skills search react --source skills-sh 搜索 skills.sh 公共目录。",
    "hermes mcp add github --command npx 从命令行添加 MCP 服务器。",
    "hermes completion bash >> ~/.bashrc 为所有命令和 profile 启用 tab 补全。",
    "hermes backup 创建整个 Hermes 主目录的 zip 备份。",
    "hermes profile create coder 创建隔离的 profile，成为独立的 coder 命令。",
    "hermes update 自动同步所有 profile 的新捆绑技能。",
    
    # --- 配置 ---
    "在 config.yaml 中设置 display.bell_on_complete: true，长任务完成时响铃。",
    "设置 display.streaming: true 以实时查看模型生成的 token。",
    "设置 display.show_reasoning: true 以查看模型的链式推理。",
    "设置 compression.threshold: 0.50 控制自动压缩触发的时机（默认：上下文的 50%）。",
    "设置 agent.max_turns: 200 让代理每轮进行更多工具调用步骤。",
    "设置 approvals.mode: smart 让 LLM 自动批准安全命令并自动拒绝危险命令。",
    "设置 fallback_model 自动故障转移到备用提供者。",
    "环境变量替换在 config.yaml 中生效：使用 ${VAR_NAME} 语法。",
    
    # --- 工具与功能 ---
    "execute_code 运行调用 Hermes 工具的 Python 脚本 — 结果不进入上下文。",
    "delegate_task 生成最多 3 个并发子代理，隔离上下文进行并行工作。",
    "web_extract 支持 PDF URL — 传递任何 PDF 链接并转换为 markdown。",
    "search_files 基于 ripgrep，比 grep 更快 — 用它代替终端 grep。",
    "patch 使用 9 种模糊匹配策略，微小的空格差异不会破坏编辑。",
    "read_file 在文件未找到时建议相似的文件名。",
    "browser_vision 截取屏幕并用 AI 分析 — 适用于 CAPTCHA 和视觉内容。",
    "image_generate 使用 FLUX 2 Pro 创建图像并自动 2 倍放大。",
    "text_to_speech 将文本转换为音频 — 在 Telegram 上作为语音气泡播放。",
    "session_search 在所有过去的对话中执行全文搜索。",
    "代理自动将偏好、更正和环境事实保存到内存。",
    
    # --- Profile ---
    "每个 profile 都有自己的配置、API 密钥、内存、会话、技能和定时任务。",
    "Profile 名称成为 shell 命令 — 'hermes profile create coder' 创建'coder'命令。",
    
    # --- 会话 ---
    "会话在第一次交互后自动生成描述性标题 — 无需手动命名。",
    '会话标题支持谱系："我的项目" → "我的项目 #2" → "我的项目 #3"。',
    "退出时，Hermes 打印带会话 ID 和统计信息的恢复命令。",
    "hermes -r SESSION_ID 通过 ID 恢复任何特定的过去会话。",
    
    # --- 内存 ---
    "内存是冻结的快照 — 更改仅在下次会话开始时出现在系统提示中。",
    "内存条目自动扫描提示注入和窃取模式。",
    "代理有两个内存存储：个人笔记（约 2200 字符）和用户配置文件（约 1375 字符）。",
    
    # --- 技能 ---
    "80 多个捆绑技能涵盖 github、creative、mlops、productivity、research 等。",
    "每个安装的技能自动成为斜杠命令 — 输入 / 查看全部。",
    "hermes skills install official/security/1password 从仓库安装可选技能。",
    "代理可以使用 skill_manage 创建自己的技能作为过程内存。",
    
    # --- 定时任务 ---
    '定时任务可以附加技能：hermes cron add --skill blogwatcher "检查新帖子"。',
    "定时任务交付目标包括 telegram、discord、slack、email、sms 等 12+ 平台。",
    "如果定时任务响应以 [SILENT] 开头，则抑制交付 — 适合仅监控任务。",
    
    # --- 语音 ---
    "如果安装了 faster-whisper，语音模式无需 API 密钥（免费本地语音转文本）。",
    "五种 TTS 提供者：Edge TTS（免费）、ElevenLabs、OpenAI、NeuTTS（免费本地）、MiniMax。",
    "/voice on 在 CLI 中启用语音模式。Ctrl+B 切换按住说话录音。",
    "流式 TTS 在生成时播放句子 — 无需等待完整响应。",
    
    # --- 网关与消息 ---
    "Hermes 运行在 18 个平台上：Telegram、Discord、Slack、WhatsApp、Signal、Matrix、email 等。",
    "hermes gateway install 将其设置为系统服务，启动时自动运行。",
    "Webhook 路由支持 HMAC 验证、速率限制和事件过滤。",
    "API 服务器提供与 Open WebUI 和 LibreChat 兼容的 OpenAI 兼容端点。",
    
    # --- 安全 ---
    "危险命令批准有 4 个级别：once、session、always（永久白名单）、deny。",
    "智能批准模式使用 LLM 自动批准安全命令并标记危险命令。",
    "SSRF 保护阻止私有网络、回环、链路本地和云元数据地址。",
    
    # --- 上下文与压缩 ---
    "上下文达到阈值时自动压缩 — 内存被刷新并总结历史。",
    "状态栏随着上下文填充变为黄色、橙色、然后红色。",
    "SOUL.md 位于 ~/.hermes/SOUL.md 是代理的主要身份 — 自定义它以塑造行为。",
    "Hermes 从 .hermes.md、AGENTS.md、CLAUDE.md 或 .cursorrules 加载项目上下文（第一个匹配）。",
    
    # --- 浏览器 ---
    "五种浏览器提供者：本地 Chromium、Browserbase、Browser Use、Camofox 和 Firecrawl。",
    "Camofox 是反检测浏览器 — 带有 C++ 指纹伪造的 Firefox 分支。",
    
    # --- MCP ---
    "MCP 服务器在 config.yaml 中配置 — 支持 stdio 和 HTTP 传输。",
    "每服务器工具过滤：tools.include 白名单和 tools.exclude 黑名单特定工具。",
    "MCP 服务器在运行时自动生成工具集 — hermes tools 可以每平台切换它们。",
    
    # --- 检查点与回滚 ---
    "没有文件修改时检查点零开销 — 默认启用。",
    "预回滚快照自动保存，以便你可以撤销撤销操作。",
    "/rollback 还撤销对话轮次，因此代理不记得回滚的更改。",
    "检查点使用 ~/.hermes/checkpoints/中的影子仓库 — 从不触碰项目的.git。",
]


# Backward compatibility: TIPS alias for existing tests/code
TIPS = TIPS_EN


def get_random_tip(exclude_recent: int = 0) -> str:
    """Return a random tip string.

    Args:
        exclude_recent: not used currently; reserved for future
            deduplication across sessions.
    """
    lang = _detect_tip_language()
    tips = TIPS_ZH if lang.startswith("zh") else TIPS_EN
    return random.choice(tips)


