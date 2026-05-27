"""Random tips shown at CLI session start to help users discover features."""

import random


# ---------------------------------------------------------------------------
# Tip corpus — one-liners covering slash commands, CLI flags, config,
# keybindings, tools, gateway, skills, profiles, and workflow tricks.
# ---------------------------------------------------------------------------

TIPS = [
    # --- Slash Commands ---
    "/background <prompt> (alias /bg or /btw) runs a task in a separate session while your current one stays free.",
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
    "/browser connect attaches browser tools to your running Chromium-family browser via CDP.",
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
    "Alt+Enter inserts a newline for multi-line input. (Windows Terminal intercepts Alt+Enter — use Ctrl+Enter instead.)",
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
    "Save money: hermes tools disables unused tools, hermes skills config trims skills down.",
    "/reasoning low or /reasoning minimal cuts thinking depth below the default (medium) — faster, cheaper responses.",
    "hermes models routes vision, compression, and aux tasks to cheaper models — cuts background token cost 85%+ without downgrading your main chat model.",

    # --- Configuration ---
    "Set display.bell_on_complete: true in config.yaml to hear a bell when long tasks finish.",
    "Set display.streaming: true to see tokens appear in real time as the model generates.",
    "Set display.show_reasoning: true to watch the model's chain-of-thought reasoning.",
    "Set display.compact: true to reduce whitespace in output for denser information.",
    "Set display.busy_input_mode: queue to queue messages instead of interrupting the agent, or steer to inject them mid-run via /steer.",
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
    "delegate_task spawns up to 3 concurrent sub-agents by default (delegation.max_concurrent_children) with isolated contexts for parallel work.",
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
    "Hermes runs on 21 messaging platforms: Telegram, Discord, Slack, WhatsApp, Signal, Matrix, IRC, Microsoft Teams, email, and more.",
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
    "agent.api_max_retries (default 3) controls how many times the agent retries a failed API call before surfacing the error — lower it for fast fallback.",
    "The gateway caches AIAgent instances per session — destroying this cache breaks Anthropic prompt caching.",
    "Any website can expose skills via /.well-known/skills/index.json — the skills hub discovers them automatically.",
    "The skills audit log at ~/.hermes/skills/.hub/audit.log tracks every install and removal operation.",
    "Stale git worktrees are auto-cleaned: 24-72h old with no unpushed commits get pruned on startup.",
    "Each profile gets its own subprocess HOME at HERMES_HOME/home/ — isolated git, ssh, npm, gh configs.",
    "HERMES_HOME_MODE env var (octal, e.g. 0701) sets custom directory permissions for web server traversal.",
    "Container mode: place .container-mode in HERMES_HOME and the host CLI auto-execs into the container.",
    "Ctrl+C has 5 priority tiers: cancel recording → cancel prompts → cancel picker → interrupt agent → exit.",
    "Every interrupt during an agent run is logged to ~/.hermes/interrupt_debug.log with timestamps.",
    "BROWSER_CDP_URL connects browser tools to any running Chromium-family browser — accepts WebSocket, HTTP, or host:port.",
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

    # --- Advanced Slash Commands ---
    '/steer <prompt> injects a note after the next tool call — nudge direction mid-task without interrupting.',
    '/goal <text> sets a standing Ralph-loop objective — Hermes auto-continues turn after turn until a judge says done.',
    '/snapshot create [label] saves a full state snapshot of Hermes config; /snapshot restore <id> reverts later.',
    '/copy [N] copies the last assistant response to your clipboard, or the Nth-from-last with a number.',
    '/redraw forces a full UI repaint, fixing terminal drift after tmux resize or mouse selection artifacts.',
    '/agents (alias /tasks) shows active agents and running background tasks across the current session.',
    '/footer toggles the gateway footer on final replies showing model, tool counts, and turn timing.',
    '/busy queue|steer|interrupt controls what pressing Enter does while Hermes is working.',
    '/topic in Telegram DMs enables user-managed multi-session topic mode — /topic <id> restores past sessions inline.',
    '/approve session|always runs a pending dangerous command with your chosen trust scope; /deny rejects it.',
    '/restart gracefully restarts the gateway after draining active runs, then pings the requester when back up.',
    '/kanban boards switch <slug> changes the active multi-project Kanban board from inside chat.',
    '/reload reloads ~/.hermes/.env into the running session — pick up new API keys without restarting.',

    # --- Cron (no-agent & scripts) ---
    'cronjob with no_agent=True runs a script on schedule and sends its stdout directly — zero tokens, zero LLM.',
    'An empty cron script stdout means silent tick — nothing is delivered, perfect for threshold watchdogs.',
    "HERMES_CRON_MAX_PARALLEL (default 4) caps how many cron jobs run per tick so bursts don't saturate your keys.",

    # --- Gateway Hooks ---
    'Gateway hooks live under ~/.hermes/hooks/<name>/ with HOOK.yaml + handler.py — handler must be named `handle`.',
    'Hook events include gateway:startup, session:start, agent:step, and command:* wildcard subscriptions.',
    'Drop a ~/.hermes/BOOT.md checklist and a gateway:startup hook runs it as a one-shot agent every boot.',

    # --- Curator ---
    'hermes curator run --dry-run previews what the curator would archive or consolidate without mutating anything.',
    "hermes curator pin <skill> hard-fences a skill against both auto-archival and the agent's skill_manage tool.",
    'hermes curator rollback restores skills from a pre-run snapshot — backups live under skills/.curator_backups/.',

    # --- Credential Pools & Routing ---
    'hermes auth reset <provider> clears all cooldowns and exhaustion flags on a credential pool.',
    'credential_pool_strategies.<provider>: round_robin cycles keys evenly instead of the fill_first default.',
    'use_gateway: true per-tool routes web, image, tts, or browser through your Nous subscription — no extra keys.',
    'provider_routing.data_collection: deny excludes data-storing providers on OpenRouter.',
    'provider_routing.require_parameters: true only routes to providers that support every param in your request.',

    # --- TUI & Dashboard ---
    'HERMES_TUI_RESUME=1 auto-re-attaches to the most recent TUI session on launch — handy after SSH drops.',
    "HERMES_TUI_THEME=light|dark|<hex> forces the TUI theme on terminals that don't set COLORFGBG.",
    'Ctrl+G or Ctrl+X Ctrl+E in the TUI opens the input buffer in $EDITOR for long multi-line prompts.',
    'The TUI renders LaTeX inline — $E=mc^2$ becomes Unicode math instead of raw TeX.',
    'hermes dashboard launches a local web UI at 127.0.0.1:9119 — zero data leaves localhost.',
    'hermes dashboard --tui embeds the full Hermes TUI in your browser via xterm.js and a WebSocket PTY.',
    'Drop a YAML in ~/.hermes/dashboard-themes/ with two palette colors to reskin the entire dashboard.',
    'Dashboard plugins are drop-in: manifest.json + JS bundle in ~/.hermes/dashboard-plugins/ — no npm build required.',
    'layoutVariant: cockpit in a dashboard theme adds a 260px left rail that plugins can populate via the sidebar slot.',

    # --- Env Vars & Config Gates ---
    "display.tool_progress_command: true exposes /verbose on messaging platforms; it's CLI-only by default.",
    'HERMES_BACKGROUND_NOTIFICATIONS=result only pings when background tasks finish (vs all/error/off).',
    'HERMES_WRITE_SAFE_ROOT restricts write_file and patch to a directory prefix; writes outside require approval.',
    'HERMES_IGNORE_RULES skips auto-injection of AGENTS.md, SOUL.md, .cursorrules, memory, and preloaded skills.',
    'HERMES_ACCEPT_HOOKS auto-approves unseen shell hooks declared in config.yaml without a TTY prompt.',
    'auxiliary.goal_judge.model routes the /goal judge to a cheap fast model to keep loop cost near zero.',
    'Checkpoints skip directories with more than 50,000 files to avoid slow git operations on massive monorepos.',

    # --- TTS ---
    'tts.provider: piper runs 44-language local TTS on CPU — voices auto-download to ~/.hermes/cache/piper-voices/.',
    'tts.providers.<name>.type: command wires any CLI TTS engine with {input_path} and {output_path} placeholders.',

    # --- API Server & Proxy ---
    'API_SERVER_ENABLED=true runs an OpenAI-compatible endpoint alongside the gateway for Open WebUI and LibreChat.',
    'GATEWAY_PROXY_URL runs a split setup: platform I/O locally, agent work delegated to a remote API server.',

    # --- Platform-specific ---
    'MATRIX_DEVICE_ID pins a stable device ID for E2EE — without it, keys rotate every start and historic decrypt breaks.',
    'TELEGRAM_WEBHOOK_SECRET is required whenever TELEGRAM_WEBHOOK_URL is set — generate with openssl rand -hex 32.',

    # --- Batch ---
    "batch_runner.py --resume content-matches completed prompts by text so dataset reorders don't re-run finished work.",

    # --- Less-Known Slash Commands ---
    '/new starts a fresh session in place (alias /reset) — fresh session ID, clean history, CLI stays open.',
    '/clear wipes the terminal screen AND starts a new session — one shortcut for a visual reset.',
    '/history prints the current conversation in-line without leaving the CLI — useful for a quick re-read.',
    '/save writes the current conversation to disk without ending the session.',
    '/status shows session info at a glance: ID, title, model, token usage, and elapsed time.',
    '/image <path> attaches a local image file for your next prompt without pasting or drag-and-drop.',
    '/platforms shows gateway and messaging-platform connection status right from inside chat.',
    '/commands paginates the full slash-command + installed-skill list — useful on platforms without tab completion.',
    '/toolsets lists every available toolset so you know what -t/--toolsets accepts.',
    '/gquota shows Google Gemini Code Assist quota usage with progress bars when that provider is active.',
    '/voice tts toggles TTS-only mode — agent replies out loud but you still type your prompts.',
    '/reload-skills re-scans ~/.hermes/skills/ so drop-in skills appear without restarting the session.',
    '/indicator kaomoji|emoji|unicode|ascii picks the TUI busy-indicator style shown during agent runs.',
    '/debug uploads a support bundle (system info + logs) and returns shareable links — works in chat too.',

    # --- CLI Subcommands & Flags ---
    'hermes -z "<prompt>" is the purest one-shot: final answer on stdout, nothing else — ideal for piping in scripts.',
    'hermes chat --pass-session-id injects the session ID into the system prompt so the agent can self-reference it.',
    'hermes chat --image path/to/pic.png attaches a local image to a single -q query without a separate upload step.',
    'hermes chat --ignore-user-config skips ~/.hermes/config.yaml — reproducible bug reports and CI runs.',
    "hermes chat --source tool tags programmatic chats so they don't clutter hermes sessions list.",
    'hermes dump --show-keys includes redacted API key fingerprints for deeper support debugging.',
    'hermes sessions rename <ID> "new title" renames any past session; hermes sessions delete <ID> removes one.',
    'hermes import restores a session export or profile archive produced by sessions export or profile export.',
    'hermes fallback manages the fallback_model chain interactively — no hand-editing config.yaml.',
    'hermes pairing rotates the DM pairing token — the first messager after rotation claims access to the bot.',
    'hermes setup walks first-time users through provider, keys, and platform wiring in one interactive flow.',
    'hermes status --deep runs the full health sweep across every component; plain hermes status is the quick view.',

    # --- Agent Behavior Env Vars ---
    'HERMES_AGENT_TIMEOUT=0 disables the gateway inactivity kill for a running agent — use for long research runs.',
    'HERMES_ENABLE_PROJECT_PLUGINS=1 auto-loads repo-local plugins from ./.hermes/plugins/ — trust-gated by design.',
    "HERMES_DISABLE_FILE_STATE_GUARD=1 turns off the 'file changed since you read it' guard on patch and write_file.",
    'HERMES_ALLOW_PRIVATE_URLS=true lets web tools hit localhost and private networks — off by default in gateway mode.',
    'HERMES_OPTIONAL_SKILLS=name1,name2 auto-installs extra optional-catalog skills on first run per profile.',
    'HERMES_BUNDLED_SKILLS points at a custom bundled-skill tree — used by Homebrew and Nix packaging.',
    'HERMES_DUMP_REQUEST_STDOUT=1 dumps every API request payload to stdout instead of log files.',
    'HERMES_OAUTH_TRACE=1 logs redacted OAuth token exchange and refresh attempts for debugging provider auth.',
    'HERMES_STREAM_RETRIES (default 3) controls mid-stream reconnect attempts on transient network errors.',

    # --- Gateway Behavior Env Vars ---
    'HERMES_GATEWAY_BUSY_ACK_ENABLED=false silences the ⚡/⏳/⏩ ack messages when a user messages a busy agent.',
    'HERMES_AGENT_NOTIFY_INTERVAL (default 180s) sets how often the gateway pings with progress on long turns.',
    'HERMES_RESTART_DRAIN_TIMEOUT (default 900s) caps how long /restart waits for in-flight runs before forcing.',
    'HERMES_CHECKPOINT_TIMEOUT (default 30s) caps filesystem checkpoint creation — raise it on huge monorepos.',

    # --- Auxiliary Tasks & Image Generation ---
    'image_gen.model in config.yaml picks the FAL model: flux-2/klein, gpt-image-2, nano-banana-pro, and more.',
    'image_gen.provider routes image generation through a plugin (OpenAI Images, Codex, FAL) instead of the default.',
    'AUXILIARY_VISION_BASE_URL + AUXILIARY_VISION_API_KEY point vision analysis at any OpenAI-compatible endpoint.',

    # --- Security ---
    'security.tirith_fail_open: false makes Hermes block commands when the tirith scanner itself errors out.',
    'TIRITH_FAIL_OPEN env var overrides the tirith_fail_open config — a quick toggle without editing config.yaml.',

    # --- Sessions & Source Tags ---
    '--source tool chats are excluded from hermes sessions list by default — set --source explicitly to see them.',
    'Session IDs are timestamp-prefixed (20250305_091523_abcd) so sorting works naturally in ls and jq.',

    # --- Misc ---
    'API_SERVER_MODEL_NAME customizes the model name on /v1/models — essential for multi-profile Open WebUI setups.',
    'Dashboard plugins are served from /dashboard-plugins/<name>/ — drop files into ~/.hermes/dashboard-plugins/.',
]


# ---------------------------------------------------------------------------
# Chinese translations for tips
# ---------------------------------------------------------------------------
TIPS_ZH = [
    # --- 斜杠命令 ---
    "/background <提示>（别名 /bg 或 /btw）在单独的会话中运行任务，当前会话不受影响。",
    "/branch 分叉当前会话，方便探索不同方向而不丢失进度。",
    "/compress 在对话变长时手动压缩上下文。",
    "/rollback 列出文件系统检查点 — 将代理修改的文件恢复到任何先前状态。",
    "/rollback diff 2 预览检查点 2 以来的更改，不恢复任何内容。",
    "/rollback 2 src/file.py 从特定检查点恢复单个文件。",
    '/title "我的项目" 为会话命名 — 之后可用 /resume 或 hermes -c 恢复。',
    "/resume 从之前命名的会话继续。",
    "/queue <提示> 将消息排队到下一轮，不中断当前任务。",
    "/undo 移除最后一组用户/助手交互。",
    "/retry 重新发送上一条消息 — 当代理回复不太对时很有用。",
    "/verbose 循环切换工具进度显示：关闭 → 新增 → 全部 → 详细。",
    "/reasoning high 增加模型的思考深度。/reasoning show 显示推理过程。",
    "/fast 切换优先处理以获得更快的 API 响应（取决于提供商）。",
    "/yolo 跳过本次会话剩余的所有危险命令审批提示。",
    "/model 让你在会话中切换模型 — 试试 /model sonnet 或 /model gpt-5。",
    "/model --global 永久更改默认模型。",
    "/personality pirate 设置有趣的个性 — 14 个内置选项，从 kawaii 到 shakespeare。",
    "/approve 批准待处理的命令 — 在审批模式下代理等待你确认。",
    "/deny 拒绝待处理的命令 — 代理将跳过该操作。",
    "/commands 列出所有可用的斜杠命令和已安装的技能命令。",
    "/stop 立即停止正在运行的代理。",
    "/status 显示当前会话信息：ID、标题、模型、token 用量和耗时。",
    "/compact 压缩对话历史以释放上下文空间 — 保留关键信息，删除冗余细节。",
    "/yolo off 重新启用危险命令审批 — 在 /yolo 之后恢复安全模式。",
    "/debug 上传支持包（系统信息 + 日志）并返回可分享链接 — 在聊天中也能用。",

    # --- CLI 标志和环境变量 ---
    'hermes -q "提示" 是最简洁的一次性调用：只有最终答案输出到 stdout，适合脚本管道。',
    "HERMES_QUIET=1 抑制 CLI 横幅和 spinner — 适合脚本和 CI。",
    "HERMES_SILENT=1 完全静默 — 不输出任何内容（适合只关心退出码的场景）。",
    "HERMES_NO_UPDATE_CHECK=1 跳过启动时的更新检查 — 加速启动。",
    "HERMES_STREAM_RETRIES（默认 3）控制瞬态网络错误时的中流重连尝试次数。",
    "hermes gateway run --port 9229 在 9229 端口启动网关，与默认实例共存。",

    # --- 配置和模型 ---
    "config.yaml 中的 model.fallbacks 定义了故障转移链 — 主模型失败时自动尝试下一个。",
    "config.yaml 中的 custom_providers 允许你添加任何 OpenAI 兼容的 API 端点。",
    "config.yaml 中的 model.context_length 覆盖自动检测 — 本地模型通常需要手动设置。",
    "provider_routing.data_collection: deny 排除会存储数据的 OpenRouter 提供商。",
    "credential_pool_strategies.<provider>: round_robin 均匀轮询密钥，而非默认的 fill_first。",
    "use_gateway: true 每个工具路由 web、image、tts 或 browser 通过你的 Nous 订阅 — 无需额外密钥。",

    # --- TUI 和面板 ---
    "HERMES_TUI_RESUME=1 启动时自动附加到最近的 TUI 会话 — SSH 断开后很方便。",
    "HERMES_TUI_THEME=light|dark|<hex> 在不设置 COLORFGBG 的终端上强制 TUI 主题。",
    "TUI 中 Ctrl+G 或 Ctrl+X Ctrl+E 在 $EDITOR 中打开输入缓冲区，适合长多行提示。",
    "TUI 内联渲染 LaTeX — $E=mc^2$ 变成 Unicode 数学而非原始 TeX。",
    "hermes dashboard 在 127.0.0.1:9119 启动本地 Web UI — 数据不离开本机。",

    # --- 环境变量和配置开关 ---
    "display.tool_progress_command: true 在消息平台上暴露 /verbose；默认仅 CLI 可用。",
    "HERMES_BACKGROUND_NOTIFICATIONS=result 仅在后台任务完成时通知（对比 all/error/off）。",
    "HERMES_WRITE_SAFE_ROOT 限制 write_file 和 patch 到目录前缀；外部写入需要审批。",
    "HERMES_IGNORE_RULES 跳过自动注入 AGENTS.md、SOUL.md、.cursorrules、memory 和预加载技能。",
    "auxiliary.goal_judge.model 将 /goal 判定器路由到便宜快速的模型，保持循环成本接近零。",
    "检查点跳过超过 50,000 个文件的目录，避免在大型 monorepo 上执行慢速 git 操作。",

    # --- TTS ---
    "tts.provider: piper 在 CPU 上运行 44 种语言的本地 TTS — 语音模型自动下载到 ~/.hermes/cache/piper-voices/。",
    "tts.providers.<name>.type: command 用 {input_path} 和 {output_path} 占位符连接任何 CLI TTS 引擎。",

    # --- API 服务器和代理 ---
    "API_SERVER_ENABLED=true 在网关旁运行 OpenAI 兼容端点，供 Open WebUI 和 LibreChat 使用。",
    "GATEWAY_PROXY_URL 运行拆分设置：本地处理平台 I/O，代理工作委派到远程 API 服务器。",

    # --- 平台特定 ---
    "MATRIX_DEVICE_ID 固定稳定的设备 ID 用于 E2EE — 否则每次启动密钥都会轮换，历史解密会失败。",
    "TELEGRAM_WEBHOOK_SECRET 在设置 TELEGRAM_WEBHOOK_URL 时必须 — 用 openssl rand -hex 32 生成。",

    # --- 批量 ---
    "batch_runner.py --resume 按文本内容匹配已完成的提示，避免数据集重排时重复运行已完成的工作。",

    # --- 不太知名的斜杠命令 ---
    "/new 在当前位置启动新会话（别名 /reset）— 新会话 ID、干净历史、CLI 保持打开。",
    "/clear 清除终端屏幕并启动新会话 — 一个快捷键完成视觉重置。",
    "/history 在 CLI 中内联打印当前对话 — 方便快速回看。",
    "/save 将当前对话写入磁盘但不结束会话。",
    "/status 显示会话信息一览：ID、标题、模型、token 用量和耗时。",
    "/image <路径> 为下一条提示附加本地图片文件，无需粘贴或拖放。",
    "/platforms 直接在聊天中显示网关和消息平台连接状态。",
    "/commands 分页显示完整斜杠命令 + 已安装技能列表 — 在没有 Tab 补全的平台上很有用。",
    "/toolsets 列出所有可用工具集，让你知道 -t/--toolsets 接受什么。",
    "/voice tts 切换纯 TTS 模式 — 代理回复出声，但你仍然输入提示。",
    "/reload-skills 重新扫描 ~/.hermes/skills/，无需重启会话即可使用新技能。",
    "/debug 上传支持包（系统信息 + 日志）并返回可分享链接 — 在聊天中也能用。",

    # --- CLI 子命令和标志 ---
    'hermes -z "提示" 是最纯粹的一次性调用：最终答案在 stdout，其他全无 — 适合脚本管道。',
    "hermes chat --pass-session-id 将会话 ID 注入系统提示，让代理可以自引用。",
    "hermes chat --image path/to/pic.png 将本地图片附加到单个 -q 查询，无需单独上传。",
    "hermes chat --ignore-user-config 跳过 ~/.hermes/config.yaml — 可复现的 bug 报告和 CI 运行。",
    'hermes sessions rename <ID> "新标题" 重命名任何过去的会话；hermes sessions delete <ID> 删除一个。',
    "hermes import 恢复由 sessions export 或 profile export 产生的会话导出或配置文件归档。",
    "hermes fallback 交互式管理 fallback_model 链 — 无需手动编辑 config.yaml。",
    "hermes pairing 轮换 DM 配对令牌 — 轮换后第一个发消息的人获得机器人的访问权。",
    "hermes setup 通过一个交互式流程引导新用户完成提供商、密钥和平台配置。",
    "hermes status --deep 运行完整的健康检查；普通的 hermes status 是快速查看。",

    # --- 代理行为环境变量 ---
    "HERMES_AGENT_TIMEOUT=0 禁用网关不活动终止 — 用于长时间研究运行。",
    "HERMES_ENABLE_PROJECT_PLUGINS=1 自动从 ./.hermes/plugins/ 加载仓库本地插件 — 内置信任门控。",
    "HERMES_ALLOW_PRIVATE_URL=true 让 web 工具访问 localhost 和私有网络 — 网关模式下默认关闭。",
    "HERMES_OPTIONAL_SKILLS=name1,name2 在首次运行时自动安装额外的可选目录技能。",
    "HERMES_DUMP_REQUEST_STDOUT=1 将每个 API 请求负载转储到 stdout 而非日志文件。",
    "HERMES_STREAM_RETRIES（默认 3）控制瞬态网络错误时的中流重连尝试次数。",

    # --- 安全 ---
    "security.tirith_fail_open: false 让 Hermes 在 tirith 扫描器本身出错时阻止命令。",
    "TIRITH_FAIL_OPEN 环境变量覆盖 tirith_fail_open 配置 — 无需编辑 config.yaml 的快速切换。",

    # --- 会话和来源标签 ---
    "--source tool 聊天默认从 hermes sessions 列表中排除 — 显式设置 --source 以显示。",
    "会话 ID 以时间戳为前缀（20250305_091523_abcd），排序在 ls 和 jq 中自然生效。",

    # --- 杂项 ---
    "API_SERVER_MODEL_NAME 自定义 /v1/models 上的模型名称 — 多配置文件 Open WebUI 设置必需。",
    "仪表板插件从 /dashboard-plugins/<name>/ 提供 — 将文件放入 ~/.hermes/dashboard-plugins/。",
]


def get_random_tip(exclude_recent: int = 0) -> str:
    """Return a random tip string in the current locale.

    Args:
        exclude_recent: not used currently; reserved for future
            deduplication across sessions.
    """
    try:
        from agent.i18n import _config_language_cached
        lang = _config_language_cached()
        if lang and lang.startswith("zh"):
            return random.choice(TIPS_ZH)
    except Exception:
        pass
    return random.choice(TIPS)


