---
name: hermes-agent
description: Complete guide to using and extending Hermes Agent — CLI usage, setup, configuration, spawning additional agents, gateway platforms, skills, voice, tools, profiles, and a concise contributor reference. Load this skill when helping users configure Hermes, troubleshoot issues, spawn agent instances, or make code contributions.
version: 2.1.0
author: Hermes Agent + Teknium
license: MIT
metadata:
  hermes:
    tags: [hermes, setup, configuration, multi-agent, spawning, cli, gateway, development]
    homepage: https://github.com/NousResearch/hermes-agent
    related_skills: [claude-code, codex, opencode]
---

# Hermes Agent

Hermes Agent is an open-source AI agent framework by Nous Research that runs in your terminal, messaging platforms, and IDEs. It belongs to the same category as Claude Code (Anthropic), Codex (OpenAI), and OpenClaw — autonomous coding and task-execution agents that use tool calling to interact with your system. Hermes works with any LLM provider (OpenRouter, Anthropic, OpenAI, DeepSeek, local models, and 15+ others) and runs on Linux, macOS, and WSL.

What makes Hermes different:

- **Self-improving through skills** — Hermes learns from experience by saving reusable procedures as skills. When it solves a complex problem, discovers a workflow, or gets corrected, it can persist that knowledge as a skill document that loads into future sessions. Skills accumulate over time, making the agent better at your specific tasks and environment.
- **Persistent memory across sessions** — remembers who you are, your preferences, environment details, and lessons learned. Pluggable memory backends (built-in, Honcho, Mem0, and more) let you choose how memory works.
- **Multi-platform gateway** — the same agent runs on Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Email, and 8+ other platforms with full tool access, not just chat.
- **Provider-agnostic** — swap models and providers mid-workflow without changing anything else. Credential pools rotate across multiple API keys automatically.
- **Profiles** — run multiple independent Hermes instances with isolated configs, sessions, skills, and memory.
- **Extensible** — plugins, MCP servers, custom tools, webhook triggers, cron scheduling, and the full Python ecosystem.

People use Hermes for software development, research, system administration, data analysis, content creation, home automation, and anything else that benefits from an AI agent with persistent context and full system access.

**This skill helps you work with Hermes Agent effectively** — setting it up, configuring features, spawning additional agent instances, troubleshooting issues, finding the right commands and settings, and understanding how the system works when you need to extend or contribute to it.

**Docs:** https://hermes-agent.nousresearch.com/docs/

## Quick Start

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# Interactive chat (default)
hermes

# Single query
hermes chat -q "What is the capital of France?"

# Setup wizard
hermes setup

# Change model/provider
hermes model

# Check health
hermes doctor
```

---

## CLI Reference

### Global Flags

```
hermes [flags] [command]

  --version, -V             Show version
  --resume, -r SESSION      Resume session by ID or title
  --continue, -c [NAME]     Resume by name, or most recent session
  --worktree, -w            Isolated git worktree mode (parallel agents)
  --skills, -s SKILL        Preload skills (comma-separate or repeat)
  --profile, -p NAME        Use a named profile
  --yolo                    Skip dangerous command approval
  --pass-session-id         Include session ID in system prompt
```

No subcommand defaults to `chat`.

### Chat

```
hermes chat [flags]
  -q, --query TEXT          Single query, non-interactive
  -m, --model MODEL         Model (e.g. anthropic/claude-sonnet-4)
  -t, --toolsets LIST       Comma-separated toolsets
  --provider PROVIDER       Force provider (openrouter, anthropic, nous, etc.)
  -v, --verbose             Verbose output
  -Q, --quiet               Suppress banner, spinner, tool previews
  --checkpoints             Enable filesystem checkpoints (/rollback)
  --source TAG              Session source tag (default: cli)
```

### Configuration

```
hermes setup [section]      Interactive wizard (model|terminal|gateway|tools|agent)
hermes model                Interactive model/provider picker
hermes config               View current config
hermes config edit          Open config.yaml in $EDITOR
hermes config set KEY VAL   Set a config value
hermes config path          Print config.yaml path
hermes config env-path      Print .env path
hermes config check         Check for missing/outdated config
hermes config migrate       Update config with new options
hermes login [--provider P] OAuth login (nous, openai-codex)
hermes logout               Clear stored auth
hermes doctor [--fix]       Check dependencies and config
hermes status [--all]       Show component status
```

### Tools & Skills

```
hermes tools                Interactive tool enable/disable (curses UI)
hermes tools list           Show all tools and status
hermes tools enable NAME    Enable a toolset
hermes tools disable NAME   Disable a toolset

hermes skills list          List installed skills
hermes skills search QUERY  Search the skills hub
hermes skills install ID    Install a skill
hermes skills inspect ID    Preview without installing
hermes skills config        Enable/disable skills per platform
hermes skills check         Check for updates
hermes skills update        Update outdated skills
hermes skills uninstall N   Remove a hub skill
hermes skills publish PATH  Publish to registry
hermes skills browse        Browse all available skills
hermes skills tap add REPO  Add a GitHub repo as skill source
```

### MCP Servers

```
hermes mcp serve            Run Hermes as an MCP server
hermes mcp add NAME         Add an MCP server (--url or --command)
hermes mcp remove NAME      Remove an MCP server
hermes mcp list             List configured servers
hermes mcp test NAME        Test connection
hermes mcp configure NAME   Toggle tool selection
```

### Gateway (Messaging Platforms)

```
hermes gateway run          Start gateway foreground
hermes gateway install      Install as background service
hermes gateway start/stop   Control the service
hermes gateway restart      Restart the service
hermes gateway status       Check status
hermes gateway setup        Configure platforms
```

Supported platforms: Telegram, Discord, Slack, WhatsApp, Signal, Email, SMS, Matrix, Mattermost, Home Assistant, DingTalk, Feishu, WeCom, API Server, Webhooks, Open WebUI.

Platform docs: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/

### Sessions

```
hermes sessions list        List recent sessions
hermes sessions browse      Interactive picker
hermes sessions export OUT  Export to JSONL
hermes sessions rename ID T Rename a session
hermes sessions delete ID   Delete a session
hermes sessions prune       Clean up old sessions (--older-than N days)
hermes sessions stats       Session store statistics
```

### Cron Jobs

```
hermes cron list            List jobs (--all for disabled)
hermes cron create SCHED    Create: '30m', 'every 2h', '0 9 * * *'
hermes cron edit ID         Edit schedule, prompt, delivery
hermes cron pause/resume ID Control job state
hermes cron run ID          Trigger on next tick
hermes cron remove ID       Delete a job
hermes cron status          Scheduler status
```

### Webhooks

```
hermes webhook subscribe N  Create route at /webhooks/<name>
hermes webhook list         List subscriptions
hermes webhook remove NAME  Remove a subscription
hermes webhook test NAME    Send a test POST
```

### Profiles

```
hermes profile list         List all profiles
hermes profile create NAME  Create (--clone, --clone-all, --clone-from)
hermes profile use NAME     Set sticky default
hermes profile delete NAME  Delete a profile
hermes profile show NAME    Show details
hermes profile alias NAME   Manage wrapper scripts
hermes profile rename A B   Rename a profile
hermes profile export NAME  Export to tar.gz
hermes profile import FILE  Import from archive
```

### Credential Pools

```
hermes auth add             Interactive credential wizard
hermes auth list [PROVIDER] List pooled credentials
hermes auth remove P INDEX  Remove by provider + index
hermes auth reset PROVIDER  Clear exhaustion status
```

### Other

```
hermes insights [--days N]  Usage analytics
hermes update               Update to latest version
hermes pairing list/approve/revoke  DM authorization
hermes plugins list/install/remove  Plugin management
hermes honcho setup/status  Honcho memory integration
hermes memory setup/status/off  Memory provider config
hermes completion bash|zsh  Shell completions
hermes acp                  ACP server (IDE integration)
hermes claw migrate         Migrate from OpenClaw
hermes uninstall            Uninstall Hermes
```

---

## Slash Commands (In-Session)

Type these during an interactive chat session.

### Session Control
```
/new (/reset)        Fresh session
/clear               Clear screen + new session (CLI)
/retry               Resend last message
/undo                Remove last exchange
/title [name]        Name the session
/compress            Manually compress context
/stop                Kill background processes
/rollback [N]        Restore filesystem checkpoint
/background <prompt> Run prompt in background
/queue <prompt>      Queue for next turn
/resume [name]       Resume a named session
```

### Configuration
```
/config              Show config (CLI)
/model [name]        Show or change model
/provider            Show provider info
/prompt [text]       View/set system prompt (CLI)
/personality [name]  Set personality
/reasoning [level]   Set reasoning (none|low|medium|high|xhigh|show|hide)
/verbose             Cycle: off → new → all → verbose
/voice [on|off|tts]  Voice mode
/yolo                Toggle approval bypass
/skin [name]         Change theme (CLI)
/statusbar           Toggle status bar (CLI)
```

### Tools & Skills
```
/tools               Manage tools (CLI)
/toolsets            List toolsets (CLI)
/skills              Search/install skills (CLI)
/skill <name>        Load a skill into session
/cron                Manage cron jobs (CLI)
/reload-mcp          Reload MCP servers
/plugins             List plugins (CLI)
```

### Info
```
/help                Show commands
/commands [page]     Browse all commands (gateway)
/usage               Token usage
/insights [days]     Usage analytics
/status              Session info (gateway)
/profile             Active profile info
```

### Exit
```
/quit (/exit, /q)    Exit CLI
```

---

## Key Paths & Config

```
~/.hermes/config.yaml       Main configuration
~/.hermes/.env              API keys and secrets
~/.hermes/skills/           Installed skills
~/.hermes/sessions/         Session transcripts
~/.hermes/logs/             Gateway and error logs
~/.hermes/auth.json         OAuth tokens and credential pools
~/.hermes/hermes-agent/     Source code (if git-installed)
```

Profiles use `~/.hermes/profiles/<name>/` with the same layout.

### Config Sections

Edit with `hermes config edit` or `hermes config set section.key value`.

| Section | Key options |
|---------|-------------|
| `model` | `default`, `provider`, `base_url`, `api_key`, `context_length` |
| `agent` | `max_turns` (90), `tool_use_enforcement` |
| `terminal` | `backend` (local/docker/ssh/modal), `cwd`, `timeout` (180) |
| `compression` | `enabled`, `threshold` (0.50), `target_ratio` (0.20) |
| `display` | `skin`, `tool_progress`, `show_reasoning`, `show_cost` |
| `stt` | `enabled`, `provider` (local/groq/openai) |
| `tts` | `provider` (edge/elevenlabs/openai/kokoro/fish) |
| `memory` | `memory_enabled`, `user_profile_enabled`, `provider` |
| `security` | `tirith_enabled`, `website_blocklist` |
| `delegation` | `model`, `provider`, `max_iterations` (50) |
| `smart_model_routing` | `enabled`, `cheap_model` |
| `checkpoints` | `enabled`, `max_snapshots` (50) |

Full config reference: https://hermes-agent.nousresearch.com/docs/user-guide/configuration

### Providers

18 providers supported. Set via `hermes model` or `hermes setup`.

| Provider | Auth | Key env var |
|----------|------|-------------|
| OpenRouter | API key | `OPENROUTER_API_KEY` |
| Anthropic | API key | `ANTHROPIC_API_KEY` |
| Nous Portal | OAuth | `hermes login --provider nous` |
| OpenAI Codex | OAuth | `hermes login --provider openai-codex` |
| GitHub Copilot | Token | `COPILOT_GITHUB_TOKEN` |
| DeepSeek | API key | `DEEPSEEK_API_KEY` |
| Hugging Face | Token | `HF_TOKEN` |
| Z.AI / GLM | API key | `GLM_API_KEY` |
| MiniMax | API key | `MINIMAX_API_KEY` |
| Kimi / Moonshot | API key | `KIMI_API_KEY` |
| Alibaba / DashScope | API key | `DASHSCOPE_API_KEY` |
| Kilo Code | API key | `KILOCODE_API_KEY` |
| Custom endpoint | Config | `model.base_url` + `model.api_key` in config.yaml |

Plus: AI Gateway, OpenCode Zen, OpenCode Go, MiniMax CN, GitHub Copilot ACP.

Full provider docs: https://hermes-agent.nousresearch.com/docs/integrations/providers

### Toolsets

Enable/disable via `hermes tools` (interactive) or `hermes tools enable/disable NAME`.

| Toolset | What it provides |
|---------|-----------------|
| `web` | Web search and content extraction |
| `browser` | Browser automation (Browserbase, Camofox, or local Chromium) |
| `terminal` | Shell commands and process management |
| `file` | File read/write/search/patch |
| `code_execution` | Sandboxed Python execution |
| `vision` | Image analysis |
| `image_gen` | AI image generation |
| `tts` | Text-to-speech |
| `skills` | Skill browsing and management |
| `memory` | Persistent cross-session memory |
| `session_search` | Search past conversations |
| `delegation` | Subagent task delegation |
| `cronjob` | Scheduled task management |
| `clarify` | Ask user clarifying questions |
| `moa` | Mixture of Agents (off by default) |
| `homeassistant` | Smart home control (off by default) |

Tool changes take effect on `/reset` (new session). They do NOT apply mid-conversation to preserve prompt caching.

---

## Voice & Transcription

### STT (Voice → Text)

Voice messages from messaging platforms are auto-transcribed.

Provider priority (auto-detected):
1. **Local faster-whisper** — free, no API key: `pip install faster-whisper`
2. **Groq Whisper** — free tier: set `GROQ_API_KEY`
3. **OpenAI Whisper** — paid: set `VOICE_TOOLS_OPENAI_KEY`

Config:
```yaml
stt:
  enabled: true
  provider: local        # local, groq, openai
  local:
    model: base          # tiny, base, small, medium, large-v3
```

### TTS (Text → Voice)

| Provider | Env var | Free? |
|----------|---------|-------|
| Edge TTS | None | Yes (default) |
| ElevenLabs | `ELEVENLABS_API_KEY` | Free tier |
| OpenAI | `VOICE_TOOLS_OPENAI_KEY` | Paid |
| Kokoro (local) | None | Free |
| Fish Audio | `FISH_AUDIO_API_KEY` | Free tier |

Voice commands: `/voice on` (voice-to-voice), `/voice tts` (always voice), `/voice off`.

---

## Spawning Additional Hermes Instances

Run additional Hermes processes as fully independent subprocesses — separate sessions, tools, and environments.

### When to Use This vs delegate_task

| | `delegate_task` | Spawning `hermes` process |
|-|-----------------|--------------------------|
| Isolation | Separate conversation, shared process | Fully independent process |
| Duration | Minutes (bounded by parent loop) | Hours/days |
| Tool access | Subset of parent's tools | Full tool access |
| Interactive | No | Yes (PTY mode) |
| Use case | Quick parallel subtasks | Long autonomous missions |

### One-Shot Mode

```
terminal(command="hermes chat -q 'Research GRPO papers and write summary to ~/research/grpo.md'", timeout=300)

# Background for long tasks:
terminal(command="hermes chat -q 'Set up CI/CD for ~/myapp'", background=true)
```

### Interactive PTY Mode (via tmux)

Hermes uses prompt_toolkit, which requires a real terminal. Use tmux for interactive spawning:

```
# Start
terminal(command="tmux new-session -d -s agent1 -x 120 -y 40 'hermes'", timeout=10)

# Wait for startup, then send a message
terminal(command="sleep 8 && tmux send-keys -t agent1 'Build a FastAPI auth service' Enter", timeout=15)

# Read output
terminal(command="sleep 20 && tmux capture-pane -t agent1 -p", timeout=5)

# Send follow-up
terminal(command="tmux send-keys -t agent1 'Add rate limiting middleware' Enter", timeout=5)

# Exit
terminal(command="tmux send-keys -t agent1 '/exit' Enter && sleep 2 && tmux kill-session -t agent1", timeout=10)
```

### Multi-Agent Coordination

```
# Agent A: backend
terminal(command="tmux new-session -d -s backend -x 120 -y 40 'hermes -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t backend 'Build REST API for user management' Enter", timeout=15)

# Agent B: frontend
terminal(command="tmux new-session -d -s frontend -x 120 -y 40 'hermes -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t frontend 'Build React dashboard for user management' Enter", timeout=15)

# Check progress, relay context between them
terminal(command="tmux capture-pane -t backend -p | tail -30", timeout=5)
terminal(command="tmux send-keys -t frontend 'Here is the API schema from the backend agent: ...' Enter", timeout=5)
```

### Session Resume

```
# Resume most recent session
terminal(command="tmux new-session -d -s resumed 'hermes --continue'", timeout=10)

# Resume specific session
terminal(command="tmux new-session -d -s resumed 'hermes --resume 20260225_143052_a1b2c3'", timeout=10)
```

### Tips

- **Prefer `delegate_task` for quick subtasks** — less overhead than spawning a full process
- **Use `-w` (worktree mode)** when spawning agents that edit code — prevents git conflicts
- **Set timeouts** for one-shot mode — complex tasks can take 5-10 minutes
- **Use `hermes chat -q` for fire-and-forget** — no PTY needed
- **Use tmux for interactive sessions** — raw PTY mode has `\r` vs `\n` issues with prompt_toolkit
- **For scheduled tasks**, use the `cronjob` tool instead of spawning — handles delivery and retry

---

## Agent Swarm Orchestration

Multi-agent swarms enable parallel task execution across specialized agents. This section covers swarm architecture, shared memory systems, context optimization, and orchestration patterns for both small coding models and large context models.

### Swarm Helper Script

A helper script is available for quick swarm management:

```bash
~/.hermes/scripts/swarm.py <command>

Commands:
  init <project_name>           Initialize swarm with directory structure
  add-task <swarm_id> <task>    Add task to queue
  spawn <swarm_id> <agent>      Spawn tmux agent (with --task, --context minimal|full)
  send <session> <msg>          Send message to running agent
  read <session> [--lines N]    Read agent output
  status <swarm_id>             Show swarm status
  update <swarm_id>             Update progress (--agent, --task, --status, --step)
  monitor <swarm_id>            Wait for completion (--timeout, --poll)
  complete <swarm_id>           Mark complete, kill agents
  cleanup <swarm_id> [--archive] Archive or delete swarm
```

**Quick Start:**
```bash
# 1. Init swarm
swarm_id=$(python3 ~/.hermes/scripts/swarm.py init "MyApp" --desc "SaaS dashboard")

# 2. Add tasks
python3 ~/.hermes/scripts/swarm.py add-task $swarm_id backend "Build REST API" --agent backend
python3 ~/.hermes/scripts/swarm.py add-task $swarm_id frontend "Build React UI" --agent frontend

# 3. Spawn agents
python3 ~/.hermes/scripts/swarm.py spawn $swarm_id backend --task "Build user auth API" --context minimal
python3 ~/.hermes/scripts/swarm.py spawn $swarm_id frontend --task "Build login page" --context minimal

# 4. Monitor
python3 ~/.hermes/scripts/swarm.py monitor $swarm_id --timeout 60

# 5. Collect & cleanup
python3 ~/.hermes/scripts/swarm.py complete $swarm_id
```

### Swarm Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         ORCHESTRATOR (Parent Agent)      │
                    │  - Task decomposition                    │
                    │  - Agent spawning & coordination         │
                    │  - Context optimization & distribution   │
                    │  - Result aggregation & synthesis        │
                    └─────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
       ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
       │   Agent A   │         │   Agent B   │         │   Agent C   │
       │  (Specialist)│         │  (Specialist)│         │  (Specialist)│
       └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │         SHARED MEMORY STORE              │
                    │  - Task state & progress                 │
                    │  - Intermediate artifacts                │
                    │  - Agent-to-agent context references     │
                    │  - Final deliverables                    │
                    └─────────────────────────────────────────┘
```

### Shared Memory System

Shared memory is the backbone of effective swarm coordination. Agents read/write to a shared store instead of passing bulky context directly.

**Storage Backends:**

| Backend | Best For | Setup |
|---------|----------|-------|
| **JSON files** | Simple, portable, small teams | File-based, no dependencies |
| **SQLite** | Structured data, queries, larger swarms | Single file, ACID compliant |
| **Redis** | Real-time, high-frequency updates | External server required |
| **Hermes Memory** | Already integrated, cross-session | `hermes memory setup` |

**JSON Shared Store (Recommended for most cases):**

```bash
# Create shared memory directory
mkdir -p ~/.hermes/swarm/<project-name>/memory

# Core files:
#   task_queue.json     - Pending tasks for each agent
#   progress.json       - Agent progress updates
#   artifacts/         - Generated files (schemas, specs, code)
#   context/           - Shared context references
#   results/           - Final deliverables
```

**Schema for task_queue.json:**
```json
{
  "swarm_id": "project-alpha-20260406",
  "created_at": "2026-04-06T12:00:00Z",
  "tasks": {
    "backend": {
      "id": "backend-001",
      "description": "Build REST API for user management",
      "status": "pending|in_progress|completed|blocked",
      "assigned_to": null,
      "dependencies": [],
      "artifacts": [],
      "progress": [],
      "result": null,
      "created_at": "...",
      "updated_at": "..."
    }
  },
  "agents": {
    "backend": { "spawned_at": null, "pid": null, "status": "idle" }
  }
}
```

**Schema for progress.json:**
```json
{
  "backend-001": {
    "agent": "backend",
    "status": "in_progress",
    "step": "Implementing JWT authentication",
    "artifacts_created": ["auth/jwt.py", "middleware/auth.go"],
    "context_updates": ["Added JWT flow to shared context"],
    "blocked_by": null,
    "ready_for_handoff": false,
    "handoff_notes": null,
    "updated_at": "2026-04-06T12:15:00Z"
  }
}
```

### Context Partitioning (Critical for Small Models)

Small coding models (7B-13B) have limited context. Optimize by partitioning context across agents and the shared store.

**Principles:**
1. **Never dump everything to every agent** — each agent only gets what it needs
2. **Reference over content** — "see shared context file X" instead of copying content
3. **Incremental updates** — agents write summaries, not full transcripts
4. **Hierarchical context** — orchestrator holds global view, agents hold local view

**Context Distribution Strategy:**

| Model Size | Strategy | Prompt Size per Agent |
|-----------|----------|----------------------|
| ≤7B params | Heavy referencing, minimal prompts | <500 tokens |
| 7B-13B | Balanced, key context inline | 500-1500 tokens |
| 13B-34B | Moderate inlining | 1500-3000 tokens |
| >34B (large ctx) | Full context when needed | No strict limit |

**For Small Models - Optimized Prompt Template:**

```
# AGENT PROMPT (Small Model Optimized)

## YOUR TASK
{task_description}

## PROJECT CONTEXT (File Reference)
Read full context: ~/hermes/swarm/<project>/memory/context/project_overview.json

## YOUR SCOPE
- What you're building: {specific_deliverable}
- Where it lives: {file_paths}
- What's already there: {existing_artifacts}

## SHARED STORE (Write Here)
~/hermes/swarm/<project>/memory/artifacts/{your_work}.json

## COORDINATION
- Check progress: ~/hermes/swarm/<project>/memory/progress.json
- Update your status: Write to progress.json on each major step
- Signal completion: Set status="completed", artifacts=[list files]

## DEPENDENCIES (Wait for These)
{blocking_tasks}

## START
Begin work now. Update progress.json every 5 minutes.
```

**For Large Context Models - Full Context Template:**

```
# AGENT PROMPT (Large Context Optimized)

## SWARM OVERVIEW
Project: {name}
Goal: {high_level_objective}
Architecture: {system_design}
Timeline: {milestones}

## FULL PROJECT CONTEXT
{include relevant project docs, schemas, specs here}

## YOUR TASK
{detailed_task_description}

## EXISTING ARTIFACTS (Don't Recreate)
{list and include relevant existing code}

## COORDINATION PROTOCOL
- Shared store: ~/hermes/swarm/<project>/memory/
- Write all artifacts to artifacts/ directory
- Update progress.json on each phase completion
- Dependencies resolved via progress.json blocking status

## COMPLETION CRITERIA
{success_conditions}

## HANDOFF
When done, write summary to artifacts/{your_task}_handoff.json including:
- What was built
- API contracts exposed
- Artifacts created
- Recommendations for dependent agents
```

### Orchestration Patterns

**Pattern 1: Boss-Agent (Recommended for Most Cases)**

```
Orchestrator → Spawns → Specialist Agents → Write to → Shared Store
     ↑                                                    │
     └──────────── Reads Results ←────────────────────────┘
```

Best for: Most multi-agent workflows. One coordinator, multiple specialists.

```python
# Orchestrator logic (pseudocode)
def run_boss_agent_swarm(task, num_agents=3):
    swarm_id = create_swarm(task)
    context = build_initial_context(task)

    # Phase 1: Decompose and spawn
    subtasks = decompose_task(task, num_agents)
    for subtask in subtasks:
        spawn_agent(subtask, swarm_id, context)

    # Phase 2: Monitor and coordinate
    while not all_complete(subtasks):
        progress = read_progress(swarm_id)
        handle_blockers(progress)
        sleep(30)

    # Phase 3: Aggregate
    results = collect_results(swarm_id)
    return synthesize_output(results)
```

**Pattern 2: Pipeline (Sequential Handoffs)**

```
Agent A → Agent B → Agent C → Agent D
   ↓         ↓         ↓         ↓
 Shared Store (Each stage reads previous stage's output)
```

Best for: Tasks with strict dependencies (data flow: parse → transform → validate → render).

**Pattern 3: Peer Network (Event-Driven)**

```
       ┌──→ Agent A ──┐
       │              │
Orchestrator ───→ Agent B ──→ Shared Store ←── Agent C
       │              │
       └──→ Agent D ──┘
```

Best for: Multiple agents working on independent tasks that occasionally need to collaborate.

**Pattern 4: Hierarchical**

```
Top-Level Orchestrator
    ├── Middle Manager A ──→ Agent A1, Agent A2
    └── Middle Manager B ──→ Agent B1, Agent B2
```

Best for: Very large tasks requiring 10+ agents. Top orchestrator manages middle managers who each handle a subdomain.

### Spawning Swarm Agents

**Initialize Swarm:**
```bash
# Create swarm directory structure
SWARM_ID="myapp-$(date +%Y%m%d-%H%M%S)"
mkdir -p ~/.hermes/swarm/$SWARM_ID/{memory/{task_queue,progress,artifacts,context},logs}

# Create initial context
cat > ~/.hermes/swarm/$SWARM_ID/memory/context/project_overview.json << 'EOF'
{
  "swarm_id": "$SWARM_ID",
  "project": "My App",
  "goal": "Build a SaaS dashboard",
  "tech_stack": ["Node.js", "React", "PostgreSQL"],
  "shared_artifacts": {}
}
EOF
```

**Spawn via tmux (recommended for interactive):**
```bash
# Spawn orchestrator
tmux new-session -d -s orchestrator -x 120 -y 40 'hermes'

# Spawn specialist agents
tmux new-session -d -s backend -x 120 -y 40 'hermes -p swarm-backend'
tmux new-session -d -s frontend -x 120 -y 40 'hermes -p swarm-frontend'
tmux new-session -d -s devops -x 120 -y 40 'hermes -p swarm-devops'

# Wait for startup, send initial tasks
sleep 8
tmux send-keys -t orchestrator 'Initialize swarm $SWARM_ID. Architecture: backend (Node.js API), frontend (React dashboard), devops (Docker + CI/CD). Tech stack: PostgreSQL, Auth0. Begin task decomposition.' Enter
```

**Spawn via delegate_task (quick subtasks):**
```python
delegate_task(
    goal="Build user authentication module with JWT",
    context=f"""
    SWARM: {swarm_id}
    PROJECT: My App SaaS dashboard
    YOUR TASK: Implement auth module

    WRITE TO: ~/.hermes/swarm/{swarm_id}/memory/artifacts/auth_module.json
    PROGRESS: ~/.hermes/swarm/{swarm_id}/memory/progress.json

    REQUIREMENTS:
    - JWT-based authentication
    - User registration/login endpoints
    - Password hashing with bcrypt
    - Middleware for protected routes

    EXISTING: No auth yet. Create from scratch.

    COMPLETION: Write final artifact and update progress.json status='completed'
    """,
    toolsets=["terminal", "file"]
)
```

**Spawn via hermes chat -q (fire-and-forget):**
```bash
hermes chat -q "Build the PostgreSQL schema for the user management system. Output to ~/.hermes/swarm/$SWARM_ID/memory/artifacts/db_schema.sql" --source swarm-backend
```

### Orchestrator Best Practices

**1. Task Decomposition**
- Break work into ~30-minute chunks per agent
- Define clear interfaces between tasks (API contracts, data schemas)
- Identify dependencies upfront, sequence accordingly

**2. Context Injection**
```python
# Build minimal context for small models
def build_minimal_context(task, agent_role):
    return f"""
    PROJECT GOAL: {task.goal}
    YOUR ROLE: {agent_role}
    YOUR TASK: {task.description}

    READ: ~/hermes/swarm/{swarm_id}/memory/context/project_overview.json
    READ: ~/hermes/swarm/{swarm_id}/memory/context/{agent_role}_spec.json

    WRITE ARTIFACTS TO: ~/hermes/swarm/{swarm_id}/memory/artifacts/
    UPDATE PROGRESS: ~/hermes/swarm/{swarm_id}/memory/progress.json

    START: {task.instructions}
    """
```

**3. Progress Monitoring**
```python
def monitor_swarm(swarm_id, timeout_minutes=60):
    start = time.time()
    while time.time() - start < timeout_minutes * 60:
        progress = read_json(f"~/.hermes/swarm/{swarm_id}/memory/progress.json")
        statuses = [p["status"] for p in progress.values()]

        if all(s == "completed" for s in statuses):
            return "SUCCESS"

        blocked = [p for p in progress if p["status"] == "blocked"]
        if blocked:
            resolve_blocker(swarm_id, blocked[0])

        sleep(30)

    return "TIMEOUT"
```

**4. Result Aggregation**
```python
def aggregate_swarm_results(swarm_id):
    artifacts_dir = f"~/.hermes/swarm/{swarm_id}/memory/artifacts"
    progress = read_json(f"~/.hermes/swarm/{swarm_id}/memory/progress.json")

    results = {
        "completed": [],
        "summaries": []
    }

    for task_id, p in progress.items():
        if p["status"] == "completed":
            artifact = read_json(f"{artifacts_dir}/{task_id}_handoff.json")
            results["completed"].append(artifact)
            results["summaries"].append(artifact["summary"])

    return synthesize(results["summaries"])
```

### Context Window Optimization Checklist

**For Small Models (≤13B):**
- [ ] Each agent prompt <1500 tokens
- [ ] Heavy use of "READ FILE X" instead of inline content
- [ ] Agents write summaries, not full outputs
- [ ] Orchestrator holds global view, distills for agents
- [ ] No agent-to-agent direct communication (all via shared store)
- [ ] Task chunks sized for 30-min completion

**For Large Models (>13B):**
- [ ] Include full relevant context inline
- [ ] Agents can handle multi-step tasks without frequent context refreshes
- [ ] More autonomy per agent (less hand-holding)
- [ ] Can maintain conversation context across multiple handoffs
- [ ] Full project docs included when helpful

### Error Handling & Recovery

**Agent Failure:**
```python
def handle_agent_failure(swarm_id, failed_agent):
    # Read what was completed
    progress = read_progress(swarm_id)

    # Respawn with same context
    task = progress[failed_agent]["task"]
    context_summary = summarize_completed_work(progress)

    new_agent = spawn_agent(task, swarm_id, f"""
    PREVIOUS AGENT FAILED. Here is completed work so far:
    {context_summary}

    Your task: {task.description}
    Continue from where it left off.
    """)
```

**Context Corruption:**
- Always validate JSON before writing
- Use atomic writes (write to .tmp, then rename)
- Keep backup of progress.json before updates

**Swarm Timeout:**
```python
# If swarm times out, collect what we have
def emergency_collect(swarm_id):
    artifacts = glob(f"~/.hermes/swarm/{swarm_id}/memory/artifacts/*.json")
    progress = read_progress(swarm_id)

    return {
        "completed": [p for p in progress if p["status"] == "completed"],
        "in_progress": [p for p in progress if p["status"] == "in_progress"],
        "artifacts": artifacts,
        "reason": "timeout"
    }
```

### Swarm Cleanup

```bash
# After completion, archive swarm
SWARM_ID="myapp-20260406-120000"
tar -czf ~/hermes/swarm/archive/$SWARM_ID.tar.gz ~/.hermes/swarm/$SWARM_ID

# Or delete if not needed
rm -rf ~/.hermes/swarm/$SWARM_ID

# Keep logs for debugging
mv ~/.hermes/swarm/$SWARM_ID/logs ~/hermes/swarm/archive/${SWARM_ID}_logs
```

### When to Use Each Approach

| Scenario | Approach | Why |
|----------|----------|-----|
| Quick research task | `delegate_task` | No shared state needed, fire-and-forget |
| App with 2-3 components | Boss-agent via tmux | Clear separation, easy coordination |
| Large system (10+ components) | Hierarchical (orchestrator + middle managers) | Too many agents for single orchestrator |
| Data pipeline | Pipeline pattern | Strict sequential dependencies |
| Real-time collaboration | Peer network + events | Agents need to react to each other |
| Single developer, complex task | Boss-agent, small model optimized | Maximizes parallel work within context limits |

---

## Troubleshooting

### Voice not working
1. Check `stt.enabled: true` in config.yaml
2. Verify provider: `pip install faster-whisper` or set API key
3. Restart gateway: `/restart`

### Tool not available
1. `hermes tools` — check if toolset is enabled for your platform
2. Some tools need env vars (check `.env`)
3. `/reset` after enabling tools

### Model/provider issues
1. `hermes doctor` — check config and dependencies
2. `hermes login` — re-authenticate OAuth providers
3. Check `.env` has the right API key

### Changes not taking effect
- **Tools/skills:** `/reset` starts a new session with updated toolset
- **Config changes:** `/restart` reloads gateway config
- **Code changes:** Restart the CLI or gateway process

### Skills not showing
1. `hermes skills list` — verify installed
2. `hermes skills config` — check platform enablement
3. Load explicitly: `/skill name` or `hermes -s name`

### Gateway issues
Check logs first:
```bash
grep -i "failed to send\|error" ~/.hermes/logs/gateway.log | tail -20
```

---

## Where to Find Things

| Looking for... | Location |
|----------------|----------|
| Config options | `hermes config edit` or [Configuration docs](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) |
| Available tools | `hermes tools list` or [Tools reference](https://hermes-agent.nousresearch.com/docs/reference/tools-reference) |
| Slash commands | `/help` in session or [Slash commands reference](https://hermes-agent.nousresearch.com/docs/reference/slash-commands) |
| Skills catalog | `hermes skills browse` or [Skills catalog](https://hermes-agent.nousresearch.com/docs/reference/skills-catalog) |
| Provider setup | `hermes model` or [Providers guide](https://hermes-agent.nousresearch.com/docs/integrations/providers) |
| Platform setup | `hermes gateway setup` or [Messaging docs](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/) |
| MCP servers | `hermes mcp list` or [MCP guide](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) |
| Profiles | `hermes profile list` or [Profiles docs](https://hermes-agent.nousresearch.com/docs/user-guide/profiles) |
| Cron jobs | `hermes cron list` or [Cron docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) |
| Memory | `hermes memory status` or [Memory docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) |
| Env variables | `hermes config env-path` or [Env vars reference](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) |
| CLI commands | `hermes --help` or [CLI reference](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) |
| Gateway logs | `~/.hermes/logs/gateway.log` |
| Session files | `~/.hermes/sessions/` or `hermes sessions browse` |
| Source code | `~/.hermes/hermes-agent/` |

---

## Contributor Quick Reference

For occasional contributors and PR authors. Full developer docs: https://hermes-agent.nousresearch.com/docs/developer-guide/

### Project Layout

```
hermes-agent/
├── run_agent.py          # AIAgent — core conversation loop
├── model_tools.py        # Tool discovery and dispatch
├── toolsets.py           # Toolset definitions
├── cli.py                # Interactive CLI (HermesCLI)
├── hermes_state.py       # SQLite session store
├── agent/                # Prompt builder, compression, display, adapters
├── hermes_cli/           # CLI subcommands, config, setup, commands
│   ├── commands.py       # Slash command registry (CommandDef)
│   ├── config.py         # DEFAULT_CONFIG, env var definitions
│   └── main.py           # CLI entry point and argparse
├── tools/                # One file per tool
│   └── registry.py       # Central tool registry
├── gateway/              # Messaging gateway
│   └── platforms/        # Platform adapters (telegram, discord, etc.)
├── cron/                 # Job scheduler
├── tests/                # ~3000 pytest tests
└── website/              # Docusaurus docs site
```

Config: `~/.hermes/config.yaml` (settings), `~/.hermes/.env` (API keys).

### Adding a Tool (3 files)

**1. Create `tools/your_tool.py`:**
```python
import json, os
from tools.registry import registry

def check_requirements() -> bool:
    return bool(os.getenv("EXAMPLE_API_KEY"))

def example_tool(param: str, task_id: str = None) -> str:
    return json.dumps({"success": True, "data": "..."})

registry.register(
    name="example_tool",
    toolset="example",
    schema={"name": "example_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: example_tool(
        param=args.get("param", ""), task_id=kw.get("task_id")),
    check_fn=check_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

**2. Add import** in `model_tools.py` → `_discover_tools()` list.

**3. Add to `toolsets.py`** → `_HERMES_CORE_TOOLS` list.

All handlers must return JSON strings. Use `get_hermes_home()` for paths, never hardcode `~/.hermes`.

### Adding a Slash Command

1. Add `CommandDef` to `COMMAND_REGISTRY` in `hermes_cli/commands.py`
2. Add handler in `cli.py` → `process_command()`
3. (Optional) Add gateway handler in `gateway/run.py`

All consumers (help text, autocomplete, Telegram menu, Slack mapping) derive from the central registry automatically.

### Agent Loop (High Level)

```
run_conversation():
  1. Build system prompt
  2. Loop while iterations < max:
     a. Call LLM (OpenAI-format messages + tool schemas)
     b. If tool_calls → dispatch each via handle_function_call() → append results → continue
     c. If text response → return
  3. Context compression triggers automatically near token limit
```

### Testing

```bash
source venv/bin/activate  # or .venv/bin/activate
python -m pytest tests/ -o 'addopts=' -q   # Full suite
python -m pytest tests/tools/ -q            # Specific area
```

- Tests auto-redirect `HERMES_HOME` to temp dirs — never touch real `~/.hermes/`
- Run full suite before pushing any change
- Use `-o 'addopts='` to clear any baked-in pytest flags

### Commit Conventions

```
type: concise subject line

Optional body.
```

Types: `fix:`, `feat:`, `refactor:`, `docs:`, `chore:`

### Key Rules

- **Never break prompt caching** — don't change context, tools, or system prompt mid-conversation
- **Message role alternation** — never two assistant or two user messages in a row
- Use `get_hermes_home()` from `hermes_constants` for all paths (profile-safe)
- Config values go in `config.yaml`, secrets go in `.env`
- New tools need a `check_fn` so they only appear when requirements are met
