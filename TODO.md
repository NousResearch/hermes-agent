# Hermes Agent - Future Improvements

> Ideas for enhancing the agent's capabilities, generated from self-analysis of the codebase.

---

## 🚨 HIGH PRIORITY - Immediate Fixes

These items need to be addressed ASAP:

### 1. SUDO Breaking Terminal Tool 🔐 ✅ COMPLETE
- [x] **Problem:** SUDO commands break the terminal tool execution (hangs indefinitely)
- [x] **Fix:** Created custom environment wrappers in `tools/terminal_tool.py`
  - `stdin=subprocess.DEVNULL` prevents hanging on interactive prompts
  - Sudo fails gracefully with clear error if no password configured
  - Same UX as Claude Code - agent sees error, tells user to run it themselves
- [x] **All 5 environments now have consistent behavior:**
  - `_LocalEnvironment` - local execution
  - `_DockerEnvironment` - Docker containers
  - `_SingularityEnvironment` - Singularity/Apptainer containers
  - `_ModalEnvironment` - Modal cloud sandboxes
  - `_SSHEnvironment` - remote SSH execution
- [x] **Optional sudo support via `SUDO_PASSWORD` env var:**
  - Shared `_transform_sudo_command()` helper used by all environments
  - If set, auto-transforms `sudo cmd` → pipes password via `sudo -S`
  - Documented in `.env.example` with security warnings
  - Works for chained commands: `cmd1 && sudo cmd2`
- [ ] **Optional future enhancements:**
  - Interactive password prompt in CLI mode only
  - Document passwordless sudo setup in /etc/sudoers for power users

### 2. Fix `browser_get_images` Tool 🖼️
- [ ] **Problem:** `browser_get_images` tool is broken/not working correctly
- [ ] **Debug:** Investigate what's failing - selector issues? async timing? 
- [ ] **Fix:** Ensure it properly extracts image URLs and alt text from pages

### 3. Better Action Logging for Debugging 📝
- [ ] **Problem:** Need better logging of agent actions for debugging
- [ ] **Implementation:**
  - Log all tool calls with inputs/outputs
  - Timestamps for each action
  - Structured log format (JSON?) for easy parsing
  - Log levels (DEBUG, INFO, ERROR)
  - Option to write to file vs stdout

### 4. Stream Thinking Summaries in Real-Time 💭
- [ ] **Problem:** Thinking/reasoning summaries not shown while streaming
- [ ] **Implementation:**
  - Use streaming API to show thinking summaries as they're generated
  - Display intermediate reasoning before final response
  - Let user see the agent "thinking" in real-time

---

## 1. Subagent Architecture (Context Isolation) 🎯

**Problem:** Long-running tools (terminal commands, browser automation, complex file operations) consume massive context. A single `ls -la` can add hundreds of lines. Browser snapshots, debugging sessions, and iterative terminal work quickly bloat the main conversation, leaving less room for actual reasoning.

**Solution:** The main agent becomes an **orchestrator** that delegates context-heavy tasks to **subagents**.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (main agent)                                      │
│  - Receives user request                                        │
│  - Plans approach                                               │
│  - Delegates heavy tasks to subagents                           │
│  - Receives summarized results                                  │
│  - Maintains clean, focused context                             │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ TERMINAL AGENT  │  │ BROWSER AGENT   │  │ CODE AGENT      │
│ - terminal tool │  │ - browser tools │  │ - file tools    │
│ - file tools    │  │ - web_search    │  │ - terminal      │
│                 │  │ - web_extract   │  │                 │
│ Isolated context│  │ Isolated context│  │ Isolated context│
│ Returns summary │  │ Returns summary │  │ Returns summary │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**How it works:**
1. User asks: "Set up a new Python project with FastAPI and tests"
2. Orchestrator plans: "I need to create files, install deps, write code"
3. Orchestrator calls: `terminal_task(goal="Create venv, install fastapi pytest", context="New project in ~/myapp")`
4. **Subagent spawns** with fresh context, only terminal/file tools
5. Subagent iterates (may take 10+ tool calls, lots of output)
6. Subagent completes → returns summary: "Created venv, installed fastapi==0.109.0, pytest==8.0.0"
7. Orchestrator receives **only the summary**, context stays clean
8. Orchestrator continues with next subtask

**Key tools to implement:**
- [ ] `terminal_task(goal, context, cwd?)` - Delegate terminal/shell work
- [ ] `browser_task(goal, context, start_url?)` - Delegate web research/automation  
- [ ] `code_task(goal, context, files?)` - Delegate code writing/modification
- [ ] Generic `delegate_task(goal, context, toolsets=[])` - Flexible delegation

**Implementation details:**
- [ ] Subagent uses same `run_agent.py` but with:
  - Fresh/empty conversation history
  - Limited toolset (only what's needed)
  - Smaller max_iterations (focused task)
  - Task-specific system prompt
- [ ] Subagent returns structured result:
  ```python
  {
    "success": True,
    "summary": "Installed 3 packages, created 2 files",
    "details": "Optional longer explanation if needed",
    "artifacts": ["~/myapp/requirements.txt", "~/myapp/main.py"],  # Files created
    "errors": []  # Any issues encountered
  }
  ```
- [ ] Orchestrator sees only the summary in its context
- [ ] Full subagent transcript saved separately for debugging

**Benefits:**
- 🧹 **Clean context** - Orchestrator stays focused, doesn't drown in tool output
- 📊 **Better token efficiency** - 50 terminal outputs → 1 summary paragraph
- 🎯 **Focused subagents** - Each agent has just the tools it needs
- 🔄 **Parallel potential** - Independent subtasks could run concurrently
- 🐛 **Easier debugging** - Each subtask has its own isolated transcript

**When to use subagents vs direct tools:**
- **Subagent**: Multi-step tasks, iteration likely, lots of output expected
- **Direct**: Quick one-off commands, simple file reads, user needs to see output

**Files to modify:** `run_agent.py` (add orchestration mode), new `tools/delegate_tools.py`, new `subagent_runner.py`

---

## 2. Context Management (complements Subagents)

**Problem:** Context grows unbounded during long conversations. Trajectory compression exists for training data post-hoc, but live conversations lack intelligent context management.

**Ideas:**
- [ ] **Incremental summarization** - Compress old tool outputs on-the-fly during conversations
  - Trigger when context exceeds threshold (e.g., 80% of max tokens)
  - Preserve recent turns fully, summarize older tool responses
  - Could reuse logic from `trajectory_compressor.py`
  
- [ ] **Semantic memory retrieval** - Vector store for long conversation recall
  - Embed important facts/findings as conversation progresses
  - Retrieve relevant memories when needed instead of keeping everything in context
  - Consider lightweight solutions: ChromaDB, FAISS, or even a simple embedding cache
  
- [ ] **Working vs. episodic memory** distinction
  - Working memory: Current task state, recent tool results (always in context)
  - Episodic memory: Past findings, tried approaches (retrieved on demand)
  - Clear eviction policies for each

**Files to modify:** `run_agent.py` (add memory manager), possibly new `tools/memory_tool.py`

---

## 3. Self-Reflection & Course Correction 🔄

**Problem:** Current retry logic handles malformed outputs but not semantic failures. Agent doesn't reason about *why* something failed.

**Ideas:**
- [ ] **Meta-reasoning after failures** - When a tool returns an error or unexpected result:
  ```
  Tool failed → Reflect: "Why did this fail? What assumptions were wrong?"
  → Adjust approach → Retry with new strategy
  ```
  - Could be a lightweight LLM call or structured self-prompt
  
- [ ] **Planning/replanning module** - For complex multi-step tasks:
  - Generate plan before execution
  - After each step, evaluate: "Am I on track? Should I revise the plan?"
  - Store plan in working memory, update as needed
  
- [ ] **Approach memory** - Remember what didn't work:
  - "I tried X for this type of problem and it failed because Y"
  - Prevents repeating failed strategies in the same conversation

**Files to modify:** `run_agent.py` (add reflection hooks in tool loop), new `tools/reflection_tool.py`

---

## 4. Tool Composition & Learning 🔧

**Problem:** Tools are atomic. Complex tasks require repeated manual orchestration of the same tool sequences.

**Ideas:**
- [ ] **Macro tools / Tool chains** - Define reusable tool sequences:
  ```yaml
  research_topic:
    description: "Deep research on a topic"
    steps:
      - web_search: {query: "$topic"}
      - web_extract: {urls: "$search_results.urls[:3]"}
      - summarize: {content: "$extracted"}
  ```
  - Could be defined in skills or a new `macros/` directory
  - Agent can invoke macro as single tool call
  
- [ ] **Tool failure patterns** - Learn from failures:
  - Track: tool, input pattern, error type, what worked instead
  - Before calling a tool, check: "Has this pattern failed before?"
  - Persistent across sessions (stored in skills or separate DB)
  
- [ ] **Parallel tool execution** - When tools are independent, run concurrently:
  - Detect independence (no data dependencies between calls)
  - Use `asyncio.gather()` for parallel execution
  - Already have async support in some tools, just need orchestration

**Files to modify:** `model_tools.py`, `toolsets.py`, new `tool_macros.py`

---

## 5. Dynamic Skills Expansion 📚

**Problem:** Skills system is elegant but static. Skills must be manually created and added.

**Ideas:**
- [ ] **Skill acquisition from successful tasks** - After completing a complex task:
  - "This approach worked well. Save as a skill?"
  - Extract: goal, steps taken, tools used, key decisions
  - Generate SKILL.md automatically
  - Store in user's skills directory
  
- [ ] **Skill templates** - Common patterns that can be parameterized:
  ```markdown
  # Debug {language} Error
  1. Reproduce the error
  2. Search for error message: `web_search("{error_message} {language}")`
  3. Check common causes: {common_causes}
  4. Apply fix and verify
  ```
  
- [ ] **Skill chaining** - Combine skills for complex workflows:
  - Skills can reference other skills as dependencies
  - "To do X, first apply skill Y, then skill Z"
  - Directed graph of skill dependencies

**Files to modify:** `tools/skills_tool.py`, `skills/` directory structure, new `skill_generator.py`

---

## 6. Task Continuation Hints 🎯

**Problem:** Could be more helpful by suggesting logical next steps.

**Ideas:**
- [ ] **Suggest next steps** - At end of a task, suggest logical continuations:
  - "Code is written. Want me to also write tests / docs / deploy?"
  - Based on common workflows for task type
  - Non-intrusive, just offer options

**Files to modify:** `run_agent.py`, response generation logic

---

## 7. Interactive Clarifying Questions Tool ❓

**Problem:** Agent sometimes makes assumptions or guesses when it should ask the user. Currently can only ask via text, which gets lost in long outputs.

**Ideas:**
- [ ] **Multiple-choice prompt tool** - Let agent present structured choices to user:
  ```
  ask_user_choice(
    question="Should the language switcher enable only German or all languages?",
    choices=[
      "Only enable German - works immediately",
      "Enable all, mark untranslated - show fallback notice",
      "Let me specify something else"
    ]
  )
  ```
  - Renders as interactive terminal UI with arrow key / Tab navigation
  - User selects option, result returned to agent
  - Up to 4 choices + optional free-text option
  
- [ ] **Implementation:**
  - Use `inquirer` or `questionary` Python library for rich terminal prompts
  - Tool returns selected option text (or user's custom input)
  - **CLI-only** - only works when running via `cli.py` (not API/programmatic use)
  - Graceful fallback: if not in interactive mode, return error asking agent to rephrase as text
  
- [ ] **Use cases:**
  - Clarify ambiguous requirements before starting work
  - Confirm destructive operations with clear options
  - Let user choose between implementation approaches
  - Checkpoint complex multi-step workflows

**Files to modify:** New `tools/ask_user_tool.py`, `cli.py` (detect interactive mode), `model_tools.py`

---

## 8. Resource Awareness & Efficiency 💰

**Problem:** No awareness of costs, time, or resource usage. Could be smarter about efficiency.

**Ideas:**
- [ ] **Tool result caching** - Don't repeat identical operations:
  - Cache web searches, extractions within a session
  - Invalidation based on time-sensitivity of query
  - Hash-based lookup: same input → cached output

- [ ] **Lazy evaluation** - Don't fetch everything upfront:
  - Get summaries first, full content only if needed
  - "I found 5 relevant pages. Want me to deep-dive on any?"

**Files to modify:** `model_tools.py`, new `resource_tracker.py`

---

## 9. Collaborative Problem Solving 🤝

**Problem:** Interaction is command/response. Complex problems benefit from dialogue.

**Ideas:**
- [ ] **Assumption surfacing** - Make implicit assumptions explicit:
  - "I'm assuming you want Python 3.11+. Correct?"
  - "This solution assumes you have sudo access..."
  - Let user correct before going down wrong path

- [ ] **Checkpoint & confirm** - For high-stakes operations:
  - "About to delete 47 files. Here's the list - proceed?"
  - "This will modify your database. Want a backup first?"
  - Configurable threshold for when to ask

**Files to modify:** `run_agent.py`, system prompt configuration

---

## 10. Project-Local Context 💾

**Problem:** Valuable context lost between sessions.

**Ideas:**
- [ ] **Project awareness** - Remember project-specific context:
  - Store `.hermes/context.md` in project directory
  - "This is a Django project using PostgreSQL"
  - Coding style preferences, deployment setup, etc.
  - Load automatically when working in that directory

- [ ] **Handoff notes** - Leave notes for future sessions:
  - Write to `.hermes/notes.md` in project
  - "TODO for next session: finish implementing X"
  - "Known issues: Y doesn't work on Windows"

**Files to modify:** New `project_context.py`, auto-load in `run_agent.py`

---

## 11. Graceful Degradation & Robustness 🛡️

**Problem:** When things go wrong, recovery is limited. Should fail gracefully.

**Ideas:**
- [ ] **Fallback chains** - When primary approach fails, have backups:
  - `web_extract` fails → try `browser_navigate` → try `web_search` for cached version
  - Define fallback order per tool type
  
- [ ] **Partial progress preservation** - Don't lose work on failure:
  - Long task fails midway → save what we've got
  - "I completed 3/5 steps before the error. Here's what I have..."
  
- [ ] **Self-healing** - Detect and recover from bad states:
  - Browser stuck → close and retry
  - Terminal hung → timeout and reset

**Files to modify:** `model_tools.py`, tool implementations, new `fallback_manager.py`

---

## 12. Tools & Skills Wishlist 🧰

*Things that would need new tool implementations (can't do well with current tools):*

### High-Impact

- [ ] **Audio/Video Transcription** 🎬 *(See also: Section 16 for detailed spec)*
  - Transcribe audio files, podcasts, YouTube videos
  - Extract key moments from video
  - Voice memo transcription for messaging integrations
  - *Provider options: Whisper API, Deepgram, local Whisper*
  
- [ ] **Diagram Rendering** 📊
  - Render Mermaid/PlantUML to actual images
  - Can generate the code, but rendering requires external service or tool
  - "Show me how these components connect" → actual visual diagram

### Medium-Impact

- [ ] **Canvas / Visual Workspace** 🖼️
  - Agent-controlled visual panel for rendering interactive UI
  - Inspired by OpenClaw's Canvas feature
  - **Capabilities:**
    - `present` / `hide` - Show/hide the canvas panel
    - `navigate` - Load HTML files or URLs into the canvas
    - `eval` - Execute JavaScript in the canvas context
    - `snapshot` - Capture the rendered UI as an image
  - **Use cases:**
    - Display generated HTML/CSS/JS previews
    - Show interactive data visualizations (charts, graphs)
    - Render diagrams (Mermaid → rendered output)
    - Present structured information in rich format
    - A2UI-style component system for structured agent UI
  - **Implementation options:**
    - Electron-based panel for CLI
    - WebSocket-connected web app
    - VS Code webview extension
  - *Would let agent "show" things rather than just describe them*

- [ ] **Document Generation** 📄
  - Create styled PDFs, Word docs, presentations
  - *Can do basic PDF via terminal tools, but limited*

- [ ] **Diff/Patch Tool** 📝
  - Surgical code modifications with preview
  - "Change line 45-50 to X" without rewriting whole file
  - Show diffs before applying
  - *Can use `diff`/`patch` but a native tool would be safer*

### Skills to Create

- [ ] **Domain-specific skill packs:**
  - DevOps/Infrastructure (Terraform, K8s, AWS)
  - Data Science workflows (EDA, model training)
  - Security/pentesting procedures
  
- [ ] **Framework-specific skills:**
  - React/Vue/Angular patterns
  - Django/Rails/Express conventions
  - Database optimization playbooks

- [ ] **Troubleshooting flowcharts:**
  - "Docker container won't start" → decision tree
  - "Production is slow" → systematic diagnosis

---

## 13. Messaging Platform Integrations 💬

**Problem:** Agent currently only works via `cli.py` which requires direct terminal access. Users may want to interact via messaging apps from their phone or other devices.

**Architecture:**
- `run_agent.py` already accepts `conversation_history` parameter and returns updated messages ✅
- Need: persistent session storage, platform monitors, session key resolution

**Implementation approach:**
```
┌─────────────────────────────────────────────────────────────┐
│  Platform Monitor (e.g., telegram_monitor.py)               │
│  ├─ Long-running daemon connecting to messaging platform    │
│  ├─ On message: resolve session key → load history from disk│
│  ├─ Call run_agent.py with loaded history                   │
│  ├─ Save updated history back to disk (JSONL)               │
│  └─ Send response back to platform                          │
└─────────────────────────────────────────────────────────────┘
```

**Platform support (each user sets up their own credentials):**
- [ ] **Telegram** - via `python-telegram-bot` or `grammy` equivalent
  - Bot token from @BotFather
  - Easiest to set up, good for personal use
- [ ] **Discord** - via `discord.py`
  - Bot token from Discord Developer Portal
  - Can work in servers (group sessions) or DMs
- [ ] **WhatsApp** - via `baileys` (WhatsApp Web protocol)
  - QR code scan to authenticate
  - More complex, but reaches most people

**Session management:**
- [ ] **Session store** - JSONL persistence per session key
  - `~/.hermes/sessions/{session_key}.jsonl`
  - Session keys: `telegram:dm:{user_id}`, `discord:channel:{id}`, etc.
- [ ] **Session expiry** - Configurable reset policies
  - Daily reset (default 4am) OR idle timeout (e.g., 2 hours)
  - Manual reset via `/reset` or `/new` command in chat
- [ ] **Session continuity** - Conversations persist across messages until reset

**Files to create:** `monitors/telegram_monitor.py`, `monitors/discord_monitor.py`, `monitors/session_store.py`

---

## 14. Scheduled Tasks / Cron Jobs ⏰

**Problem:** Agent only runs on-demand. Some tasks benefit from scheduled execution (daily summaries, monitoring, reminders).

**Ideas:**
- [ ] **Cron-style scheduler** - Run agent turns on a schedule
  - Store jobs in `~/.hermes/cron/jobs.json`
  - Each job: `{ id, schedule, prompt, session_mode, delivery }`
  - Uses APScheduler or similar Python library
  
- [ ] **Session modes:**
  - `isolated` - Fresh session each run (no history, clean context)
  - `main` - Append to main session (agent remembers previous scheduled runs)
  
- [ ] **Delivery options:**
  - Write output to file (`~/.hermes/cron/output/{job_id}/{timestamp}.md`)
  - Send to messaging channel (if integrations enabled)
  - Both
  
- [ ] **CLI interface:**
  ```bash
  # List scheduled jobs
  python cli.py --cron list
  
  # Add a job (runs daily at 9am)
  python cli.py --cron add "Summarize my email inbox" --schedule "0 9 * * *"
  
  # Quick syntax for simple intervals  
  python cli.py --cron add "Check server status" --every 30m
  
  # Remove a job
  python cli.py --cron remove <job_id>
  ```

- [ ] **Agent self-scheduling** - Let the agent create its own cron jobs
  - New tool: `schedule_task(prompt, schedule, session_mode)`
  - "Remind me to check the deployment tomorrow at 9am"
  - Agent can set follow-up tasks for itself

- [ ] **In-chat command:** `/cronjob {prompt} {frequency}` when using messaging integrations

**Files to create:** `cron/scheduler.py`, `cron/jobs.py`, `tools/schedule_tool.py`

---

## 15. Text-to-Speech (TTS) 🔊

**Problem:** Agent can only respond with text. Some users prefer audio responses (accessibility, hands-free use, podcasts).

**Ideas:**
- [ ] **TTS tool** - Generate audio files from text
  ```python
  tts_generate(text="Here's your summary...", voice="nova", output="summary.mp3")
  ```
  - Returns path to generated audio file
  - For messaging integrations: can send as voice message
  
- [ ] **Provider options:**
  - Edge TTS (free, good quality, many voices)
  - OpenAI TTS (paid, excellent quality)
  - ElevenLabs (paid, best quality, voice cloning)
  - Local options (Coqui TTS, Bark)
  
- [ ] **Modes:**
  - On-demand: User explicitly asks "read this to me"
  - Auto-TTS: Configurable to always generate audio for responses
  - Long-text handling: Summarize or chunk very long responses
  
- [ ] **Integration with messaging:**
  - When enabled, can send voice notes instead of/alongside text
  - User preference per channel

**Files to create:** `tools/tts_tool.py`, config in `cli-config.yaml`

---

## 16. Speech-to-Text / Audio Transcription 🎤

**Problem:** Users may want to send voice memos instead of typing. Agent is blind to audio content.

**Ideas:**
- [ ] **Voice memo transcription** - For messaging integrations
  - User sends voice message → transcribe → process as text
  - Seamless: user speaks, agent responds
  
- [ ] **Audio/video file transcription** - Existing idea, expanded:
  - Transcribe local audio files (mp3, wav, m4a)
  - Transcribe YouTube videos (download audio → transcribe)
  - Extract key moments with timestamps
  
- [ ] **Provider options:**
  - OpenAI Whisper API (good quality, cheap)
  - Deepgram (fast, good for real-time)
  - Local Whisper (free, runs on GPU)
  - Groq Whisper (fast, free tier available)
  
- [ ] **Tool interface:**
  ```python
  transcribe(source="audio.mp3")  # Local file
  transcribe(source="https://youtube.com/...")  # YouTube
  transcribe(source="voice_message", data=bytes)  # Voice memo
  ```

**Files to create:** `tools/transcribe_tool.py`, integrate with messaging monitors

---

## Priority Order (Suggested)

1. **🎯 Subagent Architecture** - Critical for context management, enables everything else
2. **Memory & Context Management** - Complements subagents for remaining context
3. **Self-Reflection** - Improves reliability and reduces wasted tool calls  
4. **Project-Local Context** - Practical win, keeps useful info across sessions
5. **Messaging Integrations** - Unlocks mobile access, new interaction patterns
6. **Scheduled Tasks / Cron Jobs** - Enables automation, reminders, monitoring
7. **Tool Composition** - Quality of life, builds on other improvements
8. **Dynamic Skills** - Force multiplier for repeated tasks
9. **Interactive Clarifying Questions** - Better UX for ambiguous tasks
10. **TTS / Audio Transcription** - Accessibility, hands-free use

---

## Removed Items (Unrealistic)

The following were removed because they're architecturally impossible:

- ~~Proactive suggestions / Prefetching~~ - Agent only runs on user request, can't interject
- ~~Clipboard integration~~ - No access to user's local system clipboard

The following **moved to active TODO** (now possible with new architecture):

- ~~Session save/restore~~ → See **Messaging Integrations** (session persistence)
- ~~Voice/TTS playback~~ → See **TTS** (can generate audio files, send via messaging)
- ~~Set reminders~~ → See **Scheduled Tasks / Cron Jobs**

The following were removed because they're **already possible**:

- ~~HTTP/API Client~~ → Use `curl` or Python `requests` in terminal
- ~~Structured Data Manipulation~~ → Use `pandas` in terminal
- ~~Git-Native Operations~~ → Use `git` CLI in terminal
- ~~Symbolic Math~~ → Use `SymPy` in terminal
- ~~Code Quality Tools~~ → Run linters (`eslint`, `black`, `mypy`) in terminal
- ~~Testing Framework~~ → Run `pytest`, `jest`, etc. in terminal
- ~~Translation~~ → LLM handles this fine, or use translation APIs

---

---

## 🧪 Brainstorm Ideas (Not Yet Fleshed Out)

*These are early-stage ideas that need more thinking before implementation. Captured here so they don't get lost.*

### Remote/Distributed Execution 🌐

**Concept:** Run agent on a powerful remote server while interacting from a thin client.

**Why interesting:**
- Run on beefy GPU server for local LLM inference
- Agent has access to remote machine's resources (files, tools, internet)
- User interacts via lightweight client (phone, low-power laptop)

**Open questions:**
- How does this differ from just SSH + running cli.py on remote?
- Would need secure communication channel (WebSocket? gRPC?)
- How to handle tool outputs that reference remote paths?
- Credential management for remote execution
- Latency considerations for interactive use

**Possible architecture:**
```
┌─────────────┐         ┌─────────────────────────┐
│ Thin Client │ ◄─────► │ Remote Hermes Server    │
│ (phone/web) │  WS/API │ - Full agent + tools    │
└─────────────┘         │ - GPU for local LLM     │
                        │ - Access to server files│
                        └─────────────────────────┘
```

**Related to:** Messaging integrations (could be the "server" that monitors receive from)

---

### Multi-Agent Parallel Execution 🤖🤖

**Concept:** Extension of Subagent Architecture (Section 1) - run multiple subagents in parallel.

**Why interesting:**
- Independent subtasks don't need to wait for each other
- "Research X while setting up Y" - both run simultaneously
- Faster completion for complex multi-part tasks

**Open questions:**
- How to detect which tasks are truly independent?
- Resource management (API rate limits, concurrent connections)
- How to merge results when parallel tasks have conflicts?
- Cost implications of multiple parallel LLM calls

*Note: Basic subagent delegation (Section 1) should be implemented first, parallel execution is an optimization on top.*

---

### Plugin/Extension System 🔌

**Concept:** Allow users to add custom tools/skills without modifying core code.

**Why interesting:**
- Community contributions
- Organization-specific tools
- Clean separation of core vs. extensions

**Open questions:**
- Security implications of loading arbitrary code
- Versioning and compatibility
- Discovery and installation UX

---

*Last updated: $(date +%Y-%m-%d)* 🤖
