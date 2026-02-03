# Hermes Agent - Future Improvements

> Ideas for enhancing the agent's capabilities, generated from self-analysis of the codebase.

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

## 2. Planning & Task Management 📋

**Problem:** Agent handles tasks reactively without explicit planning. Complex multi-step tasks lack structure, progress tracking, and the ability to decompose work into manageable chunks.

**Ideas:**
- [ ] **Task decomposition tool** - Break complex requests into subtasks:
  ```
  User: "Set up a new Python project with FastAPI, tests, and Docker"
  
  Agent creates plan:
  ├── 1. Create project structure and requirements.txt
  ├── 2. Implement FastAPI app skeleton
  ├── 3. Add pytest configuration and initial tests
  ├── 4. Create Dockerfile and docker-compose.yml
  └── 5. Verify everything works together
  ```
  - Each subtask becomes a trackable unit
  - Agent can report progress: "Completed 3/5 tasks"
  
- [ ] **Progress checkpoints** - Periodic self-assessment:
  - After N tool calls or time elapsed, pause to evaluate
  - "What have I accomplished? What remains? Am I on track?"
  - Detect if stuck in loops or making no progress
  - Could trigger replanning if approach isn't working
  
- [ ] **Explicit plan storage** - Persist plan in conversation:
  - Store as structured data (not just in context)
  - Update status as tasks complete
  - User can ask "What's the plan?" or "What's left?"
  - Survives context compression (plans are protected)

- [ ] **Failure recovery with replanning** - When things go wrong:
  - Record what failed and why
  - Revise plan to work around the issue
  - "Step 3 failed because X, adjusting approach to Y"
  - Prevents repeating failed strategies

**Files to modify:** `run_agent.py` (add planning hooks), new `tools/planning_tool.py`

---

## 3. Dynamic Skills Expansion 📚

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

## 4. Interactive Clarifying Questions Tool ❓

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

## 5. Collaborative Problem Solving 🤝

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

## 6. Project-Local Context 💾

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

## 6. Tools & Skills Wishlist 🧰

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

## 7. Messaging Platform Integrations 💬 ✅ COMPLETE

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
- [x] **Telegram** - via `python-telegram-bot`
  - Bot token from @BotFather
  - Easiest to set up, good for personal use
- [x] **Discord** - via `discord.py`
  - Bot token from Discord Developer Portal
  - Can work in servers (group sessions) or DMs
- [x] **WhatsApp** - via Node.js bridge (whatsapp-web.js/baileys)
  - Requires Node.js bridge setup
  - More complex, but reaches most people

**Session management:**
- [x] **Session store** - JSONL persistence per session key
  - `~/.hermes/sessions/{session_id}.jsonl`
  - Session keys: `agent:main:telegram:dm`, `agent:main:discord:group:123`, etc.
- [x] **Session expiry** - Configurable reset policies
  - Daily reset (default 4am) OR idle timeout (default 2 hours)
  - Manual reset via `/reset` or `/new` command in chat
  - Per-platform and per-type overrides
- [x] **Session continuity** - Conversations persist across messages until reset

**Files created:** `gateway/`, `gateway/platforms/`, `gateway/config.py`, `gateway/session.py`, `gateway/delivery.py`, `gateway/run.py`

**Configuration:**
- Environment variables: `TELEGRAM_BOT_TOKEN`, `DISCORD_BOT_TOKEN`, etc.
- Config file: `~/.hermes/gateway.json`
- CLI commands: `/platforms` to check status, `--gateway` to start

**Dynamic context injection:**
- Agent knows its source platform and chat
- Agent knows connected platforms and home channels
- Agent can deliver cron outputs to specific platforms

---

## 8. Text-to-Speech (TTS) 🔊

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

## 13. Speech-to-Text / Audio Transcription 🎤

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

## Recently Completed ✅

### Dangerous Command Approval System
**Implemented:** Dangerous command detection and approval for terminal tool.

**Features:**
- Pattern-based detection of dangerous commands (rm -rf, DROP TABLE, chmod 777, etc.)
- CLI prompt with options: `[o]nce | [s]ession | [a]lways | [d]eny`
- Session caching (approved patterns don't re-prompt)
- Permanent allowlist in `~/.hermes/config.yaml`
- Force flag for agent to bypass after user confirmation
- Skip check for isolated backends (Docker, Singularity, Modal)
- Helpful sudo failure messages for messaging platforms

**Files:** `tools/terminal_tool.py`, `model_tools.py`, `hermes_cli/config.py`

---

## 14. Learning Machine / Dynamic Memory System 🧠

*Inspired by [Dash](~/agent-codebases/dash) - a self-learning data agent.*

**Problem:** Agent starts fresh every session. Valuable learnings from debugging, error patterns, successful approaches, and user preferences are lost.

**Dash's Key Insight:** Separate **Knowledge** (static, curated) from **Learnings** (dynamic, discovered):

| System | What It Stores | How It Evolves |
|--------|---------------|----------------|
| **Knowledge** (Skills) | Validated approaches, templates, best practices | Curated by user |
| **Learnings** | Error patterns, gotchas, discovered fixes | Managed automatically |

**Tools to implement:**
- [ ] `save_learning(topic, learning, context?)` - Record a discovered pattern
  ```python
  save_learning(
    topic="python-ssl",
    learning="On Ubuntu 22.04, SSL certificate errors often fixed by: apt install ca-certificates",
    context="Debugging requests SSL failure"
  )
  ```
- [ ] `search_learnings(query)` - Find relevant past learnings
  ```python
  search_learnings("SSL certificate error Python")
  # Returns: "On Ubuntu 22.04, SSL certificate errors often fixed by..."
  ```

**User Profile & Memory:**
- [ ] `user_profile` - Structured facts about user preferences
  ```yaml
  # ~/.hermes/user_profile.yaml
  coding_style:
    python_formatter: black
    type_hints: always
    test_framework: pytest
  preferences:
    verbosity: detailed
    confirm_destructive: true
  environment:
    os: linux
    shell: bash
    default_python: 3.11
  ```
- [ ] `user_memory` - Unstructured observations the agent learns
  ```yaml
  # ~/.hermes/user_memory.yaml
  - "User prefers tabs over spaces despite black's defaults"
  - "User's main project is ~/work/myapp - a Django app"
  - "User often works late - don't ask about timezone"
  ```

**When to learn:**
- After fixing an error that took multiple attempts
- When user corrects the agent's approach
- When a workaround is discovered for a tool limitation
- When user expresses a preference

**Storage:** Vector database (ChromaDB) or simple YAML with embedding search.

**Files to create:** `tools/learning_tools.py`, `learning/store.py`, `~/.hermes/learnings/`

---

## 15. Layered Context Architecture 📊

*Inspired by Dash's "Six Layers of Context" - grounding responses in multiple sources.*

**Problem:** Context sources are ad-hoc. No clear hierarchy or strategy for what context to include when.

**Proposed Layers for Hermes:**

| Layer | Source | When Loaded | Example |
|-------|--------|-------------|---------|
| 1. **Project Context** | `.hermes/context.md` | Auto on cwd | "This is a FastAPI project using PostgreSQL" |
| 2. **Skills** | `skills/*.md` | On request | "How to set up React project" |
| 3. **User Profile** | `~/.hermes/user_profile.yaml` | Always | "User prefers pytest, uses black" |
| 4. **Learnings** | `~/.hermes/learnings/` | Semantic search | "SSL fix for Ubuntu" |
| 5. **External Knowledge** | Web search, docs | On demand | Current API docs, Stack Overflow |
| 6. **Runtime Introspection** | Tool calls | Real-time | File contents, terminal output |

**Benefits:**
- Clear mental model for what context is available
- Prioritization: local > learned > external
- Debugging: "Why did agent do X?" → check which layers contributed

**Files to modify:** `run_agent.py` (context loading), new `context/layers.py`

---

## 16. Evaluation System with LLM Grading 📏

*Inspired by Dash's evaluation framework.*

**Problem:** `batch_runner.py` runs test cases but lacks quality assessment.

**Dash's Approach:**
- **String matching** (default) - Check if expected strings appear
- **LLM grader** (-g flag) - GPT evaluates response quality
- **Result comparison** (-r flag) - Compare against golden output

**Implementation for Hermes:**

- [ ] **Test case format:**
  ```python
  TestCase(
    name="create_python_project",
    prompt="Create a new Python project with FastAPI and tests",
    expected_strings=["requirements.txt", "main.py", "test_"],  # Basic check
    golden_actions=["write:main.py", "write:requirements.txt", "terminal:pip install"],
    grader_criteria="Should create complete project structure with working code"
  )
  ```

- [ ] **LLM grader mode:**
  ```python
  def grade_response(response: str, criteria: str) -> Grade:
      """Use GPT to evaluate response quality."""
      prompt = f"""
      Evaluate this agent response against the criteria.
      Criteria: {criteria}
      Response: {response}
      
      Score (1-5) and explain why.
      """
      # Returns: Grade(score=4, explanation="Created all files but tests are minimal")
  ```

- [ ] **Action comparison mode:**
  - Record tool calls made during test
  - Compare against expected actions
  - "Expected terminal call to pip install, got npm install"

- [ ] **CLI flags:**
  ```bash
  python batch_runner.py eval test_cases.yaml       # String matching
  python batch_runner.py eval test_cases.yaml -g    # + LLM grading
  python batch_runner.py eval test_cases.yaml -r    # + Result comparison
  python batch_runner.py eval test_cases.yaml -v    # Verbose (show responses)
  ```

**Files to modify:** `batch_runner.py`, new `evals/test_cases.py`, new `evals/grader.py`

---

*Last updated: $(date +%Y-%m-%d)* 🤖
