# Hermes Agent - Future Improvements

---

## What We Already Have (for reference)

**43+ tools** across 13 toolsets: web (search, extract), terminal + process management, file ops (read, write, patch, search), vision, MoA reasoning, image gen, browser (10 tools via Browserbase), skills (41 skills), **todo (task planning)**, cronjobs, RL training (10 tools via Tinker-Atropos), TTS, cross-channel messaging.

**Skills Hub**: search/install/inspect/audit/uninstall/publish/snapshot across 4 registries (GitHub, ClawHub, Claude Code marketplaces, LobeHub). Security scanner with trust-aware policy. CLI (`hermes skills ...`) and `/skills` slash command. agentskills.io spec compliant.

**4 platform adapters**: Telegram, Discord, WhatsApp, Slack -- all with typing indicators, image/voice auto-analysis, dangerous command approval, interrupt support, background process watchers.

**Other**: Context compression, context files (SOUL.md, AGENTS.md), session JSONL transcripts, batch runner with toolset distributions, 13 personalities, DM pairing auth, PTY mode, model metadata caching.

---

## The Knowledge System (how Memory, Skills, Sessions, and Subagents interconnect)

These four systems form a continuum of agent intelligence. They should be thought of together:

**Types of agent knowledge:**
- **Procedural memory (Skills)** -- reusable approaches for specific task types. "How to deploy a Docker container." "How to fine-tune with Axolotl." Created when the agent works through something difficult and succeeds.
- **Declarative memory (MEMORY.md)** -- facts about the environment, projects, tools, conventions. "This repo uses Poetry, not pip." "The API key is stored in ~/.config/keys."
- **Identity memory (USER.md / memory_summary.md)** -- who the user is, how they like to work, their preferences, communication style. Persists across all sessions.
- **Error memory (Learnings)** -- what went wrong and the proven fix. "pip install fails on this system because of X; use conda instead."
- **Episodic memory (Session summaries)** -- what happened in past sessions. Searchable for when the agent needs to recall prior conversations.

**The feedback loop:** After complex tasks, especially ones involving difficulty or iteration, the agent should:
1. Ask the user for feedback: "How was that? Did it work out?"
2. If successful, offer to save: "Would you like me to save that as a skill for next time?"
3. Update general memory with any durable insights (user preferences, environment facts, lessons learned)

**Storage evolution:** Start with flat files (Phase 1), migrate to SQLite (Phase 2) when the volume of sessions and memories makes file-based storage unwieldy.

---

## 1. Subagent Architecture (Context Isolation) 🎯

**Status:** Not started
**Priority:** High -- this is foundational for scaling to complex tasks

The main agent becomes an orchestrator that delegates context-heavy tasks to subagents with isolated context. Each subagent returns a summary, keeping the orchestrator's context clean.

**Core design:**
- `delegate_task(goal, context, toolsets=[])` tool -- spawns a fresh AIAgent with its own conversation, limited toolset, and task-specific system prompt
- Parent passes a goal string + optional context blob; child returns a structured summary
- Configurable depth limit (e.g., max 2 levels) to prevent runaway recursion
- Subagent inherits the same terminal/browser session (task_id) but gets a fresh message history

**What other agents do:**
- **OpenClaw**: `sessions_spawn` + `subagents` tool with list/kill/steer actions, depth limits, rate limiting. Cross-session agent-to-agent coordination via `sessions_send`.
- **Codex**: `spawn_agent` / `send_input` / `close_agent` / `wait_for_agent` with configurable timeouts. Thread manager for concurrent agents. Also uses subagents for memory consolidation (Phase 2 spawns a dedicated consolidation agent).
- **Cline**: Up to 5 parallel subagents per invocation. Subagents get restricted tool access (read, list, search, bash, skill, attempt). Progress tracking with stats (tool calls, tokens, cost, context usage).
- **OpenCode**: `TaskTool` creates subagent sessions with permission inheritance. Resumable tasks via `task_id`. Parent-child session relationships.

**Our approach:**
- Start with a single `delegate_task` tool (like Cline's model -- simple, bounded)
- Subagent gets: goal, context excerpt, restricted toolset, fresh conversation
- Returns: summary string (success/failure + key findings + any file paths created)
- Track active subagents so parent can reference them; limit concurrency to 3
- Primary use cases: parallelizing distinct work (research two topics, work on two separate code changes), handling context-heavy tasks that would bloat the parent's context
- Later: add `send_input` for interactive subagent steering (Codex-style)
- Later: cross-session coordination for gateway (OpenClaw-style `sessions_send`)

---

## 2. Agent-Managed Skills (Create / Edit / Delete) 📚

**Status:** Not started
**Priority:** High -- skills are the agent's procedural memory

The Skills Hub (search/install/publish from 4 registries) is done. What's missing is the agent's ability to **create, edit, and delete its own skills** -- turning successful approaches into reusable task-specific knowledge.

**Skills are a form of memory.** General memory (user profile, environment facts, preferences) is broad and declarative. Skills are narrow and procedural -- they capture *how to do a specific type of task* based on proven experience. Together they form the agent's knowledge system.

**What other agents do:**
- **Cline**: `new_rule` tool for creating rules from context. Global + project-level skill directories.
- **OpenClaw**: Workspace skills that agents can write into during sessions.
- **Codex**: Phase 2 consolidation agent automatically creates skills from recurring patterns it detects across rollout summaries. Skills include: triggers, inputs, procedure steps, efficiency plans, pitfalls, verification checklists, and optional scripts/templates/examples.

### Architecture

**New `skill_manage` tool** (separate from read-only `skill_view`):
- Actions: `create`, `edit`, `delete`
- User skills stored in `~/.hermes/skills/<name>/SKILL.md` (separate from bundled `skills/` in repo)
- Same SKILL.md format (YAML frontmatter + markdown body), validated on write
- Agent sees `source` field (`"bundled"`, `"hub"`, `"user"`) on every skill -- can only edit/delete `"user"` skills

**Discovery merge**: `_find_all_skills()` and `skill_view()` search both `SKILLS_DIR` (bundled) and `USER_SKILLS_DIR` (`~/.hermes/skills/`).

### Proactive skill creation (the feedback loop)

The agent shouldn't just create skills when asked. It should **recognize when a skill is worth creating** and offer proactively. The behavior is taught via the tool description (cache-friendly, same pattern as todo tool):

**When to trigger:**
- The task involved 5+ tool calls with back-and-forth or iteration
- The agent hit errors/obstacles and recovered successfully
- The user corrected the agent's approach and the corrected version worked
- The task type is likely to recur (deployment, data processing, config setup, etc.)

**The interaction pattern:**
1. After completing a difficult task successfully, the agent asks: *"That took some figuring out. How did the result turn out for you?"*
2. If the user confirms success, the agent offers: *"Would you like me to save that approach as a skill so I can do it faster next time?"*
3. If yes, the agent creates a skill capturing: the trigger conditions, the working procedure, the pitfalls encountered, and the verification steps

This pattern doesn't require the `clarify` tool (#3) -- it works as normal conversational text. But `clarify` would make it cleaner on messaging platforms with structured choices.

### Implementation steps

**Step 1: Update discovery** (`tools/skills_tool.py`)
- Add `USER_SKILLS_DIR = Path.home() / ".hermes" / "skills"`
- Update `_find_all_skills()` to scan both dirs, tag each skill with `source`
- Update `skill_view()` to search `USER_SKILLS_DIR` as fallback

**Step 2: Validation helper** (`tools/skills_tool.py`)
- `_validate_skill_frontmatter()` -- enforce `name` (required, ≤64 chars, filesystem-safe), `description` (required, ≤1024 chars), valid YAML, non-empty body

**Step 3: `skill_manage()` function** (`tools/skills_tool.py`)
```
skill_manage(action, name, description=None, content=None, tags=None)
```
- `create`: write `~/.hermes/skills/<name>/SKILL.md`, fail if name collision with any source
- `edit`: read existing user skill, merge updates, write back (refuse bundled/hub skills)
- `delete`: remove user skill directory (refuse bundled/hub skills)

**Step 4: Register tool** (`model_tools.py`)
- Tool definition with description teaching when/how to create skills AND the proactive feedback loop behavior
- Route in `handle_skills_function_call()`
- Add to toolset mappings and `TOOLSET_REQUIREMENTS`

**Step 5: CLI commands** (`hermes_cli/skills_hub.py` + `hermes_cli/main.py`)
- `hermes skills create <name>` -- interactive or `--from-file`
- `hermes skills edit <name>` -- opens `$EDITOR` or accepts flags
- `hermes skills delete <name>` -- with confirmation
- Update `hermes skills list` to show `[user]`/`[builtin]`/`[hub]` tags

**Step 6: Slash command** -- extend `/skills` handler with `create`/`edit`/`delete`

### Tool description (teaches the LLM when and how to use skills)
```
Create, edit, or delete user-managed skills. Skills are your procedural memory --
they capture proven approaches for specific task types so you can do them faster
and better next time.

Actions:
- create: Save a new skill to ~/.hermes/skills/. Provide name, description, content.
- edit: Update an existing user skill. Only works on source="user" skills.
- delete: Remove a user skill. Only works on source="user" skills.

═══ WHEN TO CREATE A SKILL ═══

Create a skill when ALL of these are true:
1. The task type is likely to recur (not a one-off)
2. The approach was non-obvious or required iteration to get right
3. A future attempt would benefit from having the steps written down

Common triggers:
- You completed a complex task (5+ tool calls) and it succeeded
- You hit errors or obstacles during the task and found the fix
- The user corrected your approach and the corrected version worked
- You discovered a non-trivial workflow (deployment, data pipeline, config, etc.)
- The user explicitly asks you to remember how to do something

═══ THE FEEDBACK LOOP ═══

After completing a task that was difficult or iterative (errors encountered,
multiple attempts, user corrections, or 5+ tool calls with back-and-forth):

1. Ask the user for feedback: "That took some working through. How did the
   result turn out?"
2. If they confirm success: "Would you like me to save that approach as a
   skill so I can do it faster next time?"
3. If yes: create a skill with the working procedure, including:
   - When to use this skill (trigger conditions)
   - The steps that worked (numbered procedure)
   - Pitfalls encountered and how to avoid them
   - How to verify success

Do NOT trigger this feedback loop for:
- Simple tasks (single tool call, obvious approach)
- Tasks where the user seems impatient or in a hurry
- Tasks that are clearly one-off (e.g., "what time is it in Tokyo")

═══ SKILL QUALITY ═══

A good skill is specific and actionable, not generic advice. It should contain:
- Concrete trigger conditions (when does this skill apply?)
- Exact commands, file paths, or API calls that worked
- Known failure modes and their fixes
- Verification steps (how to confirm it worked)

Always confirm with the user before creating a skill.
```

### Later
- Skill chaining with dependency graphs, parameterized templates
- Publishing user skills to remote registries (already have `hermes skills publish`)
- Periodic skill review: subagent scans session summaries for recurring patterns that should become skills (Codex-style Phase 2 consolidation)

**Files:** `tools/skills_tool.py` (core logic), `model_tools.py` (registration), `hermes_cli/skills_hub.py` + `hermes_cli/main.py` (CLI)

---

## 3. Interactive Clarifying Questions ❓

**Status:** Not started
**Priority:** Medium-High -- enables the knowledge system feedback loop

Allow the agent to present structured choices to the user when it needs clarification or feedback. Rich terminal UI in CLI mode, graceful fallback on messaging platforms.

**What other agents do:**
- **Codex**: `request_user_input` tool for open-ended questions
- **Cline**: `ask_followup_question` tool with structured options
- **OpenCode**: `question` tool for asking the user

**Our approach:**
- `clarify` tool with parameters: `question` (string), `choices` (list of up to 6 strings), `allow_freetext` (bool)
- CLI mode: Rich-powered selection UI (arrow keys + number shortcuts)
- Gateway/messaging mode: numbered list with "reply with number or type your answer"
- Returns the user's selection as a string

**Use cases (beyond simple clarification):**
- Before starting expensive operations: "Which approach do you prefer?"
- **Post-task feedback**: "How did that work out?" with choices like [Worked perfectly / Mostly good / Had issues / Didn't work]
- **Skill creation offer**: "Want me to save that approach as a skill?" with [Yes / Yes, but let me review it first / No]
- **Memory update prompt**: "I noticed you prefer X. Should I remember that for future sessions?" with [Yes / No / It depends]

This tool is lightweight on its own but becomes critical for the proactive feedback loop in the knowledge system (skills, memory, learnings).

**File:** `tools/clarify_tool.py` -- presentation layer differs per platform, core logic is simple

---

## 4. Memory System 🧠

**Status:** Not started
**Priority:** High -- biggest gap vs. OpenClaw, Dash, and Codex

Persistent memory that survives across sessions. The agent remembers what it learned, who the user is, and what worked before. Memory is the general/declarative counterpart to skills (procedural memory) -- together they form the agent's knowledge system.

**What other agents do:**
- **OpenClaw**: SQLite + sqlite-vec for vector search. LRU eviction on embedding cache. Temporal decay on search results (half-life 30 days). Pre-compaction memory flush (model writes durable notes before context eviction).
- **Dash**: 6-layer context system. LearningMachine with agentic mode -- errors get diagnosed, fixed, and saved as learnings that are never repeated.
- **Codex**: 2-phase pipeline. Phase 1: dedicated model extracts raw memories from past sessions. Phase 2: consolidation subagent produces `memory_summary.md` (always in prompt), `MEMORY.md` (searchable), `skills/`. Retention cap: keep N most recent, drop rest.

### Our architecture: bounded, curated, always-visible

Two small files, both injected into the system prompt every session. The agent always sees its full memory, so it can make informed decisions about what to update or consolidate without extra reads.

**`~/.hermes/memories/MEMORY.md`** -- agent's personal notes and observations (2,200 character limit, ~800 tokens)
- Environment facts, project conventions, tool quirks, things that took effort to figure out
- One entry per line: `conda preferred over pip on this machine`

**`~/.hermes/memories/USER.md`** -- what the agent knows about the user (1,375 character limit, ~500 tokens)
- Preferences, communication style, expectations, workflow habits, corrections
- One entry per line: `Prefers plans before implementation`

Character limits (not tokens) because character counts are model-independent -- users can switch models without breaking budgets. Conversion: ~2.75 characters per token. Dates are stored internally for eviction tiebreaking but not shown in the system prompt.

### System prompt injection

Both files are injected into the system prompt with clear separators:

```
══════════════════════════════════════════════
MEMORY (your personal notes) [68% — 1,496/2,200 chars]
══════════════════════════════════════════════
User's name is Teknium, founder of Nous Research
§
This machine runs Ubuntu, conda preferred over pip
§
When working on Hermes-Agent, always test with run_agent.py before gateway
§
User strongly prefers batch approach for RL training over online

══════════════════════════════════════════════
USER PROFILE (who the user is) [72% — 990/1,375 chars]
══════════════════════════════════════════════
Prefers detailed technical discussions, no hand-holding
§
Communication style: direct, concise, expects expertise
§
Likes to discuss plans before implementation, wants to approve approach first
§
Dislikes unnecessary emoji or filler language
```

Entries are separated by `§` (section sign). The model references entries by quoting their content via the `old_text` parameter for replace/remove operations -- content-based matching, not position-based.

Injected after SOUL.md/AGENTS.md but before skills. Only injected when the respective memory is enabled.

### Bounded memory: how pruning works

**The model does the pruning, but it's cheap because it already sees everything.** Since both files are in the system prompt, the model always knows exactly what's in memory. When at the character limit:

1. Model calls `memory(action="add", target="memory", content="new thing")`
2. Tool checks: would this exceed the 2,200 char limit?
3. If yes: tool returns an error with the current usage (e.g., "2,150/2,200 chars used — consolidate or replace entries first")
4. Model (which already sees all entries + usage % in its system prompt) decides what to consolidate or replace
5. Model calls `replace` or `remove`, then retries the `add`

This costs one extra tool call when at the limit, but the model makes an **informed decision** (not blindly evicting oldest). The user's name won't get evicted because the model knows it's important. Stale or redundant entries get consolidated.

**Why not auto-evict oldest?** Because some early memories are the most important (user's name, core preferences, critical environment facts). The model, which sees everything, is the right judge of what to prune.

### The `memory` tool

**Actions:**
- `add(target, content)` -- append a new dated entry. Fails with guidance if over char limit.
- `replace(target, old_text, new_content)` -- find entry containing `old_text`, replace it. For updates and consolidation.
- `remove(target, old_text)` -- remove the entry containing `old_text`.
- `read(target)` -- return current contents. Useful after context compression when system prompt may be stale.

**Tool description (teaches the model everything):**
```
Manage your persistent memory. You have two memory stores, both visible in
your system prompt every session:

MEMORY — your personal notes and observations (2,200 character limit)
  Things worth remembering: environment facts, project conventions, tool quirks,
  things that took effort to figure out, recurring patterns.

USER PROFILE — what you know about the user (1,375 character limit)
  Preferences, communication style, expectations, workflow habits, corrections
  they've given you.

Actions:
  add(target, content)                   — append a new entry
  replace(target, old_text, new_content) — find entry matching old_text,
                                           replace it
  remove(target, old_text)               — remove entry matching old_text
  read(target)                           — return current live contents

For replace/remove, old_text is a short unique substring that identifies the
target entry — just a few words, not the full text. If your snippet matches
multiple entries, you'll get an error showing the matches so you can retry
with something more specific.

Rules:
- You can always see your current memories in the system prompt, along with
  a usage indicator showing how close you are to the limit (e.g. [68% — 1,496/2,200 chars]).
- Each entry is one line.
- When usage is high (>80%), consolidate or replace entries before adding new
  ones. Merge related entries. Remove stale or redundant info.
- Prefer REPLACING over REMOVING — update with better info rather than delete
  and re-add.
- Do not store secrets, tokens, or passwords.
- Only store things that will meaningfully help you in future sessions.

When to write memories:
- You learned something non-obvious about the environment or a project
- The user corrected you or expressed a preference
- You discovered a tool quirk, workaround, or convention
- The user explicitly asks you to remember something
- You completed something difficult (consider a skill instead if it's a full
  reusable procedure)

When NOT to write memories:
- Trivial or one-off facts that won't matter next session
- Things already captured in a skill
- Things you can easily re-discover (file contents, command outputs)
```

### Config

```yaml
memory:
  memory_enabled: true          # MEMORY.md - agent's personal notes
  user_profile_enabled: true    # USER.md - user preferences/identity
  memory_char_limit: 2200       # ~800 tokens at 2.75 chars/token
  user_char_limit: 1375         # ~500 tokens at 2.75 chars/token
```

Both default to `false` in batch_runner and RL environments (checked programmatically). Configurable per-environment.

### Long-term recall (session search)

The bounded memory is the curated layer. For unbounded "long-term memory" -- searching past session transcripts -- see SQLite State Store (#5). A separate `session_search` tool provides ripgrep-style search over the full session history. This is never injected into the system prompt; the agent searches it on demand.

### Known bug: duplicate entries are irrecoverable

If the same content gets added twice (e.g., agent retries after a failed response generation and re-saves a memory it already saved), both `replace` and `remove` fail with "Multiple entries matched" because content-based matching can't disambiguate identical entries. The user/agent has no way to fix this without manually editing the file.

**Possible fixes (pick one or combine):**
- **Prevent on add**: reject or silently deduplicate when `content` exactly matches an existing entry. Cheapest fix, prevents the problem entirely.
- **Remove-all flag**: `remove(target, old_text, remove_all=True)` deletes every match. Lets the agent nuke all copies, then re-add one if needed.
- **Index-based fallback**: when multiple matches are found, return them with indices and accept `index` parameter on retry. More complex, but most precise.

Simplest path: deduplicate on add (check before appending). One-line fix in `memory_tool.py`.

### Later (optional)
- Periodic consolidation via cronjob/subagent: reviews recent session summaries, suggests memory updates. Needs subagents (#1).
- Memory import/export: `hermes memory export` / `hermes memory import` for backup/migration.

**Files:** `tools/memory_tool.py` (tool logic + file I/O), `model_tools.py` (registration), system prompt injection in `run_agent.py`

---

## 5. SQLite State Store & Session Search 🔍

**Status:** Not started
**Priority:** High -- foundational infrastructure for memory, search, and scale

Replace the current JSONL-per-session file approach with a SQLite database. This is infrastructure that makes everything else work better at scale.

**The problem with JSONL files:**
- Currently: one `.jsonl` file per session in `~/.hermes/sessions/` and `logs/`
- At 5-10 sessions per day across 4 platforms, that's 1,500-3,600 files per year
- Searching across sessions requires ripgrep over thousands of files (slow, no filtering)
- No relational queries (e.g., "show me all sessions about Docker from last month")
- No way to store structured metadata alongside transcripts (summaries, tags, memory references)
- File system overhead: inode limits, directory listing performance, backup complexity

**What Codex does:**
- SQLite state database with tables for threads, stage1_outputs (extracted memories), and jobs (background processing queue with leases/heartbeats/watermarks)
- All session metadata, memory extraction outputs, and job coordination in one DB
- File system only used for human-readable artifacts (MEMORY.md, rollout_summaries/, skills/)

**Our approach:**

### Schema: `~/.hermes/state.db`

```sql
-- Core session data
sessions (
    id TEXT PRIMARY KEY,
    platform TEXT,           -- telegram, discord, whatsapp, slack, cli
    user_id TEXT,
    started_at INTEGER,
    ended_at INTEGER,
    summary TEXT,            -- agent-written session summary (nullable)
    tags TEXT,               -- comma-separated tags
    message_count INTEGER,
    tool_call_count INTEGER
)

-- Full message history (replaces JSONL)
messages (
    id INTEGER PRIMARY KEY,
    session_id TEXT REFERENCES sessions,
    role TEXT,               -- user, assistant, tool, system
    content TEXT,
    tool_name TEXT,          -- nullable, for tool calls
    timestamp INTEGER,
    tokens_used INTEGER
)

-- FTS5 virtual table for fast text search across messages
messages_fts USING fts5(content, content=messages, content_rowid=id)

-- Session summaries (written by agent at session end)
session_summaries (
    session_id TEXT PRIMARY KEY REFERENCES sessions,
    summary TEXT,
    keywords TEXT,
    created_at INTEGER
)

-- Learnings from errors (item #15)
learnings (
    id INTEGER PRIMARY KEY,
    pattern TEXT,
    error_type TEXT,
    fix TEXT,
    context TEXT,
    tags TEXT,
    created_at INTEGER,
    times_used INTEGER DEFAULT 0
)
```

### Migration path from JSONL
- New sessions go directly to SQLite
- Existing JSONL files imported on first run (background migration)
- `hermes migrate-sessions` CLI command for manual migration
- Keep JSONL export as an option (`hermes sessions export <id>`)

### Session search (replaces the old plan)
- **Agent tool**: `session_search(query, role_filter, date_range, platform, limit)` -- FTS5 search across messages table
- **CLI command**: `hermes sessions search <query>` with filters
- FTS5 gives us: ranking, phrase matching, boolean operators, prefix queries
- Much faster than ripgrep over thousands of files
- Filter by platform, date range, role -- impossible with flat files

### Benefits for other systems
- **Memory**: session summaries stored in `session_summaries` table, searchable by keyword
- **Learnings**: structured storage with `times_used` counter and tag search
- **Subagents**: parent-child session relationships trackable via foreign keys
- **Analytics**: token usage over time, tool call frequency, session duration -- trivially queryable
- **Cleanup**: `hermes sessions prune --older-than 90d` becomes a SQL DELETE

**Files:** `hermes_state.py` (SQLite wrapper, schema, migrations), `tools/session_search_tool.py` (agent tool), `hermes_cli/sessions.py` (CLI)

---

## 6. Local Browser Control via CDP 🌐

**Status:** Not started (currently Browserbase cloud only)
**Priority:** Medium

Support local Chrome/Chromium via Chrome DevTools Protocol alongside existing Browserbase cloud backend.

**What other agents do:**
- **OpenClaw**: Full CDP-based Chrome control with snapshots, actions, uploads, profiles, file chooser, PDF save, console messages, tab management. Uses local Chrome for persistent login sessions.
- **Cline**: Headless browser with Computer Use (click, type, scroll, screenshot, console logs)

**Our approach:**
- Add a `local` backend option to `browser_tool.py` using Playwright or raw CDP
- Config toggle: `browser.backend: local | browserbase | auto`
- `auto` mode: try local first, fall back to Browserbase
- Local advantages: free, persistent login sessions, no API key needed
- Local disadvantages: no CAPTCHA solving, no stealth mode, requires Chrome installed
- Reuse the same 10-tool interface -- just swap the backend
- Later: Chrome profile management for persistent sessions across restarts

---

## 7. Signal Integration 📡

**Status:** Not started
**Priority:** Low

New platform adapter using signal-cli daemon (JSON-RPC HTTP + SSE). Requires Java runtime and phone number registration.

**Reference:** OpenClaw has Signal support via signal-cli.

---

## 8. Plugin/Extension System 🔌

**Status:** Partially implemented (event hooks exist in `gateway/hooks.py`)
**Priority:** Medium

Full Python plugin interface that goes beyond the current hook system.

**What other agents do:**
- **OpenClaw**: Plugin SDK with tool-send capabilities, lifecycle phase hooks (before-agent-start, after-tool-call, model-override), plugin registry with install/uninstall.
- **Pi**: Extensions are TypeScript modules that can register tools, commands, keyboard shortcuts, custom UI widgets, overlays, status lines, dialogs, compaction hooks, raw terminal input listeners. Extremely comprehensive.
- **OpenCode**: MCP client support (stdio, SSE, StreamableHTTP), OAuth auth for MCP servers. Also has Copilot/Codex plugins.
- **Codex**: Full MCP integration with skill dependencies.
- **Cline**: MCP integration + lifecycle hooks with cancellation support.

**Our approach (phased):**

### Phase 1: Enhanced hooks
- Expand the existing `gateway/hooks.py` to support more events: `before-tool-call`, `after-tool-call`, `before-response`, `context-compress`, `session-end`
- Allow hooks to modify tool results (e.g., filter sensitive output)

### Phase 2: Plugin interface
- `~/.hermes/plugins/<name>/plugin.yaml` + `handler.py`
- Plugins can: register new tools, add CLI commands, subscribe to events, inject system prompt sections
- `hermes plugin list|install|uninstall|create` CLI commands
- Plugin discovery and validation on startup

### Phase 3: MCP support (industry standard)
- MCP client that can connect to external MCP servers (stdio, SSE, HTTP)
- This is the big one -- Codex, Cline, and OpenCode all support MCP
- Allows Hermes to use any MCP-compatible tool server (hundreds exist)
- Config: `mcp_servers` list in config.yaml with connection details
- Each MCP server's tools get registered as a new toolset

---

## 9. Native Companion Apps 📱

**Status:** Not started
**Priority:** Low

macOS (Swift/SwiftUI), iOS, Android apps connecting via WebSocket.

**Prerequisite:** WebSocket API on gateway (new endpoint alongside the existing HTTP flow).

**What other agents do:**
- **OpenClaw**: iOS + Android + macOS companion apps with Bonjour pairing. Voice Wake (always-on speech detection). Talk Mode (continuous conversation overlay). A2UI Canvas for agent-pushed visual content.
- **OpenCode**: Desktop app via Tauri (macOS, Windows, Linux). Also has a VS Code extension.

**Our approach:**
- MVP: Web UI with Flask/FastAPI + WebSocket (much faster to build than native apps)
- Later: Tauri desktop app wrapping the web UI
- Later: Native iOS/Android if there's demand

---

## 10. Evaluation System 📏

**Status:** Not started
**Priority:** Medium

Systematic evaluation of agent performance for batch_runner and RL training.

**What other agents do:**
- **Dash**: Evaluation system with test cases and LLM grading

**Our approach:**
- LLM grader mode for batch_runner: after each trajectory, a judge model scores the result
- Action comparison: expected tool call sequences vs. actual
- String matching baselines for simple checks
- Metrics: task completion rate, tool efficiency (calls per task), cost per task
- Export to WandB for tracking across runs

---

## 11. Layered Context Architecture 📊

**Status:** Partially implemented (context files, skills, compression exist)
**Priority:** Medium

Structured hierarchy for what goes into the system prompt, with clear priority ordering.

**Current state:** We have SOUL.md, AGENTS.md, skills, session context, and compression. But it's ad-hoc -- no explicit layering or budget allocation.

**Our approach:**
- Define explicit layers with token budgets: `project context (AGENTS.md) > skills > user profile (USER.md) > learnings > memory > session context > runtime introspection`
- Each layer has a max token budget; when total exceeds limit, lower-priority layers get summarized first
- Runtime introspection layer: current working directory, active processes, git status, recent file changes
- This becomes the backbone for the memory system (item 4) and subagent architecture (item 1)

---

## 12. Tools Wishlist 🧰

**Status:** Various
**Priority:** Mixed

### Diagram Rendering
- Mermaid/PlantUML text to PNG/SVG images
- Agent writes diagram code, tool renders it, returns image path
- Dependencies: `mmdc` (mermaid-cli) or PlantUML jar

### Document Generation
- PDFs (via WeasyPrint or reportlab), Word docs (python-docx), presentations (python-pptx)
- Agent writes content, tool generates formatted document
- Templates for common formats (report, resume, letter, slides)

### Canvas / Visual Workspace
- **OpenClaw has this**: A2UI (Agent-to-UI) -- agent pushes visual content to a Canvas surface
- For us: could be a web-based canvas (HTML/JS) that the agent can draw on
- Depends on companion app / web UI (item 9)

### Coding Agent Skill
- Orchestrate Codex CLI or Claude Code via PTY mode (already supported!)
- Create a skill document that teaches the agent how to use these tools effectively
- Not a new tool -- just a well-crafted skill + PTY terminal usage

### Domain Skill Packs
- Curated skill collections for specific domains
- DevOps: Docker, K8s, CI/CD, monitoring
- Data Science: pandas, scikit-learn, plotting, data cleaning
- Security: vulnerability scanning, OWASP, secrets management
- Each pack is a set of SKILL.md files installable via `hermes skills install <pack>`

---

## NEW: Items Discovered from Agent Codebases Analysis

---

## 13. MCP (Model Context Protocol) Support 🔗

**Status:** Not started
**Priority:** High -- this is becoming an industry standard

MCP is the protocol that Codex, Cline, and OpenCode all support for connecting to external tool servers. Supporting MCP would instantly give Hermes access to hundreds of community tool servers.

**What other agents do:**
- **Codex**: Full MCP integration with skill dependencies
- **Cline**: `use_mcp_tool` / `access_mcp_resource` / `load_mcp_documentation` tools
- **OpenCode**: MCP client support (stdio, SSE, StreamableHTTP transports), OAuth auth

**Our approach:**
- Implement an MCP client that can connect to external MCP servers
- Config: list of MCP servers in `~/.hermes/config.yaml` with transport type and connection details
- Each MCP server's tools auto-registered as a dynamic toolset
- Start with stdio transport (most common), then add SSE and HTTP
- Could also be part of the Plugin system (item 8, Phase 3) since MCP is essentially a plugin protocol

---

## 14. Permission / Safety System 🛡️

**Status:** Partially implemented (dangerous command approval in gateway)
**Priority:** Medium

Formalize the tool permission system beyond the current ad-hoc dangerous command checks.

**What other agents do:**
- **OpenCode**: Sophisticated permission system with wildcards, deny/allow/ask per tool pattern. Configurable `primary_tools` that bypass restrictions.
- **Codex**: Exec policy system for allowed/denied commands. Seatbelt (macOS) / Landlock (Linux) sandboxing. Network proxy for controlling outbound access.
- **OpenClaw**: Tool policy pipeline with allowlist/denylist per sandbox. Docker per-session sandboxing.

**Our approach:**
- Config-based permission rules: `permissions.yaml` with patterns like `terminal.*: ask`, `file.write: allow`, `browser.*: deny`
- Per-platform overrides (stricter on Discord groups than Telegram DMs)
- Tool call logging with audit trail (who ran what, when, approved by whom)
- Later: sandboxing options (Docker per-session, like OpenClaw)

---

## 15. Self-Learning from Errors 📖

**Status:** Not started
**Priority:** Medium-High -- the "error memory" layer of the knowledge system

Automatic learning loop: when tool calls fail, the agent diagnoses the error, fixes it, and saves the pattern so it never repeats the same mistake. This is the error-specific counterpart to skills (procedural memory) and MEMORY.md (declarative memory).

**What Dash does:**
- LearningMachine with agentic mode
- Success -> optionally save as Knowledge
- Error -> Diagnose -> Fix -> Save Learning (never repeated)
- 6 layers of context including institutional knowledge and runtime schema

**Our approach:**
- Part of the knowledge system, stored in the SQLite state store (#5) once available, or `~/.hermes/learnings.jsonl` as fallback
- Each learning: `{pattern, error_type, fix, context, tags, created_at, times_used}`
- `learning` tool with actions: `save`, `search`, `list`
- Before executing operations that have failed before, auto-inject relevant learnings (appended to tool responses, same pattern as todo checkpoint nudges -- cache-friendly)
- Agent prompted: "This is similar to a previous error. Here's what worked last time: ..."
- Consolidation: periodically merge similar learnings and increment `times_used`
- Triggered automatically on tool call errors OR manually by the agent
- **Relationship to skills**: if the same error pattern appears 3+ times, the agent should consider creating a skill that includes the fix as a "pitfalls" section, rather than keeping it as a standalone learning

---

## 16. Session Branching / Checkpoints 🌿

**Status:** Not started
**Priority:** Low-Medium

Save and restore conversation state at any point. Branch off to explore alternatives without losing progress.

**What other agents do:**
- **Pi**: Full branching -- create branches from any point in conversation. Branch summary entries. Parent session tracking for tree-like session structures.
- **Cline**: Checkpoints -- workspace snapshots at each step with Compare/Restore UI
- **OpenCode**: Git-backed workspace snapshots per step, with weekly gc

**Our approach:**
- `checkpoint` tool: saves current message history + working directory state as a named snapshot
- `restore` tool: rolls back to a named checkpoint
- Stored in `~/.hermes/checkpoints/<session_id>/<name>.json`
- For file changes: git stash or tar snapshot of working directory
- Useful for: "let me try approach A, and if it doesn't work, roll back and try B"
- Later: full branching with tree visualization

---

## 17. File Watcher / Project Awareness 👁️

**Status:** Not started
**Priority:** Low

Monitor the working directory for changes and notify the agent of relevant updates.

**What other agents do:**
- **Codex**: File watcher for live project change detection
- **OpenCode**: Git info integration, worktree support

**Our approach:**
- Watchdog-based file watcher on the working directory
- Inject a system message when relevant files change (e.g., "Note: `src/main.py` was modified externally")
- Filter noise: ignore `.git/`, `node_modules/`, `__pycache__/`, etc.
- Useful for pair-programming scenarios where the user is also editing files
- Could also track git status changes

---

## 18. Heartbeat System 💓

**Status:** Not started
**Priority:** Low-Medium

Periodic agent wake-up for checking reminders, monitoring tasks, and running scheduled introspection.

**What other agents do:**
- **OpenClaw**: Heartbeat runner with configurable intervals per-agent (e.g., every 30m, 10m, 15m). Ghost reminder system. Transcript pruning on heartbeat. Dynamic config update without restart. Wake-on-demand.

**Our approach:**
- `HEARTBEAT.md` file in `~/.hermes/` -- agent reads this on each heartbeat
- Configurable interval (default: 30 minutes when gateway is running)
- Runs inside an existing session (not a new one) to maintain context
- Can be used for: checking on background processes, sending reminders, running periodic tasks
- `HEARTBEAT_OK` response suppresses output when nothing needs attention
- Integrates with the existing cronjob system for scheduling

---

## 19. Programmatic Tool Calling (Code-Mediated Tool Use) 🧬

**Status:** Not started
**Priority:** High -- potentially the single biggest efficiency win for agent loops

Instead of the LLM making one tool call, reading the result, deciding what to do next, making another tool call (N round trips), the LLM writes a Python script that calls multiple tools, processes results, branches on conditions, and returns a final summary -- all in one turn.

**What Anthropic just shipped (Feb 2026):**

Anthropic's new `web_search_20260209` and `web_fetch_20260209` tools use "dynamic filtering" -- Claude writes and executes Python code that calls the search/fetch tools, filters the HTML, cross-references results, retries with different queries, and returns only what's relevant. Results: **+11% accuracy, -24% input tokens** on average across BrowseComp and DeepsearchQA. Quora/Poe found it "achieved the highest accuracy on our internal evals" and described it as behaving "like an actual researcher, writing Python to parse, filter, and cross-reference results rather than reasoning over raw HTML in context."

Source: [claude.com/blog/improved-web-search-with-dynamic-filtering](https://claude.com/blog/improved-web-search-with-dynamic-filtering)

**Why this matters for agent loops:**

The standard agent loop is:
```
LLM call -> tool call -> result -> LLM call -> tool call -> result -> LLM call -> ...
```
Every round trip costs: a full LLM inference (prompt + generation), network latency, and the growing context window carrying all previous tool results. For a 10-step task, that's 10+ LLM calls with increasingly large contexts.

With programmatic tool calling:
```
LLM call -> writes Python script -> script calls tools N times, processes results,
branches on conditions, retries on failure -> returns summary -> LLM call
```
One LLM call replaces many. The intermediate tool results never enter the context window -- they're processed in the code sandbox and only the final summary comes back. The LLM pre-plans its decision tree in code rather than making decisions one-at-a-time in the conversation.

**Which of our tools benefit most:**

| Tool | Current pattern (N round trips) | With programmatic calling (1 round trip) |
|------|--------------------------------|----------------------------------------|
| **web_search + web_extract** | Search -> read results -> pick URLs -> extract each -> read each -> synthesize | Script: search, fetch top 5, extract relevant sections, cross-reference, return summary |
| **browser (10 tools)** | navigate -> snapshot -> click -> snapshot -> type -> snapshot -> ... | Script: navigate, loop through elements, extract data, handle pagination, return structured result |
| **file ops (read, search, patch)** | Search for pattern -> read matching files -> decide which to patch -> patch each | Script: search, read all matches, filter by criteria, apply patches, verify, return diff summary |
| **session_search** | Search -> read results -> search again with refined query -> ... | Script: search with multiple queries, deduplicate, rank by relevance, return top N |
| **terminal** | Run command -> check output -> run follow-up -> check again -> ... | Script: run command, parse output, branch on exit code, run follow-ups, return final state |

### The hard problem: where does the code run?

Our tools don't all live in one place. The terminal backend can be local, Docker, Singularity, SSH, or Modal. Browser runs on Browserbase cloud. Web search/extract are Firecrawl API calls. File ops go through whatever terminal backend is active. Vision, image gen, TTS are all remote APIs.

If we just "run Python in the terminal," we hit a wall:
- **Docker/Modal/SSH backends**: The remote environment doesn't have our Python tool code, our API keys, our `handle_function_call` dispatcher, or any of the hermes-agent packages. It's a bare sandbox.
- **Local backend**: Could import our code directly, but that couples execution to local-only and creates a security mess (LLM-generated code running in the agent process).
- **API-based tools** (web, browser, vision): These need API keys and specific client libraries that aren't in the terminal backend.

**The code sandbox is NOT the terminal backend.** This is the key insight. The sandbox runs on the **agent host machine** (where `run_agent.py` lives), separate from both the LLM and the terminal backend. It calls tools through the same `handle_function_call` dispatcher that the normal agent loop uses. No inbound network connections needed -- everything is local IPC on the agent host.

### Architecture: Local subprocess with Unix domain socket RPC

```
Agent Host (where run_agent.py runs)

  ┌──────────────┐         ┌──────────────────────┐
  │ Code Sandbox │  UDS    │ Agent Process         │
  │ (child proc) │ ◄─────► │ (parent)              │
  │              │ (local) │                       │
  │ print()──────┼─stdout──┼─► captured as output  │
  │ errors───────┼─stderr──┼─► captured for debug  │
  │              │         │                       │
  │ hermes_tools │         │ handle_function_call  │
  │ .web_search()├─socket──┼─► web_search ─────────┼──► Firecrawl API
  │ .terminal()  ├─socket──┼─► terminal ───────────┼──► Docker/SSH/Modal
  │ .browser_*() ├─socket──┼─► browser ────────────┼──► Browserbase
  │ .read_file() ├─socket──┼─► read_file ──────────┼──► terminal backend
  └──────────────┘         └──────────────────────┘
```

The sandbox is a **child process** on the same machine. RPC goes over a **Unix domain socket** (not stdin/stdout/stderr -- those stay free for their natural purposes). The parent dispatches each tool call through the existing `handle_function_call` -- the exact same codepath the normal agent loop uses. Works with every terminal backend because the sandbox doesn't touch the terminal backend directly.

### RPC transport: Unix domain socket

Why not stdin/stdout/stderr for RPC?
- **stdout** is the script's natural output channel (`print()`). Multiplexing RPC and output on the same stream requires fragile marker parsing. Keep it clean: stdout = final output for the LLM.
- **stderr** is for Python errors, tracebacks, warnings, and `logging` output. Multiplexing RPC here means any stray `logging.warning()` or exception traceback corrupts the RPC stream.
- **Extra file descriptors (fd 3/4)** work on Linux/macOS but are clunky with subprocess.Popen.

A **Unix domain socket** gives a clean dedicated RPC channel:
1. Parent creates a temp UDS: `/tmp/hermes_rpc_<uuid>.sock`
2. Parent starts listening (single-client, since there's one sandbox)
3. Parent spawns child with `HERMES_RPC_SOCKET=/tmp/hermes_rpc_<uuid>.sock` in env
4. Child's `hermes_tools` module connects to the socket on first tool call
5. Protocol: newline-delimited JSON. Child writes `{"tool": "web_search", "args": {...}}\n`, reads `{"result": ...}\n` back
6. Parent reads each request, calls `handle_function_call`, writes the response
7. After child exits, parent cleans up the socket file

**Channels stay clean:**
- `stdout` → captured by parent as the tool's return value to the LLM
- `stderr` → captured by parent for error reporting (included in response on failure)
- UDS → dedicated tool call RPC (invisible to the script's normal I/O)

Works on Linux and macOS (our target platforms). Windows fallback: named pipes or the marker-on-stderr approach if we ever need it.

### The auto-generated `hermes_tools` module

The parent writes this into a temp directory before spawning the child. Each function is a thin RPC stub:

```python
# Auto-generated: /tmp/hermes_sandbox_<uuid>/hermes_tools.py
import json, os, socket

_sock = None

def _connect():
    global _sock
    if _sock is None:
        _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        _sock.connect(os.environ["HERMES_RPC_SOCKET"])
        _sock.settimeout(300)  # 5 min max per tool call
    return _sock

def _call(tool_name, args):
    """RPC: send tool call to parent, get result back."""
    conn = _connect()
    request = json.dumps({"tool": tool_name, "args": args}) + "\n"
    conn.sendall(request.encode())
    # Read response (newline-delimited)
    chunks = []
    while True:
        data = conn.recv(65536)
        if not data:
            raise RuntimeError("Agent process disconnected")
        chunks.append(data.decode())
        if chunks[-1].endswith("\n"):
            break
    raw = "".join(chunks).strip()
    # Tool responses are JSON strings; parse them into dicts
    result = json.loads(raw)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result

# --- Tool functions (one per enabled tool) ---

def web_search(query):
    """Search the web. Returns dict with 'results' list."""
    return _call("web_search", {"query": query})

def web_extract(urls):
    """Extract content from URLs. Returns markdown text."""
    return _call("web_extract", {"urls": urls})

def read_file(path, offset=1, limit=500):
    """Read a file. Returns dict with content and metadata."""
    return _call("read_file", {"path": path, "offset": offset, "limit": limit})

def terminal(command, timeout=None):
    """Run a shell command. Returns dict with stdout, exit_code."""
    return _call("terminal", {"command": command, "timeout": timeout})

def search(pattern, target="content", path=".", file_glob=None, limit=50):
    """Search file contents or find files."""
    return _call("search", {"pattern": pattern, "target": target,
                             "path": path, "file_glob": file_glob, "limit": limit})

# ... generated for each enabled tool in the session
```

This module is generated dynamically because the available tools vary per session and per toolset configuration. The generator reads the session's enabled tools and emits a function for each one that's on the sandbox allow-list.

### What Python libraries are available in the sandbox

The sandbox runs the same Python interpreter as the agent. Available imports:

**Python standard library (always available):**
`json`, `re`, `math`, `csv`, `datetime`, `collections`, `itertools`, `textwrap`, `difflib`, `html`, `urllib.parse`, `pathlib`, `hashlib`, `base64`, `string`, `functools`, `operator`, `statistics`, `io`, `os.path`

**Not restricted but discouraged via tool description:**
`subprocess`, `socket`, `requests`, `urllib.request`, `os.system` -- the tool description says "use hermes_tools for all I/O." We don't hard-block these because the user already trusts the agent with `terminal()`, which is unrestricted shell access. Soft-guiding the LLM via the description is sufficient. If it occasionally uses `import os; os.listdir()` instead of `hermes_tools.search()`, no real harm done.

**The tool description tells the LLM:**
```
Available imports:
- from hermes_tools import web_search, web_extract, read_file, terminal, ...
- Python standard library: json, re, math, csv, datetime, collections, etc.

Use hermes_tools for all I/O (web, files, commands, browser).
Use stdlib for processing between tool calls (parsing, filtering, formatting).
Print your final result to stdout.
```

### Platform support

**Linux / macOS**: Fully supported. Unix domain sockets work natively.

**Windows**: Not supported. `AF_UNIX` nominally exists on Windows 10 17063+ but is unreliable in practice, and Hermes-Agent's primary target is Linux/macOS (bash-based install, systemd gateway, etc.). The `execute_code` tool is **disabled at startup on Windows**:

```python
import sys
SANDBOX_AVAILABLE = sys.platform != "win32"

def check_sandbox_requirements():
    return SANDBOX_AVAILABLE
```

If the LLM tries to use `execute_code` on Windows, it gets: `{"error": "execute_code is not available on Windows. Use normal tool calls instead."}`. The tool is excluded from the tool schema entirely on Windows so the LLM never sees it.

### Which tools to expose in the sandbox (full audit)

The purpose of the sandbox is **reading, filtering, and processing data across multiple tool calls in code**, collapsing what would be many LLM round trips into one. Every tool needs to justify its inclusion against that purpose. The parent only generates RPC stubs for tools that pass this filter AND are enabled in the session.

**Every tool, one by one:**

| Tool | In sandbox? | Reasoning |
|------|------------|-----------|
| `web_search` | **YES** | Core use case. Multi-query, cross-reference, filter. |
| `web_extract` | **YES** | Core use case. Fetch N pages, parse, keep only relevant sections. |
| `read_file` | **YES** | Core use case. Bulk read + filter. Note: reads files on the terminal backend (Docker/SSH/Modal), not the agent host -- this is correct and intentional. |
| `search` | **YES** | Core use case. Find files/content, then process matching results. |
| `terminal` | **YES (restricted)** | Command chains with branching on exit codes. Foreground only -- `background`, `check_interval`, and `pty` parameters are stripped/blocked. |
| `write_file` | **YES (with caution)** | Scripts need to write computed outputs (generated configs, processed data). Partial-write risk if script fails midway, but same risk as normal tool calls. |
| `patch` | **YES (with caution)** | Bulk search-and-replace across files. Powerful but risky if the script's patch logic has bugs. The upside: script can read → patch → verify in a loop, which is actually safer than blind patching. |
| `browser_navigate` | **YES** | Browser automation loops are one of the biggest wins. |
| `browser_snapshot` | **YES** | Needed for reading page state in browser loops. Parent passes `user_task` from the session context. |
| `browser_click` | **YES** | Core browser automation. |
| `browser_type` | **YES** | Core browser automation. |
| `browser_scroll` | **YES** | Core browser automation. |
| `browser_back` | **YES** | Navigation within browser loops. |
| `browser_press` | **YES** | Keyboard interaction in browser loops. |
| `browser_close` | **NO** | Ends the entire browser session. If the script errors out after closing, the agent has no browser to recover with. Too destructive for unsupervised code. |
| `browser_get_images` | **NO** | Niche. Usually paired with vision analysis, which is excluded. |
| `browser_vision` | **NO** | This calls the vision LLM API -- expensive per call and requires LLM reasoning on the result. Defeats the purpose of avoiding LLM round trips. |
| `vision_analyze` | **NO** | Expensive API call per invocation. The LLM needs to SEE and reason about images directly, not filter them in code. One-shot nature. |
| `mixture_of_agents` | **NO** | This IS multiple LLM calls. Defeats the entire purpose. |
| `image_generate` | **NO** | Media generation. One-shot, no filtering logic benefits. |
| `text_to_speech` | **NO** | Media generation. One-shot. |
| `process` | **NO** | Background process management from an ephemeral script is incoherent. The script exits, but the process lives on -- who monitors it? |
| `skills_list` | **NO** | Skills are knowledge for the LLM to read and reason about. Loading a skill inside code that can't reason about it is pointless. |
| `skill_view` | **NO** | Same as above. |
| `schedule_cronjob` | **NO** | Side effect. Should not happen silently inside a script. |
| `list_cronjobs` | **NO** | Read-only but not useful in a code-mediation context. |
| `remove_cronjob` | **NO** | Side effect. |
| `send_message` | **NO** | Cross-platform side effect. Must not happen unsupervised. |
| `todo_write` | **NO** | Agent-level conversational state. Meaningless from code. |
| `todo_read` | **NO** | Same. |
| `clarify` | **NO** | Requires interactive user input. Can't block in a script. |
| `execute_code` | **NO** | No recursive sandboxing. |
| All RL tools | **NO** | Separate domain with its own execution model. |

**Summary: 14 tools in, 28+ tools out.** The sandbox exposes: `web_search`, `web_extract`, `read_file`, `write_file`, `search`, `patch`, `terminal` (restricted), `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_scroll`, `browser_back`, `browser_press`.

The allow-list is a constant in `code_execution_tool.py`, not derived from the session's enabled toolsets. Even if the session has `vision_analyze` enabled, it won't appear in the sandbox. The intersection of the allow-list and the session's enabled tools determines what's generated.

### Error handling

| Scenario | What happens |
|----------|-------------|
| **Syntax error in script** | Child exits immediately, traceback on stderr. Parent returns stderr as the tool response so the LLM sees the error and can retry. |
| **Runtime exception** | Same -- traceback on stderr, parent returns it. |
| **Tool call fails** | RPC returns the error JSON (same as normal tool errors). Script decides: retry, skip, or raise. |
| **Unknown tool called** | RPC returns `{"error": "Unknown tool: foo. Available: web_search, read_file, ..."}`. |
| **Script hangs / infinite loop** | Killed by timeout (SIGTERM, then SIGKILL after 5s). Parent returns timeout error. |
| **Parent crashes mid-execution** | Child's socket connect/read fails, gets a RuntimeError, exits. |
| **Child crashes mid-execution** | Parent detects child exit via `process.poll()`. Collects partial stdout + stderr. |
| **Slow tool call (e.g., terminal make)** | Overall timeout covers total execution. One slow call is fine if total is under limit. |
| **Tool response too large in memory** | `web_extract` can return 500KB per page. If the script fetches 10 pages, that's 5MB in the child's memory. Not a problem on modern machines, and the whole point is the script FILTERS this down before printing. |
| **User interrupt (new message on gateway)** | Parent catches the interrupt event (same as existing `_interrupt_event` in terminal_tool), sends SIGTERM to child, returns `{"status": "interrupted"}`. |
| **Script tries to call excluded tool** | RPC returns `{"error": "Tool 'vision_analyze' is not available in execute_code. Use it as a normal tool call instead."}` |
| **Script calls terminal with background=True** | RPC strips the parameter and runs foreground, or returns an error. Background processes from ephemeral scripts are not supported. |

The tool response includes structured metadata:
```json
{
  "status": "success | error | timeout | interrupted",
  "output": "...",
  "errors": "...",
  "tool_calls_made": 7,
  "duration_seconds": 12.3
}
```

### Resource limits

- **Timeout**: 120 seconds default (configurable via `config.yaml`). Parent sends SIGTERM, waits 5s, SIGKILL.
- **Tool call limit**: max 50 RPC tool calls per execution. After 50, further calls return an error. Prevents infinite tool-call loops.
- **Output size**: stdout capped at 50KB. Truncated with `[output truncated at 50KB]`. Prevents the script from flooding the LLM's context with a huge result (which would defeat the purpose).
- **Stderr capture**: capped at 10KB for error reporting.
- **No recursive sandboxing**: `execute_code` is not in the sandbox's tool list.
- **Interrupt support**: respects the same `_interrupt_event` mechanism as terminal_tool. If the user sends a new message while the sandbox is running, the child is killed and the agent can process the interrupt.

### Tool call logging and observability

Each tool call made inside the sandbox is **logged to the session transcript** for debugging, but **NOT added to the LLM conversation history** (that's the whole point -- keeping intermediate results out of context).

The parent logs each RPC-dispatched call:
```jsonl
{"type": "sandbox_tool_call", "tool": "web_search", "args": {"query": "..."}, "duration": 1.2}
{"type": "sandbox_tool_call", "tool": "web_extract", "args": {"urls": [...]}, "duration": 3.4}
```

These appear in the JSONL transcript and in verbose logging, but the LLM only sees the final `execute_code` response containing the script's stdout.

For the gateway (messaging platforms): show one typing indicator + notification for the entire `execute_code` duration. Internal tool calls are silent. Later enhancement: progress updates like "execute_code (3/7 tool calls)".

### Stateful tools work correctly

Tools like `terminal` (working directory, env vars) and `browser_*` (page state, cookies) maintain state per `task_id`. The parent passes the session's `task_id` to every RPC-dispatched `handle_function_call`. So if the script runs:
```python
terminal("cd /tmp")
terminal("pwd")  # returns /tmp -- state persists between calls
```
This works because both calls go through the same terminal environment, same as normal tool calls.

### Each `execute_code` invocation is stateless

The sandbox subprocess is fresh each time. No Python state carries over between `execute_code` calls. If the agent needs state across multiple `execute_code` calls, it should:
- Output the state as part of the result, then pass it back in the next script as a variable
- Or use the tools themselves for persistence (write to a file, then read it in the next script)

The underlying *tools* are stateful (same terminal session, same browser session), but the *Python sandbox* is not.

### When should the LLM use `execute_code` vs normal tool calls?

This goes in the tool description:

**Use `execute_code` when:**
- You need 3+ tool calls with processing logic between them
- You need to filter/reduce large tool outputs before they enter your context
- You need conditional branching (if X then do Y, else do Z)
- You need to loop (fetch N pages, process N files, retry on failure)

**Use normal tool calls when:**
- Single tool call with no processing needed
- You need to see the full result and apply complex reasoning (the LLM is better at reasoning than the code it writes)
- The task requires human interaction (clarify tool)

### Open questions for implementation

1. **Should the parent block its main thread while the sandbox runs?** Currently `handle_function_call` is synchronous, so yes -- same as any other tool call. For long sandbox runs (up to 120s), the gateway's typing indicator stays active. The agent can't process new messages during this time, but it can't during any long tool call either. Interrupt support (above) handles the "user sends a new message" case.

2. **Should `browser_snapshot` pass `user_task`?** `handle_browser_function_call` accepts `user_task` for task-aware content extraction. The parent should pass the user's original query from the session context when dispatching sandbox browser calls.

3. **Terminal parameter restrictions**: The sandbox version of `terminal` should strip/ignore: `background=True` (no background processes from ephemeral scripts), `check_interval` (gateway-only feature for background watchers), `pty=True` (interactive PTY makes no sense in a script). Only `command`, `timeout`, and `workdir` are passed through.

### Future enhancements (not MVP)

- **Concurrent tool calls via threading**: The script could use `ThreadPoolExecutor` to fetch 5 URLs in parallel. Requires the UDS client to be thread-safe (add a `threading.Lock` around socket send/receive). Significant speedup for I/O-bound workflows like multi-page web extraction.
- **Streaming progress to gateway**: Instead of one notification for the entire run, send periodic progress updates ("execute_code: 3/7 tool calls, 12s elapsed").
- **Persistent sandbox sessions**: Keep the subprocess alive between `execute_code` calls so Python variables carry over. Adds complexity but enables iterative multi-step workflows where the agent refines its script.
- **RL/batch integration**: Atropos RL environments use ToolContext instead of handle_function_call. Would need an adapter so the RPC bridge dispatches through the right mechanism per execution context.
- **Windows support**: If there's demand, fall back to TCP localhost (127.0.0.1:random_port) instead of UDS. Same protocol, different transport. Security concern: localhost port is accessible to other processes on the machine. Could mitigate with a random auth token in the RPC handshake.

### Relationship to other items
- **Subagent Architecture (#1)**: A code sandbox that calls tools IS a lightweight subagent without its own LLM inference. This handles many of the "mechanical multi-step" cases (search+filter, bulk file ops, browser loops) at near-zero LLM cost. Full subagents are still needed for tasks requiring LLM reasoning at each step.
- **Browser automation (#6)**: Biggest win. Browser workflows are 10+ round trips today. A script that navigates, clicks, extracts, paginates in a loop collapses that to 1 LLM turn.
- **Web search**: Directly matches Anthropic's dynamic filtering results.
- **File ops**: Bulk read-search-patch workflows become one call.

**Files:** `tools/code_execution_tool.py` (subprocess management, UDS server, RPC dispatch, hermes_tools generator), tool schema in `model_tools.py`

---

## Implementation Priority Order

### Tier 1: The Knowledge System + Agent Efficiency

These form two parallel tracks. The Knowledge System items depend on each other (build in order). Programmatic Tool Calling is independent and can be built in parallel.

**Track A: The Knowledge System (build in this order):**
1. **Memory System Phase 1** (file-based: memory_summary.md + MEMORY.md) -- #4
   - No infrastructure dependency. Just a new `memory` tool + prompt guidance + file read/write.
   - Gives the agent persistent identity memory (user profile) and declarative memory (facts, preferences).
   - memory_summary.md always in system prompt = immediate value every session.

2. **Agent-Managed Skills** (create/edit/delete + proactive creation) -- #2
   - Depends on: nothing (but better with memory, since the agent understands what it has learned)
   - Gives the agent procedural memory. Combined with memory, the agent now has both "what I know" and "how I do things."
   - The proactive feedback loop ("How was that? Want me to save it as a skill?") is taught via tool description.

3. **Interactive Clarifying Questions** -- #3
   - Makes the feedback loop cleaner with structured choices (especially on messaging platforms).
   - Also useful independently for pre-task clarification ("Which approach?").

4. **SQLite State Store** -- #5
   - Migrate sessions from JSONL to SQLite. Enables fast session search, structured metadata, scales to thousands of sessions.
   - Memory Phase 2 and Learnings depend on this.

5. **Self-Learning from Errors** -- #15
   - Depends on: SQLite (#5) for storage, or fallback to learnings.jsonl
   - The "error memory" layer. Auto-saves error patterns and fixes.

**Track B: Agent Efficiency (independent, build anytime):**
6. **Programmatic Tool Calling** (code-mediated tool use) -- #19
   - No dependency on the Knowledge System. Biggest efficiency win for agent loops.
   - Can be built in parallel with Track A items.

### Tier 2: Scaling & Ecosystem

7. **Subagent Architecture** -- #1
   - Benefits from the Knowledge System (subagents can read memory/skills) but doesn't require it.
   - Partially solved by Programmatic Tool Calling (#19) for mechanical multi-step tasks.
   - Once built, enables periodic memory consolidation (an optional subagent that reviews recent session summaries and updates MEMORY.md/skills).

8. **MCP Support** -- #13
   - Industry standard protocol. Instant access to hundreds of community tool servers.
   - Independent of Knowledge System.

### Tier 3: Quality of Life

9. Permission / Safety System -- #14
10. Local Browser Control via CDP -- #6
11. Layered Context Architecture -- #11 (becomes more important as memory/skills grow in size)
12. Plugin/Extension System (enhanced hooks first) -- #8
13. Evaluation System -- #10

### Tier 4: Nice to Have

14. Heartbeat System -- #18 (useful for periodic memory consolidation once subagents exist)
15. Session Branching / Checkpoints -- #16
16. File Watcher -- #17
17. Signal Integration -- #7
18. Tools Wishlist items -- #12
19. Native Companion Apps -- #9
