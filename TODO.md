# Hermes Agent - Future Improvements

> Ideas for enhancing the agent's capabilities, generated from self-analysis of the codebase.

---

## 🚨 HIGH PRIORITY - Immediate Fixes

These items need to be addressed ASAP:

### 1. SUDO Breaking Terminal Tool 🔐
- [ ] **Problem:** SUDO commands break the terminal tool execution
- [ ] **Fix:** Handle password prompts / TTY requirements gracefully
- [ ] **Options:**
  - Configure passwordless sudo for specific commands
  - Detect sudo and warn user / request alternative approach
  - Use `sudo -S` with stdin handling if password can be provided securely

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

## 1. Context Management

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

## 2. Self-Reflection & Course Correction 🔄

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

## 3. Tool Composition & Learning 🔧

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

## 4. Dynamic Skills Expansion 📚

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

## 5. Task Continuation Hints 🎯

**Problem:** Could be more helpful by suggesting logical next steps.

**Ideas:**
- [ ] **Suggest next steps** - At end of a task, suggest logical continuations:
  - "Code is written. Want me to also write tests / docs / deploy?"
  - Based on common workflows for task type
  - Non-intrusive, just offer options

**Files to modify:** `run_agent.py`, response generation logic

---

## 6. Uncertainty & Honesty Calibration 🎚️

**Problem:** Sometimes confidently wrong. Should be better calibrated about what I know vs. don't know.

**Ideas:**
- [ ] **Source attribution** - Track where information came from:
  - "According to the docs I just fetched..." vs "From my training data (may be outdated)..."
  - Let user assess reliability themselves

- [ ] **Cross-reference high-stakes claims** - Self-check for made-up details:
  - When stakes are high, verify with tools before presenting as fact
  - "Let me verify that before you act on it..."

**Files to modify:** `run_agent.py`, response generation logic

---

## 7. Resource Awareness & Efficiency 💰

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

## 8. Collaborative Problem Solving 🤝

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

## 9. Project-Local Context 💾

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

## 10. Graceful Degradation & Robustness 🛡️

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

## 11. Tools & Skills Wishlist 🧰

*Things that would need new tool implementations (can't do well with current tools):*

### High-Impact

- [ ] **Audio/Video Transcription** 🎬
  - Transcribe audio files, podcasts, YouTube videos
  - Extract key moments from video
  - Currently blind to multimedia content
  - *Could potentially use whisper via terminal, but native tool would be cleaner*
  
- [ ] **Diagram Rendering** 📊
  - Render Mermaid/PlantUML to actual images
  - Can generate the code, but rendering requires external service or tool
  - "Show me how these components connect" → actual visual diagram

### Medium-Impact

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

## Priority Order (Suggested)

1. **Memory & Context Management** - Biggest impact on complex tasks
2. **Self-Reflection** - Improves reliability and reduces wasted tool calls  
3. **Project-Local Context** - Practical win, keeps useful info across sessions
4. **Tool Composition** - Quality of life, builds on other improvements
5. **Dynamic Skills** - Force multiplier for repeated tasks

---

## Removed Items (Unrealistic)

The following were removed because they're architecturally impossible:

- ~~Proactive suggestions / Prefetching~~ - Agent only runs on user request, can't interject
- ~~Session save/restore across conversations~~ - Agent doesn't control session persistence
- ~~User preference learning across sessions~~ - Same issue
- ~~Clipboard integration~~ - No access to user's local system clipboard
- ~~Voice/TTS playback~~ - Can generate audio but can't play it to user
- ~~Set reminders~~ - No persistent background execution

The following were removed because they're **already possible**:

- ~~HTTP/API Client~~ → Use `curl` or Python `requests` in terminal
- ~~Structured Data Manipulation~~ → Use `pandas` in terminal
- ~~Git-Native Operations~~ → Use `git` CLI in terminal
- ~~Symbolic Math~~ → Use `SymPy` in terminal
- ~~Code Quality Tools~~ → Run linters (`eslint`, `black`, `mypy`) in terminal
- ~~Testing Framework~~ → Run `pytest`, `jest`, etc. in terminal
- ~~Translation~~ → LLM handles this fine, or use translation APIs

---

*Last updated: $(date +%Y-%m-%d)* 🤖
