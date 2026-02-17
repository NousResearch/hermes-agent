# Hermes Agent - Future Improvements

---

## 1. Subagent Architecture (Context Isolation) 🎯

The main agent becomes an orchestrator that delegates context-heavy tasks to subagents with isolated context. Each subagent returns a summary, keeping the orchestrator's context clean. `delegate_task(goal, context, toolsets=[])` with fresh conversation, limited toolset, task-specific system prompt.

## 2. Planning & Task Management 📋

Task decomposition tool, progress checkpoints after N tool calls, persistent plan storage that survives context compression, failure recovery with replanning.

## 3. Dynamic Skills Expansion 📚

Skill acquisition from successful tasks, parameterized skill templates, skill chaining with dependency graphs.

## 4. Interactive Clarifying Questions ❓

Multiple-choice prompt tool with rich terminal UI. Up to 4 choices + free-text. CLI-only with graceful fallback for non-interactive modes.

## 5. Memory System 🧠

Daily memory logs, long-term curated MEMORY.md, vector/semantic search, pre-compaction memory flush, user profile, learning store for error patterns and discovered fixes. *Inspired by ClawdBot's memory system.*

## 6. Heartbeat System 💓

Periodic agent wake-up that reads HEARTBEAT.md for instructions. Runs inside the main session with full context. Triggers on interval, exec completion, cron events, or manual wake. HEARTBEAT_OK suppression when nothing needs attention. *Inspired by ClawdBot's heartbeat.*

## 7. Local Browser Control via CDP 🌐

Support both local Chrome (via CDP, free) and Browserbase (cloud, paid) as browser backends. Local gives persistent login sessions but lacks CAPTCHA solving.

## 8. Signal Integration 📡

New platform adapter using signal-cli daemon (JSON-RPC HTTP + SSE). Requires Java runtime and phone number registration.

## 9. Session Transcript Search 🔍

`hermes sessions search <query>` CLI command and `session_search` agent tool. Text-based first (ripgrep over JSONL), vector search later.

## 10. Plugin/Extension System 🔌

Python plugin interface with `plugin.yaml` + `handler.py`. Discovery from `~/.hermes/plugins/`. Plugins can register tools, hooks, and CLI commands. *Inspired by ClawdBot's 36-plugin extension system.*

## 11. Native Companion Apps 📱

macOS (Swift/SwiftUI), iOS, Android apps connecting via WebSocket. Prerequisite: WS API on gateway. MVP: web UI with Flask/FastAPI. *Inspired by ClawdBot's companion apps.*

## 12. Evaluation System 📏

LLM grader mode for batch_runner, action comparison against expected tool calls, string matching baselines.

## 13. Layered Context Architecture 📊

Structured hierarchy: project context > skills > user profile > learnings > external knowledge > runtime introspection.

## 14. Tools Wishlist 🧰

- Diagram rendering (Mermaid/PlantUML to images)
- Document generation (PDFs, Word, presentations)
- Canvas / visual workspace
- Coding agent skill (Codex, Claude Code orchestration via PTY)
- Domain skill packs (DevOps, data science, security)
