# Claude Flow Architecture

Discovered at `/mnt/c/Users/<username>/.claude/` (May 2026). Claude Flow is an open-source multi-agent orchestration layer that sits ON TOP of Claude Code. It's NOT the Claude Code binary itself — it's configuration, hooks, agent definitions, and intelligence that orchestrate how Claude Code's agents operate.

## Layer Model

```
┌─────────────────────────────────────────┐
│  Claude Flow (config + orchestration)   │  ← What lives at ~/.claude/
│  hooks, agents, intelligence, sessions  │
├─────────────────────────────────────────┤
│  Claude Code (proprietary binary)       │  ← @anthropic-ai/claude-code npm
│  Bash, Read, Write, Edit tools          │
│  Agentic loop, permission system        │
└─────────────────────────────────────────┘
```

## Core Files

### Hook System (`helpers/hook-handler.cjs`)
Dispatches events to handlers. Hooks fire at these points:
- `pre-bash` — command safety validation (blocks `rm -rf /`, format commands)
- `post-edit` — records edit for session metrics + intelligence tracking
- `session-restore` — restores previous session state, initializes intelligence graph
- `session-end` — consolidates intelligence (PageRank recompute), persists state
- `pre-task` / `post-task` — task lifecycle tracking
- `route` — routes tasks to optimal agent via keyword regex

All hooks must exit 0 (Claude Code treats non-zero as hook error and skips subsequent hooks).

### Task Router (`helpers/router.js`)
Simple keyword-based routing. Maps regex patterns to agent names:
- `implement|create|build|add|write code` → `coder`
- `test|spec|coverage` → `tester`
- `review|audit|check|security` → `reviewer`
- `research|find|search|documentation` → `researcher`
- `design|architect|structure|plan` → `architect`
- `api|endpoint|server|backend|database` → `backend-dev`
- `ui|frontend|component|react|css` → `frontend-dev`
- `deploy|docker|ci|cd|pipeline` → `devops`

Default: `coder` with 0.5 confidence.

### Session Manager (`helpers/session.js`)
Tracks session lifecycle: start → restore → metrics → end.
Stores in `.claude-flow/sessions/current.json`.
Metrics tracked: edits, commands, tasks, errors.

### Memory (`helpers/memory.js`)
Simple key-value store at `.claude-flow/data/memory.json`.
Commands: get, set, delete, clear, keys.

### Intelligence Layer (`helpers/intelligence.cjs`) — 858 lines
The most sophisticated component. Implements:
- **PageRank** over a memory graph (nodes = entries, edges = co-occurrence)
- **Trigram similarity** for finding related memories (Jaccard similarity)
- **Content-hash deduplication** (FNV-1a fingerprinting)
- **Session state** integration (reads/writes session context)
- **Pending insights** log (append-only JSONL)
- Safety limits: 10MB max file size, 5000 max graph nodes

Data files (under `.claude-flow/data/`):
- `auto-memory-store.json` — accumulated memory entries
- `graph-state.json` — serialized graph (nodes + edges + PageRank scores)
- `ranked-context.json` — pre-computed ranked entries for fast lookup
- `pending-insights.jsonl` — append-only edit/task log

## Agent Definitions (`agents/`)

Five core agents, each with a rich system prompt:

| Agent | File | Role |
|-------|------|------|
| Coder | `agents/core/coder.md` | Implementation: SOLID, DRY, TDD, error handling |
| Planner | `agents/core/planner.md` | Task decomposition, dependency mapping, risk assessment |
| Researcher | `agents/core/researcher.md` | Code analysis, pattern recognition, documentation review |
| Tester | `agents/core/tester.md` | Unit/integration/E2E testing, edge cases, security testing |
| Reviewer | `agents/core/reviewer.md` | Code quality, security audit, performance analysis |

Additional agents in `agents/`: architecture, browser, consensus, development, devops, documentation, github, goal, optimization, specialized, swarm, testing, templates.

## SPARC Methodology (`agents/sparc/`)

5-phase structured development:
1. **Specification** — requirements analysis, constraints, acceptance criteria
2. **Pseudocode** — algorithm flow, data structures, edge case handling
3. **Architecture** — module design, component boundaries, tech selection
4. **Refinement** — implement, test, iterate, refactor
5. **Completion** — verify, document, deliver

Each phase has its own agent definition in `agents/sparc/`.

## Hooks Configuration (in `settings.json`)

```json
{
  "hooks": {
    "SessionStart": [{"hooks": [{"type": "command", "command": "...session-restore"}]}],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "...session-end"}]}],
    "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "...pre-bash"}]}],
    "PostToolUse": [
      {"matcher": "Bash", "hooks": [{"type": "command", "command": "...post-bash"}]},
      {"matcher": "Write|Edit|MultiEdit", "hooks": [{"type": "command", "command": "...post-edit"}]}
    ]
  }
}
```

## Current Settings (as of May 2026)

- Model: `claude-opus-4-8-20260501` (env overrides to `claude-sonnet-4-6-20260101` via proxy)
- Proxy: `https://ai-beta-five-25.vercel.app/`
- maxTokens: 8096
- Permissions: Bash, Write(*), Read(*) all allowed (defaultMode: dontAsk)
- Experimental: agent teams, flow v3, flow hooks all enabled

## Key Differences from Hermes's Native Approach

| Aspect | Claude Flow | Hermes |
|--------|-------------|--------|
| Memory | PageRank graph + trigram similarity | FTS5 search + built-in memory + holographic |
| Routing | Keyword regex → agent name | Skill-based (71 skills with triggers) |
| Sessions | JSON file in `.claude-flow/sessions/` | SQLite with FTS5 |
| Intelligence | Graph-based with PageRank | Memory tool + session_search + skills |
| Hooks | Shell commands fired by Claude Code | No equivalent (agent decides in-loop) |
| Agents | Markdown system prompts | Skill documents loaded on demand |
