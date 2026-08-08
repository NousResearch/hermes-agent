# Architecture: Hermes ↔ Cursor Two-Way Sync

How the pieces fit together. Read this before extending the skill.

## Component map

```
┌─────────────────────────────────────────────────────────────────┐
│                        Hermes Agent                              │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ Hermes CLI │  │ Hermes Kanban │  │ Hermes Sessions (state.db)│ │
│  │ hermes -z   │  │ hermes kanban │  │ ~/.hermes/state.db        │ │
│  └─────┬──────┘  └──────┬───────┘  └─────────┬─────────────────┘ │
│        │                 │                     │                  │
│        │    ┌────────────┴────────────┐        │                  │
│        │    │  cursor_kanban_lane.py   │        │                  │
│        │    │  (custom dispatcher)     │        │                  │
│        │    └────────────┬────────────┘        │                  │
│        │                 │                     │                  │
│  ┌─────┴─────────────────┴─────────────────────┴──────────────┐   │
│  │              cursor_hermes_bridge.py                       │   │
│  │  init | status | sync | handoff | watch | changes |         │   │
│  │  open-cursor | open-hermes | create-project | list-projects │   │
│  │  cursor-chats | cursor-transcripts | hermes-sessions |     │   │
│  │  add-mcp | list-mcp | remove-mcp | remove                   │   │
│  └─────────────────────────┬──────────────────────────────────┘   │
│                            │                                      │
└────────────────────────────┼──────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Shared Files     │
                    │  AGENTS.md       │ ← both agents read this
                    │  .cursor/rules/  │ ← Cursor reads this
                    │  .agent-sync/    │ ← sync state (gitignored)
                    └────────┬────────┘
                             │
┌────────────────────────────┼──────────────────────────────────────┐
│                        Cursor AI IDE                              │
│  ┌───────────────┐  ┌──────┴───────┐  ┌────────────────────────┐  │
│  │ Cursor CLI    │  │ Cursor MCP   │  │ Cursor Storage          │  │
│  │ /Applications/ │  │ ~/.cursor/   │  │ ai-tracking.db (SQLite) │  │
│  │ Cursor.app/... │  │ mcp.json    │  │ projects/<enc>/         │  │
│  └───────────────┘  └──────────────┘  │ agent-transcripts/*.jsonl│  │
│                                        └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Data flows

### 1. Context sync (Hermes → Cursor)

```
Hermes writes AGENTS.md → Cursor reads AGENTS.md on session start
         │                           │
         └─ handoff --to cursor      └─ .cursor/rules/agent-sync.mdc
            writes handoff note        tells Cursor to read state.json
```

### 2. Context sync (Cursor → Hermes)

```
Cursor edits files → Hermes reads files via file tools
         │              │
         └─ watch        └─ changes command shows what changed
            records to      since last Hermes session
            changes.jsonl
```

### 3. Kanban task dispatch (Hermes → Cursor)

```
hermes kanban create "Task" --assignee cursor
         │
         └─ cursor_kanban_lane.py tick/daemon
            ├─ finds ready @cursor tasks
            ├─ atomically claims (ready → running)
            ├─ creates workspace
            ├─ writes AGENTS.md with task context
            ├─ opens Cursor on workspace
            └─ writes kanban comment with PID
```

### 4. Kanban task completion (Cursor → Hermes)

```
User finishes in Cursor
         │
         └─ hermes kanban complete <task_id> --summary "Done"
            └─ task status: running → done
               dependents become ready
```

### 5. MCP bridge (Cursor → Hermes tools)

```
add-mcp --name "Hermes Bridge" --command python3 --args /path/to/server.py
         │
         └─ writes to ~/.cursor/mcp.json
            └─ Cursor restart picks up the MCP server
               └─ Cursor's agent can now call Hermes tools
                  (web search, terminal, memory, file ops)
```

### 6. Transcript reading (Cursor → Hermes)

```
cursor-transcripts <project>
         │
         └─ finds ~/.cursor/projects/<encoded>/agent-transcripts/
            ├─ reads JSONL files
            ├─ extracts user/assistant messages
            └─ shows conversation history (with --full for complete text)
```

## File layout

```
~/.hermes/skills/devops/cursor-hermes-sync/
├── SKILL.md                          # Main skill instructions
├── references/
│   ├── cursor-storage-schema.md      # Cursor's SQLite + JSONL + MCP formats
│   ├── sync-state-schema.md          # .agent-sync/ and registry formats
│   └── architecture.md               # This file
└── scripts/
    ├── cursor_hermes_bridge.py       # Main bridge script
    ├── cursor_kanban_lane.py          # Kanban worker lane for Cursor
    ├── test_bridge.py                # 62 tests for the bridge
    └── test_kanban_lane.py           # 45 tests for the lane
```

## Design principles

1. **Filesystem is the sync layer** — both agents work on the same repo. Git is the real version control. The bridge adds semantic tracking (what changed, when, why) on top.

2. **AGENTS.md is the shared context** — both agents read it natively. No proprietary protocol. Edit the file and both agents see it.

3. **Kanban owns the task lifecycle** — the lane script only handles `ready → running`. Completion and blocking are done from Hermes CLI. The Kanban kernel is the source of truth.

4. **Registry over filesystem scanning** — the project registry (`projects.json`) is the authoritative source for project names, like Dyad's `apps` table. Filesystem scanning is a fallback for unregistered projects.

5. **Auto-watch on handoff** — the bridge automatically records file changes before switching agents, so the receiving agent always knows what changed.