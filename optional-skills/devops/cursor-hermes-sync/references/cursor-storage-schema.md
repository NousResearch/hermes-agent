# Cursor Internal Storage Schema

Reference for the Cursor AI IDE's internal data storage, used by the
`cursor-hermes-sync` bridge script to read Cursor's state.

## Cursor AI Tracking DB

**Location:** `~/.cursor/ai-tracking/ai-code-tracking.db`
**Format:** SQLite (WAL mode — concurrent reads are safe while Cursor runs)

### Tables

#### `conversation_summaries`

Stores AI-generated summaries of Cursor agent conversations.

| Column | Type | Description |
|--------|------|-------------|
| `conversationId` | TEXT PRIMARY KEY | UUID of the conversation |
| `title` | TEXT | Auto-generated title |
| `tldr` | TEXT | One-line summary |
| `overview` | TEXT | Longer overview |
| `summaryBullets` | TEXT | JSON array of bullet points |
| `model` | TEXT | Model used (e.g., `claude-4`, `gpt-4`) |
| `mode` | TEXT | Chat mode (`agent`, `build`, `edit`, etc.) |
| `updatedAt` | INTEGER | Millisecond timestamp |

**Note:** This table only has summaries, NOT full message content. For full
messages, see the JSONL transcripts below.

#### `ai_code_hashes`

Tracks AI-generated code by content hash for provenance tracking.

| Column | Type | Description |
|--------|------|-------------|
| `hash` | TEXT PRIMARY KEY | Content hash |
| `source` | TEXT | Origin (agent, inline, etc.) |
| `fileExtension` | TEXT | File type |
| `fileName` | TEXT | File name |
| `requestId` | TEXT | API request ID |
| `conversationId` | TEXT | Links to conversation_summaries |
| `timestamp` | INTEGER | ms timestamp |
| `model` | TEXT | Model used |
| `createdAt` | INTEGER | Creation timestamp |

#### `scored_commits`

AI vs human code contribution tracking per commit.

| Column | Type | Description |
|--------|------|-------------|
| `commitHash` | TEXT | Git commit hash |
| `branchName` | TEXT | Branch name |
| `scoredAt` | INTEGER | When scored |
| `linesAdded` | INTEGER | Total lines added |
| `linesDeleted` | INTEGER | Total lines deleted |
| `tabLinesAdded` | INTEGER | Tab-completion lines |
| `composerLinesAdded` | INTEGER | Composer (agent) lines |
| `humanLinesAdded` | INTEGER | Human-typed lines |
| `v1AiPercentage` | TEXT | Legacy AI % |
| `v2AiPercentage` | TEXT | Current AI % |

#### `tracking_state`

Key-value store for tracking state.

#### `tracked_file_content`

Stores content of tracked files by hash.

#### `ai_deleted_files`

Tracks AI-deleted files.

## Cursor Agent Transcripts

**Location:** `~/.cursor/projects/<encoded-path>/agent-transcripts/`
**Format:** JSONL files (one per conversation/subagent)

### Path encoding

Cursor encodes project paths by replacing `/` with `-` and stripping the
leading `-`. Example:
- `/Volumes/Work/code/EM` → `Volumes-Work-code-EM`

### JSONL message format

Each line in a transcript file is a JSON object with this structure:

```json
{
  "role": "user" | "assistant" | "system",
  "message": {
    "content": [
      {
        "type": "text" | "tool_use" | "tool_result",
        "text": "...",          // for text type
        "name": "Read|Shell|...",// for tool_use
        "input": { ... }         // for tool_use
      }
    ]
  }
}
```

Special line types:
```json
{"type": "turn_ended", "status": "success" | "error"}
```

### Directory structure

```
~/.cursor/projects/<encoded-path>/
  agent-transcripts/
    <conversation-uuid>/
      subagents/
        <subagent-uuid>.jsonl
        <subagent-uuid>.jsonl
  agent-tools/
  canvases/
  mcps/
  terminals/
```

The main conversation transcript may be at the conversation-UUID directory
level (not inside subagents/). Subagent transcripts are spawned by agent
tools that create sub-conversations.

## Cursor MCP Server Config

**Location:** `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (workspace)

### Format

```json
{
  "mcpServers": {
    "server-name": {
      "command": "node",
      "args": ["/path/to/server.js"],
      "env": { "KEY": "value" }
    },
    "remote-server": {
      "url": "http://localhost:8082/sse"
    }
  }
}
```

After modifying this file, Cursor needs to be restarted to pick up changes.

## Cursor CLI

**Location:** `/Applications/Cursor.app/Contents/Resources/app/bin/cursor`

The `cursor` binary is a VS Code fork. Key flags:
- `cursor <folder>` — open a folder
- `cursor -n <folder>` — open in new window
- `cursor -g <file:line:col>` — go to file:line
- `cursor --chat` — open standalone chat window

On macOS, the binary is not on `$PATH` by default. The bridge script checks
both `$PATH` and the app bundle location.

## Version

Schema extracted from Cursor 3.15.1 (August 2026). Future versions may
add columns or tables. The bridge script uses `SELECT *` and is resilient
to new columns.