# Claude Code Directory Origin Map

Which files/folders in `~/.claude/` are native Claude Code vs added by Claude Flow.

## Native Claude Code (created/maintained by Claude Code itself)

```
~/.claude/
├── settings.json              Config (but MODIFIED by Claude Flow — see below)
├── settings.local.json        Local overrides (MODIFIED — Flow disabled 3 MCP servers)
├── CLAUDE.md                  Global memory/instructions (MODIFIED — rewritten by Flow)
├── .last-cleanup              Internal cleanup timestamp
├── .last-update-result.json   Auto-update result log
├── history.jsonl              Command/prompt history
├── sessions/                  Session transcripts
├── projects/                  Per-project settings (8 project dirs)
├── session-env/               Session environment snapshots
├── shell-snapshots/           Shell state captures
├── paste-cache/               Clipboard/paste buffer
├── file-history/              File edit history
├── backups/                   Config backups
├── cache/                     Cache (changelog.md)
├── downloads/                 Downloads (empty)
├── ide/                       IDE integration (empty)
├── agents/                    DIRECTORY is native (Claude Code supports custom agents)
├── commands/                  DIRECTORY is native (Claude Code supports slash commands)
└── skills/                    DIRECTORY is native (Claude Code supports custom skills)
```

## Added by Claude Flow (all content in native directories)

### agents/ — 23 subdirectories, ~98 .md files, 1 .yaml

| Directory     | Content                              | Useful? |
|---------------|--------------------------------------|---------|
| analysis/     | Code quality analysis agents         | Low     |
| architecture/ | System design agents                 | Low     |
| browser/      | Browser automation (browser-agent.yaml) | Low  |
| consensus/    | Distributed consensus protocols      | Skip    |
| **core/**     | **5 core agents: coder, planner, researcher, reviewer, tester** | **YES** |
| custom/       | Custom test agents                   | Skip    |
| data/         | ML/data model agents                 | Skip    |
| development/  | Backend API agents                   | Low     |
| devops/       | CI/CD agents                         | Low     |
| documentation/| API docs agents                      | Skip    |
| flow-nexus/   | Flow Nexus platform agents (9 files) | Skip    |
| github/       | GitHub workflow agents (13 files)    | Low     |
| goal/         | Goal-oriented planning agents        | Low     |
| optimization/ | Performance/topology agents          | Skip    |
| payments/     | Agentic payments agent               | Skip    |
| sona/         | Sona learning optimizer              | Skip    |
| sparc/        | SPARC methodology agents (4 files)   | Low     |
| specialized/  | Mobile/React Native agents           | Skip    |
| sublinear/    | Sublinear algorithm agents           | Skip    |
| swarm/        | Swarm coordination agents (3 files)  | Skip    |
| templates/    | Agent template generators (9 files)  | Skip    |
| testing/      | TDD/production validation agents     | Low     |
| v3/           | V3 architecture agents (15 files)    | Skip    |

### commands/ — 7 subdirectories, ~88 .md files

| Directory     | Content                              |
|---------------|--------------------------------------|
| analysis/     | Performance/token analysis           |
| automation/   | Self-healing, smart-spawn            |
| github/       | PR/issue/release swarm commands      |
| hooks/        | Pre/post hook lifecycle              |
| monitoring/   | Agent metrics, swarm monitor         |
| optimization/ | Topology, cache, parallel execution  |
| sparc/        | Full SPARC mode commands (26 files)  |
| claude-flow-help.md   | Flow help                   |
| claude-flow-memory.md | Flow memory                 |
| claude-flow-swarm.md  | Flow swarm                  |

### helpers/ — ALL 41 files are Claude Flow

```
hook-handler.cjs      Central hook dispatcher (264 lines)
intelligence.cjs      PageRank memory graph (1031 lines)
statusline.cjs        Status bar renderer (834 lines)
learning-service.mjs  SQLite+HNSW pattern learning (1144 lines)
auto-memory-hook.mjs  Memory bridge (368 lines)
metrics-db.mjs        Metrics SQLite DB (488 lines)
memory.js             Key-value JSON store
router.js             Agent type router
session.js            Session lifecycle manager
statusline.js         Older statusline version
github-safe.js        GitHub CLI injection prevention
+ 30 shell scripts    Swarm, hooks, checkpoints, security, workers
README.md             "Claude Flow V3 Helpers"
pre-commit            Git hook (calls @claude-flow/cli)
post-commit           Git hook (calls @claude-flow/cli)
```

## settings.json Modifications by Claude Flow

```json
{
  "env": {
    "CLAUDE_FLOW_V3_ENABLED": "true",        // Added by Flow
    "CLAUDE_FLOW_HOOKS_ENABLED": "true"       // Added by Flow
    // ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_MODEL
    // — these are user config, not Flow-specific
  },
  "hooks": {
    // ALL hooks reference .claude/helpers/hook-handler.cjs — Flow-added
    "SessionStart": [...],
    "SessionEnd": [...],
    "PreToolUse": [...],
    "PostToolUse": [...]
  },
  "effortLevel": "xhigh",                     // Could be user or Flow
  "maxTokens": 8096                            // Could be user or Flow
}
```

## settings.local.json Modifications by Claude Flow

```json
{
  "disabledMcpjsonServers": ["ruflo", "ruv-swarm", "flow-nexus"]
  // These are Flow-installed MCP servers that were disabled
}
```

## CLAUDE.md Modification by Claude Flow

Content rewritten to instruct Claude Code to use OpenCode Zen proxy with minimax-m2.5 model. Original Claude Code CLAUDE.md would typically contain project-specific instructions, not API routing.
