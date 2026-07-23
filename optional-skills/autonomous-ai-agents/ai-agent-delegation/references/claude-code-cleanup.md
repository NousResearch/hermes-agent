# Claude Code Cleanup (Removing Claude Flow)

When Claude Flow has been installed alongside Claude Code, these are the exact steps to remove it:

## What Claude Flow Added (not native to Claude Code)

| Path | Content | Action |
|------|---------|--------|
| `~/.claude/helpers/` | 41 files (hook handlers, intelligence, learning, swarm, workers) | DELETE entire directory |
| `~/.claude/agents/*` | 98 .md files across 23 subdirectories | DELETE all content, keep empty folder |
| `~/.claude/commands/*` | 88 .md files across 7 subdirectories | DELETE all content, keep empty folder |
| `~/.claude/skills/*` | skill-creator/ (1 file) | DELETE all content, keep empty folder |
| `~/.claude/CLAUDE.md` | Flow-rewritten instructions | DELETE entirely |
| `~/.claude-flow/` | Full Flow installation (agents, hooks, data, metrics, swarm) | DELETE entirely |
| `settings.json` hooks | SessionStart/SessionEnd/PreToolUse/PostToolUse referencing hook-handler.cjs | REMOVE hooks section |
| `settings.json` env | `CLAUDE_FLOW_V3_ENABLED`, `CLAUDE_FLOW_HOOKS_ENABLED` | REMOVE these keys |
| `settings.local.json` | `disabledMcpjsonServers: [ruflo, ruv-swarm, flow-nexus]` | Clear to `{}` |

## What to KEEP (native Claude Code)

- `settings.json`: env vars (ANTHROPIC_BASE_URL, AUTH_TOKEN, MODEL), permissions, effortLevel, maxTokens, autoUpdatesChannel
- All session/project/history/cache directories
- `agents/`, `commands/`, `skills/` directories (empty, ready for your own content)

## Verification After Cleanup

```bash
# Confirm no Flow remnants
find ~/.claude/ -iname "*flow*" -o -iname "*swarm*" -o -iname "*nexus*"
# Should return nothing

# Confirm no Flow npm packages
npm list -g | grep -i "flow\|swarm\|nexus"
# Should return nothing

# Confirm settings.json is clean
python3 -c "import json; s=json.load(open('$HOME/.claude/settings.json')); print('hooks:', s.get('hooks', 'NONE')); print('env:', list(s.get('env',{}).keys()))"
# hooks should be NONE, env should not include CLAUDE_FLOW_* keys
```
