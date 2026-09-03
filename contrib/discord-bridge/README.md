# Discord bridge — out-of-repo counterparts

The Discord OpenCode bridge (`plugins/platforms/discord/opencode_bridge.py`)
only sees what an agent writes into its spool directory
(`$HERMES_HOME/opencode_bridge/`). The agents themselves are separate
programs, so their side lives here: these files are **copies of what runs
on the machine**, kept in the repo so the two halves stay reviewable
together.

| File | Installed to | Purpose |
|---|---|---|
| `opencode-plugins/befehlswaechter.js` | `~/.config/opencode/plugins/` | Blocks shell commands touching paths outside the project and asks in Discord (allow once / reject) |
| `opencode-plugins/sitzungsthread.js` | `~/.config/opencode/plugins/` | Announces each OpenCode session so the bridge opens one Discord thread per session |
| `opencode-plugins/discord-melder.js.disabled` | `~/.config/opencode/plugins/` (drop `.disabled`) | Optional: posts each finished answer to a channel webhook. Superseded by session threads; kept for reference |
| `claude-code-hooks/discord-bridge.py` | `~/.claude/hooks/` | Claude Code counterpart: forwards its own permission prompts and `AskUserQuestion`, and announces sessions |

## Wire contract

Both sides speak the spool protocol documented at the top of
`opencode_bridge.py`: the agent writes `requests/<id>.json`, the bridge
answers with `decisions/<id>.json` (`once` / `reject` / `answer`), and
`notice` requests (start/prompt/result/child) need no answer at all.

## Fail-closed posture

Access stays blocked on rejection, on timeout, when the gateway is not
running or Discord is not connected, and on any malformed or mismatched
reply. There are no persistent grants: every request covers exactly one
command.

## Claude Code hook registration

```json
"hooks": {
  "PermissionRequest": [{"hooks": [{"type": "command", "command": "python3 ~/.claude/hooks/discord-bridge.py", "timeout": 360}]}],
  "PreToolUse":        [{"hooks": [{"type": "command", "command": "python3 ~/.claude/hooks/discord-bridge.py", "timeout": 360}]}],
  "UserPromptSubmit":  [{"hooks": [{"type": "command", "command": "python3 ~/.claude/hooks/discord-bridge.py", "timeout": 15}]}],
  "Stop":              [{"hooks": [{"type": "command", "command": "python3 ~/.claude/hooks/discord-bridge.py", "timeout": 15}]}]
}
```

The hook only acts when Claude Code was started by Hermes (`HERMES_HOME` in
the environment, a Hermes ancestor process, or `HERMES_CLAUDE_BRIDGE=1`).
Switch it off with `~/.claude/discord-bridge.aus` or
`HERMES_CLAUDE_BRIDGE=aus`.

## Keeping the copies in sync

These are copies, not the live files — edits made in `~/.claude/hooks/` or
`~/.config/opencode/plugins/` do not appear here by themselves. Copy them
over before committing, or symlink the live paths at this directory.
