# Claude Code Speedup — Hook Optimization (May 2026)

## Problem
Claude Code felt slow — commands took 24s for simple tasks. Root cause was hook overhead, not API latency.

## Original Config (slow)
9 hook events firing on every interaction:
- `UserPromptSubmit` → route + intelligence.getContext() on every message
- `auto-memory import/sync` on every session start/end
- `SubagentStart`, `SubagentStop`, `PreCompact`, `Notification` hooks

Per-task overhead: **0.591s** (benchmark: pre-bash alone was 158ms)

## Optimized Config (fast)
4 hook events:
- `SessionStart` → session-restore
- `SessionEnd` → session-end
- `PreToolUse` (Bash only) → pre-bash
- `PostToolUse` (Bash + Edit) → post-bash + post-edit

Removed: UserPromptSubmit, auto-memory import/sync, Subagent hooks, PreCompact, Notification.

Per-task overhead: **~0.25s** (saved 340ms/task)

## Settings file
`/mnt/c/Users/<username>/.claude/settings.json` — no longer has UserPromptSubmit/route hook. The `hook-handler.cjs` route handler was also simplified (removed intelligence.getContext() call).

## Files modified
- `~/.claude/settings.json` — hook reduction
- `~/.claude/helpers/hook-handler.cjs` — simplified route handler
- `~/.claude/CLAUDE.md` — updated from ruflo reference to Claude Code integration notes

## Benchmark Command
```bash
cd /mnt/c/Users/<username> && node .claude/helpers/hook-handler.cjs pre-bash
```
Expect: ~70-80ms, not 150ms.

## Key insight
Hook overhead is minor. **The real killer was the M2.7 routing bug** — 24s was API time, not hook time. After fixing `minimax-m2.5-free` → `minimax-m2.5`, task completed in one shot.

## Test task
```
create a file called test.js with a single console.log("hello")
```
Expected: works in one shot, no retries.

## WSL Invocation — Critical for Hermes (May 2026)

`claude` is a Windows binary, not in WSL PATH. Direct `cmd /c claude` fails from bash. Use:

```bash
/mnt/c/Windows/System32/cmd.exe /c "cd /d <WINDOWS_PATH> && claude -p 'task' --output-format json"
```

**Keys:**
- Use `/mnt/c/Windows/System32/cmd.exe` as full path from WSL
- Use `cd /d <WINDOWS_PATH>` not UNC paths
- Direct `-p 'task'` argument, NOT echo/pipes (pipes break prompt passing)
- Use `--output-format json` for parseable output, `--verbose` only when using stream-json