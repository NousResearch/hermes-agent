# Claude Code as Fallback When Codex Hits Usage Limits

## When to Use This

Codex (ChatGPT auth) has a usage limit that resets monthly. When you see:
```
ERROR: You've hit your usage limit. Upgrade to Plus to continue using Codex
```

Switch to Claude Code for the same Rust coding tasks.

## Claude Code Setup (verified 2026-06-24)

- Install: `npm install -g @anthropic-ai/claude-code`
- Auth: `claude` once (browser OAuth for claude.ai Pro/Max)
- Check: `claude auth status` -- should show `loggedIn: true, authMethod: "claude.ai"`
- Version: v2.1.187

## Claude Code Print Mode (PREFERRED for coding tasks)

```bash
cat /tmp/task-spec.md | claude -p "$(cat /tmp/task-spec.md)" \
  --dangerously-skip-permissions --max-turns 30 --model sonnet
```

Key flags:
- `-p` = print mode (non-interactive, exits when done)
- `--dangerously-skip-permissions` = auto-approve all tool use
- `--max-turns 30` = cap agentic loops
- `--model sonnet` = good balance of speed/quality for Rust tasks

## Critical: Do NOT use --bare

`--bare` skips OAuth and fails with "Not logged in". Always use `-p` without `--bare`.

## Claude Code vs Codex Comparison

| Aspect | Claude Code | Codex |
|--------|-------------|-------|
| Auth | claude.ai OAuth | ChatGPT auth |
| Usage limit | No limit observed | Monthly limit (~Jul 23 reset) |
| Print mode | `claude -p` | `codex exec` |
| Auto-approve | `--dangerously-skip-permissions` | `--dangerously-bypass-approvals-and-sandbox` |
| Rust code quality | Compiled correctly first try (both batches) | Upgraded rmcp API but hit limit before features |
| Speed | ~5 min per 2-phase batch | Hit limit before completing |
| Background mode | `terminal(background=true, notify_on_complete=true)` | Same |

## Shared Rules for Both

1. Write task specs to files, pipe via stdin (not inline in shell command)
2. Tell agents NOT to run `cargo test` in worktrees (path deps fail)
3. Write "Out of scope (do NOT touch)" sections listing files other agents own
4. Group tasks by file ownership for parallel safety
5. Controller verifies with `cargo test --all-features` after all agents complete

## Parallel Dispatch Pattern (works for both)

```bash
# Task spec files
/tmp/claude-task-1.md  # touches search.rs, lib.rs
/tmp/claude-task-2.md  # touches http_server.rs, server.rs, tools.rs

# Launch in parallel (different files = no conflict)
terminal(command="cat /tmp/claude-task-1.md | claude -p \"$(cat /tmp/claude-task-1.md)\" --dangerously-skip-permissions --max-turns 30 --model sonnet", background=true, notify_on_complete=true)

terminal(command="cat /tmp/claude-task-2.md | claude -p \"$(cat /tmp/claude-task-2.md)\" --dangerously-skip-permissions --max-turns 30 --model sonnet --add-dir /path/to/mcp-server", background=true, notify_on_complete=true)
```

## When Both Are Available

- Use Codex for parallel batch work or when Claude API is down
- Use Claude Code for Rust coding tasks (no usage limit, compiles correctly)
- Both can run in parallel if touching different files
