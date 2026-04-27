---
name: claude-code-sdk
description: "Programmatic Claude Code via the official claude-agent-sdk Python package. Persistent multi-turn sessions via session resume, parallel orchestration across projects, and per-query cost tracking. Complements the existing claude-code skill for SDK-based use cases."
version: 1.0.0
author: Raghu Thiyagharajan (0xRaghu)
license: MIT
metadata:
  hermes:
    tags: [Coding-Agent, Claude, Anthropic, SDK, Sessions, Concurrency, Cost-Tracking]
    related_skills: [claude-code, codex, opencode, hermes-agent]
---

# Claude Code SDK, Programmatic Hermes Orchestration

Delegate coding work to Claude Code through the official [`claude-agent-sdk`](https://github.com/anthropics/claude-agent-sdk-python) Python package, with a path-based session model that survives across separate Hermes `terminal` invocations. This skill complements the existing `claude-code` skill (CLI plus tmux), which orchestrates the `claude` binary as a TUI process. Both skills share the same underlying Claude Code engine; pick whichever orchestration shape fits the task.

## When to Use This Skill vs. claude-code

| Aspect | Existing `claude-code` | This skill (`claude-code-sdk`) |
|---|---|---|
| Transport | `claude` CLI plus tmux | `ClaudeSDKClient` (Python subprocess) |
| Multi-turn context | tmux PTY snapshots and pane capture | SDK session resume by ID |
| Concurrency | One tmux session per orchestration | Multiple parallel SDK sessions, no shared TUI |
| Streaming | Pane polling (`tmux capture-pane`) | Native async via `client.receive_response()` |
| Cost tracking | Read `total_cost_usd` from `--output-format json` | Captured from `ResultMessage` per query, logged |
| Cleanup | Manual `tmux kill-session` or `/exit` | Drop the session record (no live process between calls) |
| Heaviness | tmux server plus PTY plus REPL | Single Python subprocess per query, no daemon |
| Best for | Interactive multi-step work, slash commands, watching Claude live | Programmatic batch flows, parallel repos, cost-aware automation |

Both skills coexist. Reach for `claude-code-sdk` when the orchestration is fundamentally programmatic (parallel work, scripted budgets, multi-repo) and reach for `claude-code` when an interactive REPL or tmux pane gives you something the SDK does not.

## Prerequisites

- **`claude-agent-sdk`** Python package, version 0.1.68 or newer:

  ```
  pip install --upgrade claude-agent-sdk
  ```

  Note: `claude-code-sdk` (without "agent") is the predecessor package and is **deprecated**. The Anthropic migration guide is at https://docs.claude.com/en/api/agent-sdk/migration-guide.

- **Claude Code CLI** installed (the SDK wraps it):

  ```
  npm install -g @anthropic-ai/claude-code
  ```

- **Authentication** for the `claude` CLI. Either run `claude auth login` once for OAuth (Pro, Max, Enterprise) or set `ANTHROPIC_API_KEY` in your environment for API-key billing. Confirm with `claude auth status --text`.

- **Hermes terminal toolset** enabled. All commands below are issued via `terminal(command="...")` from a Hermes turn.

## Quick Reference

| Command | Purpose | Example |
|---|---|---|
| `open <project_path>` | Register a session for a project directory. Returns `{"session_id": "..."}`. | `open ~/code/my-api` |
| `query <handle> <message>` | Send one message. Resumes prior context if any. Returns text plus per-query cost. | `query 7c3a... "Refactor auth.py to use JWT"` |
| `list` | List active sessions with last activity, message count, and cumulative cost. | `list` |
| `close <handle>` | Drop the session record (idempotent). | `close 7c3a...` |
| `costs <handle>` | Sum cost.log entries for a session and compare against the in-store total. | `costs 7c3a...` |

The `<handle>` returned by `open` is the value you pass to all subsequent commands. It is a 12-char hex token managed by this skill and is independent of Claude's own internal session UUID (which is captured automatically and stored alongside the handle).

## Three Orchestration Patterns

### Pattern 1: Multi-turn refactor with persistent context

Open one session against a project, then run an analyse, refactor, verify cycle. Each step is its own Hermes turn, but the SDK resumes the conversation, so Claude remembers everything from prior turns.

```
terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py open /path/to/project", timeout=30)
# -> {"session_id": "7c3a91fb22d4", "project_path": "/path/to/project"}

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py query 7c3a91fb22d4 'Read src/auth.py and summarise the current authentication strategy in 3 bullet points.'", timeout=180)
# -> {"text": "...", "cost_usd": 0.04, "total_cost_usd": 0.04, "message_count": 1}

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py query 7c3a91fb22d4 'Refactor src/auth.py to use JWT tokens instead of session cookies. Keep the public API identical.'", timeout=300)
# Claude already knows the current strategy from the prior turn

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py query 7c3a91fb22d4 'Now run pytest and report any failures.'", timeout=300)

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py close 7c3a91fb22d4", timeout=10)
```

The conversation context survives across these four independent OS processes because the SDK resumes by Claude's session ID under the hood.

### Pattern 2: Parallel sessions across multiple projects

Open one session per repo and run them in parallel from separate Hermes turns. Each session has its own `cwd`, its own resumed history, and its own cost meter.

```
# Open three sessions
terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py open /repos/api", timeout=30)
terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py open /repos/web", timeout=30)
terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py open /repos/infra", timeout=30)

# Query each independently
terminal(command="python ... query <handle_api> 'Audit endpoints for missing input validation.'", timeout=300)
terminal(command="python ... query <handle_web> 'Audit React components for missing error boundaries.'", timeout=300)
terminal(command="python ... query <handle_infra> 'Audit Terraform for hard-coded credentials.'", timeout=300)

# Inspect all sessions
terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py list", timeout=10)

# Close them
terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py close <handle_api>", timeout=10)
# ...
```

There is no shared tmux pane to manage and no PTY to monitor. Each `query` is one `claude` subprocess that exits when the answer is in.

### Pattern 3: Cost-aware budgeting

Every `query` returns a `cost_usd` for that call and a cumulative `total_cost_usd` for the session. Watch these and stop when you cross a threshold. Combine with `--max-turns` (passed through `ClaudeAgentOptions.max_turns`) for a per-call ceiling.

```
terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py open /repos/big-monorepo", timeout=30)
# -> {"session_id": "abc123..."}

# After each query, check the running total
terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py query abc123 'Investigate slow query in lib/db/orders.ts.'", timeout=300)
# Inspect total_cost_usd in the response. If it exceeds your budget, stop.

# Detailed audit
terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py costs abc123", timeout=10)
# -> {"total_cost_usd": 0.42, "query_count": 7, "tracked_in_store": 0.42}
```

Per-query costs go to `~/.hermes/skills/claude-code-sdk/cost.log` (tab-separated `timestamp\thandle\tcost`). The `costs` subcommand sums that log per handle and returns the same number that the in-memory store tracks. Disagreement between the two surfaces a bug, so this command is also a quick health check.

## Procedure

The standard end-to-end shape:

1. `open <project_path>` to mint a handle. The handle is a short hex token; record it in your turn.
2. `query <handle> <message>` for each turn. The first call has no prior session to resume from; subsequent calls automatically resume Claude's session.
3. Optionally call `list` between turns to see all live sessions, their cumulative costs, and last activity.
4. Call `costs <handle>` when you want a clean per-session cost summary.
5. Call `close <handle>` when done. This drops the record from `~/.hermes/skills/claude-code-sdk/sessions.json`.

State on disk:

- `~/.hermes/skills/claude-code-sdk/sessions.json`: the session store. Keys are handles, values are records (`project_path`, `created_at`, `last_activity`, `message_count`, `total_cost_usd`, `claude_session_id`).
- `~/.hermes/skills/claude-code-sdk/cost.log`: append-only log of per-query costs.

### Idle TTL and what it actually means

Idle records are reaped on every CLI invocation. Default TTL is **3600 seconds (1 hour)**, override with `HERMES_CLAUDE_SDK_IDLE_TTL` (seconds). Override the per-query timeout with `HERMES_CLAUDE_SDK_QUERY_TIMEOUT` (default 300 seconds).

The TTL is purely local bookkeeping. When a record is reaped:

- The record is removed from `sessions.json` so the handle is no longer addressable through this skill.
- **No process is killed.** Each `query` already opens and closes its own `claude` subprocess; nothing lives between calls.
- **The underlying Claude conversation is NOT deleted.** Claude Code persists session state under its own state directory, indexed by the Claude session UUID. The data is still on disk; the user has just lost the short handle that pointed to it.

The TTL has **no relationship** to Claude's subscription billing windows (the 5-hour rolling quota for Pro/Max plans). That window is a server-side quota concept and does not expire conversations.

Why 1 hour was chosen as the default:

- Short enough that abandoned handles do not accumulate forever in `sessions.json`.
- Long enough to survive normal interruptions: a coffee break, a meeting, a context switch to another task.
- Aligns with the typical "human idle" threshold; sub-hour idleness usually means the user stepped away briefly, beyond an hour usually means they moved on.
- Independent of Claude's 5-hour subscription window so the two mechanisms are not entangled. If you want behaviour that mirrors that window, set `HERMES_CLAUDE_SDK_IDLE_TTL=18000`.

## Pitfalls

1. **Use `claude-agent-sdk`, not `claude-code-sdk`.** The latter is the deprecated predecessor and will not provide the `session_id` and `resume` options used here. Migration guide: https://docs.claude.com/en/api/agent-sdk/migration-guide.

2. **First-query cost is higher than steady state.** A new session pays for system-prompt cache creation plus initial project context. In testing, the first one-word reply ran ~$0.20, while subsequent queries on the same resumed session ran $0.03 to $0.05. Plan budgets accordingly.

3. **`total_cost_usd` is an estimate and may be `None`.** The official SDK docs note this. The skill defensively logs only when the value is present and treats absence as "unknown" rather than zero. Do not display a fake $0.00 to the user when the SDK returns `None`.

4. **Sessions resume from a single `cwd` only.** The `project_path` is fixed at `open` time and reused for every `query` on that handle. You cannot re-target a handle to a different directory mid-session. If you need a different `cwd`, open a new handle.

5. **Concurrent queries on the same handle conflict.** Each `query` spawns a `claude` subprocess that holds the session. Running two queries on the same handle in parallel from two Hermes turns is unsupported. Open separate handles for parallel work, even on the same project.

6. **Set `max_turns` for runaway protection.** Without it, Claude can iterate through many tool-use loops on a complex query and rack up cost. To bound this, modify `_run_query` to pass `max_turns=N` into `ClaudeAgentOptions`, or wrap your query with explicit "stop after N steps" instructions. The SDK also exposes `max_budget_usd` for hard cost ceilings.

7. **The reaper runs on every invocation, not in the background.** An idle session past the TTL (default 1 hour) is dropped on the next `open`, `query`, `list`, or `close` call, not by a daemon. If your Hermes turn is silent for longer than the TTL, the next call cleans up. See "Idle TTL and what it actually means" above for the full semantics.

8. **The SDK's `query()` async iterator is an alternative to `ClaudeSDKClient`.** This skill uses `ClaudeSDKClient` because it carries forward more cleanly to richer features (tool callbacks, hooks). For a one-shot stateless call, the `query()` iterator is lighter; both honour the `resume` option identically.

9. **Skill state lives under `~/.hermes/skills/claude-code-sdk/`.** If you have multiple Hermes profiles or want a sandboxed location, set `HERMES_CLAUDE_SDK_STATE_DIR` to override.

## Verification

Run this from a Hermes turn to confirm the skill works end to end:

```
terminal(command="python -c 'import claude_agent_sdk; print(claude_agent_sdk.__version__)'", timeout=10)
# Expect: 0.1.68 or newer

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py --help", timeout=10)
# Expect: usage line listing open, query, list, close, costs

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py open /tmp", timeout=10)
# Expect: {"session_id": "<12-hex>", "project_path": "/private/tmp"}

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py query <handle> 'Reply only with the word OK.'", timeout=120)
# Expect: text "OK", cost_usd populated (or null on subscription auth without per-call billing visibility)

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py query <handle> 'Repeat your previous answer.'", timeout=120)
# Expect: text "OK" again. Proves resume works across separate processes.

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py costs <handle>", timeout=10)
# Expect: total_cost_usd matches in-store total

terminal(command="python skills/autonomous-ai-agents/claude-code-sdk/scripts/session_manager.py close <handle>", timeout=10)
# Expect: {"status": "closed"}
```

If the second query does not echo "OK", session resume is failing. Likely causes: SDK older than 0.1.68 (no `session_id` field on `ClaudeAgentOptions`), `cwd` mismatch, or the underlying `claude` CLI being out of date.

## Rules for Hermes Agents

1. **Use this skill when the task is programmatic.** Multi-repo audits, batch refactors, scripted budgets. Use `claude-code` when the task wants an interactive REPL.
2. **Always `close` when done.** Records sit in `sessions.json` until the reaper or an explicit close removes them. Stale records are harmless but cluttering.
3. **Watch `total_cost_usd` after each query.** The cost surfaces directly in the JSON response; budget checks belong in your loop, not in user-facing prose.
4. **Open one handle per project.** Do not try to share a single handle across multiple cwds; resume will not survive that.
5. **One query at a time per handle.** Open more handles for parallel work.
6. **Treat `cost_usd: null` as "unknown".** Some auth modes do not surface per-call cost. Do not display $0.00 in that case.

## References

- Official SDK repository: https://github.com/anthropics/claude-agent-sdk-python
- Official SDK docs: https://docs.claude.com/en/api/agent-sdk/python
- Cost tracking docs: https://docs.claude.com/en/api/agent-sdk/cost-tracking
- Migration guide (`claude-code-sdk` to `claude-agent-sdk`): https://docs.claude.com/en/api/agent-sdk/migration-guide
- Upstream project this skill is derived from: https://github.com/0xRaghu/perplexity-claude-agent
