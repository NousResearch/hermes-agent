# Claude Code MCP Bridge - Development Notes

Goal

Enable the **Claude Code MCP bridge** in the `hermes-swarm` Docker deployment on `tusker@10.0.0.231`. The goal is to connect Hermes to Claude Code as an MCP server — Claude Code acts as tool executor, OpenCode provides the LLM via the sampling protocol.

Architecture

```
ClaudeCodeMCPBridge.chat(prompt)
  → stdio_client()  [primary async transport]
    → spawns: claude mcp serve
      → ClientSession(read_stream, write_stream, sampling_callback=_sampling_callback)
      → session.initialize()
      → session.list_tools()
      → optional Agent/SubTask probes
  → pty_stdio_client() [legacy fallback]
```

## Key Files

- `agent/claude_code_bridge.py` — Main bridge implementation
  - stdio-first MCP transport
  - PTY fallback transport
  - `_sampling_callback` — handles `sampling/createMessage` by calling Hermes LLM
- `test_claude_mcp_handshake.py` — basic initialize/tools-list handshake harness
- `test_agent_subtask.py` — startup-variant matrix for Agent/SubTask probes
- `test_agent_subtask_seq.py` — simple sequencing probe
- `test_agent_subtask_seq_ext.py` — plan/worktree sequencing probe
- `test_agent_subtask_seq_ext2.py` — deeper variant + sequencing probe
- `.claude/agents/mcp-local-smoke.md` — local project smoke subagent for MCP discovery

## Current Status

### Working locally

- `claude mcp serve` works over **direct stdio**.
- MCP `initialize` succeeds.
- `tools/list` succeeds.
- Rich tool surface appears when `CLAUDE_CODE_SIMPLE` is **not forced**.
- Exposed tools include `Agent`, `TaskOutput`, `TaskStop`, `EnterPlanMode`, `EnterWorktree`, and others.
- `EnterPlanMode` works.
- `EnterWorktree` works and creates a real worktree.

### Current blocker

`Agent` is exposed as an MCP tool, but **subagent types are not resolving at runtime** in `claude mcp serve`.

Observed failures include:
- `general-purpose`
- `Plan`
- `Explore`
- `swarm:worker`
- `swarm:coordinator`

Typical error:

```text
Agent type 'general-purpose' not found. Available agents:
```

This happens even though:

```bash
claude agents
```

shows those agents are installed and available.

## What We Learned

### 1. The missing Task/SubTask surface was mostly caused by simple mode

When `CLAUDE_CODE_SIMPLE=1` is forced, Claude MCP only exposes a reduced tool surface like:
- `Bash`
- `Read`
- `Edit`

Without simple mode, MCP exposes a much richer surface including:
- `Agent`
- `TaskOutput`
- `TaskStop`
- `EnterPlanMode`
- `EnterWorktree`
- `TodoWrite`
- and others

### 2. `Task` is now `Agent`

Anthropic docs indicate `Task` was renamed to `Agent` in Claude Code 2.1.63.
Older `Task(...)` references are aliases.

### 3. Agent spawning may not be fully wired in MCP serve runtime

Despite exposing the `Agent` tool and listing agents via `claude agents`, `claude mcp serve` in this environment appears to expose the schema without resolving the backing agent registry correctly.

## Harnesses

### 1. Basic handshake

```bash
source venv/bin/activate
python3 test_claude_mcp_handshake.py
```

Confirms:
- initialize works
- tools/list works

### 2. Startup-variant Agent probe matrix

```bash
source venv/bin/activate
python3 test_agent_subtask.py
```

This tests startup variants:
- default
- `--setting-sources user,project,local`
- inline `--agents`
- settings + inline `--agents`

And probes agent types:
- built-ins (`general-purpose`, `Plan`, `Explore`)
- plugin agents (`swarm:worker`, `swarm:coordinator`)
- project-local smoke agent (`mcp-local-smoke`)
- inline smoke agent (`mcp-inline-smoke`)

### 3. Sequencing probe

```bash
python3 test_agent_subtask_seq.py
```

### 4. Extended plan/worktree sequencing probe

```bash
python3 test_agent_subtask_seq_ext.py
```

### 5. Extended sequencing + startup variant matrix

```bash
python3 test_agent_subtask_seq_ext2.py
```

This combines:
- startup variants
- repeated `EnterPlanMode` / `EnterWorktree`
- built-in/plugin/custom agent probes

Optional fork-subagent flag:

```bash
CLAUDE_CODE_FORK_SUBAGENT=1 python3 test_agent_subtask_seq_ext2.py
```

## Local Smoke Agent

Project-local MCP smoke agent:

```text
.claude/agents/mcp-local-smoke.md
```

Definition:

```markdown
---
name: mcp-local-smoke
description: Local MCP smoke test agent
tools: Read
model: haiku
---

Reply with exactly MCP_SMOKE_OK
```

This helps distinguish whether the issue is:
- built-in/plugin agent resolution only, or
- all Agent spawning in MCP serve

## Recommended Next Probes

### A. If inline agent works but built-ins fail

Then MCP Agent execution works, but built-in/plugin registry loading is broken.

### B. If local smoke agent works but built-ins fail

Then project agent discovery works, but built-in/plugin agent loading is broken.

### C. If neither inline nor local smoke agent works

Then `Agent` tool exposure may be superficial in MCP serve, and actual subagent execution may not be supported in this runtime path.

### D. Compare with Agent SDK

Anthropic docs strongly suggest subagents are a first-class feature in the **Agent SDK**. If MCP serve continues failing for real Agent spawning, compare behavior with SDK-based subagent invocation.

## Environment Toggles

### Force simple mode (reduced tool surface)

```bash
export CLAUDE_CODE_USE_SIMPLE=1
```

or legacy toggle:

```bash
export HERMES_MCP_SIMPLE_MODE=1
```

### Try forked subagent behavior

```bash
export CLAUDE_CODE_FORK_SUBAGENT=1
```

Note: docs mention this for interactive mode, SDK, and `claude -p`; it may not help `mcp serve`.

## Practical Summary

Right now we have proven:
- MCP transport works
- richer tool surface works
- `Agent` tool is visible
- session-state tools like `EnterPlanMode` and `EnterWorktree` work

But we have **not yet proven successful Agent/SubTask execution via MCP serve** because the runtime cannot currently resolve agent types, even though they exist in normal Claude CLI contexts.
