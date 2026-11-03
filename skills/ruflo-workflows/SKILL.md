---
name: ruflo-workflows
description: Orchestrate multi-agent swarms and shared memory from Hermes.
version: 1.1.0
author: Louis Ling
license: MIT
platforms: [linux, macos]
tags: [ruflo, swarm, federation, multi-agent, tencentdb]
category: autonomous-ai-agents
metadata:
  hermes:
    config:
      - key: ruflo.mcp_command
        description: Command to start Ruflo MCP server
        default: "npx ruflo@latest mcp start"
        prompt: Ruflo MCP start command
      - key: ruflo.federation_endpoint
        description: Ruflo federation WebSocket endpoint for cross-machine peering
        default: "ws://localhost:9100"
        prompt: Ruflo federation endpoint
      - key: ruflo.shared_service_id
        description: TencentDB memory namespace shared across profiles
        default: "hermes-shared-memory"
        prompt: Shared TencentDB service ID
---

# Ruflo Workflows

## When to Use
- Spawn a multi-agent swarm (coding, review, test, full) for a complex task
- Run several coding agents in parallel on independent worktrees
- Persist findings to shared memory all Hermes profiles can see
- Cross-machine agent collaboration (Ruflo federation, requires Ruflo installed)

## Prerequisites

Run the healthcheck FIRST — it tells you which mode is available:

```bash
bash ~/.hermes/scripts/ruflo-healthcheck.sh
```

| Mode | Requires | What works |
|---|---|---|
| **A: Ruflo MCP** | `npx ruflo@latest mcp start` registered in Hermes (`/mcp list`) | `ruflo__*` tools: swarm, federation, AgentDB memory |
| **B: Hermes-native** | nothing extra (always available) | `delegate_task`, `terminal(background=true)`, `npx` coding agents, `tdai_*` shared memory |
| **Shared memory** | TDAI gateway on `:8420` (launchd-supervised, auth-enforced) | `tdai_memory_search` (L1 semantic, verified working) + `tdai_conversation_search` (L0 instant) — all profiles share `hermes-shared-memory` |

## Quick Reference

| Task | Mode A (Ruflo MCP) | Mode B (Hermes-native) |
|---|---|---|
| Spawn swarm | `ruflo__swarm_init(topology, maxAgents, strategy, config)` | `bash ~/.hermes/skills/ruflo-workflows/scripts/ruflo-swarm.sh "<goal>" [type]` → prints delegate_task payload |
| Swarm status | `ruflo__swarm_status(swarmId)` | `process(action=list)` |
| Swarm health | `ruflo__swarm_health(swarmId)` | — |
| Swarm shutdown | `ruflo__swarm_shutdown(swarmId)` | `process(action=kill)` |
| Single agent | `ruflo__agent_spawn` / `agent_execute` | `delegate_task(goal, context)` |
| List agents | `ruflo__agent_list()` | `process(action=list)` |
| Store memory | `ruflo__memory_store(key, value, namespace)` | `bash ~/.hermes/skills/ruflo-workflows/scripts/ruflo-memory.sh store "<content>" [tag]` |
| Search memory (L1) | `ruflo__memory_search(query, limit)` | `bash .../ruflo-memory.sh search "<query>" [limit]` |
| Recent memory (L0, instant) | `ruflo__memory_search` (lags) | `bash .../ruflo-memory.sh recent "<query>" [limit]` |
| Verify setup | — | `bash ~/.hermes/scripts/ruflo-healthcheck.sh` |
| Federation | `ruflo__federation_join/send/peers` | `send_message` (human channels) |
| Goal planning | `ruflo__goal_plan(goal)` | `plan` skill (markdown plan to `.hermes/plans/`) |

## Procedure

### 1. Verify setup (always first)
```bash
bash ~/.hermes/scripts/ruflo-healthcheck.sh
# Prints: ruflo installed? MCP registered? gateway up? auth enforced? L0/L1 search OK?
```

### 2. Mode A — Ruflo MCP (verified working)

Spawn a swarm (verified against ruflo v3.34.0 — task detail goes in `config`):
```
delegate_task(
  goal="Call ruflo__swarm_init with topology='hierarchical', maxAgents=3, strategy='balanced', config={goal:'<task>', task:'<detail>'}. Report swarmId. Then swarm_health, then swarm_shutdown.",
  context="Ruflo MCP tools available: ruflo__swarm_init, ruflo__swarm_status, ruflo__swarm_health, ruflo__swarm_shutdown"
)
```

Lifecycle to use with a running swarm:
```
ruflo__swarm_status(swarmId)    # status, agentCount, taskCount
ruflo__swarm_health(swarmId)    # coordinator/agents/persistence/topology checks
ruflo__swarm_shutdown(swarmId)  # graceful teardown
```

Cross-machine federation (both hosts must run Ruflo):
```
ruflo__federation_init(nodeId="<this-host>", endpoint="ws://<tailscale-ip>:9100", agentTypes=["coder","reviewer"])
ruflo__federation_join(endpoint="ws://<peer-tailscale-ip>:9100")
ruflo__federation_send(targetNodeId="<peer>", messageType="agent-handoff", payload={task: "..."})
```

### 3. Mode B — Hermes-native fallback (works today, no Ruflo install)

Generate the right payload for the task size:
```bash
bash ~/.hermes/skills/ruflo-workflows/scripts/ruflo-swarm.sh "Build a REST API with auth" coding
# → prints delegate_task / background terminal payload, ready to execute
```

Parallel coding agents on isolated worktrees:
```
delegate_task(tasks=[
  {goal: "Implement feature X in worktree A", context: "Run: git worktree add ../wt-a main; cd ../wt-a; npx claude-code -p '<task>'"},
  {goal: "Review PR #N", context: "Run: npx claude-code -p 'review this diff' or use gh pr review"},
])
```

Long-running background agent:
```
terminal(background=true, notify_on_complete=true, command="cd <repo> && npx claude-code -p '<big task>'")
process(action=poll)  # check progress
process(action=kill)  # if stuck
```

### 4. Shared memory (all profiles see this)

CLI helpers (no Python needed — the agent or user runs these):
```bash
bash ~/.hermes/skills/ruflo-workflows/scripts/ruflo-memory.sh store "Finding: API latency 200ms" findings
bash ~/.hermes/skills/ruflo-workflows/scripts/ruflo-memory.sh recent "findings" 5   # L0, instant
bash ~/.hermes/skills/ruflo-workflows/scripts/ruflo-memory.sh search "latency" 5   # L1, semantic
```

Or the provider tools in-session:
```python
# Store (writes to TencentDB L0 via provider)
memory(action="add", target="<topic>", content="<finding>")

# Read back — instant (L0 raw turns):
tdai_conversation_search(query="<topic>", limit=5)

# Read back — semantic (L1 atomic, async pipeline, may lag):
tdai_memory_search(query="<topic>", limit=5)
```

All profiles share namespace `hermes-shared-memory` (config `ruflo.shared_service_id`), so a finding stored by the researcher profile is visible to software_developer, project_manager, etc.

## Pitfalls
- **Provider routing**: ruflo's `agent_execute` is launched via `~/.hermes/scripts/ruflo-mcp.sh`. Defaults to **Ollama mode** (`RUFLO_PROVIDER=ollama`) — injects `OLLAMA_API_KEY`, forces ruflo's Ollama path, executes swarm agents via Ollama Cloud using `OLLAMA_DEFAULT_MODEL=deepseek-v4-flash:0731`. No Anthropic key or proxy needed. Set `RUFLO_PROVIDER=anthropic` to use Claude Code's OAuth token instead (reads `~/.claude/.credentials.json`, Bearer works directly at `api.anthropic.com`). Requires patched `resolveOllamaModel` in `agent-execute-core.js` (maps `claude-*` → Ollama default).
- **A local patch** to ruflo's `agent-execute-core.js` makes it honor `ANTHROPIC_BASE_URL` and send `Authorization: Bearer` when `ANTHROPIC_AUTH_TOKEN` is set (OAuth-compatible). This lives in node_modules — `npm update`/reinstall of ruflo will wipe it; re-apply from the test file or git note. It does NOT affect `api.anthropic.com` usage when no env is set.
- Mode A tools are registered per-profile under `mcp_servers.ruflo` in each profile's `config.yaml` (stdio → wrapper, 333 tools). They load when a Hermes session starts in that profile — restart the session if a tool is missing.
- The healthcheck auto-detects Ruflo via `~/.local/bin/ruflo` (npm global) — if you install elsewhere, update the candidate list in `ruflo-healthcheck.sh`.
- Ruflo's own AgentDB (`ruflo__memory_search`, HNSW+sql.js) is SEPARATE from TencentDB (`tdai_*`). Store cross-profile knowledge in TencentDB; Ruflo AgentDB is per-Ruflo-instance.
- Gateway requires auth: `Authorization: Bearer <key>` + `x-tdai-service-id` headers. The provider sets these from `TDAI_MEMORY_API_KEY` / `TDAI_MEMORY_SERVICE_ID` in each profile's `.env` — don't clear them.
- L1 (`tdai_memory_search`) is verified working against shared namespace `hermes-shared-memory`; for brand-new writes it may lag L0 briefly while the async pipeline indexes. Use `tdai_conversation_search` for just-written data.
- Hard-linked `.env` files: edits to one profile's `.env` may silently apply to others (shared inode). Use atomic replace (temp file + `os.replace`) when editing.
- Federation needs Tailscale (or reachable WSS) between hosts; trust ladder starts at UNTRUSTED — only `discovery` until interactions promote it.
- Config corruption: two profiles were found with broken `known_plugin_toolsets` YAML (flattened `headroom:` key). If a profile's config won't parse, check that block — repair template: `known_plugin_toolsets: {headroom: [headroom_retrieve, ccr_mirror], cli: [spotify]}`.

## Verification
- `bash ~/.hermes/scripts/ruflo-healthcheck.sh` → all green
- `tdai_conversation_search(query="ruflo", limit=3)` returns entries (L0 works)
- If Mode A: `ruflo__agent_list()` returns agents; `ruflo__memory_search` returns stored keys
- Cross-profile: store from one profile's session, read from another
