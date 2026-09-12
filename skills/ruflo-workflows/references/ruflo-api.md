# Ruflo API Reference

## MCP Tools (via `claude mcp add ruflo -- npx ruflo@latest mcp start`)

### Swarm Coordination
- `ruflo__swarm_init(goal, agentSpec?, topology?)` — Spawn multi-agent swarm
- `ruflo__agent_spawn(type, goal, context?)` — Spawn single specialized agent
- `ruflo__agent_list()` — List active agents with status
- `ruflo__agent_kill(agentId)` — Terminate agent
- `ruflo__swarm_status(swarmId)` — Get swarm progress

### Vector Memory (AgentDB + HNSW)
- `ruflo__memory_store(key, value, namespace?, ttl?)` — Store with semantic embedding
- `ruflo__memory_search(query, limit?, namespace?)` — Semantic search (~0.99 recall@10)
- `ruflo__memory_delete(key, namespace?)` — Remove entry
- `ruflo__memory_namespace_list()` — List namespaces

### Agent Federation (Zero-Trust Peering)
- `ruflo__federation_init(nodeId, endpoint, agentTypes[])` — Initialize this node
- `ruflo__federation_join(endpoint)` — Join peer by endpoint
- `ruflo__federation_peers()` — List discovered peers + trust levels
- `ruflo__federation_send(targetNodeId, messageType, payload)` — Send typed envelope
- `ruflo__federation_query(targetNodeId, query)` — Synchronous query (ATTESTED+)
- `ruflo__federation_status()` — Node + peer trust summary
- `ruflo__federation_trust(peerId, level?)` — View/adjust trust (operator)
- `ruflo__federation_audit()` — Read audit log (operator)
- `ruflo__federation_breaker_status()` — Per-peer circuit breaker state

### GOAP Planning (goal.ruv.io)
- `ruflo__goal_plan(goal, constraints?)` — Plain English → executable plan tree
- `ruflo__goal_agents()` — Live agent dashboard data

## CLI Commands
```bash
npx ruflo@latest init wizard          # Interactive setup
npx ruflo@latest init                 # Quick non-interactive
npx ruflo@latest mcp start            # Start MCP server
npx ruflo@latest swarm "goal"         # Spawn swarm from CLI
npx ruflo@latest federation init      # Init federation node
npx ruflo@latest federation join <ep> # Join peer
npx ruflo@latest goal "goal"          # GOAP plan
```

## Web UIs
- **Chat + MCP**: https://flo.ruv.io (self-host: `ruflo/src/ruvocal/`)
- **Goal Planner**: https://goal.ruv.io (self-host: `v3/goal_ui/`)
- **Agent Dashboard**: https://goal.ruv.io/agents

## Architecture
```
User → Ruflo CLI/MCP → Router → Swarm → Agents → Memory → LLM Providers
                          ↑                           |
                          +---- Learning Loop <-------+
```
- 100+ specialized agents (coding, review, test, security, arch, docs)
- Hierarchical, mesh, adaptive swarm topologies
- SONA neural patterns, ReasoningBank, trajectory learning
- HNSW AgentDB (~1.9x faster at 20k, 3.2-4.7x at 5k)
- 12 background workers (audit, optimize, testgaps, etc.)
- 33 native Claude Code plugins + 21 npm plugins
- Multi-provider: Claude, GPT, Gemini, Cohere, Ollama
- Security: AIDefence, CVE remediation, path traversal prevention
- Federation: mTLS + ed25519, PII pipeline, WireGuard mesh (ADR-111)