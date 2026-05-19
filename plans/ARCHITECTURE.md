# Hermes Three-Service Architecture

> Reference document for plans 001, 002, 003, and Atlas plan 012.
> Living document — update as services are built and interfaces are confirmed.

## The Vision

Hermes is the **orchestration layer**. It does one thing well: manage agent turns, route tool calls, and coordinate user interaction. Everything else — memory, skills, self-improvement — is a peer service, connected via MCP.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         HERMES AGENT                                         │
│                    (agentic orchestration)                                   │
│                                                                              │
│  Responsibilities:                                                           │
│  - Turn management (receive message → tool loop → respond)                  │
│  - Gateway routing (Slack, Discord, CLI, API)                                │
│  - Identity resolution (who is asking, from where, with what scope)         │
│  - Tool dispatch (MCP connections to peer services)                          │
│  - NO persistent storage of its own beyond session cache                    │
└───────────┬─────────────────────┬──────────────────────┬───────────────────┘
            │ MCP                 │ MCP                  │ MCP (future)
            ▼                     ▼                      ▼
┌───────────────────┐  ┌──────────────────────┐  ┌──────────────────────────┐
│  ATLAS            │  │  SKILLS SERVICE       │  │  SELF-IMPROVEMENT (TBD)  │
│  (memory)         │  │  (learned tasks)      │  │  (eval + evolution)      │
│  localhost:8000   │  │  localhost:8001        │  │  localhost:8002           │
│                   │  │                       │  │                           │
│  RDF-grounded     │  │  Git-backed SKILL.md  │  │  Fitness scoring         │
│  knowledge store  │  │  registry with CSS    │  │  Skill extraction        │
│  SPARQL queries   │  │  scope resolution     │  │  Atlas ontology eval     │
│  PROV-O provenance│  │  personal→team→global │  │  (feeds Atlas + Skills)  │
│  contradiction    │  │  MCP: list/view/      │  │                           │
│  detection        │  │  promote/search       │  │                           │
│  Plans: Atlas     │  │  Plans: 003           │  │  Plans: TBD              │
│    001-014        │  │                       │  │                           │
└───────────────────┘  └──────────────────────┘  └──────────────────────────┘
```

## Service Responsibilities

### Hermes (orchestration)
- Owns: turn lifecycle, tool routing, identity/session management, gateway integrations
- Does NOT own: facts about the world, learned procedures, ontology
- MCP clients: Atlas (memory), Skills Service (tasks), platform gateways

### Atlas (memory substrate)
- Owns: all facts, relationships, decisions, and artifacts — RDF triples with provenance
- Does NOT own: how to perform tasks (that's skills), what to do next (that's Hermes)
- Serves: `search_knowledge`, `ingest`, `get_entity`, `get_timeline`, contradiction detection
- Plans: Atlas 001–014 (army-of-one repo)

### Skills Service (learned tasks)
- Owns: SKILL.md files across three scopes (personal, team, global)
- Does NOT own: facts (Atlas), turn logic (Hermes)
- Serves: `list_skills`, `view_skill`, `promote_skill`, `search_skills`
- Scope model: personal overrides team overrides global (CSS specificity)
- Collaboration: Git-backed, team promotion via PR
- Plans: Hermes Plan 003

### Self-Improvement Service (future, TBD)
- Owns: skill quality evaluation, Atlas ontology fitness measurement, agent behavior scoring
- Feeds: promotes high-quality skills (→ Skills Service), contributes ontology improvements (→ Atlas)
- Dependencies: Atlas (for context + evaluation data), Skills Service (to read + promote skills)
- Plans: partially covered by Atlas Plan 014 (ontology fitness); Hermes side TBD

## Interface Contract (MCP)

All three services expose MCP servers. Hermes connects to them via `~/.hermes/config.yaml`:

```yaml
mcp:
  servers:
    atlas:
      url: http://localhost:8000/mcp
      auth: bearer ${ATLAS_TOKEN}
    skills:
      url: http://localhost:8001/mcp
      auth: bearer ${SKILLS_TOKEN}
    # self-improvement: TBD
```

The `mcp_atlas_*` and `mcp_skills_*` tool prefixes are registered automatically via the native MCP client.

## Scoping Model

All three services use the same three-scope model, inherited from Plan 001:

```
personal  (user_id)    → most specific, always wins
  team    (team_id)    → shared workspace, writable by members
  global  (no id)      → hermes-agent defaults, read-only to agents
```

- Atlas scopes knowledge by `fence` (already implemented, see Plan 013)
- Skills Service scopes SKILL.md files by registry (Plan 003)
- Self-improvement scopes eval data by agent identity (TBD)

## Deployment Model

### Local / Personal (today)
```
[Blake's laptop]
  Atlas: localhost:8000  (docker-compose in army-of-one/)
  Skills: localhost:8001 (make up in hermes-skills-service/)
  Hermes: CLI / Slack gateway
```

### Team / SaaS (Plan 001-E)
```
[EC2 / ECS]
  Atlas: persistent EFS volume (Plan 005)
  Skills: S3 backend (Plan 003-F, HERMES_MODE=saas)
  Hermes: stateless worker pool
  Neon PostgreSQL: sessions + conversations
```

## Plan Cross-References

| What | Plan | Status |
|------|------|--------|
| Identity model (HermesIdentity, scope resolution) | Hermes 001-0 | DRAFT |
| Multi-user SaaS (sessions, cloud storage, deployment) | Hermes 001 | DRAFT |
| Hermes self-organization (directory layout, MCP pool) | Hermes 002 | DRAFT |
| Skills Service (scoped registries, MCP, Git, S3) | Hermes 003 | DRAFT |
| Atlas memory substrate (v1 complete) | Atlas 001–008 | COMPLETE |
| Atlas intelligence (RAG++, contradiction detection) | Atlas 003 | CODE-COMPLETE |
| Atlas→Hermes memory connector | Atlas 012 | DRAFT |
| Atlas self-improvement (ontology fitness) | Atlas 014 | IN PROGRESS |
| Self-improvement service (Hermes side) | TBD | Not planned |
