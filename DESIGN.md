# Single Gateway, Multiple Agents — Design Document

## Overview

Hermes Agent today runs one `hermes gateway run` process per profile (`HERMES_HOME`). Each process binds exactly one model, one system prompt (SOUL.md), one MEMORY.md, and one skill set. Deploying multiple personalities requires N gateway processes, N supervisors, N ports, N tunnels, and N memory footprints. This is the bottleneck behind issues #23735, #7517, #9514, and #12099.

This document describes the MVP implementation that enables a **single gateway process to host N isolated AI agents**, routing inbound messages by platform/chat/thread/user metadata while keeping each agent's memory, skills, and configuration fully separate.

---

## Goals

- Run N concurrent agent loops inside one gateway process
- Route inbound messages to the correct agent by declarative rules
- Keep agent memory, skills, SOUL.md, and model config isolated
- Zero behavior change for existing single-profile installations

## Non-Goals (Future PRs)

- Per-agent token bucket or priority queue
- Filesystem isolation guards
- Per-agent process supervision
- A2A (agent-to-agent) communication

---

## Architecture

### 1. Agent Identity — `agent_id`

Every session now carries an `agent_id` field. The default is `"main"`, preserving backward compatibility.

**Session key format** (updated):
```
agent:<id>:<platform>:<chat_type>:<chat_id>[:<thread_id>][:user_id]
```

`build_session_key` now reads `source.agent_id` and embeds it in the prefix. Existing keys without an agent prefix continue to work because the default is `"main"`.

**Files touched:**
- `gateway/session.py` — `SessionSource.agent_id`, `SessionEntry.agent_id`, `build_session_key`
- `hermes_state.py` — SQLite migration adds `agent_id TEXT NOT NULL DEFAULT 'main'`

### 2. AgentProfile + ContextVar

A new `AgentProfile` dataclass holds per-agent configuration:

```python
@dataclass
class AgentProfile:
    id: str = "main"
    home_dir: Optional[Path] = None      # profile-specific HERMES_HOME
    soul_md_path: Path = field(init=False)
    memory_dir: Path = field(init=False)
    skills_dir: Path = field(init=False)
    sessions_path: Path = field(init=False)
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    enabled_toolsets: Optional[list] = None
    disabled_toolsets: Optional[list] = None
    config_overrides: dict = field(default_factory=dict)
```

A `ContextVar` propagates the active profile through async call chains:

```python
_current_agent_profile: ContextVar[Optional[AgentProfile]] = ContextVar(...)

def get_active_profile() -> Optional[AgentProfile]: ...
def set_active_profile(p: AgentProfile) -> Token: ...
@contextmanager
def use_profile(p: Optional[AgentProfile]): ...
```

Path getters (`get_hermes_home`, `get_skills_dir`, `get_memory_dir`) lazily import `get_active_profile` and return the profile's `home_dir` when set. This avoids circular imports because `hermes_constants.py` is import-safe.

**Files touched:**
- `agent/profile.py` — new module
- `hermes_constants.py` — `get_hermes_home`, `get_skills_dir` read ContextVar
- `tools/memory_tool.py` — `get_memory_dir` reads ContextVar
- `agent/prompt_builder.py` — SOUL.md path reads ContextVar

### 3. Declarative Routing

Gateway config gains three top-level keys:

```yaml
default_agent: main

agents:
  main:
    model: anthropic/claude-sonnet-4-6
  coder:
    home_dir: ~/.hermes/profiles/coder
    model: anthropic/claude-opus-4-6
    enabled_toolsets: [filesystem, terminal]
  research:
    home_dir: ~/.hermes/profiles/research

routes:
  - match: { platform: telegram, chat_id: "-1001234", thread_id: "42" }
    agent: coder
  - match: { platform: telegram, chat_id: "-1001234" }
    agent: research
  - match: { platform: slack, guild_id: "T0ABC" }
    agent: coder
```

**Resolution order** (first match wins):
1. Declarative `routes` — first matching rule
2. `select_agent` plugin hook — first non-None return value
3. `default_agent` config key
4. Fallback `"main"`

**Match keys supported:** `platform`, `chat_id`, `thread_id`, `user_id`, `user_id_alt`, `guild_id`, `parent_chat_id`

**Files touched:**
- `gateway/config.py` — `agents`, `routes`, `default_agent` schema
- `gateway/agent_routing.py` — new resolver module
- `gateway/platforms/base.py` — `_attach_agent_id()` stamps `agent_id` before `build_session_key`
- `hermes_cli/plugins.py` — `select_agent` hook registration

### 4. GatewayRunner Multi-Agent Loading

`GatewayRunner.__init__` calls `load_agent_registry(self.config)` and stores `self._agent_registry: Dict[str, AgentProfile]`.

**Message handling flow:**
1. Adapter receives message
2. `_attach_agent_id(event)` resolves agent via routing table + hook
3. `use_profile(registry[source.agent_id or "main"])` wraps `_handle_message_with_agent`
4. Inside that block, `_resolve_session_agent_runtime` and `_apply_profile_toolsets` read the active profile via `get_active_profile()` and overlay `profile.model`, `profile.provider`, `profile.base_url`, `profile.api_key_env`, `profile.enabled_toolsets`, `profile.disabled_toolsets` on top of the gateway defaults
5. `AIAgent` is constructed with the per-agent model/runtime/toolsets; the active ContextVar means all path getters inside the agent loop (SOUL.md, memories/, skills/, sessions.json) return profile-specific paths

**Precedence:** session `/model` override (highest) → profile override → gateway default. The default `"main"` profile is a no-op overlay so legacy single-agent installs see zero behavior change.

**Cache safety:** `_agent_cache` and `_running_agents` are keyed by `session_key`, which now includes `agent_id`. No code changes needed — the cache is naturally multi-agent safe.

**Files touched:**
- `gateway/run.py` — registry loading, profile wrapping, `_apply_profile_runtime_overrides`, `_apply_profile_toolsets`, hook `agent_id` kwargs

### 5. Cron + Delivery Context Propagation

**Cron jobs:**
- `CronJob` gains `agent_id` field (defaults to `"main"`)
- `create_job` detects active profile and stores correct `agent_id`
- Jobs are stored per-profile in `<home_dir>/cron/jobs.json`
- `scheduler.tick()` reads `job["agent_id"]` and wraps execution in `use_profile()`

**Delivery:**
- `DeliveryTarget` gains `agent_id` field (None means inherit from origin)
- `DeliveryRouter` accepts registry, wraps each delivery in `use_profile()`

**Files touched:**
- `cron/jobs.py` — `agent_id` field, per-profile storage
- `cron/scheduler.py` — `tick()` profile wrapping
- `gateway/delivery.py` — `DeliveryTarget.agent_id`, router profile wrapping

### 6. Hook Payloads

Every `invoke_hook` call now passes `agent_id=` so plugin callbacks can branch on the active agent:

- `gateway/run.py` — `on_session_finalize`, `pre_gateway_dispatch`, `on_session_reset`
- `run_agent.py` — `on_session_start`, `pre_llm_call`, `post_llm_call`, etc.
- `model_tools.py` — `post_tool_call`, `transform_tool_result`
- `tools/approval.py`, `tools/terminal_tool.py`, `tools/delegate_tool.py`

### 7. CLI

New `hermes agent` subcommand group:

```bash
hermes agent list          # table: id / model / home_dir / route count
hermes agent add <id>      # --from-profile, --model, --home-dir
hermes agent remove <id>   # warns about affected routes
hermes agent show <id>     # paths, routes, model, SOUL summary
```

**Files touched:**
- `hermes_cli/agent.py` — new module
- `hermes_cli/main.py` — subparser registration, `_BUILTIN_SUBCOMMANDS`

---

## Business Model & Deployment Scenarios

### Entity Relationship

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT (1 per server)                        │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      GATEWAY PROCESS (1)                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │    │
│  │  │  Agent:main │  │  Agent:coder│  │ Agent:research            │    │
│  │  │  (default)  │  │  (profile)  │  │  (profile)  │             │    │
│  │  │             │  │             │  │             │             │    │
│  │  │ SOUL.md     │  │ SOUL.md     │  │ SOUL.md     │             │    │
│  │  │ MEMORY.md   │  │ MEMORY.md   │  │ MEMORY.md   │             │    │
│  │  │ skills/     │  │ skills/     │  │ skills/     │             │    │
│  │  │ sessions.db │  │ sessions.db │  │ sessions.db │             │    │
│  │  │ cron/jobs   │  │ cron/jobs   │  │ cron/jobs   │             │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │    │
│  │                                                                  │    │
│  │  Shared: adapters, config.yaml, _agent_cache, hook registry     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    PLATFORM ADAPTERS (shared)                    │    │
│  │   Telegram  Slack  Discord  WeChat  Feishu  Matrix  ...         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Scenario Matrix

| Scenario | Gateways | Agents | Use Case |
|----------|----------|--------|----------|
| **Single user, single personality** | 1 | 1 (`main`) | Default Hermes install. One person, one model, one memory space. |
| **Single user, multiple personalities** | 1 | N | One person routes different chats to different agents (e.g., work chat → professional agent, hobby chat → casual agent). |
| **Multi-tenant (team/org)** | 1 | N | One gateway serves a team. Each member or project gets an isolated agent with its own memory and skills. |
| **High-availability / sharding** | N | N per gateway | Multiple gateway processes behind a load balancer, each hosting a subset of agents. Used when one process cannot handle all traffic. |
| **Environment separation** | N | 1 per gateway | Separate gateways for dev/staging/prod, each with one agent. Not solved by this PR — still requires N processes. |

### When Do You Need Multiple Gateways?

| Reason | Solution |
|--------|----------|
| CPU/memory limits — one process cannot host all agents | Run N gateway processes, shard agents across them |
| Geographic latency — users in different regions | Gateway per region, each with relevant agents |
| Isolation requirements — agents must not share a process | Separate gateway processes (this PR does not change that) |
| Different Hermes versions — some agents need different code | Separate gateway processes |

**This PR solves:** One gateway process hosting multiple isolated agents.
**This PR does NOT solve:** Process-level isolation or horizontal scaling.

---

## Data Flow Diagram

```
Inbound Message
    |
    v
[Adapter] ──► [_attach_agent_id]
    |              |
    |              ├── routes match ──► agent_id = "coder"
    |              ├── select_agent hook ──► (optional override)
    |              └── default_agent ──► "main"
    |
    v
[use_profile(registry[agent_id])]
    |
    v
[GatewayRunner._handle_message_with_agent]
    |
    ├──► build_session_key ──► "agent:coder:telegram:dm:123"
    ├──► AIAgent(profile=registry["coder"])
    │         └── run_conversation ──► use_profile(self._profile)
    │                    ├──► get_hermes_home() ──► ~/.hermes/profiles/coder
    │                    ├──► SOUL.md ──► profiles/coder/SOUL.md
    │                    ├──► memories ──► profiles/coder/memories/
    │                    └──► skills ──► profiles/coder/skills/
    └──► invoke_hook(..., agent_id="coder")
```

---

## Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| No `agents:` / `routes:` config | Registry = `{"main": AgentProfile()}`, all messages route to `"main"` |
| Existing SQLite sessions rows | Migration adds `agent_id` column with default `"main"` |
| Existing `sessions.json` | `from_dict` defaults `agent_id` to `"main"` |
| Existing `cron/jobs.json` | Load defaults missing `agent_id` to `"main"` |
| `HERMES_HOME` env | Still the default profile home; sub-profiles have independent dirs |
| `hermes -p coder gateway run` | CLI path unchanged; ContextVar layers on top |

---

## Testing

### Automated Tests Added (62)

- `tests/gateway/test_agent_routing.py` — 25 tests for `resolve_agent_id`, route matching, declaration order, invalid routes
- `tests/agent/test_profile_contextvar.py` — 25 tests for `AgentProfile`, ContextVar, `use_profile`, async isolation (`gather`, sibling tasks), `load_agent_registry`
- `tests/gateway/test_session.py` — 12 tests for `build_session_key` with `agent_id` across DM, group, thread, WhatsApp, shared group

### Existing Test Updates

- `tests/gateway/test_session_boundary_hooks.py` — hook assertions updated for `agent_id=None`
- `tests/test_model_tools.py` — hook call signatures updated for `agent_id=None`
- `hermes_cli/main.py` — `_BUILTIN_SUBCOMMANDS` adds `"agent"`

### Manual Test Checklist

1. **Basic routing — main fallback**: Message to unmatched chat → routes to `main`, session_key prefix `agent:main:...`
2. **Precise routing — thread match**: Message to Telegram forum topic 42 → routes to `coder`, session_key prefix `agent:coder:...`
3. **Memory isolation**: Say "I'm Alice" in topic 42, "I'm Bob" in another topic → each agent remembers only its own name
4. **SOUL.md isolation**: Different SOUL.md per profile → responses match respective personalities
5. **Skills isolation**: Enable filesystem toolset for coder only → research agent cannot access filesystem
6. **Session reset boundary**: `/new` in topic 42 → `on_session_finalize` receives `agent_id="coder"`
7. **Cron job isolation**: Create cron job in coder profile → file lands in `profiles/coder/cron/jobs.json`
8. **Delivery routing**: Trigger delivery from coder session → `DeliveryTarget` inherits `agent_id`, executes in coder context
9. **select_agent hook**: Plugin returns `"research"` → overrides route match
10. **Gateway restart recovery**: Restart gateway, message previous chat → restores existing session with correct `agent_id`
11. **Backward compat — no multi-agent config**: Delete `agents:` and `routes:` → all messages route to `main`
12. **Backward compat — existing SQLite**: Old sessions.db auto-migrates, old rows backfill to `"main"`

---

## Files Changed

| File | Change |
|------|--------|
| `agent/profile.py` | **New** — `AgentProfile`, ContextVar, `load_agent_registry` |
| `gateway/agent_routing.py` | **New** — `resolve_agent_id`, `_route_matches` |
| `gateway/session.py` | `SessionSource.agent_id`, `SessionEntry.agent_id`, `build_session_key` rewrite |
| `gateway/run.py` | Registry loading, `use_profile` wrapping, hook `agent_id` kwargs |
| `gateway/platforms/base.py` | `_attach_agent_id`, `set_routing_context` |
| `gateway/config.py` | `agents`, `routes`, `default_agent` schema |
| `gateway/delivery.py` | `DeliveryTarget.agent_id`, router profile wrapping |
| `hermes_constants.py` | `get_hermes_home`, `get_skills_dir` read ContextVar |
| `run_agent.py` | `AIAgent` accepts `profile`, sets ContextVar |
| `cron/jobs.py` | `CronJob.agent_id`, per-profile storage, backward-compat constants |
| `cron/scheduler.py` | `tick()` profile wrapping |
| `hermes_cli/agent.py` | **New** — `hermes agent` subcommand |
| `hermes_cli/main.py` | Register `agent` subparser, `_BUILTIN_SUBCOMMANDS` |
| `hermes_cli/plugins.py` | `select_agent` hook registration |
| `hermes_state.py` | SQLite migration for `agent_id` column |
| `model_tools.py` | Hook `agent_id` kwargs |
| `tools/approval.py` | Hook `agent_id` kwargs |
| `tools/terminal_tool.py` | Hook `agent_id` kwargs |
| `tools/delegate_tool.py` | Hook `agent_id` kwargs |
| `scripts/release.py` | AUTHOR_MAP entry |
| `cli-config.yaml.example` | Multi-agent config examples |
| `website/docs/user-guide/messaging/multi-agent.md` | **New** — documentation |
| `tests/agent/test_profile_contextvar.py` | **New** — 25 tests |
| `tests/gateway/test_agent_routing.py` | **New** — 25 tests |
| `tests/gateway/test_session.py` | +12 tests for `build_session_key` |
| `tests/gateway/test_session_boundary_hooks.py` | Updated hook assertions |
| `tests/test_model_tools.py` | Updated hook assertions |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| ContextVar + `create_task` timing | Always `set_active_profile` before `create_task`; existing `copy_context` patterns in `run_agent.py` followed |
| Module-level path snapshots | `cron/jobs.py` keeps `HERMES_DIR`/`CRON_DIR` as backward-compat fallbacks; production code uses dynamic getters |
| Provider client cache sharing | Cache key already includes api_key/base_url/region; different agents with different keys → separate cache entries |
| `_agent_cache` capacity | N agents × M sessions may exceed LRU cap; monitoring TODO left for future |
| Plugin rewrites `event.source` after `_attach_agent_id` | Documented: `pre_gateway_dispatch` fires after agent resolution; plugin changing chat does not trigger re-resolution |

---

## Verification Commands

```bash
# Unit tests
pytest tests/gateway/test_agent_routing.py -v
pytest tests/agent/test_profile_contextvar.py -v
pytest tests/gateway/test_session.py -v
pytest tests/gateway/test_session_boundary_hooks.py -v

# Backward compat
pytest tests/ -v -x

# Manual smoke
hermes agent add coder --model anthropic/claude-opus-4-6
# Edit ~/.hermes/config.yaml with routes
hermes gateway run
# Send messages to different routed chats and verify isolation
```
