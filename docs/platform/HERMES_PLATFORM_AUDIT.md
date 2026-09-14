# Hermes Platform Audit

**Phase 0 deliverable — repository audit and proposed target architecture.**
No code has been changed. This document ends with a decision list that needs your approval
before Phase 1 begins.

| | |
|---|---|
| Repository | `yahyeameer/hermes-agen-` (fork of NousResearch/hermes-agent) |
| Version audited | `0.21.1`, HEAD `4bdd64b` |
| Scale | 5,987 Python files / ~797k LOC · ~3,000 TS/TSX files · 3,991 test files |
| License | MIT |
| Audit date | 2026-09-12 |

---

## 0. Executive summary

**Hermes is not a blank slate and it is not a business platform. It is a single-operator
autonomous agent runtime that already contains roughly 70–80% of the "Hermes Core" the PRD
describes — and almost none of the business layer.**

The parts the PRD asks for that already exist, production-grade:

- A real task kernel — SQLite Kanban with WAL + `BEGIN IMMEDIATE` compare-and-swap claiming,
  attempt history, heartbeats, a consecutive-failure circuit breaker, and typed block reasons.
- A real dispatcher that claims ready tasks and spawns isolated worker processes, with
  per-profile concurrency caps, stale/orphan detection and log rotation.
- A tool registry with per-tool schemas, availability checks and toolset grouping.
- A nine-module human approval gate, including blocks that fire *before* any bypass mode.
- A provider abstraction with 40 pluggable model providers, Bedrock and Anthropic included.
- Pluggable memory backends, MCP integration, a cron scheduler, a React admin dashboard with
  OIDC/OAuth auth providers, and a multi-platform gateway.

The parts that do not exist at all:

- **No declarative agent definition.** An "agent" is a directory on disk created by hand.
- **No supervisor.** The dispatcher is a scheduler, not a reasoning coordinator.
- **No knowledge/RAG layer.** No ingestion, chunking, embedding, or vector store anywhere.
- **No infrastructure as code.** Zero `.tf`, CDK or CloudFormation files.
- **No business-action approval policy.** Approvals gate *dangerous shell commands*, not
  *"issue a refund"*.
- **No token/cost kill switch** per customer or per agent.

**The strategic recommendation is therefore the opposite of a refactor.** The highest-value,
lowest-risk path is an *additive* package that composes existing primitives and touches core
only through extension points that already exist (profiles, distributions, plugins, the tool
registry, the approval gate, config). Rewriting the orchestration engine would destroy the
most valuable thing in this repository and permanently fork you from a fast-moving upstream.

---

## 1. Current architecture

### 1.1 Process topology

Hermes is a set of **cooperating processes over a shared filesystem root** (`$HERMES_HOME`,
default `~/.hermes`), not a service mesh.

```
$HERMES_HOME/                     ← the real isolation boundary today
  config.yaml  .env  SOUL.md      ← configuration, secrets, agent persona
  state.db  kanban.db             ← SQLite, WAL mode
  profiles/<name>/                ← a full nested Hermes root per profile
  skills/  cron/  memories/  sessions/  logs/
```

Five entry points share that root:

| Process | Entry | Role |
|---|---|---|
| CLI / TUI | `cli.py` (4,635 lines) → `run_agent.py` | Interactive operator session |
| Gateway | `gateway/run.py` (5,537 lines) | Chat platforms, cron, Kanban watchers, API server |
| Dispatcher | `hermes_cli/kanban_db_dispatch.py::run_daemon` | Claims tasks, spawns workers |
| Worker | `hermes -p <profile>` subprocess | One task, one process, then exits |
| Dashboard | `hermes_cli/web_routers/*` + `web/` (React) | Admin UI |

### 1.2 Layering

```
  Surfaces      CLI/TUI · Gateway (Telegram, Discord, Slack, WhatsApp, Signal,
                Teams, Feishu…) · OpenAI-compatible API server · Dashboard · MCP server
       │
  Agent loop    run_agent.py · agent/conversation_loop.py · agent/context_engine.py
                agent/context_compressor.py (4,931 lines) · compaction, caching
       │
  Tools         tools/registry.py → 260 modules · toolsets.py (named groups)
                tools/approval*.py (9 modules) · MCP client · code-execution RPC
       │
  Coordination  hermes_cli/kanban_db.py (4,136) + kanban_db_dispatch.py (2,349)
                kanban_swarm · kanban_decompose · delegate_tool (in-process subagents)
       │
  State         hermes_state*.py (22 modules) · SQLite + FTS5 · memory · skills
       │
  Providers     providers/ registry → plugins/model-providers/ (40 profiles)
                agent/bedrock_adapter.py · anthropic_adapter.py · transports/
```

### 1.3 Extension points that already exist

This list is the foundation of the whole proposal. Each is a supported way to change
behaviour **without editing core**:

1. `plugins/` — loader + storage, with bundled categories for model providers, memory,
   platforms, dashboard auth, context engines, cron providers, observability, browser.
   User plugins under `$HERMES_HOME/plugins/` **override** bundled ones by name.
2. `providers/` — pip-installed plugins via the `hermes_agent.plugins` entry point.
3. **Profiles** — `$HERMES_HOME/profiles/<name>/`, each a complete nested Hermes root with
   its own `config.yaml`, `SOUL.md`, `mcp.json`, `skills/`, `cron/`, state and credentials.
4. **Distributions** — `hermes_cli/profile_distribution.py`: a profile packaged as a git repo
   with a versioned `distribution.yaml` manifest, install/update, `env_requires` declarations,
   and a `distribution_owned` / `USER_OWNED_EXCLUDE` split that protects customer data on upgrade.
5. `toolsets.py` — named tool groups, extensible from the registry.
6. **Skills** — 199 `SKILL.md` files, progressive disclosure, `agentskills.io`-compatible.
7. **Blueprints** — a skill whose frontmatter declares an automation; bridges to cron.
8. `mcp.json` — per-profile MCP servers; 65 optional MCP integrations bundled.

---

## 2. Current agent architecture

### 2.1 There is no first-class agent object

Grep the repository for an agent definition and you will not find one. There are three
distinct things that behave like "an agent":

| Mechanism | Isolation | Persistence | Configured by |
|---|---|---|---|
| **Profile** | Separate directory, separate SQLite, separate credentials | Durable | Hand-created directory tree |
| **Kanban worker** | Separate OS process, separate workspace/git worktree | Per task | `assignee` column + dispatcher flags |
| **Subagent** (`delegate_task`) | Separate context window, same process | Per call | `delegation:` config block |

**The profile is the real agent boundary**, and the Kanban `assignee` column holds a profile
name — the dispatcher literally shells out to `hermes -p <assignee>`. That is a genuine
multi-agent architecture with the strongest isolation of the three options (process + filesystem
+ credentials). It is exactly the model the PRD wants. It is just not *declarative*.

### 2.2 Per-task overrides already exist

The `tasks` table carries per-task `model_override`, `provider_override`, `reasoning_effort`,
`skills`, `max_runtime_seconds`, `max_retries`, `goal_mode`, `goal_max_turns`. So a large part
of the PRD's agent property list is already schema-backed — at the *task* level rather than the
*agent* level.

### 2.3 Coverage against the PRD's agent model

| PRD property | Today | Where |
|---|---|---|
| id / name / description | ~ | Profile directory name + `distribution.yaml` |
| role, system instructions | ✅ | `SOUL.md` per profile |
| model, provider, temperature | ✅ | `config.yaml` `model:` + per-task override |
| capabilities / tools | ✅ | `enabled_toolsets`, registry availability checks |
| permissions | ⚠️ | Approval gate + deny globs — command-shaped, not business-shaped |
| memory policy | ⚠️ | Global `memory:` config, not per agent |
| knowledge sources | ❌ | No knowledge system |
| allowed agents | ❌ | Any profile can create a card for any assignee |
| task types | ❌ | Not modelled |
| approval requirements | ⚠️ | Dangerous-command detection only |
| retry policy / timeout | ✅ | `max_retries`, `max_runtime_seconds`, failure limit |
| concurrency limits | ✅ | `kanban.max_in_progress_per_profile` |
| enabled/disabled | ⚠️ | Implicit (profile exists or does not) |

### 2.4 There is no supervisor

`run_daemon` is a **scheduler**: poll → claim ready tasks → spawn workers → reap. It does not
reason, decompose, or route. Three modules come close and are the right foundation:

- `kanban_decompose.py` — splits a triage card into child cards.
- `kanban_specify.py` — turns a vague request into a specified card.
- `kanban_swarm.py` — plants a fixed graph: *planning root → parallel workers → verifier →
  synthesizer*, with a JSON "blackboard" stored as structured task comments. Explicitly
  "deliberately no second scheduler."

That last design note is the single best architectural signal in the repository, and the
proposal below follows it.

---

## 3. Current Kanban / task architecture

**This is the strongest asset in the repository and must be preserved as-is.**

Statuses: `triage → todo → scheduled → ready → running → blocked → review → done → archived`.

Tables: `tasks`, `task_links` (parent/child DAG), `task_comments`, `task_events`, `task_runs`,
`task_attachments`, `kanban_notify_subs`.

Concurrency correctness:

- WAL + `BEGIN IMMEDIATE` + compare-and-swap on `status`/`claim_lock`. SQLite serialises
  writers, so exactly one claimer wins and losers see zero rows — no retries, no distributed
  lock service. **The PRD's "avoid two agents doing the same task" requirement is already met.**
- `claim_expires`, `last_heartbeat_at`, `worker_pid` drive stale and orphan reclamation.

Reliability, already implemented:

- `consecutive_failures` + `DEFAULT_FAILURE_LIMIT = 2` circuit breaker, reset only on success —
  with an explicit comment refusing to reset on spawn, which would allow infinite loops.
- `block_kind` typed blocks; `dependency` blocks route to `todo`, others to `blocked`.
- `block_recurrences` + `BLOCK_RECURRENCE_LIMIT` → routes to `triage` so a cron cannot spin
  a card forever.
- `task_runs` records every attempt with `outcome` ∈ {completed, blocked, crashed, timed_out,
  spawn_failed, gave_up, reclaimed}.
- `idempotency_key` column present.
- Memory-aware concurrency caps in the dispatcher.

Human-in-the-loop is wired end to end: `review` status, `kanban_request_review` /
`kanban_request_changes` tools, and `gateway/kanban_watchers_notifier.py` pushing
completed/blocked events back to the chat thread that created the card.

**Forward-compat warning:** `tasks.workflow_template_id` and `current_step_key` exist, are
written when a task opts into a template, and are explicitly documented as *not yet consulted
for routing* — upstream is planning a v2 workflow router. Building a competing workflow engine
on these columns would collide with upstream.

---

## 4. Current memory system

Three unrelated mechanisms, correctly separated (the PRD's section 12 is largely satisfied):

| Layer | Implementation | Notes |
|---|---|---|
| Short-term | `agent/context_engine.py`, `context_compressor.py`, `conversation_compression.py` | Sophisticated compaction and prompt caching |
| Long-term curated | `tools/memory_tool_store.py` → `MEMORY.md` / `USER.md` | Bounded char budgets, atomic writes, file locks |
| Session recall | `hermes_state_fts.py` — SQLite FTS5 with CJK-bigram tokenizer | Searches *past conversations*, not documents |
| Pluggable external | `plugins/memory/` — mem0, honcho, supermemory, byterover, hindsight, holographic, openviking, retaindb | Vector stores only via third-party backends |

Security note worth carrying forward: curated memory is scanned by
`tools/threat_patterns.first_threat_message(scope="strict")` before it is written, because
memory enters the system prompt and a poisoned entry would persist across sessions. **Any
knowledge-retrieval layer we add must reuse this scanner** — customer documents are exactly
the same injection vector.

**Gap:** memory is per-`$HERMES_HOME`, not per-agent and not per-organization.

---

## 5. Current tool / MCP architecture

`tools/registry.py` is a clean central registry: each tool module calls `register()` at import
declaring schema, handler, toolset membership and an availability `check_fn`. `model_tools.py`
queries the registry rather than keeping parallel structures, and the import chain is documented
as cycle-safe.

- 260 tool modules; ~100k LOC.
- `toolsets.py` groups tools into named sets (`web`, `file`, `terminal`, `browser`, `kanban`,
  `delegation`, `code_execution`, `memory`, `safe`, …) with `includes` composition, per-platform
  bundles, and a webhook-safe set that deliberately excludes file/system execution because
  webhook payloads are untrusted.
- Tool output is budgeted (`tools/budget_config.py`): 100k chars/result default, 200k/turn,
  50k for `mcp_` tools, with disk spillover.
- MCP: `mcp_serve.py` (Hermes as a server), per-profile `mcp.json` clients, 65 bundled optional
  MCP integrations, `mcp_security.validate_mcp_server_entry`, and MCP elicitation used as an
  approval transport.

### The approval gate

`tools/approval.py` plus eight leaf modules. Order of evaluation:

1. **Floors that fire before any bypass** — hardline patterns, `sudo -S` password piping, and
   the user's own `approvals.deny` globs. These run *before* yolo mode / `approvals.mode: off` /
   cron approve-mode.
2. Permanent allowlist match.
3. Dangerous-pattern detection over normalised, de-obfuscated command variants (so `r\m` and
   `git st""atus` cannot sidestep a rule).
4. Optional LLM guardian verdict (`approval_smart`).
5. Human decision via CLI prompt, plugin transport, MCP elicitation, or a blocking gateway
   round-trip to a chat platform.

Note `_YOLO_MODE_FROZEN` is read once at import specifically so a skill running in-process
cannot set the env var and bypass every check — the codebase is thinking about prompt-injection
escalation. **This gate is a genuine security asset. Extend it; do not replace it.**

**Gap:** it gates *shell commands*. The PRD needs a declarative, per-agent, per-tool policy —
*"the Accounting Agent may call `issue_refund` only with human approval"* — which is a different
axis and currently unrepresentable.

---

## 6. Current database

**SQLite everywhere. No PostgreSQL, no Redis, no networked database of any kind.**

| File | Contents |
|---|---|
| `state.db` | Sessions, messages, FTS5 indexes, per-model token usage, gateway routing, titles |
| `kanban.db` | The task kernel (per board: `<root>/kanban/boards/<slug>/`) |
| `response_store.db` | Spilled large tool results |

`hermes_state*.py` is 22 modules / ~15k LOC covering schema, migrations, WAL, compression,
repair, read pooling, portability and maintenance. It is careful, well-tested code.

**The constraint this creates is the most important technical fact in this audit:**

- SQLite's correctness guarantees hold for **multiple processes on one host**. They do not hold
  across hosts. SQLite on EFS/NFS is a documented corruption risk.
- Therefore **the first AWS deployment must be single-node** (one EC2 instance, or one ECS task
  with an attached EBS volume). Horizontal scale-out requires a real database abstraction.
- **That abstraction does not exist.** Raw `sqlite3` calls, `sqlite3.Row` access and SQLite-specific
  SQL (FTS5, `BEGIN IMMEDIATE`, `INSERT OR IGNORE`) are spread across roughly 40 modules. Porting
  to Postgres is a multi-month project, not a configuration change.

This is not a blocker. A single well-provisioned node handles a very large number of business
agents. But it must be stated plainly in customer sizing, and **we should not pre-build the
abstraction** — that is exactly the over-engineering the PRD's section 28 forbids.

---

## 7. Current authentication and security

**Strong:**

- No hard-coded secrets found. Credentials live in `$HERMES_HOME/.env` and `auth.json`, both on
  the `USER_OWNED_EXCLUDE` list so a distribution upgrade can never overwrite or ship them.
- `RedactingFormatter` on every log handler — secrets are scrubbed before they reach disk.
- Secret sources: Bitwarden (`hermes secrets bitwarden`), 1Password, OS keyring, an encrypted
  browser credential vault with TOTP support.
- Dashboard auth is a plugin surface (`plugins/dashboard_auth/`: basic, nous, self_hosted, drain)
  with PKCE, JWKS verification and token refresh.
- Gateway authorisation: per-platform `<PLATFORM>_ALLOWED_USERS` allowlists, separate group-chat
  allowlists, explicit `ALLOW_ALL_USERS` opt-out.
- API server requires `API_SERVER_KEY`; dashboard binds `127.0.0.1` by default with compose
  comments explaining why exposing it is unsafe.
- Untrusted-content handling: webhook payloads get a restricted toolset; MCP results are wrapped;
  memory writes are injection-scanned.
- Supply chain: all 92 direct dependencies exact-pinned, with a written rationale citing the
  Mini Shai-Hulud worm; `osv-scanner` and lockfile-diff in CI.

**Gaps for a customer deployment:**

- No TLS termination anywhere — HTTPS is assumed to be someone else's job.
- No AWS Secrets Manager / Parameter Store integration.
- No RBAC in the dashboard — authenticated means fully privileged.
- No audit log of *human* actions (who approved what, who changed config).
- `tasks.tenant` exists but is **a filter, not an enforced boundary**: no row-level checks, no
  organization concept in sessions or memory. Today the real boundary is the profile directory.

---

## 8. Current deployment

- `Dockerfile` (26 KB, multi-stage, hadolint-linted) with s6-overlay supervision, `cont-init.d`
  hooks for UID/GID remapping and profile reconciliation, and services dropping privileges via
  `s6-setuidgid`.
- `docker-compose.yml`: `gateway` + `dashboard`, `network_mode: host`, volume `~/.hermes:/opt/data`,
  dashboard pinned to localhost.
- Installers for Linux/macOS/WSL2/Termux/native Windows; a Nix flake; extensive CI
  (`ci.yaml` with detect/tests/tests-os/lint/js-tests/installer-tests/rust-tests/osv-scanner/
  uv-lockfile, plus e2e installer workflows for macOS, Linux and Windows).

**What is missing for the PRD, completely:**

| Requirement | Status |
|---|---|
| Infrastructure as code | **None.** Zero `.tf`, `cdk.json` or CloudFormation files |
| HTTPS / TLS | None |
| Automated backups | None (manual `hermes backup` only) |
| CloudWatch integration | None |
| Secrets Manager | None |
| RDS / ElastiCache | Not applicable — SQLite (see §6) |
| S3 | `boto3` present only for Bedrock auth; no object storage integration |
| Deployment documentation | `docker.md` and `aws-bedrock.md` only — Bedrock is *inference*, not hosting |

Also relevant to a customer deployment: the fork carries a Nous Research free-tier onboarding
path (`HERMES_GUEST_ONBOARDING`), auto-update checks, telemetry config and Nous branding. All of
these must be disabled in a business distribution.

---

## 9. Current testing

**A genuine strength: 3,991 test files.** Organised by subsystem — `agent/`, `cli/`, `gateway/`,
`hermes_cli/`, `tools/`, `hermes_state/`, `plugins/`, `providers/`, `security/`, `e2e/`,
`integration/`, `conformance/`, `perf_guards/`, `docker/`, `install/`.

Tests exist for the things that matter here: subagent lifecycle, subagent failure notices,
supervisor recovery, delegation timeouts, worktree isolation, approval flows, state repair.

**Missing:** multi-agent coordination end-to-end (supervisor → several agents → aggregate),
anything for a knowledge layer, and anything for a deployment/IaC layer.

---

## 10. Existing reusable components — keep as Hermes Core

Everything in this table is business-agnostic today and should be preserved unchanged.

| Capability | Module(s) | Verdict |
|---|---|---|
| Task kernel | `hermes_cli/kanban_db.py` | **Production-ready.** Do not touch |
| Dispatcher | `hermes_cli/kanban_db_dispatch.py` | **Production-ready.** Extend via `spawn_fn` hook |
| Swarm topology | `hermes_cli/kanban_swarm.py` | Ready — the supervisor's foundation |
| Tool registry | `tools/registry.py`, `toolsets.py` | Ready |
| Approval gate | `tools/approval*.py` | Ready; needs a policy layer above it |
| Provider abstraction | `providers/`, 40 plugins, Bedrock adapter | **PRD §19 already satisfied** |
| Memory (3-layer) | `tools/memory_tool*`, `context_engine`, FTS5 | **PRD §12 largely satisfied** |
| MCP | `mcp_serve.py`, `tools/mcp_*`, 65 integrations | Ready |
| Cron | `cron/`, `tools/cronjob_tools.py` | Ready |
| Logging + redaction | `hermes_logging.py` | Ready; needs a JSON formatter |
| Usage accounting | `hermes_state_usage.py` | Ready; needs budget enforcement |
| Dashboard + auth | `hermes_cli/web_routers/`, `web/`, `plugins/dashboard_auth/` | Ready; needs RBAC |
| Profiles + distributions | `hermes_cli/profiles.py`, `profile_distribution.py` | **The template mechanism already exists** |
| Skills | `skills/`, 199 SKILL.md | Ready — the knowledge-authoring surface |
| Docker image | `Dockerfile`, `docker/` | Ready for single-node production |

---

## 11. Components that need refactoring

Ordered by how much they block the PRD.

1. **Agent definition — additive, not a refactor.** Introduce a declarative `AgentSpec` (YAML)
   that *materialises* a profile. The profile mechanism stays; we add the declarative front door.
2. **Permission model — extend, do not replace.** Add a policy resolver that maps
   (agent, tool, action) → allow / require-approval / deny, and feeds the existing approval gate
   and toolset resolution. The command-shaped floors stay as the last line of defence.
3. **Distributions — generalise from one profile to a set.** Today `distribution.yaml` packages
   one profile. A business template needs N agents + a board + workflows + knowledge config.
   This is a manifest extension plus a multi-profile installer, not a new engine.
4. **Logging — add a JSON formatter and correlation IDs.** `hermes_logging.py` is text-oriented.
   `task_runs` + `task_events` already give a per-task audit trail; what is missing is an ID that
   spans gateway → agent → tool → model call. Roughly 70% of the PRD's §14 trace is achievable
   today; a request-id contextvar plus a JSON handler closes the rest.
5. **Config — add a validated overlay.** `config.yaml` is a large untyped YAML dict read through
   `cfg_get`. Do not rewrite it; add a typed (pydantic) platform config that validates the
   business layer and compiles down into profile `config.yaml` files.
6. **Very large modules — treat as no-go zones.** `agent/auxiliary_client.py` (7,588 lines),
   the platform adapters (6–7k each), `hermes_cli/gateway.py` (6,314), `cli.py` (4,635). These
   are fragile by size and by upstream-merge cost. Avoid editing them; where a hook is genuinely
   needed, add the smallest possible one and record it (see §15, risk 1).

---

## 12. Missing capabilities

| # | Capability | PRD § | Size | Notes |
|---|---|---|---|---|
| 1 | Declarative agent definitions | 4 | M | The keystone — everything else composes on it |
| 2 | Supervisor / orchestrator agent | 3 | M | Build on `kanban_swarm` + `kanban_decompose` |
| 3 | Business template packaging | 5 | M | Extend `distribution.yaml` to multi-profile |
| 4 | Knowledge / retrieval layer | 8 | **L** | **Largest net-new build.** Nothing exists |
| 5 | Business-action permissions | 21 | M | Policy layer above the approval gate |
| 6 | Declarative approval requirements | 11 | S | Gate exists; needs config-driven triggers |
| 7 | Token / cost budgets and kill switch | 20 | S | Usage is tracked; nothing enforces a ceiling |
| 8 | Structured JSON logs + correlation IDs | 14 | S | |
| 9 | CloudWatch shipping | 14 | S | |
| 10 | Infrastructure as code | 17 | M | Nothing exists |
| 11 | HTTPS / TLS termination | 15 | S | ALB + ACM |
| 12 | Automated backups | 15 | S | SQLite `.backup` → S3 |
| 13 | AWS Secrets Manager integration | 18 | S | A new secret source alongside Bitwarden/1Password |
| 14 | Organization / customer config model | 6 | S | Branding, labels, notification channels |
| 15 | Dashboard RBAC + human audit log | 22 | M | Currently all-or-nothing |
| 16 | Enforced tenancy boundary | 7 | **Deferred** | One deployment per customer instead — see §13.5 |

---

## 13. Proposed target architecture

### 13.1 The governing principle

> **Add a business layer. Do not modify the core. Compose what exists.**

Every PRD requirement is satisfied either by an existing primitive, or by new code in a new
top-level package that reaches core only through documented extension points.

### 13.2 Layout

```
hermes-agent/
├── <existing core — unchanged>
│
├── platform/                     ← NEW: the entire business layer
│   ├── config/                   Typed schema: organization, agents, tools,
│   │                             knowledge, permissions, limits, branding
│   ├── agents/                   AgentSpec loader + profile materialiser
│   ├── supervisor/               Decompose → route → Kanban → collect
│   ├── knowledge/                Ingest → extract → chunk → index → retrieve
│   ├── policy/                   Permission + approval policy resolver
│   ├── templates/                Business templates
│   │   ├── customer-support/
│   │   └── operations/
│   ├── observability/            JSON formatter, correlation IDs, CloudWatch
│   └── cli/                      `hermes platform ...` subcommands
│
├── deploy/                       ← NEW
│   ├── terraform/                Single-node AWS, fully variable-driven
│   └── docker/                   Production compose overlay
│
└── docs/platform/                ← NEW (this document lives here)
```

### 13.3 How each PRD concept maps onto an existing primitive

| PRD concept | Implementation | New code? |
|---|---|---|
| Agent | `AgentSpec` YAML → materialised profile | Loader only |
| Agent isolation | Profile dir + worker process + git worktree | **None** |
| Supervisor | A profile running a decompose/route loop | Loop only |
| Task system | Kanban kernel, unchanged | **None** |
| Delegation | Kanban card with `assignee` = agent id | **None** |
| Concurrency safety | `BEGIN IMMEDIATE` CAS claim | **None** |
| Human approval | Existing gate + declarative policy | Policy layer |
| Business template | Extended multi-profile distribution | Manifest + installer |
| Versioning / upgrade | `distribution.yaml` + owned/user-owned split | **None** |
| Tools | Registry + toolsets, scoped per agent | Scoping only |
| Integrations | MCP `mcp.json` per agent | **None** |
| Model abstraction | Provider plugins | **None** |
| Memory | 3-layer memory, scoped per profile | Scoping only |
| Knowledge | New subsystem | **All of it** |
| Error handling | Circuit breaker, blocks, runs | **None** |
| Observability | Existing logs + JSON + correlation IDs | Formatter |
| Cost control | Usage rows + new budget enforcer | Enforcer |
| Deployment | Existing Docker + new Terraform | Terraform |

### 13.4 The agent specification (proposed)

```yaml
# platform/templates/customer-support/agents/support.yaml
id: customer-support
name: Customer Support Agent
role: customer_support
enabled: true

model:
  provider: ${HERMES_PROVIDER:-bedrock}     # never hard-coded
  name: ${HERMES_MODEL}
  reasoning_effort: medium

instructions: prompts/customer-support.md    # becomes the profile's SOUL.md

tools:
  toolsets: [web, file, knowledge]
  allow: [crm_lookup, ticket_create, email_send]
  deny:  [terminal, execute_code]            # compiles to approvals.deny floors

knowledge:
  sources: [company_handbook, product_docs]  # per-agent scoping, not global

permissions:
  - read_customers
  - create_ticket

approval:
  required_for: [refund, account_deletion, email_send_external]

limits:
  max_concurrent_tasks: 3        # → kanban.max_in_progress_per_profile
  max_task_runtime_seconds: 900  # → tasks.max_runtime_seconds
  max_retries: 2                 # → tasks.max_retries
  max_turns: 60                  # → agent.max_turns
  daily_token_budget: 2_000_000  # → new enforcer

delegation:
  may_assign_to: [knowledge-agent, ticket-agent]
```

The loader compiles this into a profile directory. **No orchestration code changes** to add an
agent, and **no core code changes** to add a template — which is PRD acceptance scenarios A, B
and C, met by construction.

### 13.5 Multi-tenancy: the explicit recommendation

**One customer → one deployment → the customer's AWS account. Do not build multi-tenancy now.**

The PRD itself prefers this (§7), and the architecture supports it naturally because the profile
directory and its SQLite files are already a hard boundary. What we will do instead, at near-zero
cost, is *avoid foreclosing* it:

- Keep `tasks.tenant` populated with the organization id from day one.
- Put the organization id on every structured log line.
- Never add new code that assumes a single global `$HERMES_HOME`.

That keeps the door open without paying for a door we are not walking through.

### 13.6 AWS target — deliberately small

```
Route 53 → ACM → Application Load Balancer (HTTPS)
                          │
              ┌───────────┴───────────┐
              │   EC2 (or one ECS task)│
              │  ┌──────────────────┐  │
              │  │ hermes gateway   │  │
              │  │ hermes dispatcher│  │
              │  │ hermes dashboard │  │
              │  └────────┬─────────┘  │
              │      EBS (gp3)         │  ← SQLite lives here
              └───────────┬────────────┘
                          │
   S3 (backups, documents, attachments) · Secrets Manager · CloudWatch Logs
                          │
             Bedrock or an external model provider
```

No Kubernetes, no microservices, no event bus, no service mesh, no RDS, no ElastiCache —
because §6 means SQLite is the store, and nothing in the requirements justifies the rest. IaC in
**Terraform**, chosen over CDK because the deployment path has no Node/TypeScript build today and
Terraform hands over to a customer's AWS account more cleanly.

---

## 14. Migration plan

Each phase ends in something demonstrable, and no phase requires the next one to exist.

### Phase 1 — Core stabilisation (2–3 weeks)
Typed platform config schema · `AgentSpec` loader and profile materialiser · permission/approval
policy resolver feeding the existing gate · token-budget enforcer with a hard kill switch ·
JSON log formatter + correlation-ID propagation · `hermes platform` CLI. **Core edits: only
additive hook points, each recorded.** *Exit:* an agent defined in YAML runs a Kanban task.

### Phase 2 — Template engine (2–3 weeks)
Multi-profile distribution manifest · installer that materialises N agents + a board · supervisor
loop on `kanban_decompose`/`kanban_swarm` · the two demonstration templates (Customer Support,
Operations). *Exit:* `hermes platform install customer-support` produces a working multi-agent
deployment.

### Phase 3 — Knowledge (2–3 weeks)
Ingestion (reuse `firecrawl-anydoc` extraction already in core) · chunking · **SQLite FTS5 index
first — no vector database until retrieval quality proves it necessary** · per-agent source
scoping · a `knowledge_search` tool · injection scanning on every retrieved chunk via the
existing `threat_patterns` scanner. *Exit:* an agent answers from a customer PDF with citations.

### Phase 4 — AWS deployment (2 weeks)
Terraform for the §13.6 topology · S3 backup of SQLite via the online `.backup` API on a schedule
· Secrets Manager as a new secret source · CloudWatch log shipping · ALB/ACM HTTPS · a deployment
runbook. *Exit:* a new customer account goes from empty to running Hermes from variables alone.

### Phase 5 — Testing (1.5–2 weeks)
Unit (agent spec, policy resolution, budget enforcement, template loading) · integration
(materialisation, dispatch, knowledge retrieval) · **end-to-end multi-agent coordination** ·
failure tests (model timeout, tool failure, DB lock, invalid config, duplicate claim, budget
exhaustion) · security tests (cross-agent tool access denial, injection through knowledge).

### Phase 6 — Demo (1 week)
A seeded environment showing request → supervisor → agents → Kanban → tools → knowledge →
human approval → result → trace.

**Total: ~11–15 developer-weeks for one developer.**

Sequencing note: Phases 1 and 2 are the commercial unlock (templates deployable, no core edits).
Phase 3 is the largest single build and the most uncertain. Phase 4 is small but has a hard
dependency on a real AWS account to test against.

---

## 15. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | **Upstream divergence.** This is a fork of a fast-moving repo. Every core edit compounds merge cost | **High** | Additive `platform/` package. Maintain `docs/platform/CORE_PATCHES.md` listing every core file touched and why. Treat that list as a budget to be kept near zero |
| 2 | **SQLite ceiling.** Single-node only; no horizontal scale | **High** | Document explicitly in sizing. Single node is sufficient for the stated use cases. Do **not** pre-build a DB abstraction |
| 3 | **Kanban v2 collision.** `workflow_template_id` / `current_step_key` are reserved for an upstream router | Medium | Do not build a competing workflow engine. Express workflows as `task_links` DAGs + swarm shape |
| 4 | **Process cost per agent.** Each worker is a full Hermes process (hundreds of MB) | Medium | "Hundreds of agents" = hundreds of *definitions*, not concurrent processes. Concurrency caps already exist; state the distinction in customer sizing |
| 5 | **Prompt injection via customer documents** | **High** | Reuse `threat_patterns` on every retrieved chunk; keep knowledge retrieval read-only; never let retrieved text reach the approval path |
| 6 | **Cost runaway** | Medium | Phase 1 budget enforcer is non-negotiable. `delegation.max_concurrent_children` warns that >10 multiplies cost linearly |
| 7 | **Branding, telemetry, auto-update and the Nous free-tier path in customer deployments** | Medium | A "business" base distribution that disables update checks, telemetry, guest onboarding, achievements/pet plugins and Nous branding |
| 8 | **Surface sprawl.** Desktop app, Termux, evals, batch trajectory tooling are all irrelevant here | Low | Leave in the fork, exclude from the production image and disable in the base distribution |
| 9 | **Dashboard has no RBAC** — authenticated equals fully privileged | Medium | Scope Phase 1 to a read-only customer role; full RBAC is post-Phase-5 |
| 10 | **Over-engineering pressure** (PRD §28) | Medium | The phase exit criteria above are the guardrail: nothing ships that no exit criterion requires |

---

## 16. Estimated implementation phases

| Phase | Scope | Effort | Depends on |
|---|---|---|---|
| 0 | Repository audit | **Complete** | — |
| 1 | Core stabilisation: config, agent spec, policy, budgets, logging | 2–3 wks | Approval of §13 |
| 2 | Template engine + supervisor + 2 templates | 2–3 wks | Phase 1 |
| 3 | Knowledge system (FTS5 first) | 2–3 wks | Phase 1 |
| 4 | AWS deployment (Terraform, single-node) | 2 wks | Phase 2; a test AWS account |
| 5 | Testing across all layers | 1.5–2 wks | Phases 2–4 |
| 6 | Demo environment | 1 wk | Phase 5 |
| | **Total** | **11–15 wks** | |

Phases 3 and 4 can run in parallel if you want to compress the calendar; they touch disjoint code.

---

## Decisions needed before Phase 1

1. **Additive `platform/` package rather than a core refactor** — confirm. This is the
   load-bearing decision; everything else follows from it.
2. **One deployment per customer; no multi-tenancy build now** (§13.5) — confirm.
3. **Single-node AWS with SQLite on EBS; no RDS in v1** (§6, §13.6) — confirm, and confirm you
   accept the scale-out implication.
4. **Terraform over CDK/CloudFormation** (§13.6) — confirm.
5. **Knowledge starts on FTS5, no vector database until retrieval quality demands one** (Phase 3)
   — confirm.
6. **Customer Support + Operations as the two demonstration templates** — confirm, or name
   different ones.
7. **Upstream tracking policy** — do you intend to keep merging from `NousResearch/hermes-agent`?
   The answer changes how strictly we enforce risk 1. If yes, the core-patch budget is near zero.
   If no, some refactors that are currently off the table become viable.

Nothing will be built until these are settled.
