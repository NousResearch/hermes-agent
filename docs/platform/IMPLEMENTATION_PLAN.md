# Platform Implementation Plan

**Answers the ten questions in the approved product direction, then names the smallest Phase 1.**

Boundary rules live in `ARCHITECTURE_BOUNDARIES.md` and govern everything here. Evidence for the
claims about the existing codebase is in `HERMES_PLATFORM_AUDIT.md` and `BRAND_SURFACE_AUDIT.md`.

Product working name: **NOVA** — a configuration value, never an identifier.

---

## 1. Existing Hermes extension points we reuse

Each row is an upstream-supported seam, verified in the audit. Using these is what keeps the
patch budget near zero.

| Platform need | Seam we ride | Core change |
|---|---|---|
| An agent | **Profile** — `$HERMES_HOME/profiles/<name>/`: own config, `SOUL.md`, `mcp.json`, skills, cron, state, credentials | none |
| Agent isolation | Dispatcher spawns `hermes -p <assignee>` — process + filesystem + credential isolation | none |
| Packaged agent set, versioned, upgradable | **Distribution** — `distribution.yaml`, install/update, `env_requires`, distribution-owned vs. user-owned split that protects customer data | none |
| Task coordination | **Kanban kernel** — WAL + `BEGIN IMMEDIATE` compare-and-swap claiming, `task_runs`, heartbeats, circuit breaker, typed blocks | none |
| Worker spawn control | Dispatcher `spawn_fn` hook | none |
| CLI/TUI branding | **Skin** — `load_skin()` reads `$HERMES_HOME/skins/<name>.yaml` *before* built-ins; carries `agent_name`, `welcome`, `goodbye`, `response_label`, `prompt_symbol`, `help_header`, `banner_logo`, colors | none |
| Gateway string branding | **Locale overlay** — `locales/*.yaml` behind `agent/i18n.py`, falls back English → key | none |
| Agent voice | `SOUL.md`, distribution-owned per profile | none |
| Tool scoping | `enabled_toolsets` + registry `check_fn` | none |
| Integrations | Per-profile `mcp.json`; 65 bundled MCP integrations | none |
| Model backends | `plugins/model-providers/` + `hermes_agent.plugins` entry point; 40 profiles incl. Bedrock | none |
| Human approval | `tools/approval*.py` — floors that fire before any bypass, gateway round-trip, MCP elicitation | none |
| Dashboard auth | `plugins/dashboard_auth/` — PKCE, JWKS, refresh | none |
| Tracing | `plugins/observability/` | none |
| Usage accounting | `hermes_state_usage.py` — per-model token rows | none |
| Contributor routing | `AGENTS.md` routing table → per-area `AGENTS.md` | **1 table row** |

## 2. Files and modules that remain untouched

Enumerated in `ARCHITECTURE_BOUNDARIES.md` §2. In summary: `hermes_cli/`, `hermes_state*`,
`agent/`, `run_agent.py`, `cli.py`, `gateway/`, `tools/`, `tests/`, the 225 `HERMES_*` variables,
`~/.hermes`, the toolset names, the wire identifiers, the plugin entry-point group, the upstream
URLs, and the `Hermes-3`/`Hermes-4` model identifiers.

The kernel modules deserve a specific note: `kanban_db.py` and `kanban_db_dispatch.py` are the
most valuable code in the repository and are already correct under concurrency. **We extend them
through `spawn_fn` and read models. We do not edit them.**

## 3. Platform-owned modules we create

```
platform/
├── identity/      Branding config + projectors (skin, locale, persona, theme tokens)
├── config/        Typed tenant schema: organization, agents, integrations,
│                  knowledge, permissions, limits — validation and compilation
├── agents/        AgentSpec → profile materialiser
├── policy/        (agent, tool, action) → allow | require-approval | deny
├── control/       Platform Control API: read models + narrow typed commands
├── supervisor/    Decompose → route → Kanban → collect (built on kanban_swarm)
├── knowledge/     Ingest → extract → chunk → index (FTS5 first) → retrieve
├── observability/ JSON formatter, correlation IDs, CloudWatch shipping
├── budget/        Token and cost ceilings with a hard kill switch
└── cli/           `nova` — platform and deployment operations only

deploy/
├── terraform/     Single-node AWS, fully variable-driven
└── docker/        Production compose overlay

customer/          A tenant bundle (git or S3), not committed here
```

Nothing in `nova/` is imported by anything upstream. The dependency arrow points one way.

## 4. Where the identity/branding layer lives

`nova/identity/`, and the architecture is **compile, don't look up**.

One source of truth per tenant:

```yaml
# customer/<tenant>/identity.yaml
product_name:   "Acme Intelligence"        # default: "NOVA AI Workforce"
company_name:   "Acme Corporation"
logo:           s3://acme-nova-assets/logo.svg
favicon:        s3://acme-nova-assets/favicon.png
theme:
  accent:       "#1f6f5c"
  surface:      "#ffffff"
support:
  email:        support@acme.example
  url:          https://help.acme.example
agents:
  customer-support:  { display_name: "Acme Support Assistant" }
  operations:        { display_name: "Acme Ops" }
messages:
  welcome:      "Welcome to Acme Intelligence."
```

Four **projectors** compile that into surfaces the runtime already reads:

| Projector | Output | Consumed by | Core change |
|---|---|---|---|
| Skin | `$HERMES_HOME/skins/<tenant>.yaml` | CLI, TUI, banner, prompt | none |
| Locale | locale overlay | Gateway, slash commands | none |
| Persona | per-profile `SOUL.md` | The agent's own voice | none |
| Theme | JSON from `GET /platform/v1/identity` | Platform dashboard, email templates | none |

**Core never imports `platform.identity`.** It loads a skin file and a locale file, which it
already does for its own reasons. That is what makes white-label free of patch-budget cost.

Rebranding a deployment is: edit `identity.yaml`, re-run the projectors, restart. No build, no
source change, no redeploy of a different image.

## 5. How the customer dashboard communicates with the runtime

**Why a new control plane is required, not preferred.** `hermes_cli/web_server.py` mounts its
FastAPI app *inside the gateway process*, and `web_server_gateway.py:325` `_spawn_hermes_action()`
acts on the runtime by launching `hermes <subcommand>` subprocesses inheriting `os.environ`. It is
a correct engineering tool and an unacceptable customer surface.

```
Customer operator
   │  HTTPS, OIDC session, RBAC
   ▼
Platform Dashboard          React. Knows no Hermes path, profile dir or table name
   │  GET/POST /platform/v1/...
   ▼
Platform Control API        FastAPI, separate process, typed request/response
   │
   ├── read models  ──────►  read-only connection: tasks, task_runs, task_events,
   │                         session_model_usage
   └── typed commands ────►  a fixed verb set over existing kernel functions
   ▼
Hermes Runtime              dispatcher, workers, gateway — never reached from the UI
```

**Read models** (read-only, no privileged path): agent inventory and status, task lists and
detail, run history with outcomes, failures, approvals queue, activity and audit trail from
`task_events`, usage and cost from the usage tables.

**Typed commands** (the complete v1 verb set — anything not listed is not possible through the
API): create task, comment, block, unblock, request review, approve, reject, cancel task, enable
agent, disable agent, pause agent (set its concurrency cap to zero), update agent config, update
branding, update organization settings.

Each maps onto an existing kernel function or a config write. **Arbitrary subprocess spawn from a
web handler is forbidden.** Process lifecycle (start/stop the dispatcher) goes through the
supervised service, not `Popen` from a request.

Transport is HTTP from day one even though both sides are same-host in v1 — so the boundary is
real, testable and relocatable later. The Hermes dashboard and CLI remain on localhost for
engineering.

## 6. How the AWS deployment boundary works

```
Customer AWS Account  (their account, their credentials, their review)
│
├── ALB + ACM ─────────────── HTTPS
├── EC2 / one ECS task
│     ├── Platform Control API
│     ├── Platform Dashboard (static)
│     ├── Hermes gateway
│     └── Hermes dispatcher + workers
├── EBS (gp3) ─────────────── SQLite: state.db, kanban.db
├── S3 ────────────────────── backups, documents, attachments, brand assets
├── Secrets Manager ───────── model credentials, integration credentials
├── CloudWatch ────────────── structured logs, metrics, alarms
└── IAM
      ├── runtime task role ───── no customer-data permissions
      └── integration roles ───── one per integration, assumed on demand
                                  trust policy names the runtime role
                                        │
                                        ▼
                          Customer systems: RDS, CRM, ERP, S3, internal APIs, analytics
```

Both halves deploy together — never the platform alone with the runtime elsewhere. **Bedrock is
the inference layer, not the hosting target.**

The IAM shape is the part a customer security team will actually review: the runtime role can do
nothing to customer data on its own; each integration is a separately-scoped role it may assume,
so adding an integration is an auditable IAM change rather than a permissions broadening. **No
root or admin credentials are ever requested or held** — deployment is a Terraform module the
customer's own DevOps runs.

Single-node in v1, because SQLite's guarantees hold across processes on one host and not across
hosts. Documented in the deployment guide, not discovered in production.

## 7. How customer/tenant configuration is represented

A **tenant bundle** — a versioned directory, delivered by git or S3, never committed to this repo:

```
customer/<tenant>/
├── organization.yaml     tenant id, legal name, region, contacts, locale, timezone
├── identity.yaml         branding (§4)
├── agents/*.yaml         one AgentSpec per agent
├── integrations/*.yaml   MCP servers, APIs, databases — each naming its IAM role
├── knowledge/*.yaml      sources and per-agent scoping
├── permissions.yaml      roles, business actions, approval requirements
├── limits.yaml           concurrency, timeouts, retries, token budgets
└── manifest.yaml         bundle version + minimum platform version
```

Validated by the typed schema in `nova/config/`, then **compiled** into profiles, skins,
locale overlays, `mcp.json`, toolset scoping and board configuration. A bad bundle fails
validation before anything is written — not halfway through materialisation.

Tenancy stance is unchanged from the platform audit: **one customer, one deployment**. We do not
build multi-tenancy, but we do not foreclose it — the tenant id is stamped on `tasks.tenant` and
on every structured log line from day one, and no new code assumes a single global `$HERMES_HOME`.

## 8. How white-label branding flows through the UI

1. The dashboard boots and calls `GET /platform/v1/identity`.
2. The Control API returns resolved identity: product name, company name, logo and favicon URLs
   (S3, presigned), theme tokens, support contact, agent display names.
3. The React app writes theme tokens to CSS custom properties on the root element and sets
   `document.title` and the favicon link. **No per-customer build, no rebuild, no separate image.**
4. Agent display names come from the AgentSpec, never from a profile directory name — so
   `customer-support` renders as "Acme Support Assistant" without the identifier changing.
5. Email and notification templates render from the same resolved identity object.

One deployment serves `NOVA AI Workforce`, `Acme Intelligence` or `Bristol Foods AI Workforce`
by changing `identity.yaml`.

## 9. How Hermes compatibility is preserved

**Add names beside the old ones; never replace.**

| Surface | Mechanism |
|---|---|
| Environment | `NOVA_FOO` resolved first, falls back to `HERMES_FOO`. All 225 keep working |
| Home directory | `$NOVA_HOME` → `~/.nova` → `$HERMES_HOME` → `~/.hermes` |
| CLI | `nova` added to `[project.scripts]`; `hermes` untouched |
| HTTP headers | Emit both, accept either |
| URL scheme | `nova://` registered in addition to `hermes://` |
| Plugin discovery | `nova.plugins` **and** `hermes_agent.plugins` |
| Toolsets | `nova-*` aliases resolving to the same definitions |
| PyPI | `hermes-agent` remains the installed distribution |

Aliases do not expire in v1. An existing Hermes installation keeps working unchanged, and
upstream mergeability is untouched because every one of these is additive.

Enforced by `scripts/check_protected_identifiers.py`, with `[Hh]ermes-[0-9]` as an absolute
exclusion so no sweep can ever reach the model identifiers.

## 10. How upstream updates continue to merge cleanly

Three mechanics, in order of leverage:

1. **New paths conflict with nothing.** `nova/`, `deploy/`, `docs/platform/` and `customer/`
   do not exist upstream. The overwhelming majority of platform code has a zero conflict surface
   by construction.
2. **Replace brand-owned files wholesale; never line-edit upstream prose.** A rewritten `README.md`
   is one conflict settled in one command; a `README.md` with 400 swapped words is 200 conflicts on
   every upstream docs commit. A `merge=brandours` driver in `.gitattributes` resolves these
   automatically.
3. **A patch budget.** `CORE_PATCHES.md` records every upstream-owned file we touch and why,
   reviewed at each merge. Growth is the signal that a seam was missed.

Update procedure: fetch → merge → upstream tests → platform tests → integration and security
tests → release a platform version.

---

# Phase 1 — the smallest useful implementation

**Goal:** prove the whole architecture end to end on the narrowest possible slice.

**Acceptance:** *a tenant bundle produces a branded CLI and a read-only customer dashboard listing
that tenant's agents and tasks — with **zero** entries in the patch budget beyond one `AGENTS.md`
routing row.*

Zero core patches is the point. If Phase 1 needs a core edit, a seam was missed.

### In scope

| # | Deliverable | Rides |
|---|---|---|
| 1 | `nova/config/` — typed tenant schema, loader, validation | new code |
| 2 | `nova/identity/` — branding config + **the skin projector only** | existing skins dir |
| 3 | `nova/agents/` — AgentSpec schema + profile materialiser | existing profiles dir |
| 4 | `platform/control/` — **read-only** Control API: `/identity`, `/agents`, `/tasks`, `/tasks/{id}`, `/health` | read-only SQLite |
| 5 | Minimal read-only dashboard consuming those five endpoints, themed from `/identity` | new code |
| 6 | `scripts/check_protected_identifiers.py` + CI wiring | existing guardrail pattern |
| 7 | Boundary docs, `nova/AGENTS.md`, patch-budget ledger | **delivered with this plan** |

Read-only is deliberate: there is no privileged write path to get wrong on day one, and the read
models are the part every later feature depends on.

### Explicitly out of scope for Phase 1

Supervisor; knowledge/RAG; write and control commands; approvals UI; Terraform; environment and
home-directory aliases (not needed until a NOVA-named install exists); the `nova` CLI; locale,
persona and theme projectors beyond the skin; cost enforcement; RBAC beyond a single admin role.

### Sequence

| Step | Work | Effort |
|---|---|---|
| 1.1 | Tenant schema + validation + one example bundle | 3–4 d |
| 1.2 | Identity model + skin projector + `nova platform brand apply` | 2–3 d |
| 1.3 | AgentSpec + profile materialiser + round-trip test | 3–4 d |
| 1.4 | Control API read models + the five endpoints | 4–5 d |
| 1.5 | Read-only dashboard, themed from `/identity` | 3–4 d |
| 1.6 | Guardrails + CI | 1–2 d |
| | **Total** | **~3 weeks** |

### How we will know Phase 1 worked

1. Two tenant bundles differing only in `identity.yaml` produce two visibly different branded
   deployments from one build.
2. `docs/platform/CORE_PATCHES.md` contains exactly one entry.
3. `git merge upstream/main` completes with conflicts only in files we replaced wholesale.
4. The dashboard makes no request to any Hermes endpoint — verifiable from its network log.
5. `scripts/check_protected_identifiers.py` passes, and fails when a test fixture renames
   `Hermes-4-405B`.

Criterion 4 is the one worth watching. It is the easiest to violate under time pressure and the
most expensive to undo later.
