# Architecture Boundaries

**Authoritative. Read this before adding, moving, or renaming anything.**

This repository is a fork of the upstream Hermes Agent runtime, extended with a platform layer
that turns it into a reusable, white-labelable enterprise agent platform. Those two things are
kept strictly apart. This document says where the line is, why it is there, and what happens if
it is crossed.

Working name of the product is **NOVA**. It is a *display name held in configuration*, never an
identifier in code. See §4.

---

## 1. The two halves

| | Upstream-owned | Platform-owned |
|---|---|---|
| **Paths** | everything not listed opposite | `nova/`, `deploy/`, `docs/platform/`, `customer/` |
| **Origin** | `NousResearch/hermes-agent` | this fork |
| **Change policy** | **patch budget — near zero** | free |
| **Merge behaviour** | merges from upstream land here | upstream never touches these paths |
| **May import** | never imports `platform.*` | may import upstream modules |

The dependency arrow points **one way**: platform code imports Hermes; Hermes code never imports
platform code and must never learn that the platform exists.

That single rule is what keeps `git merge upstream/main` a routine operation. A conditional in
core that reads a platform config is not a small compromise — it inverts the arrow and makes
every future upstream change a potential conflict.

---

## 2. Do not touch

These are load-bearing for mergeability, correctness, or live installations. The audits in
`HERMES_PLATFORM_AUDIT.md` and `BRAND_SURFACE_AUDIT.md` have the measured detail.

### 2.1 Modules — never rename, never restructure

| Module | Why |
|---|---|
| `hermes_cli/` (481 files, imported 10,150×) | Renaming rewrites every import and conflicts with nearly every upstream commit |
| `hermes_state*.py` (22 modules) | Session, message, FTS and usage storage |
| `hermes_cli/kanban_db.py`, `kanban_db_dispatch.py` | The task kernel and dispatcher. Production-grade; extend via the `spawn_fn` hook, never by editing |
| `agent/`, `run_agent.py`, `cli.py` | The agent loop and its surfaces |
| `gateway/`, `gateway/platforms/` | Gateway and ~20 platform adapters |
| `tools/`, `tools/approval*.py` | Tool registry and the approval gate. Extend with a policy layer above it |
| `hermes_constants.py`, `hermes_logging.py` | Home resolution, log redaction |
| `tests/` (3,991 files) | Not shipped. Editing them is pure merge cost |

### 2.2 Identifiers — never rename; alias instead

| Identifier | Breaks on rename |
|---|---|
| `HERMES_*` environment variables (**225 distinct**) | Every existing deployment, container, systemd unit and CI job |
| `$HERMES_HOME` / `~/.hermes` | All state: `state.db`, `kanban.db`, credentials, profiles, skills |
| `hermes-*` toolset names | Written into `config.yaml`; a rename silently disables toolsets |
| `X-Hermes-*` HTTP headers (9) | Gateway / sidecar / relay wire protocol |
| `hermes://` URL scheme (7 routes) | OS-registered deep links, plugin and blueprint install links |
| `hermes_agent.plugins` entry-point group | Every pip-installed third-party plugin |
| `compat_manifest.json` import paths | External plugins using the compatibility shim |
| Console scripts `hermes`, `hermes-agent`, `hermes-acp` | Scripts, cron entries, runbooks, muscle memory |
| PyPI distribution `hermes-agent` | How the runtime is installed and updated |

### 2.3 Upstream references — never rewrite

`github.com/NousResearch/*`, `nousresearch.com`, and the upstream remote. Required technically
(install, update, issue links) and in some cases legally (MIT attribution).

### 2.4 The package is named `nova/`, never `platform/`

A top-level `platform/` package **shadows the Python standard library's `platform`
module**, which 40 upstream modules import for OS detection. The failure is
`AttributeError: module 'platform' has no attribute 'system'` in every process started
from the repository root — confusing, wide, and nowhere near the change that caused it.

`scripts/check_protected_identifiers.py` fails if `platform/__init__.py` ever appears.
Do not "tidy" the package name back.

### 2.5 The one that will bite you

> **`Hermes-4-405B`, `NousResearch/Hermes-3-Llama-3.1-70B`, `nousresearch/hermes-4-405b` are
> large language models — a different product that happens to share a name with the runtime.**

189 occurrences, in provider profiles, the model-routing regexes in `hermes_cli/model_switch.py`,
and auxiliary-client configuration. A rename here silently breaks model resolution, and it
surfaces as a 404 from an inference provider rather than as a failing test.

`scripts/check_protected_identifiers.py` enforces `[Hh]ermes-[0-9]` as an absolute exclusion.
Never add an exception to it.

---

## 3. Extension points — use these instead of editing core

Every one of these is an upstream-supported seam. Reaching for one of these instead of a core
edit is the single most useful habit in this repository.

| Need | Seam | Core edit? |
|---|---|---|
| New model backend | `plugins/model-providers/`, `hermes_agent.plugins` entry point | none |
| New agent | A **profile** — `$HERMES_HOME/profiles/<name>/` with its own config, persona, MCP, skills | none |
| Packaged, versioned agent bundle | A **distribution** — `distribution.yaml` with install/update and an owned vs. user-owned split | none |
| Branding: CLI/TUI name, welcome, prompt, banner, colors | A **skin** — `$HERMES_HOME/skins/<name>.yaml`; `load_skin()` prefers user skins over built-ins | none |
| Branding: gateway and slash-command strings | A **locale overlay** — `locales/*.yaml` behind `agent/i18n.py` | none |
| Agent persona and voice | `SOUL.md` — distribution-owned, per profile | none |
| Tool scoping per agent | `enabled_toolsets` + registry availability checks | none |
| Integrations | Per-profile `mcp.json` | none |
| Scheduled work | `cron/`, blueprints in skill frontmatter | none |
| Worker spawn behaviour | The dispatcher's `spawn_fn` hook | none |
| Dashboard authentication | `plugins/dashboard_auth/` | none |
| Tracing | `plugins/observability/` | none |
| Giving one agent a new tool | A **plugin** in `<profile>/plugins/<name>/` with `register(ctx)` + `provides_tools`. A worker runs with its profile as the runtime home, so plugin discovery scopes it to exactly that agent | none |
| Document extraction (PDF, Office, OpenDocument) | The runtime's own extractors, borrowed through `AgentRuntime.extract_text()` — imported **lazily, inside an adapter** | none |
| Creating work for an agent | `kanban_db.create_task` — the runtime's own API for its own shared board (`kanban.db`, not the conversation state on `NEVER_WRITE`), called lazily from an adapter. Never raw SQL against that schema | none |
| Customer model endpoints and credentials | `deployment.yaml` → the adapter's provider translator. Endpoints and the **names** of credential variables in `<profile>/config.yaml`; the values in `<profile>/.env`, which is on `NEVER_WRITE` and owned by the operator | none |

**If you are about to edit core, you have probably missed a seam above. Check first.**

---

## 4. Identity and branding

Branding is **tenant configuration compiled into existing surfaces** — never a string in code,
and never a runtime lookup from core.

- One source: the tenant's `identity.yaml`.
- Projected by `nova/identity/` into a skin, a locale overlay, a `SOUL.md`, and theme tokens
  served to the platform dashboard.
- **Core never imports the identity layer.** It reads a skin file and a locale file, which it
  already does for its own reasons.

`NOVA` is a default display value in configuration. It must never appear as a module name, an
environment-variable prefix, a database column, a class name, or a wire identifier. The same
build serves `NOVA AI Workforce`, `Acme Intelligence` and `Bristol Foods AI Workforce` with no
source change.

---

## 5. The control-plane boundary

**The Hermes dashboard is not, and will not become, the customer dashboard.**

This is a factual constraint, not a preference. `hermes_cli/web_server.py` mounts its FastAPI app
**inside the gateway process**, and `_spawn_hermes_action()` acts on the runtime by launching
`hermes <subcommand>` subprocesses that inherit `os.environ`. It is a same-host, fully-privileged
engineering tool, correct for its purpose and unsuitable for customer operators.

The customer path is:

```
Customer operator
   -> Platform Dashboard        (platform-owned React app)
   -> Platform Control API      (platform-owned, separate process, typed commands)
   -> Agent Platform            (nova/agents, nova/policy)
   -> Hermes Runtime            (upstream — never reached directly from the UI)
```

Rules:

1. The platform dashboard talks **only** to the Platform Control API. It never calls a Hermes
   endpoint and never learns a Hermes path, profile directory or table name.
2. The Control API reads runtime state through **read models** — a read-only connection to
   `kanban.db`, `task_events`, `task_runs`, and the usage tables — never by scraping logs.
3. The Control API writes through **narrow typed commands** that map onto existing kernel
   functions. Arbitrary subprocess spawn from a web handler is forbidden. *Implemented in
   Phase 8*: four verbs (`release`, `reject`, `resume`, `annotate`) plus objective
   submission, each asking the runtime for a transition it already owns rather than writing
   a status — a control plane that can write any state can write an inconsistent one.
4. The Hermes dashboard and CLI stay, bound to localhost, for engineering and debugging.
   *Channels follow the same rule* (Phase 9): the runtime's 22 platform adapters, its webhook
   verification and its delivery ledger stay exactly where they are, and NOVA compiles a
   customer-facing declaration into the gateway configuration they already read. All of
   NOVA's runtime-specific channel knowledge lives in one module,
   `nova/runtime/hermes/channels.py`; `nova/channels/` contains no runtime vocabulary.
5. A future `nova` CLI covers platform and deployment operations only. **Do not re-implement the
   Hermes CLI.**

---

## 6. The deployment boundary

One customer, one deployment, inside **the customer's own AWS account**. The deployment provisions
the platform *and* the agent runtime together — never the platform alone with the runtime
elsewhere. Amazon Bedrock is the inference layer, not the hosting target.

Agent access to customer systems is explicit and least-privilege:

- The runtime's task role carries **no customer-data permissions by default**.
- Each integration is a separately-scoped IAM role, assumed per integration, whose trust policy
  names the runtime role. Adding an integration is an auditable IAM change the customer's own
  security team can review.
- **Agents never receive broad access to the customer's AWS account.**
- Deployment is Terraform the customer's DevOps team runs with their own credentials. **We never
  require or hold root or admin credentials.**

This is implemented, not aspirational: the root module is `deploy/aws`, and `nova deploy render`
derives the per-integration roles from the tenant bundle so that what an agent may reach cannot
drift from what was declared. The three refusals that keep the last bullet true — no wildcard
action, no `"*"` resource, no `iam`/`sts`/`organizations`/`account`/`kms` action on an
integration — are enforced twice, in the generator and again in the module's own variable
validation, because the tfvars file can be hand-edited. `deploy/aws/README.md` records what has
been verified and what has not.

Storage is SQLite on an attached volume, so v1 is deliberately **single-node**. This is a
documented constraint, not an oversight: SQLite's guarantees hold across processes on one host
and not across hosts, and SQLite on EFS/NFS risks corruption.

---

## 7. The patch budget

Every edit to an upstream-owned file is recorded in `docs/platform/CORE_PATCHES.md` with one line
saying why. The target is a handful of small, additive hook points.

**The list is reviewed at every upstream merge. If it is growing, something has gone wrong.**

Before adding an entry, answer: *which extension point in §3 did I fail to find?*

---

## 8. Upstream update procedure

1. Fetch upstream.
2. Merge or rebase.
3. Run the **upstream** test suite.
4. Run the **platform** test suite.
5. Run integration and security tests.
6. Release a new platform version.

Platform paths do not participate in step 2 — upstream has never heard of them. Brand-owned files
resolve automatically via the `merge=brandours` driver in `.gitattributes`. Only the hook points
in the patch budget can genuinely conflict.

---

## 9. Guardrails

| Guardrail | Enforces |
|---|---|
| `scripts/check_protected_identifiers.py` | §2.2 and §2.4 — no rename of protected identifiers, absolute `[Hh]ermes-[0-9]` exclusion |
| Import-direction check | §1 — no upstream module imports `platform.*` |
| `test_platform_layer_never_imports_the_runtime_at_module_level` | §1 — NOVA loads with only the stdlib and PyYAML, adapters included |
| `test_only_an_adapter_may_import_the_runtime_lazily` | §1 — a borrowed runtime capability stays inside `nova/runtime/<adapter>/` |
| `test_the_adapter_packages_still_import_without_their_runtime` | §1 — proves the lazy import stayed lazy, in a subprocess |
| `tests/platform/test_plugin_activation.py` | §3 — a plugin NOVA installs is enabled, registers its hook, and is permitted by the policy guarding it |
| Patch-budget check | §7 — an upstream-owned file changed without a ledger entry fails CI |
| `merge=brandours` in `.gitattributes` | §8 — brand-owned files resolve without manual attention |

---

## 10. Five things that would break this architecture

1. Renaming `hermes_cli` or any module in §2.1 to make the repository look branded.
2. Adding a platform import, or a platform-aware conditional, to an upstream-owned file.
3. Pointing the customer dashboard at a Hermes endpoint because it was quicker.
4. Replacing a `HERMES_*` variable or `~/.hermes` instead of aliasing it.
5. A find-and-replace that catches `Hermes-4-405B`.
6. Letting retrieved customer documents into the system prompt, or scoping a knowledge
   search by telling the model which corpora it may read rather than filtering in SQL.
7. Writing a second decomposer, scheduler or task store because the runtime's was not
   shaped exactly like NOVA's model of one — or writing `kanban.db` with raw SQL.
8. Shipping a control whose only evidence is a unit test that calls NOVA's own code. Every
   seam with the runtime — plugin enablement, hook registration, tool availability — has to
   be proven in a live worker, because each of those failed silently once already
   (`LIVE_RUN.md`).
9. Writing a credential anywhere NOVA owns — a profile config, an audit record, a log line —
   or adding a command that takes one. NOVA writes the **name** of a secret and never its
   value; `<profile>/.env` belongs to the operator and is on `NEVER_WRITE` (`PHASE_6.md`).

If a change you are about to make matches one of these, stop and re-read §3.
