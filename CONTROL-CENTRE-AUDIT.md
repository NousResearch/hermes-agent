# Control Centre audit

Written before any code changed. Every claim below was checked against the source, and the
file:line references are there so you can check them too.

The headline: **almost everything you asked for already exists in Hermes.** The Control
Centre's problem is not missing runtime capability — it is that NOVA exposes a narrow,
mostly read-only slice of it. Two exceptions are real and are called out in §5.

One finding changes the design of the whole task, so it is first.

---

## 0. The finding that shapes everything: where an agent's Soul actually lives

You asked for `SOUL.md` to be editable. Hermes has an endpoint for exactly that —
`PUT /api/profiles/{name}/soul` (`hermes_cli/web_routers/profiles.py:855`). **The Control
Centre must not use it**, and the reason matters.

An agent's Soul is a *derived* file. The real chain:

```
bundle/prompts/operations.md          ← the source of truth (you edit this)
  → AgentSpec.instructions              nova/spec/agent.py:383
  → build_persona()                     nova/runtime/hermes/materialize.py:415
      (adds tenant branding + the knowledge briefing)
  → <profile>/SOUL.md                   written by materialize
  → prompt_builder identity slot #1      agent/prompt_builder.py:1453
```

`nova apply` **overwrites** `<profile>/SOUL.md` from the bundle every time it runs. So an
edit written straight to the profile would work, persist, survive a reload — and then
vanish the next time anyone applied the bundle, with no error and no trace. That is worse
than not shipping the feature.

**Therefore:** Control Centre edits write to the *bundle*, and then run the existing apply
path. The bundle stays the single source of truth, the frontend never becomes authoritative,
and "did my change apply?" has a real answer — the materialiser reports it.

This also answers your §8 architecture requirement without inventing anything:

```
Control Centre UI → NOVA Control API → validate + write bundle → existing apply path
                                     → Hermes core → profiles / cron / gateway
```

---

## 1. Existing Control Centre capabilities

Eleven screens, of which **nine are read-only**.

| Screen | Reads | Can change anything? |
|---|---|---|
| Overview | health, agents, tasks, channels | no |
| Agents | agent list + detail | **no** — this is the gap you hit |
| Objectives | declared objectives | submit only |
| Work | the Kanban board | decide (release / changes / resume / note) |
| Approvals | blocked work | decide |
| Activity | audit decisions | no |
| Knowledge | corpora | no |
| Channels | declared connections | apply the whole declaration |
| Automations | schedules + governance | pause / resume / delete / declare |
| Policies | compiled policy | no |
| Usage | budget observations | no |

The entire write surface is **five routes** (`nova/control/auth.py:79-90`):
`/work/decide`, `/objectives/submit`, `/channels/apply`, `/automations/decide`,
`/automations/create`.

**What it displays but does not control** — your §1 question, answered directly:

- **Agents.** Name, description, model, tools, permissions, channels, knowledge and limits
  are all rendered. None is editable. There is no create, no delete, no duplicate.
- **Policies.** The compiled policy is shown in full. It can only be changed by editing
  `policy.yaml` on the server.
- **Knowledge.** Corpora and their grants are listed; grants cannot be changed.
- **Channels.** Shows the 6 providers NOVA knows about (§2) and can re-apply the existing
  declaration. It cannot add, configure, disconnect or test a connection.
- **Usage.** Shows token and duration figures. These are **observations, not limits** — the
  UI already says so, and that stays true.

## 2. Existing Hermes capabilities

### Channels — 22 plugins, with a machine-readable manifest each

`plugins/platforms/` contains: a2a, buzz, dingtalk, discord, email, feishu, google_chat,
homeassistant, irc, line, matrix, mattermost, ntfy, photon, raft, simplex, slack, sms,
teams, telegram, wecom, whatsapp. Plus non-plugin adapters in the `Platform` enum
(`gateway/config.py:167`): local, whatsapp_cloud, signal, api_server, webhook,
msgraph_webhook, weixin, bluebubbles, qqbot, yuanbao, relay.

Each plugin ships a `plugin.yaml` carrying exactly what a connect-a-channel UI needs:

```yaml
label: Slack
description: >
  Slack gateway adapter…
requires_env:
  - name: SLACK_BOT_TOKEN
    description: "Slack bot token (xoxb-...)"
    prompt: "Slack Bot Token (xoxb-...)"
    url: "https://api.slack.com/apps"
    password: true          # ← the UI must mask this
optional_env: [...]
```

There is already a parser: `hermes_cli/plugins_manifest.py` (`PluginManifest`, line 323;
`parse` at 461). **NOVA does not use it.** `nova/channels/providers.py:127` is a
hand-written tuple of **6** providers, with a comment explaining the subset as deliberate.
That was a reasonable call for a phase that shipped channel *governance*; it is the wrong
call for a channel *management* surface, and it is what you hit.

### Agent profiles — full lifecycle, already implemented

`hermes_cli/profiles.py`: `list_profiles` (685), `get_profile_dir` (232),
`profile_exists` (240), `read_profile_meta` (606), `write_profile_meta` (618),
`set_profile_display_name` (652), `rename_profile` (1635), `delete_profile` (1133),
`export_profile` (1516), `import_profile` (1542) — export+import is a **duplicate**
primitive, which covers your "clone an agent" ask without new code.

`validate_profile_name` (189) enforces `[a-z0-9][a-z0-9_-]{0,63}` plus a reserved-name list.
**This is the path-traversal defence** (§9): a name that passes cannot contain `/`, `.` or
`..`, so no path built from it can escape. Reuse it rather than writing another check.

### Scheduling — a complete scheduler

`cron/jobs.py`: `create_job` (1682), `get_job` (1811), `list_jobs` (1847),
**`update_job` (1960)**, `pause_job` (2080), `resume_job` (2093), `remove_job` (2209).
`cron/executions.py`: `list_executions` (322), plus the full execution lifecycle.

NOVA already uses create/pause/resume/remove (Phase 12). **`update_job` is unused** — which
is why you cannot edit a schedule, only delete and re-declare it.

### Runtime telemetry

- Work: `kanban_db.list_tasks`, `list_runs` (4139), `list_events` (1962) — already tenant-scoped.
- Cron: `cron/executions.list_executions` — real execution records with status and errors.
- Logs: `agent.log`, `errors.log`, `gateway.log` per profile (`hermes_logging.py:195`).
- Scheduler liveness: heartbeat, already surfaced by the Automations banner.

### Model and tools

Per-profile `config.yaml` holds provider/model; `hermes_cli/web_routers/profiles.py:101`
(`_write_profile_model`) is the existing writer. Toolsets come from `toolsets.py` and the
plugin registry.

## 3. Missing integrations

| # | Gap | Hermes has it? | Severity |
|---|---|---|---|
| 1 | Agent Soul/instructions not editable | yes — but via the bundle, per §0 | **blocking** |
| 2 | Only 6 of 22+ channels exposed | yes — `plugins_manifest` | **blocking** |
| 3 | No agent create / edit / delete / duplicate | yes — `profiles.py` | **blocking** |
| 4 | Schedules cannot be edited | yes — `update_job` | high |
| 5 | No per-agent profile page | partly | high |
| 6 | No channel connect / disconnect / test | partly — §5 | high |
| 7 | Model not changeable from the UI | yes — bundle + `_write_profile_model` | medium |
| 8 | Tool grants not changeable | yes — bundle | medium |
| 9 | No execution history for schedules | yes — `list_executions` | medium |
| 10 | No logs / errors surface | yes — per-profile log files | medium |
| 11 | Agent status is coarse | partly | medium |

## 4. Backend APIs and services to reuse

Reuse these; do not reimplement any of them.

| Need | Reuse | Where |
|---|---|---|
| Channel catalogue | `PluginManifest` + `parse` | `hermes_cli/plugins_manifest.py:323,461` |
| Platform ids | `Platform` enum + `_scan_bundled_plugin_platforms` | `gateway/config.py:167,223` |
| Agent name safety | `validate_profile_name` | `hermes_cli/profiles.py:189` |
| Agent lifecycle | `create/rename/delete/export/import_profile` | `hermes_cli/profiles.py` |
| Agent metadata | `read_profile_meta` / `write_profile_meta` | `hermes_cli/profiles.py:606,618` |
| Model write | `_write_profile_model` | `hermes_cli/web_routers/profiles.py:101` |
| Schedules | `create/update/pause/resume/remove_job` | `cron/jobs.py` |
| Execution history | `list_executions` | `cron/executions.py:322` |
| Work + runs + events | `kanban_db` | tenant-scoped already |
| Atomic file writes | `utils.atomic_write_text`, `materialize.atomic_write` | — |
| Validation | `nova.spec` loaders | raise `SpecError` with field names |
| Audit | `nova.audit.AuditLog` | intent → committed/failed |
| RBAC + CSRF | `nova/control/auth.py`, `server.py` | already enforced |

## 5. Required new APIs

Two genuine gaps, and I want to be precise about their limits rather than promise a button
that cannot work.

**a. Channel connection test.** Hermes has no uniform "test this connection" call; each
adapter connects in its own way inside the gateway process. A *credential presence* check is
honest and useful (are the manifest's `requires_env` names set in the agent's `.env`?) and
NOVA already does it (`nova/runtime/hermes/channels.py: readiness`). A true round-trip test
would mean running each adapter's handshake — that is adapter-by-adapter work, not one API.
**Plan: ship presence-checking, label it as such, and do not call it a connection test.**

**b. Writing credentials.** `.env` is on `materialize.NEVER_WRITE` deliberately — NOVA never
holds a customer secret. "Connect a channel" from the UI therefore means: declare the
connection, grant agents, and *show the operator which variables to set and where*. Making
the UI accept a bot token would reverse a security property the platform is built on. I will
not do that without you telling me to, and §9 of your brief argues against it too.

Everything else is a NOVA-side route over an existing Hermes primitive:

```
POST /agents                    create        POST /agents/<id>/soul       edit instructions
POST /agents/<id>               edit          POST /agents/<id>/duplicate  clone
POST /agents/<id>/archive       disable       POST /automations/<id>/update edit schedule
POST /channels/declare          add/remove    GET  /agents/<id>/runtime    status, runs, logs
```

## 6. Implementation plan

Ordered so each step is independently shippable and testable.

1. **Channel registry from manifests.** Read `plugin.yaml` through `plugins_manifest`, keep
   the existing hand-written capability notes as annotations where they exist, mark the rest
   `unknown` rather than guessing. Catalogue goes from 6 → 22+.
2. **Bundle writer.** One module that edits bundle files atomically, validates by reloading
   the whole bundle before committing, and refuses on `SpecError`. Every Control Centre
   write goes through it. This is the piece that makes changes persist correctly.
3. **Agent lifecycle routes** over the bundle writer + `profiles.py`.
4. **Soul editing** — bundle prompt file, then apply, then report the materialiser's result.
5. **Schedule editing** via `update_job`, plus execution history from `list_executions`.
6. **Agent profile page** assembling identity / model / capabilities / channels / schedules /
   runtime from the routes above.
7. **Runtime visibility** — runs, events, per-profile errors.

## 7. Testing plan

Backend, as automated tests:

- Persistence: create → edit → reload from disk → state matches.
- Soul: edit → apply → assert `<profile>/SOUL.md` contains it → re-apply → still correct.
- The §0 trap, as a regression test: a direct profile write is overwritten by apply, and the
  bundle write is not.
- Channel catalogue: every `plugins/platforms/*/plugin.yaml` appears; no invented entry.
- Schedules: create → update → assert the runtime record changed → pause → resume → delete.
- Tenant isolation: every new route refuses cross-tenant access.
- RBAC: a viewer is refused on every new write route.
- Audit: every mutation emits intent → committed/failed with the human actor.
- Failure cases: invalid config, invalid schedule, unknown agent, unauthorised request,
  malformed body, oversized body, cross-site write.

End-to-end, in a browser against a real control plane: create an agent, edit its Soul, save,
confirm applied, refresh, confirm it persisted.

**Not claimable without a live scheduler:** that an automation *executed*. Hermes' cron
ticker runs inside the gateway (`hermes_cli/cron.py:70`); there is no standalone daemon. The
UI will keep distinguishing a declared schedule from a recorded execution.


---

## 8. What landed, and what has not

Added after implementation, so this document does not read as a plan that was never checked
against its result.

### Done, and verified

| # | Item | Verified by |
|---|---|---|
| 1 | Channel catalogue is the runtime's 22, read from `plugin.yaml` | a test asserting catalogue == platforms on disk |
| 2 | Bundle writer: validated, atomic, traversal-proof | unit tests incl. a `../../` escape attempt |
| 3 | Agent create / update / duplicate / archive / restore / delete | 34 tests in `test_agent_management.py` |
| 4 | Soul editing that survives a later apply | applies twice around the edit, checks the runtime's own `SOUL.md` |
| 5 | Soul editor in the Control Centre | driven in a real browser: edit → save → "Applied" → reload → still there |
| 6 | Schedule editing via `update_job` | adapter test; the objective is deliberately not settable |
| 7 | Execution history from `list_executions` | exposed at `/agents/<id>/automations` |
| 8 | Channels screen lists all 22 with credential names | rendered in a browser |
| 9 | RBAC, audit, CSRF on every new route | a viewer is refused on all seven; intent → committed/failed with the actor |

### Not done

Stated plainly rather than left to be discovered.

- **Agent create/edit forms in the UI.** The routes exist, are tested and are callable; the
  Control Centre has no form for them yet, so creating an agent today means calling the API.
  The Soul editor is the pattern the rest should follow.
- **Schedule editing in the UI.** Same: route done, no form.
- **Channel connect/disconnect from the UI.** Deliberate, per §5b. Connecting means writing
  a customer's secret, and `.env` is on `materialize.NEVER_WRITE` so that NOVA cannot hold
  one. The screen says what each platform needs and where to put it.
- **Connection testing.** Per §5a there is no uniform Hermes call for it. Credential
  *presence* is reported and labelled as that.
- **Logs surface.** Per-profile `agent.log` / `errors.log` are readable but not exposed. The
  right shape is a bounded tail on an authenticated route; it is not written yet.
- **Live-proven cron execution.** Unchanged and unchangeable from here: Hermes' ticker runs
  inside the gateway, so a control plane with no gateway cannot observe an execution.
